"""
training/distill.py — Distillation vers LLM Maison avec plusieurs backends teacher.
--profile small|medium|large : definir avant import via LLM_PROFILE ou en premier dans argv.

Backends:
  transformers — HuggingFace (4-bit) comme avant
  ollama       — API localhost:11434 (pas de logits → distillation séquence CE)
  lmstudio     — API OpenAI-compatible localhost:1234 + logprobs si dispo
  llamacpp     — llama-cpp-python + .gguf (logits bruts si logits_all=True)

Loss:
  Avec logits : alpha * KL(teacher || student) + (1-alpha) * CE
  Sans logits (ollama) : CE sur la suite générée par le teacher (sequence-level).
  La CE est masquée sur le prompt : on ne supervise que la continuation (pas le re-predict du wiki).

Vocab: pendant la distillation avec tokenizer teacher, utiliser un ModelConfig
dont vocab_size = len(teacher_tokenizer). Après distillation, re-tokenizer BPE pour finetune.

Usage:
  python -m training.distill --mode online --backend ollama
  python -m training.distill --mode online --backend lmstudio --api-url http://localhost:1234
  python -m training.distill --mode online --backend llamacpp --model-path /path/to/model.gguf
  python -m training.distill --mode offline --phase generate --backend transformers
  python -m training.distill --backend transformers   # teacher HF 4-bit + KL sur logits (kd > 0)
  python -m training.distill --backend transformers --fresh   # repartir de zero (ignore checkpoints)
"""
import os
import json
import time
import math
import random
import sys
import argparse
from pathlib import Path

for _i, _a in enumerate(sys.argv):
    if _a == "--profile" and _i + 1 < len(sys.argv):
        os.environ["LLM_PROFILE"] = sys.argv[_i + 1]
        break
from datetime import datetime
from typing import Optional, Tuple, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import DST_CFG, MDL_CFG, MODEL_DIR, LOG_DIR, DATA_DIR, PROCESSED_DIR, TRN_CFG, ROOT, TEACHER_TOKENIZER_DIR
from config import ModelConfig
from model.transformer import LLMMaison

TEACHER_LOGITS_DIR = PROCESSED_DIR / "teacher_logits"
TOP_K_LOGITS = 100
OFFLINE_BATCH_SIZE = 4
OFFLINE_MAX_LENGTH = 512

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"


def _targets_only_continuation(ttok, full_text, prompt_len_chars, ids):
    """
    CE uniquement sur la suite (continuation), pas sur le re-predict du prompt.
    Evite que le student apprenne surtout a continuer du wiki comme un article.
    Utilise offset_mapping du tokenizer pour trouver le premier token apres prompt.
    """
    targets = ids.clone()
    if prompt_len_chars <= 0 or ids.size(1) < 3:
        return targets
    try:
        enc = ttok(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
        om = enc.get("offset_mapping")
        if not om:
            return targets
        if isinstance(om[0], (list, tuple)) and len(om[0]) == 2 and isinstance(om[0][0], (list, tuple)):
            om = om[0]
        cont_start = None
        for i, span in enumerate(om):
            if span is None:
                continue
            s, e = span[0], span[1]
            if s is not None and int(s) >= prompt_len_chars:
                cont_start = i
                break
        if cont_start is None or cont_start < 1:
            return targets
        # Pas de loss sur les predictions dont la cible est encore dans le prompt
        targets[:, 1:cont_start] = -1
    except Exception:
        pass
    return targets


def load_texts(max_docs=10000):
    """
    Mix conversations + livres + wiki + instructions + code + web.
    Retourne une liste de strings (meme interface qu'avant) — la boucle distill ne change pas.
    """
    from crawler.cleaner import normalize_after_json_load

    MIX_RATIOS = {
        "conversations": 0.30,
        "books": 0.25,
        "wikipedia": 0.15,
        "instructions": 0.15,
        "code": 0.10,
        "web": 0.05,
    }

    all_texts = []

    # 1. Wikipedia — jsonl existants sauf mixed_* (deja melanges ailleurs)
    wiki_target = max(1, int(max_docs * MIX_RATIOS["wikipedia"]))
    wiki_texts = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        if f.name.startswith("mixed_"):
            continue
        for line in open(f, "r", encoding="utf-8", errors="replace"):
            try:
                t = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            t = normalize_after_json_load(t)
            if len(t) > 200:
                wiki_texts.append(t[:2000])
            if len(wiki_texts) >= wiki_target:
                break
        if len(wiki_texts) >= wiki_target:
            break
    all_texts.extend(wiki_texts)
    print(f"[MIX] Wikipedia: {len(wiki_texts)} textes", flush=True)

    conv_target = max(1, int(max_docs * MIX_RATIOS["conversations"]))
    conv_texts = _load_conversations(conv_target)
    all_texts.extend(conv_texts)
    print(f"[MIX] Conversations: {len(conv_texts)} textes", flush=True)

    books_target = max(1, int(max_docs * MIX_RATIOS["books"]))
    books_texts = _load_books(books_target)
    all_texts.extend(books_texts)
    print(f"[MIX] Books: {len(books_texts)} textes", flush=True)

    inst_target = max(1, int(max_docs * MIX_RATIOS["instructions"]))
    inst_texts = _load_instructions(inst_target)
    all_texts.extend(inst_texts)
    print(f"[MIX] Instructions: {len(inst_texts)} textes", flush=True)

    code_target = max(1, int(max_docs * MIX_RATIOS["code"]))
    code_texts = _load_code(code_target)
    all_texts.extend(code_texts)
    print(f"[MIX] Code: {len(code_texts)} textes", flush=True)

    web_target = max(1, int(max_docs * MIX_RATIOS["web"]))
    web_texts = _load_web(web_target)
    all_texts.extend(web_texts)
    print(f"[MIX] Web: {len(web_texts)} textes", flush=True)

    rng = random.Random(42)
    rng.shuffle(all_texts)
    print(f"[MIX] Total: {len(all_texts)} textes (cap max_docs={max_docs})", flush=True)
    return all_texts[:max_docs]


def _load_conversations(target):
    """Conversations HF — format <|user|> / <|assistant|>."""
    texts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        conv_buffer = {}
        for row in ds:
            if len(texts) >= target:
                break
            lang = (row.get("lang") or "").lower()
            if lang and lang not in ("fr", "en"):
                continue
            parent_id = row.get("parent_id")
            msg_id = row.get("message_id")
            text = (row.get("text") or "").strip()
            role = row.get("role", "")
            if not text or len(text) < 20:
                continue
            if parent_id is None:
                conv_buffer[msg_id] = f"<|user|>\n{text}\n"
            elif parent_id in conv_buffer and role == "assistant":
                full = conv_buffer[parent_id] + f"<|assistant|>\n{text}\n"
                if len(full) > 100:
                    texts.append(full[:2000])
                del conv_buffer[parent_id]
            elif parent_id in conv_buffer and role == "prompter":
                conv_buffer[msg_id] = conv_buffer.pop(parent_id) + f"<|user|>\n{text}\n"
        print(f"  [CONV] OASST2: {len(texts)}", flush=True)
    except Exception as e:
        print(f"  [CONV] OASST2 failed: {e}", flush=True)

    if len(texts) < target:
        try:
            from datasets import load_dataset

            for split in ("train_sft", "train"):
                try:
                    ds = load_dataset(
                        "HuggingFaceH4/ultrachat_200k", split=split, streaming=True
                    )
                    break
                except Exception:
                    ds = None
            if ds is None:
                raise RuntimeError("ultrachat split not found")
            count = 0
            for row in ds:
                if len(texts) >= target:
                    break
                messages = row.get("messages") or []
                if not messages:
                    continue
                parts = []
                for msg in messages:
                    role = (msg.get("role") or "").lower()
                    content = (msg.get("content") or "").strip()
                    if not content:
                        continue
                    if role == "user":
                        parts.append(f"<|user|>\n{content}\n")
                    elif role == "assistant":
                        parts.append(f"<|assistant|>\n{content}\n")
                text = "".join(parts)
                if len(text) > 100:
                    texts.append(text[:2000])
                    count += 1
            print(f"  [CONV] UltraChat: +{count}", flush=True)
        except Exception as e:
            print(f"  [CONV] UltraChat failed: {e}", flush=True)

    if len(texts) < target:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train",
                streaming=True,
            )
            count = 0
            for row in ds:
                if len(texts) >= target:
                    break
                convos = row.get("conversations") or []
                text = ""
                for turn in convos:
                    role = "user" if turn.get("from") == "human" else "assistant"
                    content = (turn.get("value") or "").strip()
                    if content:
                        text += f"<|{role}|>\n{content}\n"
                if len(text) > 100:
                    texts.append(text[:2000])
                    count += 1
            print(f"  [CONV] ShareGPT: +{count}", flush=True)
        except Exception as e:
            print(f"  [CONV] ShareGPT failed: {e}", flush=True)

    return texts[:target]


def _load_books(target):
    """
    French-PD-Books peut bloquer 10–30+ min au premier acces (145 shards).
    Par defaut on remplit le quota avec Wikipedia FR (articles longs) en premier.
    Pour forcer French-PD: DISTILL_USE_FRENCH_PD=1
    """
    texts = []
    # 1) Wikipedia FR long d'abord — rapide une fois en cache, pas de blocage initial
    try:
        from datasets import load_dataset

        for subset in ("20231101.fr", "20220301.fr"):
            try:
                ds = load_dataset(
                    "wikimedia/wikipedia", subset, split="train", streaming=True
                )
                break
            except Exception:
                ds = None
        if ds is not None:
            count = 0
            for row in ds:
                if len(texts) >= target:
                    break
                text = row.get("text") or ""
                if len(text) > 2000:
                    texts.append(text[:2000])
                    count += 1
            print(f"  [BOOKS] Wikipedia FR long: {len(texts)}", flush=True)
    except Exception as e:
        print(f"  [BOOKS] Wikipedia FR failed: {e}", flush=True)

    # 2) French-PD-Books seulement si explicitement demande (sinon risque de blocage long)
    if len(texts) < target and os.environ.get("DISTILL_USE_FRENCH_PD") == "1":
        try:
            from datasets import load_dataset

            print("  [BOOKS] French-PD-Books (DISTILL_USE_FRENCH_PD=1) — peut etre tres long...", flush=True)
            ds = load_dataset("PleIAs/French-PD-Books", split="train", streaming=True)
            for row in ds:
                if len(texts) >= target:
                    break
                text = row.get("text") or row.get("content") or ""
                if not isinstance(text, str) or len(text) < 500:
                    continue
                for i in range(0, len(text), 2000):
                    chunk = text[i : i + 2000]
                    if len(chunk) > 200:
                        texts.append(chunk)
                    if len(texts) >= target:
                        break
            print(f"  [BOOKS] French-PD-Books: total {len(texts)}", flush=True)
        except Exception as e:
            print(f"  [BOOKS] French-PD-Books failed: {e}", flush=True)
    elif len(texts) < target:
        print(
            f"  [BOOKS] French-PD-Books skip (evite blocage). "
            f"Actuel {len(texts)}/{target} — pour completer: DISTILL_USE_FRENCH_PD=1",
            flush=True,
        )

    return texts[:target]


def _load_instructions(target):
    texts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        for row in ds:
            if len(texts) >= target:
                break
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()
            if not instruction or not output:
                continue
            prompt = f"{instruction}\n{inp}".strip() if inp else instruction
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{output}\n"
            if len(text) > 50:
                texts.append(text[:2000])
                count += 1
        print(f"  [INST] Alpaca: {count}", flush=True)
    except Exception as e:
        print(f"  [INST] Alpaca failed: {e}", flush=True)

    if len(texts) < target:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "databricks/databricks-dolly-15k", split="train", streaming=True
            )
            count = 0
            for row in ds:
                if len(texts) >= target:
                    break
                instruction = (row.get("instruction") or "").strip()
                context = (row.get("context") or "").strip()
                response = (row.get("response") or "").strip()
                if not instruction or not response:
                    continue
                prompt = f"{instruction}\n{context}".strip() if context else instruction
                text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
                if len(text) > 50:
                    texts.append(text[:2000])
                    count += 1
            print(f"  [INST] Dolly: +{count}", flush=True)
        except Exception as e:
            print(f"  [INST] Dolly failed: {e}", flush=True)

    return texts[:target]


def _load_code(target):
    texts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        count = 0
        for row in ds:
            if len(texts) >= target:
                break
            prompt = (row.get("prompt") or "").strip()
            completion = (row.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            text = f"<|user|>\n{prompt}\n<|assistant|>\n```\n{completion}\n```\n"
            if len(text) > 50:
                texts.append(text[:2000])
                count += 1
        print(f"  [CODE] CodeAlpaca: {count}", flush=True)
    except Exception as e:
        print(f"  [CODE] CodeAlpaca failed: {e}", flush=True)
    return texts[:target]


def _load_web(target):
    """CulturaX peut etre tres lourd — skip sauf DISTILL_USE_CULTURAX=1."""
    texts = []
    if os.environ.get("DISTILL_USE_CULTURAX") != "1":
        print("  [WEB] CulturaX skip (set DISTILL_USE_CULTURAX=1 pour activer)", flush=True)
        return texts[:target]
    try:
        from datasets import load_dataset

        ds = load_dataset("uonlp/CulturaX", "fr", split="train", streaming=True)
        count = 0
        for row in ds:
            if len(texts) >= target:
                break
            text = row.get("text") or ""
            if len(text) < 200:
                continue
            words = text.split()
            if len(words) < 30:
                continue
            if len(set(words)) / max(len(words), 1) < 0.3:
                continue
            texts.append(text[:2000])
            count += 1
            if count >= target:
                break
        print(f"  [WEB] CulturaX FR: {count}", flush=True)
    except Exception as e:
        print(f"  [WEB] CulturaX failed: {e}", flush=True)
    return texts[:target]


def _api_base(url: str, backend: str) -> str:
    url = (url or "").rstrip("/")
    if not url:
        url = DEFAULT_OLLAMA_URL if backend == "ollama" else DEFAULT_LMSTUDIO_URL
    return url


def _ensure_llamacpp_cuda_dll_dirs():
    """
    Windows + wheel cu121 : llama.dll charge ggml-cuda.dll qui dépend de
    cudart64_12 / cublas64_12. Sans ces dossiers sur le chemin de chargement
    des DLL, import llama_cpp échoue. Les wheels nvidia-* dans le venv
    fournissent les bons binaires sans installer CUDA 12.1 en global.
    """
    if sys.platform != "win32":
        return
    try:
        import site

        for site_dir in site.getsitepackages():
            for sub in (
                "llama_cpp/lib",
                "nvidia/cuda_runtime/bin",
                "nvidia/cublas/bin",
            ):
                p = Path(site_dir) / sub
                if p.is_dir():
                    try:
                        os.add_dll_directory(str(p.resolve()))
                    except OSError:
                        pass
    except Exception:
        pass


def _load_teacher_tokenizer():
    """Charge le tokenizer teacher en local uniquement (pas d'appel Hugging Face)."""
    from transformers import AutoTokenizer

    path = getattr(DST_CFG, "teacher_tokenizer", None) or str(TEACHER_TOKENIZER_DIR)
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        print(f"[DISTILL] Dossier tokenizer absent: {path}")
        print("  Cree teacher_tokenizer/ et copie tokenizer.json + tokenizer_config.json (voir teacher_tokenizer/README.md)")
        raise FileNotFoundError(f"teacher_tokenizer dir missing: {path}")

    has_tok = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
    if not has_tok:
        print(f"[DISTILL] Pas de tokenizer.json (ni tokenizer.model) dans: {path}")
        print("  Copie les fichiers tokenizer depuis un modele Llama 3 (voir teacher_tokenizer/README.md)")
        raise FileNotFoundError(f"no tokenizer files in {path}")

    try:
        ttok = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
        )
    except Exception as e:
        print(f"[DISTILL] Echec chargement tokenizer local: {e}")
        print("  Verifie que tokenizer.json + tokenizer_config.json (+ special_tokens_map.json) sont presents.")
        raise

    if ttok.pad_token is None:
        ttok.pad_token = ttok.eos_token
    print(f"[DISTILL] Tokenizer charge en local depuis: {path}")
    return ttok


def _student_config_for_teacher_vocab(vocab_size: int) -> ModelConfig:
    """Copy MDL_CFG with teacher vocab_size (embedding + lm_head aligned)."""
    d = {k: v for k, v in MDL_CFG.__dict__.items() if not k.startswith("_")}
    d["vocab_size"] = vocab_size
    return ModelConfig(**d)


# ── Backend: transformers — teacher HF 4-bit, forward logits uniquement (pas de génération) ──

# Ordre de secours si meta-llama gated / indisponible
TEACHER_TRANSFORMERS_FALLBACKS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "unsloth/llama-3-8b-Instruct",
    "NousResearch/Meta-Llama-3-8B-Instruct",
]


def _get_teacher_transformers_model():
    """
    Charge le teacher une seule fois (4-bit, device_map auto). Essaie plusieurs repos
    si accès refusé. Forward uniquement → logits pour KL ; pas de generate().
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if getattr(_get_teacher_transformers_model, "_model", None) is not None:
        return _get_teacher_transformers_model._model

    # Liste des repos à essayer : config d'abord, puis fallbacks sans doublon
    primary = getattr(DST_CFG, "teacher_model", None) or TEACHER_TRANSFORMERS_FALLBACKS[0]
    repos = [primary] + [r for r in TEACHER_TRANSFORMERS_FALLBACKS if r != primary]
    seen = set()
    repos = [r for r in repos if not (r in seen or seen.add(r))]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    last_err = None
    for repo in repos:
        try:
            print(f"[DISTILL] Chargement teacher transformers: {repo} ...", flush=True)
            # Tout sur GPU : un seul device (cuda:0) pour eviter CPU offload
            teacher = AutoModelForCausalLM.from_pretrained(
                repo,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16,
                quantization_config=bnb,
            )
            teacher.eval()
            _get_teacher_transformers_model._model = teacher
            _get_teacher_transformers_model._repo = repo
            print(
                f"[DISTILL] Teacher OK: {repo} — forward logits seulement (kd non nul).",
                flush=True,
            )
            return teacher
        except Exception as e:
            last_err = e
            print(f"[DISTILL] Echec {repo}: {e}", flush=True)

    print("[DISTILL] Accès teacher HF refusé ou indisponible.", flush=True)
    print("  1) huggingface-cli login  (token avec accès meta-llama si besoin)", flush=True)
    print("  2) --backend ollama  (pas de logits -> kd=0, CE sequence)", flush=True)
    if last_err is not None:
        raise last_err
    raise RuntimeError("Aucun teacher transformers chargé.")


def _get_teacher_transformers_tokenizer():
    """
    Tokenizer aligné sur le teacher HF (vocab 128256). Obligatoire pour encoder
    avant teacher(**enc) : le tokenizer local peut avoir +4 tokens (128260) et
    tout id >= 128256 fait planter l'embedding CUDA (srcIndex < srcSelectDimSize).
    """
    from transformers import AutoTokenizer

    if getattr(_get_teacher_transformers_tokenizer, "_tok", None) is not None:
        return _get_teacher_transformers_tokenizer._tok
    _get_teacher_transformers_model()  # charge modèle + définit _repo
    repo = getattr(_get_teacher_transformers_model, "_repo", None) or DST_CFG.teacher_model
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _get_teacher_transformers_tokenizer._tok = tok
    vs = getattr(tok, "vocab_size", None) or len(tok)
    print(
        f"[DISTILL] Teacher tokenizer HF ({repo}) — vocab_size={vs} (évite OOB embedding).",
        flush=True,
    )
    return tok


def _teacher_forward_transformers(enc, dev):
    """Teacher forward pass — PAS de génération. Retourne logits (B, T, V)."""
    teacher = _get_teacher_transformers_model()
    # Device du teacher (device_map auto peut répartir; on pousse les inputs sur le même device que la 1re couche)
    teacher_dev = next(teacher.parameters()).device
    enc = {k: v.to(teacher_dev) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
    # Sécurité : aucun input_id ne doit dépasser la taille réelle des embeddings
    emb = teacher.get_input_embeddings()
    if emb is not None:
        n_emb = emb.num_embeddings
        if enc["input_ids"].max().item() >= n_emb or enc["input_ids"].min().item() < 0:
            raise RuntimeError(
                f"[DISTILL] input_ids hors plage pour le teacher (max id doit être < {n_emb}). "
                "Utilise _get_teacher_transformers_tokenizer() pour encoder, pas le tokenizer local + special tokens."
            )
    with torch.no_grad():
        return teacher(**enc).logits


# ── Backend: lmstudio (OpenAI completions + logprobs) ─────────────

def _lmstudio_completions_logprobs(
    prompt: str, api_url: str, model: str = "llama3", top_logprobs: int = 5
) -> Optional[dict]:
    import requests
    url = f"{api_url.rstrip('/')}/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": top_logprobs,
    }
    r = requests.post(url, json=body, timeout=120)
    if r.status_code != 200:
        return None
    return r.json()


# ── Backend: ollama (sequence-level only) ────────────────────────

def _peek_text(s, head=120, tail=80):
    """Apercu une ligne pour logs (evite JSON illisible)."""
    s = (s or "").replace("\r", " ").replace("\n", " | ")
    if len(s) <= head:
        return s
    if len(s) <= head + tail:
        return s
    return s[:head] + " [...] " + s[-tail:]


def debug_ollama(api_url: str, model: str) -> None:
    """Ecrit la reponse brute Ollama dans logs/ollama_debug.txt (generate + chat, prompt court)."""
    import requests
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "ollama_debug.txt"
    base = api_url.rstrip("/")
    lines = [f"api_url={base} model={model}\n"]

    def dump(label, r):
        lines.append(f"\n--- {label} status={r.status_code} ---\n")
        lines.append(r.text[:12000] if r.text else "(no body)")
        lines.append("\n")

    # Generate minimal
    body = {"model": model, "prompt": "Say hello in 3 words.", "stream": False, "options": {"num_predict": 64}}
    if "gpt-oss" in model.lower():
        body["options"]["think"] = "low"
    r = requests.post(f"{base}/api/generate", json=body, timeout=120)
    dump("POST /api/generate", r)

    # Chat minimal
    body2 = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in 3 words."}],
        "stream": False,
        "options": {"num_predict": 64},
    }
    if "gpt-oss" in model.lower():
        body2["options"]["think"] = "low"
    r2 = requests.post(f"{base}/api/chat", json=body2, timeout=120)
    dump("POST /api/chat", r2)

    out.write_text("".join(lines), encoding="utf-8")
    print(f"[DEBUG] Ecrit {out}")
    try:
        j = r.json()
        print(f"[DEBUG] generate keys: {list(j.keys()) if isinstance(j, dict) else type(j)}")
        if isinstance(j, dict):
            for k in ("response", "thinking", "error"):
                v = j.get(k)
                if v is not None:
                    print(f"[DEBUG]   {k} len={len(str(v))}")
    except Exception as e:
        print(f"[DEBUG] parse generate: {e}")


def _distill_peek_student(student, ttok, dev, step: int, max_new: int = 40) -> None:
    """Affiche un mini extrait genere par le student dans le terminal (amusant)."""
    raw = student.module if hasattr(student, "module") else student
    was_training = raw.training
    raw.eval()
    try:
        from model.generate import generate, GenConfig
        prompt = "Bonjour"
        enc = ttok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev)
        if input_ids.size(1) < 1:
            return
        cfg = GenConfig(max_new_tokens=max_new, temperature=0.85, top_k=40, top_p=0.92)
        if getattr(ttok, "eos_token_id", None) is not None:
            cfg.stop_tokens = [ttok.eos_token_id]
        with torch.inference_mode():
            out = generate(raw, input_ids, cfg)
        new = out[0, input_ids.size(1) :].tolist()
        text = ttok.decode(new, skip_special_tokens=True).strip()
        text = text.replace("\n", " ").strip()
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"[STUDENT] {text}", flush=True)
    except Exception as e:
        print(f"[STUDENT] <erreur {e}>", flush=True)
    finally:
        raw.train(was_training)


def _ollama_check(api_url: str) -> bool:
    """Verifie que c'est bien Ollama sur ce port (GET /api/tags)."""
    import requests
    base = api_url.rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_accumulate_streaming_body(raw: str) -> str:
    """Ollama peut renvoyer du NDJSON (une ligne JSON par chunk) meme avec stream:false mal gere."""
    if not raw or not raw.strip():
        return ""
    parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(j, dict):
            continue
        if j.get("error"):
            continue
        # Chaque chunk stream ajoute un morceau dans "response"
        if "response" in j and j["response"] is not None:
            parts.append(str(j["response"]))
        if "thinking" in j and j["thinking"] is not None and not parts:
            parts.append(str(j["thinking"]))
        # Chat stream: message.content en delta selon versions
        msg = j.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            parts.append(str(msg["content"]))
        if j.get("done") is True:
            break
    return "".join(parts).strip()


def _ollama_extract_text(j, max_thinking_chars: int = 2000) -> str:
    """Ollama /api/generate ou /api/chat. gpt-oss met souvent tout dans 'thinking' et laisse 'response' vide."""
    if not j or not isinstance(j, dict):
        return ""
    if j.get("error"):
        return ""
    # /api/generate classique
    s = j.get("response")
    if s:
        return str(s).strip()
    # gpt-oss: contenu dans thinking si response vide (distill CE sur ce texte = imite le raisonnement)
    th = j.get("thinking")
    if th:
        th = str(th).strip()
        if len(th) > max_thinking_chars:
            th = th[:max_thinking_chars] + "..."
        return th
    # /api/chat
    msg = j.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if c:
            return str(c).strip()
        # Chat + thinking dans message
        if msg.get("thinking"):
            th = str(msg["thinking"]).strip()
            return th[:max_thinking_chars] if len(th) > max_thinking_chars else th
    for k in ("text", "output", "completion"):
        if j.get(k):
            return str(j[k]).strip()
    return ""


def _ollama_generate(api_url: str, model: str, prompt: str, max_tokens: int = 256) -> str:
    import requests
    base = api_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    url_gen = f"{base}/api/generate"
    # num_predict + num_ctx: certains modeles vides si contexte trop court
    opts = {
        "temperature": 0.7,
        "num_predict": max_tokens,
        "num_ctx": 8192,
    }
    # gpt-oss: sans think low, response reste vide et tout part dans thinking (voir logs/ollama_debug)
    if "gpt-oss" in (model or "").lower():
        opts["think"] = "low"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": opts,
    }
    r = requests.post(url_gen, json=body, headers=headers, timeout=300)
    text = ""
    if r.status_code == 200:
        # Corps unique JSON
        try:
            j = r.json()
            if isinstance(j, dict) and j.get("error"):
                print(f"[DISTILL] Ollama generate error: {j.get('error')}", flush=True)
            else:
                text = _ollama_extract_text(j)
        except Exception:
            j = {}
        # Pas un seul JSON -> NDJSON stream aggrégé dans le body
        if not text and r.text:
            text = _ollama_accumulate_streaming_body(r.text)
        # Toujours vide -> stream explicite
        if not text:
            body["stream"] = True
            rs = requests.post(
                url_gen, json=body, headers=headers, timeout=300, stream=True
            )
            if rs.status_code == 200:
                buf = []
                for line in rs.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        if isinstance(j, dict) and j.get("response") is not None:
                            buf.append(str(j["response"]))
                        if j.get("done"):
                            break
                    except Exception:
                        pass
                text = "".join(buf).strip()

    # Generate vide ou 404 -> /api/chat (gpt-oss souvent mieux en chat)
    if not text:
        url_chat = f"{base}/api/chat"
        chat_opts = {"temperature": 0.7, "num_predict": max_tokens, "num_ctx": 8192}
        if "gpt-oss" in (model or "").lower():
            chat_opts["think"] = "low"
        body_chat = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Continue le texte suivant de facon naturelle, sans repeter le prompt."},
                {"role": "user", "content": prompt[-4000:] if len(prompt) > 4000 else prompt},
            ],
            "stream": False,
            "options": chat_opts,
        }
        r2 = requests.post(url_chat, json=body_chat, headers=headers, timeout=300)
        if r2.status_code == 200:
            try:
                j2 = r2.json()
                if isinstance(j2, dict) and j2.get("error"):
                    print(f"[DISTILL] Ollama chat error: {j2.get('error')}", flush=True)
                else:
                    text = _ollama_extract_text(j2)
            except Exception:
                pass
            if not text and r2.text:
                text = _ollama_accumulate_streaming_body(r2.text)
        if not text:
            body_chat["stream"] = True
            r3 = requests.post(
                url_chat, json=body_chat, headers=headers, timeout=300, stream=True
            )
            if r3.status_code == 200:
                buf = []
                for line in r3.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        if isinstance(j, dict):
                            msg = j.get("message")
                            if isinstance(msg, dict) and msg.get("content"):
                                buf.append(str(msg["content"]))
                            if j.get("done"):
                                break
                    except Exception:
                        pass
                text = "".join(buf).strip()

    if not text and r.status_code not in (200,) and r.status_code != 404:
        r.raise_for_status()

    if not text:
        if not getattr(_ollama_generate, "_warned_empty", False):
            _ollama_generate._warned_empty = True
            raw = ""
            try:
                raw = (r.text or "")[:800]
            except Exception:
                pass
            print(
                f"[DISTILL] Ollama toujours vide — modele={model!r}. "
                f"Debut corps brut (800c): {raw!r}",
                flush=True,
            )
            print(
                "[DISTILL] Test manuel: ollama run gpt-oss:20b  puis tape un mot. "
                "Si rien ne sort, le modele ne genere pas via API.",
                flush=True,
            )

    return text


def _ollama_chat_logprobs(api_url: str, model: str, messages: list) -> Optional[dict]:
    """Try /api/chat with logprobs if server supports it."""
    import requests
    url = f"{api_url.rstrip('/')}/api/chat"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    # Some Ollama versions accept logprobs in options
    try:
        body["options"]["logprobs"] = True
    except Exception:
        pass
    r = requests.post(url, json=body, timeout=120)
    if r.status_code != 200:
        return None
    return r.json()


# ── Online distillation unified ───────────────────────────────────

def _load_llamacpp_llm(
    model_path: str,
    n_gpu_layers: int,
    *,
    verbose: bool = False,
    use_mmap: bool = True,
):
    """Charge Llama ; lève ValueError si llama_load_model_from_file échoue."""
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=512,
        logits_all=True,
        verbose=verbose,
        n_gpu_layers=n_gpu_layers,
        use_mmap=use_mmap,
    )


def distill_online(
    backend: str = "transformers",
    api_url: str = "",
    model_path: str = "",
    ollama_model: str = "llama3",
    lmstudio_model: str = "llama3",
    use_deepspeed_override: Optional[bool] = None,
    llamacpp_n_gpu_layers: int = -1,
    llamacpp_verbose: bool = False,
    llamacpp_no_mmap: bool = False,
    fresh: bool = False,
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backend = backend.lower()
    api_url = _api_base(api_url, backend)

    if backend == "ollama":
        if not _ollama_check(api_url):
            print(f"[DISTILL] Ollama introuvable sur {api_url} (GET /api/tags != 200).")
            print("  - Lance Ollama (app ou ollama serve) et verifie le port 11434.")
            print("  - ollama pull llama3   (ou le nom exact: ollama list)")
            print("  - Si autre port: python -m training.distill --backend ollama --api-url http://127.0.0.1:PORT")
            sys.exit(1)
        print(f"[DISTILL] Ollama OK sur {api_url}")

    ttok = _load_teacher_tokenizer()
    # Tokens de conversation (texte mix) — etend le vocab avant creation du student
    CHAT_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    try:
        to_add = [t for t in CHAT_TOKENS if t not in ttok.get_vocab()]
        if to_add:
            ttok.add_tokens(to_add, special_tokens=True)
            print(f"[DISTILL] +{len(to_add)} special tokens: {to_add}", flush=True)
    except Exception as e:
        print(f"[DISTILL] add_tokens skip: {e}", flush=True)
    vocab_size = len(ttok)
    student_cfg = _student_config_for_teacher_vocab(vocab_size)
    print(f"[DISTILL] Backend={backend} teacher_vocab_size={vocab_size} -> student embedding matches")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = None
    if not fresh:
        # Priorite aux checkpoints "globaux" (resume generique)
        for name in ("distill_latest.pt", "distill_final.pt"):
            p = MODEL_DIR / name
            if p.exists():
                resume_path = p
                break
        # Backend transformers : si pas de distill_latest/final, on reprend
        # automatiquement sur le plus grand distill_s{step}.pt disponible.
        if resume_path is None and backend == "transformers":
            try:
                pattern = str(MODEL_DIR / "distill_s*.pt")
                import glob as _glob

                candidates = []
                for path in _glob.glob(pattern):
                    name = os.path.basename(path)
                    # nom attendu: distill_s{step}.pt
                    if not name.startswith("distill_s") or not name.endswith(".pt"):
                        continue
                    middle = name[len("distill_s") : -len(".pt")]
                    try:
                        s = int(middle)
                    except ValueError:
                        continue
                    candidates.append((s, path))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    step_candidate, path_candidate = candidates[0]
                    resume_path = Path(path_candidate)
                    print(
                        f"[DISTILL] Reprise auto: utilise {resume_path.name} (step≈{step_candidate}) car aucun distill_latest.pt / distill_final.pt",
                        flush=True,
                    )
            except Exception as e:
                print(f"[DISTILL] Resume auto distill_s*.pt ignore ({e})", flush=True)
    else:
        print(
            "[DISTILL] --fresh : pas de reprise checkpoint — student neuf, step=0, DeepSpeed OK.",
            flush=True,
        )

    step = 0
    text_cursor = 0  # indice du prochain batch dans texts — evite de repasser les memes phrases au redemarrage
    if resume_path:
        try:
            ck = torch.load(str(resume_path), map_location=dev, weights_only=False)
            ck_vocab = (ck.get("config") or {}).get("vocab_size")
            if ck_vocab is not None and ck_vocab != vocab_size:
                print(f"[DISTILL] Resume ignore: vocab checkpoint={ck_vocab} != teacher {vocab_size}")
                resume_path = None
            else:
                student, ck = LLMMaison.load_checkpoint(str(resume_path), dev)
                step = int(ck.get("step", 0))
                text_cursor = int(ck.get("text_cursor", 0))
                print(f"[DISTILL] Reprise auto depuis {resume_path.name} step={step} text_cursor={text_cursor}")
        except Exception as e:
            print(f"[DISTILL] Resume echoue ({e}) -> depart zero")
            resume_path = None

    if not resume_path:
        student = LLMMaison(student_cfg).to(dev)
    if getattr(DST_CFG, "gradient_checkpointing", False) and hasattr(student, "enable_gradient_checkpointing"):
        student.enable_gradient_checkpointing()

    # Reprise: pas de DeepSpeed (pas d'optimizer state dans .pt) -> AdamW stable
    use_deepspeed = getattr(DST_CFG, "use_deepspeed", False) and dev == "cuda" and not resume_path
    if use_deepspeed_override is False:
        use_deepspeed = False
    student_engine = None
    opt = None
    if use_deepspeed:
        try:
            import deepspeed
            ds_config_path = ROOT / "deepspeed_config.json"
            if not ds_config_path.exists():
                print("[DISTILL] deepspeed_config.json absent -> PyTorch AdamW")
                use_deepspeed = False
            else:
                with open(ds_config_path, encoding="utf-8") as f:
                    ds_config = json.load(f)
                ds_config["train_micro_batch_size_per_gpu"] = DST_CFG.batch_size
                ds_config["gradient_accumulation_steps"] = DST_CFG.gradient_accumulation
                ds_config["gradient_clipping"] = 1.0
                # train_batch_size = micro * num_gpus * grad_accum (1 GPU)
                ds_config["train_batch_size"] = (
                    DST_CFG.batch_size * 1 * DST_CFG.gradient_accumulation
                )
                ds_config.setdefault("optimizer", {})
                ds_config["optimizer"] = {
                    "type": "AdamW",
                    "params": {
                        "lr": DST_CFG.lr,
                        "weight_decay": 0.01,
                        "betas": [0.9, 0.95],
                    },
                }
                student_engine, opt, _, _ = deepspeed.initialize(
                    model=student,
                    config=ds_config,
                )
                student = student_engine  # wrapper: forward/backward/step sur engine
                print("[DISTILL] DeepSpeed ZeRO-2 + optimizer offload CPU actif")
        except Exception as e:
            print(f"[DISTILL] DeepSpeed indisponible ({e}) -> PyTorch AdamW")
            use_deepspeed = False
            student_engine = None

    if not use_deepspeed:
        opt = torch.optim.AdamW(student.parameters(), lr=DST_CFG.lr, weight_decay=0.01)
    texts = load_texts()
    print(f"[DISTILL] {len(texts)} textes")

    if step >= DST_CFG.max_steps:
        print(f"[DISTILL] Deja step {step} >= max_steps {DST_CFG.max_steps} — rien a faire.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / "distill_log.jsonl"
    T = DST_CFG.temperature
    # Reprise sans repasser les memes phrases: rotation de la liste pour commencer au prochain batch
    bs = DST_CFG.batch_size
    if not texts:
        print("[DISTILL] Aucun texte — arret.")
        return
    if text_cursor > 0:
        text_cursor = min(text_cursor, len(texts) - 1)
        texts = texts[text_cursor:] + texts[:text_cursor]
        print(f"[DISTILL] Liste rotatee de {text_cursor} textes — pas de reprise sur les memes prompts", flush=True)
        text_cursor = 0  # on sauvegardera le prochain offset relatif apres chaque batch

    for epoch in range(999):
        for i in range(0, len(texts), DST_CFG.batch_size):
            if step >= DST_CFG.max_steps:
                break
            batch = texts[i : i + DST_CFG.batch_size]
            if not batch:
                continue

            batch_log_snippets = []  # rempli si backend ollama
            ce_loss = None  # rempli si backend transformers (pour logs ce / kd)

            if backend == "transformers":
                # Encoder avec le tokenizer DU TEACHER uniquement : le ttok local peut avoir
                # vocab 128260 (+4 special) → ids >= 128256 crashent l'embedding CUDA du teacher.
                teacher_tok = _get_teacher_transformers_tokenizer()
                enc = teacher_tok(
                    batch,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to(dev)
                with torch.no_grad():
                    teacher_logits = _teacher_forward_transformers(enc, dev)
                teacher_logits = teacher_logits.to(dev)
                input_ids_dev = enc["input_ids"].to(dev)
                so = student(input_ids_dev, targets=input_ids_dev)
                student_logits = so["logits"]
                ce_loss = so["loss"]
                min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
                # KL teacher || student (distillation classique sur logits)
                kd = (
                    F.kl_div(
                        F.log_softmax(student_logits[..., :min_vocab] / T, dim=-1),
                        F.softmax(teacher_logits[..., :min_vocab] / T, dim=-1),
                        reduction="batchmean",
                    )
                    * (T * T)
                )
                loss = DST_CFG.alpha_kd * kd + DST_CFG.alpha_ce * ce_loss

            elif backend == "lmstudio":
                # Pas de logits alignés vocab → distillation séquence (CE sur prompt + suite générée)
                import requests
                loss = None
                n_sub = 0
                url = f"{api_url.rstrip('/')}/v1/completions"
                for text in batch:
                    prompt = text[:1500]
                    try:
                        r = requests.post(
                            url,
                            json={
                                "model": lmstudio_model,
                                "prompt": prompt,
                                "max_tokens": 128,
                                "temperature": 0.7,
                            },
                            timeout=120,
                        )
                        continuation = ""
                        if r.status_code == 200:
                            continuation = r.json()["choices"][0].get("text", "")
                    except Exception:
                        continuation = ""
                    full_text = prompt + continuation
                    enc = ttok(
                        full_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=False,
                    )
                    ids = enc.input_ids.to(dev)
                    if ids.size(1) < 2:
                        continue
                    so = student(ids, targets=ids)
                    loss = so["loss"] if loss is None else loss + so["loss"]
                    n_sub += 1
                if loss is None or n_sub == 0:
                    continue
                loss = loss / n_sub
                kd = torch.tensor(0.0, device=dev)

            elif backend == "ollama":
                loss = None
                batch_log_snippets = []  # pour JSONL + console enrichie
                if step == 0 and i == 0:
                    print("[DISTILL] 1er batch Ollama peut etre long (20B)...", flush=True)
                for bi, text in enumerate(batch):
                    prompt = text[:1500]
                    continuation = _ollama_generate(api_url, ollama_model, prompt, max_tokens=128)
                    if not (continuation or "").strip():
                        continuation = " "
                    # Une seule ligne = ce que le teacher envoie
                    t_line = (continuation or "").replace("\n", " ").replace("\r", " ").strip()
                    if len(t_line) > 600:
                        t_line = t_line[:600] + "..."
                    print(f"[TEACHER] {t_line}", flush=True)
                    cont_preview = _peek_text(continuation, 180, 100)
                    prompt_tail = _peek_text(prompt[-200:] if len(prompt) > 200 else prompt, 100, 60)
                    batch_log_snippets.append(
                        {
                            "prompt_tail": prompt_tail,
                            "continuation": cont_preview,
                            "continuation_len": len(continuation),
                        }
                    )
                    full_text = prompt + continuation
                    enc = ttok(
                        full_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=False,
                    )
                    ids = enc.input_ids.to(dev)
                    if ids.size(1) < 2:
                        continue
                    # CE uniquement sur la suite teacher, pas sur le prompt wiki (sinon article infini)
                    targets = _targets_only_continuation(ttok, full_text, len(prompt), ids)
                    so = student(ids, targets=targets)
                    loss_b = so["loss"]
                    loss = loss_b if loss is None else loss + loss_b
                if loss is None:
                    continue
                loss = loss / len(batch)
                kd = torch.tensor(0.0, device=dev)

            elif backend == "llamacpp":
                if not model_path or not Path(model_path).exists():
                    print("[DISTILL] --model-path required for llamacpp")
                    sys.exit(1)
                _ensure_llamacpp_cuda_dll_dirs()
                try:
                    from llama_cpp import Llama
                except ImportError:
                    print("[DISTILL] pip install llama-cpp-python")
                    sys.exit(1)
                except OSError as e:
                    print("[DISTILL] llama_cpp DLL load failed:", e)
                    print("  Windows+cu121: pip install nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1")
                    print("  Puis relance (distill enregistre les dossiers DLL automatiquement).")
                    sys.exit(1)
                llm = getattr(distill_online, "_llamacpp", None)
                cached_path = getattr(distill_online, "_llamacpp_path", None)
                cached_ngl = getattr(distill_online, "_llamacpp_n_gpu_layers", None)
                if (
                    llm is None
                    or cached_path != model_path
                    or cached_ngl != llamacpp_n_gpu_layers
                ):
                    if llamacpp_n_gpu_layers != 0:
                        print(f"[DISTILL] llamacpp n_gpu_layers={llamacpp_n_gpu_layers} (-1 = all VRAM)", flush=True)
                    use_mmap = not llamacpp_no_mmap
                    llm = None
                    ngl_effective = llamacpp_n_gpu_layers
                    last_err: Optional[BaseException] = None
                    # Ordre: params demandés → verbose pour voir stderr llama.cpp → CPU pour isoler OOM → no mmap Windows
                    for ngl, verb, mmap in (
                        (llamacpp_n_gpu_layers, llamacpp_verbose, use_mmap),
                        (llamacpp_n_gpu_layers, True, use_mmap),
                        (0, True, False),
                        (llamacpp_n_gpu_layers, True, False),
                    ):
                        try:
                            llm = _load_llamacpp_llm(
                                model_path, ngl, verbose=verb, use_mmap=mmap
                            )
                            ngl_effective = ngl
                            if ngl != llamacpp_n_gpu_layers:
                                print(
                                    f"[DISTILL] llamacpp chargé avec n_gpu_layers={ngl} (fallback). "
                                    "Si OOM: réduis --llamacpp-gpu-layers ou libère VRAM.",
                                    flush=True,
                                )
                            break
                        except ValueError as e:
                            last_err = e
                            if "Failed to load model" not in str(e):
                                raise
                            if not verb and ngl == llamacpp_n_gpu_layers:
                                print(
                                    "[DISTILL] Llama load failed — message llama.cpp (stderr) au prochain essai verbose:",
                                    flush=True,
                                )
                            continue
                    if llm is None:
                        print(
                            "[DISTILL] Impossible de charger le GGUF. Pistes:",
                            flush=True,
                        )
                        print(
                            "  - OOM: --llamacpp-gpu-layers 0 ou 20 (student déjà sur GPU).",
                            flush=True,
                        )
                        print(
                            "  - Windows: --llamacpp-no-mmap si mmap pose problème.",
                            flush=True,
                        )
                        print(
                            "  - llama-cpp 0.3.4 ancien: GGUF récent → ré-exporter avec llama.cpp aligné.",
                            flush=True,
                        )
                        raise last_err  # type: ignore[misc]
                    distill_online._llamacpp = llm
                    distill_online._llamacpp_path = model_path
                    distill_online._llamacpp_n_gpu_layers = ngl_effective
                loss = None
                kd = torch.tensor(0.0, device=dev)
                for text in batch:
                    prompt = text[:1500]
                    # Génération teacher puis CE student (même vocab si modèle aligné HF)
                    out = llm.create_completion(
                        prompt, max_tokens=128, temperature=0.7
                    )
                    continuation = (out or {}).get("choices", [{}])[0].get("text", "")
                    full_text = prompt + continuation
                    enc = ttok(
                        full_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=False,
                    )
                    ids = enc.input_ids.to(dev)
                    if ids.size(1) < 2:
                        continue
                    so = student(ids, targets=ids)
                    loss = so["loss"] if loss is None else loss + so["loss"]
                if loss is None:
                    continue
                loss = loss / max(len(batch), 1)
                # Si logits_all + eval disponible, on pourrait ajouter KD ici selon binding
            else:
                print(f"Unknown backend: {backend}")
                sys.exit(1)

            if use_deepspeed:
                student.backward(loss)
                student.step()
            else:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
            step += 1

            kd_val = kd.item() if isinstance(kd, torch.Tensor) else float(kd)
            ce_val = ce_loss.item() if isinstance(ce_loss, torch.Tensor) else None

            if backend == "transformers":
                # Logger kd et ce séparément (kd doit être > 0 si logits teacher OK)
                log_entry = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "step": step,
                    "batch_offset": i,
                    "loss": round(loss.item(), 4),
                    "kd": round(kd_val, 4),
                    "ce": round(ce_val, 4) if ce_val is not None else None,
                    "backend": backend,
                    "teacher_repo": getattr(_get_teacher_transformers_model, "_repo", None),
                }
                if step % max(1, getattr(DST_CFG, "log_interval", 10)) == 0 or step <= 3:
                    print(
                        f"[DISTILL {step}/{DST_CFG.max_steps}] loss={loss.item():.4f} "
                        f"kd={kd_val:.4f} ce={ce_val:.4f} backend=transformers",
                        flush=True,
                    )
                with open(logf, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                _distill_peek_student(student, ttok, dev, step, max_new=48)

            if backend == "ollama":
                # Un seul fichier ecrase a chaque batch (evite 50k x 2.5 Go)
                latest_path = MODEL_DIR / "distill_latest.pt"
                try:
                    if use_deepspeed:
                        student.module.save_checkpoint(
                            str(latest_path),
                            step,
                            extra={
                                "distill_backend": backend,
                                "teacher_vocab_size": vocab_size,
                                "overwrite": "each_step",
                                "text_cursor": (i + bs) % len(texts) if texts else 0,
                            },
                        )
                    else:
                        student.save_checkpoint(
                            str(latest_path),
                            step,
                            extra={
                                "distill_backend": backend,
                                "teacher_vocab_size": vocab_size,
                                "overwrite": "each_step",
                                "text_cursor": (i + bs) % len(texts) if texts else 0,
                            },
                        )
                    print(f"[DISTILL] Checkpoint ecrase -> {latest_path.name} (step {step})", flush=True)
                except Exception as e:
                    print(f"[DISTILL] Save latest failed: {e}", flush=True)

                # Log enrichi chaque step + apercu textes
                log_entry = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "step": step,
                    "batch_offset": i,
                    "loss": round(loss.item(), 4),
                    "kd": round(kd_val, 4),
                    "backend": backend,
                    "snippets": batch_log_snippets,
                }
                print(
                    f"[DISTILL {step}/{DST_CFG.max_steps}] loss={loss.item():.4f}",
                    flush=True,
                )
                with open(logf, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                # Apres chaque batch: le student "parle" un peu dans le terminal
                _distill_peek_student(student, ttok, dev, step, max_new=48)
            elif backend != "transformers" and step % 10 == 0:
                print(
                    f"[DISTILL {step}/{DST_CFG.max_steps}] loss={loss.item():.4f} kd={kd_val:.4f} backend={backend}",
                    flush=True,
                )
                with open(logf, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "step": step,
                                "loss": round(loss.item(), 4),
                                "kd": round(kd_val, 4),
                                "backend": backend,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            save_every = max(1, getattr(DST_CFG, "save_interval", 2000))
            if backend != "ollama" and step > 0 and step % save_every == 0:
                print(f"[DISTILL] Checkpoint step {step} -> distill_s{step}.pt", flush=True)
                if use_deepspeed:
                    tag = f"step_{step}"
                    student.save_checkpoint(str(MODEL_DIR), tag, client_state={"step": step})
                    raw = student.module
                    raw.save_checkpoint(
                        str(MODEL_DIR / f"distill_s{step}.pt"),
                        step,
                        extra={"distill_backend": backend, "teacher_vocab_size": vocab_size},
                    )
                else:
                    student.save_checkpoint(
                        str(MODEL_DIR / f"distill_s{step}.pt"),
                        step,
                        extra={"distill_backend": backend, "teacher_vocab_size": vocab_size},
                    )

        if step >= DST_CFG.max_steps:
            break

    if use_deepspeed:
        student.save_checkpoint(str(MODEL_DIR), "final", client_state={"step": step})
        student.module.save_checkpoint(
            str(MODEL_DIR / "distill_final.pt"),
            step,
            extra={"distill_backend": backend, "teacher_vocab_size": vocab_size},
        )
    else:
        student.save_checkpoint(
            str(MODEL_DIR / "distill_final.pt"),
            step,
            extra={"distill_backend": backend, "teacher_vocab_size": vocab_size},
        )
    print(f"[DISTILL] Fini — {step} steps (backend={backend})")


# ── Offline generate/train (transformers only for logits) ──────────

def generate_teacher_targets(max_batches=None, backend: str = "transformers", model_path: str = ""):
    if backend != "transformers":
        print("[OFFLINE-GEN] Logits offline only implemented for --backend transformers")
        print("[OFFLINE-GEN] For ollama/lmstudio use online mode with sequence CE.")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = _get_teacher_transformers_model()
    teacher_tok = _get_teacher_transformers_tokenizer()
    texts = load_texts(max_docs=DST_CFG.max_samples)
    TEACHER_LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    batch_idx = 0
    start_time = time.time()
    for i in range(0, len(texts), OFFLINE_BATCH_SIZE):
        if max_batches and batch_idx >= max_batches:
            break
        batch_texts = texts[i : i + OFFLINE_BATCH_SIZE]
        if not batch_texts:
            continue
        enc = teacher_tok(
            batch_texts,
            return_tensors="pt",
            max_length=OFFLINE_MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to(dev)
        with torch.no_grad():
            logits = teacher(**enc).logits
            probs = F.softmax(logits / DST_CFG.temperature, dim=-1)
            top_probs, top_ids = torch.topk(probs, TOP_K_LOGITS, dim=-1)
        torch.save(
            {
                "input_ids": enc.input_ids.cpu(),
                "attention_mask": enc.attention_mask.cpu(),
                "top_ids": top_ids.cpu(),
                "top_probs": top_probs.cpu().to(torch.float16),
            },
            TEACHER_LOGITS_DIR / f"batch_{batch_idx:06d}.pt",
        )
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"[OFFLINE-GEN] Batch {batch_idx} ETA {(time.time()-start_time)/batch_idx*(len(texts)//OFFLINE_BATCH_SIZE-batch_idx)/60:.1f} min")
    with open(TEACHER_LOGITS_DIR / "meta.json", "w") as f:
        json.dump(
            {
                "n_batches": batch_idx,
                "teacher_model": getattr(
                    _get_teacher_transformers_model, "_repo", DST_CFG.teacher_model
                ),
                "teacher_vocab_size": getattr(teacher_tok, "vocab_size", len(teacher_tok)),
                "backend": backend,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"[OFFLINE-GEN] Done {batch_idx} batches -> {TEACHER_LOGITS_DIR}")


class TeacherLogitsDataset(Dataset):
    def __init__(self, logits_dir=TEACHER_LOGITS_DIR):
        self.logits_dir = Path(logits_dir)
        self.batch_files = sorted(self.logits_dir.glob("batch_*.pt"))
        if not self.batch_files:
            raise FileNotFoundError(f"No batch files in {logits_dir}")
        self.meta = {}
        mp = self.logits_dir / "meta.json"
        if mp.exists():
            self.meta = json.load(open(mp))

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        return torch.load(self.batch_files[idx], map_location="cpu", weights_only=True)


def reconstruct_distribution(top_ids, top_probs, vocab_size, device):
    B, T, K = top_ids.shape
    full_probs = torch.zeros(B, T, vocab_size, device=device)
    full_probs.scatter_(2, top_ids.to(device), top_probs.to(device))
    remaining_mass = (1.0 - top_probs.sum(dim=-1, keepdim=True).to(device)).clamp(min=0)
    uniform_prob = remaining_mass / max(vocab_size - K, 1)
    mask = torch.ones(B, T, vocab_size, device=device)
    mask.scatter_(2, top_ids.to(device), 0)
    full_probs = full_probs + mask * uniform_prob
    return full_probs / full_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)


def train_from_targets(resume_checkpoint=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda" and TRN_CFG.mixed_precision
    dt = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    ds = TeacherLogitsDataset()
    meta_vocab = None
    if ds.meta.get("teacher_vocab_size"):
        meta_vocab = ds.meta["teacher_vocab_size"]
    if resume_checkpoint:
        student, ckpt = LLMMaison.load_checkpoint(resume_checkpoint, dev)
        if getattr(DST_CFG, "gradient_checkpointing", False) and hasattr(student, "enable_gradient_checkpointing"):
            student.enable_gradient_checkpointing()
        start_step = ckpt.get("step", 0)
    else:
        cfg = MDL_CFG
        if meta_vocab:
            cfg = _student_config_for_teacher_vocab(meta_vocab)
        student = LLMMaison(cfg).to(dev)
        if getattr(DST_CFG, "gradient_checkpointing", False) and hasattr(student, "enable_gradient_checkpointing"):
            student.enable_gradient_checkpointing()
        start_step = 0
    dc, nd = [], []
    for n, p in student.named_parameters():
        (dc if p.dim() >= 2 else nd).append(p)
    opt = torch.optim.AdamW(
        [{"params": dc, "weight_decay": 0.01}, {"params": nd, "weight_decay": 0.0}],
        lr=DST_CFG.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    logf = LOG_DIR / "distill_offline_log.jsonl"
    step = start_step
    Ttemp = DST_CFG.temperature
    V = student.cfg.vocab_size

    while step < DST_CFG.max_steps:
        for batch_data in ds:
            if step >= DST_CFG.max_steps:
                break
            input_ids = batch_data["input_ids"].to(dev)
            top_ids = batch_data["top_ids"]
            top_probs = batch_data["top_probs"].float()
            teacher_probs = reconstruct_distribution(top_ids, top_probs, V, dev)
            with torch.cuda.amp.autocast(enabled=amp, dtype=dt):
                out = student(input_ids, targets=input_ids)
                student_log_probs = F.log_softmax(out["logits"] / Ttemp, dim=-1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * Ttemp * Ttemp
                loss = DST_CFG.alpha_kd * kd_loss + DST_CFG.alpha_ce * out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"[OFFLINE-TRAIN {step}] loss={loss.item():.4f}")
                with open(logf, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(loss.item(), 4)}) + "\n")
            save_every = max(1, getattr(DST_CFG, "save_interval", 2000))
            if step > 0 and step % save_every == 0:
                student.save_checkpoint(str(MODEL_DIR / f"distill_offline_s{step}.pt"), step, opt)
        # re-parcourir le dataset (plusieurs epochs) jusqu'à max_steps
    student.save_checkpoint(str(MODEL_DIR / "distill_offline_final.pt"), step)
    print(f"[OFFLINE-TRAIN] Done {step} steps")


# Repos publics (non gated) avec tokenizer Llama 3 — meta-llama/* exige login + licence
TOKENIZER_DOWNLOAD_FALLBACKS = [
    "Xenova/llama3-tokenizer",   # dedie tokenizer, pas de gate
    "unsloth/llama-3-8b",       # tokenizer identique famille Llama 3
]


def download_teacher_tokenizer(repo_id: Optional[str], dest: Path) -> None:
    """Telecharge uniquement les fichiers tokenizer (pas le modele) dans dest."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[DISTILL] pip install huggingface_hub")
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
    ]
    candidates = []
    if repo_id:
        candidates.append(repo_id)
    for r in TOKENIZER_DOWNLOAD_FALLBACKS:
        if r not in candidates:
            candidates.append(r)

    last_err = None
    for rid in candidates:
        print(f"[DISTILL] Download tokenizer only from {rid} -> {dest}")
        try:
            snapshot_download(
                repo_id=rid,
                local_dir=str(dest),
                allow_patterns=patterns,
                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
            )
        except Exception as e:
            last_err = e
            print(f"[DISTILL] Echec {rid}: {e}")
            continue
        if (dest / "tokenizer.json").exists() or (dest / "tokenizer.model").exists():
            print(f"[DISTILL] OK depuis {rid} — relance: python -m training.distill --backend ollama")
            return
        last_err = RuntimeError("tokenizer.json absent apres download")

    print("[DISTILL] Tous les repos ont echoue. meta-llama/* = gated -> huggingface-cli login + licence,")
    print("  ou copie manuelle dans teacher_tokenizer/ (voir teacher_tokenizer/README.md)")
    if last_err:
        raise last_err
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Knowledge Distillation")
    p.add_argument(
        "--download-teacher-tokenizer",
        nargs="?",
        const="Xenova/llama3-tokenizer",
        default=None,
        metavar="REPO",
        help="Telecharge seulement le tokenizer (defaut: Xenova/llama3-tokenizer, public). Ex: --download-teacher-tokenizer unsloth/llama-3-8b",
    )
    p.add_argument("--mode", choices=["online", "offline"], default="online")
    p.add_argument("--phase", choices=["generate", "train"])
    p.add_argument("--backend", choices=["transformers", "ollama", "lmstudio", "llamacpp"], default="transformers")
    p.add_argument(
        "--teacher-model",
        type=str,
        default="",
        metavar="REPO",
        help="HF repo du teacher (transformers). Defaut: DistillConfig.teacher_model. Ex: unsloth/llama-3-8b-Instruct",
    )
    p.add_argument("--api-url", type=str, default="", help="Base URL for ollama/lmstudio (default localhost)")
    p.add_argument("--model-path", type=str, default="", help="Path to .gguf for llamacpp")
    p.add_argument(
        "--llamacpp-gpu-layers",
        type=int,
        default=-1,
        metavar="N",
        help="llamacpp: layers on GPU (-1=all, 0=CPU only). Defaut -1 pour utiliser CUDA si wheel GPU installee.",
    )
    p.add_argument(
        "--llamacpp-verbose",
        action="store_true",
        help="llamacpp: afficher stderr llama.cpp (utile si Failed to load model).",
    )
    p.add_argument(
        "--llamacpp-no-mmap",
        action="store_true",
        help="llamacpp: use_mmap=False (souvent utile si chargement echoue sur Windows).",
    )
    p.add_argument("--ollama-model", type=str, default="llama3")
    p.add_argument("--lmstudio-model", type=str, default="llama3")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--profile", choices=["small", "medium", "large"], default=None,
                   help="Doit etre en tete de commande ou via LLM_PROFILE pour affecter MDL_CFG")
    p.add_argument("--no-deepspeed", action="store_true", help="Forcer PyTorch AdamW (sans ZeRO)")
    p.add_argument("--save-every", type=int, default=None, metavar="N",
                   help="Checkpoint tous les N steps (defaut: DistillConfig.save_interval)")
    p.add_argument("--debug-ollama", action="store_true", help="Test API Ollama + ecrit logs/ollama_debug.txt puis quitte")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Repartir de zero : ignore distill_latest.pt / distill_final.pt, student initialise au hasard, step=0.",
    )
    args = p.parse_args()
    if args.save_every is not None and args.save_every > 0:
        DST_CFG.save_interval = args.save_every
        print(f"[DISTILL] save_interval={DST_CFG.save_interval}")

    if args.download_teacher_tokenizer is not None:
        download_teacher_tokenizer(args.download_teacher_tokenizer, TEACHER_TOKENIZER_DIR)
        return

    if args.debug_ollama:
        api = _api_base(args.api_url, "ollama")
        debug_ollama(api, args.ollama_model)
        return

    if getattr(args, "teacher_model", "").strip():
        DST_CFG.teacher_model = args.teacher_model.strip()
        print(f"[DISTILL] teacher_model={DST_CFG.teacher_model}")

    if args.mode == "online":
        distill_online(
            backend=args.backend,
            api_url=args.api_url,
            model_path=args.model_path,
            ollama_model=args.ollama_model,
            lmstudio_model=args.lmstudio_model,
            use_deepspeed_override=False if getattr(args, "no_deepspeed", False) else None,
            llamacpp_n_gpu_layers=args.llamacpp_gpu_layers,
            llamacpp_verbose=getattr(args, "llamacpp_verbose", False),
            llamacpp_no_mmap=getattr(args, "llamacpp_no_mmap", False),
            fresh=getattr(args, "fresh", False),
        )
    elif args.mode == "offline":
        if args.phase == "generate":
            generate_teacher_targets(args.max_batches, backend=args.backend, model_path=args.model_path)
        elif args.phase == "train":
            train_from_targets(args.resume)
        else:
            print("Use --phase generate or --phase train")
            sys.exit(1)


if __name__ == "__main__":
    main()
