"""
training/finetune.py — Fine-tuning conversations (LoRA par defaut).
Usage: python -m training.finetune [--full] [--profile large]
"""
import os
import sys
from pathlib import Path

for i, a in enumerate(sys.argv):
    if a == "--profile" and i + 1 < len(sys.argv):
        os.environ["LLM_PROFILE"] = sys.argv[i + 1]
        break

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from torch.utils.data import DataLoader

from config import FT_CFG, MDL_CFG, MODEL_DIR, LOG_DIR, CONV_DIR, TOKENIZER_DIR, PROFILE
from model.transformer import LLMMaison
from model.lora import apply_lora, save_lora
from training.dataset import ConvDataset, collate_pad
from tokenizer.bpe import BPETokenizer


def load_convs():
    convs = []
    for f in sorted(CONV_DIR.glob("*.jsonl")):
        for line in open(f, "r", encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(c, list) and len(c) >= 2:
                convs.append(c)
    return convs


def _config_match(ckpt_cfg, current_cfg):
    if not ckpt_cfg:
        return False
    keys = ("d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "d_ff")
    for k in keys:
        if ckpt_cfg.get(k) != getattr(current_cfg, k, None):
            return False
    return True


def finetune(use_lora=True):
    # 3B sur 4080 : full finetune = OOM — LoRA toujours
    if not use_lora:
        print("[FT] Full finetune 3B = OOM sur 4080. LoRA force.")
        use_lora = True

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = MODEL_DIR / "pretrain_final.pt"
    if not ckpt_path.exists():
        ckpt_path = MODEL_DIR / "distill_final.pt"
    if not ckpt_path.exists():
        print("[FT] Pas de checkpoint. Lance training.pretrain d'abord.")
        return

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_cfg = ck.get("config")
    if not _config_match(ckpt_cfg, MDL_CFG):
        print(f"[FT] Config mismatch: checkpoint d_model={ckpt_cfg.get('d_model')} layers={ckpt_cfg.get('n_layers')}")
        print(f"[FT] vs current profile={PROFILE} d_model={MDL_CFG.d_model} n_layers={MDL_CFG.n_layers}")
        print("[FT] Charge un checkpoint du meme profil ou lance pretrain avec --profile correspondant.")
        sys.exit(1)

    model, _ = LLMMaison.load_checkpoint(str(ckpt_path), dev)
    if getattr(FT_CFG, "gradient_checkpointing", False) and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    tok = BPETokenizer.load(str(TOKENIZER_DIR))
    if use_lora:
        apply_lora(model, FT_CFG.lora_targets, FT_CFG.lora_r, FT_CFG.lora_alpha, FT_CFG.lora_dropout)

    convs = load_convs()
    if not convs:
        print(f"[FT] Pas de conversations dans {CONV_DIR}")
        return

    ds = ConvDataset(convs, tok, FT_CFG.max_seq_len)
    dl = DataLoader(ds, batch_size=FT_CFG.batch_size, shuffle=True, collate_fn=collate_pad, drop_last=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=FT_CFG.lr, weight_decay=0.01)
    logf = LOG_DIR / "finetune_log.jsonl"
    step = 0
    model.train()

    for ep in range(FT_CFG.epochs):
        el = 0.0
        nb = 0
        for bx, by in dl:
            bx, by = bx.to(dev), by.to(dev)
            loss = model(bx, targets=by)["loss"] / FT_CFG.gradient_accumulation
            loss.backward()
            if (step + 1) % FT_CFG.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            el += loss.item() * FT_CFG.gradient_accumulation
            nb += 1
            step += 1
            if step % 10 == 0:
                a = el / nb
                print(f"[FT E{ep+1} S{step}] loss={a:.4f} profile={PROFILE}")
                with open(logf, "a") as f:
                    f.write(json.dumps({"epoch": ep + 1, "step": step, "loss": round(a, 4), "profile": PROFILE}) + "\n")
        print(f"[FT] Epoch {ep+1} loss={el/max(nb,1):.4f}")

    if use_lora:
        save_lora(model, str(MODEL_DIR / "lora.pt"))
    else:
        model.save_checkpoint(str(MODEL_DIR / "finetune_full.pt"), step, extra={"profile": PROFILE})
    print("[FT] Termine")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--profile", choices=["small", "medium", "large"], default=None,
                   help="Doit matcher le checkpoint (sinon exit).")
    args, _ = p.parse_known_args()
    finetune(not args.full)
