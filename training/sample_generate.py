"""
Voir ce que dit le student (checkpoint distill) sans passer par le BPE.
Usage:
  python -m training.sample_generate
  python -m training.sample_generate --prompt "Bonjour, raconte une blague courte."
  python -m training.sample_generate --ckpt checkpoints/distill_final.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config import MODEL_DIR, TEACHER_TOKENIZER_DIR
from model.transformer import LLMMaison
from model.generate import generate, GenConfig


def main():
    p = argparse.ArgumentParser(description="Generation avec checkpoint distill + tokenizer teacher")
    p.add_argument("--ckpt", type=str, default=None, help="Chemin .pt (defaut: distill_latest.pt puis distill_final.pt)")
    p.add_argument("--prompt", type=str, default="Bonjour, comment vas-tu ?", help="Prompt texte")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if not ckpt_path or not ckpt_path.exists():
        for name in ("distill_latest.pt", "distill_final.pt", "pretrain_final.pt"):
            c = MODEL_DIR / name
            if c.exists():
                ckpt_path = c
                break
    if not ckpt_path or not ckpt_path.exists():
        print("[!] Aucun checkpoint. Lance la distill ou pretrain d'abord.")
        sys.exit(1)

    print(f"[LOAD] {ckpt_path}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = LLMMaison.load_checkpoint(str(ckpt_path), dev)
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TEACHER_TOKENIZER_DIR), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(dev)
    if input_ids.size(1) < 1:
        print("[!] Prompt vide apres tokenize")
        sys.exit(1)

    cfg = GenConfig(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=50, top_p=0.9)
    if tok.eos_token_id is not None:
        cfg.stop_tokens = [tok.eos_token_id]

    with torch.inference_mode():
        out_ids = generate(model, input_ids, cfg)

    # Decode seulement les nouveaux tokens
    new_tokens = out_ids[0, input_ids.size(1) :].tolist()
    text = tok.decode(new_tokens, skip_special_tokens=True)
    print()
    print("--- Prompt ---")
    print(args.prompt)
    print("--- Student ---")
    print(text or "(rien genere)")
    print()


if __name__ == "__main__":
    main()
