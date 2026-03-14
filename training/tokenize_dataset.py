"""
Tokenisation + packing optionnel pour sequences longues.
Le distill actuel consomme du texte brut via load_texts() + troncature ;
ce script peut pre-tokeniser en .pt ou jsonl d'ids si besoin.

Usage:
  python -m training.tokenize_dataset --input data_raw --output data_processed/packed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Tokenize JSONL texte -> sequences (optionnel)")
    p.add_argument("--input", type=str, default="data_raw", help="Repertoire jsonl")
    p.add_argument("--output", type=str, default="data_processed/packed", help="Sortie")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-docs", type=int, default=1000, help="Limite pour test")
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("pip install transformers")
        sys.exit(1)

    from config import TEACHER_TOKENIZER_DIR

    tok = AutoTokenizer.from_pretrained(str(TEACHER_TOKENIZER_DIR), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    buffer: list[int] = []
    n_seq = 0
    max_docs = args.max_docs

    for fp in sorted(inp.glob("*.jsonl")):
        if max_docs <= 0:
            break
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if max_docs <= 0:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or ""
                except json.JSONDecodeError:
                    continue
                if len(text) < 50:
                    continue
                ids = tok.encode(text, add_special_tokens=False)
                buffer.extend(ids)
                max_docs -= 1
                while len(buffer) >= args.seq_len + 1:
                    chunk = buffer[: args.seq_len + 1]
                    buffer = buffer[args.seq_len :]
                    # Exemple : ecrire une ligne json d'ids (alleger selon besoin)
                    with open(out / f"seq_{n_seq:08d}.json", "w") as fo:
                        json.dump({"ids": chunk}, fo)
                    n_seq += 1

    print(f"[tokenize_dataset] {n_seq} sequences -> {out}")


if __name__ == "__main__":
    main()
