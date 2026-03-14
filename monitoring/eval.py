"""
monitoring/eval.py — Évaluation automatique du modèle.
Perplexité, diversité de génération, comparaison de checkpoints.
Usage: python -m monitoring.eval --checkpoint checkpoints/best.pt
"""
import json
import math
import argparse
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from config import (
    CHECKPOINT_DIR, PROCESSED_DIR, LOG_DIR, TOKENIZER_DIR,
    MDL_CFG, TRN_CFG
)
from model.transformer import LLMMaison
from model.generate import generate, GenConfig
from tokenizer import BPETokenizer


EVAL_PROMPTS_FR = [
    "La France est un pays",
    "L'intelligence artificielle permet de",
    "Le réchauffement climatique est causé par",
    "En mathématiques, le théorème de Pythagore",
    "La Révolution française a commencé en",
    "Les avantages du machine learning sont",
    "Pour apprendre à programmer, il faut",
    "La philosophie de Descartes repose sur",
    "L'économie mondiale est influencée par",
    "Les neurosciences étudient",
]

EVAL_PROMPTS_EN = [
    "Artificial intelligence is transforming",
    "The history of computing began",
    "Climate change affects the planet by",
    "In physics, quantum mechanics describes",
    "The internet has revolutionized",
    "Machine learning models can",
    "Programming languages are designed to",
    "The human brain processes information",
    "Economic growth depends on",
    "Scientific research requires",
]

EVAL_PROMPTS = EVAL_PROMPTS_FR + EVAL_PROMPTS_EN


class ValDataset(Dataset):
    """Validation dataset from val.bin or last 5% of train.bin."""
    
    def __init__(self, path=None, seq_len=None, val_ratio=0.05):
        self.sl = seq_len or TRN_CFG.seq_len
        val_path = path or str(PROCESSED_DIR / "val.bin")
        train_path = str(PROCESSED_DIR / "train.bin")
        
        if Path(val_path).exists():
            self.data = np.memmap(val_path, dtype=np.uint16, mode="r")
            print(f"[EVAL] Loaded val.bin: {len(self.data):,} tokens")
        elif Path(train_path).exists():
            full_data = np.memmap(train_path, dtype=np.uint16, mode="r")
            split_idx = int(len(full_data) * (1 - val_ratio))
            self.data = full_data[split_idx:]
            print(f"[EVAL] Using last {val_ratio*100:.0f}% of train.bin: {len(self.data):,} tokens")
        else:
            raise FileNotFoundError(f"No validation data found at {val_path} or {train_path}")
        
        self.n = len(self.data) // (self.sl + 1)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        s = i * (self.sl + 1)
        c = self.data[s:s + self.sl + 1].astype(np.int64)
        return torch.from_numpy(c[:-1]), torch.from_numpy(c[1:])


def compute_perplexity(model, device="cpu", batch_size=8, max_batches=None):
    """Compute perplexity on validation set."""
    try:
        ds = ValDataset()
    except FileNotFoundError as e:
        print(f"[EVAL] {e}")
        return None
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x, targets=y)
            loss = out["loss"].item()
            n_tokens = y.numel()
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "batches": i + 1 if max_batches else len(dl)
    }


def extract_trigrams(text):
    """Extract word trigrams from text."""
    words = text.lower().split()
    if len(words) < 3:
        return []
    return [tuple(words[i:i+3]) for i in range(len(words) - 2)]


def compute_diversity(model, tokenizer, device="cpu", max_new_tokens=200):
    """Compute generation diversity metrics."""
    model.eval()
    cfg = GenConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        rep_penalty=1.0,
        use_cache=True
    )
    
    results = []
    all_trigrams = []
    loop_detections = 0
    
    for prompt in EVAL_PROMPTS:
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = generate(model, inp, cfg)
        
        generated_ids = out[0, len(ids):].tolist()
        text = tokenizer.decode(generated_ids)
        
        trigrams = extract_trigrams(text)
        all_trigrams.extend(trigrams)
        
        trigram_counts = Counter(trigrams)
        has_loop = any(c > 3 for c in trigram_counts.values())
        if has_loop:
            loop_detections += 1
        
        results.append({
            "prompt": prompt,
            "generated": text[:200] + ("..." if len(text) > 200 else ""),
            "length": len(generated_ids),
            "has_loop": has_loop
        })
    
    unique_trigrams = len(set(all_trigrams))
    total_trigrams = len(all_trigrams)
    diversity_score = unique_trigrams / total_trigrams if total_trigrams > 0 else 0
    
    return {
        "diversity_score": diversity_score,
        "unique_trigrams": unique_trigrams,
        "total_trigrams": total_trigrams,
        "loop_detections": loop_detections,
        "total_prompts": len(EVAL_PROMPTS),
        "samples": results[:5]
    }


def load_model(checkpoint_path, device="cpu"):
    """Load model from checkpoint."""
    model, ckpt = LLMMaison.load_checkpoint(checkpoint_path, device)
    model.eval()
    step = ckpt.get("step", 0)
    return model, step


def evaluate_checkpoint(checkpoint_path, device="cpu", tokenizer=None, max_batches=100):
    """Full evaluation of a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"{'='*60}")
    
    model, step = load_model(checkpoint_path, device)
    print(f"[EVAL] Loaded checkpoint at step {step}")
    print(f"[EVAL] Model: {model.count_params()/1e6:.1f}M params")
    
    results = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "device": device
    }
    
    print("\n[EVAL] Computing perplexity...")
    ppl = compute_perplexity(model, device, max_batches=max_batches)
    if ppl:
        results["perplexity"] = ppl
        print(f"  Perplexity: {ppl['perplexity']:.2f}")
        print(f"  Avg Loss:   {ppl['avg_loss']:.4f}")
        print(f"  Tokens:     {ppl['total_tokens']:,}")
    
    if tokenizer:
        print("\n[EVAL] Computing diversity...")
        div = compute_diversity(model, tokenizer, device)
        results["diversity"] = div
        print(f"  Diversity Score: {div['diversity_score']:.3f}")
        print(f"  Unique/Total:    {div['unique_trigrams']}/{div['total_trigrams']}")
        print(f"  Loop detections: {div['loop_detections']}/{div['total_prompts']}")
        
        print("\n[EVAL] Sample generations:")
        for s in div["samples"][:3]:
            print(f"  Prompt: {s['prompt'][:40]}...")
            print(f"  Output: {s['generated'][:80]}...")
            print()
    
    return results


def compare_checkpoints(ckpt1_path, ckpt2_path, device="cpu", tokenizer=None):
    """Compare two checkpoints side by side."""
    r1 = evaluate_checkpoint(ckpt1_path, device, tokenizer)
    r2 = evaluate_checkpoint(ckpt2_path, device, tokenizer)
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Checkpoint 1':>15} {'Checkpoint 2':>15} {'Delta':>12}")
    print("-"*70)
    
    s1, s2 = r1.get("step", 0), r2.get("step", 0)
    print(f"{'Step':<25} {s1:>15,} {s2:>15,} {s2-s1:>+12,}")
    
    if "perplexity" in r1 and "perplexity" in r2:
        p1 = r1["perplexity"]["perplexity"]
        p2 = r2["perplexity"]["perplexity"]
        print(f"{'Perplexity':<25} {p1:>15.2f} {p2:>15.2f} {p2-p1:>+12.2f}")
        
        l1 = r1["perplexity"]["avg_loss"]
        l2 = r2["perplexity"]["avg_loss"]
        print(f"{'Avg Loss':<25} {l1:>15.4f} {l2:>15.4f} {l2-l1:>+12.4f}")
    
    if "diversity" in r1 and "diversity" in r2:
        d1 = r1["diversity"]["diversity_score"]
        d2 = r2["diversity"]["diversity_score"]
        print(f"{'Diversity':<25} {d1:>15.3f} {d2:>15.3f} {d2-d1:>+12.3f}")
        
        lo1 = r1["diversity"]["loop_detections"]
        lo2 = r2["diversity"]["loop_detections"]
        print(f"{'Loop Detections':<25} {lo1:>15} {lo2:>15} {lo2-lo1:>+12}")
    
    return r1, r2


def save_eval_log(results):
    """Append results to eval log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")
    print(f"\n[EVAL] Results saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument("--compare", type=str, nargs=2, help="Compare two checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches for perplexity")
    parser.add_argument("--no-diversity", action="store_true", help="Skip diversity eval")
    args = parser.parse_args()
    
    tokenizer = None
    if not args.no_diversity:
        try:
            tokenizer = BPETokenizer.load(str(TOKENIZER_DIR))
            print(f"[EVAL] Loaded tokenizer: {tokenizer.size} tokens")
        except Exception as e:
            print(f"[EVAL] Could not load tokenizer: {e}")
            print("[EVAL] Diversity evaluation will be skipped")
    
    if args.compare:
        r1, r2 = compare_checkpoints(args.compare[0], args.compare[1], args.device, tokenizer)
        save_eval_log(r1)
        save_eval_log(r2)
    elif args.checkpoint:
        results = evaluate_checkpoint(args.checkpoint, args.device, tokenizer, args.max_batches)
        save_eval_log(results)
    else:
        ckpts = sorted(CHECKPOINT_DIR.glob("*.pt"))
        if not ckpts:
            print(f"[EVAL] No checkpoints found in {CHECKPOINT_DIR}")
            print("Usage: python -m monitoring.eval --checkpoint path/to/model.pt")
            return
        
        latest = ckpts[-1]
        print(f"[EVAL] Using latest checkpoint: {latest}")
        results = evaluate_checkpoint(str(latest), args.device, tokenizer, args.max_batches)
        save_eval_log(results)


if __name__ == "__main__":
    main()
