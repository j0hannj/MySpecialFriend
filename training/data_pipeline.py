"""
Pipeline de mix multi-sources -> JSONL data_raw/*.jsonl (champ "text").
Compatible avec load_texts() dans distill.py.

Usage:
  pip install datasets pyyaml
  python -m training.data_pipeline --config training/mix_config.yaml --max-docs 50000
  python -m training.data_pipeline --list-sources   # dry run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

# ROOT = repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: Optional[str] = None) -> dict:
    import yaml
    p = Path(path or ROOT / "training" / "mix_config.yaml")
    if not p.is_file():
        p = ROOT / "mix_config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quality_filter(text: str) -> bool:
    if not text or len(text) < 100:
        return False
    words = text.split()
    if len(words) < 20:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return False
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.6:
        return False
    lines = text.split("\n")
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
    if avg_line_len < 20 and len(lines) > 10:
        return False
    return True


def dedup_filter(seen_hashes: set, text: str) -> bool:
    h = hashlib.md5(text[:500].encode("utf-8", errors="ignore")).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True


def _registry() -> Dict[str, Callable[[], Iterator[str]]]:
    from training.sources import (
        stream_books,
        stream_code,
        stream_conversations,
        stream_instructions,
        stream_web,
        stream_wikipedia,
    )

    return {
        "conversations": stream_conversations,
        "books_articles": stream_books,
        "wikipedia": stream_wikipedia,
        "instructions_qa": stream_instructions,
        "code": stream_code,
        "web_crawl": stream_web,
    }


def build_mixed_stream(
    config_path: Optional[str] = None,
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
    use_quality_filter: bool = True,
    use_dedup: bool = True,
    seed: int = 42,
) -> Iterator[tuple[str, str]]:
    """
    Yields (source_name, text) selon les poids du mix.
    Les sources s'epuisent une par une ; poids renormalises.
    """
    config = load_config(config_path)
    mix = config.get("mix") or {}
    rng = random.Random(seed)
    registry = _registry()
    # Iterateurs par source (un seul cycle chacun pour simplifier — recharger pour plus)
    iterators: Dict[str, Iterator[str]] = {}
    weights: Dict[str, float] = {}
    for name, w in mix.items():
        if name not in registry or w <= 0:
            continue
        iterators[name] = registry[name]()
        weights[name] = float(w)

    if not iterators:
        logger.error("Aucune source active dans mix_config.")
        return

    source_names = list(iterators.keys())
    source_weights = [weights[n] for n in source_names]
    exhausted = set()
    seen_hashes = set()
    total_chars = 0
    doc_count = 0
    max_c = max_chars or (config.get("max_tokens") or 10_000_000) * 4

    while len(exhausted) < len(iterators):
        if max_docs is not None and doc_count >= max_docs:
            break
        if total_chars >= max_c:
            break
        chosen = rng.choices(source_names, weights=source_weights, k=1)[0]
        if chosen in exhausted:
            continue
        try:
            doc = next(iterators[chosen])
        except StopIteration:
            exhausted.add(chosen)
            idx = source_names.index(chosen)
            source_weights[idx] = 0.0
            s = sum(source_weights)
            if s > 0:
                source_weights = [w / s for w in source_weights]
            logger.info("Source epuisee: %s", chosen)
            continue
        if not doc or len(doc) < 50:
            continue
        if use_quality_filter and not quality_filter(doc):
            continue
        if use_dedup and not dedup_filter(seen_hashes, doc):
            continue
        total_chars += len(doc)
        doc_count += 1
        yield chosen, doc


def write_jsonl(
    config_path: Optional[str] = None,
    max_docs: int = 10_000,
    shard_size: int = 5000,
    callback: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    config = load_config(config_path)
    out_dir = ROOT / (config.get("output_dir") or "data_raw")
    prefix = config.get("output_prefix") or "mixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    shard_idx = 0
    lines_in_shard = 0
    f = None

    def open_shard():
        nonlocal f, shard_idx, lines_in_shard
        if f:
            f.close()
        path = out_dir / f"{prefix}_{shard_idx:05d}.jsonl"
        f = open(path, "w", encoding="utf-8")
        paths.append(path)
        lines_in_shard = 0
        logger.info("Ecriture %s", path)

    open_shard()
    source_counts: Dict[str, int] = {}

    try:
        for source_name, text in build_mixed_stream(config_path=config_path, max_docs=max_docs):
            rec = {"text": text, "source": source_name}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            lines_in_shard += 1
            if lines_in_shard >= shard_size:
                shard_idx += 1
                open_shard()
            if callback and sum(source_counts.values()) % 5000 == 0:
                callback(str(source_counts))
    finally:
        if f:
            f.close()

    logger.info("Termine. Fichiers: %s — counts %s", paths, source_counts)
    return paths


def main():
    p = argparse.ArgumentParser(description="Mix multi-sources -> data_raw JSONL")
    p.add_argument("--config", type=str, default=None, help="Chemin mix_config.yaml")
    p.add_argument("--max-docs", type=int, default=20_000, help="Nombre max de documents a ecrire")
    p.add_argument("--shard-size", type=int, default=5000, help="Lignes par fichier jsonl")
    p.add_argument("--list-sources", action="store_true", help="Affiche les sources et quitte")
    p.add_argument("--no-quality-filter", action="store_true")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()

    if args.list_sources:
        reg = _registry()
        cfg = load_config(args.config)
        mix = cfg.get("mix") or {}
        print("Sources enregistrees:")
        for k in reg:
            print(f"  {k}: weight={mix.get(k, 0)}")
        return

    write_jsonl(
        config_path=args.config,
        max_docs=args.max_docs,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
