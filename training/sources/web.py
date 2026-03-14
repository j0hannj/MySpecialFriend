"""Web crawl — OSCAR FR (subset) ou fallback leger."""
import logging
from typing import Iterator

logger = logging.getLogger(__name__)


def _try_load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
        return load_dataset(*args, **kwargs)
    except Exception as e:
        logger.warning("datasets.load_dataset unavailable: %s", e)
        return None


def stream_web() -> Iterator[str]:
    # OSCAR-2301 fr — peut necessiter auth ou etre volumineux
    try:
        ds = _try_load_dataset("oscar-corpus/OSCAR-2301", "fr", split="train", streaming=True, trust_remote_code=True)
        if ds is None:
            return
        n = 0
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) > 200:
                yield text[:8000]
                n += 1
                if n >= 50_000:
                    break
    except Exception as e:
        logger.warning("OSCAR/web stream failed: %s", e)
