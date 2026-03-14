"""Code + instructions code (CodeAlpaca)."""
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


def stream_code() -> Iterator[str]:
    ds = _try_load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            prompt = (row.get("prompt") or row.get("instruction") or "").strip()
            completion = (row.get("completion") or row.get("output") or "").strip()
            if not prompt or not completion:
                continue
            text = f"<|user|>\n{prompt}\n<|assistant|>\n```\n{completion}\n```\n"
            yield text
    except Exception as e:
        logger.warning("CodeAlpaca stream failed: %s", e)
