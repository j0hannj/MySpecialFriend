"""Livres / articles longs — chunks ~4000 chars."""
import logging
from typing import Iterator

logger = logging.getLogger(__name__)
CHUNK = 4000


def _try_load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
        return load_dataset(*args, **kwargs)
    except Exception as e:
        logger.warning("datasets.load_dataset unavailable: %s", e)
        return None


def _chunk(text: str) -> Iterator[str]:
    text = text.strip()
    if len(text) < 300:
        return
    for i in range(0, len(text), CHUNK):
        chunk = text[i : i + CHUNK]
        if len(chunk) > 200:
            yield chunk


def stream_books() -> Iterator[str]:
    # PleIAs/French-PD-Books peut etre lourd — streaming
    ds = _try_load_dataset("PleIAs/French-PD-Books", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            if isinstance(text, str):
                yield from _chunk(text)
    except Exception as e:
        logger.warning("French-PD-Books stream failed: %s", e)
