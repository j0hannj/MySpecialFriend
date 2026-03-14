"""Wikipedia FR — reduit dans le mix via ratio config."""
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


def stream_wikipedia() -> Iterator[str]:
    # 20231101.fr ou autre subset selon dispo
    for subset in ("20231101.fr", "20220301.fr"):
        ds = _try_load_dataset("wikimedia/wikipedia", subset, split="train", streaming=True, trust_remote_code=True)
        if ds is not None:
            break
    else:
        return
    try:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) > 500:
                yield text
    except Exception as e:
        logger.warning("Wikipedia stream failed: %s", e)
