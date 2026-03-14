"""Conversations : OASST2, UltraChat, ShareGPT — format <|user|> / <|assistant|>."""
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

CHAT_USER = "<|user|>\n"
CHAT_ASST = "<|assistant|>\n"


def _try_load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
        return load_dataset(*args, **kwargs)
    except Exception as e:
        logger.warning("datasets.load_dataset unavailable: %s", e)
        return None


def _yield_oasst2() -> Iterator[str]:
    ds = _try_load_dataset("OpenAssistant/oasst2", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            lang = (row.get("lang") or "").lower()
            if lang and lang not in ("fr", "en"):
                continue
            text = (row.get("text") or "").strip()
            if len(text) > 30:
                yield text
    except Exception as e:
        logger.warning("OASST2 stream failed: %s", e)


def _yield_ultrachat() -> Iterator[str]:
    # split peut varier selon version
    for split in ("train_sft", "train"):
        ds = _try_load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True, trust_remote_code=True)
        if ds is not None:
            break
    else:
        return
    try:
        for row in ds:
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
                    parts.append(f"{CHAT_USER}{content}\n")
                elif role == "assistant":
                    parts.append(f"{CHAT_ASST}{content}\n")
            if parts:
                yield "".join(parts)
    except Exception as e:
        logger.warning("UltraChat stream failed: %s", e)


def _yield_sharegpt() -> Iterator[str]:
    ds = _try_load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if ds is None:
        return
    try:
        for row in ds:
            convos = row.get("conversations") or []
            parts = []
            for turn in convos:
                from_h = turn.get("from")
                role = "user" if from_h == "human" else "assistant"
                content = (turn.get("value") or "").strip()
                if not content:
                    continue
                parts.append(f"<|{role}|>\n{content}\n")
            text = "".join(parts)
            if len(text) > 100:
                yield text
    except Exception as e:
        logger.warning("ShareGPT stream failed: %s", e)


def stream_conversations() -> Iterator[str]:
    """Mix interne des sources conversationnelles."""
    yield from _yield_oasst2()
    yield from _yield_ultrachat()
    yield from _yield_sharegpt()
