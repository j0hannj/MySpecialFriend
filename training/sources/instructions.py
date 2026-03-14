"""Instructions / Q&A : Alpaca, Dolly, FQuAD — format chat."""
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


def _fmt(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"


def _yield_alpaca() -> Iterator[str]:
    ds = _try_load_dataset("tatsu-lab/alpaca", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()
            if not instruction or not output:
                continue
            prompt = f"{instruction}\n{inp}".strip() if inp else instruction
            yield _fmt(prompt, output)
    except Exception as e:
        logger.warning("Alpaca stream failed: %s", e)


def _yield_dolly() -> Iterator[str]:
    ds = _try_load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            context = (row.get("context") or "").strip()
            response = (row.get("response") or "").strip()
            if not instruction or not response:
                continue
            prompt = f"{instruction}\n{context}".strip() if context else instruction
            yield _fmt(prompt, response)
    except Exception as e:
        logger.warning("Dolly stream failed: %s", e)


def _yield_fquad() -> Iterator[str]:
    ds = _try_load_dataset("manu/fquad2", split="train", streaming=True, trust_remote_code=True)
    if ds is None:
        return
    try:
        for row in ds:
            question = (row.get("question") or "").strip()
            answers = row.get("answers") or {}
            texts = answers.get("text") if isinstance(answers, dict) else None
            if not question or not texts:
                continue
            answer = texts[0] if isinstance(texts, list) else str(texts)
            if answer:
                yield _fmt(question, answer)
    except Exception as e:
        logger.warning("FQuAD stream failed: %s", e)


def stream_instructions() -> Iterator[str]:
    yield from _yield_alpaca()
    yield from _yield_dolly()
    yield from _yield_fquad()
