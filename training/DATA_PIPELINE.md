# Data pipeline — mix conversations + wiki + instructions

## Prérequis

```bash
pip install datasets pyyaml
```

(Ajoute `datasets` et `pyyaml` à ton `requirements.txt` si tu veux les figer.)

## Config

`training/mix_config.yaml` — ratios `mix.*`, `output_dir` (défaut `data_raw`), `output_prefix` (défaut `mixed`).

## Générer le JSONL

```bash
python -m training.data_pipeline --config training/mix_config.yaml --max-docs 50000
```

Produit `data_raw/mixed_00000.jsonl`, … avec `{"text": "...", "source": "conversations"}`.

`load_texts()` dans `distill.py` lit **tous** les `*.jsonl` de `data_raw` et ne garde que le champ `text` — les fichiers `mixed_*.jsonl` sont donc pris en même temps que ton wiki.

## Lister les sources sans télécharger

```bash
python -m training.data_pipeline --list-sources
```

## Special tokens `<|user|>` / `<|assistant|>`

Le pipeline écrit du texte avec ces marqueurs. Si ton tokenizer teacher ne les a pas, ils sont traités comme du texte brut (sous-mots BPE) — ça reste utilisable. Pour les ajouter au tokenizer maison plus tard : `add_special_tokens` + `resize_token_embeddings`.

## Sources HF

Les loaders utilisent `streaming=True` et `try/except` : si un dataset est indisponible (gated, rename), la source est simplement ignorée. Tu peux compléter `training/sources/*.py` avec d’autres `hf_id`.
