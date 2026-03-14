# Tokenizer teacher (local)

Une fois les fichiers en place, la distillation **ne refait plus aucun appel** au hub (chargement `local_files_only=True`).

## Remplir le dossier automatiquement (une fois)

Avec un repo HF qui contient le tokenizer Llama 3 (souvent ~quelques Mo seulement) :

```powershell
# Sans argument = repo public Xenova/llama3-tokenizer (pas de compte HF requis)
python -m training.distill --download-teacher-tokenizer

# Ou explicitement
python -m training.distill --download-teacher-tokenizer Xenova/llama3-tokenizer
```

**Évite `meta-llama/*`** : tout est gated (401 sans login + licence). Les repos ci-dessus sont publics et tokenizer compatible Ollama `llama3`.

---

# Tokenizer teacher (copie manuelle)

La distillation avec `--backend ollama` a besoin du **tokenizer Llama 3** pour aligner les `input_ids` du student sur le vocab du teacher. Tout est chargé **en local** : aucun appel réseau.

## Fichiers à mettre dans ce dossier

Copie ici les fichiers d’un modèle Llama 3 (Instruct de préférence), par ex. depuis :

- un clone HF déjà téléchargé :  
  `.../hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/<hash>/`
- ou un dossier où tu as extrait le modèle.

Fichiers **minimum** pour `AutoTokenizer.from_pretrained(..., local_files_only=True)` :

| Fichier | Rôle |
|--------|------|
| `tokenizer.json` | BPE / SentencePiece sérialisé (souvent le plus important) |
| `tokenizer_config.json` | Config tokenizer |
| `special_tokens_map.json` | Si présent dans le repo source, copie-le aussi |

Optionnels selon le repo : `added_tokens.json`, `chat_template.jinja`.

## Vérification

```powershell
cd llm-maison
dir teacher_tokenizer
# tu dois voir au moins tokenizer.json et tokenizer_config.json
```

Puis :

```powershell
python -m training.distill --backend ollama
```

Si un fichier manque, le message d’erreur transformers indiquera lequel.

## Alternative : chemin absolu

Tu peux aussi mettre les fichiers ailleurs et dans `config.py` :

```python
teacher_tokenizer: str = r"D:\modeles\llama3-tokenizer"
```
