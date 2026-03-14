# LLM Maison — Guide de mise en œuvre

Comment **installer**, **configurer** et **enchaîner** les étapes pour faire tourner le projet de bout en bout.

---

## 1. Prérequis

| Élément | Détail |
|--------|--------|
| **Python** | 3.10+ recommandé |
| **RAM** | ~32 Go confortable pour pretrain + crawls (moins possible en réduisant batch/seq) |
| **GPU** | Optionnel pour crawlers / tokenizer ; **recommandé** pour `training.pretrain` et `training.distill` (transformers) |
| **Disque** | Quelques Go pour corpus + `train.bin` + checkpoints (~600 Mo–2 Go par checkpoint selon précision) |

**Optionnel selon usage :**

- **LLaMA 3 HF** : compte Hugging Face + accès modèle, `transformers` + `bitsandbytes` (CUDA).
- **Ollama / LM Studio** : pas de GPU obligatoire côté Python si le teacher tourne déjà en local.
- **llama-cpp** : `pip install llama-cpp-python` + fichier `.gguf`.

---

## 2. Installation

### 2.1 Cloner / se placer dans le repo

```bash
cd chemin/vers/llm-maison
```

### 2.2 Environnement virtuel (recommandé)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2.3 Dépendances

```bash
pip install -U pip
pip install -r requirements.txt
```

**Optionnel :**

```bash
pip install gradio          # agent/web_ui.py
pip install mwparserfromhell # crawler Wikipedia dump (nettoyage wiki)
```

**PyTorch** : si besoin d’une build CUDA précise, installer `torch` depuis [pytorch.org](https://pytorch.org) **avant** ou **après** `requirements.txt` (éviter les conflits).

### 2.4 Vérification rapide

```bash
python -m model.transformer
```

Sans GPU / sans torch, cette commande échouera sur l’import `torch` — installer torch CPU au minimum.

---

## 3. Dossiers créés automatiquement

`config.py` crée à l’import :

| Dossier | Rôle |
|---------|------|
| `data_raw/` | JSONL crawlés (Wikipedia, Reddit, web…) |
| `data_processed/` | `train.bin`, `teacher_logits/`, etc. |
| `checkpoints/` | `.pt` du modèle |
| `tokenizer_model/` | `tokenizer.json` du BPE |
| `logs/` | `*_log.jsonl`, `eval_log.jsonl`, etc. |
| `conversations/` | Export chat pour finetune |

Rien à créer à la main pour démarrer.

---

## 4. Pipeline recommandé (ordre des étapes)

### Étape A — Données brutes

Au moins un fichier dans `data_raw/` (sinon le tokenizer ne pourra pas s’entraîner).

```bash
# Wikipedia (API, lent si beaucoup d’articles)
python -m crawler.wikipedia_crawler

# Ou dump (gros volume, une seule fois)
python -m crawler.wikipedia_crawler --mode dump --lang fr

# Reddit (self-posts, old.reddit JSON)
python -m crawler.reddit_crawler
```

**Astuce** : pour tester vite, un simple fichier texte suffit :

```bash
echo "Ton texte de test sur plusieurs lignes..." > data_raw/test.txt
```

### Étape B — Nettoyage

```bash
python -m crawler.cleaner
```

Produit un corpus nettoyé/dédoublonné utilisable en amont du tokenizer (selon ce que fait ton `cleaner` sur les fichiers présents).

### Étape C — Tokenizer + `train.bin`

```bash
python -m tokenizer.train_tokenizer
```

Lit `data_raw/*.jsonl` (et `.txt`), entraîne le BPE, écrit :

- `tokenizer_model/tokenizer.json`
- `data_processed/train.bin`
- `data_processed/stats.json`

**Sans cette étape**, `training.pretrain` lève `FileNotFoundError` sur `train.bin`.

### Étape D — Test modèle

```bash
python -m model.transformer
```

Forward + backward sur config par défaut.

### Étape E — Pré-entraînement

```bash
python -m training.pretrain
```

- Utilise `data_processed/train.bin`
- Checkpoints dans `checkpoints/step*.pt` et `pretrain_final.pt`
- Logs : `logs/pretrain_log.jsonl`

**CPU** : possible mais très lent ; réduire `TRN_CFG.max_steps` / `batch_size` dans `config.py` pour essais.

### Étape F — Distillation (optionnel)

**Teacher en local sans charger HF en RAM :**

```bash
# Ollama (localhost:11434)
python -m training.distill --mode online --backend ollama --api-url http://localhost:11434

# LM Studio (localhost:1234)
python -m training.distill --mode online --backend lmstudio --api-url http://localhost:1234
```

**Teacher Hugging Face (GPU + VRAM) :**

```bash
python -m training.distill --mode online --backend transformers
```

**Offline (logits pré-calculés, uniquement transformers pour la phase generate) :**

```bash
python -m training.distill --mode offline --phase generate
python -m training.distill --mode offline --phase train
```

### Étape G — Fine-tune conversationnel (optionnel)

Après avoir des conversations dans `conversations/` (ex. via `agent.chat`) :

```bash
python -m training.finetune
```

Charge `pretrain_final.pt` ou `distill_final.pt` selon présence.

### Étape H — Chat

```bash
# Sans modèle local (outils seulement)
python -m agent.chat --mode tools

# LLaMA via Ollama (défaut)
python -m agent.chat --mode llama --backend ollama

# LM Studio
python -m agent.chat --mode llama --backend lmstudio --lmstudio-url http://localhost:1234

# Notre modèle (checkpoint + tokenizer)
python -m agent.chat --mode local
```

### Étape I — Monitoring

```bash
python -m monitoring.dashboard
```

---

## 5. Menu tout-en-un

```bash
python run.py
```

Lance un menu numéroté qui appelle les mêmes modules (crawl, tokenizer, pretrain, distill, chat, etc.).

---

## 6. Profils de modèle (small / medium / large)

| Profil  | Ordre de grandeur | Usage |
|---------|-------------------|--------|
| `small` | ~150M | Pipeline rapide, tests |
| `medium`| ~1.5B | Pretrain sérieux sur 4080 16GB |
| `large` | ~3B   | LoRA finetune surtout (full pretrain très lourd) |

**Choisir le profil** (avant tout import qui lit `config`) :

```bash
# Windows PowerShell
$env:LLM_PROFILE="medium"; python -m training.pretrain

# Ou en premier argument (pretrain/finetune/distill lisent sys.argv)
python -m training.pretrain --profile medium
```

`TrainConfig` adapte batch/accum/lr pour `small` (plus rapide). `medium`/`large` utilisent gradient checkpointing si activé.

---

## 7. Configuration utile (`config.py`)

Tout est centralisé ; pour une **première mise en œuvre** tu peux toucher :

| Dataclass | Usage |
|-----------|--------|
| `CrawlerConfig` | `delay`, `wiki_min_chars`, subs Reddit |
| `TrainConfig` | `batch_size`, `max_steps`, `seq_len` |
| `DistillConfig` | `teacher_model`, `alpha_kd` / `alpha_ce`, `batch_size` |
| `AgentConfig` | température, `memory_path` |

Pas besoin de dupliquer les chemins : `DATA_DIR`, `CHECKPOINT_DIR`, etc. sont déjà définis.

---

## 7. Problèmes fréquents

| Symptôme | Piste |
|----------|--------|
| `No module named 'torch'` | `pip install torch` (ou suivi du site PyTorch) |
| `train.bin` introuvable | Lancer `python -m tokenizer.train_tokenizer` après avoir mis des données dans `data_raw/` |
| Reddit 429 | Augmenter `CRW_CFG.delay` ; attendre ; réduire le nombre de subs |
| Distill OOM GPU | Utiliser `--backend ollama` ou `--backend lmstudio` ; ou offline phase train seulement |
| Chat local sans réponse | Vérifier `checkpoints/pretrain_final.pt` ou `distill_final.pt` + `tokenizer_model/` |
| bitsandbytes sur Windows | Parfois capricieux ; privilégier backend Ollama/LM Studio pour la distillation |

---

## 8. Récap commandes par objectif

| Objectif | Commande |
|----------|----------|
| Avoir des données | `python -m crawler.wikipedia_crawler` ou `reddit_crawler` |
| Préparer le corpus | `python -m crawler.cleaner` |
| Tokenizer + binaire | `python -m tokenizer.train_tokenizer` |
| Entraîner from scratch | `python -m training.pretrain` |
| Distiller sans HF en RAM | `python -m training.distill --mode online --backend ollama` |
| Parler à l’agent (Ollama) | `python -m agent.chat --mode llama` |
| Voir l’état du projet | `python -m monitoring.dashboard` |

---

*Guide de mise en œuvre — à adapter selon ta machine et ton backend teacher. Pour le détail du modèle, voir [MODELE.md](MODELE.md).*
