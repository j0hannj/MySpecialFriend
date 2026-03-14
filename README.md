# 🧠 LLM Maison — De A à Z

Un LLM construit from scratch : tokenizer BPE, transformer LLaMA-style (~150M params),
crawlers web, distillation depuis LLaMA 3, fine-tuning conversationnel, agent autonome.
Tourne sur 32GB RAM.

## Documentation

| Doc | Contenu |
|-----|--------|
| **[docs/MISE_EN_OEUVRE.md](docs/MISE_EN_OEUVRE.md)** | **Mise en œuvre** : prérequis, installation, pipeline pas à pas, commandes, dépannage |
| [docs/MODELE.md](docs/MODELE.md) | Fonctionnel : architecture, forward, KV cache, checkpoints |

## Architecture

```
llm-maison/
├── config.py                  # TOUS les paramètres (source unique de vérité)
├── requirements.txt           # pip install -r requirements.txt
├── run.py                     # Point d'entrée (menu interactif)
│
├── tokenizer/
│   ├── bpe.py                 # BPE from scratch (256 bytes → merges → vocab)
│   └── train_tokenizer.py     # Entraîne le tokenizer + tokenize le corpus
│
├── model/
│   ├── transformer.py         # Le modèle complet (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── generate.py            # Génération de texte (temp, top-k, top-p)
│   └── lora.py                # LoRA pour fine-tuning léger
│
├── crawler/
│   ├── wikipedia_crawler.py   # Wikipedia via API MediaWiki
│   ├── reddit_crawler.py      # Reddit via old.reddit.com JSON
│   ├── web_crawler.py         # Crawler web généraliste
│   └── cleaner.py             # Nettoyage + déduplication
│
├── training/
│   ├── mix_config.yaml        # Ratios du mix (conversations, wiki, instructions…)
│   ├── data_pipeline.py       # Mix multi-sources → data_raw/*.jsonl
│   ├── sources/               # Wikipedia, conversations HF, Alpaca, CodeAlpaca…
│   ├── tokenize_dataset.py    # Packing optionnel (ids par séquence)
│   ├── dataset.py             # Datasets (mmap pretrain + conversations finetune)
│   ├── pretrain.py            # Pré-entraînement (cosine LR, grad accum, AMP)
│   ├── distill.py             # Distillation LLaMA 3 → notre modèle
│   └── finetune.py            # Fine-tuning avec LoRA
│
├── agent/
│   ├── tools.py               # Outils : search, fetch, python, calcul, fichiers
│   ├── memory.py              # Mémoire persistante (JSON auditable)
│   ├── agent.py               # Agent autonome (boucle ReAct)
│   ├── chat.py                # Interface de chat (sauvegarde auto)
│   └── auto_learner.py        # Apprentissage autonome (crawl + learn en fond)
│
├── monitoring/
│   └── dashboard.py           # Stats entraînement, mémoire, données
│
├── data_raw/                  # Données crawlées brutes
├── data_processed/            # Données tokenizées (.bin)
├── checkpoints/               # Sauvegardes modèle (.pt)
├── tokenizer_model/           # Tokenizer entraîné (JSON)
├── conversations/             # Historique chat (pour finetune)
└── logs/                      # Logs d'entraînement (JSONL)
```

## Pipeline

```
1. CRAWL       python -m crawler.wikipedia_crawler
               python -m crawler.reddit_crawler
   MIX (opt.)  pip install datasets pyyaml
               python -m training.data_pipeline --max-docs 50000
               # Écrit data_raw/mixed_*.jsonl — load_texts() les lit avec le wiki
2. CLEAN       python -m crawler.cleaner
3. TOKENIZE    python -m tokenizer.train_tokenizer
4. PRETRAIN    python -m training.pretrain
5. DISTILL     python -m training.distill       (si LLaMA 3 dispo)
6. FINETUNE    python -m training.finetune      (après tes conversations)
7. CHAT        python -m agent.chat --mode local
8. AUTO-LEARN  python -m agent.auto_learner --hours 2
9. MONITOR     python -m monitoring.dashboard
```

## Quickstart

```bash
cd llm-maison
pip install -r requirements.txt

# Option A : Menu interactif
python run.py

# Option B : Pipeline manuelle
python -m crawler.wikipedia_crawler      # ~1h pour 1000 articles
python -m crawler.cleaner
python -m tokenizer.train_tokenizer
python -m model.transformer              # Test de santé
python -m training.pretrain              # GPU recommandé
python -m agent.chat                     # Chat (mode tools)
```

## Specs du modèle

| Paramètre | Valeur |
|-----------|--------|
| Architecture | Decoder-only (GPT/LLaMA style) |
| Params | ~150M |
| d_model | 768 |
| n_layers | 12 |
| n_heads | 12 (Q) / 4 (KV) → Grouped Query Attention |
| d_ff | 2048 (SwiGLU) |
| Context | 1024 tokens |
| Vocab | 32K (BPE from scratch) |
| Positional | RoPE (Rotary) |
| Norm | RMSNorm (pre-norm) |
| RAM estimée | ~0.6GB fp16, ~1.2GB fp32 |

## Audit

Tout est auditable :
- **tokenizer_model/tokenizer.json** → chaque merge BPE lisible
- **logs/*.jsonl** → chaque step d'entraînement loggé
- **agent_memory.json** → chaque fait appris avec source et timestamp
- **conversations/*.jsonl** → historique complet des échanges
- **data_raw/*.jsonl** → données brutes crawlées
