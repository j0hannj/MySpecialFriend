# LLM Maison — Documentation fonctionnelle (modèle)

Ce document décrit **comment le modèle fonctionne** dans le projet : architecture, données, entraînement, inférence. Pour la liste des commandes, voir `run.py` ou les docstrings des modules.

---

## 1. Vue d’ensemble

**LLM Maison** est un **modèle causal (decoder-only)** de type LLaMA :

- **~150M paramètres** (config par défaut dans `config.py`).
- **Vocabulaire** : BPE maison 32k (`tokenizer/bpe.py`) ou vocab aligné teacher pendant la distillation.
- **Rôle** : prédire le token suivant à partir du contexte (langage modélisé).

Le flux typique :

1. **Données brutes** → crawlers → `data_raw/*.jsonl`
2. **Nettoyage** → `crawler/cleaner.py` → corpus prêt
3. **Tokenizer** → `tokenizer/train_tokenizer.py` → `tokenizer_model/` + `data_processed/train.bin`
4. **Pré-entraînement** → `training/pretrain.py` → `checkpoints/pretrain_final.pt`
5. **(Optionnel)** Distillation → `training/distill.py` → `distill_final.pt`
6. **(Optionnel)** Fine-tune → `training/finetune.py` → LoRA ou full
7. **Inférence** → `model/generate.py` + tokenizer, ou agent `agent/chat.py --mode local`

---

## 2. Architecture du modèle (`model/transformer.py`)

### 2.1 Blocs principaux

| Composant | Rôle |
|-----------|------|
| **Embedding** | `ids` → vecteurs `d_model` (768 par défaut) |
| **N couches** | 12 blocs identiques (Transformer) |
| **RMSNorm** | Normalisation pré-attention / pré-FFN (style LLaMA) |
| **RoPE** | Encodage positionnel dans Q et K (pas d’embedding position absolu) |
| **GQA (Grouped Query Attention)** | 12 têtes Q, **4 têtes K/V** ; K/V répétées pour matcher Q |
| **SwiGLU FFN** | `gate` + `up` → SiLU × up → `down` (FFN 2048 par défaut) |
| **Tête LM** | Si `tie_embeddings=True` : logits = produit scalaire avec la matrice d’embedding (pas de `lm_head` séparé) |

### 2.2 Config par défaut (`ModelConfig` / `MDL_CFG`)

| Paramètre | Valeur | Effet |
|-----------|--------|--------|
| `vocab_size` | 32_000 | Taille du softmax / embedding |
| `d_model` | 768 | Dimension cachée |
| `n_heads` | 12 | Têtes d’attention (Q) |
| `n_kv_heads` | 4 | Têtes K/V (GQA) |
| `n_layers` | 12 | Profondeur |
| `d_ff` | 2048 | Taille intermédiaire FFN |
| `max_seq_len` | 1024 | Contexte max (RoPE précalculé jusqu’à cette longueur) |
| `tie_embeddings` | True | Poids partagés embedding / projection vocab |

### 2.3 Forward — entrées / sorties

```text
forward(ids, targets=None, start_pos=0, past_kv=None)
```

- **Entrée** : `ids` shape `(batch, seq_len)` — indices de tokens.
- **Sortie** (dict) :
  - `logits` : `(batch, seq_len, vocab_size)` — non normalisés ; appliquer softmax pour probas.
  - `loss` : si `targets` fourni, cross-entropy sur le **décalage** classique (prédire `targets[t]` depuis `logits[t-1]`).
  - `kv_cache` : liste par couche `(k, v)` pour **génération incrémentale** (KV cache).

### 2.4 KV cache (génération rapide)

En génération autoregressive, au lieu de recalculer K/V sur toute la séquence à chaque nouveau token :

1. **Premier passage** : forward sur tout le prompt → récupérer `kv_cache`.
2. **Pas suivants** : forward sur **le dernier token seulement** + `past_kv` + `start_pos` mis à jour.

C’est ce que fait `model/generate.py` quand `GenConfig.use_cache=True`.

---

## 3. Tokenizer (`tokenizer/bpe.py`)

- **BPE from scratch** : 256 bytes + tokens spéciaux + merges jusqu’à `vocab_size`.
- **Fichiers** : `tokenizer_model/tokenizer.json` (merges + vocab lisible).
- **Spéciaux** : `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<SEP>`, `<USER>`, `<ASST>`, `<SYS>` — utilisés pour le chat / finetune.
- **Prétrain** : `train_tokenizer.py` lit `data_raw/*.jsonl`, entraîne le BPE, écrit `data_processed/train.bin` (uint16, suite de tokens pour LM).

---

## 4. Données d’entraînement

### 4.1 Prétrain (`PretrainDataset`)

- **Source** : `data_processed/train.bin` (mmap).
- **Découpage** : fenêtres de longueur `seq_len + 1` ; une ligne = `(input[:-1], target[1:])` (next-token prediction).
- **Pas de padding** dans le pretrain par défaut : batch de blocs fixes.

### 4.2 Fine-tune (`ConvDataset`)

- **Source** : conversations JSON dans `conversations/` (listes de `{role, content}`).
- **Format interne** : BOS + `<USER>` / `<ASST>` + tokens du texte + EOS ; padding avec `collate_pad` (pad sur input, `-1` sur targets pour ignorer au loss).

---

## 5. Entraînement

### 5.1 Prétrain (`training/pretrain.py`)

- **Loss** : cross-entropy sur tout le vocab (sauf positions ignorées).
- **Optim** : AdamW, découpage weight decay (2D vs bias), **gradient accumulation**, **mixed precision** si CUDA.
- **LR** : warmup puis cosine jusqu’à `min_lr`.
- **Checkpoints** : tous les `save_interval` steps + `pretrain_final.pt`.

### 5.2 Distillation (`training/distill.py`)

- **Idée** : un **teacher** (LLaMA 3 ou API locale) fournit des cibles plus riches qu’un simple next-token sur corpus seul.
- **Loss** (quand logits alignés) :  
  `alpha_kd * KL(student || teacher) + alpha_ce * CE(labels)`  
  avec température sur les softmax.
- **Backends** : `transformers` (HF 4-bit), `ollama`, `lmstudio`, `llamacpp` — si pas de logits (Ollama/LM Studio), **CE séquence** sur texte généré par le teacher (moins fin que KL).
- **Vocab** : en distillation avec tokenizer teacher, le student peut être créé avec `vocab_size = len(teacher_tokenizer)` ; après coup, repasser par le BPE pour finetune si besoin.

### 5.3 Fine-tune (`training/finetune.py`)

- Charge `pretrain_final.pt` ou `distill_final.pt`.
- **LoRA** par défaut sur les projets attention + FFN.
- Loss = CE sur les conversations formatées.

---

## 6. Inférence / génération (`model/generate.py`)

1. **Encoder** le prompt avec le tokenizer → `input_ids`.
2. **Boucle** : à chaque pas, prendre les logits du **dernier** token → sampling :
   - `temperature`, `top_k`, `top_p` (nucleus)
   - `rep_penalty` sur les tokens déjà présents dans la séquence
3. **Arrêt** : `max_new_tokens` ou token dans `stop_tokens` (ex. EOS).
4. **Décoder** les nouveaux ids en texte.

Fonction utilitaire : `generate_text(model, tokenizer, prompt, cfg, device)`.

---

## 7. Checkpoints

- **Fichier** : `.pt` avec `model_state`, `config` (dict pour reconstruire `ModelConfig`), `step`, optionnel `optimizer_state`.
- **Chargement** : `LLMMaison.load_checkpoint(path, device)` — recrée le modèle avec la config sauvegardée (important si `vocab_size` a changé après distillation).

---

## 8. Où modifier quoi (résumé)

| Objectif | Fichier / entrée |
|----------|-------------------|
| Taille du modèle | `config.py` → `ModelConfig` |
| Longueur de contexte | `max_seq_len` (+ regénérer données si besoin) |
| Corpus pretrain | `data_raw/` puis `train_tokenizer` + `pretrain` |
| Hyperparams train | `TrainConfig` dans `config.py` |
| Génération (temp, top-p…) | `GenConfig` dans `model/generate.py` ou appelant |
| Agent / chat local | `agent/chat.py --mode local` + checkpoint + `TOKENIZER_DIR` |

---

## 9. Commandes utiles

```bash
# Santé du modèle (forward + loss + backward)
python -m model.transformer

# Menu tout-en-un
python run.py

# Dashboard (logs, checkpoints, mémoire)
python -m monitoring.dashboard
```

---

*Document fonctionnel — à faire évoluer avec le code. Dernière mise à jour alignée sur `model/transformer.py`, `config.py`, `training/dataset.py`, `model/generate.py`.*
