# 🔧 Prompts Cursor — Morceaux à compléter

Tout le squelette fonctionne tel quel. Ces prompts sont pour les améliorations
que tu peux faire toi-même dans Cursor. Copie-colle le prompt, ouvre le fichier.

---

## 1. ⚡ KV Cache pour inférence rapide

**Fichier :** `model/transformer.py` + `model/generate.py`

```
Ajouter un KV cache pour accélérer la génération autoregressive.

Dans GQAttention:
1. Ajouter un paramètre past_kv=None au forward()
2. Si past_kv est fourni, ne calculer K/V que pour les nouveaux tokens
3. Concaténer avec le cache existant
4. Retourner le nouveau cache en plus de la sortie
5. Ajouter clear_cache() sur LLMMaison

Dans generate():
1. Au premier step, forward complet → récupérer le cache
2. Aux steps suivants, ne passer que le dernier token + cache
3. Devrait être ~10-20x plus rapide

Le KV cache par couche = (batch, n_kv_heads, seq_len, d_head).
12 couches × 1024 tokens × 4 heads × 64 dim × 2 bytes = ~6MB, négligeable.
Tester que les outputs sont identiques avec et sans cache.
```

---

## 2. 📚 Wikipedia Dump Parser (rapide)

**Fichier :** `crawler/wikipedia_crawler.py`

```
Ajouter une fonction download_and_parse_dump() qui :

1. Télécharge le dump frwiki-latest-pages-articles.xml.bz2
   depuis https://dumps.wikimedia.org/frwiki/latest/
   Utiliser requests avec stream=True, afficher la progression.

2. Parser en streaming avec xml.etree.ElementTree.iterparse()
   Namespace: {http://www.mediawiki.org/xml/export-0.10/}
   Pour chaque <page>, extraire <title> et <text>
   Ignorer les <redirect/>, les Discussion:, Utilisateur:, etc.

3. Nettoyer le markup MediaWiki :
   - {{ templates }} → supprimer
   - [[Lien|Texte]] → garder "Texte"
   - [[Lien]] → garder "Lien"
   - == Titres == → garder "Titres"
   - {| tableaux |} → supprimer
   - [[Catégorie:...]], [[Fichier:...]] → supprimer
   - Balises HTML → supprimer
   Utiliser mwparserfromhell si dispo, sinon regex.

4. Sauvegarder en .jsonl, même format que le crawler API.

Le dump frwiki fait ~2.5GB bz2. Utiliser bz2.open() en streaming.
C'est 100x plus rapide que l'API pour > 10k articles.
```

---

## 3. 🧪 Distillation Offline (sans gros GPU)

**Fichier :** `training/distill.py`

```
Ajouter le mode offline à distill.py. Deux phases :

PHASE 1 — generate_teacher_targets :
- Charger LLaMA 3 en 4-bit
- Pour chaque texte du corpus :
  → Tokenizer avec le tokenizer LLaMA
  → Forward pass → logits
  → Garder top-100 tokens par position (pas les 128K complets)
  → Sauvegarder : data_processed/teacher_logits/batch_XXXX.pt
    Format: {"input_ids": LongTensor(B,T), "top_ids": LongTensor(B,T,100),
             "top_probs": FloatTensor(B,T,100)}
- Batch size 4, max_length 512
- Afficher progression + ETA

PHASE 2 — train_from_targets :
- Charger les .pt files un par un
- Reconstruire distribution sparse :
    P(token) = top_probs[i] si token dans top_ids
    P(token) = (1 - sum(top_probs)) / (vocab - 100) sinon
- KL divergence student vs distribution reconstruite
- Loss = alpha * KD + (1-alpha) * CE
- Même logging/checkpointing que distill_online()

CLI: python -m training.distill --mode offline --phase generate
     python -m training.distill --mode offline --phase train
```

---

## 4. 🗺️ Mapping de vocabulaire pour la distillation

**Fichier :** nouveau `tokenizer/vocab_mapper.py`

```
Créer un mapper entre notre vocab BPE (32K) et celui de LLaMA 3 (128K).

Classe SharedVocabMapper:
  __init__(our_tokenizer, llama_tokenizer)

  Stratégie :
  1. Pour chaque token dans notre vocab, encoder le string avec LLaMA
  2. Pour chaque token LLaMA, encoder le string avec notre tokenizer
  3. Construire une matrice sparse de correspondance M
     M[our_id, llama_id] = score de correspondance

  project_teacher_logits(llama_logits) → our_logits:
     Projeter les logits 128K → 32K via la matrice M
     Normaliser pour que ça reste une distribution valide

  Alternative simple :
     Pour chaque texte, tokenizer avec les deux tokenizers.
     Aligner par position dans le texte source.
     Créer des paires (our_token_id, llama_token_id).

Ceci rend la distillation propre quand les vocabs diffèrent.
```

---

## 5. 📊 Évaluation automatique du modèle

**Fichier :** nouveau `monitoring/eval.py`

```
Créer un script d'évaluation automatique :

1. Perplexité :
   - Charger les 5% de données de validation (split dans train_tokenizer.py)
   - Calculer exp(avg_loss) sur tout le val set
   - Logger dans logs/eval_log.jsonl

2. Diversité de génération :
   - 20 prompts fixes (FR + EN, questions variées, à définir)
   - Générer 200 tokens pour chaque
   - unique_trigrams / total_trigrams → score de diversité
   - Détecter les boucles (trigram répété > 3x)

3. Comparaison entre checkpoints :
   - Charger 2 .pt files
   - Calculer perplexité et diversité pour les deux
   - Afficher un tableau comparatif

CLI: python -m monitoring.eval --checkpoint checkpoints/best.pt
```

---

## 6. 🔔 Notifications (optionnel)

**Fichier :** nouveau `agent/notifications.py`

```
Système de notifications pour le monitoring :

Backends :
- ConsoleNotifier : print() (défaut)
- FileNotifier : logs/notifications.jsonl
- TelegramNotifier : bot Telegram (optionnel, token dans config)
- DiscordNotifier : webhook Discord (optionnel)

Événements :
- Fait intéressant appris par l'auto-learner
- Milestone d'entraînement (loss < seuil)
- Erreur critique
- Fin de session

Config dans config.py :
@dataclass
class NotifConfig:
    backend: str = "console"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""

Intégrer dans auto_learner.py et pretrain.py.
```

---

## 7. 🌐 Interface Web (bonus)

**Fichier :** nouveau `agent/web_ui.py`

```
Créer une interface web simple avec Flask ou Gradio :

Option A (Gradio, plus simple) :
- pip install gradio
- Interface chat avec historique
- Onglet "Mémoire" pour voir les faits
- Onglet "Stats" pour le dashboard
- Bouton "Explorer un sujet"

Option B (Flask, plus contrôle) :
- Chat WebSocket en temps réel
- Dashboard avec graphiques (Chart.js)
- Visualisation de la mémoire
- Logs d'entraînement en live

Dans les deux cas :
- Réutiliser Agent, AgentMemory, les tools existants
- Sauvegarder les conversations comme agent/chat.py le fait
```
