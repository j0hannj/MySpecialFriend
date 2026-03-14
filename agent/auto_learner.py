"""
agent/auto_learner.py — Agent d'exploration autonome piloté par LLaMA via Ollama.

Boucle : LLaMA choisit quoi apprendre -> DuckDuckGo -> fetch pages ->
LLaMA extrait faits + sujets connexes -> mémoire + file FIFO -> priorisation LLaMA.

Usage:
  python -m agent.auto_learner --hours 4
  python -m agent.auto_learner --follow-interests
  python -m agent.auto_learner --topic "physique quantique" --deep
"""
import json
import re
import time
import random
import sys
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, LOG_DIR, CONV_DIR, AGT_CFG
from agent.tools import search_web, fetch_url
from agent.memory import AgentMemory
from agent.notifications import notify

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
MAX_QUEUE = 80
PAGES_PER_TOPIC = 5


def ollama_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Appel Ollama sans charger de modèle en RAM."""
    try:
        import requests
        r = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,
        )
        r.raise_for_status()
        return (r.json() or {}).get("response", "").strip()
    except Exception as e:
        print(f"[AUTO] Ollama error: {e}")
        return ""


def memory_summary(memory: AgentMemory, max_facts: int = 15) -> str:
    """Résumé compact pour le prompt LLaMA."""
    facts = memory.data.get("facts", [])[-max_facts:]
    topics = memory.data.get("topics_discussed", [])[-20:]
    lines = []
    if topics:
        lines.append("Sujets déjà explorés: " + "; ".join(topics[-15:]))
    if facts:
        lines.append("Extraits de faits connus:")
        for f in facts:
            cat = f.get("category", "")
            prefix = f"[{cat}] " if cat else ""
            lines.append(f"- {prefix}{f['fact'][:120]}")
    return "\n".join(lines) if lines else "(Mémoire presque vide — propose un sujet généraliste intéressant.)"


def parse_llama_facts_topics(raw: str):
    """
    Parse une réponse LLaMA structurée attendue :
    FAITS:
    - fait1 [CAT:science]
    SUJETS:
    - sujet1
    """
    facts = []
    topics = []
    current = None
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("FAITS") or up.startswith("FAIT"):
            current = "facts"
            continue
        if up.startswith("SUJETS") or up.startswith("SUJET") or up.startswith("CONNEXES"):
            current = "topics"
            continue
        if line.startswith("-") or line.startswith("*") or re.match(r"^\d+[\).]", line):
            content = re.sub(r"^[-*\d\).)\s]+", "", line).strip()
            if len(content) < 15:
                continue
            if current == "facts":
                cat = None
                m = re.search(r"\[CAT:\s*([^\]]+)\]", content, re.I)
                if m:
                    cat = m.group(1).strip()
                    content = re.sub(r"\[CAT:[^\]]*\]", "", content).strip()
                if content:
                    facts.append((content, cat))
            elif current == "topics":
                topics.append(content)
    # Fallback : lignes longues sans section = faits
    if not facts and not topics and len(raw) > 50:
        for line in raw.split("\n"):
            line = line.strip().lstrip("-*").strip()
            if 40 < len(line) < 400:
                facts.append((line, None))
    return facts[:12], topics[:8]


def read_recent_conversation_topics(max_files: int = 5) -> list:
    """Lit CONV_DIR et extrait des sujets potentiels depuis les messages user."""
    topics = []
    files = sorted(CONV_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    for fp in files:
        try:
            for line in open(fp, "r", encoding="utf-8", errors="replace"):
                try:
                    conv = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(conv, list):
                    for m in conv:
                        if m.get("role") == "user":
                            c = (m.get("content") or "").strip()
                            if 20 < len(c) < 200:
                                topics.append(c[:150])
        except Exception:
            pass
    return list(dict.fromkeys(topics))[:30]


class AutoLearner:
    def __init__(self, ollama_url: str = None, ollama_model: str = None):
        global OLLAMA_URL, OLLAMA_MODEL
        if ollama_url:
            OLLAMA_URL = ollama_url.rstrip("/")
        if ollama_model:
            OLLAMA_MODEL = ollama_model
        self.memory = AgentMemory()
        self.output_dir = DATA_DIR / "auto_learned"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_file = self.output_dir / "auto_learned.jsonl"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.sessions_log = LOG_DIR / "learning_sessions.jsonl"
        self.queue = deque()
        self.seen_topics = set(t.lower() for t in self.memory.data.get("topics_discussed", []))
        self.stats = {"pages": 0, "facts": 0, "topics": 0, "errors": 0, "skipped_dup": 0}
        self.session_learned = []  # faits de la session pour résumé final

    def _enqueue(self, topic: str):
        t = topic.strip()
        if not t or len(t) < 5:
            return
        key = t.lower()[:80]
        if key in self.seen_topics:
            return
        if len(self.queue) >= MAX_QUEUE:
            return
        self.queue.append(t)
        self.seen_topics.add(key)

    def ask_next_topic(self) -> str:
        prompt = f"""Tu es un curateur d'apprentissage. Réponds en UNE ligne seulement : un sujet de recherche web précis (en français ou anglais).

Question: Qu'est-ce que je devrais apprendre maintenant ?

Ce que je sais déjà (extrait):
{memory_summary(self.memory)}

Règles:
- Propose UN sujet que je ne connais pas encore (pas dans la liste explorée).
- Sujet concret, searchable (ex: "effet Tambora 1816 conséquences", "LoRA fine-tuning expliqué").
- Une seule ligne, pas de guillemets, pas d'explication.

Sujet:"""
        out = ollama_generate(prompt, max_tokens=80, temperature=0.9)
        line = out.split("\n")[0].strip().strip('"').strip("'")
        if line.lower().startswith("sujet"):
            line = line.split(":", 1)[-1].strip()
        return line[:200] if line else ""

    def prioritize_queue(self):
        """LLaMA choisit le prochain sujet dans la file."""
        if len(self.queue) <= 1:
            return
        batch = list(self.queue)[:15]
        prompt = f"""Parmi ces sujets à explorer, lequel est le plus utile à apprendre en premier ? Réponds par UNE ligne = le sujet exact repris de la liste.

Liste:
{chr(10).join(f"- {t}" for t in batch)}

Réponse (une ligne, copie exacte d'un sujet de la liste):"""
        out = ollama_generate(prompt, max_tokens=100, temperature=0.3)
        choice = out.split("\n")[0].strip().strip("-").strip()
        for i, t in enumerate(batch):
            if t.lower() in choice.lower() or choice.lower() in t.lower():
                while self.queue and self.queue[0] != t:
                    self.queue.rotate(-1)
                return

    def extract_from_page(self, text: str, url: str) -> tuple:
        prompt = f"""Tu es un extracteur de connaissances. Texte source (extrait):

{text[:3500]}

Tâche:
1) Liste 3 à 7 faits importants ou concepts clés (une ligne par fait, commencer par - ).
2) Liste 3 sujets connexes à explorer ensuite (section SUJETS:, une ligne par - ).

Format obligatoire:
FAITS:
- fait1 [CAT:science]   (CAT optionnel: science, histoire, tech, culture, autre)
SUJETS:
- sujet connexe 1
- sujet connexe 2
"""
        raw = ollama_generate(prompt, max_tokens=600, temperature=0.4)
        facts, topics = parse_llama_facts_topics(raw)
        return facts, topics

    def explore(self, topic: str, deep: bool = False):
        print(f"\n[AUTO] Topic: {topic[:70]}...")
        if self.memory.fact_exists_similar(topic, threshold=0.6):
            print("[AUTO] Skip (trop proche de faits existants)")
            self.stats["skipped_dup"] += 1
            return

        self.stats["topics"] += 1
        self.memory.add_topic(topic)

        results = search_web(topic, max_results=8)
        if isinstance(results, list) and results and "error" in results[0]:
            print(f"[AUTO] Search error: {results[0].get('error')}")
            self.stats["errors"] += 1
            return

        urls_done = set()
        for r in results[:PAGES_PER_TOPIC]:
            url = r.get("url", "")
            title = r.get("title", "")
            if not url or url in urls_done:
                continue
            urls_done.add(url)

            print(f"[AUTO] Page: {title[:55]}...")
            page_text = fetch_url(url, max_chars=6000)
            if not page_text or len(page_text) < 200 or page_text.startswith("[ERREUR"):
                continue

            self.stats["pages"] += 1
            doc = {
                "text": page_text,
                "source": "auto_learn",
                "url": url,
                "title": title,
                "topic": topic,
                "ts": datetime.now().isoformat(),
            }
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            facts, related = self.extract_from_page(page_text, url)
            for fact, cat in facts:
                if self.memory.fact_exists_similar(fact, threshold=0.55):
                    continue
                # Contradiction simple : même sujet, forme négative (heuristique)
                if self._might_contradict(fact):
                    fact = "[À vérifier] " + fact
                self.memory.add_fact(fact, source=url, category=cat)
                self.stats["facts"] += 1
                self.session_learned.append({"fact": fact, "category": cat})
                print(f"  + {fact[:75]}...")

            for t in related:
                self._enqueue(t)

            if deep:
                # Liens : recherche "site:..." ou re-search avec titre
                body = r.get("body") or ""
                if body and len(body) > 30:
                    self._enqueue(f"{topic} {body[:60]}")

            time.sleep(2)

    def _might_contradict(self, fact: str) -> bool:
        """Heuristique légère ; LLaMA pourrait faire mieux en batch."""
        neg = (" pas ", " non ", " jamais ", " aucun ", " false ", " not ")
        if not any(n in fact.lower() for n in neg):
            return False
        for f in self.memory.data.get("facts", [])[-50:]:
            if f["fact"][:40] in fact or fact[:40] in f["fact"]:
                return True
        return False

    def session_summary(self):
        if not self.session_learned:
            return
        facts_lines = "\n".join(f"- {x['fact']}" for x in self.session_learned[-40:])
        prompt = f"""Résume en 5-10 lignes ce que l'agent a appris pendant cette session (faits + thèmes). Français.

Faits enregistrés:
{facts_lines}

Résumé session:"""
        summary = ollama_generate(prompt, max_tokens=400, temperature=0.5)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "n_facts": self.stats["facts"],
            "n_pages": self.stats["pages"],
            "n_topics": self.stats["topics"],
        }
        with open(self.sessions_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n[AUTO] Résumé session -> {self.sessions_log}")

    def run(
        self,
        hours: float = 1.0,
        seed_topics: list = None,
        follow_interests: bool = False,
        deep: bool = False,
    ):
        end_time = time.time() + hours * 3600

        if follow_interests:
            for t in read_recent_conversation_topics():
                self._enqueue(t)
            print(f"[AUTO] --follow-interests: {len(self.queue)} sujets depuis conversations/")

        if seed_topics:
            for t in seed_topics:
                self._enqueue(t)

        print("╔══════════════════════════════════════════════════╗")
        print("║  AUTO-LEARNER — Exploration pilotée (Ollama)    ║")
        print(f"║  Durée: {hours}h  |  Ollama: {OLLAMA_URL}        ║")
        print("╚══════════════════════════════════════════════════╝")

        cycle = 0
        while time.time() < end_time:
            cycle += 1
            if self.queue:
                self.prioritize_queue()
                topic = self.queue.popleft()
            else:
                topic = self.ask_next_topic()
                if not topic:
                    topic = random.choice(
                        self.memory.data.get("topics_discussed", []) or ["actualités sciences 2024"]
                    )

            try:
                self.explore(topic, deep=deep)
            except KeyboardInterrupt:
                print("\n[Interrompu]")
                break
            except Exception as e:
                print(f"[AUTO] Erreur: {e}")
                self.stats["errors"] += 1

            wait = random.uniform(8, 20)
            remaining = (end_time - time.time()) / 60
            if remaining > 0:
                print(f"[AUTO] Pause {wait:.0f}s — ~{remaining:.0f} min restantes")
            time.sleep(wait)

        self.session_summary()

        print(f"\n{'='*50}")
        print(f"[AUTO] Session terminée — {cycle} cycles")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        print(f"  Mémoire: {self.memory.stats()}")
        if self.out_file.exists():
            print(f"  Données brutes: {self.out_file} ({self.out_file.stat().st_size/1e6:.1f} MB)")
        print(f"{'='*50}")

        notify(
            "Auto-learner session complete",
            event="session_end",
            extra=self.stats,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Auto-learner piloté par Ollama")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--topics", type=str, default="", help="Sujets initiaux, virgules")
    p.add_argument("--topic", type=str, default="", help="Un sujet + exploration")
    p.add_argument("--deep", action="store_true", help="Exploration plus profonde")
    p.add_argument("--follow-interests", action="store_true", help="Déduit sujets depuis conversations/")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="llama3")
    args = p.parse_args()

    seed = [t.strip() for t in args.topics.split(",") if t.strip()]
    if args.topic:
        seed.insert(0, args.topic)

    AutoLearner(ollama_url=args.ollama_url, ollama_model=args.model).run(
        hours=args.hours,
        seed_topics=seed or None,
        follow_interests=args.follow_interests,
        deep=args.deep,
    )
