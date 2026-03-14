"""
agent/memory.py — Mémoire persistante de l'agent.
Tout est dans agent_memory.json, tu peux l'ouvrir pour auditer.

Structure :
  facts[]           → Chaque fait appris avec source + timestamp
  preferences{}     → Tes préférences détectées
  topics_discussed  → Sujets explorés
  conversation_count → Compteur de sessions
"""
import json
import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGT_CFG


class AgentMemory:

    def __init__(self, path: str = None):
        self.path = Path(path or AGT_CFG.memory_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "facts": [],
            "preferences": {},
            "topics_discussed": [],
            "conversation_count": 0,
            "last_updated": None,
        }

    def save(self):
        self.data["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ── Faits ──

    def add_fact(self, fact: str, source: str = "conversation", category: str = None):
        """Ajoute un fait (pas de doublons). category optionnelle : science, histoire, tech, etc."""
        existing = {f["fact"] for f in self.data["facts"]}
        if fact not in existing:
            entry = {
                "fact": fact,
                "source": source,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if category:
                entry["category"] = category
            self.data["facts"].append(entry)
            self.save()

    def fact_exists_similar(self, text: str, threshold: float = 0.5) -> bool:
        """True si un fait déjà présent est assez proche (mots communs)."""
        t_words = set(text.lower().split())
        if len(t_words) < 3:
            return False
        for f in self.data["facts"]:
            f_words = set(f["fact"].lower().split())
            if not f_words:
                continue
            overlap = len(t_words & f_words) / max(len(t_words), 1)
            if overlap >= threshold:
                return True
        return False

    def search_facts(self, query: str, top_k: int = 5) -> list:
        """Recherche substring + mots communs."""
        q = query.lower()
        scored = []
        for f in self.data["facts"]:
            text = f["fact"].lower()
            if q in text:
                scored.append((f, 1.0))
            else:
                q_words = set(q.split())
                t_words = set(text.split())
                overlap = len(q_words & t_words)
                if overlap > 0:
                    scored.append((f, overlap / max(len(q_words), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:top_k]]

    # ── Préférences ──

    def add_preference(self, key: str, value):
        self.data["preferences"][key] = value
        self.save()

    # ── Topics ──

    def add_topic(self, topic: str):
        if topic not in self.data["topics_discussed"]:
            self.data["topics_discussed"].append(topic)
            self.save()

    # ── Contexte pour le prompt ──

    def get_context(self, query: str = "", max_facts: int = 10) -> str:
        """Retourne un résumé mémoire à injecter dans le prompt agent."""
        parts = []
        if self.data["preferences"]:
            parts.append("Préférences: " + json.dumps(
                self.data["preferences"], ensure_ascii=False))

        relevant = (self.search_facts(query, max_facts) if query
                    else self.data["facts"][-max_facts:])
        if relevant:
            parts.append("Faits connus:\n" + "\n".join(
                f"- {f['fact']}" for f in relevant))

        return "\n".join(parts) if parts else "(mémoire vide)"

    # ── Stats ──

    def increment_conversations(self):
        self.data["conversation_count"] += 1
        self.save()

    def stats(self) -> str:
        return (f"Faits: {len(self.data['facts'])}, "
                f"Préfs: {len(self.data['preferences'])}, "
                f"Conversations: {self.data['conversation_count']}, "
                f"Topics: {len(self.data['topics_discussed'])}")
