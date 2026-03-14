"""
agent/agent.py — L'agent autonome.

Boucle ReAct :
  1. THINK   → L'agent réfléchit à la question
  2. ACT     → Il utilise un outil (search, fetch, python...)
  3. OBSERVE → Il lit le résultat
  4. REPEAT  → Jusqu'à la réponse finale ou max_steps

Modes :
  - "llama"  → LLaMA 3 comme cerveau (nécessite GPU)
  - "local"  → Notre LLMMaison (après entraînement)
  - "tools"  → Outils seulement, pas de LLM
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import TOOLS, execute_tool, list_tools
from agent.memory import AgentMemory
from config import AGT_CFG


class Agent:

    SYSTEM_PROMPT = """Tu es un agent IA personnel autonome. Tu as accès à des outils.

OUTILS DISPONIBLES :
{tools}

FORMAT D'UTILISATION D'UN OUTIL :
<tool>NOM_OUTIL</tool>
<args>{{"arg1": "valeur1"}}</args>

FORMAT DE RÉPONSE FINALE :
<answer>Ta réponse ici</answer>

MÉMOIRE :
{memory}

Réfléchis étape par étape. Utilise les outils quand nécessaire.
Si tu connais la réponse, utilise directement <answer>."""

    def __init__(
        self,
        mode: str = "tools",
        model=None,
        tokenizer=None,
        llm_backend: str = "ollama",
    ):
        self.mode = mode
        self.model = model
        self.tokenizer = tokenizer
        self.memory = AgentMemory()
        self.llm = None
        self.llm_backend = (llm_backend or "ollama").lower()
        self._ollama_url = "http://localhost:11434"
        self._lmstudio_url = "http://localhost:1234"
        self._api_model = "llama3"

        if mode == "llama" and self.llm_backend == "transformers":
            self._load_llama()

    # ── Chargement LLaMA 3 ──────────────────────────────────────────

    def _load_llama(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import BitsAndBytesConfig
            from config import DST_CFG
            import torch

            print("[AGENT] Chargement LLaMA 3 (4-bit)...")
            self.llama_tok = AutoTokenizer.from_pretrained(DST_CFG.teacher_model)
            self.llm = AutoModelForCausalLM.from_pretrained(
                DST_CFG.teacher_model,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            )
            self.llm.eval()
            print("[AGENT] LLaMA 3 OK ✓")
        except Exception as e:
            print(f"[AGENT] Pas de LLaMA 3 : {e}")
            print("[AGENT] Fallback mode 'tools' (outils seuls)")
            self.mode = "tools"

    # ── Appel au LLM ────────────────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Route vers le bon backend LLM."""
        # Ollama — pas de chargement en RAM
        if self.mode == "llama" and self.llm_backend == "ollama":
            try:
                import requests
                r = requests.post(
                    f"{self._ollama_url.rstrip('/')}/api/generate",
                    json={
                        "model": self._api_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": AGT_CFG.temperature,
                            "top_p": AGT_CFG.top_p,
                            "repeat_penalty": AGT_CFG.rep_penalty,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=300,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                print(f"[AGENT] Ollama error: {e}")
                return ""

        # LM Studio — API OpenAI-compatible
        if self.mode == "llama" and self.llm_backend == "lmstudio":
            try:
                import requests
                r = requests.post(
                    f"{self._lmstudio_url.rstrip('/')}/v1/completions",
                    json={
                        "model": self._api_model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": AGT_CFG.temperature,
                        "top_p": AGT_CFG.top_p,
                    },
                    timeout=300,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0].get("text", "")
            except Exception as e:
                print(f"[AGENT] LM Studio error: {e}")
                return ""

        if self.mode == "llama" and self.llm_backend == "transformers" and self.llm is not None:
            import torch
            inputs = self.llama_tok(
                prompt, return_tensors="pt",
                max_length=2048, truncation=True
            )
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.llm.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=AGT_CFG.temperature,
                    top_p=AGT_CFG.top_p,
                    do_sample=True,
                    repetition_penalty=AGT_CFG.rep_penalty,
                )
            return self.llama_tok.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        elif self.mode == "local" and self.model is not None:
            from model.generate import generate_text, GenConfig
            return generate_text(
                self.model, self.tokenizer, prompt,
                GenConfig(
                    max_new_tokens=max_tokens,
                    temperature=AGT_CFG.temperature,
                    top_p=AGT_CFG.top_p,
                    top_k=AGT_CFG.top_k,
                    rep_penalty=AGT_CFG.rep_penalty,
                ),
            )
        else:
            return ""

    # ── Boucle ReAct ────────────────────────────────────────────────

    def process(self, user_message: str) -> str:
        """
        Traite un message avec la boucle think/act/observe.
        Retourne la réponse finale.
        """
        # Mode tools-only : pas de LLM, heuristique simple
        if self.mode == "tools":
            return self._tools_only(user_message)

        # Mode LLM (llama ou local)
        system = self.SYSTEM_PROMPT.format(
            tools=list_tools(),
            memory=self.memory.get_context(user_message),
        )
        conversation = f"{system}\n\nUtilisateur: {user_message}\n\nAssistant:"

        for step in range(AGT_CFG.max_steps):
            response = self._call_llm(conversation)

            # Réponse finale ?
            answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if answer:
                final = answer.group(1).strip()
                if AGT_CFG.auto_learn:
                    self._auto_learn(user_message, final)
                return final

            # Appel d'outil ?
            tool_match = re.search(r"<tool>(.*?)</tool>", response)
            args_match = re.search(r"<args>(.*?)</args>", response, re.DOTALL)

            if tool_match:
                tool_name = tool_match.group(1).strip()
                tool_args = {}
                if args_match:
                    try:
                        tool_args = json.loads(args_match.group(1).strip())
                    except json.JSONDecodeError:
                        tool_args = {}

                result = execute_tool(tool_name, **tool_args)
                conversation += (
                    f"\n{response}\n\n"
                    f"Résultat {tool_name}:\n{result[:1500]}\n\n"
                    f"Assistant:"
                )
            else:
                # Ni réponse ni outil → forcer la réponse
                conversation += (
                    f"\n{response}\n\n"
                    f"Donne ta réponse finale avec <answer>...</answer>\n\n"
                    f"Assistant:"
                )

        # Timeout
        return response.strip() if response else "Pas de réponse (max steps atteint)."

    def _tools_only(self, msg: str) -> str:
        """Mode sans LLM : détection heuristique de l'intent → outil."""
        ml = msg.lower()

        # Recherche web
        search_triggers = ["cherche", "search", "trouve", "c'est quoi",
                           "qu'est-ce", "who is", "what is", "actualité"]
        if any(t in ml for t in search_triggers):
            results = execute_tool("search_web", query=msg)
            return f"Résultats pour '{msg}':\n\n{results}"

        # Calcul
        if any(c in msg for c in "+-*/") and any(c.isdigit() for c in msg):
            return execute_tool("calculator", expression=msg)

        # Date
        if any(t in ml for t in ["date", "heure", "time", "jour"]):
            return execute_tool("get_datetime")

        return (
            "[Mode tools] Pas de LLM chargé pour répondre librement.\n"
            "Commandes dispo: /search <query>, /explore <topic>, /memory\n"
            "Ou lance avec: python -m agent.chat --mode llama --backend ollama"
        )

    # ── Auto-apprentissage ──────────────────────────────────────────

    def _auto_learn(self, question: str, answer: str):
        """Extrait des faits de la conversation pour enrichir la mémoire."""
        if len(answer) < 100:
            return
        prompt = (
            "Extrais les 3 faits les plus importants de cet échange. "
            "Un fait par ligne, juste le fait, pas de numéro.\n\n"
            f"Question: {question}\n"
            f"Réponse: {answer}\n\n"
            "Faits:"
        )
        facts_text = self._call_llm(prompt, max_tokens=200)
        for line in facts_text.strip().split("\n"):
            line = line.strip().lstrip("-•*0123456789.").strip()
            if 20 < len(line) < 300:
                self.memory.add_fact(line, source="auto_learn")

    # ── Exploration autonome ────────────────────────────────────────

    def auto_explore(self, topic: str):
        """L'agent explore un sujet de lui-même."""
        print(f"\n[AGENT] 🔍 Exploration autonome : '{topic}'")
        self.memory.add_topic(topic)

        results = execute_tool("search_web", query=topic, max_results=5)
        try:
            results_list = json.loads(results) if isinstance(results, str) else results
        except json.JSONDecodeError:
            results_list = []

        for r in results_list[:3]:
            url = r.get("url", "")
            if not url:
                continue

            print(f"[AGENT] 📖 {r.get('title', url)[:60]}...")
            text = execute_tool("fetch_url", url=url, max_chars=3000)

            if not text or len(text) < 200 or text.startswith("[ERREUR"):
                continue

            # Extraire des faits (avec LLM si backend disponible)
            has_llm = self.mode != "tools" and (
                self.llm is not None or self.llm_backend in ("ollama", "lmstudio")
            )
            if has_llm:
                prompt = (
                    "Extrais 5 faits importants de ce texte. "
                    "Un par ligne, sans numéro.\n\n"
                    f"{text[:2000]}\n\nFaits:"
                )
                facts = self._call_llm(prompt, max_tokens=300)
                for line in facts.strip().split("\n"):
                    line = line.strip().lstrip("-•*0123456789.").strip()
                    if 20 < len(line) < 300:
                        self.memory.add_fact(line, source=url)
                        print(f"  💡 {line[:80]}...")
            else:
                # Heuristique sans LLM : premières phrases
                import re as _re
                for s in _re.split(r'(?<=[.!?])\s+', text)[:5]:
                    s = s.strip()
                    if 30 < len(s) < 300:
                        self.memory.add_fact(s, source=url)

        self.memory.save()
        print(f"[AGENT] Exploration terminée. {self.memory.stats()}")
