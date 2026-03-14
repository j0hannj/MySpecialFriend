"""
agent/tools.py — Outils que l'agent peut invoquer.
Chaque tool = une fonction pure avec docstring explicite.
Le registre TOOLS en bas permet l'introspection automatique.
"""
import json
import subprocess
import datetime
from pathlib import Path


def search_web(query: str, max_results: int = 5) -> list:
    """Recherche DuckDuckGo (zéro API key)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            results = list(d.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
                for r in results
            ]
    except ImportError:
        return [{"error": "pip install duckduckgo-search"}]
    except Exception as e:
        return [{"error": str(e)}]


def fetch_url(url: str, max_chars: int = 5000) -> str:
    """Télécharge une page et extrait le texte propre."""
    try:
        import requests
        r = requests.get(url, timeout=15, headers={"User-Agent": "LLM-Maison/1.0"})
        r.raise_for_status()
        try:
            import trafilatura
            text = trafilatura.extract(r.text) or ""
        except ImportError:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"[ERREUR] {e}"


def get_datetime() -> str:
    """Date et heure actuelles."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_python(code: str) -> str:
    """Exécute du Python dans un subprocess (timeout 30s)."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30
        )
        out = result.stdout
        if result.stderr:
            out += f"\n[STDERR] {result.stderr}"
        return out[:3000]
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] > 30 secondes"
    except Exception as e:
        return f"[ERREUR] {e}"


def calculator(expression: str) -> str:
    """Évalue une expression mathématique simple."""
    try:
        # Sécurité : que des chars mathématiques
        allowed = set("0123456789+-*/.()% ")
        if not all(c in allowed for c in expression):
            return "[ERREUR] Caractères non autorisés"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"[ERREUR] {e}"


def read_file(path: str) -> str:
    """Lit un fichier local (max 5000 chars)."""
    try:
        return Path(path).read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"[ERREUR] {e}"


def write_file(path: str, content: str) -> str:
    """Écrit dans un fichier local."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return f"OK — {len(content)} chars → {path}"
    except Exception as e:
        return f"[ERREUR] {e}"


# ═══════════════════════════════════════════════════════════════════
# REGISTRE — L'agent introspect ce dict pour savoir quoi utiliser
# ═══════════════════════════════════════════════════════════════════

TOOLS = {
    "search_web":  {"fn": search_web,  "desc": "Rechercher sur internet. Args: query (str), max_results (int, opt)"},
    "fetch_url":   {"fn": fetch_url,   "desc": "Lire le contenu d'une URL. Args: url (str)"},
    "get_datetime": {"fn": get_datetime, "desc": "Date/heure actuelle. Pas d'args."},
    "calculator":  {"fn": calculator,  "desc": "Calculer une expression math. Args: expression (str)"},
    "run_python":  {"fn": run_python,  "desc": "Exécuter du code Python. Args: code (str)"},
    "read_file":   {"fn": read_file,   "desc": "Lire un fichier local. Args: path (str)"},
    "write_file":  {"fn": write_file,  "desc": "Écrire un fichier. Args: path (str), content (str)"},
}


def list_tools() -> str:
    """Description lisible de tous les outils."""
    return "\n".join(f"- {name}: {info['desc']}" for name, info in TOOLS.items())


def execute_tool(name: str, **kwargs) -> str:
    """Exécute un outil par son nom."""
    if name not in TOOLS:
        return f"[ERREUR] Outil inconnu: {name}. Dispo: {list(TOOLS.keys())}"
    try:
        result = TOOLS[name]["fn"](**kwargs)
        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return f"[ERREUR] {name}: {e}"
