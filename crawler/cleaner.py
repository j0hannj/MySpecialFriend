"""
crawler/cleaner.py — Nettoyage + dédup des JSONL dans data_raw.
Wikipedia : retire titres == ==, templates {{ }}, liens [[ ]], HTML, etc.

Usage:
  python -m crawler.cleaner
  python -m crawler.cleaner --min-chars 300
  python -m crawler.cleaner --in-place   # écrase chaque fichier (backup .bak)
"""
import json
import hashlib
import re
import unicodedata
import sys
import argparse
import shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


def _strip_wiki_headings(text):
    """== Titre == ou === Sous-titre === → saut de ligne + titre sans '='."""
    # Lignes du type ====== ... ======
    def repl_heading(m):
        inner = m.group(1).strip()
        if not inner:
            return "\n"
        return "\n\n" + inner + "\n\n"

    text = re.sub(r"^[ \t]*={2,}\s*([^=\n]+?)\s*={2,}\s*$", repl_heading, text, flags=re.MULTILINE)
    return text


def _strip_wiki_templates(text):
    """Supprime {{ ... }} (itéré pour imbrication partielle)."""
    for _ in range(20):
        new = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.DOTALL)
        if new == text:
            break
        text = new
    # Restes {{ sans fermeture
    text = re.sub(r"\{\{[^}]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[^{]*\}\}", "", text, flags=re.MULTILINE)
    return text


def _strip_wiki_links(text):
    """[[Lien|Texte]] → Texte ; [[Lien]] → Lien."""
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    return text


def _strip_html_and_entities(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    return text


def _strip_tables_wiki(text):
    """{| ... |} tables wiki → supprimées par blocs."""
    for _ in range(10):
        new = re.sub(r"\{\|[^{}]*?\|\}", "", text, flags=re.DOTALL)
        if new == text:
            break
        text = new
    return text


# Sections wiki en fin d'article : à couper (évite refs / liens externes en bruit)
def truncate_wiki_tail(text):
    """Coupe le texte avant la première section Notes/Références/Voir aussi/etc."""
    if not text:
        return text
    # Ligne seule = section wiki à supprimer (évite dépendre des accents NFC/NFD)
    def is_tail_section(low_line):
        if not low_line:
            return False
        if low_line.startswith("notes et ") and "rences" in low_line:
            return True  # Notes et références / references
        if low_line.startswith("voir aussi"):
            return True
        if low_line.startswith("bibliographie"):
            return True
        if low_line.startswith("liens externes") or low_line.startswith("external links"):
            return True
        if low_line in ("références", "references"):
            return True
        return False

    lines = text.split("\n")
    out_lines = []
    for line in lines:
        low_line = line.lower().strip()
        if is_tail_section(low_line):
            break
        out_lines.append(line)
    if len(out_lines) < len(lines):
        text = "\n".join(out_lines).rstrip()
    return text


def normalize_after_json_load(text):
    """
    Garantit un texte prêt pour BPE/pretrain après json.loads(line):
    - json.loads interprète déjà \\n en vrai newline ; ici on sécurise le reste.
    - \\r\\n -> \\n
    - séquences littérales backslash+n (mauvais export) -> newline
    - tronque les queues wiki (Notes et références, ...)
    """
    if not text:
        return ""
    t = text
    # Windows / vieux exports
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Si beaucoup de "\n" littéraux (2 chars) sans vrais newlines -> décodage foireux
    if t.count("\\n") >= 2 and t.count("\n") < t.count("\\n"):
        t = t.replace("\\n", "\n").replace("\\t", " ")
    t = truncate_wiki_tail(t)
    return t


def clean_text(text):
    """Nettoyage agressif pour corpus wiki / web."""
    if not text:
        return ""
    t = normalize_after_json_load(text)
    t = unicodedata.normalize("NFKC", t)
    t = _strip_wiki_templates(t)
    t = _strip_tables_wiki(t)
    t = _strip_wiki_links(t)
    t = _strip_wiki_headings(t)
    t = _strip_html_and_entities(t)
    # URLs
    t = re.sub(r"https?://\S+", "", t)
    # Catégorie / Fichier restants
    t = re.sub(r"\[\[Catégorie:[^\]]*\]\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[Category:[^\]]*\]\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[Fichier:[^\]]*\]\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[File:[^\]]*\]\]", "", t, flags=re.IGNORECASE)
    # Espaces / newlines
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = truncate_wiki_tail(t)
    return t.strip()


def clean(d=None, min_c=200, in_place=False):
    d = Path(d or DATA_DIR)
    if in_place:
        files = [f for f in d.glob("*.jsonl")]
    else:
        out = d / "corpus_clean.jsonl"
        files = [f for f in d.glob("*.jsonl") if f.name != "corpus_clean.jsonl"]

    if not files:
        print(f"[CLEAN] Rien à traiter dans {d}")
        return

    seen = set()
    st = Counter()

    if in_place:
        for fp in files:
            bak = fp.with_suffix(fp.suffix + ".bak")
            shutil.copy2(fp, bak)
            print(f"[CLEAN] Backup -> {bak}")
            tmp = fp.with_suffix(".tmp.jsonl")
            with open(tmp, "w", encoding="utf-8") as o:
                for line in open(fp, "r", encoding="utf-8", errors="replace"):
                    st["total"] += 1
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        st["err"] += 1
                        continue
                    t = clean_text(doc.get("text", ""))
                    if len(t) < min_c:
                        st["short"] += 1
                        continue
                    h = hashlib.md5(re.sub(r"\s+", " ", t.lower()).encode()).hexdigest()
                    if h in seen:
                        st["dup"] += 1
                        continue
                    seen.add(h)
                    alpha = sum(c.isalpha() for c in t)
                    if alpha / max(len(t), 1) < 0.4:
                        st["bad"] += 1
                        continue
                    doc["text"] = t
                    doc["chars"] = len(t)
                    o.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    st["kept"] += 1
            tmp.replace(fp)
            print(f"[CLEAN] Reecrit -> {fp}")
        print(f"[CLEAN] Total:{st['total']} Gardés:{st['kept']} Dups:{st['dup']} Short:{st['short']} Bad:{st['bad']}")
        return

    with open(out, "w", encoding="utf-8") as o:
        for fp in files:
            for line in open(fp, "r", encoding="utf-8", errors="replace"):
                st["total"] += 1
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    st["err"] += 1
                    continue
                t = clean_text(doc.get("text", ""))
                if len(t) < min_c:
                    st["short"] += 1
                    continue
                h = hashlib.md5(re.sub(r"\s+", " ", t.lower()).encode()).hexdigest()
                if h in seen:
                    st["dup"] += 1
                    continue
                seen.add(h)
                alpha = sum(c.isalpha() for c in t)
                if alpha / max(len(t), 1) < 0.4:
                    st["bad"] += 1
                    continue
                doc["text"] = t
                doc["chars"] = len(t)
                o.write(json.dumps(doc, ensure_ascii=False) + "\n")
                st["kept"] += 1

    print(
        f"[CLEAN] Total:{st['total']} Gardés:{st['kept']} Dups:{st['dup']} "
        f"Short:{st['short']} Bad:{st['bad']} Err:{st['err']} -> {out}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Nettoyage data_raw/*.jsonl")
    p.add_argument("--dir", type=str, default=None, help="Dossier (défaut: DATA_DIR)")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Réécrit chaque .jsonl (backup .bak). Sinon sort corpus_clean.jsonl",
    )
    args = p.parse_args()
    clean(d=args.dir, min_c=args.min_chars, in_place=args.in_place)
