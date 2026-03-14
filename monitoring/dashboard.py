"""
monitoring/dashboard.py — Dashboard : stats entraînement, mémoire, données.
Tout vient de fichiers JSON/JSONL → auditable à la main.
Usage: python -m monitoring.dashboard
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_DIR, DATA_DIR, PROCESSED_DIR, MODEL_DIR, CONV_DIR, ROOT


def read_log(path: Path) -> list:
    entries = []
    if path.exists():
        for line in open(path, "r"):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def show_data():
    section("📁 DONNÉES")
    # Brutes
    raw = list(DATA_DIR.glob("*.jsonl")) + list(DATA_DIR.glob("*.txt"))
    total = sum(f.stat().st_size for f in raw) if raw else 0
    print(f"  Fichiers bruts    : {len(raw)}")
    print(f"  Taille totale     : {total / 1e6:.1f} MB")

    # Auto-learned
    auto_dir = DATA_DIR / "auto_learned"
    if auto_dir.exists():
        af = list(auto_dir.glob("*.jsonl"))
        if af:
            n = sum(1 for f in af for _ in open(f))
            print(f"  Auto-appris       : {n} documents")

    # Tokenisé
    bin_f = PROCESSED_DIR / "train.bin"
    if bin_f.exists():
        import numpy as np
        data = np.memmap(str(bin_f), dtype=np.uint16, mode="r")
        print(f"  Tokens (train.bin): {len(data):,}")
        print(f"  Taille .bin       : {bin_f.stat().st_size / 1e6:.1f} MB")


def show_model():
    section("🧠 MODÈLE")
    ckpts = sorted(MODEL_DIR.glob("*.pt"))
    if not ckpts:
        print("  Aucun checkpoint")
        return
    for c in ckpts[-5:]:
        print(f"  {c.name:<30} {c.stat().st_size / 1e6:>8.1f} MB")

    # Info du dernier
    import torch
    try:
        d = torch.load(str(ckpts[-1]), map_location="cpu", weights_only=False)
        if "config" in d:
            cfg = d["config"]
            n = sum(v.numel() for v in d["model_state"].values())
            print(f"\n  Dernier checkpoint : {ckpts[-1].name}")
            print(f"  Paramètres         : {n / 1e6:.1f}M")
            print(f"  Step               : {d.get('step', '?')}")
            print(f"  d_model={cfg.get('d_model')}, n_layers={cfg.get('n_layers')}, "
                  f"n_heads={cfg.get('n_heads')}")
    except Exception as e:
        print(f"  (erreur lecture: {e})")


def show_training():
    section("📊 ENTRAÎNEMENT")
    for name in ["pretrain_log.jsonl", "distill_log.jsonl", "finetune_log.jsonl"]:
        log = read_log(LOG_DIR / name)
        if not log:
            print(f"  {name}: (vide)")
            continue
        last = log[-1]
        print(f"\n  {name} — {len(log)} entrées")
        print(f"    Dernier step : {last.get('step', '?')}")
        print(f"    Dernière loss: {last.get('loss', '?')}")
        if len(log) > 10:
            early = [e["loss"] for e in log[:10] if "loss" in e]
            recent = [e["loss"] for e in log[-10:] if "loss" in e]
            if early and recent:
                e_avg, r_avg = sum(early) / len(early), sum(recent) / len(recent)
                print(f"    Loss début   : {e_avg:.4f}")
                print(f"    Loss récente : {r_avg:.4f}")
                print(f"    Δ            : {e_avg - r_avg:+.4f}")


def show_memory():
    section("🧠 MÉMOIRE AGENT")
    mem_f = ROOT / "agent_memory.json"
    if not mem_f.exists():
        print("  (pas encore de mémoire)")
        return
    m = json.loads(mem_f.read_text())
    print(f"  Faits        : {len(m.get('facts', []))}")
    print(f"  Préférences  : {len(m.get('preferences', {}))}")
    print(f"  Topics       : {len(m.get('topics_discussed', []))}")
    print(f"  Conversations: {m.get('conversation_count', 0)}")
    print(f"  Dernière MAJ : {m.get('last_updated', 'jamais')}")
    facts = m.get("facts", [])
    if facts:
        print(f"\n  5 derniers faits:")
        for f in facts[-5:]:
            print(f"    • {f['fact'][:80]}...")


def show_conversations():
    section("💬 CONVERSATIONS (fine-tuning)")
    files = sorted(CONV_DIR.glob("*.jsonl"))
    if not files:
        print("  Aucune conversation sauvée")
        return
    n_conv = n_msg = 0
    for f in files:
        for line in open(f):
            try:
                c = json.loads(line)
                n_conv += 1
                n_msg += len(c) if isinstance(c, list) else 0
            except:
                pass
    print(f"  Fichiers      : {len(files)}")
    print(f"  Conversations : {n_conv}")
    print(f"  Messages total: {n_msg}")


def dashboard():
    print("╔═══════════════════════════════════════════════════╗")
    print("║        🧠 LLM MAISON — Monitoring Dashboard        ║")
    print("╚═══════════════════════════════════════════════════╝")
    show_data()
    show_model()
    show_training()
    show_memory()
    show_conversations()
    print(f"\n{'═'*52}")
    print("Tous les fichiers sont en JSON/JSONL → ouvre-les pour auditer.")


if __name__ == "__main__":
    dashboard()
