#!/usr/bin/env python3
"""
run.py — Point d'entrée unique. Un seul fichier pour tout piloter.
Usage: python run.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

MENU = """
╔═══════════════════════════════════════════════════════╗
║              🧠 LLM MAISON — Menu Principal            ║
╠═══════════════════════════════════════════════════════╣
║                                                        ║
║  ─── DONNÉES ───                                       ║
║   1. Crawler Wikipedia                                 ║
║   2. Crawler Reddit                                    ║
║   3. Crawler Web (URLs)                                ║
║   4. Nettoyer + dédupliquer les données                ║
║                                                        ║
║  ─── TOKENIZER ───                                     ║
║   5. Entraîner le tokenizer BPE                        ║
║                                                        ║
║  ─── MODÈLE ───                                        ║
║   6. Tests de santé du modèle                          ║
║   7. Pré-entraîner from scratch                        ║
║   8. Distiller depuis LLaMA 3                          ║
║   9. Fine-tuner sur mes conversations                  ║
║                                                        ║
║  ─── AGENT ───                                         ║
║  10. Chat interactif                                   ║
║  11. Chat + LLaMA 3                                    ║
║  12. Apprentissage autonome (auto-learner)             ║
║                                                        ║
║  ─── MONITORING ───                                    ║
║  13. Dashboard                                         ║
║                                                        ║
║   0. Quitter                                           ║
╚═══════════════════════════════════════════════════════╝"""

COMMANDS = {
    "1":  [sys.executable, "-m", "crawler.wikipedia_crawler"],
    "2":  [sys.executable, "-m", "crawler.reddit_crawler"],
    "3":  [sys.executable, "-m", "crawler.web_crawler"],
    "4":  [sys.executable, "-m", "crawler.cleaner"],
    "5":  [sys.executable, "-m", "tokenizer.train_tokenizer"],
    "6":  [sys.executable, "-m", "model.transformer"],
    "7":  [sys.executable, "-m", "training.pretrain"],
    "8":  [sys.executable, "-m", "training.distill"],
    "9":  [sys.executable, "-m", "training.finetune"],
    "10": [sys.executable, "-m", "agent.chat", "--mode", "tools"],
    "11": [sys.executable, "-m", "agent.chat", "--mode", "llama"],
    "12": [sys.executable, "-m", "agent.auto_learner"],
    "13": [sys.executable, "-m", "monitoring.dashboard"],
}

if __name__ == "__main__":
    while True:
        print(MENU)
        choice = input("\n  Choix > ").strip()
        if choice == "0":
            print("\n  À plus Johann ! 🤙\n")
            break
        elif choice in COMMANDS:
            cmd = COMMANDS[choice]
            print(f"\n  → {' '.join(cmd)}\n")
            try:
                subprocess.run(cmd, cwd=str(ROOT))
            except KeyboardInterrupt:
                print("\n  [Interrompu]")
            input("\n  Entrée pour continuer...")
        else:
            print(f"  [?] Choix invalide: {choice}")
