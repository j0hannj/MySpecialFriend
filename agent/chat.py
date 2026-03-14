"""
agent/chat.py — Interface de chat interactive.
Sauvegarde auto les conversations → finetune plus tard.

Usage:
  python -m agent.chat                      # Mode tools (outils seuls)
  python -m agent.chat --mode llama       # LLaMA via Ollama (défaut)
  python -m agent.chat --mode llama --backend lmstudio
  python -m agent.chat --mode llama --backend transformers
  python -m agent.chat --mode local         # Avec notre modèle entraîné

Commandes dans le chat :
  /search <query>    Recherche web
  /explore <topic>   Exploration autonome
  /memory            Voir ce que l'agent sait
  /stats             Statistiques
  /learn <fait>      Enseigner un fait manuellement
  /save              Forcer la sauvegarde
  /quit              Quitter
"""
import json
import sys
import datetime
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONV_DIR, MODEL_DIR, TOKENIZER_DIR
from agent.agent import Agent


def save_conversation(messages: list, conv_dir: Path) -> Path:
    """Sauvegarde en .jsonl pour le fine-tuning ultérieur."""
    conv_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = conv_dir / f"conv_{ts}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(messages, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="LLM Maison — Chat Agent")
    parser.add_argument("--mode", choices=["llama", "local", "tools"],
                        default="tools")
    args = parser.parse_args()

    # ── Banner ──
    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║          🧠 LLM MAISON — Chat Agent               ║")
    print("╠═══════════════════════════════════════════════════╣")
    backend_line = f"{args.mode}" + (f" / {args.backend}" if args.mode == "llama" else "")
    print(f"║  Mode : {backend_line:<42}║")
    print("║  /search /explore /memory /stats /learn /quit      ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    # ── Init agent ──
    model, tokenizer = None, None

    if args.mode == "local":
        try:
            from model.transformer import LLMMaison
            from tokenizer.bpe import BPETokenizer
            ckpt = MODEL_DIR / "pretrain_final.pt"
            if not ckpt.exists():
                ckpt = MODEL_DIR / "distill_final.pt"
            if ckpt.exists():
                model, _ = LLMMaison.load_checkpoint(str(ckpt))
                tokenizer = BPETokenizer.load(str(TOKENIZER_DIR))
            else:
                print("[!] Pas de checkpoint trouvé → fallback mode tools")
                args.mode = "tools"
        except Exception as e:
            print(f"[!] Erreur chargement modèle: {e} → fallback tools")
            args.mode = "tools"

    agent = Agent(
        mode=args.mode,
        model=model,
        tokenizer=tokenizer,
        llm_backend=args.backend if args.mode == "llama" else "ollama",
    )
    if args.mode == "llama":
        agent._ollama_url = args.ollama_url
        agent._lmstudio_url = args.lmstudio_url
        agent._api_model = args.api_model
    messages = []
    agent.memory.increment_conversations()

    # ── Boucle de chat ──
    while True:
        try:
            user_input = input("\n🧑 Toi > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[Au revoir !]")
            break

        if not user_input:
            continue

        # ── Commandes spéciales ──

        if user_input in ("/quit", "/exit", "/q"):
            if messages:
                p = save_conversation(messages, CONV_DIR)
                print(f"[💾 Conversation sauvée → {p}]")
            print("[Au revoir !]")
            break

        if user_input.startswith("/search "):
            from agent.tools import execute_tool
            r = execute_tool("search_web", query=user_input[8:])
            print(f"\n🔍 Résultats:\n{r}")
            continue

        if user_input.startswith("/explore "):
            agent.auto_explore(user_input[9:])
            continue

        if user_input == "/memory":
            print(f"\n🧠 Mémoire:\n{agent.memory.get_context()}")
            continue

        if user_input == "/stats":
            print(f"\n📊 {agent.memory.stats()}")
            continue

        if user_input.startswith("/learn "):
            fact = user_input[7:]
            agent.memory.add_fact(fact, source="manual")
            print(f"[✓ Appris: {fact}]")
            continue

        if user_input == "/save":
            if messages:
                p = save_conversation(messages, CONV_DIR)
                print(f"[💾 → {p}]")
            else:
                print("[Rien à sauvegarder]")
            continue

        # ── Message normal → agent ──
        messages.append({"role": "user", "content": user_input})

        response = agent.process(user_input)
        messages.append({"role": "assistant", "content": response})

        print(f"\n🤖 Agent:\n{response}")

        # Auto-save toutes les 5 échanges (10 messages)
        if len(messages) % 10 == 0:
            p = save_conversation(messages, CONV_DIR)
            print(f"[💾 Auto-save → {p}]")


if __name__ == "__main__":
    main()
