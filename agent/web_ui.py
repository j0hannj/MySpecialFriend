"""
agent/web_ui.py — Interface web Gradio pour LLM Maison.
Chat, mémoire, stats, exploration.

Usage:
    python -m agent.web_ui                    # Mode tools (sans LLM)
    python -m agent.web_ui --mode llama       # Avec LLaMA 3
    python -m agent.web_ui --mode local       # Avec notre modèle
    python -m agent.web_ui --port 7861        # Port personnalisé
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from agent.agent import Agent
from agent.memory import AgentMemory
from agent.tools import execute_tool, list_tools
from config import (
    LOG_DIR, DATA_DIR, CHECKPOINT_DIR, PROCESSED_DIR, ROOT,
    MDL_CFG, TRN_CFG, AGT_CFG
)


class WebUI:
    """Gradio-based web interface for LLM Maison."""
    
    def __init__(self, mode: str = "tools", model=None, tokenizer=None):
        self.agent = Agent(mode=mode, model=model, tokenizer=tokenizer)
        self.memory = self.agent.memory
        self.chat_history = []
    
    def chat(self, message: str, history: list) -> tuple:
        """Process chat message and return response."""
        if not message.strip():
            return history, ""
        
        if message.startswith("/"):
            response = self._handle_command(message)
        else:
            response = self.agent.process(message)
        
        history.append((message, response))
        self.memory.increment_conversations()
        
        return history, ""
    
    def _handle_command(self, cmd: str) -> str:
        """Handle slash commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "/help":
            return """**Available Commands:**
- `/search <query>` - Search the web
- `/explore <topic>` - Auto-explore a topic
- `/memory` - Show memory stats
- `/facts` - Show recent facts
- `/clear` - Clear chat history
- `/tools` - List available tools"""
        
        elif command == "/search":
            if not args:
                return "Usage: /search <query>"
            results = execute_tool("search_web", query=args)
            return f"**Search results for '{args}':**\n\n{results}"
        
        elif command == "/explore":
            if not args:
                return "Usage: /explore <topic>"
            self.agent.auto_explore(args)
            return f"**Exploration of '{args}' complete.**\n\n{self.memory.stats()}"
        
        elif command == "/memory":
            return f"**Memory Stats:**\n\n{self.memory.stats()}"
        
        elif command == "/facts":
            facts = self.memory.data.get("facts", [])[-10:]
            if not facts:
                return "No facts in memory yet."
            lines = [f"- {f['fact'][:100]}..." if len(f['fact']) > 100 else f"- {f['fact']}" 
                     for f in facts]
            return "**Recent Facts:**\n\n" + "\n".join(lines)
        
        elif command == "/clear":
            self.chat_history = []
            return "Chat history cleared."
        
        elif command == "/tools":
            return f"**Available Tools:**\n\n```\n{list_tools()}\n```"
        
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def get_memory_display(self) -> str:
        """Get formatted memory content for display."""
        facts = self.memory.data.get("facts", [])
        prefs = self.memory.data.get("preferences", {})
        topics = self.memory.data.get("topics_discussed", [])
        
        output = []
        
        output.append(f"## Memory Overview")
        output.append(f"- **Total Facts:** {len(facts)}")
        output.append(f"- **Preferences:** {len(prefs)}")
        output.append(f"- **Topics Explored:** {len(topics)}")
        output.append(f"- **Conversations:** {self.memory.data.get('conversation_count', 0)}")
        output.append(f"- **Last Updated:** {self.memory.data.get('last_updated', 'Never')}")
        
        if topics:
            output.append(f"\n## Recent Topics")
            for t in topics[-10:]:
                output.append(f"- {t}")
        
        if facts:
            output.append(f"\n## Recent Facts ({len(facts)} total)")
            for f in facts[-15:]:
                source = f.get("source", "unknown")
                ts = f.get("timestamp", "")[:10]
                output.append(f"- [{ts}] {f['fact'][:150]}...")
                output.append(f"  *Source: {source}*")
        
        if prefs:
            output.append(f"\n## Preferences")
            for k, v in prefs.items():
                output.append(f"- **{k}:** {v}")
        
        return "\n".join(output)
    
    def get_stats_display(self) -> str:
        """Get system stats for dashboard."""
        output = []
        
        output.append("## System Dashboard")
        output.append("")
        
        output.append("### Data")
        raw_files = list(DATA_DIR.glob("*.jsonl")) + list(DATA_DIR.glob("*.txt"))
        raw_size = sum(f.stat().st_size for f in raw_files) / 1e6 if raw_files else 0
        output.append(f"- **Raw data files:** {len(raw_files)}")
        output.append(f"- **Raw data size:** {raw_size:.1f} MB")
        
        train_bin = PROCESSED_DIR / "train.bin"
        if train_bin.exists():
            import numpy as np
            try:
                data = np.memmap(str(train_bin), dtype=np.uint16, mode="r")
                output.append(f"- **Tokenized data:** {len(data):,} tokens")
                output.append(f"- **train.bin size:** {train_bin.stat().st_size / 1e6:.1f} MB")
            except Exception:
                output.append("- **Tokenized data:** Error reading")
        else:
            output.append("- **Tokenized data:** Not found")
        
        output.append("")
        output.append("### Model")
        ckpts = sorted(CHECKPOINT_DIR.glob("*.pt"))
        output.append(f"- **Checkpoints:** {len(ckpts)}")
        if ckpts:
            latest = ckpts[-1]
            output.append(f"- **Latest:** {latest.name} ({latest.stat().st_size / 1e6:.1f} MB)")
        
        output.append(f"- **Config:** {MDL_CFG.n_layers}L × {MDL_CFG.d_model}D × {MDL_CFG.n_heads}H")
        output.append(f"- **Params:** ~{MDL_CFG.count_params() / 1e6:.0f}M")
        
        output.append("")
        output.append("### Training Logs")
        for log_name in ["pretrain_log.jsonl", "distill_log.jsonl", "finetune_log.jsonl"]:
            log_path = LOG_DIR / log_name
            if log_path.exists():
                try:
                    lines = log_path.read_text().strip().split("\n")
                    entries = [json.loads(l) for l in lines if l.strip()]
                    if entries:
                        last = entries[-1]
                        output.append(f"- **{log_name}:** {len(entries)} entries, "
                                      f"last step={last.get('step', '?')}, loss={last.get('loss', '?')}")
                except Exception:
                    output.append(f"- **{log_name}:** Error reading")
            else:
                output.append(f"- **{log_name}:** Not found")
        
        output.append("")
        output.append("### Agent")
        output.append(f"- **Mode:** {self.agent.mode}")
        output.append(f"- {self.memory.stats()}")
        
        return "\n".join(output)
    
    def explore_topic(self, topic: str) -> str:
        """Explore a topic and return results."""
        if not topic.strip():
            return "Please enter a topic to explore."
        
        self.agent.auto_explore(topic)
        return f"**Exploration complete!**\n\n{self.memory.stats()}\n\n" + self.get_memory_display()
    
    def search_facts(self, query: str) -> str:
        """Search facts in memory."""
        if not query.strip():
            return "Please enter a search query."
        
        results = self.memory.search_facts(query, top_k=10)
        if not results:
            return f"No facts found matching '{query}'."
        
        output = [f"**Facts matching '{query}':**\n"]
        for f in results:
            output.append(f"- {f['fact']}")
            output.append(f"  *Source: {f.get('source', 'unknown')}*\n")
        
        return "\n".join(output)
    
    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""
        
        with gr.Blocks(title="LLM Maison", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🏠 LLM Maison")
            gr.Markdown(f"*Mode: {self.agent.mode} | {self.memory.stats()}*")
            
            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        show_label=False
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message or /help for commands...",
                            scale=4,
                            show_label=False
                        )
                        send_btn = gr.Button("Send", scale=1)
                    
                    gr.Markdown("*Commands: /help, /search, /explore, /memory, /facts, /tools*")
                    
                    msg.submit(self.chat, [msg, chatbot], [chatbot, msg])
                    send_btn.click(self.chat, [msg, chatbot], [chatbot, msg])
                
                with gr.Tab("🧠 Memory"):
                    memory_display = gr.Markdown(value=self.get_memory_display)
                    
                    with gr.Row():
                        refresh_memory_btn = gr.Button("🔄 Refresh")
                    
                    gr.Markdown("### Search Facts")
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Search in memory...",
                            scale=4
                        )
                        search_btn = gr.Button("Search", scale=1)
                    search_results = gr.Markdown()
                    
                    refresh_memory_btn.click(self.get_memory_display, outputs=memory_display)
                    search_btn.click(self.search_facts, inputs=search_input, outputs=search_results)
                    search_input.submit(self.search_facts, inputs=search_input, outputs=search_results)
                
                with gr.Tab("📊 Stats"):
                    stats_display = gr.Markdown(value=self.get_stats_display)
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                    refresh_stats_btn.click(self.get_stats_display, outputs=stats_display)
                
                with gr.Tab("🔍 Explore"):
                    gr.Markdown("### Auto-Explore a Topic")
                    gr.Markdown("The agent will search the web and extract facts about the topic.")
                    
                    with gr.Row():
                        explore_input = gr.Textbox(
                            label="Topic",
                            placeholder="Enter a topic to explore...",
                            scale=4
                        )
                        explore_btn = gr.Button("🚀 Explore", scale=1)
                    
                    explore_output = gr.Markdown()
                    
                    explore_btn.click(self.explore_topic, inputs=explore_input, outputs=explore_output)
                    explore_input.submit(self.explore_topic, inputs=explore_input, outputs=explore_output)
                    
                    gr.Markdown("### Suggested Topics")
                    gr.Markdown("""
- Intelligence artificielle 2025
- Machine learning transformers
- Physique quantique vulgarisation
- Python bonnes pratiques
- Deep learning architectures
""")
                
                with gr.Tab("🛠️ Tools"):
                    gr.Markdown("### Available Tools")
                    gr.Markdown(f"```\n{list_tools()}\n```")
                    
                    gr.Markdown("### Manual Tool Execution")
                    with gr.Row():
                        tool_name = gr.Dropdown(
                            choices=["search_web", "fetch_url", "calculator", "get_datetime"],
                            label="Tool"
                        )
                        tool_args = gr.Textbox(
                            label="Arguments (JSON)",
                            placeholder='{"query": "hello world"}'
                        )
                    tool_btn = gr.Button("Execute Tool")
                    tool_output = gr.Textbox(label="Output", lines=10)
                    
                    def run_tool(name, args_str):
                        try:
                            args = json.loads(args_str) if args_str.strip() else {}
                            return execute_tool(name, **args)
                        except json.JSONDecodeError:
                            return "Error: Invalid JSON arguments"
                        except Exception as e:
                            return f"Error: {e}"
                    
                    tool_btn.click(run_tool, inputs=[tool_name, tool_args], outputs=tool_output)
        
        return demo


def main():
    parser = argparse.ArgumentParser(description="LLM Maison Web UI")
    parser.add_argument("--mode", choices=["tools", "llama", "local"], default="tools",
                        help="Agent mode (tools=no LLM, llama=LLaMA 3, local=our model)")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for local mode)")
    args = parser.parse_args()
    
    model = None
    tokenizer = None
    
    if args.mode == "local":
        if args.checkpoint:
            from model.transformer import LLMMaison
            from tokenizer import BPETokenizer
            from config import TOKENIZER_DIR
            
            print(f"[WEB] Loading model from {args.checkpoint}...")
            model, _ = LLMMaison.load_checkpoint(args.checkpoint, "cuda" if __import__("torch").cuda.is_available() else "cpu")
            model.eval()
            
            print(f"[WEB] Loading tokenizer from {TOKENIZER_DIR}...")
            tokenizer = BPETokenizer.load(str(TOKENIZER_DIR))
        else:
            print("[WEB] Warning: No checkpoint specified for local mode. Falling back to tools mode.")
            args.mode = "tools"
    
    print(f"[WEB] Starting LLM Maison Web UI...")
    print(f"[WEB] Mode: {args.mode}")
    
    ui = WebUI(mode=args.mode, model=model, tokenizer=tokenizer)
    demo = ui.build_interface()
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
