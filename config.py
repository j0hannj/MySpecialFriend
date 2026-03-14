"""
CONFIG.PY — Source unique pour TOUS les paramètres du projet.
Profils : small (~150M), medium (~1.5B), large (~3B) via --profile ou LLM_PROFILE.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data_raw"
PROCESSED_DIR = ROOT / "data_processed"
CHECKPOINT_DIR= Path(os.environ.get("CHECKPOINT_DIR", "")) if os.environ.get("CHECKPOINT_DIR") else (ROOT / "checkpoints")
TOKENIZER_DIR = ROOT / "tokenizer_model"
# Dossier local pour le tokenizer Llama 3 (sans appel HF). Voir teacher_tokenizer/README.md
TEACHER_TOKENIZER_DIR = ROOT / "teacher_tokenizer"
LOG_DIR       = ROOT / "logs"
CONV_DIR      = ROOT / "conversations"

for _d in [DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, TOKENIZER_DIR, TEACHER_TOKENIZER_DIR, LOG_DIR, CONV_DIR]:
    _d.mkdir(exist_ok=True)


def get_model_profile():
    """Profil depuis CLI --profile <name> ou env LLM_PROFILE (défaut: 3b pour distill plus costaud)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--profile" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower()
    return os.environ.get("LLM_PROFILE", "3b").lower()


@dataclass
class TokenizerConfig:
    vocab_size: int = 32_000
    min_frequency: int = 2
    special_tokens: list = field(default_factory=lambda: [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>",
        "<USER>", "<ASST>", "<SYS>",
    ])

@dataclass
class ModelConfig:
    """~3B params (profil large). Presets MODEL_SMALL / MODEL_MEDIUM pour --profile."""
    vocab_size: int = 32_000
    d_model: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 26
    d_ff: int = 5632
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    def count_params(self):
        emb = self.vocab_size * self.d_model
        attn = self.n_layers * ((self.n_heads + 2*self.n_kv_heads) * self.d_head * self.d_model + self.d_model**2)
        ffn = self.n_layers * 3 * self.d_model * self.d_ff
        return emb + attn + ffn


# --- Presets (RTX 4080 16GB / 32GB RAM) ---
MODEL_SMALL = ModelConfig(
    vocab_size=32_000,
    d_model=768,
    n_heads=12,
    n_kv_heads=4,
    n_layers=12,
    d_ff=2048,
    max_seq_len=1024,
)

MODEL_MEDIUM = ModelConfig(
    vocab_size=32_000,
    d_model=1536,
    n_heads=16,
    n_kv_heads=4,
    n_layers=24,
    d_ff=4096,
    max_seq_len=1024,
)

MODEL_LARGE = ModelConfig(
    vocab_size=32_000,
    d_model=2048,
    n_heads=32,
    n_kv_heads=8,
    n_layers=26,
    d_ff=5632,
    max_seq_len=1024,
)

# ~3B params — student plus costaud pour la distillation
MODEL_3B = ModelConfig(
    vocab_size=32_000,
    d_model=2560,
    n_heads=32,
    n_kv_heads=8,
    n_layers=32,
    d_ff=9728,
    max_seq_len=1024,
)

_PROFILES = {
    "small": MODEL_SMALL,
    "medium": MODEL_MEDIUM,
    "large": MODEL_LARGE,
    "3b": MODEL_3B,
}

PROFILE = get_model_profile()
MDL_CFG = _PROFILES.get(PROFILE, MODEL_3B)
if PROFILE not in _PROFILES:
    print(f"[CONFIG] Profil inconnu '{PROFILE}', fallback 3b")
    PROFILE = "3b"
    MDL_CFG = MODEL_3B
_nparams = MDL_CFG.count_params()
print(f"[CONFIG] Profil: {PROFILE} — {MDL_CFG.d_model}d, {MDL_CFG.n_layers}L, ~{_nparams/1e6:.0f}M params")


@dataclass
class TrainConfig:
    """Aligné distillation 3B + DeepSpeed ZeRO (accum souvent géré par DS)."""
    batch_size: int = 2
    gradient_accumulation: int = 8
    seq_len: int = 1024
    lr: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100_000
    eval_interval: int = 500
    save_interval: int = 2000
    log_interval: int = 10
    mixed_precision: bool = True
    num_workers: int = 4
    train_split: float = 0.95
    gradient_checkpointing: bool = True
    use_deepspeed: bool = True


@dataclass
class DistillConfig:
    # teacher_model = chargement HF (gated) — utilisé seulement si backend=transformers
    teacher_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Tokenizer 100% local : fichiers dans teacher_tokenizer/ (voir teacher_tokenizer/README.md)
    teacher_tokenizer: str = str(TEACHER_TOKENIZER_DIR)
    temperature: float = 2.0
    alpha_kd: float = 0.7
    alpha_ce: float = 0.3
    max_steps: int = 50_000
    save_interval: int = 500  # checkpoint tous les N steps (etait 2000) — plus frequent = plus de Go sur disque
    max_samples: int = 20_000  # charge corpus distill
    lr: float = 5e-5
    batch_size: int = 2
    load_in_4bit: bool = True
    gradient_accumulation: int = 8
    gradient_checkpointing: bool = True
    backend: str = "ollama"
    api_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    model_path: str = ""

@dataclass
class FinetuneConfig:
    """3B : LoRA uniquement sur 4080 — use_lora toujours True en pratique."""
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 8
    warmup_ratio: float = 0.1
    max_seq_len: int = 1024
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True
    use_deepspeed: bool = False

@dataclass
class CrawlerConfig:
    output_dir: str = str(DATA_DIR)
    delay: float = 1.5
    max_retries: int = 3
    timeout: int = 30
    user_agent: str = "LLM-Maison/1.0 (Educational Research)"
    wiki_lang: str = "fr"
    wiki_max_articles: int = 50_000
    wiki_min_chars: int = 500
    reddit_subs: list = field(default_factory=lambda: [
        "france", "AskReddit", "explainlikeimfive", "science",
        "worldnews", "todayilearned", "AskScience", "philosophy",
        "MachineLearning", "LocalLLaMA", "learnprogramming",
    ])
    reddit_max_per_sub: int = 5000
    reddit_min_score: int = 10

@dataclass
class AgentConfig:
    max_steps: int = 10
    search_top_k: int = 5
    memory_path: str = str(ROOT / "agent_memory.json")
    auto_learn: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    rep_penalty: float = 1.15

@dataclass
class NotifConfig:
    backend: str = "console"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""
    min_importance: int = 1

# Instances globales (MDL_CFG déjà défini par profil ci-dessus)
tok_cfg     = TokenizerConfig()
model_cfg   = MDL_CFG
train_cfg   = TrainConfig()
# Petit profil seulement : pretrain rapide si --profile small
if PROFILE == "small":
    train_cfg.batch_size = 8
    train_cfg.gradient_accumulation = 8
    train_cfg.lr = 3e-4
    train_cfg.min_lr = 3e-5
    train_cfg.warmup_steps = 1000
    train_cfg.gradient_checkpointing = False
distill_cfg = DistillConfig()
ft_cfg      = FinetuneConfig()
crawl_cfg   = CrawlerConfig()
agent_cfg   = AgentConfig()
notif_cfg   = NotifConfig()

TOK_CFG   = tok_cfg
TRN_CFG   = train_cfg
DST_CFG   = distill_cfg
FT_CFG    = ft_cfg
CRW_CFG   = crawl_cfg
AGT_CFG   = agent_cfg
NTF_CFG   = notif_cfg
MODEL_DIR = CHECKPOINT_DIR

if __name__ == "__main__":
    n = MDL_CFG.count_params()
    print(f"LLM Maison — {MDL_CFG.n_layers}L x {MDL_CFG.d_model}D x {MDL_CFG.n_heads}H")
    print(f"~{n/1e6:.0f}M params | ctx: {MDL_CFG.max_seq_len} | GQA: {MDL_CFG.n_kv_heads}KV | profile={PROFILE}")
