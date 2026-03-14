"""training/pretrain.py — Pretrain avec profils small/medium/large.
Usage: python -m training.pretrain --profile medium
       LLM_PROFILE=medium python -m training.pretrain
"""
import os
import sys
from pathlib import Path

# --profile doit etre applique avant import config
for i, a in enumerate(sys.argv):
    if a == "--profile" and i + 1 < len(sys.argv):
        os.environ["LLM_PROFILE"] = sys.argv[i + 1]
        break

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader

from config import MDL_CFG, TRN_CFG, MODEL_DIR, LOG_DIR, PROFILE
from model.transformer import LLMMaison
from training.dataset import PretrainDataset
from agent.notifications import notify


def get_lr(s, wu, mx, hi, lo):
    if s < wu:
        return hi * (s + 1) / wu
    if s >= mx:
        return lo
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * (s - wu) / (mx - wu)))


def pretrain():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"[TRAIN] GPU: {props.name}, VRAM: {vram_gb:.1f} GB")
        if vram_gb < 12 and PROFILE == "medium":
            print("[WARN] VRAM < 12GB — risque OOM avec medium. Utilise --profile small ou gradient_checkpointing.")
        if vram_gb < 20 and PROFILE == "large":
            print("[WARN] VRAM < 20GB — large risque OOM en pretrain full. Prefere LoRA finetune.")

    amp = dev == "cuda" and TRN_CFG.mixed_precision
    dt = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16

    model = LLMMaison(MDL_CFG).to(dev)
    if getattr(TRN_CFG, "gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    ds = PretrainDataset()
    dl = DataLoader(
        ds,
        batch_size=TRN_CFG.batch_size,
        shuffle=True,
        num_workers=TRN_CFG.num_workers,
        pin_memory=dev == "cuda",
        drop_last=True,
    )
    dc, nd = [], []
    for n, p in model.named_parameters():
        (dc if p.dim() >= 2 else nd).append(p)
    opt = torch.optim.AdamW(
        [{"params": dc, "weight_decay": TRN_CFG.weight_decay}, {"params": nd, "weight_decay": 0.0}],
        lr=TRN_CFG.lr,
        betas=(TRN_CFG.beta1, TRN_CFG.beta2),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / "pretrain_log.jsonl"
    step = 0
    toks = 0
    run = 0.0
    t0 = time.time()
    eff_batch = TRN_CFG.batch_size * TRN_CFG.gradient_accumulation
    print(f"[TRAIN] profile={PROFILE} steps={TRN_CFG.max_steps} eff_batch={eff_batch}")
    model.train()

    while step < TRN_CFG.max_steps:
        for bx, by in dl:
            bx, by = bx.to(dev), by.to(dev)
            lr = get_lr(step, TRN_CFG.warmup_steps, TRN_CFG.max_steps, TRN_CFG.lr, TRN_CFG.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            with torch.cuda.amp.autocast(enabled=amp, dtype=dt):
                loss = model(bx, targets=by)["loss"] / TRN_CFG.gradient_accumulation
            scaler.scale(loss).backward()
            if (step + 1) % TRN_CFG.gradient_accumulation == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRN_CFG.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run += loss.item() * TRN_CFG.gradient_accumulation
            toks += bx.numel()
            step += 1
            if step % TRN_CFG.log_interval == 0:
                a = run / TRN_CFG.log_interval
                e = time.time() - t0
                print(f"[{step:>6}] loss={a:.4f} lr={lr:.2e} tok/s={toks/e:.0f}")
                with open(logf, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(a, 4), "profile": PROFILE}) + "\n")
                run = 0.0
            if step % TRN_CFG.save_interval == 0:
                model.save_checkpoint(
                    str(MODEL_DIR / f"step{step}.pt"),
                    step,
                    opt,
                    extra={"profile": PROFILE},
                )
                notify(
                    f"Checkpoint step {step}",
                    event="checkpoint",
                    extra={"step": step, "profile": PROFILE},
                )
            if step >= TRN_CFG.max_steps:
                break
    model.save_checkpoint(str(MODEL_DIR / "pretrain_final.pt"), step, opt, extra={"profile": PROFILE})
    notify(
        "Pre-training complete",
        event="session_end",
        extra={"steps": step, "profile": PROFILE},
    )


if __name__ == "__main__":
    pretrain()
