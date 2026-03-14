"""model/lora.py — LoRA pour fine-tuning léger (~2-5M params au lieu de 150M)."""
import torch, torch.nn as nn
from typing import List

class LoRALinear(nn.Module):
    def __init__(self, orig, r=16, alpha=32, dropout=0.05):
        super().__init__(); self.orig=orig; orig.weight.requires_grad=False
        if orig.bias is not None: orig.bias.requires_grad=False
        self.scale=alpha/r
        self.A=nn.Parameter(torch.randn(orig.in_features,r)*0.01)
        self.B=nn.Parameter(torch.zeros(r,orig.out_features))
        self.drop=nn.Dropout(dropout)
    def forward(self, x): return self.orig(x)+(self.drop(x)@self.A@self.B)*self.scale

def apply_lora(model, targets:List[str], r=16, alpha=32, dropout=0.05):
    for p in model.parameters(): p.requires_grad=False
    n=0
    for name, mod in list(model.named_modules()):
        if name.split(".")[-1] in targets and isinstance(mod, nn.Linear):
            lora=LoRALinear(mod,r,alpha,dropout); n+=lora.A.numel()+lora.B.numel()
            parts=name.split("."); parent=model
            for p in parts[:-1]: parent=parent[int(p)] if p.isdigit() else getattr(parent,p)
            setattr(parent,parts[-1],lora)
    print(f"[LoRA] {n/1e6:.2f}M params sur {targets}"); return n

def save_lora(model, path):
    s={}
    for n,m in model.named_modules():
        if isinstance(m,LoRALinear): s[f"{n}.A"]=m.A.data; s[f"{n}.B"]=m.B.data
    torch.save(s,path); print(f"[LoRA] → {path}")

def load_lora(model, path):
    s=torch.load(path,weights_only=True)
    for n,m in model.named_modules():
        if isinstance(m,LoRALinear) and f"{n}.A" in s: m.A.data=s[f"{n}.A"]; m.B.data=s[f"{n}.B"]
