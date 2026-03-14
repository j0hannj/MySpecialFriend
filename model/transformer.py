"""
model/transformer.py — Transformer LLaMA-style: RMSNorm, RoPE, GQA, SwiGLU.
~150M params. python -m model.transformer pour les tests.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).parent.parent))
from config import MDL_CFG

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps=eps; self.w=nn.Parameter(torch.ones(d))
    def forward(self, x):
        return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)).type_as(x)*self.w

def precompute_rope(dim, mlen, theta=10000.0, dev="cpu"):
    f=1.0/(theta**(torch.arange(0,dim,2,device=dev).float()/dim))
    a=torch.outer(torch.arange(mlen,device=dev).float(),f); return a.cos(),a.sin()

def apply_rope(x, cos, sin, start_pos=0):
    d=x.shape[-1]//2; T=x.shape[2]
    c,s=cos[start_pos:start_pos+T][None,None],sin[start_pos:start_pos+T][None,None]
    x1,x2=x[...,:d],x[...,d:]
    return torch.cat([x1*c-x2*s,x1*s+x2*c],-1)

class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.nh=cfg.n_heads; self.nkv=cfg.n_kv_heads; self.dh=cfg.d_head; self.rep=self.nh//self.nkv
        self.q_proj=nn.Linear(cfg.d_model,self.nh*self.dh,bias=False)
        self.k_proj=nn.Linear(cfg.d_model,self.nkv*self.dh,bias=False)
        self.v_proj=nn.Linear(cfg.d_model,self.nkv*self.dh,bias=False)
        self.o_proj=nn.Linear(self.nh*self.dh,cfg.d_model,bias=False)
        self.drop=nn.Dropout(cfg.dropout)

    def forward(self, x, rc, rs, start_pos=0, past_kv=None):
        B,T,_=x.shape
        q=self.q_proj(x).view(B,T,self.nh,self.dh).transpose(1,2)
        k=self.k_proj(x).view(B,T,self.nkv,self.dh).transpose(1,2)
        v=self.v_proj(x).view(B,T,self.nkv,self.dh).transpose(1,2)
        q=apply_rope(q,rc,rs,start_pos)
        k=apply_rope(k,rc,rs,start_pos)
        if past_kv is not None:
            pk,pv=past_kv
            k=torch.cat([pk,k],dim=2)
            v=torch.cat([pv,v],dim=2)
        new_kv=(k,v)
        kexp=k.repeat_interleave(self.rep,1) if self.rep>1 else k
        vexp=v.repeat_interleave(self.rep,1) if self.rep>1 else v
        if hasattr(F,"scaled_dot_product_attention"):
            y=F.scaled_dot_product_attention(q,kexp,vexp,is_causal=(past_kv is None),dropout_p=self.drop.p if self.training else 0.0)
        else:
            S=kexp.shape[2]
            s=(q@kexp.transpose(-2,-1))/math.sqrt(self.dh)
            if past_kv is None:
                s.masked_fill_(torch.triu(torch.ones(T,S,device=x.device,dtype=torch.bool),1),float("-inf"))
            y=self.drop(F.softmax(s,-1))@vexp
        return self.o_proj(y.transpose(1,2).contiguous().view(B,T,-1)),new_kv

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj=nn.Linear(cfg.d_model,cfg.d_ff,bias=False)
        self.up_proj=nn.Linear(cfg.d_model,cfg.d_ff,bias=False)
        self.down_proj=nn.Linear(cfg.d_ff,cfg.d_model,bias=False)
        self.drop=nn.Dropout(cfg.dropout)
    def forward(self, x): return self.drop(self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.an=RMSNorm(cfg.d_model,cfg.norm_eps); self.attn=GQAttention(cfg)
        self.fn=RMSNorm(cfg.d_model,cfg.norm_eps); self.ffn=SwiGLUFFN(cfg)

    def forward(self, x, rc, rs, start_pos=0, past_kv=None):
        attn_out,new_kv=self.attn(self.an(x),rc,rs,start_pos,past_kv)
        x=x+attn_out
        return x+self.ffn(self.fn(x)),new_kv

class LLMMaison(nn.Module):
    def __init__(self, cfg=None):
        super().__init__(); self.cfg=cfg=cfg or MDL_CFG
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.d_model); self.drop=nn.Dropout(cfg.dropout)
        self.layers=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm=RMSNorm(cfg.d_model,cfg.norm_eps)
        self.lm_head=None if cfg.tie_embeddings else nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        rc,rs=precompute_rope(cfg.d_head,cfg.max_seq_len,cfg.rope_theta)
        self.register_buffer("rc",rc,persistent=False); self.register_buffer("rs",rs,persistent=False)
        self.use_gradient_checkpointing = False
        self.gradient_checkpointing = False  # alias pour config / hasattr
        self.apply(self._iw); print(f"[MODEL] {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")

    def enable_gradient_checkpointing(self):
        """Active le gradient checkpointing — économise ~40% VRAM, un peu plus lent."""
        self.use_gradient_checkpointing = True
        self.gradient_checkpointing = True
        print("[MODEL] Gradient checkpointing ON — ~40% VRAM en moins")

    def _iw(self, m):
        if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)

    def forward(self, ids, targets=None, start_pos=0, past_kv=None):
        x=self.drop(self.tok_emb(ids))
        new_kv_list=[]
        use_ckpt = self.use_gradient_checkpointing and self.training and past_kv is None
        for i,L in enumerate(self.layers):
            layer_past=past_kv[i] if past_kv is not None else None
            if use_ckpt:
                # Checkpoint ne garde pas KV cache ; OK en entraînement (past_kv toujours None)
                def _run_block(block, hidden, rc, rs):
                    out, _ = block(hidden, rc, rs, 0, None)
                    return out
                x = torch_checkpoint(_run_block, L, x, self.rc, self.rs, use_reentrant=False)
                new_kv_list.append(None)
            else:
                x,new_kv=L(x,self.rc,self.rs,start_pos,layer_past)
                new_kv_list.append(new_kv)
        x=self.norm(x)
        logits=F.linear(x,self.tok_emb.weight) if self.cfg.tie_embeddings else self.lm_head(x)
        out={"logits":logits,"kv_cache":new_kv_list}
        if targets is not None:
            out["loss"]=F.cross_entropy(logits[:,:-1].contiguous().view(-1,self.cfg.vocab_size),
                                        targets[:,1:].contiguous().view(-1),ignore_index=-1)
        return out

    def clear_cache(self):
        """Clear any external KV cache (for stateless generation)."""
        pass

    def count_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if not trainable or p.requires_grad)

    def save_checkpoint(self, path, step=0, optimizer=None, extra=None):
        d={"model_state":self.state_dict(),"step":step,
           "config":{k:v for k,v in self.cfg.__dict__.items() if not k.startswith("_")}}
        if optimizer: d["optimizer_state"]=optimizer.state_dict()
        if extra: d.update(extra)
        torch.save(d,path); print(f"[SAVE] → {path} (step {step})")

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        from config import ModelConfig
        ck=torch.load(path,map_location=device,weights_only=False)
        m=cls(ModelConfig(**ck["config"])).to(device); m.load_state_dict(ck["model_state"]); return m,ck

if __name__=="__main__":
    m=LLMMaison(MDL_CFG); x=torch.randint(0,MDL_CFG.vocab_size,(2,64))
    o=m(x); assert o["logits"].shape==(2,64,MDL_CFG.vocab_size); print("✓ Forward")
    assert "kv_cache" in o and len(o["kv_cache"])==MDL_CFG.n_layers; print("✓ KV cache returned")
    o=m(x,targets=x); o["loss"].backward(); print(f"✓ Loss={o['loss'].item():.4f}, Backward OK")
    # Test incremental decoding with KV cache
    m.eval()
    with torch.no_grad():
        out1=m(x[:,:32])
        cache=out1["kv_cache"]
        out2=m(x[:,32:33],start_pos=32,past_kv=cache)
        assert out2["logits"].shape==(2,1,MDL_CFG.vocab_size); print("✓ KV cache incremental OK")
    print(f"✓ {m.count_params()/1e6:.1f}M params\n✅ OK")
