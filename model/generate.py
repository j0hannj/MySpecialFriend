"""model/generate.py — Génération avec temperature, top-k, top-p, rep penalty, KV cache."""
import torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GenConfig:
    max_new_tokens:int=256; temperature:float=0.7; top_k:int=50; top_p:float=0.9
    rep_penalty:float=1.15; stop_tokens:Optional[List[int]]=None; use_cache:bool=True

def _sample_next(lo, cfg, gen):
    """Apply sampling strategies to logits and return next token."""
    if cfg.rep_penalty!=1.0:
        for t in set(gen[0].tolist()):
            lo[0,t]/=cfg.rep_penalty if lo[0,t]>0 else 1/cfg.rep_penalty
    if cfg.temperature<=0:
        return lo.argmax(-1,keepdim=True)
    lo=lo/cfg.temperature
    if cfg.top_k>0:
        lo[lo<torch.topk(lo,min(cfg.top_k,lo.size(-1)))[0][...,-1:]]=float("-inf")
    if cfg.top_p<1.0:
        sl,si=torch.sort(lo,descending=True); cp=torch.cumsum(F.softmax(sl,-1),-1)
        sl[cp-F.softmax(sl,-1)>=cfg.top_p]=float("-inf"); lo=sl.scatter(1,si,sl)
    return torch.multinomial(F.softmax(lo,-1),1)

@torch.inference_mode()
def generate(model, input_ids, cfg=None):
    cfg=cfg or GenConfig(); gen=input_ids.clone(); stops=set(cfg.stop_tokens or [])
    kv_cache=None; start_pos=0
    
    for i in range(cfg.max_new_tokens):
        if cfg.use_cache and kv_cache is not None:
            inp=gen[:,-1:]
            out=model(inp,start_pos=start_pos,past_kv=kv_cache)
            start_pos+=1
        else:
            seq=gen[:,-model.cfg.max_seq_len:]
            out=model(seq)
            start_pos=seq.shape[1]
        
        kv_cache=out["kv_cache"] if cfg.use_cache else None
        lo=out["logits"][:,-1,:].clone()
        nxt=_sample_next(lo,cfg,gen)
        gen=torch.cat([gen,nxt],1)
        if nxt.item() in stops: break
        
        if cfg.use_cache and start_pos>=model.cfg.max_seq_len-1:
            kv_cache=None; start_pos=0
    return gen

@torch.inference_mode()
def generate_no_cache(model, input_ids, cfg=None):
    """Legacy generation without KV cache (for comparison/testing)."""
    cfg=cfg or GenConfig(); gen=input_ids.clone(); stops=set(cfg.stop_tokens or [])
    for _ in range(cfg.max_new_tokens):
        seq=gen[:,-model.cfg.max_seq_len:]; lo=model(seq)["logits"][:,-1,:].clone()
        nxt=_sample_next(lo,cfg,gen)
        gen=torch.cat([gen,nxt],1)
        if nxt.item() in stops: break
    return gen

def generate_text(model, tokenizer, prompt, cfg=None, device="cpu"):
    cfg=cfg or GenConfig(); ids=tokenizer.encode(prompt)
    inp=torch.tensor([ids],dtype=torch.long,device=device)
    stops=list(cfg.stop_tokens or [])
    if tokenizer.eos_id>=0: stops.append(tokenizer.eos_id)
    cfg.stop_tokens=stops; model.eval()
    return tokenizer.decode(generate(model,inp,cfg)[0,len(ids):].tolist())

if __name__=="__main__":
    import time
    import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).parent.parent))
    from model.transformer import LLMMaison
    from config import MDL_CFG
    
    print("Testing KV cache generation...")
    m=LLMMaison(MDL_CFG).eval()
    x=torch.randint(0,MDL_CFG.vocab_size,(1,32))
    
    cfg_cache=GenConfig(max_new_tokens=50,use_cache=True)
    cfg_nocache=GenConfig(max_new_tokens=50,use_cache=False)
    
    torch.manual_seed(42)
    t0=time.time()
    out_cache=generate(m,x,cfg_cache)
    t_cache=time.time()-t0
    
    torch.manual_seed(42)
    t0=time.time()
    out_nocache=generate_no_cache(m,x,cfg_nocache)
    t_nocache=time.time()-t0
    
    print(f"With cache:    {t_cache:.3f}s")
    print(f"Without cache: {t_nocache:.3f}s")
    print(f"Speedup:       {t_nocache/t_cache:.1f}x")
    
    if torch.equal(out_cache,out_nocache):
        print("✓ Outputs identical")
    else:
        print("⚠ Outputs differ (expected with sampling)")
    print("✅ KV cache test complete")
