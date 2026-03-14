"""training/dataset.py — Datasets mmap (pretrain) + conversations (finetune)."""
import json,numpy as np,torch; from torch.utils.data import Dataset; from pathlib import Path
import sys; sys.path.insert(0,str(Path(__file__).parent.parent)); from config import PROCESSED_DIR,TRN_CFG

class PretrainDataset(Dataset):
    def __init__(self,path=None,seq_len=None):
        self.sl=seq_len or TRN_CFG.seq_len; p=path or str(PROCESSED_DIR/"train.bin")
        if not Path(p).exists(): raise FileNotFoundError(f"{p}")
        self.data=np.memmap(p,dtype=np.uint16,mode="r"); self.n=len(self.data)//(self.sl+1)
        print(f"[DATA] {len(self.data):,} tokens, {self.n:,} chunks")
    def __len__(self): return self.n
    def __getitem__(self,i):
        s=i*(self.sl+1); c=self.data[s:s+self.sl+1].astype(np.int64)
        return torch.from_numpy(c[:-1]),torch.from_numpy(c[1:])

class ConvDataset(Dataset):
    def __init__(self,convs,tok,ml=1024):
        self.samples=[]
        for conv in convs:
            ids=[tok.bos_id]
            for m in conv:
                r=tok.user_id if m["role"]=="user" else tok.asst_id
                if r and r>=0: ids.append(r)
                ids.extend(tok.encode(m["content"]))
            ids.append(tok.eos_id)
            if 10<len(ids)<=ml: self.samples.append(ids)
        print(f"[DATA] {len(self.samples)} conversations")
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ids=self.samples[i]; return torch.tensor(ids[:-1],dtype=torch.long),torch.tensor(ids[1:],dtype=torch.long)

def collate_pad(batch,pad_id=256):
    xs,ys=zip(*batch); ml=max(len(x) for x in xs)
    px=torch.full((len(xs),ml),pad_id,dtype=torch.long); py=torch.full((len(ys),ml),-1,dtype=torch.long)
    for i,(x,y) in enumerate(zip(xs,ys)): px[i,:len(x)]=x; py[i,:len(y)]=y
    return px,py
