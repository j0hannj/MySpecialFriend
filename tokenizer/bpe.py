"""
tokenizer/bpe.py — BPE from scratch. 256 bytes → merges → vocab_size.
Auditable: self.merges stocke chaque fusion, save() produit du JSON lisible.
"""
import json, re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict

class BPETokenizer:
    SPLIT = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+", re.UNICODE)

    def __init__(self, vocab_size=32000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<PAD>","<UNK>","<BOS>","<EOS>","<SEP>","<USER>","<ASST>","<SYS>"]
        self.merges: List[Tuple[int,int]] = []
        self.vocab: Dict[int,bytes] = {}
        self._trained = False
        self._init_vocab()

    @property
    def n_base(self): return 256 + len(self.special_tokens)
    @property
    def size(self): return len(self.vocab)
    def special_id(self, name):
        try: return 256 + self.special_tokens.index(name)
        except ValueError: return -1
    @property
    def pad_id(self): return self.special_id("<PAD>")
    @property
    def bos_id(self): return self.special_id("<BOS>")
    @property
    def eos_id(self): return self.special_id("<EOS>")
    @property
    def user_id(self): return self.special_id("<USER>")
    @property
    def asst_id(self): return self.special_id("<ASST>")

    def _init_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate(self.special_tokens):
            self.vocab[256+i] = t.encode("utf-8")

    def train(self, texts, verbose=True):
        if verbose: print(f"[BPE] Training — target {self.vocab_size} tokens, {len(texts)} docs")
        wf = Counter()
        for t in texts:
            for w in self.SPLIT.findall(t): wf[w] += 1
        splits = {}
        for w, f in wf.items():
            k = tuple(w.encode("utf-8")); splits[k] = splits.get(k,0) + f
        n_merges = self.vocab_size - self.n_base
        for step in range(n_merges):
            pairs = Counter()
            for seq, f in splits.items():
                for i in range(len(seq)-1): pairs[(seq[i],seq[i+1])] += f
            if not pairs: break
            best = max(pairs, key=pairs.get)
            nid = self.n_base + step
            ns = {}
            for seq, f in splits.items():
                r = []; i = 0
                while i < len(seq):
                    if i < len(seq)-1 and seq[i]==best[0] and seq[i+1]==best[1]: r.append(nid); i+=2
                    else: r.append(seq[i]); i+=1
                k = tuple(r); ns[k] = ns.get(k,0) + f
            splits = ns
            self.merges.append(best)
            self.vocab[nid] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose and (step+1)%1000==0:
                print(f"[BPE] {step+1}/{n_merges} — '{self.vocab[nid].decode('utf-8',errors='replace')}'")
        self._trained = True
        if verbose: print(f"[BPE] Done — {len(self.vocab)} tokens")

    def encode(self, text, add_special=False):
        assert self._trained, "Train or load first!"
        # Build merge lookup once
        if not hasattr(self, '_merge_map'):
            self._merge_map = {p: self.n_base+i for i, p in enumerate(self.merges)}
        words = self.SPLIT.findall(text)
        all_ids = []
        for w in words:
            ids = list(w.encode("utf-8"))
            for pair in self.merges:
                if pair not in self._merge_map: continue
                nid = self._merge_map[pair]
                new = []; i = 0
                while i < len(ids):
                    if i < len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]: new.append(nid); i+=2
                    else: new.append(ids[i]); i+=1
                ids = new
            all_ids.extend(ids)
        if add_special: all_ids = [self.bos_id] + all_ids + [self.eos_id]
        return all_ids

    def decode(self, ids):
        return b"".join(self.vocab.get(i, b"?") for i in ids if i in self.vocab and i not in
                        {self.special_id(s) for s in self.special_tokens}).decode("utf-8", errors="replace")

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        d = {"vocab_size":self.vocab_size,"special_tokens":self.special_tokens,"merges":self.merges,
             "vocab_readable":{str(k):v.decode("utf-8",errors="replace") for k,v in self.vocab.items() if k>=self.n_base}}
        with open(p/"tokenizer.json","w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)
        print(f"[BPE] Saved → {p/'tokenizer.json'}")

    @classmethod
    def load(cls, path):
        with open(Path(path)/"tokenizer.json","r",encoding="utf-8") as f: d=json.load(f)
        t = cls(d["vocab_size"], d["special_tokens"])
        t.merges = [tuple(m) for m in d["merges"]]
        for i,(a,b) in enumerate(t.merges): t.vocab[t.n_base+i] = t.vocab[a]+t.vocab[b]
        t._trained = True
        t._merge_map = {p: t.n_base+i for i, p in enumerate(t.merges)}
        print(f"[BPE] Loaded — {len(t.vocab)} tokens"); return t

    def inspect(self, n=10):
        print(f"\n[AUDIT] {len(self.vocab)} tokens, {len(self.merges)} merges")
        for i in range(max(0,len(self.merges)-n), len(self.merges)):
            tid=self.n_base+i; print(f"  [{tid}] '{self.vocab[tid].decode('utf-8',errors='replace')}' ← {self.merges[i]}")
