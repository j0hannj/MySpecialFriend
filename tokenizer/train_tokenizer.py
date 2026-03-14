"""
tokenizer/train_tokenizer.py — Entraîne le BPE puis tokenize le corpus.
Usage: python -m tokenizer.train_tokenizer
"""
import json,sys,numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).parent.parent))
from config import TOK_CFG,DATA_DIR,TOKENIZER_DIR,PROCESSED_DIR
from tokenizer.bpe import BPETokenizer

def load_corpus(d, max_docs=None):
    from crawler.cleaner import normalize_after_json_load
    texts=[]
    for f in sorted(d.glob("*.jsonl"))+sorted(d.glob("*.txt")):
        if f.suffix==".jsonl":
            for line in open(f,"r",encoding="utf-8",errors="replace"):
                try:
                    t=json.loads(line).get("text","")
                except json.JSONDecodeError:
                    continue
                t=normalize_after_json_load(t)
                if len(t)>=100: texts.append(t)
                if max_docs and len(texts)>=max_docs: return texts
        elif f.suffix==".txt":
            t=f.read_text("utf-8")
            if len(t)>=100: texts.append(t)
    return texts

if __name__=="__main__":
    print("="*50+"\nTOKENIZER BPE — ENTRAÎNEMENT\n"+"="*50)
    texts=load_corpus(DATA_DIR)
    if not texts:
        print(f"[!] Pas de données dans {DATA_DIR}")
        print("    Lance d'abord un crawler, ou crée un fichier test:")
        print(f"    echo 'Bonjour le monde...' > {DATA_DIR}/test.txt")
        sys.exit(1)
    print(f"{len(texts)} documents chargés")
    tok=BPETokenizer(TOK_CFG.vocab_size, TOK_CFG.special_tokens)
    tok.train(texts)
    tok.save(str(TOKENIZER_DIR))
    # Tokenize
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    all_ids=[]
    for t in tqdm(texts,desc="Tokenization"):
        all_ids.extend(tok.encode(t,add_special=True))
    arr=np.array(all_ids,dtype=np.uint16)
    out=PROCESSED_DIR/"train.bin"; arr.tofile(str(out))
    print(f"[TOK] {len(all_ids):,} tokens → {out} ({out.stat().st_size/1e6:.1f}MB)")
    with open(PROCESSED_DIR/"stats.json","w") as f:
        json.dump({"n_tokens":len(all_ids),"n_docs":len(texts),"vocab_size":tok.size},f,indent=2)
    # Test roundtrip
    test="Salut Johann, comment ça va ?"
    ids=tok.encode(test); back=tok.decode(ids)
    print(f"\nTest: '{test}' → {len(ids)} tokens → '{back}'")
    assert back==test, f"ROUNDTRIP FAIL: '{back}'"
    print("✓ Roundtrip OK"); tok.inspect(10)
