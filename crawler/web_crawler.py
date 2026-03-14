"""crawler/web_crawler.py — Crawler web généraliste. python -m crawler.web_crawler"""
import json,time,sys; from pathlib import Path; from urllib.parse import urljoin,urlparse; from collections import deque
sys.path.insert(0,str(Path(__file__).parent.parent))
import requests; from bs4 import BeautifulSoup; from config import CRW_CFG,DATA_DIR
try: import trafilatura; HAS_T=True
except: HAS_T=False

class WebCrawler:
    def __init__(self,cfg=None):
        self.cfg=cfg or CRW_CFG; self.s=requests.Session(); self.s.headers["User-Agent"]=self.cfg.user_agent; self.seen=set(); DATA_DIR.mkdir(parents=True,exist_ok=True)
    def extract(self,url):
        try:
            r=self.s.get(url,timeout=self.cfg.timeout); r.raise_for_status()
            t=trafilatura.extract(r.text) if HAS_T else BeautifulSoup(r.text,"html.parser").get_text("\n",strip=True)
            if t and len(t)>=200: return {"url":url,"text":t[:5000],"chars":len(t),"source":"web","domain":urlparse(url).netloc}
        except: pass
        return None
    def crawl_urls(self,urls,out_name="web.jsonl"):
        out=DATA_DIR/out_name; n=0
        with open(out,"w",encoding="utf-8") as f:
            for u in urls:
                if u in self.seen: continue
                self.seen.add(u); d=self.extract(u)
                if d: f.write(json.dumps(d,ensure_ascii=False)+"\n"); n+=1; print(f"[WEB] {n}: {u}")
                time.sleep(self.cfg.delay)
        print(f"[WEB] {n} pages → {out}")
    def crawl_deep(self,seed,max_p=500):
        out=DATA_DIR/"web_deep.jsonl"; q=deque([seed]); n=0
        with open(out,"w",encoding="utf-8") as f:
            while q and n<max_p:
                u=q.popleft()
                if u in self.seen: continue
                self.seen.add(u); d=self.extract(u)
                if d: f.write(json.dumps(d,ensure_ascii=False)+"\n"); n+=1
                try:
                    r=self.s.get(u,timeout=15); soup=BeautifulSoup(r.text,"html.parser"); dom=urlparse(u).netloc
                    for a in soup.find_all("a",href=True)[:20]:
                        h=urljoin(u,a["href"])
                        if urlparse(h).netloc==dom and h not in self.seen: q.append(h)
                except: pass
                time.sleep(self.cfg.delay)
        print(f"[WEB] {n} pages → {out}")

if __name__=="__main__": WebCrawler().crawl_urls(["https://fr.wikipedia.org/wiki/Intelligence_artificielle"])
