"""crawler/reddit_crawler.py — Reddit via old.reddit.com JSON. python -m crawler.reddit_crawler"""
import json,time,sys; from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import requests; from config import CRW_CFG,DATA_DIR

class RedditCrawler:
    def __init__(self,cfg=None):
        self.cfg=cfg or CRW_CFG; self.s=requests.Session(); self.s.headers["User-Agent"]=self.cfg.user_agent; DATA_DIR.mkdir(parents=True,exist_ok=True)
    def get_posts(self,sub,sort="top",after=None):
        p={"limit":100,"t":"all","raw_json":1}; url=f"https://old.reddit.com/r/{sub}/{sort}.json"
        if after: p["after"]=after
        try:
            r=self.s.get(url,params=p,timeout=30)
            if r.status_code==429: time.sleep(60); return self.get_posts(sub,sort,after)
            r.raise_for_status(); d=r.json(); posts=[]
            for c in d.get("data",{}).get("children",[]):
                x=c.get("data",{})
                if x.get("score",0)>=self.cfg.reddit_min_score and x.get("is_self") and x.get("selftext"):
                    posts.append({"title":x["title"],"text":x["selftext"],"score":x["score"],"subreddit":sub,"id":x["id"],"source":"reddit"})
            return posts,d.get("data",{}).get("after")
        except Exception as e: print(f"[REDDIT] {e}"); return [],None
    def crawl(self,max_per=200):
        out=DATA_DIR/"reddit.jsonl"; tp=tc=0
        with open(out,"w",encoding="utf-8") as f:
            for sub in self.cfg.reddit_subs:
                af=None; sc=0
                while sc<max_per:
                    ps,af=self.get_posts(sub,after=af)
                    if not ps: break
                    for p in ps: f.write(json.dumps(p,ensure_ascii=False)+"\n"); sc+=1; tp+=1; time.sleep(self.cfg.delay)
                    if not af: break
                print(f"[REDDIT] r/{sub}: {sc}")
        print(f"[REDDIT] {tp} posts → {out}")

if __name__=="__main__": RedditCrawler().crawl(200)
