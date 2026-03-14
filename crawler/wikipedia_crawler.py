"""crawler/wikipedia_crawler.py — Wikipedia via API ou dump. python -m crawler.wikipedia_crawler"""
import json,time,sys,re,bz2; from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).parent.parent))
import requests; from config import CRW_CFG,DATA_DIR

WIKI_NS = "{http://www.mediawiki.org/xml/export-0.10/}"
SKIP_PREFIXES = ("Discussion:", "Utilisateur:", "Wikipédia:", "Fichier:", "MediaWiki:",
                 "Modèle:", "Aide:", "Catégorie:", "Portail:", "Projet:", "Référence:",
                 "Module:", "Talk:", "User:", "Wikipedia:", "File:", "Template:",
                 "Help:", "Category:", "Portal:", "Draft:")

def clean_wikitext(text):
    """Clean MediaWiki markup. Uses mwparserfromhell if available, else regex."""
    try:
        import mwparserfromhell
        wikicode = mwparserfromhell.parse(text)
        for template in wikicode.filter_templates(): wikicode.remove(template)
        for tag in wikicode.filter_tags(): wikicode.remove(tag)
        for comment in wikicode.filter_comments(): wikicode.remove(comment)
        text = wikicode.strip_code()
    except ImportError:
        text = re.sub(r'\{\{[^}]*\}\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{\|[^}]*\|\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\[\[Catégorie:[^\]]*\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[Fichier:[^\]]*\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[File:[^\]]*\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[Image:[^\]]*\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
        text = re.sub(r"'''?([^']*?)'''?", r'\1', text)
        text = re.sub(r'==+\s*([^=]+?)\s*==+', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\{\{[^}]*$', '', text)
        text = re.sub(r'^[^{]*\}\}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def download_and_parse_dump(lang="fr", max_articles=None, out_name=None):
    """
    Download and parse Wikipedia XML dump in streaming mode.
    
    Args:
        lang: Wikipedia language code (default: "fr")
        max_articles: Max articles to extract (None = all)
        out_name: Output filename (default: wikipedia_dump_{lang}.jsonl)
    """
    dump_url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
    out_file = DATA_DIR / (out_name or f"wikipedia_dump_{lang}.jsonl")
    cache_file = DATA_DIR / f"{lang}wiki-latest-pages-articles.xml.bz2"
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not cache_file.exists():
        print(f"[DUMP] Downloading {dump_url}...")
        print(f"[DUMP] This may take a while (~2.5GB for frwiki)")
        try:
            from tqdm import tqdm
            r = requests.get(dump_url, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(cache_file, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except ImportError:
            r = requests.get(dump_url, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(cache_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * 1024 * 1024) < 8192:
                        print(f"[DUMP] Downloaded {downloaded / 1e9:.2f}GB / {total / 1e9:.2f}GB")
        print(f"[DUMP] Download complete: {cache_file}")
    else:
        print(f"[DUMP] Using cached dump: {cache_file}")
    
    print(f"[DUMP] Parsing {cache_file} → {out_file}")
    count = 0
    chars = 0
    skipped = 0
    
    with bz2.open(cache_file, 'rt', encoding='utf-8') as bzf, \
         open(out_file, 'w', encoding='utf-8') as out:
        
        context = ET.iterparse(bzf, events=('end',))
        title = None
        
        for event, elem in context:
            tag = elem.tag.replace(WIKI_NS, '')
            
            if tag == 'title':
                title = elem.text
            elif tag == 'redirect':
                title = None
            elif tag == 'text' and title:
                if any(title.startswith(p) for p in SKIP_PREFIXES):
                    skipped += 1
                    title = None
                    elem.clear()
                    continue
                
                raw_text = elem.text or ''
                if '#REDIRECT' in raw_text.upper() or '#REDIRECTION' in raw_text.upper():
                    skipped += 1
                    title = None
                    elem.clear()
                    continue
                
                text = clean_wikitext(raw_text)
                
                if len(text) >= CRW_CFG.wiki_min_chars:
                    doc = {
                        "title": title,
                        "text": text,
                        "source": "wikipedia_dump",
                        "lang": lang,
                        "chars": len(text)
                    }
                    out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    count += 1
                    chars += len(text)
                    
                    if count % 10000 == 0:
                        print(f"[DUMP] {count:,} articles, {chars/1e6:.1f}M chars, {skipped:,} skipped")
                    
                    if max_articles and count >= max_articles:
                        break
                
                title = None
            
            elem.clear()
    
    print(f"[DUMP] Done: {count:,} articles, {chars/1e6:.1f}M chars")
    print(f"[DUMP] Output: {out_file}")
    return out_file

class WikiCrawler:
    def __init__(self, cfg=None):
        self.cfg=cfg or CRW_CFG; self.s=requests.Session(); self.s.headers["User-Agent"]=self.cfg.user_agent
        self.api=f"https://{self.cfg.wiki_lang}.wikipedia.org/w/api.php"; DATA_DIR.mkdir(parents=True,exist_ok=True)
    def random_titles(self,n=500):
        t=[]
        while len(t)<n:
            r=self.s.get(self.api,params={"action":"query","list":"random","rnnamespace":0,"rnlimit":min(50,n-len(t)),"format":"json"},timeout=30)
            for p in r.json().get("query",{}).get("random",[]): t.append(p["title"])
            time.sleep(self.cfg.delay)
        return t
    def get_article(self, title):
        r=self.s.get(self.api,params={"action":"query","titles":title,"prop":"extracts","explaintext":True,"format":"json"},timeout=30)
        for pid,p in r.json().get("query",{}).get("pages",{}).items():
            if pid!="-1":
                t=p.get("extract","")
                if len(t)>=self.cfg.wiki_min_chars: return {"title":p.get("title",title),"text":t,"source":"wikipedia","lang":self.cfg.wiki_lang,"chars":len(t)}
        return None
    def crawl(self, n=1000):
        out=DATA_DIR/f"wikipedia_{self.cfg.wiki_lang}.jsonl"; print(f"[WIKI] Crawl {n} articles → {out}")
        titles=self.random_titles(n); ct=ch=0
        with open(out,"w",encoding="utf-8") as f:
            for i,t in enumerate(titles):
                a=self.get_article(t)
                if a: f.write(json.dumps(a,ensure_ascii=False)+"\n"); ct+=1; ch+=a["chars"]
                if (i+1)%100==0: print(f"[WIKI] {i+1}/{n} — {ct} articles, {ch/1e6:.1f}M chars")
                time.sleep(self.cfg.delay)
        print(f"[WIKI] {ct} articles, {ch/1e6:.1f}M chars")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(description="Wikipedia crawler (API or dump)")
    p.add_argument("--mode", choices=["api", "dump"], default="api", help="Crawl mode")
    p.add_argument("--n", type=int, default=1000, help="Number of articles (API mode)")
    p.add_argument("--lang", type=str, default="fr", help="Wikipedia language")
    p.add_argument("--max", type=int, default=None, help="Max articles (dump mode)")
    args = p.parse_args()
    
    if args.mode == "api":
        WikiCrawler().crawl(args.n)
    else:
        download_and_parse_dump(lang=args.lang, max_articles=args.max)
