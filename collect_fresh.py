"""
Collect a small FRESH labelled URL set for the infra-feature experiments.

Phishing : OpenPhish community feed (a plain text list). This script only downloads the
           list. It NEVER visits the phishing URLs. Check OpenPhish's terms of use first, and
           never open these URLs in your normal browser.
Legit    : random domains from the Tranco top list. Each domain's homepage is fetched (legitimate
           sites only) and ONE URL is sampled from {homepage + same-site links}, so legitimate
           URLs also have paths and don't all follow a fixed 'www.' pattern.

Labels follow PhiUSIIL: label 1 = legitimate, 0 = phishing (phish = 1 - label).
Each row also gets 'source' and 'collected_at' (needed later for temporal splits).

Usage (from the phish folder):
    python collect_fresh.py --n 300
    python collect_fresh.py --n 300 --phish-file feed.txt --tranco-csv top-1m.csv   # offline fallbacks
"""
import argparse, io, os, random, re, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
import tldextract

TLD = tldextract.TLDExtract(suffix_list_urls=())
UA = {"User-Agent": "Mozilla/5.0 (PhishGuard-Z research)"}
HREF = re.compile(r'href\s*=\s*["\']([^"\'#>\s]+)', re.I)


def get_phish(path):
    if path:
        text = open(path, encoding="utf-8", errors="ignore").read()
    else:
        text = requests.get("https://openphish.com/feed.txt", headers=UA, timeout=30).text
    urls = [u.strip() for u in text.splitlines() if u.strip().lower().startswith(("http://", "https://"))]
    return list(dict.fromkeys(urls))


def get_tranco(path, top):
    if path:
        df = pd.read_csv(path, header=None, names=["rank", "domain"])
    else:
        raw = requests.get("https://tranco-list.eu/top-1m.csv.zip", headers=UA, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=["rank", "domain"])
    return df[df["rank"] <= top]["domain"].astype(str).tolist()


def pick_legit(domain, seed):
    """One URL from the legitimate site: its homepage or a same-site link (login pages included)."""
    rng = random.Random(f"{seed}-{domain}")
    home = f"https://{domain}/"
    cands = [home]
    try:
        r = requests.get(home, headers=UA, timeout=6)
        reg = TLD(urlsplit(r.url).hostname or "").registered_domain
        for h in HREF.findall(r.text[:300000])[:2000]:
            u = urljoin(r.url, h)
            sp = urlsplit(u)
            if sp.scheme in ("http", "https") and TLD(sp.hostname or "").registered_domain == reg:
                cands.append(u)
    except Exception:
        pass
    return rng.choice(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="URLs per class")
    ap.add_argument("--top", type=int, default=20000, help="sample legit domains from Tranco rank <= this")
    ap.add_argument("--out", default="data/raw/fresh_urls.csv")
    ap.add_argument("--phish-file", default="")
    ap.add_argument("--tranco-csv", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    phish = get_phish(args.phish_file)
    print(f"phishing feed: {len(phish)} unique URLs")
    phish = rng.sample(phish, min(args.n, len(phish)))

    domains = get_tranco(args.tranco_csv, args.top)
    domains = rng.sample(domains, min(int(args.n * 1.5), len(domains)))
    with ThreadPoolExecutor(args.workers) as ex:
        legit = list(ex.map(lambda d: pick_legit(d, args.seed), domains))
    legit = list(dict.fromkeys(legit))[:args.n]
    print(f"legitimate URLs: {len(legit)}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = pd.DataFrame(
        [(u, 0, 1, "openphish", today) for u in phish] + [(u, 1, 0, "tranco", today) for u in legit],
        columns=["URL", "label", "phish", "source", "collected_at"])
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    sp = df.URL.map(lambda u: int(len(urlsplit(u).path.strip("/")) > 0 or bool(urlsplit(u).query)))
    print("share of URLs with a path/query by class (0 = phishing, 1 = legit):")
    print(sp.groupby(df.label).mean().round(3).to_string())
    print("saved:", args.out, df.shape)


if __name__ == "__main__":
    main()