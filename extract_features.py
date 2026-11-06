"""
PhishGuard-Z feature extractor.

LEXICAL features (offline, fast): URL length, path depth, entropy, suspicious tokens,
    TLD characteristics, randomness score.  Computed on the NORMALIZED url
    (scheme and leading www. removed) so the dataset's 'https://www.' tell cannot leak in.
INFRA features (--infra, needs internet): DNS records, IP/ASN, domain age, registrar.
    These do NOT connect to the target website (DNS, RDAP and Team Cymru lookups only).
REDIRECT features (--redirects): number of redirects. This DOES contact the URL,
    so run it only inside an isolated VM/Docker container, never on a machine you care about.

Usage (from the phish folder):
    pip install tldextract dnspython requests
    python extract_features.py --in data/processed/train.csv --out data/features/train_lex.csv
    python extract_features.py --in data/raw/fresh_urls.csv --out data/features/fresh.csv --infra --limit 500
    python extract_features.py --in data/raw/fresh_urls.csv --out data/features/fresh.csv --infra --redirects --limit 200

Input CSV needs a URL column (default 'URL'). A 'label' / 'phish' / 'reg_domain' column is carried over if present.
The output is appended chunk by chunk and the run can be resumed (already-done URLs are skipped).

WARNING (time bias): infra features are measured TODAY. Old phishing URLs are mostly dead
(no DNS, no WHOIS) while popular legitimate sites always resolve, so on an old dataset
'does not resolve' and 'domain age' become label proxies. Use fresh URLs collected the same
day (PhishTank / OpenPhish / URLhaus + a legitimate list) for infra features.
"""
import argparse, math, re, os, time, socket, ipaddress, threading, warnings
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urljoin

import numpy as np
import pandas as pd
import tldextract

warnings.filterwarnings("ignore")
_TLD = tldextract.TLDExtract(suffix_list_urls=())      # bundled suffix list, no network

# ---- hand-made starting lists: refine them with statistics from your TRAIN split ----
SUSPICIOUS_TOKENS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "security", "account",
    "update", "confirm", "banking", "password", "passwd", "credential", "wallet", "support",
    "billing", "invoice", "payment", "webscr", "authenticate", "recover", "unlock", "suspend",
    "alert", "refund", "helpdesk", "claim", "prize",
]
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google", "gmail", "facebook",
    "instagram", "whatsapp", "amazon", "netflix", "ebay", "dropbox", "linkedin", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "hdfc", "icici", "paytm", "dhl", "fedex", "usps",
    "coinbase", "binance", "metamask",
]
RISKY_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "icu", "click", "link", "work", "zip", "mov", "buzz",
    "cyou", "cfd", "sbs", "rest", "monster", "quest", "support", "country", "stream", "download",
    "loan", "men", "bid", "win", "party", "cam", "fit",
}
VOWELS, CONSONANTS = set("aeiou"), set("bcdfghjklmnpqrstvwxyz")


# ====================================================================== lexical
def norm_url(u):
    u = str(u).strip().lower()
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)     # drop scheme
    u = re.sub(r"^www\d*\.", "", u)                 # drop leading www.
    return u


def shannon(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in Counter(s).values())


def max_run(s, chars):
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch in chars else 0
        best = max(best, cur)
    return best


def lexical(raw_url):
    raw_url = str(raw_url)
    u = norm_url(raw_url)
    host = path = query = ""
    netloc, port = "", None
    try:
        sp = urlsplit("//" + u)
        netloc, host, path, query = sp.netloc, (sp.hostname or ""), sp.path, sp.query
        try:
            port = sp.port
        except ValueError:
            port = None
    except ValueError:
        host = re.split(r"[/?#]", u, maxsplit=1)[0]
        netloc = host

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    ext = _TLD(host)
    label = host if is_ip else ext.domain               # registered-domain label, e.g. 'paypal-login'
    suffix = "" if is_ip else ext.suffix
    tld = suffix.split(".")[-1] if suffix else ""
    reg = host if is_ip else (f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain)
    sub_path = ext.subdomain + "/" + path

    letters = [c for c in label if c.isalpha()]
    vowel_ratio = sum(c in VOWELS for c in letters) / len(letters) if letters else 0.0
    cons_run = max_run(label, CONSONANTS)
    label_digit_ratio = sum(c.isdigit() for c in label) / len(label) if label else 0.0

    # heuristic randomness score in [0,1]: few vowels, long consonant runs, many digits.
    # The components are also saved separately so a model can learn its own mix.
    randomness = (
        (1 - min(vowel_ratio / 0.35, 1.0)) + min(max(cons_run - 3, 0) / 4, 1.0) + min(label_digit_ratio * 2, 1.0)
    ) / 3 if label else 0.0

    host_path = host + path
    segs = [s for s in path.split("/") if s]
    n_digits = sum(c.isdigit() for c in u)

    feats = {
        # --- URL length & path depth
        "url_length": len(u),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": len(segs),
        "num_params": len(query.split("&")) if query else 0,
        "subdomain_depth": len([s for s in ext.subdomain.split(".") if s]),
        # --- structure / obfuscation
        "num_dots_host": host.count("."),
        "num_hyphens_host": host.count("-"),
        "digit_ratio_url": n_digits / len(u) if u else 0.0,
        "is_ip_host": is_ip,
        "has_at_symbol": int("@" in netloc),
        "has_nonstd_port": int(port not in (None, 80, 443)),
        "num_pct_encoded": len(re.findall(r"%[0-9a-f]{2}", u)),
        "has_punycode": int("xn--" in host),
        "non_ascii_chars": sum(ord(c) > 127 for c in raw_url),
        # --- entropy
        "url_entropy": shannon(u),
        "host_entropy": shannon(host),
        "path_entropy": shannon(path),
        # --- suspicious tokens
        "susp_token_count": sum(t in host_path + query for t in SUSPICIOUS_TOKENS),
        "susp_in_host": sum(t in host for t in SUSPICIOUS_TOKENS),
        "susp_in_path": sum(t in path + query for t in SUSPICIOUS_TOKENS),
        "brand_in_subdomain_or_path": int(any(b in sub_path and b not in label for b in BRANDS)),
        "brand_inside_label": int(any(b in label and b != label for b in BRANDS)),
        # --- TLD characteristics
        "tld": tld,
        "tld_length": len(tld),
        "tld_is_country": int(len(tld) == 2 and tld.isalpha()),
        "tld_multi_part": int("." in suffix),
        "tld_in_risky_list": int(tld in RISKY_TLDS),
        # --- randomness of the registered-domain label
        "label_length": len(label),
        "label_entropy": shannon(label),
        "label_vowel_ratio": vowel_ratio,
        "label_max_consonant_run": cons_run,
        "label_digit_ratio": label_digit_ratio,
        "randomness_score": randomness,
    }
    return feats, host, reg, tld, is_ip


# ====================================================================== infra (no contact with the site)
try:
    import dns.resolver
    import requests
    _res = dns.resolver.Resolver(configure=False)
    _res.nameservers = ["1.1.1.1", "8.8.8.8"]
    _res.timeout, _res.lifetime = 2.0, 4.0
except ImportError:
    _res = None

DNS_COLS = dict(dns_resolves=np.nan, a_count=np.nan, aaaa_count=np.nan, dns_ttl_min=np.nan,
                ns_count=np.nan, mx_count=np.nan, txt_count=np.nan, has_spf=np.nan)
IP_COLS = dict(asn=np.nan, asn_cc="")
RDAP_COLS = dict(rdap_ok=0, domain_age_days=np.nan, days_to_expiry=np.nan, days_since_update=np.nan, registrar="")
REDIR_COLS = dict(redirect_ok=0, redirect_count=np.nan, redirect_final_domain_differs=np.nan)


def _q(name, rtype):
    if _res is None or not name:
        return None
    try:
        return _res.resolve(name, rtype, raise_on_no_answer=False)
    except Exception:
        return None


def _n(ans):
    return len(ans.rrset) if ans is not None and ans.rrset else 0


@lru_cache(maxsize=500000)
def dns_features(host, reg):
    out = dict(DNS_COLS)
    out["_ip"] = ""
    a, aaaa = _q(host, "A"), _q(host, "AAAA")
    if a is None and aaaa is None:
        out["dns_resolves"] = 0                      # NXDOMAIN / timeout
    else:
        out.update(a_count=_n(a), aaaa_count=_n(aaaa), dns_resolves=int(_n(a) + _n(aaaa) > 0))
        if _n(a):
            out["dns_ttl_min"] = a.rrset.ttl
            out["_ip"] = a.rrset[0].address
    if reg:
        for key, rtype in (("ns_count", "NS"), ("mx_count", "MX")):
            r = _q(reg, rtype)
            out[key] = _n(r) if r is not None else np.nan
        r = _q(reg, "TXT")
        if r is not None:
            out["txt_count"] = _n(r)
            out["has_spf"] = int(any(b"v=spf1" in b"".join(rd.strings) for rd in r.rrset)) if _n(r) else 0
    return out


@lru_cache(maxsize=500000)
def asn_info(ip):
    """IP -> ASN / country via Team Cymru's DNS service (free, no API key, IPv4 only)."""
    out = dict(IP_COLS)
    try:
        ipaddress.IPv4Address(ip)
        r = _q(".".join(reversed(ip.split("."))) + ".origin.asn.cymru.com", "TXT")
        if r is not None and r.rrset:
            parts = [p.strip() for p in b"".join(r.rrset[0].strings).decode().split("|")]
            out["asn"] = int(parts[0].split()[0])
            out["asn_cc"] = parts[2]
    except Exception:
        pass
    return out


_boot, _boot_lock, _sems = {}, threading.Lock(), {}


def _rdap_base(tld):
    with _boot_lock:
        if not _boot:
            try:
                for tlds, urls in requests.get("https://data.iana.org/rdap/dns.json", timeout=10).json()["services"]:
                    for t in tlds:
                        _boot[t.lower()] = urls[0]
            except Exception:
                _boot["__failed__"] = ""
        if _boot.get("__failed__") is not None and len(_boot) == 1:
            return None
        return _boot.get(tld.lower())


def _dt(s):
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


@lru_cache(maxsize=500000)
def rdap_features(reg, tld):
    """Domain age / registrar through RDAP (the modern WHOIS). Many ccTLDs have no RDAP -> NaN."""
    out = dict(RDAP_COLS)
    base = _rdap_base(tld) if (reg and tld and _res is not None) else None
    if not base:
        return out
    with _boot_lock:
        sem = _sems.setdefault(base, threading.Semaphore(3))      # be polite to each registry
    j = None
    for attempt in range(2):
        with sem:
            try:
                r = requests.get(base.rstrip("/") + "/domain/" + reg, timeout=8,
                                 headers={"Accept": "application/rdap+json"})
            except Exception:
                return out
        if r.status_code == 429:
            time.sleep(2 + 3 * attempt)
            continue
        if r.status_code == 200:
            try:
                j = r.json()
            except Exception:
                return out
        break
    if not j:
        return out
    now = datetime.now(timezone.utc)
    ev = {e.get("eventAction"): _dt(e.get("eventDate")) for e in j.get("events", [])}
    out["rdap_ok"] = 1
    if ev.get("registration"):
        out["domain_age_days"] = (now - ev["registration"]).days
    if ev.get("expiration"):
        out["days_to_expiry"] = (ev["expiration"] - now).days
    if ev.get("last changed"):
        out["days_since_update"] = (now - ev["last changed"]).days
    try:
        for ent in j.get("entities", []):
            if "registrar" in ent.get("roles", []):
                for item in ent["vcardArray"][1]:
                    if item[0] == "fn":
                        out["registrar"] = item[3]
    except Exception:
        pass
    return out


# ====================================================================== redirects (CONTACTS the URL)
def _public_host(host):
    try:
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(host, None))
    except Exception:
        return False


def redirect_features(raw_url, max_hops=10):
    """Follow redirects hop by hop. Headers only, body is never read; non-public IPs are refused."""
    out = dict(REDIR_COLS)
    cur = raw_url if re.match(r"^[a-z][a-z0-9+.-]*://", raw_url, re.I) else "http://" + raw_url
    start_reg = _TLD(urlsplit(cur).hostname or "").registered_domain
    hops = 0
    try:
        for _ in range(max_hops + 1):
            sp = urlsplit(cur)
            if sp.scheme not in ("http", "https") or not sp.hostname or not _public_host(sp.hostname):
                return out
            r = requests.get(cur, allow_redirects=False, timeout=5, stream=True, verify=False,
                             headers={"User-Agent": "Mozilla/5.0 (PhishGuard-Z research crawler)"})
            r.close()
            loc = r.headers.get("Location")
            if r.status_code in (301, 302, 303, 307, 308) and loc:
                cur, hops = urljoin(cur, loc), hops + 1
            else:
                break
        out.update(redirect_ok=1, redirect_count=hops,
                   redirect_final_domain_differs=int(_TLD(urlsplit(cur).hostname or "").registered_domain != start_reg))
    except Exception:
        pass
    return out


# ====================================================================== driver
def infra_features(args_tuple):
    raw_url, host, reg, tld, is_ip, do_redirects = args_tuple
    out = {}
    if is_ip:
        d = dict(DNS_COLS)
        d["_ip"] = host
    else:
        d = dns_features(host, reg)
    ip = d.get("_ip", "")
    out.update({k: v for k, v in d.items() if k != "_ip"})
    out.update(asn_info(ip) if ip else IP_COLS)
    out.update(RDAP_COLS if is_ip else rdap_features(reg, tld))
    if do_redirects:
        out.update(redirect_features(raw_url))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url-col", default="URL")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--limit", type=int, default=0, help="class-balanced random sample of this many rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infra", action="store_true", help="add DNS / IP-ASN / RDAP features (internet needed)")
    ap.add_argument("--redirects", action="store_true", help="add redirect count (CONTACTS the URL, isolate it!)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.limit and args.limit < len(df):
        if args.label_col in df.columns:
            per = args.limit // df[args.label_col].nunique()
            df = df.groupby(args.label_col, group_keys=False).apply(
                lambda g: g.sample(min(len(g), per), random_state=args.seed))
        else:
            df = df.sample(args.limit, random_state=args.seed)
    df = df.drop_duplicates(subset=args.url_col).reset_index(drop=True)

    keep = [c for c in (args.url_col, args.label_col, "phish", "reg_domain") if c in df.columns]
    df = df[keep]

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=[args.url_col])[args.url_col])
        print(f"resuming: {len(done)} URLs already in {args.out}")
    df = df[~df[args.url_col].isin(done)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    CHUNK = 200 if args.infra else 20000
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for start in range(0, len(df), CHUNK):
            chunk = df.iloc[start:start + CHUNK]
            lex = [lexical(u) for u in chunk[args.url_col]]
            rows = pd.DataFrame([f[0] for f in lex])
            if args.infra:
                jobs = [(u, h, rg, t, ip, args.redirects)
                        for u, (_, h, rg, t, ip) in zip(chunk[args.url_col], lex)]
                rows = pd.concat([rows, pd.DataFrame(list(ex.map(infra_features, jobs)))], axis=1)
            rows = pd.concat([chunk.reset_index(drop=True), rows], axis=1)
            rows.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            print(f"{min(start + CHUNK, len(df))}/{len(df)} rows done, {time.time() - t0:.0f}s elapsed")
    print("saved:", args.out)


if __name__ == "__main__":
    main()