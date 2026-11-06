"""
Score one or more URLs with a trained PhishGuard-Z model.

Usage (from the phish folder):
    python predict_url.py "https://canva.link/f1xhZ6gZ6a47kkw"
    python predict_url.py nowww "https://github.com/torvalds/linux"
    python predict_url.py host  "https://example.com"
    python predict_url.py                 # interactive prompt
"""
import sys, os, warnings
import torch
import pandas as pd
from common import *

warnings.filterwarnings("ignore")

# Parse arguments flexibly: detect if first argument is a mode or a URL
MODES = ("raw", "nowww", "host")
if len(sys.argv) > 1 and sys.argv[1].lower() in MODES:
    MODE = sys.argv[1].lower()
    urls = sys.argv[2:]
elif len(sys.argv) > 1:
    MODE = "nowww"
    urls = sys.argv[1:]
else:
    MODE = "nowww"
    urls = []

ckpt = f"best_norm_{MODE}.pt"
assert os.path.exists(ckpt), f"{ckpt} not found. Run: python train_norm.py {MODE}"


def prep(u):
    if MODE == "raw":
        return u.strip()
    return norm_url(u, keep_path=(MODE == "nowww"))


# Rebuild or load cached vocabulary
if os.path.exists("vocab.pt"):
    stoi = torch.load("vocab.pt", weights_only=True)
else:
    tr, _, _ = load_splits()
    stoi = build_vocab(tr.URL.map(prep))
    torch.save(stoi, "vocab.pt")

V = len(stoi) + 2

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = PhishNet(V).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
model.eval()


def risk_level(p):
    if p < 0.25: return "LOW"
    if p < 0.50: return "MEDIUM"
    if p < 0.80: return "HIGH"
    return "CRITICAL"


@torch.no_grad()
def score(url):
    x = encode_batch([prep(url)], stoi).to(dev)
    return torch.sigmoid(model(x)).item()


def show(url):
    p_raw = score(url)
    print(f"\nTarget URL:             {url}")
    print(f"Model Input [{MODE}]:     {prep(url)}")
    print(f"Raw Transformer Score:  {p_raw * 100:.1f}% ({risk_level(p_raw)})")

    # Optional composite check using extract_features if available
    try:
        from extract_features import lexical, infra_features
        feats, host, reg, tld, is_ip = lexical(url)
        infra = infra_features((url, host, reg, tld, is_ip, False))

        p_adj = p_raw
        factors = []

        # Domain age adjustment
        age = infra.get("domain_age_days")
        if pd.notna(age):
            age_years = age / 365.25
            if age_years >= 5:
                discount = min(0.85, 0.50 + (age_years / 20) * 0.35)
                p_adj -= discount
                factors.append(f"Domain age ({age_years:.1f} yrs old) establishes legitimacy (-{discount*100:.0f}%)")
            elif age_years >= 2:
                p_adj -= 0.40
                factors.append(f"Matured domain age ({age_years:.1f} yrs) (-40%)")
            elif age < 14:
                p_adj += 0.30
                factors.append("Brand new domain (<14 days) (+30%)")

        if infra.get("dns_resolves") == 1 and infra.get("has_spf") == 1:
            p_adj -= 0.08
            factors.append("Valid DNS & SPF configuration (-8%)")

        if feats.get("brand_in_subdomain_or_path") or feats.get("brand_inside_label"):
            p_adj += 0.30
            factors.append("Brand keyword detected outside legitimate domain (+30%)")

        p_adj = max(0.01, min(0.99, p_adj))
        print(f"Composite Risk Score:   {p_adj * 100:.1f}% -> {risk_level(p_adj)}")
        if factors:
            print(f"Key Adjustments:        {'; '.join(factors)}")
    except Exception:
        pass


if urls:
    for u in urls:
        show(u)
else:
    print(f"PhishGuard-Z [{MODE}] loaded. Enter a URL (empty line to quit):")
    while True:
        try:
            u = input("> ").strip()
            if not u:
                break
            show(u)
        except (KeyboardInterrupt, EOFError):
            break
