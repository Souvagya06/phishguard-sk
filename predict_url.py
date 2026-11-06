"""
Score one or more URLs with a trained PhishGuard-Z model (static, URL-text only).

Usage (from the phish folder):
    python predict_url.py raw   "https://www.google.com" "http://secure-paypal-login.xyz/verify?token=8f9a"
    python predict_url.py nowww "https://github.com/torvalds/linux"
    python predict_url.py host  "https://example.com"

With no URLs given, it asks for them one per line (empty line to quit).
The mode must match a checkpoint you trained: best_norm_<mode>.pt
"""
import sys, os, warnings
import torch
from common import *

warnings.filterwarnings("ignore")

MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"
assert MODE in ("raw", "nowww", "host"), "mode must be raw, nowww or host"
ckpt = f"best_norm_{MODE}.pt"
assert os.path.exists(ckpt), f"{ckpt} not found. Run: python train_norm.py {MODE}"


def prep(u):
    if MODE == "raw":
        return u.strip()
    return norm_url(u, keep_path=(MODE == "nowww"))


# Rebuild the exact training vocabulary (deterministic: sorted characters of TRAIN urls)
tr, _, _ = load_splits()
stoi = build_vocab(tr.URL.map(prep))
V = len(stoi) + 2

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = PhishNet(V).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev))
model.eval()


def risk_level(p):                      # thresholds are configurable
    if p < 0.25: return "LOW"
    if p < 0.50: return "MEDIUM"
    if p < 0.80: return "HIGH"
    return "CRITICAL"


@torch.no_grad()
def score(url):
    x = encode_batch([prep(url)], stoi).to(dev)
    return torch.sigmoid(model(x)).item()


def show(url):
    p = score(url)
    print(f"\nURL:                  {url}")
    print(f"Model input:          {prep(url)}")
    print(f"Phishing probability: {p * 100:.1f}%")
    print(f"Risk level:           {risk_level(p)}")


urls = sys.argv[2:]
if urls:
    for u in urls:
        show(u)
else:
    print(f"[{MODE}] model loaded. Enter a URL (empty line to quit):")
    while True:
        u = input("> ").strip()
        if not u:
            break
        show(u)