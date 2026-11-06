"""
PhishGuard-Z Web Dashboard (Flask).
Single-command full interactive UI for zero-day phishing detection & multi-layered intelligence.

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""
import os, sys, warnings
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

from common import PhishNet, norm_url, encode_batch
from extract_features import lexical, infra_features, SUSPICIOUS_TOKENS, BRANDS, RISKY_TLDS
from inspect_url import compute_composite_risk

app = Flask(__name__)

# Load model and cached vocab
dev = "cuda" if torch.cuda.is_available() else "cpu"
stoi = None
model = None

if os.path.exists("vocab.pt") and os.path.exists("best_norm_nowww.pt"):
    stoi = torch.load("vocab.pt", weights_only=True)
    model = PhishNet(len(stoi) + 2).to(dev)
    model.load_state_dict(torch.load("best_norm_nowww.pt", map_location=dev, weights_only=True))
    model.eval()
    print("[+] PhishGuard-Z Transformer Model loaded successfully.")


def analyze_url(url: str):
    url = url.strip()
    if not url:
        return {"error": "Please enter a valid URL."}

    # 1. Lexical analysis
    feats, host, reg, tld, is_ip = lexical(url)
    lower_u = url.lower()
    matched_tokens = [t for t in SUSPICIOUS_TOKENS if t in lower_u]
    matched_brands = [b for b in BRANDS if b in lower_u]

    # 2. Infra analysis (DNS, ASN, RDAP, Redirects)
    try:
        infra = infra_features((url, host, reg, tld, is_ip, True))
    except Exception as e:
        infra = {"dns_resolves": 0, "error": str(e)}

    # 3. AI Transformer prediction
    raw_prob = 0.5
    if model is not None and stoi is not None:
        clean = norm_url(url, keep_path=True)
        x = encode_batch([clean], stoi).to(dev)
        with torch.no_grad():
            raw_prob = torch.sigmoid(model(x)).item()

    # 4. Composite multi-layered risk calculation
    comp_prob, comp_risk, adjustments = compute_composite_risk(raw_prob, feats, infra)

    domain_age = infra.get("domain_age_days")
    age_str = f"{int(domain_age)} days ({domain_age/365.25:.1f} yrs)" if pd.notna(domain_age) else "Unknown / Private"

    return {
        "url": url,
        "clean_input": norm_url(url, keep_path=True),
        "composite_prob": round(comp_prob * 100, 1),
        "composite_risk": comp_risk,
        "raw_transformer_prob": round(raw_prob * 100, 1),
        "adjustments": adjustments,
        "lexical": {
            "url_length": feats.get("url_length", len(url)),
            "host_length": feats.get("host_length", len(host)),
            "path_depth": feats.get("path_depth", 0),
            "num_params": feats.get("num_params", 0),
            "url_entropy": round(feats.get("url_entropy", 0.0), 3),
            "host_entropy": round(feats.get("host_entropy", 0.0), 3),
            "path_entropy": round(feats.get("path_entropy", 0.0), 3),
            "randomness_score": round(feats.get("randomness_score", 0.0), 3),
            "longest_consonant_run": feats.get("label_max_consonant_run", 0),
            "is_ip_host": bool(feats.get("is_ip_host", 0)),
            "tld": f".{feats.get('tld', '')}",
            "is_country_tld": bool(feats.get("tld_is_country", 0)),
            "is_risky_tld": bool(feats.get("tld_in_risky_list", 0)),
            "suspicious_tokens": matched_tokens,
            "matched_brands": matched_brands,
            "brand_impersonation": bool(feats.get("brand_in_subdomain_or_path") or feats.get("brand_inside_label")),
        },
        "infra": {
            "dns_resolves": bool(infra.get("dns_resolves") == 1),
            "a_records": int(infra.get("a_count")) if pd.notna(infra.get("a_count")) else 0,
            "aaaa_records": int(infra.get("aaaa_count")) if pd.notna(infra.get("aaaa_count")) else 0,
            "ns_records": int(infra.get("ns_count")) if pd.notna(infra.get("ns_count")) else 0,
            "has_spf": bool(infra.get("has_spf") == 1),
            "asn": infra.get("asn") if pd.notna(infra.get("asn")) else "Unknown",
            "asn_country": infra.get("asn_cc", "") or "Unknown",
            "domain_age": age_str,
            "registrar": infra.get("registrar") or "Unknown / Hidden",
            "redirect_count": int(infra.get("redirect_count")) if pd.notna(infra.get("redirect_count")) else 0,
            "redirect_domain_changed": bool(infra.get("redirect_final_domain_differs") == 1),
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL parameter missing"}), 400
    res = analyze_url(url)
    return jsonify(res)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n" + "=" * 60)
    print(f"  PhishGuard-Z Web Server Running on http://127.0.0.1:{port}")
    print(f"=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
