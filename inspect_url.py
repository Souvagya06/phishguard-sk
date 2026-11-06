"""
Inspect any user-given URL with full PhishGuard-Z feature extraction:
- URL length & path depth
- Shannon entropy
- Suspicious tokens & brand targeting
- TLD characteristics
- Randomness score
- DNS records
- IP & ASN information
- Domain age & registrar information (RDAP)
- Redirect count & hop analysis
- AI Transformer model score (PhishNet)
- Composite Multi-Layered Verdict (AI + Infrastructure)

Usage:
    python inspect_url.py "https://canva.link/f1xhZ6gZ6a47kkw"
    python inspect_url.py                 # interactive prompt
"""
import sys, os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from extract_features import lexical, infra_features, SUSPICIOUS_TOKENS, BRANDS, RISKY_TLDS


def compute_composite_risk(raw_prob, feats, infra):
    """
    Combines the character Transformer score with Infrastructure & WHOIS intelligence.
    Eliminates false positives on aged domains, shortened links, and corporate infrastructure.
    """
    p = raw_prob
    factors = []

    # 1. Domain Age & Registrar Trust
    age = infra.get("domain_age_days")
    if pd.notna(age):
        age_years = age / 365.25
        if age_years >= 5:
            discount = min(0.85, 0.50 + (age_years / 20) * 0.35)
            p -= discount
            factors.append(f"Domain age ({age_years:.1f} years old) establishes strong trust (-{discount*100:.0f}%)")
        elif age_years >= 2:
            p -= 0.40
            factors.append(f"Matured domain ({age_years:.1f} years old) (-40%)")
        elif age_years >= 1:
            p -= 0.20
            factors.append("Domain age > 1 year (-20%)")
        elif age < 14:
            p += 0.35
            factors.append("Brand new domain (<14 days old) (+35%)")

    # 2. Infrastructure Health
    if infra.get("dns_resolves") == 1 and infra.get("has_spf") == 1:
        p -= 0.08
        factors.append("Legitimate Mail & DNS Configuration (-8%)")
    elif infra.get("dns_resolves") == 0 and pd.isna(age):
        p += 0.10
        factors.append("Domain fails to resolve DNS & registration (+10%)")

    # 3. Brand Impersonation & Tokens
    if feats.get("brand_in_subdomain_or_path") or feats.get("brand_inside_label"):
        p += 0.35
        factors.append("Targeted Brand name detected outside legitimate domain (+35%)")

    if feats.get("is_ip_host"):
        p += 0.30
        factors.append("Direct IP address used as host (+30%)")

    p = max(0.01, min(0.99, p))
    risk = "CRITICAL" if p >= 0.80 else "HIGH" if p >= 0.50 else "MEDIUM" if p >= 0.25 else "LOW"
    return p, risk, factors


def inspect_url(raw_url: str, check_infra: bool = True, check_redirects: bool = True, run_model: bool = True):
    print("\n" + "=" * 65)
    print("  PHISHGUARD-Z FULL URL INSPECTOR")
    print("=" * 65)
    print(f"Target URL: {raw_url}")
    print("-" * 65)

    # 1. Lexical features
    feats, host, reg, tld, is_ip = lexical(raw_url)

    lower_u = raw_url.lower()
    matched_tokens = [t for t in SUSPICIOUS_TOKENS if t in lower_u]
    matched_brands = [b for b in BRANDS if b in lower_u]

    print("\n[1] URL STRUCTURE & LENGTH")
    print(f"  • Total URL Length:      {feats['url_length']}")
    print(f"  • Host Length:           {feats['host_length']}")
    print(f"  • Path Length:           {feats['path_length']}")
    print(f"  • Path Depth (segments): {feats['path_depth']}")
    print(f"  • Subdomain Depth:       {feats['subdomain_depth']}")
    print(f"  • Number of Query Params:{feats['num_params']}")
    print(f"  • Is IP as Host:         {'YES' if feats['is_ip_host'] else 'NO'}")

    print("\n[2] ENTROPY & RANDOMNESS")
    print(f"  • URL Shannon Entropy:   {feats['url_entropy']:.3f}")
    print(f"  • Host Entropy:          {feats['host_entropy']:.3f}")
    print(f"  • Path Entropy:          {feats['path_entropy']:.3f}")
    print(f"  • Randomness Score:      {feats['randomness_score']:.3f} (0=readable, 1=machine-generated)")
    print(f"  • Label Vowel Ratio:     {feats['label_vowel_ratio']:.2f}")
    print(f"  • Longest Consonant Run: {feats['label_max_consonant_run']}")
    print(f"  • Digit Ratio:           {feats['label_digit_ratio']:.2f}")

    print("\n[3] SUSPICIOUS TOKENS & BRAND TARGETING")
    print(f"  • Suspicious Token Count:{feats['susp_token_count']}")
    print(f"  • Tokens Detected:       {', '.join(matched_tokens) if matched_tokens else 'None'}")
    print(f"  • Brand Keywords Found:  {', '.join(matched_brands) if matched_brands else 'None'}")
    print(f"  • Brand Impersonation:   {'YES (in subdomain/path)' if feats['brand_in_subdomain_or_path'] else 'None'}")

    print("\n[4] TLD CHARACTERISTICS")
    print(f"  • TLD:                   .{feats['tld']}")
    print(f"  • TLD Length:            {feats['tld_length']}")
    print(f"  • Country-Code TLD:      {'YES' if feats['tld_is_country'] else 'NO'}")
    print(f"  • Known Risky TLD:       {'YES (Flagged)' if feats['tld_in_risky_list'] else 'NO'}")

    infra = {}
    if check_infra:
        print("\n[5] INFRASTRUCTURE, DNS & IP")
        try:
            infra = infra_features((raw_url, host, reg, tld, is_ip, check_redirects))

            resolves = infra.get("dns_resolves")
            dns_status = "Resolves" if resolves == 1 else ("Does not resolve" if resolves == 0 else "Query failed / offline")
            print(f"  • DNS Resolution:        {dns_status}")
            print(f"  • A Records (IPv4):      {infra.get('a_count', 'N/A')}")
            print(f"  • AAAA Records (IPv6):   {infra.get('aaaa_count', 'N/A')}")
            print(f"  • Name Servers (NS):     {infra.get('ns_count', 'N/A')}")
            print(f"  • Mail Servers (MX):     {infra.get('mx_count', 'N/A')}")
            print(f"  • SPF Record Configured: {'YES' if infra.get('has_spf') == 1 else 'NO'}")
            print(f"  • Minimum DNS TTL:       {infra.get('dns_ttl_min', 'N/A')} s")
            print(f"  • ASN / Network:         {infra.get('asn', 'N/A')} ({infra.get('asn_cc', 'Unknown')})")

            print("\n[6] DOMAIN REGISTRATION & AGE (RDAP/WHOIS)")
            age = infra.get("domain_age_days")
            print(f"  • Domain Age:            {f'{int(age)} days ({age/365.25:.1f} years)' if pd.notna(age) else 'Unknown / Private'}")
            days_exp = infra.get("days_to_expiry")
            print(f"  • Days to Expiry:        {f'{int(days_exp)} days' if pd.notna(days_exp) else 'Unknown'}")
            print(f"  • Registrar:             {infra.get('registrar') or 'Unknown / Hidden'}")

            if check_redirects:
                print("\n[7] REDIRECT ANALYSIS")
                hops = infra.get("redirect_count")
                differs = infra.get("redirect_final_domain_differs")
                print(f"  • Number of Redirects:   {int(hops) if pd.notna(hops) else '0'}")
                print(f"  • Final Domain Changed:  {'YES (Suspicious Hop)' if differs == 1 else 'NO'}")

        except Exception as e:
            print(f"  [!] Infrastructure check skipped/failed: {e}")

    # 3. Model Prediction & Multi-Layered Verdict
    if run_model and os.path.exists("best_norm_nowww.pt"):
        print("\n[8] PHISHGUARD-Z MULTI-LAYERED ANALYSIS")
        try:
            import torch
            from common import PhishNet, norm_url, encode_batch

            if os.path.exists("vocab.pt"):
                stoi = torch.load("vocab.pt", weights_only=True)
            else:
                from common import load_splits, build_vocab
                tr, _, _ = load_splits()
                stoi = build_vocab(tr.URL.map(lambda u: norm_url(u, keep_path=True)))
                torch.save(stoi, "vocab.pt")

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = PhishNet(len(stoi) + 2).to(dev)
            model.load_state_dict(torch.load("best_norm_nowww.pt", map_location=dev, weights_only=True))
            model.eval()

            clean = norm_url(raw_url, keep_path=True)
            x = encode_batch([clean], stoi).to(dev)
            with torch.no_grad():
                raw_prob = torch.sigmoid(model(x)).item()

            comp_prob, comp_risk, adjustments = compute_composite_risk(raw_prob, feats, infra)

            print(f"  • Raw Transformer Score: {raw_prob * 100:.1f}%")
            if adjustments:
                print("  • Infrastructure Adjustments:")
                for adj in adjustments:
                    print(f"      - {adj}")
            print(f"  -------------------------------------------------")
            print(f"  • FINAL COMPOSITE VERDICT:{comp_prob * 100:.1f}% -> {comp_risk}")

        except Exception as e:
            print(f"  [!] Model inference skipped: {e}")

    print("\n" + "=" * 65 + "\n")


def main():
    if len(sys.argv) > 1:
        for u in sys.argv[1:]:
            inspect_url(u)
    else:
        print("PhishGuard-Z Full URL Inspector")
        print("Enter a URL to test (empty line to quit):")
        while True:
            try:
                u = input("> ").strip()
                if not u:
                    break
                inspect_url(u)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
