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
- AI Transformer model risk score (PhishNet)

Usage:
    python inspect_url.py "https://example.com/login"
    python inspect_url.py                 # interactive prompt
"""
import sys, os, warnings
import pandas as pd
import numpy as np

# Suppress noise
warnings.filterwarnings("ignore")

from extract_features import lexical, infra_features, SUSPICIOUS_TOKENS, BRANDS, RISKY_TLDS


def inspect_url(raw_url: str, check_infra: bool = True, check_redirects: bool = True, run_model: bool = True):
    print("\n" + "=" * 65)
    print(f"  PHISHGUARD-Z URL INSPECTOR")
    print("=" * 65)
    print(f"Target URL: {raw_url}")
    print("-" * 65)

    # 1. Lexical features
    feats, host, reg, tld, is_ip = lexical(raw_url)

    # Find matched suspicious tokens and brands
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
    print(f"  • Randomness Score:      {feats['randomness_score']:.3f} (0=human readable, 1=high randomness)")
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

    # 2. Infra & Network features
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
            print(f"  • Domain Age:            {f'{int(age)} days ({age/365:.1f} years)' if pd.notna(age) else 'Unknown / Private'}")
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

    # 3. Model Prediction (if available)
    if run_model and os.path.exists("best_norm_nowww.pt"):
        print("\n[8] PHISHGUARD-Z AI TRANSFORMER MODEL")
        try:
            import torch
            from common import load_splits, build_vocab, PhishNet, norm_url, encode_batch
            tr, _, _ = load_splits()
            stoi = build_vocab(tr.URL.map(lambda u: norm_url(u, keep_path=True)))
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = PhishNet(len(stoi) + 2).to(dev)
            model.load_state_dict(torch.load("best_norm_nowww.pt", map_location=dev))
            model.eval()

            clean = norm_url(raw_url, keep_path=True)
            x = encode_batch([clean], stoi).to(dev)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            risk = "CRITICAL" if prob >= 0.8 else "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= 0.25 else "LOW"
            print(f"  • Phishing Probability:  {prob * 100:.1f}%")
            print(f"  • Final Verdict:         {risk}")
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

