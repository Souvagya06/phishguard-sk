"""
Ablation runs for PhishGuard-Z (character-level Transformer).

Usage (from the phish folder):
    python train_norm.py raw      # original URLs (reproduces the first run)
    python train_norm.py nowww    # scheme and leading www. removed, path kept
    python train_norm.py host     # scheme and www. removed, host only

Each run saves best_norm_<mode>.pt and appends a line to results_norm.csv.
"""
import sys, glob, csv, os
import torch, pandas as pd, tldextract
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from common import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"
assert MODE in ("raw", "nowww", "host"), "mode must be raw, nowww or host"
EPOCHS = 5

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", "| mode:", MODE)


def prep(u):
    if MODE == "raw":
        return u
    return norm_url(u, keep_path=(MODE == "nowww"))


def shortcut_rule_f1(d):
    """Rule: URL does NOT start with https://www.  =>  phishing (uses raw URLs)."""
    pred = (~d.URL.str.startswith("https://www.")).astype(int)
    return f1_score(d.phish, pred)


tr, va, te = load_splits()

# ---- LegitPhish (cross-dataset), loaded before normalization ----
f = glob.glob(r"E:\Hackathons\Phisguard-Z\phish\data\raw\url_features_extract*")[0]
lp = pd.read_csv(f)
lab = next(c for c in ["ClassLabel", "label", "Label", "class"] if c in lp.columns)
lp["phish"] = 1 - lp[lab]
lp["reg"] = lp.URL.map(lambda u: (lambda e: f"{e.domain}.{e.suffix}".lower())(tldextract.extract(u)))
before = len(lp)
lp = lp[~lp.reg.isin(set(tr.reg_domain))].reset_index(drop=True)   # drop domains seen in training
print(f"LegitPhish: {before} -> {len(lp)} after removing PhiUSIIL-train domains")

# exact score of the trivial prefix rule on the raw URLs (before normalization)
rule_te, rule_lp = shortcut_rule_f1(te), shortcut_rule_f1(lp)
print(f"Rule 'not https://www. => phishing'  F1: PhiUSIIL test={rule_te:.4f}  LegitPhish={rule_lp:.4f}")

# ---- apply the chosen URL preprocessing ----
for d in (tr, va, te, lp):
    d["URL"] = d.URL.map(prep)
print("example after prep:", tr.URL.iloc[0])

stoi = build_vocab(tr.URL)          # vocab from TRAIN only
V = len(stoi) + 2


def tens(df):
    return encode_batch(df.URL.tolist(), stoi), torch.tensor(df.phish.values, dtype=torch.float32)


print("encoding...")
Xtr, ytr = tens(tr); Xva, yva = tens(va); Xte, yte = tens(te); Xlp, ylp = tens(lp)
print("encoded:", Xtr.shape, Xva.shape, Xte.shape, Xlp.shape)

model = PhishNet(V).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)


@torch.no_grad()
def predict(X):
    model.eval()
    return torch.cat([torch.sigmoid(model(X[i:i + 1024].to(dev))).cpu()
                      for i in range(0, len(X), 1024)]).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(F1=f1_score(y, pred), ROC_AUC=roc_auc_score(y, p),
                PR_AUC=average_precision_score(y, p), FPR=fp / (fp + tn), FNR=fn / (fn + tp))


def report(name, y, p):
    m = metrics(y, p)
    print(f"{name}: F1={m['F1']:.4f} ROC-AUC={m['ROC_AUC']:.4f} "
          f"PR-AUC={m['PR_AUC']:.4f} FPR={m['FPR']:.4f} FNR={m['FNR']:.4f}")
    return m


ckpt = f"best_norm_{MODE}.pt"
best = 0
for ep in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    f1 = f1_score(yva.numpy(), predict(Xva) >= 0.5)
    print(f"epoch {ep + 1}  val F1 {f1:.4f}")
    if f1 > best:
        best = f1
        torch.save(model.state_dict(), ckpt)

model.load_state_dict(torch.load(ckpt))
m_te = report(f"[{MODE}] PhiUSIIL test (domain-disjoint)", yte.numpy(), predict(Xte))
m_lp = report(f"[{MODE}] LegitPhish (cross-dataset)", ylp.numpy(), predict(Xlp))

# ---- save one row per run for the ablation table ----
new = not os.path.exists("results_norm.csv")
with open("results_norm.csv", "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["mode", "phiusiil_F1", "phiusiil_ROC_AUC", "phiusiil_FPR", "phiusiil_FNR",
                    "legitphish_F1", "legitphish_ROC_AUC", "legitphish_FPR", "legitphish_FNR",
                    "rule_F1_phiusiil", "rule_F1_legitphish"])
    w.writerow([MODE, f"{m_te['F1']:.4f}", f"{m_te['ROC_AUC']:.4f}", f"{m_te['FPR']:.4f}", f"{m_te['FNR']:.4f}",
                f"{m_lp['F1']:.4f}", f"{m_lp['ROC_AUC']:.4f}", f"{m_lp['FPR']:.4f}", f"{m_lp['FNR']:.4f}",
                f"{rule_te:.4f}", f"{rule_lp:.4f}"])
print("saved row to results_norm.csv")