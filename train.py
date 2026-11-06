import glob, torch, pandas as pd, tldextract
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from common import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

tr, va, te = load_splits()
# tr = tr.sample(40000, random_state=0)     # uncomment for a quick CPU test run
stoi = build_vocab(tr.URL); V = len(stoi) + 2

def tens(df):
    return encode_batch(df.URL.tolist(), stoi), torch.tensor(df.phish.values, dtype=torch.float32)

Xtr, ytr = tens(tr); Xva, yva = tens(va); Xte, yte = tens(te)

model = PhishNet(V).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

@torch.no_grad()
def predict(X):
    model.eval()
    return torch.cat([torch.sigmoid(model(X[i:i+1024].to(dev))).cpu()
                      for i in range(0, len(X), 1024)]).numpy()

def report(name, y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"{name}: F1={f1_score(y, pred):.4f} ROC-AUC={roc_auc_score(y, p):.4f} "
          f"PR-AUC={average_precision_score(y, p):.4f} FPR={fp/(fp+tn):.4f} FNR={fn/(fn+tp):.4f}")

best = 0
for ep in range(5):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    f1 = f1_score(yva.numpy(), predict(Xva) >= 0.5)
    print(f"epoch {ep+1}  val F1 {f1:.4f}")
    if f1 > best:
        best = f1; torch.save(model.state_dict(), "best.pt")

model.load_state_dict(torch.load("best.pt"))
report("PhiUSIIL test (domain-disjoint)", yte.numpy(), predict(Xte))

# ---- cross-dataset: LegitPhish ----
f = glob.glob(r"E:\Hackathons\Phisguard-Z\phish\data\raw\url_features_extract*")[0]
lp = pd.read_csv(f)
lab = next(c for c in ["ClassLabel", "label", "Label", "class"] if c in lp.columns)
lp["phish"] = 1 - lp[lab]
lp["reg"] = lp.URL.map(lambda u: (lambda e: f"{e.domain}.{e.suffix}".lower())(tldextract.extract(u)))
before = len(lp)
lp = lp[~lp.reg.isin(set(tr.reg_domain))]          # drop domains seen in training
print(f"LegitPhish: {before} -> {len(lp)} after removing PhiUSIIL-train domains")
Xlp, ylp = tens(lp)
report("LegitPhish (cross-dataset)", ylp.numpy(), predict(Xlp))

from urllib.parse import urlparse
def has_path(u):
    p = urlparse(u)
    return int(len(p.path.strip("/")) > 0 or bool(p.query))

for name, d in [("PhiUSIIL train", tr), ("LegitPhish", lp)]:
    hp = d.URL.map(has_path)
    print(name, "share of URLs with a path/query:\n", hp.groupby(d.phish).mean())
    print("  F1 of the rule 'path => phishing':", f1_score(d.phish, hp))

p_lp = predict(Xlp)
m = lp.URL.map(has_path).values == 1
report("LegitPhish, only URLs with path/query", ylp.numpy()[m], p_lp[m])

nopath = ~te.URL.map(has_path).astype(bool)
Xn, yn = tens(te[nopath])
report("PhiUSIIL test, bare-domain URLs only", yn.numpy(), predict(Xn))

# other possible shortcuts: how URLs start
for name, d in [("PhiUSIIL train", tr), ("LegitPhish", lp)]:
    print(name, "share starting with https://www.:\n",
          d.URL.str.startswith("https://www.").groupby(d.phish).mean())