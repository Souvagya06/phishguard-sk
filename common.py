import re
import pandas as pd, torch, torch.nn as nn

P = r"E:\Hackathons\Phisguard-Z\phish\data\processed"
MAXLEN = 200


def load_splits():
    tr, va, te = [pd.read_csv(rf"{P}\{n}.csv") for n in ("train", "val", "test")]
    for d in (tr, va, te):
        d["phish"] = 1 - d["label"]          # 1 = phishing from here on
    return tr, va, te


def norm_url(u, keep_path=True):
    """Remove dataset artifacts: scheme (http/https) and leading www.
    keep_path=False keeps only the host (drops path, query, fragment)."""
    u = str(u).strip().lower()
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)   # drop scheme
    u = re.sub(r"^www\d*\.", "", u)                # drop leading www.
    if not keep_path:
        u = re.split(r"[/?#]", u, maxsplit=1)[0]   # host only
    return u


def build_vocab(urls):                        # call on TRAIN urls only
    return {c: i + 2 for i, c in enumerate(sorted(set("".join(urls))))}


def encode_batch(urls, stoi):
    out = torch.zeros(len(urls), MAXLEN, dtype=torch.long)
    for i, u in enumerate(urls):
        ids = [stoi.get(c, 1) for c in u[:MAXLEN]]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


class URLTransformer(nn.Module):
    def __init__(self, vocab, d=128, heads=4, layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN, d))
        enc = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)

    def forward(self, x):
        mask = x == 0
        h = self.enc(self.emb(x) + self.pos, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0)
        return h.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)


class PhishNet(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.enc = URLTransformer(vocab)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)