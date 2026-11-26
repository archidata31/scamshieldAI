"""Deep-learning classifiers: BiGRU and fine-tuned DistilBERT.

PyTorch and Transformers are imported lazily, so the rest of ScamShield (logistic regression, SHAP,
the web app) keeps working when they are not installed. Install with:

    pip install -r requirements-deep.txt

Every scorer exposes the same interface as a scikit-learn model:
    scorer.predict_proba(list_of_texts) -> numpy array of shape (n, 2)   # [P(legit), P(scam)]
so the app, the ensemble and LIME can treat all models identically.
"""
import importlib.util
import json
import re
import time
from collections import Counter

import numpy as np

from . import config
from .features import normalize_text

MAX_LEN_GRU = 120
BIGRU_FILES = ("model.pt", "vocab.json", "config.json")


class DeepUnavailable(RuntimeError):
    """The deep model is not installed or not trained yet (message explains how to fix it)."""


def _has(module):
    return importlib.util.find_spec(module) is not None


def status(name):
    """Availability of a deep model without importing torch."""
    if name == "bigru":
        trained = all((config.BIGRU_DIR / f).exists() for f in BIGRU_FILES)
        need = ["torch"]
    elif name == "distilbert":
        trained = (config.BERT_DIR / "config.json").exists()
        need = ["torch", "transformers"]
    else:
        raise ValueError(f"unknown deep model: {name}")
    missing = [m for m in need if not _has(m)]
    if missing:
        msg = f"Install {' and '.join(missing)}: pip install -r requirements-deep.txt"
    elif not trained:
        msg = f"Not trained yet. Run: python -m scamshield.train_deep --model {name}"
    else:
        msg = "Ready"
    return {"name": name, "trained": trained, "runtime": not missing, "available": trained and not missing,
            "message": msg}


# ------------------------------------------------------------------ BiGRU: text -> ids (pure Python)
def tokenize(text):
    """Normalised word / punctuation tokens (URLs -> 'urltoken', numbers -> 'numx')."""
    return re.findall(r"[a-z]+|[^\sa-z]", normalize_text(text))


def build_vocab(texts, max_size=30000, min_freq=2):
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    itos = ["<pad>", "<unk>"] + [w for w, n in counts.most_common(max_size) if n >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(text, vocab, max_len=MAX_LEN_GRU):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)[:max_len]]
    return ids or [1]


def pad_batch(seqs):
    width = max(len(s) for s in seqs)
    return [s + [0] * (width - len(s)) for s in seqs]


# ------------------------------------------------------------------ BiGRU model
def make_bigru(vocab_size, emb=128, hid=96):
    import torch
    import torch.nn as nn

    class BiGRU(nn.Module):
        """Embedding -> bidirectional GRU -> [mean-pool ; max-pool] -> dropout -> linear logit."""

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hid * 4, 1)

        def forward(self, x):
            mask = (x != 0).unsqueeze(-1)                         # (B, T, 1)
            h, _ = self.gru(self.emb(x))                          # (B, T, 2H)
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            mx = h.masked_fill(~mask, -1e9).max(1).values
            return self.fc(self.drop(torch.cat([mean, mx], dim=1))).squeeze(-1)

    return BiGRU()


class BiGRUScorer:
    def __init__(self, path=None):
        import torch
        path = path or config.BIGRU_DIR
        cfg = json.loads((path / "config.json").read_text())
        self.vocab = json.loads((path / "vocab.json").read_text())
        self.max_len = cfg["max_len"]
        self.torch = torch
        self.model = make_bigru(len(self.vocab), cfg["emb"], cfg["hid"])
        self.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        self.model.eval()

    def predict_proba(self, texts):
        texts, torch, out = list(texts), self.torch, []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                seqs = [encode(t, self.vocab, self.max_len) for t in texts[i:i + 256]]
                x = torch.tensor(pad_batch(seqs), dtype=torch.long)
                out.append(torch.sigmoid(self.model(x)).numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return np.column_stack([1 - p, p])


def train_bigru(texts, labels, val_texts, val_labels, epochs=8, batch=64, lr=2e-3, seed=42, out=None, log=print):
    import torch
    import torch.nn as nn

    out = out or config.BIGRU_DIR
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(texts)
    X, y = [encode(t, vocab) for t in texts], np.asarray(labels, dtype=np.float32)
    Xv, yv = [encode(t, vocab) for t in val_texts], np.asarray(val_labels, dtype=np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_bigru(len(vocab)).to(dev)
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(float(y.sum()), 1.0)], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def batches(seqs, ys, shuffle):
        order = rng.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = torch.tensor(pad_batch([seqs[j] for j in idx]), dtype=torch.long, device=dev)
            yield xb, torch.tensor(ys[idx], dtype=torch.float32, device=dev)

    best_loss, best_state = float("inf"), None
    log(f"BiGRU: {len(X)} train / {len(Xv)} val messages, vocab {len(vocab)}, device {dev}")
    for ep in range(1, epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for xb, yb in batches(X, y, True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(yb)
        model.eval()
        val = 0.0
        with torch.no_grad():
            for xb, yb in batches(Xv, yv, False):
                val += loss_fn(model(xb), yb).item() * len(yb)
        val /= len(yv)
        log(f"  epoch {ep}/{epochs}  train loss {run / len(y):.4f}  val loss {val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "config.json").write_text(json.dumps({"emb": 128, "hid": 96, "max_len": MAX_LEN_GRU,
                                                 "epochs": epochs, "best_val_loss": round(best_loss, 4)}))
    return {"epochs": epochs, "best_val_loss": round(best_loss, 4), "vocab": len(vocab)}


# ------------------------------------------------------------------ DistilBERT
class BertScorer:
    def __init__(self, path=None, max_len=128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = path or config.BERT_DIR
        self.torch, self.max_len = torch, max_len
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()

    def predict_proba(self, texts):
        texts, torch, out = list(texts), self.torch, []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i:i + 32], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt")
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))


def train_distilbert(texts, labels, val_texts, val_labels, model_name="distilbert-base-uncased", epochs=2,
                     batch=16, lr=5e-5, max_len=128, out=None, log=print):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    out = out or config.BERT_DIR
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"DistilBERT: loading '{model_name}' (first run downloads ~260 MB) ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)
    texts, y = list(texts), np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = torch.tensor(len(y) / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=dev)

    steps_per_epoch = int(np.ceil(len(texts) / batch))
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    log(f"  {len(texts)} train messages, {epochs} epochs, {total} steps, device {dev}")

    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            yb = torch.tensor(y[idx], device=dev)
            loss = F.cross_entropy(model(**enc).logits, yb, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                log(f"  step {step}/{total}  loss {loss.item():.4f}  elapsed {el / 60:.1f} min  "
                    f"eta {el / step * (total - step) / 60:.1f} min")

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(val_texts), 32):
            enc = tok(list(val_texts[i:i + 32]), truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            pred = model(**enc).logits.argmax(-1).cpu().numpy()
            correct += int((pred == np.asarray(val_labels[i:i + 32])).sum())
    val_acc = correct / max(len(val_texts), 1)
    log(f"  validation accuracy {val_acc:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return {"epochs": epochs, "base_model": model_name, "val_accuracy": round(val_acc, 4), "train_size": len(texts)}


# ------------------------------------------------------------------ factory
def load_scorer(name):
    st = status(name)
    if not st["available"]:
        raise DeepUnavailable(st["message"])
    return BiGRUScorer() if name == "bigru" else BertScorer()
