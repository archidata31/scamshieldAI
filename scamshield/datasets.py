"""Dataset loading: synthetic corpus + any real datasets in data/raw + user feedback."""
import hashlib

import pandas as pd

from . import config

COLUMNS = ["text", "label", "category", "split", "source", "group"]


def _normalise_label(v):
    return 1 if str(v).strip().lower() in {"1", "spam", "phishing", "phishing email", "scam", "smishing", "fraud"} else 0


def _read_labelled(path):
    """Read a CSV/TSV with (text,label) columns, or the SMS Spam Collection layouts."""
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, encoding="latin-1", on_bad_lines="skip")
    cols = {c.lower().strip(): c for c in df.columns}
    if {"v1", "v2"} <= set(cols):
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})
    else:
        text_col = next((cols[c] for c in ("text", "message", "sms", "body", "email text", "email_text") if c in cols), None)
        label_col = next((cols[c] for c in ("label", "class", "category", "type", "email type", "email_type") if c in cols), None)
        if text_col is None or label_col is None:
            raise ValueError(f"{path}: need a text column and a label column, found {list(df.columns)}")
        df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, 1500)   # keep long emails manageable
    df["label"] = df["label"].map(_normalise_label)
    return df[df.text.str.len() > 3].drop_duplicates("text")


def _split_real(df, name):
    """Deterministic 80/20 split by hashing the text (stable across runs and machines)."""
    h = df.text.map(lambda t: int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % 100)
    df = df.assign(split=["test" if x < 20 else "train" for x in h],
                   category="real", source=name)
    df["group"] = [f"{name}-{i}" for i in range(len(df))]
    return df


def load_all(include_real=True, include_feedback=False):
    frames = []
    synth = config.DATA_DIR / "synthetic.csv"
    if not synth.exists():
        raise FileNotFoundError(f"{synth} not found. Run: python -m scamshield.data_gen")
    frames.append(pd.read_csv(synth))

    if include_real and config.RAW_DIR.exists():
        for f in sorted(config.RAW_DIR.glob("*")):
            if f.suffix.lower() in {".csv", ".tsv", ".txt"}:
                frames.append(_split_real(_read_labelled(f), f.stem))
                print(f"loaded real dataset {f.name}: {len(frames[-1])} messages")
    if include_feedback and config.FEEDBACK_PATH.exists():
        fb = pd.read_csv(config.FEEDBACK_PATH)
        fb = fb.rename(columns={"user_label": "label"})[["text", "label"]].dropna()
        fb["label"] = fb.label.map(_normalise_label)
        frames.append(fb.assign(split="train", category="feedback", source="feedback",
                                group=[f"fb-{i}" for i in range(len(fb))]))
        print(f"loaded {len(fb)} feedback rows")
    return pd.concat(frames, ignore_index=True)[COLUMNS]
