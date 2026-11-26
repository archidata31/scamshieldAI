"""Train and evaluate the deep-learning models (BiGRU, DistilBERT) on the same split as train.py.

    pip install -r requirements-deep.txt
    python -m scamshield.train_deep --model bigru                      # a few minutes on CPU
    python -m scamshield.train_deep --model distilbert --max-samples 4000   # roughly 10-25 min on CPU
    python -m scamshield.train_deep --model all

Results are written to models/deep_card.json and merged into the model comparison table and chart.
"""
import argparse
import json
from datetime import datetime, timezone

import matplotlib
import pandas as pd

from . import config, deep
from .datasets import load_all
from .train import plot_comparison, scores

matplotlib.use("Agg")

ROW_NAMES = {"bigru": "BiGRU (deep learning)", "distilbert": "DistilBERT (fine-tuned transformer)"}


def evaluate(scorer, test, name):
    prob = scorer.predict_proba(test.text.tolist())[:, 1]
    t = test.assign(prob=prob)
    by_source = {s: {**scores(g.label, g.prob), "n": int(len(g))} for s, g in t.groupby("source") if g.label.nunique() == 2}
    return {"model": name, **scores(test.label, prob)}, by_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bigru", "distilbert", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=None, help="default: 8 for BiGRU, 2 for DistilBERT")
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="DistilBERT only: cap on training messages (0 = use all). Keeps CPU training time reasonable")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--bert-name", default="distilbert-base-uncased",
                    help="any Hugging Face sequence-classification checkpoint, e.g. distilbert-base-multilingual-cased")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--feedback", action="store_true")
    a = ap.parse_args()

    df = load_all(include_real=not a.no_real, include_feedback=a.feedback)
    train, test = df[df.split == "train"], df[df.split == "test"]
    val = train.sample(frac=0.1, random_state=1)
    fit = train.drop(val.index)
    print(f"fit={len(fit)}  val={len(val)}  test={len(test)}  sources={sorted(df.source.unique())}")

    names = ["bigru", "distilbert"] if a.model == "all" else [a.model]
    card = json.loads(config.DEEP_CARD_PATH.read_text()) if config.DEEP_CARD_PATH.exists() else {"models": {}}
    for name in names:
        st = deep.status(name)
        if not st["runtime"]:
            raise SystemExit(f"\n{st['message']}")
        t0 = pd.Timestamp.now()
        if name == "bigru":
            info = deep.train_bigru(fit.text.tolist(), fit.label.tolist(), val.text.tolist(), val.label.tolist(),
                                    epochs=a.epochs or 8, batch=a.batch or 64)
        else:
            sub = fit.sample(n=a.max_samples, random_state=0) if a.max_samples and len(fit) > a.max_samples else fit
            info = deep.train_distilbert(sub.text.tolist(), sub.label.tolist(), val.text.tolist(), val.label.tolist(),
                                         model_name=a.bert_name, epochs=a.epochs or 2, batch=a.batch or 16,
                                         max_len=a.max_len)
        minutes = round((pd.Timestamp.now() - t0).total_seconds() / 60, 1)
        print(f"evaluating {name} on {len(test)} held-out messages ...")
        row, by_source = evaluate(deep.load_scorer(name), test, ROW_NAMES[name])
        print("  ", row)
        for s, v in by_source.items():
            print(f"   [{s}]", v)
        card["models"][name] = {"row": row, "by_source": by_source, "training": {**info, "minutes": minutes},
                                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ensemble = average of logistic regression and every available deep model (same logic the app uses)
    from .predict import ScamShield
    shield = ScamShield()
    ens = shield.proba_by_model(test.text.tolist(), "ensemble")
    ens_row = {"model": "Ensemble (LR + deep models)", **scores(test.label, ens)}
    card["ensemble"] = {"row": ens_row}
    print("ensemble:", ens_row)

    config.DEEP_CARD_PATH.write_text(json.dumps(card, indent=2))
    base = json.loads(config.CARD_PATH.read_text())["model_comparison"] if config.CARD_PATH.exists() else []
    rows = base + [m["row"] for m in card["models"].values()] + [ens_row]
    plot_comparison(rows, config.MODEL_DIR)
    print(f"saved {config.DEEP_CARD_PATH} and refreshed models/model_comparison.png")


if __name__ == "__main__":
    main()
