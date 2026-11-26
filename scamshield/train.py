"""Train, evaluate and package ScamShield.

    python -m scamshield.data_gen
    python -m scamshield.download_data      # optional but recommended: adds real SMS data
    python -m scamshield.train [--no-real] [--feedback]

Outputs (models/): scamshield.joblib, model_card.json and evaluation plots.

Evaluation design
  * Synthetic data: the test split uses message templates never seen in training.
  * Real data: deterministic 80/20 split by message hash.
  * Alert thresholds are chosen on out-of-fold predictions from grouped cross-validation
    (never on the test set), then reported on the test set.
"""
import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import __version__, config
from .datasets import load_all
from . import explain
from .features import build_pipeline, feature_names

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY, TEAL, RED, GREY = "#16315f", "#0f9d8f", "#d92d20", "#8a94a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold", "figure.dpi": 130})

CANDIDATES = [  # name, model, use_features
    ("TF-IDF + Naive Bayes", "nb", False),
    ("TF-IDF + Logistic Regression", "lr", False),
    ("TF-IDF + fraud features + Linear SVM", "svm", True),
    ("TF-IDF + fraud features + Logistic Regression (ScamShield)", "lr", True),
]


def scores(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, prob), 4),
            "pr_auc": round(average_precision_score(y, prob), 4)}


def pick_thresholds(y, prob, p_target=0.97, r_target=0.95):
    """scam = lowest threshold with precision >= target; suspicious = highest threshold with recall >= target."""
    p, r, t = precision_recall_curve(y, prob)
    ok = np.where(p[:-1] >= p_target)[0]
    scam = float(t[ok[0]]) if len(ok) else config.DEFAULT_SCAM
    ok = np.where(r[:-1] >= r_target)[0]
    susp = float(t[ok[-1]]) if len(ok) else config.DEFAULT_SUSPICIOUS
    scam = float(np.clip(scam, 0.5, 0.85))
    susp = float(np.clip(susp, 0.15, scam - 0.25))  # keep a usable "suspicious" band
    return round(susp, 3), round(scam, 3)


def plot_curves(y, prob, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y, prob)
    ax[0].plot(fpr, tpr, color=TEAL, lw=2.4)
    ax[0].plot([0, 1], [0, 1], "--", color=GREY)
    ax[0].set(title=f"ROC curve (AUC {roc_auc_score(y, prob):.3f})", xlabel="False positive rate", ylabel="True positive rate")
    p, r, _ = precision_recall_curve(y, prob)
    ax[1].plot(r, p, color=NAVY, lw=2.4)
    ax[1].set(title=f"Precision-Recall (AP {average_precision_score(y, prob):.3f})", xlabel="Recall",
              ylabel="Precision", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out / "roc_pr.png")
    plt.close(fig)


def plot_confusion(y, pred, out):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if v > cm.max() / 2 else NAVY)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Scam"], yticklabels=["Legit", "Scam"],
           xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png")
    plt.close(fig)


def plot_calibration(y, prob, out):
    frac, mean = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    ax.plot(mean, frac, "o-", color=TEAL, lw=2, label="ScamShield")
    ax.set(title="Calibration", xlabel="Predicted probability", ylabel="Observed scam rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png")
    plt.close(fig)


def plot_features(model, out):
    names = [n.replace("word:", "").replace("char:", "chars:") for n in feature_names(model)]
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    top = pd.concat([coef.nsmallest(12), coef.nlargest(12)]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.barh(top.index, top.values, color=[TEAL if v < 0 else RED for v in top.values])
    ax.set(title="Strongest signals: scam (red) vs legit (teal)", xlabel="Logistic-regression weight")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "top_features.png")
    plt.close(fig)


def plot_comparison(rows, out):
    names = [r["model"].replace(" (ScamShield)", "").replace("TF-IDF + ", "").replace("fraud features + ", "features + ")
             .replace(" (fine-tuned transformer)", "").replace(" (deep learning)", "") for r in rows]
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.55 * len(rows)))
    ypos = np.arange(len(rows))
    ax.barh(ypos, [r["f1"] for r in rows], color=[NAVY if "ScamShield" in r["model"] else TEAL if r["model"].startswith(("BiGRU", "DistilBERT", "Ensemble")) else GREY for r in rows])
    ax.set(yticks=ypos, yticklabels=names, xlim=(0, 1.05), title="Model comparison - F1 on held-out test set")
    ax.invert_yaxis()
    for i, r in enumerate(rows):
        ax.text(r["f1"] + .01, i, f"{r['f1']:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-real", action="store_true", help="ignore datasets in data/raw")
    ap.add_argument("--feedback", action="store_true", help="include user feedback from data/feedback.csv")
    ap.add_argument("--out", default=str(config.MODEL_DIR))
    a = ap.parse_args()

    df = load_all(include_real=not a.no_real, include_feedback=a.feedback)
    train, test = df[df.split == "train"], df[df.split == "test"]
    print(f"train={len(train)}  test={len(test)}  sources={sorted(df.source.unique())}")

    # 1) model comparison on the held-out test set
    comparison = []
    fitted = {}
    for name, kind, feats in CANDIDATES:
        m = build_pipeline(model=kind, use_features=feats).fit(train.text, train.label)
        prob = m.predict_proba(test.text)[:, 1]
        comparison.append({"model": name, **scores(test.label, prob)})
        fitted[name] = m
        print(f"  {name:62s} F1={comparison[-1]['f1']:.4f}  AUC={comparison[-1]['roc_auc']:.4f}")
    model = fitted[CANDIDATES[-1][0]]

    # 2) thresholds from grouped out-of-fold predictions (no test leakage)
    oof = cross_val_predict(build_pipeline("lr", True), train.text, train.label,
                            groups=train.group, cv=GroupKFold(5), method="predict_proba")[:, 1]
    t_susp, t_scam = pick_thresholds(train.label.values, oof)
    cv = scores(train.label.values, oof, t_scam)
    print(f"thresholds: suspicious>={t_susp}  scam>={t_scam}   grouped-CV: {cv}")

    # 3) final evaluation
    prob = model.predict_proba(test.text)[:, 1]
    pred = (prob >= t_scam).astype(int)
    overall = scores(test.label, prob, t_scam)
    flagged = (prob >= t_susp).astype(int)
    flag_stats = {"precision": round(precision_score(test.label, flagged, zero_division=0), 4),
                  "recall": round(recall_score(test.label, flagged, zero_division=0), 4)}
    t = test.assign(prob=prob, pred=pred)
    by_source = {s: {**scores(g.label, g.prob, t_scam), "n": int(len(g))} for s, g in t.groupby("source")
                 if g.label.nunique() == 2}
    by_cat = {c: round(float((g.pred == g.label).mean()), 4) for c, g in t.groupby("category")}
    tn, fp, fn, tp = confusion_matrix(test.label, pred).ravel()
    print("overall (verdict = scam):", overall)
    print("flagged (suspicious or scam):", flag_stats)
    for s, v in by_source.items():
        print(f"  [{s}]", v)

    out = config.MODEL_DIR if a.out == str(config.MODEL_DIR) else __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "scamshield.joblib")
    plot_curves(test.label, prob, out)
    plot_confusion(test.label, pred, out)
    plot_calibration(test.label, prob, out)
    plot_features(model, out)
    deep_rows = []
    if config.DEEP_CARD_PATH.exists():     # keep previously trained deep models in the comparison chart
        deep_rows = [m["row"] for m in json.loads(config.DEEP_CARD_PATH.read_text()).get("models", {}).values()]
    plot_comparison(comparison + deep_rows, out)

    # SHAP: store the reference point and draw global importance plots
    mean = explain.save_background(model, train.text, config.SHAP_BG_PATH if a.out == str(config.MODEL_DIR) else out / "shap_background.joblib")
    explain.global_shap_plots(model, mean, test.text.sample(min(250, len(test)), random_state=0), out)

    card = {
        "name": "ScamShield", "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "algorithm": "Word + character TF-IDF and 19 fraud signals into class-balanced logistic regression",
        "data": {"train": int(len(train)), "test": int(len(test)),
                 "by_source": {s: int(n) for s, n in df.source.value_counts().items()},
                 "scam_ratio_train": round(float(train.label.mean()), 3),
                 "real_data_used": bool((df.source != "synthetic").any() and (df.source != "feedback").any())},
        "thresholds": {"suspicious": t_susp, "scam": t_scam},
        "metrics": {"test": overall, "flagged": flag_stats, "grouped_cv": cv, "by_source": by_source, "by_category_accuracy": by_cat,
                    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
        "model_comparison": comparison,
        "limitations": [
            "Synthetic messages are template-generated; accuracy on them over-states real-world performance.",
            "Public real data is mostly English promotional spam, not Indian banking fraud.",
            "Scammers adapt: retrain regularly using the in-app feedback log.",
            "The tool gives a risk estimate, not proof. Verify via your bank's official channels.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"saved model, model card and plots to {out}")


if __name__ == "__main__":
    main()
