"""Model explanations: LIME (any model) and SHAP (exact values for the logistic-regression model).

LIME   - perturbs the message (drops words), asks the model what it thinks, and fits a small local
         linear model. Works for logistic regression, BiGRU, DistilBERT and the ensemble.
SHAP   - for a linear model with independent features the Shapley value of feature j is exactly
             phi_j = coef_j * (x_j - E[x_j])
         which is what shap.LinearExplainer (interventional) computes. We evaluate it directly, so
         explanations are instant, and base_value + sum(phi) equals the model's log-odds exactly
         (checked in the tests). The optional `shap` package is used only for the beeswarm plot.
"""
import joblib
import numpy as np

from . import config
from .features import FEATURE_NAMES, feature_names

SIGNAL_LABELS = {
    "n_urls": "number of links", "shortened_url": "shortened link", "suspicious_tld": "suspicious domain ending",
    "ip_url": "link to a raw IP address", "hyphenated_domain": "look-alike domain", "url_keyword": "scam keyword inside link",
    "official_domain": "official domain", "urgency_terms": "urgency words", "asks_sensitive_info": "asks for OTP / PIN / ID",
    "safety_warning": "'do not share' warning", "reward_bait": "prize / refund bait", "fee_request": "upfront fee request",
    "phone_numbers": "phone number", "upi_id": "UPI ID", "money_mention": "money amount", "upper_ratio": "CAPITAL letters",
    "exclamations": "exclamation marks", "digit_ratio": "share of digits", "log_length": "message length",
}


class MissingDependency(RuntimeError):
    """An optional package (lime / shap) is not installed."""


def describe_feature(name):
    if name.startswith("[signal] "):
        key = name[9:]
        return "signal", SIGNAL_LABELS.get(key, key.replace("_", " "))
    if name.startswith("word:"):
        return "word", f"word “{name[5:]}”"
    return "char", f"characters “{name[5:]}”"


# ---------------------------------------------------------------- LIME
def lime_explain(proba_fn, text, num_features=10, num_samples=300):
    """proba_fn(list[str]) -> array (n, 2) of [P(legit), P(scam)]. Returns [{word, weight}] toward 'scam'."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as e:
        raise MissingDependency("LIME is not installed. Run: pip install lime") from e
    explainer = LimeTextExplainer(class_names=["legit", "scam"], random_state=42)
    exp = explainer.explain_instance(text, proba_fn, num_features=num_features,
                                     num_samples=num_samples, labels=(1,))
    return [{"word": w, "weight": round(float(v), 4)} for w, v in exp.as_list(label=1)]


# ---------------------------------------------------------------- SHAP (linear, exact)
class LinearShap:
    def __init__(self, model, mean):
        self.union = model.named_steps["feats"]
        clf = model.named_steps["clf"]
        self.coef = clf.coef_[0]
        self.mean = np.asarray(mean, dtype=float)
        self.names = feature_names(model)
        self.base = float(clf.intercept_[0] + self.coef @ self.mean)   # expected log-odds

    @classmethod
    def load(cls, model, path=config.SHAP_BG_PATH):
        if not path.exists():
            raise FileNotFoundError("SHAP background missing. Retrain with: python -m scamshield.train")
        return cls(model, joblib.load(path)["mean"])

    def values(self, text):
        x = self.union.transform([text]).toarray().ravel()
        return self.coef * (x - self.mean), x

    def explain(self, text, top=10):
        sv, x = self.values(text)
        kinds = [describe_feature(n)[0] for n in self.names]
        is_char = np.array([k == "char" for k in kinds])
        main_idx = np.where(~is_char & (np.abs(sv) > 1e-9))[0]
        order = main_idx[np.argsort(-np.abs(sv[main_idx]))]
        shown = order[:top]
        rows = [{"feature": describe_feature(self.names[i])[1], "kind": kinds[i], "shap": round(float(sv[i]), 4)}
                for i in shown]
        rest = float(sv[order[top:]].sum()) if len(order) > top else 0.0
        if abs(rest) > 1e-6:
            rows.append({"feature": "all other words and signals (combined)", "kind": "other", "shap": round(rest, 4)})
        char_total = float(sv[is_char].sum())
        rows.append({"feature": "character patterns (combined)", "kind": "char", "shap": round(char_total, 4)})
        logit = self.base + float(sv.sum())
        return {"method": "shap", "base_logit": round(self.base, 4), "logit": round(logit, 4),
                "probability": round(float(1 / (1 + np.exp(-logit))), 4), "contributions": rows}


def save_background(model, train_texts, path=config.SHAP_BG_PATH):
    """Store the training-set mean of every model column (the SHAP reference point)."""
    X = model.named_steps["feats"].transform(list(train_texts))
    mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mean": mean}, path)
    return mean


def global_shap_plots(model, mean, texts, out_dir, top=20, sample=250, chunk=50):
    """Mean |SHAP| bar chart (always) and a SHAP beeswarm (when the `shap` package is installed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    texts = list(texts)[:sample]
    ls = LinearShap(model, mean)
    X = ls.union.transform(texts)

    def chunks():
        for i in range(0, X.shape[0], chunk):
            yield i, ls.coef * (X[i:i + chunk].toarray() - ls.mean)

    total = np.zeros(len(ls.coef))
    for _, sv in chunks():
        total += np.abs(sv).sum(axis=0)
    mean_abs = total / X.shape[0]
    kinds = np.array([describe_feature(n)[0] for n in ls.names])
    cand = np.where(kinds != "char")[0]
    sel = cand[np.argsort(-mean_abs[cand])[:top]]
    labels = [describe_feature(ls.names[i])[1] for i in sel]

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.barh(labels[::-1], mean_abs[sel][::-1], color="#0f9d8f")
    ax.set(title="SHAP: average impact of each word / signal", xlabel="mean |SHAP value| (log-odds)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_importance.png", dpi=130)
    plt.close(fig)

    try:   # optional beeswarm using the shap package
        import shap
        sv_top = np.vstack([sv[:, sel] for _, sv in chunks()])
        x_top = np.vstack([X[i:i + chunk].toarray()[:, sel] for i in range(0, X.shape[0], chunk)])
        shap.summary_plot(sv_top, x_top, feature_names=labels, show=False, max_display=top)
        plt.gcf().set_size_inches(8, 6.5)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_beeswarm.png", dpi=130)
        plt.close("all")
        print("saved SHAP beeswarm (shap package)")
    except ImportError:
        print("note: `shap` package not installed - skipped beeswarm plot (pip install shap)")
    except Exception as e:  # noqa: BLE001 - plotting must never break training
        print(f"note: beeswarm plot skipped ({type(e).__name__}: {e})")
    return list(zip(labels, mean_abs[sel].round(4).tolist()))
