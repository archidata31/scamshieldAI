"""Inference, model selection and explanations.

Models:  lr (fast, default) | bigru | distilbert | ensemble (average of all available)
Word highlighting is always computed with the logistic-regression model (fast, in log-odds); LIME and
SHAP are available on demand through ScamShield.explain().
"""
import json
import re
import sys
import time
from collections import OrderedDict

import joblib
import numpy as np

from . import __version__, config, deep, explain
from .features import rule_reasons, scam_type

HIGH, MED, SAFE = 1.0, 0.5, -0.35          # log-odds change needed to highlight a word
MAX_TOKENS = 200
CACHE_SIZE = 512
MODEL_IDS = ("lr", "bigru", "distilbert", "ensemble")
MODEL_LABELS = {"lr": "Logistic Regression (fast)", "bigru": "BiGRU", "distilbert": "DistilBERT",
                "ensemble": "Ensemble (average)"}

ADVICE = {
    "scam": ["Do not click the link, call the number or share any OTP, PIN or CVV.",
             "Contact your bank only through its official app or the number printed on your card.",
             "Report it: call 1930 or visit cybercrime.gov.in. Forward SMS scams to 1909."],
    "suspicious": ["Do not act on this message yet.",
                   "Verify with the sender through the official app, website or helpline.",
                   "If it asks for money, codes or personal details, treat it as a scam."],
    "safe": ["No strong scam indicators were found.",
             "Still, never share an OTP, PIN or CVV with anyone, even someone claiming to be your bank."],
}
LABELS = {"scam": "Likely scam", "suspicious": "Suspicious", "safe": "Likely safe"}


class ModelUnavailable(ValueError):
    """Requested model cannot be used (not trained / package missing / unknown)."""


class ScamShield:
    def __init__(self, model_path=config.MODEL_PATH, card_path=config.CARD_PATH):
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run: python -m scamshield.data_gen && python -m scamshield.train")
        self.model = joblib.load(model_path)
        self.card = json.loads(card_path.read_text()) if card_path.exists() else {}
        th = self.card.get("thresholds", {})
        self.t_susp = th.get("suspicious", config.DEFAULT_SUSPICIOUS)
        self.t_scam = th.get("scam", config.DEFAULT_SCAM)
        self._cache = OrderedDict()
        self._scorers = {}            # loaded deep scorers (also the injection point for tests)
        self._shap = None

    # -- model registry ----------------------------------------------------
    def models_info(self):
        info = [{"id": "lr", "label": MODEL_LABELS["lr"], "available": True, "message": "Ready"}]
        for name in ("bigru", "distilbert"):
            st = deep.status(name)
            if name in self._scorers:
                st = {**st, "available": True, "message": "Ready"}
            info.append({"id": name, "label": MODEL_LABELS[name], "available": st["available"], "message": st["message"]})
        n_deep = sum(m["available"] for m in info[1:])
        info.append({"id": "ensemble", "label": MODEL_LABELS["ensemble"], "available": n_deep > 0,
                     "message": "Ready" if n_deep else "Needs at least one trained deep model"})
        return info

    def available(self):
        return [m["id"] for m in self.models_info() if m["available"]]

    def scorer(self, name):
        if name == "lr":
            return self.model
        if name not in ("bigru", "distilbert"):
            raise ModelUnavailable(f"Unknown model '{name}'.")
        if name not in self._scorers:
            try:
                self._scorers[name] = deep.load_scorer(name)
            except deep.DeepUnavailable as e:
                raise ModelUnavailable(f"{MODEL_LABELS[name]} is not available. {e}") from e
        return self._scorers[name]

    def members(self):
        """Models that take part in the ensemble."""
        return [n for n in ("lr", "bigru", "distilbert") if n in self.available()]

    def proba_by_model(self, texts, name):
        """P(scam) for each text under the chosen model; returns a 1-D numpy array."""
        texts = list(texts)
        if name == "ensemble":
            members = self.members()
            if len(members) < 2:
                raise ModelUnavailable("Ensemble needs at least one trained deep model. "
                                       "Run: python -m scamshield.train_deep --model bigru")
            return np.mean([self.scorer(m).predict_proba(texts)[:, 1] for m in members], axis=0)
        return self.scorer(name).predict_proba(texts)[:, 1]

    def proba2(self, name):
        """Function texts -> (n, 2) probabilities, the format LIME expects."""
        def fn(texts):
            p = self.proba_by_model(list(texts), name)
            return np.column_stack([1 - p, p])
        return fn

    def logit(self, texts):
        return self.model.decision_function(texts)

    def thresholds(self, name):
        return (self.t_susp, self.t_scam) if name == "lr" else (config.DEFAULT_SUSPICIOUS, config.DEFAULT_SCAM)

    def band(self, p, name="lr"):
        lo, hi = self.thresholds(name)
        return "scam" if p >= hi else "suspicious" if p >= lo else "safe"

    # -- public API ----------------------------------------------------------
    def check(self, text, explain=True, model="lr"):
        text = (text or "").strip()
        if not text:
            return {"verdict": "empty", "label": "Nothing to analyse", "probability": 0.0, "risk_score": 0,
                    "segments": [], "reasons": [], "advice": [], "scam_type": None, "votes": {}, "model": model}
        key = (text, explain, model)
        if key in self._cache:
            self._cache.move_to_end(key)
            return dict(self._cache[key])
        t0 = time.perf_counter()
                wanted = self.members() if model == "ensemble" else [model]
        raw = {m: float(self.proba_by_model([text], m)[0]) for m in wanted}
        if model == "ensemble":
            if len(raw) < 2:
                raise ModelUnavailable("Ensemble needs at least one trained deep model. "
                                       "Run: python -m scamshield.train_deep --model bigru")
            p = float(np.mean(list(raw.values())))
        elif model in raw:
            p = raw[model]
        else:
            p = float(self.proba_by_model([text], model)[0])      # raises ModelUnavailable with a helpful message
        votes = {m: round(v, 4) for m, v in raw.items()}
        verdict = self.band(p, model)
        out = {
            "probability": round(p, 4), "risk_score": int(round(p * 100)), "verdict": verdict,
            "label": LABELS[verdict], "scam_type": scam_type(text) if verdict != "safe" else None,
            "advice": ADVICE[verdict], "model": model, "model_label": MODEL_LABELS[model], "votes": votes,
            "reasons": [{"kind": k, "text": t} for k, t in rule_reasons(text)],
            "segments": self.highlight(text, verdict) if explain else [{"text": text, "level": "none", "score": 0.0}],
            "thresholds": dict(zip(("suspicious", "scam"), self.thresholds(model))),
            "model_version": __version__,
        }
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._cache[key] = out
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return dict(out)

    def check_many(self, texts, model="lr"):
        """Fast bulk scoring (no per-word explanations)."""
        clean = [(t or "").strip() for t in texts]
        probs = self.proba_by_model([t or "." for t in clean], model)
        rows = []
        for t, p in zip(clean, probs):
            v = self.band(float(p), model) if t else "empty"
            rows.append({"text": t, "risk_score": int(round(float(p) * 100)) if t else 0, "verdict": v,
                         "label": LABELS.get(v, "Empty"), "scam_type": scam_type(t) if v in ("scam", "suspicious") else None})
        return rows

    def highlight(self, text, verdict="scam"):
        spans = list(re.finditer(r"\S+", text))[:MAX_TOKENS]
        variants = [(text[:m.start()] + " " + text[m.end():]) for m in spans]
        base = float(self.logit([text])[0])
        scores = [base - float(q) for q in self.logit(variants)] if variants else []
        segs, pos = [], 0
        for m, s in zip(spans, scores):
            if m.start() > pos:
                segs.append({"text": text[pos:m.start()], "level": "none", "score": 0.0})
            level = "high" if s >= HIGH else "med" if s >= MED else "safe" if s <= SAFE else "none"
            if verdict == "safe" and level == "med":      # keep genuine messages visually calm
                level = "none"
            segs.append({"text": m.group(), "level": level, "score": round(s, 3)})
            pos = m.end()
        if pos < len(text):
            segs.append({"text": text[pos:], "level": "none", "score": 0.0})
        return segs

    def explain(self, text, method="shap", model="lr"):
        """LIME (any model) or exact SHAP (logistic-regression model)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Nothing to explain.")
        if method == "shap":
            if model != "lr":
                raise ModelUnavailable("SHAP is computed for the logistic-regression model. "
                                       "Use LIME to explain BiGRU, DistilBERT or the ensemble.")
            if self._shap is None:
                self._shap = explain.LinearShap.load(self.model)
            out = self._shap.explain(text)
        elif method == "lime":
            if model in ("bigru", "distilbert"):
                self.scorer(model)                      # fail early with a clear message if unavailable
            n = 300 if model == "lr" else 150          # deep models are slower: fewer perturbations
            out = {"method": "lime", "contributions": explain.lime_explain(self.proba2(model), text, num_samples=n)}
        else:
            raise ValueError("method must be 'lime' or 'shap'")
        out["model"] = model
        return out


if __name__ == "__main__":
    shield = ScamShield()
    for msg in sys.argv[1:] or ["Your SBI account is blocked. Update KYC now http://sbi-kyc.xyz/a1"]:
        r = shield.check(msg)
        print(f"{r['label']} ({r['risk_score']}/100) {r['scam_type'] or ''} :: {msg}")
        for f in r["reasons"]:
            print(f"   [{f['kind']}] {f['text']}")
