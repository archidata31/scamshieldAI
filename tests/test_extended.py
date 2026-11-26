"""Tests for email data, deep-model plumbing, ensemble, LIME and SHAP.

PyTorch / LIME are not required: they are replaced by small stand-ins so the logic around them is tested
on any machine. (Training the real BiGRU / DistilBERT needs `pip install -r requirements-deep.txt`.)
"""
import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scamshield import app as app_module
from scamshield import config, deep, explain
from scamshield.app import app
from scamshield.download_data import parse_email, read_corpus_archive
from scamshield.predict import ModelUnavailable, ScamShield

SCAM = "Dear customer, your SBI account will be blocked today. Update KYC: http://sbi-kyc.xyz/a1"
HAM = "Hi, are we still meeting for coffee at 5?"


class FakeScorer:
    """Stands in for a BiGRU / DistilBERT scorer: constant probability for every text."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, texts):
        n = len(list(texts))
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class EmailTests(unittest.TestCase):
    def test_parse_plain_and_html(self):
        plain = b"Subject: Verify now\nContent-Type: text/plain\n\nHello\n> quoted\nClick http://x.xyz/a to verify.\n"
        html = b"Subject: Prize\nMIME-Version: 1.0\nContent-Type: text/html\n\n<html><style>p{}</style><p>You <b>won</b></p></html>\n"
        t1, t2 = parse_email(plain), parse_email(html)
        self.assertIn("Verify now", t1)
        self.assertNotIn("quoted", t1)
        self.assertIn("You won", t2)
        self.assertNotIn("<", t2)

    def test_archive_in_memory(self):
        buf = io.BytesIO()
        body = b"Subject: Win a prize today\n\n" + b"Claim your lottery reward now. " * 5
        with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
            for name, data in [("spam/0001", body), ("spam/cmds", b"junk")]:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        rows = read_corpus_archive(buf.getvalue(), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 1)


class DeepHelpersTests(unittest.TestCase):
    def test_tokenize_and_vocab(self):
        toks = deep.tokenize("URGENT! Pay Rs 999 at http://bit.ly/x now")
        self.assertIn("urltoken", toks)
        self.assertIn("numx", toks)
        vocab = deep.build_vocab(["pay now", "pay later", "pay now please"], min_freq=2)
        self.assertEqual(vocab["<pad>"], 0)
        self.assertIn("pay", vocab)
        self.assertNotIn("later", vocab)          # below min_freq

    def test_encode_pad(self):
        vocab = {"<pad>": 0, "<unk>": 1, "pay": 2}
        self.assertEqual(deep.encode("pay zzz", vocab), [2, 1])
        self.assertEqual(deep.encode("", vocab), [1])
        self.assertEqual(deep.pad_batch([[2, 1], [2]]), [[2, 1], [2, 0]])
        self.assertEqual(len(deep.encode("pay " * 500, vocab)), deep.MAX_LEN_GRU)

    def test_status_messages(self):
        for name in ("bigru", "distilbert"):
            st = deep.status(name)
            self.assertEqual(set(st), {"name", "trained", "runtime", "available", "message"})
        with self.assertRaises(ValueError):
            deep.status("nope")


class ShapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = ScamShield()

    def test_additivity_exact(self):
        ls = explain.LinearShap.load(self.s.model)
        for t in (SCAM, HAM):
            sv, _ = ls.values(t)
            self.assertAlmostEqual(ls.base + sv.sum(), float(self.s.model.decision_function([t])[0]), places=8)

    def test_explain_structure(self):
        out = self.s.explain(SCAM, "shap")
        self.assertEqual(out["method"], "shap")
        self.assertTrue(out["contributions"])
        self.assertAlmostEqual(out["probability"], self.s.check(SCAM)["probability"], places=3)
        self.assertTrue(any(c["shap"] > 0 for c in out["contributions"]))

    def test_shap_only_for_lr(self):
        with self.assertRaises(ModelUnavailable):
            self.s.explain(SCAM, "shap", model="bigru")


class LimeTests(unittest.TestCase):
    def fake_lime(self):
        calls = {}

        class Exp:
            def as_list(self, label=1):
                return [("kyc", 0.4123), ("hello", -0.1)]

        class Explainer:
            def __init__(self, **kw):
                calls["init"] = kw

            def explain_instance(self, text, fn, num_features, num_samples, labels):
                out = fn(["a b", text])
                calls["shape"] = out.shape
                calls["labels"] = labels
                return Exp()

        lime = types.ModuleType("lime")
        lt = types.ModuleType("lime.lime_text")
        lt.LimeTextExplainer = Explainer
        lime.lime_text = lt
        return {"lime": lime, "lime.lime_text": lt}, calls

    def test_lime_plumbing_lr_and_deep(self):
        mods, calls = self.fake_lime()
        s = ScamShield()
        s._scorers["bigru"] = FakeScorer(0.9)
        with mock.patch.dict(sys.modules, mods):
            r = s.explain(SCAM, "lime", model="lr")
            self.assertEqual(r["contributions"][0], {"word": "kyc", "weight": 0.4123})
            self.assertEqual(calls["shape"], (2, 2))
            self.assertEqual(calls["labels"], (1,))
            r2 = s.explain(SCAM, "lime", model="bigru")
            self.assertEqual(r2["model"], "bigru")

    def test_lime_missing_package(self):
        with mock.patch.dict(sys.modules, {"lime": None, "lime.lime_text": None}):
            with self.assertRaises(explain.MissingDependency):
                ScamShield().explain(SCAM, "lime")


class EnsembleTests(unittest.TestCase):
    def test_votes_and_ensemble(self):
        s = ScamShield()
        s._scorers["bigru"] = FakeScorer(0.9)
        s._scorers["distilbert"] = FakeScorer(0.3)
        self.assertEqual(s.members(), ["lr", "bigru", "distilbert"])
        r = s.check(HAM, model="ensemble")
        self.assertEqual(set(r["votes"]), {"lr", "bigru", "distilbert"})
        expected = np.mean([s.proba_by_model([HAM], "lr")[0], 0.9, 0.3])
        self.assertAlmostEqual(r["probability"], round(float(expected), 4), places=3)
        self.assertEqual(s.check(HAM, model="bigru")["probability"], 0.9)
        self.assertEqual(s.check(HAM, model="bigru")["verdict"], "scam")      # deep models use default thresholds
        rows = s.check_many([SCAM, HAM], model="ensemble")
        self.assertEqual(len(rows), 2)

    def test_unavailable_deep_model_message(self):
        s = ScamShield()
        s._scorers.clear()
        if not deep.status("bigru")["available"]:
            with self.assertRaises(ModelUnavailable):
                s.check(HAM, model="bigru")
            with self.assertRaises(ModelUnavailable):
                s.check(HAM, model="ensemble")
        with self.assertRaises(ModelUnavailable):
            s.scorer("gpt")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_models_endpoint(self):
        ids = [m["id"] for m in self.c.get("/api/models").get_json()["models"]]
        self.assertEqual(ids, ["lr", "bigru", "distilbert", "ensemble"])

    def test_bad_model_rejected(self):
        self.assertEqual(self.c.post("/api/check", json={"text": HAM, "model": "zzz"}).status_code, 400)
        self.assertEqual(self.c.post("/api/explain", json={"text": HAM, "method": "nope"}).status_code, 400)
        self.assertEqual(self.c.post("/api/explain", json={"text": ""}).status_code, 400)

    def test_check_with_injected_deep_model(self):
        shield = app_module.shield()
        shield._scorers["bigru"] = FakeScorer(0.97)
        try:
            r = self.c.post("/api/check", json={"text": HAM, "model": "bigru"}).get_json()
            self.assertEqual(r["model"], "bigru")
            self.assertIn("bigru", r["votes"])
            self.assertEqual(r["verdict"], "scam")
        finally:
            shield._scorers.pop("bigru", None)
            shield._cache.clear()

    def test_explain_shap_and_lime_missing(self):
        r = self.c.post("/api/explain", json={"text": SCAM, "method": "shap"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["contributions"])
        with mock.patch.dict(sys.modules, {"lime": None, "lime.lime_text": None}):
            r = self.c.post("/api/explain", json={"text": SCAM, "method": "lime"})
            self.assertEqual(r.status_code, 501)
            self.assertIn("pip install lime", r.get_json()["error"])

    def test_model_card_merges_deep_results(self):
        with tempfile.TemporaryDirectory() as d:
            old, config.DEEP_CARD_PATH = config.DEEP_CARD_PATH, Path(d) / "deep_card.json"
            try:
                row = {"model": "BiGRU (deep learning)", "precision": .9, "recall": .9, "f1": .9, "roc_auc": .95, "pr_auc": .9}
                config.DEEP_CARD_PATH.write_text(json.dumps({"models": {"bigru": {"row": row, "by_source": {}, "training": {}}},
                                                              "ensemble": {"row": {**row, "model": "Ensemble (LR + deep models)"}}}))
                names = [m["model"] for m in self.c.get("/api/model").get_json()["model_comparison"]]
                self.assertIn("BiGRU (deep learning)", names)
                self.assertIn("Ensemble (LR + deep models)", names)
            finally:
                config.DEEP_CARD_PATH = old

    def test_new_plots_whitelisted(self):
        self.assertEqual(self.c.get("/plots/shap_importance.png").status_code, 200)
        self.assertEqual(self.c.get("/plots/secret.png").status_code, 404)


if __name__ == "__main__":
    unittest.main()
