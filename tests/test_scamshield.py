import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scamshield import config
from scamshield.app import app
from scamshield.datasets import _read_labelled
from scamshield.features import FEATURE_NAMES, extract_features, find_urls, scam_type
from scamshield.predict import ScamShield
from scamshield.train import pick_thresholds

SCAMS = [
    "Dear customer, your SBI account will be blocked today. Update KYC: http://sbi-kyc.xyz/a1",
    "Congratulations you won a lottery! Pay Rs 999 fee to rahul22@ybl to claim prize",
    "Approve the collect request and enter your UPI PIN to receive refund of Rs 4999",
    "Your Aadhaar is linked to a money laundering case. Call 9876543210 now to avoid arrest",
    "Aapka HDFC account aaj band ho jayega. Turant KYC update karein: http://bit.ly/kyc9Xz",
]
LEGIT = [
    "Rs 2,500 debited from A/c XX4821 on 20-11-2026 at Swiggy. Not you? Call the number on your card.",
    "Your OTP is 482913. Do not share it with anyone. -ICICI Bank",
    "Hi, are we still meeting for coffee at 5?",
    "Your Amazon order has shipped. Track at https://www.amazon.in/gp/your-orders",
    "Kal meeting 11 baje hai, time pe aa jana",
]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = ScamShield()

    def test_scams_flagged(self):
        for m in SCAMS:
            self.assertNotEqual(self.s.check(m)["verdict"], "safe", m)
            self.assertGreaterEqual(self.s.check(m)["probability"], self.s.t_susp, m)

    def test_legit_not_flagged(self):
        for m in LEGIT:
            self.assertEqual(self.s.check(m)["verdict"], "safe", m)

    def test_highlights_cover_text_exactly(self):
        for m in (SCAMS[0], LEGIT[0]):
            r = self.s.check(m)
            self.assertEqual("".join(x["text"] for x in r["segments"]), m)
        self.assertTrue(any(x["level"] == "high" for x in self.s.check(SCAMS[0])["segments"]))

    def test_output_contract(self):
        r = self.s.check(SCAMS[0])
        for k in ("probability", "risk_score", "verdict", "label", "advice", "reasons", "segments", "scam_type"):
            self.assertIn(k, r)
        self.assertEqual(self.s.check("   ")["verdict"], "empty")

    def test_check_many_matches_single(self):
        rows = self.s.check_many(SCAMS + LEGIT)
        self.assertEqual(len(rows), 10)
        for row, m in zip(rows, SCAMS + LEGIT):
            self.assertEqual(row["verdict"], self.s.check(m, explain=False)["verdict"])

    def test_scam_type(self):
        self.assertIn("arrest", scam_type(SCAMS[3]).lower())
        self.assertIn("KYC", scam_type(SCAMS[0]))
        self.assertIsNone(scam_type("see you at lunch"))


class FeatureTests(unittest.TestCase):
    def test_feature_shape(self):
        self.assertEqual(extract_features(["hello", "bit.ly/x"]).shape, (2, len(FEATURE_NAMES)))

    def test_url_detection(self):
        self.assertEqual(find_urls("visit bit.ly/abc123. now"), ["bit.ly/abc123"])
        self.assertEqual(find_urls("mail me at joe@gmail.com"), [])

    def test_thresholds_ordered(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
        lo, hi = pick_thresholds(y, p)
        self.assertLess(lo, hi)


class LoaderTests(unittest.TestCase):
    def test_uci_format(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "sms.tsv"
            f.write_text("label\ttext\nham\tsee you soon friend\nspam\tWINNER claim your prize now\n", encoding="utf-8")
            df = _read_labelled(f)
            self.assertEqual(sorted(df.label), [0, 1])

    def test_kaggle_v1_v2(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "spam.csv"
            f.write_text("v1,v2\nham,hello there my friend\nspam,free entry win cash prize\n", encoding="utf-8")
            self.assertEqual(sorted(_read_labelled(f).label), [0, 1])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_pages_and_health(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")
        self.assertEqual(self.c.get("/plots/not_a_plot.png").status_code, 404)

    def test_check(self):
        r = self.c.post("/api/check", json={"text": SCAMS[0]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["verdict"], "scam")
        self.assertEqual(self.c.post("/api/check", json={"text": ""}).get_json()["verdict"], "empty")

    def test_validation(self):
        self.assertEqual(self.c.post("/api/check", json={"text": 5}).status_code, 400)
        self.assertEqual(self.c.post("/api/check", json={"text": "x" * (config.MAX_TEXT_CHARS + 1)}).status_code, 400)
        self.assertEqual(self.c.post("/api/batch", json={"texts": []}).status_code, 400)
        self.assertEqual(self.c.post("/api/feedback", json={"text": "hi", "label": "maybe"}).status_code, 400)

    def test_batch_csv_upload(self):
        body = "id,text\n1," + '"' + SCAMS[0] + '"' + "\n2," + '"' + LEGIT[2] + '"' + "\n"
        r = self.c.post("/api/batch", data={"file": (io.BytesIO(body.encode()), "m.csv")}, content_type="multipart/form-data")
        d = r.get_json()
        self.assertEqual(d["summary"]["total"], 2)
        self.assertEqual(d["summary"]["scam"], 1)

    def test_feedback_written(self):
        with tempfile.TemporaryDirectory() as d:
            old, config.FEEDBACK_PATH = config.FEEDBACK_PATH, Path(d) / "fb.csv"
            try:
                r = self.c.post("/api/feedback", json={"text": "hello", "label": "legit", "predicted": "safe"})
                self.assertEqual(r.status_code, 200)
                rows = list(csv.reader(config.FEEDBACK_PATH.open(encoding="utf-8")))
                self.assertEqual(rows[1][1:], ["hello", "safe", "legit"])
            finally:
                config.FEEDBACK_PATH = old

    def test_model_card(self):
        card = self.c.get("/api/model").get_json()
        self.assertIn("thresholds", card)
        self.assertIn("model_comparison", card)

    def test_security_headers(self):
        h = self.c.get("/").headers
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        self.assertIn("Content-Security-Policy", h)


if __name__ == "__main__":
    unittest.main()
