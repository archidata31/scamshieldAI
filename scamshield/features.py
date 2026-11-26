"""Text normalisation, hand-built fraud features and the model pipeline."""
import math
import re

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?<![@\w-])[\w-]+(?:\.[\w-]+)*\.(?:com|in|xyz|top|co|net|org|me|ly|link|click|online|site|icu|live|vip|buzz|cc|app|sbi)\b(?:/\S*)?",
    re.I,
)
SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "t.ly", "goo.gl", "rb.gy", "is.gd",
              "shorturl.at", "ow.ly", "tiny.cc"}
BAD_TLDS = {"xyz", "top", "click", "link", "icu", "live", "buzz", "vip", "online", "site", "cc", "cfd", "sbs"}
OFFICIAL = {"amazon.in", "amazon.com", "flipkart.com", "hdfcbank.com", "sbi.co.in", "onlinesbi.sbi",
            "icicibank.com", "axisbank.com", "swiggy.com", "zomato.com", "irctc.co.in", "myntra.com",
            "google.com", "microsoft.com", "paytm.com", "phonepe.com", "kotak.com", "incometax.gov.in"}
URL_KEYWORDS = re.compile(r"kyc|verify|secure|update|login|signin|claim|refund|reward|bonus|prize|gift|unlock|reactivate|suspend", re.I)

URGENCY_RE = re.compile(
    r"\b(urgent(?:ly)?|immediate(?:ly)?|asap|right now|at once|tonight|within \d+\s*(?:hours?|hrs?|mins?|minutes?)"
    r"|expires?|expired|expiring|final notice|last chance|limited time|hurry|avoid|blocked|suspended|suspension"
    r"|freeze|frozen|deactivated|disconnected|closed|turant|abhi|band ho jayega|block ho (?:gaya|jayega)|arrest)\b", re.I)
SENSITIVE_RE = re.compile(r"\b(otp|pin|cvv|password|passcode|aadhaar|aadhar|pan|card number|atm|code)\b", re.I)
ASK_VERB_RE = re.compile(r"\b(share|enter|send|tell|submit|confirm|verify|provide|type|give|approve|batayein|bhejein|daalein|dalein)\b", re.I)
WARNING_RE = re.compile(r"\b(do not|don't|dont|never)\s+(?:share|disclose|give)\b|\bnot you\?|\bignore if\b|\bshare na karein\b|\bkisi ko na batayein\b", re.I)
REWARD_RE = re.compile(r"\b(won|winner|lottery|lucky|prize|cashback|refund|reward|free gift|gift voucher|congratulations"
                       r"|congrats|selected|shortlisted|pre-?approved|guaranteed|inaam|badhai ho)\b", re.I)
FEE_RE = re.compile(r"\b(processing|registration|clearance|handling|release|advance)\s+fee\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
UPI_RE = re.compile(r"\b[\w.\-]{2,}@[a-z]{2,}\b(?![.\w])", re.I)
MONEY_RE = re.compile(r"(?:rs\.?|₹|inr)\s*[\d,]+", re.I)

FEATURE_NAMES = [
    "n_urls", "shortened_url", "suspicious_tld", "ip_url", "hyphenated_domain", "url_keyword",
    "official_domain", "urgency_terms", "asks_sensitive_info", "safety_warning", "reward_bait",
    "fee_request", "phone_numbers", "upi_id", "money_mention", "upper_ratio", "exclamations",
    "digit_ratio", "log_length",
]


def find_urls(text):
    return [m.group().rstrip(".,;:!)\"'") for m in URL_RE.finditer(text)]


def domain_of(url):
    d = re.sub(r"^https?://", "", url.lower())
    return re.split(r"[/?#]", d)[0].split(":")[0].removeprefix("www.")


def _is_official(dom):
    return any(dom == o or dom.endswith("." + o) for o in OFFICIAL)


def normalize_text(t):
    t = t.lower()
    t = URL_RE.sub(" urltoken ", t)
    t = re.sub(r"\d+", " numx ", t)
    return t


def _one(text):
    urls = find_urls(text)
    doms = [domain_of(u) for u in urls]
    tlds = [d.rsplit(".", 1)[-1] for d in doms]
    n = max(len(text), 1)
    letters = [c for c in text if c.isalpha()]
    return [
        min(len(urls), 3),
        int(any(d in SHORTENERS for d in doms)),
        int(any(t in BAD_TLDS for t in tlds)),
        int(any(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", d) for d in doms)),
        int(any(d.count("-") >= 1 and not _is_official(d) for d in doms)),
        int(any(URL_KEYWORDS.search(u) for u in urls)),
        int(any(_is_official(d) for d in doms)),
        min(len(URGENCY_RE.findall(text)), 4),
        int(bool(SENSITIVE_RE.search(text) and ASK_VERB_RE.search(text) and not WARNING_RE.search(text))),
        int(bool(WARNING_RE.search(text))),
        min(len(REWARD_RE.findall(text)), 3),
        int(bool(FEE_RE.search(text))),
        min(len(PHONE_RE.findall(text)), 2),
        int(bool(UPI_RE.search(text))),
        int(bool(MONEY_RE.search(text))),
        sum(c.isupper() for c in letters) / max(len(letters), 1),
        min(text.count("!"), 3),
        sum(c.isdigit() for c in text) / n,
        math.log1p(len(text)),
    ]


def extract_features(texts):
    return np.array([_one(str(t)) for t in texts], dtype=float)


def rule_reasons(text):
    """Human-readable red / green flags, independent of the ML model."""
    flags = []
    urls = find_urls(text)
    doms = [domain_of(u) for u in urls]
    if any(d in SHORTENERS for d in doms):
        flags.append(("risk", "Shortened link hides the real destination"))
    if any(d.rsplit(".", 1)[-1] in BAD_TLDS for d in doms):
        flags.append(("risk", "Link uses a domain ending commonly seen in scams (.xyz, .top, .click ...)"))
    if any(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", d) for d in doms):
        flags.append(("risk", "Link points to a raw IP address instead of a real website"))
    if any(d.count("-") >= 1 and not _is_official(d) for d in doms):
        flags.append(("risk", "Look-alike domain that imitates a brand or bank"))
    if urls and all(_is_official(d) for d in doms):
        flags.append(("safe", "Link goes to a known official domain"))
    if URGENCY_RE.search(text):
        flags.append(("risk", "Creates urgency or threatens account closure / disconnection"))
    if SENSITIVE_RE.search(text) and ASK_VERB_RE.search(text) and not WARNING_RE.search(text):
        flags.append(("risk", "Asks you to share or enter an OTP, PIN, password or ID details"))
    if WARNING_RE.search(text):
        flags.append(("safe", "Contains a standard warning not to share codes (typical of genuine alerts)"))
    if REWARD_RE.search(text):
        flags.append(("risk", "Prize, cashback or refund bait"))
    if FEE_RE.search(text):
        flags.append(("risk", "Asks for an upfront fee to release money or a prize"))
    if PHONE_RE.search(text):
        flags.append(("risk", "Pushes you to call or message an unknown mobile number"))
    if UPI_RE.search(text) and re.search(r"\b(send|pay|return|transfer)\b", text, re.I):
        flags.append(("risk", "Asks you to send money to a UPI ID"))
    return flags


TYPE_PATTERNS = {
    "Digital-arrest / authority impersonation": r"\barrest\b|\bcbi\b|cyber ?crime|money laundering|illegal (?:items|activity)|narcotics|video call",
    "KYC / account-block scam": r"\bkyc\b|re-?kyc|account (?:will be |is |has been )?(?:blocked|suspended|closed|frozen|on hold)|net ?banking is locked|band ho jayega|unfreeze|card is deactivated",
    "UPI / refund scam": r"collect request|upi pin|cashback|by mistake|scan (?:the |this )?qr|wrong transfer|refund",
    "Lottery / prize scam": r"lottery|lucky|you (?:have )?won|\bwinner\b|free (?:iphone|gift)|gift voucher|inaam|\bkbc\b",
    "Delivery / utility / SIM scam": r"customs|parcel|courier|india post|fastag|electricity|gas connection|sim (?:will be )?deactivated|\btrai\b|disconnected",
    "Fake job / investment / loan": r"work from home|part-?time job|earn rs|per day|returns|stock tips|instant loan|crypto|pre-?approved|\bloan\b|profit",
    "OTP / credential phishing": r"\botp\b|password|\bcvv\b|sign in|login|verify your (?:identity|account)|card upgrade|atm pin|tax refund|income tax",
}


def scam_type(text):
    """Best-guess scam family from keyword evidence (None if nothing matches)."""
    best, best_n = None, 0
    for name, pat in TYPE_PATTERNS.items():
        n = len(re.findall(pat, text, re.I))
        if n > best_n:
            best, best_n = name, n
    return best


def feature_names(model):
    """Names of every column the classifier sees, in order (word:..., char:..., [signal] ...)."""
    names = []
    for n, tr in model.named_steps["feats"].transformer_list:
        if n == "hand":
            names += [f"[signal] {f}" for f in FEATURE_NAMES]
        else:
            names += [f"{n}:{f}" for f in tr.get_feature_names_out()]
    return names


def build_pipeline(model="lr", use_features=True, C=1.0):
    parts = [
        ("word", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True, max_features=60000)),
    ]
    if use_features:
        parts.append(("hand", Pipeline([
            ("f", FunctionTransformer(extract_features)),
            ("s", StandardScaler(with_mean=False)),
        ])))
    clf = {
        "lr": LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
        "svm": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", max_iter=20000), cv=3),
        "nb": ComplementNB(alpha=0.3),
    }[model]
    return Pipeline([("feats", FeatureUnion(parts)), ("clf", clf)])
