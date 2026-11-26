"""Download public datasets into data/raw/ (run this on your own machine).

    python -m scamshield.download_data                 # UCI SMS Spam Collection (5,574 real SMS)
    python -m scamshield.download_data --emails        # + SpamAssassin public email corpus (~4,600 real emails)
    python -m scamshield.download_data --insecure      # only if a corporate proxy breaks TLS verification

For modern *phishing* emails, also download the Kaggle "Phishing Email Dataset" CSV (login required)
and drop it in data/raw/ - columns "Email Text" and "Email Type" are recognised automatically.
"""
import argparse
import csv
import io
import re
import ssl
import sys
import tarfile
import urllib.error
import urllib.request
from email import policy
from email.parser import BytesParser

from . import config

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
CORPUS = "https://spamassassin.apache.org/old/publiccorpus/"
EMAIL_ARCHIVES = [  # (file, label) label 1 = spam/phishing, 0 = legitimate
    ("20030228_easy_ham.tar.bz2", 0), ("20030228_hard_ham.tar.bz2", 0),
    ("20030228_spam.tar.bz2", 1), ("20050311_spam_2.tar.bz2", 1),
]
MAX_EMAIL_BYTES = 400_000
MAX_EMAIL_CHARS = 1500


def fetch(url, insecure):
    ctx = ssl._create_unverified_context() if insecure else None
    req = urllib.request.Request(url, headers={"User-Agent": "ScamShield/3.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return r.read()


# ---------------------------------------------------------------- emails
def parse_email(raw: bytes) -> str:
    """Subject + plain-text body of a raw RFC-822 message (HTML stripped, quotes removed)."""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        subject = str(msg.get("subject", "") or "")
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part is not None else ""
        if part is not None and part.get_content_type() == "text/html":
            body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", body)
            body = re.sub(r"<[^>]+>", " ", body)
    except Exception:  # noqa: BLE001 - malformed old mail is common; skip it
        return ""
    body = "\n".join(l for l in str(body).splitlines() if not l.lstrip().startswith(">"))
    text = re.sub(r"\s+", " ", f"{subject}. {body}").strip()
    return text[:MAX_EMAIL_CHARS]


def read_corpus_archive(data: bytes, label: int):
    """Read a .tar.bz2 of raw emails entirely in memory (nothing is extracted to disk)."""
    rows = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.size > MAX_EMAIL_BYTES or m.name.rsplit("/", 1)[-1] == "cmds":
                continue
            raw = tf.extractfile(m).read()
            text = parse_email(raw)
            if len(text) > 40:
                rows.append((text, label))
    return rows


def download_emails(insecure):
    rows = []
    for name, label in EMAIL_ARCHIVES:
        print(f"  downloading {name} ...")
        rows += read_corpus_archive(fetch(CORPUS + name, insecure), label)
    out = config.RAW_DIR / "emails_spamassassin.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    spam = sum(l for _, l in rows)
    print(f"saved {len(rows)} emails ({spam} spam, {len(rows) - spam} legitimate) to {out}")


# ---------------------------------------------------------------- SMS
def download_sms(insecure):
    out = config.RAW_DIR / "sms_spam.tsv"
    import zipfile
    try:
        print("Downloading UCI SMS Spam Collection ...")
        z = zipfile.ZipFile(io.BytesIO(fetch(UCI_URL, insecure)))
        out.write_text("label\ttext\n" + z.read("SMSSpamCollection").decode("latin-1"), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - try the mirror before giving up
        print(f"  primary source failed ({type(e).__name__}); trying mirror ...")
        out.write_text("label\ttext\n" + fetch(MIRROR_URL, insecure).decode("utf-8", "replace"), encoding="utf-8")
    print(f"saved {sum(1 for _ in out.open(encoding='utf-8')) - 1} messages to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    ap.add_argument("--emails", action="store_true", help="also download the SpamAssassin email corpus")
    ap.add_argument("--emails-only", action="store_true", help="download only the email corpus")
    a = ap.parse_args()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not a.emails_only:
            download_sms(a.insecure)
        if a.emails or a.emails_only:
            download_emails(a.insecure)
    except (urllib.error.URLError, ssl.SSLError, OSError) as e:
        print(f"\nCould not download automatically: {e}")
        print("Manual option: save the files into data/raw/ yourself:\n"
              "  SMS    https://archive.ics.uci.edu/dataset/228/sms+spam+collection (file SMSSpamCollection -> sms_spam.tsv, first line 'label<TAB>text')\n"
              f"  Emails {CORPUS} (see scamshield/download_data.py for the file names)")
        if "CERTIFICATE" in str(e).upper():
            print("TLS error: retry with --insecure")
        sys.exit(1)


if __name__ == "__main__":
    main()
