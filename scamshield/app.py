"""ScamShield web app and JSON API.

    python -m scamshield.app -> http://127.0.0.1:5000
"""

import csv
import hmac
import io
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, request, send_from_directory

from . import __version__, config, storage
from .explain import MissingDependency
from .predict import MODEL_IDS, ModelUnavailable, ScamShield


log = logging.getLogger("scamshield")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

STATIC = config.ROOT / "scamshield" / "static"

app = Flask(
    __name__,
    static_folder=str(STATIC),
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Initialize SQLite database
storage.init_db()

_shield, _lock = None, threading.Lock()
_hits = defaultdict(deque)
_fb_lock = threading.Lock()

PLOTS = {
    "roc_pr",
    "confusion_matrix",
    "calibration",
    "top_features",
    "model_comparison",
    "shap_importance",
    "shap_beeswarm"
}


def shield():
    global _shield
    with _lock:
        if _shield is None:
            _shield = ScamShield()
    return _shield


@app.before_request
def rate_limit():
    if not request.path.startswith("/api/"):
        return None

    now, q = time.time(), _hits[request.remote_addr]

    while q and now - q[0] > 60:
        q.popleft()

    if len(q) >= config.RATE_LIMIT_PER_MIN:
        return jsonify(error="Too many requests, slow down."), 429

    q.append(now)
    return None


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'"
    )

    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"

    return resp


@app.errorhandler(FileNotFoundError)
def no_model(e):
    return jsonify(error=str(e)), 503


@app.errorhandler(ModelUnavailable)
def model_unavailable(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(MissingDependency)
def missing_dep(e):
    return jsonify(error=str(e)), 501


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File too large (max 2 MB)."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/plots/<name>.png")
def plot(name):
    if name not in PLOTS:
        abort(404)

    return send_from_directory(
        config.MODEL_DIR,
        f"{name}.png",
        max_age=300
    )


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        version=__version__,
        model_loaded=_shield is not None
    )


def full_card():
    """Merge base model card and deep-learning results."""

    card = dict(shield().card)

    if config.DEEP_CARD_PATH.exists():
        deep = json.loads(config.DEEP_CARD_PATH.read_text())

        rows = [
            m["row"]
            for m in deep.get("models", {}).values()
        ]

        if deep.get("ensemble"):
            rows.append(deep["ensemble"]["row"])

        card["model_comparison"] = (
            list(card.get("model_comparison", [])) + rows
        )

        card["deep"] = {
            k: {
                "training": v.get("training"),
                "by_source": v.get("by_source"),
                "trained_at": v.get("trained_at")
            }
            for k, v in deep.get("models", {}).items()
        }

    card["plots"] = sorted(
        n for n in PLOTS
        if (config.MODEL_DIR / f"{n}.png").exists()
    )

    return card


@app.get("/api/model")
def model_card():
    return jsonify(full_card())


@app.get("/api/models")
def models():
    return jsonify(models=shield().models_info())


@app.post("/api/check")
def check():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify(error="'text' must be a string."), 400

    if len(text) > config.MAX_TEXT_CHARS:
        return jsonify(
            error=f"Message too long (max {config.MAX_TEXT_CHARS} characters)."
        ), 400

    model = data.get("model", "lr")

    if model not in MODEL_IDS:
        return jsonify(
            error=f"Unknown model '{model}'. Choose one of: {', '.join(MODEL_IDS)}."
        ), 400

    result = shield().check(text, model=model)

    storage.save_scan(text, result, model)

    return jsonify(result)


@app.post("/api/explain")
def explain_message():
    """LIME or SHAP explanation of one message."""

    data = request.get_json(silent=True) or {}

    text = data.get("text", "")
    method = data.get("method", "shap")
    model = data.get("model", "lr")

    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' is required."), 400

    if len(text) > config.MAX_TEXT_CHARS:
        return jsonify(
            error=f"Message too long (max {config.MAX_TEXT_CHARS} characters)."
        ), 400

    if method not in ("lime", "shap") or model not in MODEL_IDS:
        return jsonify(
            error="method must be 'lime' or 'shap'; model one of "
            + ", ".join(MODEL_IDS)
        ), 400

    return jsonify(
        shield().explain(text, method=method, model=model)
    )


@app.post("/api/batch")
def batch():
    """Accept CSV upload or JSON texts and return scored rows."""

    texts = []

    if "file" in request.files:
        raw = request.files["file"].read().decode(
            "utf-8-sig", "replace"
        )

        csv_rows = list(csv.reader(io.StringIO(raw)))

        if not csv_rows:
            return jsonify(error="The file is empty."), 400

        head = [c.strip().lower() for c in csv_rows[0]]

        col = next(
            (
                head.index(c)
                for c in ("text", "message", "sms", "body")
                if c in head
            ),
            None
        )

        body = csv_rows[1:] if col is not None else csv_rows
        col = 0 if col is None else col

        texts = [
            r[col]
            for r in body
            if len(r) > col and r[col].strip()
        ]

    else:
        payload = request.get_json(silent=True) or {}

        texts = [
            t for t in payload.get("texts", [])
            if isinstance(t, str) and t.strip()
        ]

    if not texts:
        return jsonify(
            error="No messages found. Use a CSV with a 'text' column, one message per row."
        ), 400

    truncated = len(texts) > config.MAX_BATCH_ROWS

    texts = [
        t[:config.MAX_TEXT_CHARS]
        for t in texts[:config.MAX_BATCH_ROWS]
    ]

    payload = request.get_json(silent=True) or {}

    model = request.form.get("model") or payload.get("model", "lr")

    if model not in MODEL_IDS:
        return jsonify(error=f"Unknown model '{model}'."), 400

    rows = shield().check_many(texts, model=model)

    # Save every bulk scan in SQLite
    for r in rows:
        storage.save_scan(r["text"], r, model)

    for i, r in enumerate(rows, 1):
        r["index"] = i

    summary = {
        v: sum(r["verdict"] == v for r in rows)
        for v in ("scam", "suspicious", "safe")
    }

    return jsonify(
        rows=rows,
        summary={**summary, "total": len(rows)},
        truncated=truncated,
        limit=config.MAX_BATCH_ROWS
    )


@app.post("/api/feedback")
def feedback():
    """Save user feedback to CSV and SQLite."""

    d = request.get_json(silent=True) or {}

    text = d.get("text", "")
    label = d.get("label")

    if (
        not isinstance(text, str)
        or not text.strip()
        or label not in ("scam", "legit")
    ):
        return jsonify(
            error="Need 'text' and 'label' ('scam' or 'legit')."
        ), 400

    config.FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with _fb_lock:
        new = not config.FEEDBACK_PATH.exists()

        with config.FEEDBACK_PATH.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as f:
            w = csv.writer(f)

            if new:
                w.writerow([
                    "timestamp",
                    "text",
                    "predicted",
                    "user_label"
                ])

            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                text[:config.MAX_TEXT_CHARS],
                d.get("predicted", ""),
                label
            ])

    storage.save_feedback(
        text[:config.MAX_TEXT_CHARS],
        d.get("predicted", ""),
        label
    )

    log.info("feedback received: %s", label)

    return jsonify(status="saved")


@app.get("/api/history")
def history():
    """Return the latest 100 stored scans (admin only)."""

    key = os.environ.get("ADMIN_KEY", "")
    sent = request.headers.get("X-Admin-Key", "")

    if not key or not hmac.compare_digest(sent.encode(), key.encode()):
        abort(404)

    records = storage.get_recent_scans(100)

    return jsonify(
        records=records,
        total=len(records)
    )


def main():
    log.info(
        "ScamShield %s on http://%s:%s",
        __version__,
        config.HOST,
        config.PORT
    )

    try:
        from waitress import serve
        serve(app, host=config.HOST, port=config.PORT)

    except ImportError:
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=False
        )


if __name__ == "__main__":
    main()