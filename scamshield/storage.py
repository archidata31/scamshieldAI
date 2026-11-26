import sqlite3
import json
from datetime import datetime, timezone

from . import config


DB_PATH = config.DATA_DIR / "scamshield.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                model TEXT,
                probability REAL,
                risk_score INTEGER,
                verdict TEXT,
                scam_type TEXT,
                votes_json TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted TEXT,
                user_label TEXT
            )
        """)


def save_scan(text, result, model):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO scan_history
            (timestamp, text, model, probability, risk_score,
             verdict, scam_type, votes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            text,
            model,
            result.get("probability"),
            result.get("risk_score"),
            result.get("verdict"),
            result.get("scam_type"),
            json.dumps(result.get("votes", {}))
        ))


def save_feedback(text, predicted, user_label):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO user_feedback
            (timestamp, text, predicted, user_label)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            text,
            predicted,
            user_label
        ))


def get_recent_scans(limit=100):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM scan_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]