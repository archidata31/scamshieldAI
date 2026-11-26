"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SCAMSHIELD_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = Path(os.environ.get("SCAMSHIELD_MODELS", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "scamshield.joblib"
CARD_PATH = MODEL_DIR / "model_card.json"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
DEEP_CARD_PATH = MODEL_DIR / "deep_card.json"
SHAP_BG_PATH = MODEL_DIR / "shap_background.joblib"
BIGRU_DIR = MODEL_DIR / "bigru"
BERT_DIR = MODEL_DIR / "distilbert"

MAX_TEXT_CHARS = int(os.environ.get("SCAMSHIELD_MAX_CHARS", 5000))
MAX_BATCH_ROWS = int(os.environ.get("SCAMSHIELD_MAX_BATCH", 500))
RATE_LIMIT_PER_MIN = int(os.environ.get("SCAMSHIELD_RATE_LIMIT", 240))
HOST = os.environ.get("SCAMSHIELD_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5000))

# Fallback verdict thresholds (training writes data-driven ones into the model card)
DEFAULT_SUSPICIOUS = 0.35
DEFAULT_SCAM = 0.65
