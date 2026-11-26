#!/usr/bin/env sh
# One-time setup for the full NLP pipeline (macOS / Linux).
set -e
cd "$(dirname "$0")"
pip install -r requirements-deep.txt
python3 -m scamshield.download_data --emails
python3 -m scamshield.train
python3 -m scamshield.train_deep --model bigru
python3 -m scamshield.train_deep --model distilbert --max-samples 4000
echo "Done. Start the app with ./run.sh"
