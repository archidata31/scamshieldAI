#!/usr/bin/env sh
# ScamShield launcher for macOS / Linux
cd "$(dirname "$0")" || exit 1
[ -f data/synthetic.csv ] || python3 -m scamshield.data_gen
[ -f models/scamshield.joblib ] || python3 -m scamshield.train
python3 -m scamshield.app
