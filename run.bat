@echo off
REM ScamShield launcher for Windows. Double-click or run from the project folder.
cd /d "%~dp0"
if not exist data\synthetic.csv python -m scamshield.data_gen
if not exist models\scamshield.joblib python -m scamshield.train
echo.
echo Starting ScamShield at http://127.0.0.1:5000  (press Ctrl+C to stop)
python -m scamshield.app
