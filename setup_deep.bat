@echo off
REM One-time setup for the full NLP pipeline on Windows (run from this folder).
REM Installs deep-learning packages, downloads SMS + email datasets, trains every model.
cd /d "%~dp0"
set TRUST=--trusted-host pypi.org --trusted-host files.pythonhosted.org
echo [1/5] Installing packages (torch is a large download)...
pip install %TRUST% -r requirements-deep.txt || goto :fail
echo [2/5] Downloading datasets...
python -m scamshield.download_data --emails || goto :fail
echo [3/5] Training TF-IDF + features + Logistic Regression...
python -m scamshield.train || goto :fail
echo [4/5] Training BiGRU...
python -m scamshield.train_deep --model bigru || goto :fail
echo [5/5] Fine-tuning DistilBERT (10-25 minutes on CPU)...
python -m scamshield.train_deep --model distilbert --max-samples 4000 || goto :fail
echo.
echo Done. Start the app with run.bat
goto :eof
:fail
echo.
echo A step failed. Read the message above. For SSL errors try: pip install pip-system-certs
exit /b 1
