@echo off
cd /d "%~dp0"

echo Activating venv...
if not exist "venv" (
    echo venv not found! Please run setup_venv.bat first.
    pause
    exit /b
)
call venv\Scripts\activate

echo Checking/Installing dependencies...
pip install -r backend/requirements.txt --quiet
pip install tqdm colorama --quiet

echo Starting Evaluation Script...
python -m backend.evaluation.run_evals

pause
