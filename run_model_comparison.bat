@echo off
cd /d "%~dp0"

echo Activating venv...
if not exist "venv" (
    echo venv not found! Please run setup_venv.bat first.
    pause
    exit /b
)
call venv\Scripts\activate

echo Checking dependencies...
pip install lightgbm colorama scikit-learn pandas --quiet

echo Running Model Comparison...
python -m backend.evaluation.compare_models

pause
