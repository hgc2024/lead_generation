@echo off
cd /d "%~dp0"

echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r backend/requirements.txt

echo Setup complete. To run the app use run_app.bat
pause
