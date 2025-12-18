@echo off
cd /d "%~dp0"

echo ===================================================
echo [1/3] Setting up Python Backend Environment...
echo ===================================================
if not exist "venv" (
    echo Creating venv...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing Backend Requirements...
pip install -r backend/requirements.txt --quiet

echo ===================================================
echo [2/3] Setting up React Frontend Environment...
echo ===================================================
cd frontend
if not exist "node_modules" (
    echo Installing Node Modules...
    call npm install
)
cd ..

echo ===================================================
echo [3/3] Starting Hybrid AI Sales Agent...
echo ===================================================
echo Starting Backend on Port 8000...
start "Backend" cmd /k "venv\Scripts\activate && uvicorn backend.main:app --reload --port 8000"

echo Starting Frontend on Port 5173...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo Done! Browser should open shortly at http://localhost:5173
echo Please ensure you have set GEMINI_API_KEY if you want to generate emails.
pause
