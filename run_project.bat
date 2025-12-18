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
pip install -r backend/requirements.txt
pip install lightgbm --quiet

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
echo.
echo [IMPORTANT] OLLAMA CHECK:
echo Ensure you have Ollama installed and have pulled the model by running:
echo "ollama pull mistral"
echo in a separate terminal window functionality.
pause
