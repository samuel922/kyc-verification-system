@echo off
echo Starting Face Swap Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Python and Node.js found!
echo.

REM Start AI Service
echo Starting AI Service...
cd backend\ai-service
start "AI Service" cmd /k "python main.py"
cd ..\..

REM Wait a moment for AI service to start
timeout /t 3 /nobreak >nul

REM Start API Server
echo Starting API Server...
cd backend\api-server
start "API Server" cmd /k "npm run dev"
cd ..\..

echo.
echo ========================================
echo   Face Swap Application Started!
echo ========================================
echo.
echo AI Service:    http://localhost:8000
echo Frontend:      http://localhost:3001
echo.
echo Wait a few seconds for services to fully start,
echo then open http://localhost:3001 in your browser
echo.
echo Press any key to close this window...
pause >nul