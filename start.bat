@echo off
echo ======================================================================
echo          FootballAI Analytics - Full Stack Launcher
echo ======================================================================
echo.

set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    if exist "venv\Scripts\python.exe" (
        set VENV_DIR=venv
    ) else (
        echo [ERROR] Virtual environment not found!
        echo Please ensure .venv or venv exists and has dependencies installed.
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment (%VENV_DIR%)...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "FootballAI Backend" cmd /k "call %VENV_DIR%\Scripts\activate.bat && python app.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React frontend on http://localhost:5173 ...
start "FootballAI Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ======================================================================
echo  Backend  ^>  http://localhost:8000   (API + Video Processing)
echo  Frontend ^>  http://localhost:5173   (Modern React UI)
echo ======================================================================
echo.
echo Both windows are open. Close them to stop the servers.
pause
