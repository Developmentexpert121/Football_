Write-Host "======================================================================" -ForegroundColor Green
Write-Host "          Statcut Football Analytics - Web App Launcher" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

$VENV_DIR = ".venv"
if (-not (Test-Path "$VENV_DIR\Scripts\python.exe")) {
    if (Test-Path "venv\Scripts\python.exe") {
        $VENV_DIR = "venv"
    } else {
        Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
        Write-Host "Please ensure .venv or venv exists and has dependencies installed." -ForegroundColor Red
        Exit
    }
}

Write-Host "[INFO] Activating virtual environment ($VENV_DIR)..." -ForegroundColor Yellow
& "$VENV_DIR\Scripts\Activate.ps1"

Write-Host "[INFO] Starting FastAPI application server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Web UI is available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

python app.py
