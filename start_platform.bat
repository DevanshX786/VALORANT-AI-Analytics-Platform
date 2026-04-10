@echo off
setlocal
title VALORANT ML Platform Local Launcher
color 0B

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON_EXE=%ROOT%venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
	set "PYTHON_EXE=python"
)

echo =======================================================
echo    Booting VALORANT AI Analytics Platform (LOCAL)...
echo =======================================================
echo Root: %ROOT%
echo Python: %PYTHON_EXE%
echo.

echo [1/2] Starting FastAPI backend on http://127.0.0.1:8010
start "VALORANT Backend (Local)" cmd /k "cd /d "%ROOT%" && "%PYTHON_EXE%" -m uvicorn backend.api:app --host 127.0.0.1 --port 8010 --timeout-keep-alive 75"

echo Waiting for backend warm-up...
timeout /t 8 /nobreak >nul

echo [2/2] Starting frontend static server on http://127.0.0.1:8080
start "VALORANT Frontend (Local)" cmd /k "cd /d "%ROOT%" && "%PYTHON_EXE%" -m http.server 8080 --bind 127.0.0.1 --directory frontend"

timeout /t 2 /nobreak >nul

echo.
echo =======================================================
echo Local stack online.
echo Backend : http://127.0.0.1:8010
echo Frontend: http://127.0.0.1:8080
echo =======================================================
start "" http://127.0.0.1:8080
endlocal
