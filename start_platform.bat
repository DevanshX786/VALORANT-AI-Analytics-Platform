@echo off
title VALORANT ML Platform Local Launcher
color 0b

echo =======================================================
echo    Booting up the VALORANT AI Analytics Platform...
echo =======================================================
echo.

echo [1/2] Spinning up the FastAPI Machine Learning Backend...
echo (It will load heavy CSV data and cache the XGBoost models)
start "Backend Fastapi Server" cmd /k "venv\Scripts\python -m uvicorn backend.api:app --reload  --port 8000"

:: Wait 18 seconds for the ML architecture & Pandas logic to finish buffering in memory
echo Waiting 18 seconds for Backend Model Compilation...
timeout /t 18 /nobreak >nul

echo [2/2] Booting the HTML UI Webserver (Port 8080)...
start "Frontend Web Server" cmd /k "venv\Scripts\python -m http.server 8080 --directory frontend"

:: Wait incredibly fast to let the Python server bind the 8080 local port natively
timeout /t 2 /nobreak >nul

echo.
echo =======================================================
echo All Systems Online! Launching your default Web Browser...
echo =======================================================
start http://localhost:8080
