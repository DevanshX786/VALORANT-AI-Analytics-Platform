@echo off
title VALORANT ML Shutdown Engine
color 0c

echo =======================================================
echo    Terminating VALORANT AI Analytics Servers
echo =======================================================
echo.

echo [1/2] Terminating Uvicorn ML Backend (Port 8000)...
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"

echo [2/2] Terminating HTML Frontend Server (Port 8080)...
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"

echo.
echo =======================================================
echo Shutdown Complete. All background tasks destroyed!
echo =======================================================
timeout /t 3 >nul
