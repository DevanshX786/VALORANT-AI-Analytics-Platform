@echo off
setlocal
title VALORANT ML Shutdown Engine
color 0C

echo =======================================================
echo    Stopping VALORANT AI Analytics Platform (LOCAL)
echo =======================================================
echo.

echo [1/2] Stopping backend on port 8010...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$p=(Get-NetTCPConnection -LocalPort 8010 -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess -Unique ^| Where-Object { $_ -gt 0 }); if($p){$p ^| ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; Write-Output ('Stopped PID ' + $_ + ' on 8010') }} else { Write-Output 'No process found on 8010' }"

echo [2/2] Stopping frontend on port 8080...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$p=(Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess -Unique ^| Where-Object { $_ -gt 0 }); if($p){$p ^| ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; Write-Output ('Stopped PID ' + $_ + ' on 8080') }} else { Write-Output 'No process found on 8080' }"

echo.
echo =======================================================
echo Local services stopped.
echo =======================================================
timeout /t 2 /nobreak >nul
endlocal
