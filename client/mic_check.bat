@echo off
title RVC LAN Voice - mic check (you hear what Discord hears)
cd /d "%~dp0\.."

rem Usage: mic_check.bat [monitor device keyword]   default: WH-CH720N
set MON=%~1
if "%MON%"=="" set MON=WH-CH720N

if not exist "client\.venv\Scripts\python.exe" (
    echo client\.venv not found - see run_client.bat
    pause
    exit /b 1
)
echo Converted voice goes to CABLE Input (Discord) AND to "%MON%" so you can hear it.
echo Close this window to stop.
"client\.venv\Scripts\python.exe" client\client_windows.py --monitor "%MON%"
pause
