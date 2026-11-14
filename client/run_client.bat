@echo off
title RVC LAN Voice Client (Windows 0% GPU)
cd /d "%~dp0\.."

set PYTHONPATH=%CD%
if exist "client\.venv\Scripts\python.exe" (
    "client\.venv\Scripts\python.exe" client\client_windows.py
) else (
    echo client\.venv not found. Create it with:
    echo   py -3.12 -m venv client\.venv
    echo   client\.venv\Scripts\python.exe -m pip install numpy websockets PySide6
)
pause
