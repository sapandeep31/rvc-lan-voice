@echo off
rem One-time setup of the voice client on Windows (the PC with the mic). Safe to re-run.
rem Needs Python 3.12 from python.org (tick "Add python.exe to PATH") and VB-Audio Virtual Cable.
cd /d "%~dp0\..\.."

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python 3.12 not found. Install it from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
)
if not exist "client\.venv\Scripts\python.exe" (
    echo [*] Creating client\.venv ...
    py -3.12 -m venv client\.venv
)
echo [*] Installing client packages ...
"client\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
"client\.venv\Scripts\python.exe" -m pip install --quiet numpy websockets PySide6
if errorlevel 1 (
    echo [!] Package install failed.
    pause
    exit /b 1
)
echo.
echo [OK] Client installed.
echo      1. Install VB-Audio Virtual Cable if you have not: https://vb-audio.com/Cable/
echo      2. In config.json -^> "client": set "server_ip" and "server_port" to what the server window prints
echo         (same PC as the server: "server_ip": "127.0.0.1"), and "input_device_keyword" to part of your mic's name.
echo      3. Run setup\windows\start_client.bat
pause
