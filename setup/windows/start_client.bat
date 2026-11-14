@echo off
rem Start the voice client: mic -> server -> "CABLE Input". In Discord/games pick "CABLE Output" as the microphone.
rem Optional: start_client.bat "Headphones keyword"  also plays the converted voice to those headphones (mic check).
cd /d "%~dp0\..\.."
if not exist "client\.venv\Scripts\python.exe" (
    echo [!] Not installed yet. Run setup\windows\install_client.bat
    pause
    exit /b 1
)
if "%~1"=="" (
    "client\.venv\Scripts\python.exe" client\client_windows.py
) else (
    "client\.venv\Scripts\python.exe" client\client_windows.py --monitor "%~1"
)
pause
