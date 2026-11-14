@echo off
rem Start the voice server on Windows. Port from config.json ("server" -> "port") unless given: start_server.bat 8010
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
    echo [!] Not installed yet. Run setup\windows\install_server.bat
    pause
    exit /b 1
)
set PORT=%~1
if "%PORT%"=="" for /f %%p in ('.venv\Scripts\python.exe -c "import json;print(json.load(open('config.json'))['server']['port'])"') do set PORT=%%p
echo ==================================================
echo   Voice server starting on port %PORT% (model loads in ~30 s)
echo   Client on this PC:      "server_ip": "127.0.0.1"
echo   Client on another PC:   "server_ip": this PC's IPv4 address (run ipconfig)
echo   Control panel:          http://127.0.0.1:%PORT%/panel
echo   Allow Python through Windows Firewall when asked, so other devices can connect.
echo ==================================================
.venv\Scripts\python.exe -m uvicorn --app-dir server server:app --host 0.0.0.0 --port %PORT%
pause
