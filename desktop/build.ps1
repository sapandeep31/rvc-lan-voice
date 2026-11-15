# Builds dist\RVCVoice.exe. Run from the repo root on Windows with Python 3.12:
#   powershell -ExecutionPolicy Bypass -File desktop\build.ps1
$ErrorActionPreference = "Stop"
python -m pip install --quiet -r desktop\requirements-build.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# --uac-admin: the app asks for administrator rights at launch (needed to install the VB-Cable driver)
# --paths: manager.py imports server\audio_devices.py and setup\download_models.py
# --add-data: the engine's code and pages, copied into %LOCALAPPDATA%\RVCVoice\app at launch
python -m PyInstaller --noconfirm --clean --onefile --windowed --uac-admin --name RVCVoice `
    --icon desktop\icon.ico --paths server --paths setup `
    --add-data "server;server" --add-data "setup\download_models.py;setup" `
    --add-data "config.json;." --add-data "desktop\ui;desktop\ui" `
    desktop\manager.py
exit $LASTEXITCODE
