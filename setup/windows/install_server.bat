@echo off
rem One-time setup of the voice SERVER on a Windows machine with an NVIDIA GPU. Safe to re-run.
rem Note: hosting on the same gaming PC uses its GPU; a second machine (or a Mac) keeps games at full FPS.
cd /d "%~dp0\..\.."

where uv >nul 2>&1
if errorlevel 1 (
    echo [!] uv is required. Install it in PowerShell with:
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)
rem voiceconversion 0.0.8 installs only on exactly Python 3.12.3
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating Python 3.12.3 environment in .venv ...
    uv venv --python 3.12.3 .venv
)
echo [*] Installing PyTorch with CUDA (first run downloads ~3 GB) ...
uv pip install --quiet --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail
echo [*] Installing server dependencies ...
uv pip install --quiet --python .venv\Scripts\python.exe "voiceconversion==0.0.8" fastapi "uvicorn[standard]" websockets "numpy<2" pydantic
if errorlevel 1 goto :fail
echo [*] Downloading voice models ...
.venv\Scripts\python.exe setup\download_models.py
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -c "import torch; print('[*] NVIDIA GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT available, will run on CPU (slow)')"
echo.
echo [OK] Installed. Start the server with setup\windows\start_server.bat
pause
exit /b 0

:fail
echo [!] Install failed, see the error above.
pause
exit /b 1
