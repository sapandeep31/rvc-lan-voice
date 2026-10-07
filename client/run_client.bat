@echo off
title RVC LAN Voice Client (Windows 0% GPU)
cd /d "%~dp0\.."

set PYTHONPATH=%CD%
call "C:\AVoc\.venv\Scripts\activate.bat"
python client\client_windows.py
pause
