@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo RTlation is not set up yet. Please run setup.bat or setup-all.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m rtlation --open %*
