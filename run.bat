@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo RTlation ist noch nicht eingerichtet. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m rtlation --open %*
