@echo off
rem Runs setup.ps1. Options are passed on, e.g.:  setup.bat -LimitRate 7M
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
