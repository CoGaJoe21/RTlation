@echo off
rem Startet setup.ps1. Optionen werden durchgereicht, z.B.:  setup.bat -LimitRate 7M
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
