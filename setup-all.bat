@echo off
setlocal
rem Installs RTlation with all models (both speech recognition and both voice
rem models). Ollama must be installed and running.
rem Asks at the start whether the downloads should be limited.

echo RTlation - setup with all models (about 14 GB to download)
echo.
echo Limit the download speed so your connection stays usable?
echo Enter a number in MB/s (e.g. 8), or just press Enter for full speed.
echo.

:ask
set "LIMIT="
set /p "LIMIT=Limit in MB/s: "
if not defined LIMIT goto full
rem Whole numbers from 1 only.
echo %LIMIT%| findstr /r "^[1-9][0-9]*$" >nul || (
    echo Please enter a whole number such as 8, or just press Enter.
    goto ask
)
echo.
echo Downloads limited to %LIMIT% MB/s.
call "%~dp0setup.bat" -LimitRate %LIMIT%M -Stt large-v3-turbo,large-v3 -Tts chatterbox,qwen3
goto :eof

:full
echo.
echo Downloading at full speed.
call "%~dp0setup.bat" -Stt large-v3-turbo,large-v3 -Tts chatterbox,qwen3
