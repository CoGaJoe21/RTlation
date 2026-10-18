@echo off
setlocal
rem Richtet RTlation mit allen Modellen ein (beide Spracherkennungs- und beide
rem Sprachausgabe-Modelle). Ollama muss installiert sein und laufen.
rem Fragt zu Beginn, ob die Downloads gedrosselt werden sollen.

echo RTlation - Einrichtung mit allen Modellen (rund 14 GB Download)
echo.
echo Soll der Download gedrosselt werden, damit die Leitung frei bleibt?
echo Zahl in MB/s eingeben (z.B. 8) oder einfach Enter fuer volle Geschwindigkeit.
echo.

:ask
set "LIMIT="
set /p "LIMIT=Limit in MB/s: "
if not defined LIMIT goto full
rem Nur ganze Zahlen ab 1 zulassen.
echo %LIMIT%| findstr /r "^[1-9][0-9]*$" >nul || (
    echo Bitte eine ganze Zahl eingeben, z.B. 8, oder nur Enter.
    goto ask
)
echo.
echo Downloads werden auf %LIMIT% MB/s begrenzt.
call "%~dp0setup.bat" -LimitRate %LIMIT%M -Stt large-v3-turbo,large-v3 -Tts chatterbox,qwen3
goto :eof

:full
echo.
echo Downloads mit voller Geschwindigkeit.
call "%~dp0setup.bat" -Stt large-v3-turbo,large-v3 -Tts chatterbox,qwen3
