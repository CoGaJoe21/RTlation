@echo off
rem Richtet RTlation mit allen Modellen ein (beide Spracherkennungs- und beide
rem Sprachausgabe-Modelle) und begrenzt die Downloads auf 8 MB/s.
rem Ollama muss installiert sein und laufen.
call "%~dp0setup.bat" -LimitRate 8M -Stt large-v3-turbo,large-v3 -Tts chatterbox,qwen3
