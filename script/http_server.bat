@echo off
REM ===================================================================
REM  Bridge server: serve ALL trained models as an OpenAI compatible API
REM
REM  No model selection needed: it scans model\ and release\ at startup,
REM  lists every .pt it finds, and loads a model only when DSH actually
REM  talks to it (switching models unloads the previous one).
REM
REM  Usage (double-click, or run from the project root):
REM    script\http_server.bat                  (port 49128, auto device)
REM    script\http_server.bat 9000             (custom port)
REM    script\http_server.bat 9000 cpu         (force CPU inference)
REM    script\http_server.bat 9000 cuda 2      (keep 2 models in VRAM)
REM
REM  Then in DSH: Settings -> Models -> Add provider -> Custom provider
REM    Base URL      http://127.0.0.1:49128/v1
REM    API protocol  openai-completions
REM    API key       local        (any non-empty value)
REM    Model         click "fetch available models" -> pick any model id
REM
REM  Keep this window open while DSH uses the model. Ctrl+C stops it.
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM ===================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0.."

set "SERVER=%~dp0http_server.py"
if not exist "%SERVER%" (
  echo [ERROR] server not found: %SERVER%
  pause
  exit /b 1
)

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no python found in PATH.
  echo         Install Python first, then run this script again.
  pause
  exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=49128"

set "DEVICE=%~2"
set "DEVARG="
if not "%DEVICE%"=="" set "DEVARG=--device %DEVICE%"

set "MAXLOAD=%~3"
set "LOADARG="
if not "%MAXLOAD%"=="" set "LOADARG=--max-loaded %MAXLOAD%"

echo.
echo [python] %PY%
echo [port  ] %PORT%
if not "%DEVICE%"=="" echo [device] %DEVICE%
if not "%MAXLOAD%"=="" echo [loaded] %MAXLOAD%
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" "%SERVER%" --port %PORT% %DEVARG% %LOADARG%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [server stopped]
if not "%RC%"=="0" echo [server exited with code %RC%]
pause
exit /b %RC%
