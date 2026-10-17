@echo off
REM ===================================================================
REM  Bridge server: serve a trained model as an OpenAI compatible API
REM
REM  Usage (double-click, or run from the project root):
REM    script\http_server.bat                  (asks which model, then port 49128)
REM    script\http_server.bat chat_demo        (skip the question)
REM    script\http_server.bat chat_demo 9000   (custom port)
REM    script\http_server.bat chat_demo 9000 cpu   (force CPU inference)
REM
REM  Then in DSH: Settings -> Models -> Add provider -> Custom provider
REM    Base URL      http://127.0.0.1:49128/v1
REM    API protocol  openai-completions
REM    API key       local        (any non-empty value)
REM    Model         chat_demo    (or click "fetch available models")
REM
REM  Keep this window open while DSH uses the model. Ctrl+C stops it.
REM  Ends with PAUSE so error messages stay on screen.
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM ===================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0.."

set "SERVER=%~dp0http_server.py"
set "HELPER=%~dp0train_chat_helper.py"
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

REM ---- 1) which model: argument wins, otherwise ask ----
set "MODEL=%~1"
if "!MODEL!"=="" (
  if not exist "%HELPER%" (
    echo [ERROR] helper not found: %HELPER%
    pause
    exit /b 1
  )
  set "PICKFILE=%TEMP%\dsh_model_%RANDOM%%RANDOM%.txt"
  "%PY%" "%HELPER%" --pick-model --pick-out "!PICKFILE!"
  if exist "!PICKFILE!" for /f "usebackq delims=" %%P in ("!PICKFILE!") do set "MODEL=%%P"
  if exist "!PICKFILE!" del "!PICKFILE!" >nul 2>nul
  if "!MODEL!"=="" (
    echo [ERROR] no model selected.
    echo         Run this .bat by double-clicking it, or name a model:
    echo         script\http_server.bat chat_demo
    pause
    exit /b 1
  )
) else (
  if not exist "!MODEL!" if exist "model\!MODEL!.pt" set "MODEL=model\!MODEL!.pt"
  if not exist "!MODEL!" if exist "model\!MODEL!" set "MODEL=model\!MODEL!"
  if not exist "!MODEL!" (
    echo [ERROR] model not found: !MODEL!
    echo         Looked for model\!MODEL! and model\!MODEL!.pt
    pause
    exit /b 1
  )
)

set "PORT=%~2"
if "%PORT%"=="" set "PORT=49128"

set "DEVICE=%~3"
set "DEVARG="
if not "%DEVICE%"=="" set "DEVARG=--device %DEVICE%"

echo.
echo [python] %PY%
echo [model ] !MODEL!
echo [port  ] %PORT%
if not "%DEVICE%"=="" echo [device] %DEVICE%
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" "%SERVER%" --model "!MODEL!" --port %PORT% %DEVARG%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [server stopped]
if not "%RC%"=="0" echo [server exited with code %RC%]
pause
exit /b %RC%
