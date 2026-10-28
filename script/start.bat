@echo off
REM ===================================================================
REM  dlqq-llm WebUI launcher (the only entry point)
REM
REM  Double-click: starts the local web UI and opens the browser.
REM  Closing THIS window stops everything: training, the bridge server,
REM  and any signature job started from the page.
REM  Closing the browser tab alone does NOT stop anything.
REM
REM  Optional arguments:
REM     start.bat --port 8080     use a fixed port
REM     start.bat --no-browser    do not open the browser automatically
REM
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

set "SCRIPT=code\webui.py"
if not exist "%SCRIPT%" (
  echo [ERROR] not found: %SCRIPT%
  pause
  exit /b 1
)

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no python found in PATH.
  echo         Install Python 3.10+ first, then run this again.
  pause
  exit /b 1
)

echo [python] %PY%
echo [start ] launching the web UI ...
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" "%SCRIPT%" %*

set "RC=%ERRORLEVEL%"
echo.
echo [exit code %RC%]
pause
exit /b %RC%
