@echo off
REM ===================================================================
REM  dlqq-llm WebUI launcher (the only entry point)
REM
REM  Double-click: starts the local web UI and opens the browser.
REM  Closing THIS window stops everything: training, the bridge server,
REM  and any signature job started from the page.
REM  Closing the browser tab alone does NOT stop anything.
REM
REM  Runtime: prefers the bundled .python_env\python.exe (standard
REM  library only, no torch). If it is missing, falls back to a Python
REM  from PATH.
REM
REM  Optional arguments:
REM     start.bat --port 8080     use a fixed port
REM     start.bat --no-browser    do not open the browser automatically
REM     start.bat --no-lan        do not start the LAN chat page
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

REM ---- 1) bundled runtime (preferred) ---------------------------------
set "PY=.python_env\python.exe"
set "PYFLAGS=-s -P"
if exist "%PY%" goto run

REM ---- 2) fall back to a Python from PATH ------------------------------
echo [WARN] .python_env\python.exe not found; using Python from PATH.
set "PY="
set "PYFLAGS="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no Python found.
  echo         Re-extract the full package; it ships .python_env.
  echo         Or rebuild it with .tools\make_python_env.bat.
  pause
  exit /b 1
)

:run
echo [python] %PY%
echo [start ] launching the web UI ...
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" %PYFLAGS% "%SCRIPT%" %*

set "RC=%ERRORLEVEL%"
echo.
echo [exit code %RC%]
pause
exit /b %RC%
