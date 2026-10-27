@echo off
REM ===================================================================
REM  Compute SHA-256 signatures for every .pt weight in this repo
REM
REM  For each  model\<name>\<name>.pt  it writes  <name>.pt.sha256
REM  right next to the weight (existing signature files are overwritten).
REM
REM  Usage: double-click, or run from the repo root:
REM         script\compute_signature.bat
REM         script\compute_signature.bat --quiet
REM         script\compute_signature.bat --file model\dlqq_llm\dlqq_llm.pt
REM
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM        All the real work (and every Chinese message) is in
REM        script\model_signature.py.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

set "TOOL=%~dp0model_signature.py"
if not exist "%TOOL%" (
  echo [ERROR] tool not found: %TOOL%
  pause
  exit /b 1
)

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no python found in PATH.
  echo         Install Python 3 first, then run this script again.
  pause
  exit /b 1
)

echo.
echo [python] %PY%
echo [mode  ] compute
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" "%TOOL%" --mode compute %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [done] signature files written next to each .pt
if not "%RC%"=="0" echo [failed] exit code %RC%
pause
exit /b %RC%
