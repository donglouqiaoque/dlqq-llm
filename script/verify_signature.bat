@echo off
REM ===================================================================
REM  Verify every .pt weight against its SHA-256 signature file
REM
REM  For each  model\<name>\<name>.pt  it reads  <name>.pt.sha256
REM  and re-computes the digest. Exit code 0 = all pass, 1 = mismatch.
REM
REM  Usage: double-click, or run from the repo root:
REM         script\verify_signature.bat
REM         script\verify_signature.bat --quiet
REM         script\verify_signature.bat --file model\chat_demo\chat_demo.pt
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
echo [mode  ] verify
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%PY%" "%TOOL%" --mode verify %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [ok] every weight matches its signature
if not "%RC%"=="0" echo [FAILED] verification failed -- weight may be corrupted or replaced
pause
exit /b %RC%
