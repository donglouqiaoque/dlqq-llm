@echo off
REM ===================================================================
REM  Environment self-check  --  run this whenever something is missing
REM
REM  Usage: double-click, or run:  script\torch_check.bat
REM
REM  This bat only locates a python and starts the checker. All the
REM  real checking is done in python (script\torch_check.py), which
REM  can also re-launch itself with another interpreter if the first
REM  one has no torch.
REM
REM  Ends with PAUSE so the result stays on screen.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

set "CHECKER=%~dp0torch_check.py"
if not exist "%CHECKER%" (
  echo [ERROR] checker not found: %CHECKER%
  pause
  exit /b 1
)

REM ---- 1) every python on PATH: keep the first that imports torch ----
set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do call :try "%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do call :try "%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do call :try "%%P"

REM ---- 2) none of them has torch: hand over to the checker anyway ----
REM      torch_check.py re-launches itself with another python from PATH.
if not defined PY (
  for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
)
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"

if not defined PY (
  echo [ERROR] no python found in PATH.
  echo         Install Python first:  https://www.python.org/downloads/
  pause
  exit /b 1
)

echo [python] %PY%
echo.
"%PY%" "%CHECKER%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [result] all checks passed
if not "%RC%"=="0" echo [result] something failed -- see the items marked FAIL above ^(exit code %RC%^)
echo.
pause
exit /b %RC%

:try
if defined PY exit /b 0
"%~1" -c "import torch" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0
