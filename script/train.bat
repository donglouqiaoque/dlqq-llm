@echo off
REM ===================================================================
REM  Train a model  --  pick a config file, nothing else to type
REM
REM  Double-click it: it lists every *.properties under properties\ and
REM  asks for a number. All training parameters (corpus, model name,
REM  layers, steps, learning rate, python interpreter ...) come from
REM  that file.
REM
REM  Optional: script\train.bat big      (pick properties\big.properties)
REM            script\train.bat my.properties
REM
REM  NOTE: this file must stay pure ASCII -- non-ASCII bytes break the
REM  cmd parser. Chinese messages belong to the Python helper.
REM ===================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0.."

set "HELPER=%~dp0train_chat_helper.py"
set "CFGDIR=%~dp0..\properties"

if not exist "%HELPER%" (
  echo [ERROR] helper not found: %HELPER%
  pause
  exit /b 1
)
if not exist "code\train_llm.py" (
  echo [ERROR] code\train_llm.py not found -- run this script from the project.
  pause
  exit /b 1
)

REM ---- 1) which config file ----
set "PY0="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY0 set "PY0=%%P"
if not defined PY0 for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY0 set "PY0=%%P"
if not defined PY0 (
  echo [ERROR] no python found in PATH -- cannot list config files.
  pause
  exit /b 1
)

set "PICK="
if not "%~1"=="" (
  set "PICK=%~1"
  if not exist "!PICK!" if exist "%CFGDIR%\%~1" set "PICK=%CFGDIR%\%~1"
  if not exist "!PICK!" if exist "%CFGDIR%\%~1.properties" set "PICK=%CFGDIR%\%~1.properties"
) else (
  REM the python helper prints the list, asks for a number, and writes the
  REM chosen path to a temp file -- more robust than capturing stdout
  set "PICKFILE=%TEMP%\dsh_pick_%RANDOM%%RANDOM%.txt"
  "!PY0!" "%HELPER%" --pick-config --pick-out "!PICKFILE!"
  if exist "!PICKFILE!" for /f "usebackq delims=" %%P in ("!PICKFILE!") do set "PICK=%%P"
  if exist "!PICKFILE!" del "!PICKFILE!" >nul 2>nul
)

if not defined PICK (
  echo [ERROR] no config selected.
  echo         Run this .bat by double-clicking it ^(it needs a real console^),
  echo         or name a config:  script\train.bat chat_demo
  pause
  exit /b 1
)

if not exist "!PICK!" (
  echo [ERROR] config not found: !PICK!
  echo         Looked in: %CFGDIR%
  pause
  exit /b 1
)

REM ---- 2) python: if the config names one, it MUST work -- no fallback ----
set "CFGPY="
for /f "usebackq tokens=1,* delims==" %%A in ("!PICK!") do (
  if /i "%%A"=="python" set "CFGPY=%%B"
)
if defined CFGPY for /f "tokens=* delims= " %%C in ("!CFGPY!") do set "CFGPY=%%C"
if defined CFGPY if not exist "!CFGPY!" (
  echo [ERROR] python from config not found:
  echo         !CFGPY!
  echo         Fix the python= line in: !PICK!
  pause
  exit /b 1
)
REM "path\" exists only for directories -- the config must point at python.exe
if defined CFGPY if exist "!CFGPY!\" (
  echo [ERROR] python from config is a directory, point it at python.exe:
  echo         !CFGPY!
  echo         Fix the python= line in: !PICK!
  pause
  exit /b 1
)
REM the file may exist yet not be a working interpreter (store stub, broken venv)
if defined CFGPY (
  "!CFGPY!" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] python from config cannot run:
    echo         !CFGPY!
    echo         Fix the python= line in: !PICK!
    pause
    exit /b 1
  )
)

REM only when the config leaves python= empty do we use PATH
set "PY="
if defined CFGPY set "PY=!CFGPY!"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no python found. Add one line to the config file:
  echo         python=D:\path\to\python.exe
  pause
  exit /b 1
)

echo.
echo [config]  !PICK!
echo [python]  !PY!
echo.

REM unbuffered: otherwise the training log sits in Python's buffer and the
REM window looks frozen for minutes when the output is captured/redirected
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

"!PY!" "%HELPER%" --mode train --config "!PICK!"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [done] model saved under model\
if not "%RC%"=="0" echo [exit code %RC%]
pause
exit /b %RC%
