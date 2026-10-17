@echo off
REM ===================================================================
REM  Chat with a trained model  --  interactive
REM
REM  Usage: double-click, or run:
REM         script\chat.bat                (Enter = default model chat_demo.pt)
REM         script\chat.bat "model name"   (skip the question)
REM
REM  The model name is asked for at run time and validated by the
REM  helper script. No model name is written into this file.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

set "HELPER=%~dp0train_chat_helper.py"
if not exist "%HELPER%" (
  echo [ERROR] helper not found: %HELPER%
  pause
  exit /b 1
)
if not exist "code\chat.py" (
  echo [ERROR] code\chat.py not found -- run this script from the project.
  pause
  exit /b 1
)

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] python.exe not found in PATH.
  pause
  exit /b 1
)

REM ---- optional arg: a model name (skips the menu) ----
set "WANT=%~1"
if "%WANT%"=="" (
  "%PY%" "%HELPER%" --mode chat
) else (
  "%PY%" "%HELPER%" --mode chat -m "%WANT%"
)
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [exit code %RC%]
pause
exit /b %RC%
