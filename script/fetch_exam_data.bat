@echo off
REM ===================================================================
REM  Download Chinese exam datasets from ModelScope (modelscope.cn)
REM  and build the demo corpus.
REM
REM  Domestic only: no GitHub, no HuggingFace, no login needed.
REM
REM  Double-click it. Output:
REM     src\chat_demo.jsonl              demo corpus (all questions)
REM     ai_res\doc\exam\exam_eval.jsonl  eval split, if any
REM
REM  Optional: fetch_exam_data.bat --datasets c-eval gaokao bio
REM            fetch_exam_data.bat --list
REM
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERROR] no python found in PATH.
  pause
  exit /b 1
)

set "SCRIPT=%~dp0fetch_exam_data.py"
if not exist "%SCRIPT%" (
  echo [ERROR] not found: %SCRIPT%
  pause
  exit /b 1
)

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

echo [python]  %PY%
echo [source]  ModelScope (modelscope.cn)
echo [output]  src\chat_demo.jsonl  (demo corpus)
echo           ai_res\doc\exam      (raw files + eval split)
echo.

set "ARGS="
:collect
if "%~1"=="" goto run
set "ARGS=%ARGS% %~1"
shift
goto collect

:run
"%PY%" "%SCRIPT%" --out ai_res\doc\exam --all-to-train --single src\chat_demo.jsonl %ARGS%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [done] see src\chat_demo.jsonl
if not "%RC%"=="0" echo [exit code %RC%]
echo.
echo REMINDER: the demo trains on these questions, so label the source
echo           in README (ModelScope: C-Eval / gaokao-benchmark / ...).
echo           Do NOT report benchmark scores on the same questions.
pause
exit /b %RC%
