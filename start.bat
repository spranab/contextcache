@echo off
REM ContextCache — one-command startup for Windows
REM Usage:
REM   start.bat          — Demo mode (no GPU)
REM   start.bat --live   — Live mode (requires GPU)

cd /d "%~dp0"

set MODE=--demo
set EXTRA_ARGS=

:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--live" (
    set MODE=
    shift
    goto :parse_args
)
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto :parse_args
:done_args

REM Create venv if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate
call .venv\Scripts\activate.bat

REM Install dependencies
if defined MODE (
    echo Installing demo dependencies...
    pip install -q fastapi uvicorn pydantic pyyaml
) else (
    echo Installing full dependencies (including PyTorch)...
    pip install -q -r requirements.txt
)

echo.
python scripts/serve/launch.py %MODE% %EXTRA_ARGS%
