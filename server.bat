@echo off
setlocal enabledelayedexpansion

:: デフォルト値の設定
set HOST=0.0.0.0
set PORT=7860

:: コマンドライン引数の解析
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

echo Starting server on %HOST%:%PORT%...
venv\Scripts\python.exe -m uvicorn main:app --host %HOST% --port %PORT%