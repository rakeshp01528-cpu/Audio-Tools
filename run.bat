@echo off
REM Double-click to open the Audio Tools menu. Arguments go to the CLI,
REM e.g.  run.bat normalize "C:\music\song.wav" -p spotify -f mp3
REM Relative file paths are resolved from the folder you run this in.
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
"%PY%" -c "import audio_tools, numpy" >nul 2>&1
if errorlevel 1 (
    echo First run: setting things up...
    set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
    "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%HERE%setup.ps1'" >nul 2>&1
    "%PS%" -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup.ps1"
    if errorlevel 1 ( pause & exit /b 1 )
    echo.
)
"%PY%" -m audio_tools %*
if errorlevel 1 pause
