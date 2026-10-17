@echo off
REM Double-click to set everything up. Runs setup.ps1 with the execution policy bypassed for this process only.
setlocal
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
pushd "%~dp0" || (echo Could not open "%~dp0" & pause & exit /b 1)
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%~dp0setup.ps1'" >nul 2>&1
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo.
    echo Setup did not finish cleanly ^(exit code %RC%^).
    echo If PowerShell refused to run the script, right-click setup.ps1, choose Properties, tick "Unblock", then try again.
)
echo.
pause
exit /b %RC%
