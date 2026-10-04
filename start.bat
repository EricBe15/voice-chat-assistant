@echo off
rem Starts the assistant. The first run sets everything up (.venv, packages, models, llama.cpp).
rem   start.bat            the assistant
rem   start.bat --debug    also save every heard phrase to logs\phrases
setlocal
cd /d "%~dp0"
set "CHECK=import sys, tkinter; sys.exit(sys.version_info < (3, 12))"
set "PY="
set "PYARG="

rem The venv's own Python once it exists, otherwise Python 3.12+ with tkinter from this PC
if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -c "%CHECK%" >nul 2>&1 && set "PY=.venv\Scripts\python.exe"
if not defined PY py -3 -c "%CHECK%" >nul 2>&1 && set "PY=py" && set "PYARG=-3"
if not defined PY python -c "%CHECK%" >nul 2>&1 && set "PY=python"
if defined PY goto run

echo Python 3.12 or newer wasn't found.
where winget >nul 2>&1 || goto no_python
choice /m "Install Python 3.13 now (winget)"
if errorlevel 2 goto no_python
winget install --id Python.Python.3.13 -e --accept-source-agreements --accept-package-agreements
set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if exist "%PY%" goto run

:no_python
echo Install Python 3.12 or newer from https://www.python.org/downloads/ and run this again.
pause
exit /b 1

:run
"%PY%" %PYARG% bootstrap.py %*
if errorlevel 1 pause
exit /b %errorlevel%
