@echo off
rem DTU launcher for Windows: double-click to start the GUI.
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.9+ is required: https://www.python.org/downloads/
  echo Check "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
%PY% -c "import numpy" >nul 2>nul || %PY% -m pip install --user -r requirements.txt
%PY% -m dtu %*
if errorlevel 1 pause
