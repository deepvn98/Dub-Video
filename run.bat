@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :setup
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 goto :setup
if not exist ".venv\.dub_sync_ready_v7" goto :setup
goto :run

:setup
call setup.bat
if errorlevel 1 exit /b 1
if not exist ".venv\.dub_sync_ready_v7" exit /b 1

:run
".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
exit /b %errorlevel%
