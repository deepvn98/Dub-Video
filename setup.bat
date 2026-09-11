@echo off
cd /d "%~dp0"
python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 (
    set "DUB_PYTHON=python"
    goto :python_found
)
py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 (
    set "DUB_PYTHON=py -3"
    goto :python_found
)
echo.
echo Khong tim thay Python. Hay cai Python 3.10 tro len, bat tuy chon Add Python to PATH, roi chay lai setup.bat.
pause
exit /b 1

:python_found
where ffmpeg >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
where ffprobe >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
rem --clear also repairs a copied/stale environment that points to Python
rem installed at a different path on another computer.
%DUB_PYTHON% -m venv --clear .venv
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
type nul > ".venv\.dub_sync_ready_v7"

echo.
echo Cai dat Dub Sync da hoan tat.
pause
exit /b 0

:ffmpeg_missing
echo.
echo Khong tim thay FFmpeg hoac ffprobe trong PATH. Hay cai FFmpeg, mo lai cua so nay, roi chay setup.bat.
pause
exit /b 1

:error
echo.
echo Cai dat that bai. Hay cai Python 3.10 tro len va thu lai.
pause
exit /b 1
