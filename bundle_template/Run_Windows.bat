@echo off
title AgyMolden One-Click Launcher

cd /d "%~dp0"

echo ========================================================
echo               AgyMolden Launcher (Windows)
echo ========================================================
echo.

REM 1. Check Python installation
set "PY_CMD="
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not detected on your system.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM 2. Locate .whl package without nested parenthesis conflicts
set "WHL_FILE="
for %%f in (*.whl) do (
    set "WHL_FILE=%%f"
)

if "%WHL_FILE%"=="" (
    echo [ERROR] No .whl file found in this folder.
    echo Please make sure the zip file was completely extracted.
    echo.
    pause
    exit /b 1
)

echo Package found: %WHL_FILE%
echo.

REM 3. Create or update virtual environment
if not exist "agymolden_env\Scripts\python.exe" (
    echo [1/2] Setting up local isolated environment, please wait...
    %PY_CMD% -m venv agymolden_env
    call agymolden_env\Scripts\activate.bat
    python -m pip install --upgrade pip >nul 2>&1
    echo [2/2] Installing AgyMolden and dependencies...
    pip install "%WHL_FILE%"
) else (
    call agymolden_env\Scripts\activate.bat
    echo Updating local environment...
    pip install --upgrade --force-reinstall --no-deps "%WHL_FILE%" >nul 2>&1
)

echo.
echo Starting AgyMolden...
start "" agymolden_env\Scripts\agymolden.exe
exit /b 0
