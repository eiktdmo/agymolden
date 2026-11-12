@echo off
title AgyMolden One-Click Launcher
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ========================================================
echo               AgyMolden Launcher (Windows)
echo ========================================================
echo.

REM 1. Check Python installation across python, py, and standard Windows paths
set "PY_CMD="
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
) else (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3"
    ) else (
        py --version >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=py"
        )
    )
)

REM Check common AppData / ProgramFiles install paths if not in PATH
if "%PY_CMD%"=="" (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" (
            set "PY_CMD=%%D\python.exe"
        )
    )
)

REM 2. If Python is still missing, attempt automated 1-click winget install
if "%PY_CMD%"=="" (
    echo [NOTICE] Python 3 was not detected in your system PATH.
    echo Attempting automatic installation via Windows Package Manager (winget)...
    echo.
    where winget >nul 2>&1
    if not errorlevel 1 (
        echo Installing official Python 3.12 (silent background setup)...
        winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
        if not errorlevel 1 (
            echo.
            echo [SUCCESS] Python 3.12 installed successfully!
            REM Refresh path for this session
            for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
                if exist "%%D\python.exe" set "PY_CMD=%%D\python.exe"
            )
            if "%PY_CMD%"=="" (
                python --version >nul 2>&1 && set "PY_CMD=python"
            )
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3 was not detected on your system.
    echo.
    echo Please install Python 3 (3.10, 3.11, or 3.12) from:
    echo   https://www.python.org/downloads/
    echo.
    echo [IMPORTANT] Make sure to check the box "Add Python to PATH" during setup!
    echo.
    pause
    exit /b 1
)

echo Found Python engine: %PY_CMD%

REM 3. Locate .whl package
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

REM 4. Create or update virtual environment
if not exist "agymolden_env\Scripts\python.exe" (
    echo [1/2] Setting up local isolated environment in .\agymolden_env...
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
