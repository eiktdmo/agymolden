@echo off
REM ====================================================================
REM  AgyMolden Windows Setup & Standalone Build Script
REM ====================================================================

echo ========================================================
echo        AgyMolden - Windows Setup & Executable Builder
echo ========================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 from python.org and check 'Add to PATH'.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
)

echo [2/4] Activating environment and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[all]" pyinstaller

echo [3/4] Compiling standalone agymolden.exe using PyInstaller...
pyinstaller agymolden.spec --noconfirm --clean

echo.
echo ========================================================
if exist "dist\agymolden.exe" (
    echo [SUCCESS] Windows executable created successfully!
    echo Binary located at: dist\agymolden.exe
) else (
    echo [FAILED] Build did not produce dist\agymolden.exe. Check output above.
)
echo ========================================================
pause
