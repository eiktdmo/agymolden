@echo off
setlocal enabledelayedexpansion
rem ==============================================================================
rem AgyMolden Automated 1-Command Installer (Windows — no conda required)
rem ==============================================================================
cd /d "%~dp0"
set "AGY_ROOT=%~dp0"
if "%AGY_ROOT:~-1%"=="\" set "AGY_ROOT=%AGY_ROOT:~0,-1%"

set "NO_PAUSE=0"
if /i "%~1"=="--no-pause" set "NO_PAUSE=1"

echo ==========================================================
echo          AgyMolden Automatic Source Installer
echo ==========================================================
echo.

rem 1. Detect Python 3.9+ (prefer Windows 'py -3' launcher, then 'python')
set "PY_CMD="
py -3 -c "import sys, venv; assert sys.version_info >= (3, 9)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3"
) else (
    python -c "import sys, venv; assert sys.version_info >= (3, 9)" >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.9 or newer was not found.
    echo Download Python 3.11+ from https://www.python.org/downloads/windows/
    echo Make sure to check "Add Python to PATH" during installation.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

for /f "delims=" %%V in ('%PY_CMD% -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%V"
echo [OK] Found Python %PY_VER% (%PY_CMD%)

rem 2. Create isolated .venv
echo.
echo Setting up isolated virtual environment in .venv...
if not exist ".venv\Scripts\python.exe" (
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv virtual environment.
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )
)

rem 3. Install core dependencies first (must succeed), then optional chemistry extras
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1

echo Installing core AgyMolden dependencies...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Core installation failed. Check your internet connection and try again.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Installing optional chemistry extras (best-effort)...
for %%P in (basis_set_exchange rdkit ase spglib) do (
    ".venv\Scripts\python.exe" -m pip install %%P >nul 2>&1 && echo   [OK] %%P || echo   [--] %%P skipped ^(optional^)
)

rem 4. Verify installation
echo.
".venv\Scripts\python.exe" -c "import agymolden, PyQt6, numpy, scipy, matplotlib; print('[OK] Verified AgyMolden', agymolden.__version__)"
if errorlevel 1 (
    echo [ERROR] Verification import failed.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

rem 5. Create Desktop shortcut (handles spaces in folder paths safely)
echo.
echo Creating Desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = $env:AGY_ROOT; $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'AgyMolden.lnk')); $s.TargetPath = [System.IO.Path]::Combine($root, '.venv', 'Scripts', 'pythonw.exe'); $s.Arguments = '\"' + [System.IO.Path]::Combine($root, 'run.py') + '\"'; $s.WorkingDirectory = $root; $s.Description = 'AgyMolden Molecular Builder and Viewer'; $s.Save()" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Desktop shortcut created: AgyMolden.lnk
)

echo.
echo ==========================================================
echo [OK] Installation completed successfully!
echo.
echo Launch AgyMolden anytime with:
echo    .venv\Scripts\agymolden.exe
echo    py -3 run.py
echo    (or double-click the AgyMolden Desktop shortcut)
echo ==========================================================
echo.
if "%NO_PAUSE%"=="0" pause
