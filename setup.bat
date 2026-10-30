@echo off
rem ==============================================================================
rem AgyMolden Automated Source Installer (Windows)
rem ==============================================================================
cd /d "%~dp0"

echo ==========================================================
echo          AgyMolden Automatic Source Installer            
echo ==========================================================
echo.

rem Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not found in your PATH.
    echo Please install Python 3.9+ from https://www.python.org/ and check "Add Python to PATH".
    pause
    exit /b 1
)

echo Setting up Python virtual environment in .venv...
if not exist .venv (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Upgrading pip and installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -e ".[all]"
if %errorlevel% neq 0 (
    echo Falling back to core dependencies...
    .venv\Scripts\pip.exe install -e .
)

echo.
echo Creating Desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'AgyMolden.lnk')); $s.TargetPath = [System.IO.Path]::Combine('%~dp0', '.venv', 'Scripts', 'pythonw.exe'); $s.Arguments = [System.IO.Path]::Combine('%~dp0', 'run.py'); $s.WorkingDirectory = '%~dp0'; $s.Description = 'AgyMolden Molecular Builder and Viewer'; $s.Save()" >nul 2>&1
if %errorlevel% equ 0 (
    echo Desktop shortcut created: AgyMolden.lnk
)

echo.
echo ==========================================================
echo Installation completed successfully!
echo.
echo You can run AgyMolden with:
echo    python run.py
echo (or double-click the AgyMolden desktop shortcut)
echo ==========================================================
echo.
pause