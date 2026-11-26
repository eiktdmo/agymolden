@echo off
cd /d "%~dp0"
if not exist .venv call windows-installer.bat
.venv\Scripts\pip install pyinstaller
if "%1"=="--onedir" (
    set AGYMOLDEN_ONEDIR=1
    .venv\Scripts\pyinstaller --noconfirm agymolden.spec
    echo Directory bundle: dist\agymolden\
) else (
    .venv\Scripts\pyinstaller --noconfirm agymolden.spec
    echo Executable: dist\agymolden.exe
)
