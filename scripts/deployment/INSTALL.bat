@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM SERS Clinical Webapp - One-Click Installer
REM ============================================================

REM Auto-navigate to project root (parent of scripts\deployment)
cd /d "%~dp0..\..\"

title SERS Clinical Webapp - Installer

color 0B
cls
echo.
echo  ============================================================
echo.
echo                SERS Cancer Screening System
echo                    SOLUM Healthcare
echo.
echo                  Installer v1.0
echo.
echo  ============================================================
echo.
echo   Install location: %CD%
echo.
echo   This installer will automatically:
echo     1. Check Python environment
echo     2. Install required libraries
echo     3. Build executable (5-10 minutes)
echo     4. Create desktop shortcut
echo.
echo  ============================================================
echo.
pause
echo.

REM -------------------------------------------------------------
REM Step 1: Check Python
REM -------------------------------------------------------------
echo [1/5] Checking Python environment...
where python >nul 2>nul
if errorlevel 1 (
    color 0C
    echo.
    echo  ERROR: Python is not installed.
    echo.
    echo  Please install Python 3.10 or later from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo       OK - !PYVER!
echo.

REM -------------------------------------------------------------
REM Step 2: Verify project files
REM -------------------------------------------------------------
echo [2/5] Verifying project files...
if not exist "scripts\deployment\sers_clinical_webapp.py" (
    color 0C
    echo       ERROR: Project files not found.
    echo       Run this installer from inside the SERS-Clinical-App folder.
    echo       Current path: %CD%
    pause
    exit /b 1
)
echo       OK - Project files verified

REM Check at least one model exists
set MODEL_FOUND=0
if exist "artifacts\usersnet\current\manifest.json" (
    set MODEL_FOUND=1
    echo       OK - uSERS-Net model found
)
if exist "artifacts\baselines\lr-fusion\v1.0.0\manifest.json" (
    set MODEL_FOUND=1
    echo       OK - LR model found
)
if !MODEL_FOUND! == 0 (
    color 0C
    echo.
    echo       ERROR: No model files found.
    echo       Need either artifacts\usersnet\current or artifacts\baselines\lr-fusion\v1.0.0 folder.
    echo.
    echo       If using OneDrive, right-click the folder and select
    echo       "Always keep on this device" to download files.
    pause
    exit /b 1
)
echo.

REM -------------------------------------------------------------
REM Step 3: Install dependencies
REM -------------------------------------------------------------
echo [3/5] Installing libraries... (1-2 minutes)
echo.
python -m pip install --quiet --upgrade pip
python -m pip install --quiet pyinstaller fastapi uvicorn jinja2 python-multipart joblib scikit-learn scipy xgboost numpy pandas
if errorlevel 1 (
    color 0C
    echo.
    echo       ERROR: Failed to install libraries.
    echo       Please check your internet connection and try again.
    pause
    exit /b 1
)
echo       OK - All libraries installed
echo.

REM -------------------------------------------------------------
REM Step 4: Build executable
REM -------------------------------------------------------------
echo [4/5] Building executable... (5-10 minutes, please wait)
echo.

REM Clean previous build
if exist build rmdir /s /q build >nul 2>&1
if exist dist rmdir /s /q dist >nul 2>&1

REM Run PyInstaller
pyinstaller --clean --noconfirm scripts\deployment\sers_clinical.spec
if errorlevel 1 (
    color 0C
    echo.
    echo       ERROR: Build failed.
    echo       Please check the error messages above.
    pause
    exit /b 1
)

if not exist "dist\SERS_Clinical\SERS_Clinical.exe" (
    color 0C
    echo.
    echo       ERROR: Build completed but executable not found.
    pause
    exit /b 1
)
echo.
echo       OK - Build complete: dist\SERS_Clinical\SERS_Clinical.exe
echo.

REM -------------------------------------------------------------
REM Step 5: Create desktop shortcut
REM -------------------------------------------------------------
echo [5/5] Creating desktop shortcut...

set EXEPATH=%CD%\dist\SERS_Clinical\SERS_Clinical.exe
set ICONPATH=%CD%\scripts\deployment\static\icon.ico
set SHORTCUT=%USERPROFILE%\Desktop\SERS Clinical.lnk

REM Use PowerShell to create shortcut
powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$sc = $ws.CreateShortcut('%SHORTCUT%');" ^
    "$sc.TargetPath = '%EXEPATH%';" ^
    "$sc.WorkingDirectory = '%CD%\dist\SERS_Clinical';" ^
    "$sc.Description = 'SERS Cancer Screening System';" ^
    "if (Test-Path '%ICONPATH%') { $sc.IconLocation = '%ICONPATH%' };" ^
    "$sc.Save()"

if exist "%SHORTCUT%" (
    echo       OK - Desktop shortcut created
) else (
    echo       WARNING: Failed to create shortcut (create manually)
)
echo.

REM -------------------------------------------------------------
REM Done!
REM -------------------------------------------------------------
color 0A
echo.
echo  ============================================================
echo.
echo                  Installation Complete!
echo.
echo  ============================================================
echo.
echo   How to run:
echo     1. Double-click "SERS Clinical" icon on desktop
echo     2. Or run directly:
echo        %EXEPATH%
echo.
echo   Default accounts:
echo     - admin   / admin123  (Admin)
echo     - tech1   / admin123  (Lab Technician)
echo     - doctor1 / admin123  (Clinician)
echo.
echo   Browser will open automatically (http://127.0.0.1:8080)
echo.
echo  ============================================================
echo.

REM Ask if user wants to launch now
choice /c YN /m "Launch now"
if !errorlevel! == 1 (
    echo.
    echo Starting program...
    start "" "%EXEPATH%"
    timeout /t 3 >nul
)

exit /b 0
