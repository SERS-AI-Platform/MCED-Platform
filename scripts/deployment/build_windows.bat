@echo off
REM ============================================================
REM AECD Software - Windows Build Script
REM ============================================================
REM
REM Prerequisites:
REM   - Python 3.10+ installed
REM   - All project dependencies installed (pip install -e .)
REM
REM Usage:
REM   1. Open Command Prompt in the project root
REM   2. Run: scripts\deployment\build_windows.bat
REM
REM Output:
REM   dist\SERS_Clinical\SERS_Clinical.exe
REM ============================================================

echo.
echo ============================================================
echo   AECD Software - Build Script
echo ============================================================
echo.

cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo ERROR: Failed to locate the project root
    pause
    exit /b 1
)

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Install build dependencies
echo [1/4] Installing build dependencies...
pip install pyinstaller fastapi uvicorn jinja2 python-multipart pyyaml joblib scikit-learn scipy xgboost numpy pandas
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Verify model artifacts exist
echo.
echo [2/4] Verifying model artifacts...
set MODEL_FOUND=0
if exist "artifacts\usersnet\current\manifest.json" (
    set MODEL_FOUND=1
    echo OK: uSERS-Net model found at artifacts\usersnet\current
)
if exist "artifacts\usersnet\v1.0.0\manifest.json" (
    set MODEL_FOUND=1
    echo OK: uSERS-Net model found at artifacts\usersnet\v1.0.0
)
if exist "artifacts\baselines\lr-fusion\v1.0.0\manifest.json" (
    set MODEL_FOUND=1
    echo OK: LR fallback model found at artifacts\baselines\lr-fusion\v1.0.0
)
if %MODEL_FOUND% == 0 (
    echo ERROR: No supported model artifacts found under %CD%\artifacts
    echo Restore artifacts\usersnet\v1.0.0 or artifacts\baselines\lr-fusion\v1.0.0
    pause
    exit /b 1
)

REM Clean previous build
echo.
echo [3/4] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller
echo.
echo [4/4] Building executable with PyInstaller...
echo This may take 5-10 minutes...
echo.
pyinstaller --clean scripts\deployment\sers_clinical.spec
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo.
echo   Executable: dist\SERS_Clinical\SERS_Clinical.exe
echo   Size:
dir /s dist\SERS_Clinical\SERS_Clinical.exe | findstr /I "SERS_Clinical.exe"
echo.
echo   To run:
echo     1. Navigate to dist\SERS_Clinical
echo     2. Double-click SERS_Clinical.exe
echo     3. Browser will open automatically at http://127.0.0.1:8080
echo.
echo   To create a Desktop shortcut:
echo     Right-click SERS_Clinical.exe ^> Send to ^> Desktop (create shortcut)
echo.
pause
