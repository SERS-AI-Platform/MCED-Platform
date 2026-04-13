@echo off
REM SERS Pipeline - Initial Setup Script for Windows
REM Run this once after cloning/extracting the repository

echo ========================================
echo SERS Analysis Pipeline - Initial Setup
echo ========================================
echo.

REM Check if Git is available
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Check if in correct directory
if not exist "pyproject.toml" (
    echo ERROR: Please run this script from the project root directory
    echo        where pyproject.toml is located
    pause
    exit /b 1
)

echo [1/5] Initializing Git repository...
if not exist ".git" (
    git init
    echo Git repository initialized.
) else (
    echo Git repository already exists.
)

echo.
echo [2/5] Setting up Git configuration...
git config core.autocrlf true
echo Configured line endings for Windows.

echo.
echo [3/5] Creating initial commit...
git add -A
git commit -m "chore: initial commit - SERS analysis pipeline v0.1.0" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Initial commit created.
) else (
    echo Already committed or nothing to commit.
)

echo.
echo [4/5] Creating develop branch...
git branch develop 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Develop branch created.
) else (
    echo Develop branch already exists.
)

echo.
echo [5/5] Installing pre-commit hooks...
where pip >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pip install pre-commit >nul 2>nul
    if exist ".pre-commit-config.yaml" (
        pre-commit install
        echo Pre-commit hooks installed.
    )
) else (
    echo Skipping pre-commit (pip not found)
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo   1. Create a repository on GitHub
echo   2. Run: git remote add origin https://github.com/YOUR_ORG/sers-analysis.git
echo   3. Run: git push -u origin main
echo   4. Run: git push origin develop
echo.
echo To install the package:
echo   pip install -e .
echo   OR
echo   conda env create -f environment.yml
echo.
pause
