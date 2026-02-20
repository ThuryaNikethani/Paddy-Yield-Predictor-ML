@echo off
REM Smart Paddy Monitoring System - Setup Script (Windows CMD)

echo ========================================
echo Smart Paddy Monitoring System
echo Installation Script
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   X Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo   √ Python found

echo.
echo [2/4] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo   X pip not found! Please install pip.
    pause
    exit /b 1
)
echo   √ pip found

echo.
echo [3/4] Installing Python packages...
echo   This may take 5-10 minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   X Installation failed!
    pause
    exit /b 1
)
echo.
echo   √ All packages installed successfully!

echo.
echo [4/4] Creating directories...
if not exist "models" mkdir models
if not exist "uploads" mkdir uploads
if not exist "templates" mkdir templates
echo   √ Directories created

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Run: train_models.bat    (Train AI models)
echo 2. Run: run.bat             (Start application)
echo.
pause
