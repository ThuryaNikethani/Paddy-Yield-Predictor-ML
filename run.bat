@echo off
REM Smart Paddy Monitoring System - Run Script (Windows CMD)

echo ========================================
echo Smart Paddy Monitoring System
echo Starting Application...
echo ========================================
echo.

echo Checking trained models...
set MODELS_OK=1

if not exist "models\disease_classification_model.h5" (
    echo   X Missing: disease_classification_model.h5
    set MODELS_OK=0
)
if not exist "models\yield_prediction_model.pkl" (
    echo   X Missing: yield_prediction_model.pkl
    set MODELS_OK=0
)
if not exist "models\yield_loss_model.pkl" (
    echo   X Missing: yield_loss_model.pkl
    set MODELS_OK=0
)

if "%MODELS_OK%"=="0" (
    echo.
    echo WARNING: Models not found! Please train the models first.
    echo   Run: train_models.bat
    echo.
    pause
)

echo.
echo ========================================
echo Starting Flask Backend Server...
echo ========================================
echo.
echo Access the application at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python app.py
pause
