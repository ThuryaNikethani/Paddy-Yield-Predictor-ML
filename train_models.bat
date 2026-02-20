@echo off
REM Smart Paddy Monitoring System - Model Training Script (Windows CMD)

echo ========================================
echo Smart Paddy Monitoring System
echo Model Training Script
echo ========================================
echo.

echo [1/2] Training Disease Classification Model...
echo   Dataset: 1,320+ rice leaf disease images
echo   Model: CNN with MobileNetV2 transfer learning
echo   Expected time: 30-60 minutes
echo.
python train_disease_model.py
if errorlevel 1 (
    echo.
    echo   X Disease model training failed!
    pause
    exit /b 1
)
echo.
echo   √ Disease classification model trained successfully!

echo.
echo ========================================
echo.

echo [2/2] Training Yield Prediction Model...
echo   Dataset: Weather, location, and production data
echo   Model: Gradient Boosting Regressor
echo   Expected time: 5-10 minutes
echo.
python train_yield_model.py
if errorlevel 1 (
    echo.
    echo   X Yield model training failed!
    pause
    exit /b 1
)
echo.
echo   √ Yield prediction model trained successfully!

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Models saved in 'models/' directory
echo.
echo Next Step:
echo Run: run.bat    (Start the application)
echo.
pause
