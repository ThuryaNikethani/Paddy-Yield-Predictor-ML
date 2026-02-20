# Smart Paddy Monitoring System - Model Training Script
# This script trains both disease classification and yield prediction models

Write-Host "========================================" -ForegroundColor Green
Write-Host "Smart Paddy Monitoring System" -ForegroundColor Green
Write-Host "Model Training Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$totalSteps = 2
$currentStep = 0

# Train Disease Classification Model
$currentStep++
Write-Host "[$currentStep/$totalSteps] Training Disease Classification Model..." -ForegroundColor Cyan
Write-Host "  Dataset: 1,320+ rice leaf disease images" -ForegroundColor White
Write-Host "  Model: CNN with MobileNetV2 transfer learning" -ForegroundColor White
Write-Host "  Expected time: 30-60 minutes" -ForegroundColor Yellow
Write-Host ""

python train_disease_model.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  ✓ Disease classification model trained successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ✗ Disease model training failed!" -ForegroundColor Red
    Write-Host "  Please check the error messages above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Train Yield Prediction Model
$currentStep++
Write-Host "[$currentStep/$totalSteps] Training Yield Prediction Model..." -ForegroundColor Cyan
Write-Host "  Dataset: Weather, location, and production data" -ForegroundColor White
Write-Host "  Model: Gradient Boosting Regressor" -ForegroundColor White
Write-Host "  Expected time: 5-10 minutes" -ForegroundColor Yellow
Write-Host ""

python train_yield_model.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  ✓ Yield prediction model trained successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ✗ Yield model training failed!" -ForegroundColor Red
    Write-Host "  Please check the error messages above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Models saved in 'models/' directory:" -ForegroundColor Yellow
Write-Host "  • disease_classification_model.h5" -ForegroundColor White
Write-Host "  • yield_prediction_model.pkl" -ForegroundColor White
Write-Host "  • yield_loss_model.pkl" -ForegroundColor White
Write-Host "  • feature_scaler.pkl" -ForegroundColor White
Write-Host "  • season_encoder.pkl" -ForegroundColor White
Write-Host ""
Write-Host "Next Step:" -ForegroundColor Yellow
Write-Host "Run: .\run.ps1    (Start the application)" -ForegroundColor White
Write-Host ""
