# Smart Paddy Monitoring System - Run Script
# This script starts the Flask backend server

Write-Host "========================================" -ForegroundColor Green
Write-Host "Smart Paddy Monitoring System" -ForegroundColor Green
Write-Host "Starting Application..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if models exist
$modelsExist = $true
$requiredModels = @(
    "models/disease_classification_model.h5",
    "models/yield_prediction_model.pkl",
    "models/yield_loss_model.pkl"
)

Write-Host "Checking trained models..." -ForegroundColor Cyan
foreach ($model in $requiredModels) {
    if (Test-Path $model) {
        Write-Host "  ✓ Found: $model" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $model" -ForegroundColor Red
        $modelsExist = $false
    }
}

if (-not $modelsExist) {
    Write-Host ""
    Write-Host "⚠ Models not found! Please train the models first." -ForegroundColor Yellow
    Write-Host "  Run: .\train_models.ps1" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting Flask Backend Server..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access the application at:" -ForegroundColor Yellow
Write-Host "  http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Yellow
Write-Host "  • GET  /api/health      - Health check" -ForegroundColor White
Write-Host "  • GET  /api/locations   - Get locations" -ForegroundColor White
Write-Host "  • POST /api/predict     - Analyze & predict" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

python app.py
