# Smart Paddy Monitoring System - Setup Script
# This script installs all required dependencies

Write-Host "========================================" -ForegroundColor Green
Write-Host "Smart Paddy Monitoring System" -ForegroundColor Green
Write-Host "Installation Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host ""
Write-Host "[2/4] Checking pip..." -ForegroundColor Cyan
try {
    $pipVersion = pip --version 2>&1
    Write-Host "  ✓ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ pip not found! Please install pip." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "[3/4] Installing Python packages..." -ForegroundColor Cyan
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Yellow
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  ✓ All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ✗ Installation failed! Please check the error messages above." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "[4/4] Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "templates" | Out-Null
Write-Host "  ✓ Directories created" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\train_models.ps1    (Train AI models)" -ForegroundColor White
Write-Host "2. Run: .\run.ps1             (Start application)" -ForegroundColor White
Write-Host ""
