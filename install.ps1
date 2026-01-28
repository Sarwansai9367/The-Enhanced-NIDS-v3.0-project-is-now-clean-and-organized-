# Real-Time NIDS Installation Script
# Run this to set up the complete system

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  Real-Time Network Intrusion Detection System (NIDS)" -ForegroundColor Cyan
Write-Host "  Installation & Setup Script" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found. Please install Python 3.8 or higher" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "[2/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "  ✓ pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Check for Npcap (Windows)
Write-Host ""
Write-Host "[4/6] Checking for Npcap (required for packet capture)..." -ForegroundColor Yellow
$npcapPath = "C:\Windows\System32\Npcap"
if (Test-Path $npcapPath) {
    Write-Host "  ✓ Npcap is installed" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Npcap not found" -ForegroundColor Yellow
    Write-Host "    For real packet capture, install Npcap from:" -ForegroundColor Yellow
    Write-Host "    https://nmap.org/npcap/" -ForegroundColor Cyan
    Write-Host "    (Not required for simulation mode)" -ForegroundColor Gray
}

# Create necessary directories
Write-Host ""
Write-Host "[5/6] Creating directories..." -ForegroundColor Yellow
$directories = @("datasets", "templates", "logs", "models")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  ✓ Created $dir/" -ForegroundColor Green
    } else {
        Write-Host "  ✓ $dir/ already exists" -ForegroundColor Gray
    }
}

# Download sample dataset
Write-Host ""
Write-Host "[6/6] Checking for training dataset..." -ForegroundColor Yellow
$datasetPath = "datasets\KDDTrain+.txt"
if (Test-Path $datasetPath) {
    Write-Host "  ✓ NSL-KDD dataset found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ NSL-KDD dataset not found" -ForegroundColor Yellow
    Write-Host "    The system will auto-download on first run" -ForegroundColor Gray
    Write-Host "    Or manually download from:" -ForegroundColor Yellow
    Write-Host "    https://www.unb.ca/cic/datasets/nsl.html" -ForegroundColor Cyan
}

# Installation complete
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host "  ✓ Installation Complete!" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Quick Start Options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Run Quick Demo (recommended):" -ForegroundColor White
Write-Host "     python quickstart.py --demo" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Run Original System (dataset-based):" -ForegroundColor White
Write-Host "     python main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Run Real-Time System:" -ForegroundColor White
Write-Host "     python realtime_nids.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  4. Start Web Dashboard:" -ForegroundColor White
Write-Host "     python dashboard.py" -ForegroundColor Yellow
Write-Host "     Then open: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  5. Run Test Suite:" -ForegroundColor White
Write-Host "     python test_realtime.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "   - REALTIME_README.md - Complete real-time features guide" -ForegroundColor Gray
Write-Host "   - README.md - Original system documentation" -ForegroundColor Gray
Write-Host "   - QUICKSTART.md - Quick start guide" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠️  Important Notes:" -ForegroundColor Yellow
Write-Host "   - For real packet capture, run as Administrator" -ForegroundColor Gray
Write-Host "   - Simulation mode works without admin privileges" -ForegroundColor Gray
Write-Host "   - Configure alerts in alert_config.json" -ForegroundColor Gray
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# Ask to run quick demo
$runDemo = Read-Host "Would you like to run the quick demo now? (y/n)"
if ($runDemo -eq "y" -or $runDemo -eq "Y") {
    Write-Host ""
    Write-Host "Starting quick demo..." -ForegroundColor Green
    Write-Host ""
    python quickstart.py --demo
}
