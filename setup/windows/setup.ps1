# Neuroevolution Project Setup Script for Windows
# This script sets up a Python 3.10 virtual environment and installs dependencies
# Run this script in PowerShell: .\setup\windows\setup.ps1

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Get the directory where this script is located
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "Neuroevolution Project Setup" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

# Change to project root directory
Set-Location $PROJECT_ROOT
Write-ColorOutput "✓ Changed to project directory: $PROJECT_ROOT" "Green"
Write-Host ""

# Check if Python 3.10 is available
Write-ColorOutput "Checking for Python 3.10..." "Cyan"

# Try different Python commands
$pythonCmd = $null
$pythonVersion = $null

# Try python3.10 first
try {
    $versionOutput = & python3.10 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "python3.10"
        $pythonVersion = ($versionOutput -split ' ')[1]
    }
} catch {
    # Try py -3.10 (Python Launcher)
    try {
        $versionOutput = & py -3.10 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "py -3.10"
            $pythonVersion = ($versionOutput -split ' ')[1]
        }
    } catch {
        # Try python (might be 3.10)
        try {
            $versionOutput = & python --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $version = ($versionOutput -split ' ')[1]
                if ($version -match '^3\.10') {
                    $pythonCmd = "python"
                    $pythonVersion = $version
                }
            }
        } catch {
            # Python not found
        }
    }
}

if (-not $pythonCmd) {
    Write-ColorOutput "✗ Python 3.10 not found!" "Red"
    Write-ColorOutput "Please install Python 3.10 first:" "Yellow"
    Write-Host "  Option 1: Download from python.org:"
    Write-Host "    https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "  Option 2: Install via Microsoft Store:"
    Write-Host "    Search for 'Python 3.10' in Microsoft Store"
    Write-Host ""
    Write-Host "  Make sure to check 'Add Python to PATH' during installation!"
    Write-Host ""
    exit 1
}

Write-ColorOutput "✓ Found Python $pythonVersion" "Green"
Write-Host ""

# Check if venv already exists
if (Test-Path "venv") {
    Write-ColorOutput "⚠ Virtual environment 'venv' already exists." "Yellow"
    $response = Read-Host "Do you want to remove it and create a new one? (y/N)"
    if ($response -match '^[Yy]$') {
        Write-ColorOutput "Removing existing virtual environment..." "Cyan"
        Remove-Item -Recurse -Force venv
        Write-ColorOutput "✓ Removed existing virtual environment" "Green"
    } else {
        Write-ColorOutput "Skipping virtual environment creation." "Yellow"
        Write-Host ""
        Write-ColorOutput "Installing/updating dependencies..." "Cyan"
        
        # Activate existing venv
        & "venv\Scripts\Activate.ps1"
        if ($LASTEXITCODE -ne 0) {
            & "venv\Scripts\activate.bat"
        }
        
        & pip install --upgrade pip
        if (Test-Path "requirements.txt") {
            & pip install -r requirements.txt
        } else {
            Write-ColorOutput "✗ requirements.txt not found!" "Red"
            exit 1
        }
        Write-ColorOutput "✓ Dependencies installed" "Green"
        Write-Host ""
        Write-ColorOutput "========================================" "Green"
        Write-ColorOutput "Setup Complete!" "Green"
        Write-ColorOutput "========================================" "Green"
        Write-Host ""
        Write-ColorOutput "To activate the virtual environment, run:" "Yellow"
        Write-ColorOutput "  venv\Scripts\Activate.ps1" "Cyan"
        Write-Host ""
        exit 0
    }
    Write-Host ""
}

# Create virtual environment with Python 3.10
Write-ColorOutput "Creating virtual environment with Python 3.10..." "Cyan"
& $pythonCmd -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "✗ Failed to create virtual environment!" "Red"
    exit 1
}
Write-ColorOutput "✓ Virtual environment created" "Green"
Write-Host ""

# Activate virtual environment
Write-ColorOutput "Activating virtual environment..." "Cyan"
try {
    & "venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        # Fallback to batch file if PowerShell script fails (execution policy issue)
        & "venv\Scripts\activate.bat"
    }
} catch {
    # Try batch file activation
    & "venv\Scripts\activate.bat"
}
Write-ColorOutput "✓ Virtual environment activated" "Green"
Write-Host ""

# Upgrade pip
Write-ColorOutput "Upgrading pip..." "Cyan"
& pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "✗ Failed to upgrade pip!" "Red"
    exit 1
}
Write-ColorOutput "✓ pip upgraded" "Green"
Write-Host ""

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-ColorOutput "✗ requirements.txt not found!" "Red"
    Write-ColorOutput "Creating a basic requirements.txt file..." "Yellow"
    @"
torch>=2.0.0
numpy>=1.24.0
"@ | Out-File -FilePath "requirements.txt" -Encoding utf8
    Write-ColorOutput "✓ Created requirements.txt" "Green"
    Write-Host ""
}

# Install dependencies
Write-ColorOutput "Installing dependencies from requirements.txt..." "Cyan"
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "✗ Failed to install dependencies!" "Red"
    exit 1
}
Write-ColorOutput "✓ Dependencies installed" "Green"
Write-Host ""

# Verify installation
Write-ColorOutput "Verifying installation..." "Cyan"
try {
    $torchVersion = & python -c "import torch; print(torch.__version__)" 2>&1
    $numpyVersion = & python -c "import numpy; print(numpy.__version__)" 2>&1
    Write-Host "PyTorch version: $torchVersion"
    Write-Host "NumPy version: $numpyVersion"
    Write-ColorOutput "✓ Installation verified" "Green"
} catch {
    Write-ColorOutput "✗ Verification failed!" "Red"
    Write-Host $_.Exception.Message
    exit 1
}
Write-Host ""

# Success message
Write-ColorOutput "========================================" "Green"
Write-ColorOutput "Setup Complete!" "Green"
Write-ColorOutput "========================================" "Green"
Write-Host ""
Write-ColorOutput "To activate the virtual environment, run:" "Yellow"
Write-ColorOutput "  venv\Scripts\Activate.ps1" "Cyan"
Write-Host ""
Write-ColorOutput "If you get an execution policy error, run:" "Yellow"
Write-ColorOutput "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" "Cyan"
Write-ColorOutput "  (Then try activating again)" "Cyan"
Write-Host ""
Write-ColorOutput "Or use the batch file instead:" "Yellow"
Write-ColorOutput "  venv\Scripts\activate.bat" "Cyan"
Write-Host ""
Write-ColorOutput "To deactivate the virtual environment, run:" "Yellow"
Write-ColorOutput "  deactivate" "Cyan"
Write-Host ""
Write-ColorOutput "To run the project, use:" "Yellow"
Write-ColorOutput "  python run_all.py" "Cyan"
Write-Host ""

