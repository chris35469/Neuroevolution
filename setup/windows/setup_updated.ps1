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
Write-ColorOutput "[OK] Changed to project directory: $PROJECT_ROOT" "Green"
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
    Write-ColorOutput "[ERROR] Python 3.10 not found!" "Red"
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

Write-ColorOutput "[OK] Found Python $pythonVersion" "Green"
Write-Host ""
