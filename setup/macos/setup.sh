#!/bin/zsh
#
# Neuroevolution Project Setup Script for macOS
# This script sets up a Python 3.10 virtual environment and installs dependencies
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
if [[ -n "$ZSH_VERSION" ]]; then
    SCRIPT_DIR="$( cd "$( dirname "${(%):-%x}" )" && pwd )"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Neuroevolution Project Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓${NC} Changed to project directory: $PROJECT_ROOT"
echo ""

# Check if Python 3.10 is available
echo -e "${BLUE}Checking for Python 3.10...${NC}"
if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}✗${NC} Python 3.10 not found!"
    echo -e "${YELLOW}Please install Python 3.10 first:${NC}"
    echo "  Option 1: Install via Homebrew:"
    echo "    brew install python@3.10"
    echo ""
    echo "  Option 2: Download from python.org:"
    echo "    https://www.python.org/downloads/"
    echo ""
    exit 1
fi

PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Found Python $PYTHON_VERSION"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual environment 'venv' already exists."
    read "response?Do you want to remove it and create a new one? (y/N): "
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing virtual environment...${NC}"
        rm -rf venv
        echo -e "${GREEN}✓${NC} Removed existing virtual environment"
    else
        echo -e "${YELLOW}Skipping virtual environment creation.${NC}"
        echo ""
        echo -e "${BLUE}Installing/updating dependencies...${NC}"
        source venv/bin/activate
        pip install --upgrade pip
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        else
            echo -e "${RED}✗${NC} requirements.txt not found!"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} Dependencies installed"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Setup Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
        echo -e "  ${BLUE}source venv/bin/activate${NC}"
        echo ""
        exit 0
    fi
    echo ""
fi

# Create virtual environment with Python 3.10
echo -e "${BLUE}Creating virtual environment with Python 3.10...${NC}"
python3.10 -m venv venv
echo -e "${GREEN}✓${NC} Virtual environment created"
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗${NC} requirements.txt not found!"
    echo -e "${YELLOW}Creating a basic requirements.txt file...${NC}"
    cat > requirements.txt << EOF
torch>=2.0.0
numpy>=1.24.0
EOF
    echo -e "${GREEN}✓${NC} Created requirements.txt"
    echo ""
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python3 -c "import torch; import numpy; print(f'PyTorch version: {torch.__version__}'); print(f'NumPy version: {numpy.__version__}')" || {
    echo -e "${RED}✗${NC} Verification failed!"
    exit 1
}
echo -e "${GREEN}✓${NC} Installation verified"
echo ""

# Success message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
echo -e "  ${BLUE}source venv/bin/activate${NC}"
echo ""
echo -e "${YELLOW}To deactivate the virtual environment, run:${NC}"
echo -e "  ${BLUE}deactivate${NC}"
echo ""
echo -e "${YELLOW}To run the project, use:${NC}"
echo -e "  ${BLUE}python3 run_all.py${NC}"
echo ""

