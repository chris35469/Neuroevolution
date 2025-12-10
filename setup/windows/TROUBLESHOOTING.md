# PyTorch DLL Loading Error Troubleshooting

## Error: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

This is a common Windows issue with PyTorch. Here are solutions in order of likelihood to fix:

### Solution 1: Install Visual C++ Redistributables (Most Common Fix)

PyTorch requires Microsoft Visual C++ Redistributables. Download and install:

1. **Visual C++ Redistributable 2015-2022 (x64)**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run the installer and restart your computer

2. **Visual C++ Redistributable 2015-2022 (x86)** (if needed)
   - Download from: https://aka.ms/vs/17/release/vc_redist.x86.exe

### Solution 2: Reinstall PyTorch with CPU-only Version

If you don't need GPU support, try the CPU-only version:

```powershell
# Activate your virtual environment first
.\venv\Scripts\Activate.ps1

# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Solution 3: Reinstall PyTorch with Specific Index

Try installing from PyTorch's official index:

```powershell
# Activate your virtual environment first
.\venv\Scripts\Activate.ps1

# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall from official index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Solution 4: Use Conda Instead of pip

If pip continues to have issues, consider using conda:

```powershell
# Install Miniconda or Anaconda first, then:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Solution 5: Check Python Version Compatibility

Ensure you're using Python 3.10 (as required). PyTorch 2.0+ requires Python 3.8-3.11.

### Solution 6: Clean Reinstall

If all else fails, try a complete clean reinstall:

```powershell
# Deactivate and remove virtual environment
deactivate
Remove-Item -Recurse -Force venv

# Recreate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy>=1.24.0
```

### Quick Test

After applying any solution, test if PyTorch loads correctly:

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')"
```

If this works, your PyTorch installation is fixed!







