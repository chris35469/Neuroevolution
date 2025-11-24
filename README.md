This project is not yet ready for generic use. I am using it for drone recorded data currently.
I'll work on redoing it to be useful for other neuroevolutionary projects later...

## Setup

This project requires Python 3.10 and uses a virtual environment to manage dependencies.

### Prerequisites

- **Python 3.10** (must be installed and available in your PATH)
- **Git** (to clone the repository)

### macOS Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd Neuroevolution
   ```

2. **Run the setup script**:
   ```bash
   ./setup/macos/setup.sh
   ```
   
   Or if you prefer:
   ```bash
   zsh setup/macos/setup.sh
   ```

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Verify installation**:
   ```bash
   python3 run_all.py
   ```

### Windows Setup

1. **Clone the repository** (if you haven't already):
   ```powershell
   git clone <repository-url>
   cd Neuroevolution
   ```

2. **Run the setup script**:
   ```powershell
   .\setup\windows\setup.ps1
   ```
   
   **Note**: If you encounter an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Then try running the setup script again.

3. **Activate the virtual environment**:
   ```powershell
   venv\Scripts\Activate.ps1
   ```
   
   If PowerShell scripts are blocked, use the batch file instead:
   ```powershell
   venv\Scripts\activate.bat
   ```

4. **Verify installation**:
   ```powershell
   python run_all.py
   ```

### Manual Setup (Alternative)

If you prefer to set up manually:

1. **Create a virtual environment**:
   - macOS/Linux: `python3.10 -m venv venv`
   - Windows: `py -3.10 -m venv venv` or `python3.10 -m venv venv`

2. **Activate the virtual environment**:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\Activate.ps1` or `venv\Scripts\activate.bat`

3. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting

- **Python 3.10 not found**: Make sure Python 3.10 is installed and added to your PATH. You can verify by running `python3.10 --version` (macOS) or `py -3.10 --version` (Windows).

- **Virtual environment activation fails**: Make sure you're in the project root directory and the `venv` folder exists.

- **Dependencies fail to install**: Ensure you have an active internet connection and pip is up to date. Try upgrading pip first: `pip install --upgrade pip`

- **PowerShell execution policy error** (Windows): Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in an administrator PowerShell window.