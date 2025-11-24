#!/usr/bin/env python3
"""
Script to run all run_*.py files in separate terminal windows.
Works on macOS using osascript to open new Terminal windows.
"""

import os
import glob
import subprocess
import sys
from pathlib import Path
from utils import json_to_csv

# ============================================================================
# Global Configuration Variables
# These can be imported by run_*.py files to share common settings
# ============================================================================

# Genetic Algorithm Parameters
GENERATIONS = 20
POPULATION_SIZE = 20
BEST_INDIVIDUALS_SIZE = 7  # The number of best individuals to keep from every generation (elites)
CROSSOVER_RATE = 0.5  # Probability of taking weights from second parent
MUTATION_RATE = 0.1  # Probability of mutating each weight
MUTATION_STRENGTH = 0.1  # Strength of mutation (standard deviation of noise)

# Training Configuration (for run_train_*.py files)
TRAINING_EPOCHS = 10  # Number of epochs to train each individual
TRAINING_LEARNING_RATE = 0.001  # Learning rate for training
TRAINING_VERBOSE = False  # Set to True to see training progress for each individual

# Configuration Flags
SAVE_MODELS = False  # Set to True if you need to save all models (SLOW!)
SAVE_CSV_RESULTS = True  # Set to False to skip CSV saving

# ============================================================================

def get_run_files():
    """Find all run_*.py files in the current directory."""
    current_dir = Path(__file__).parent
    current_script = Path(__file__).name  # Get run_all.py filename
    run_files = glob.glob(str(current_dir / 'run_*.py'))
    # Filter out files in venv or other subdirectories, and exclude run_all.py itself
    run_files = [f for f in run_files 
                 if os.path.dirname(f) == str(current_dir) 
                 and os.path.basename(f) != current_script]
    return sorted(run_files)

def run_in_terminal_macos(script_path):
    """Open a new Terminal window and run the script on macOS."""
    script_path = os.path.abspath(script_path)
    script_name = os.path.basename(script_path)
    working_dir = os.path.dirname(script_path)
    
    # AppleScript to open new Terminal window and run the script
    applescript = f'''
    tell application "Terminal"
        activate
        do script "cd '{working_dir}' && python3 {script_name}"
    end tell
    '''
    
    subprocess.run(['osascript', '-e', applescript], check=False)

def run_in_terminal_linux(script_path):
    """Open a new terminal window and run the script on Linux."""
    script_path = os.path.abspath(script_path)
    script_name = os.path.basename(script_path)
    working_dir = os.path.dirname(script_path)
    
    # Try different terminal emulators
    terminals = ['gnome-terminal', 'xterm', 'konsole', 'terminator']
    
    for term in terminals:
        try:
            if term == 'gnome-terminal':
                subprocess.Popen([term, '--', 'bash', '-c', 
                                f'cd "{working_dir}" && python3 {script_name}; exec bash'])
            elif term == 'xterm':
                subprocess.Popen([term, '-e', 
                                f'bash -c "cd \\"{working_dir}\\" && python3 {script_name}; exec bash"'])
            elif term == 'konsole':
                subprocess.Popen([term, '-e', 
                                f'bash -c "cd \\"{working_dir}\\" && python3 {script_name}; exec bash"'])
            elif term == 'terminator':
                subprocess.Popen([term, '-e', 
                                f'bash -c "cd \\"{working_dir}\\" && python3 {script_name}; exec bash"'])
            return
        except FileNotFoundError:
            continue
    
    print(f"Error: No terminal emulator found. Please run manually: python3 {script_path}")

def run_in_terminal_windows(script_path):
    """Open a new command prompt window and run the script on Windows."""
    script_path = os.path.abspath(script_path)
    script_name = os.path.basename(script_path)
    working_dir = os.path.dirname(script_path)
    
    # Use start command to open new cmd window
    cmd = f'start cmd /k "cd /d "{working_dir}" && python {script_name}"'
    subprocess.Popen(cmd, shell=True)

def main():
    """Main function to run all run_*.py files in separate terminals."""
    run_files = get_run_files()
    
    if not run_files:
        print("No run_*.py files found in the current directory.")
        return
    
    print(f"Found {len(run_files)} run file(s):")
    for i, run_file in enumerate(run_files, 1):
        print(f"  {i}. {os.path.basename(run_file)}")
    
    # Convert JSON to CSV once before running all scripts
    json_file = 'data/dummy_data.json'
    csv_output_dir = 'data/csv_output'
    
    if os.path.exists(json_file):
        print(f"\nConverting JSON to CSV (one-time setup)...")
        try:
            json_to_csv(json_file, csv_output_dir)
            print(f"✓ JSON to CSV conversion complete")
        except Exception as e:
            print(f"⚠ Warning: JSON to CSV conversion failed: {e}")
            print(f"  Continuing anyway - scripts will handle their own data loading...")
    else:
        print(f"\n⚠ Warning: {json_file} not found. Skipping conversion.")
        print(f"  Scripts will handle their own data loading...")
    
    print(f"\nOpening {len(run_files)} terminal window(s)...")
    
    # Determine OS and use appropriate method
    system = sys.platform
    
    for run_file in run_files:
        script_name = os.path.basename(run_file)
        print(f"  → Launching {script_name}...")
        
        if system == 'darwin':  # macOS
            run_in_terminal_macos(run_file)
        elif system.startswith('linux'):  # Linux
            run_in_terminal_linux(run_file)
        elif system == 'win32':  # Windows
            run_in_terminal_windows(run_file)
        else:
            print(f"  Warning: Unsupported OS ({system}). Running directly:")
            subprocess.Popen([sys.executable, run_file])
    
    print(f"\n✓ All {len(run_files)} script(s) launched in separate terminals!")
    print("  Each script will run independently.")

if __name__ == '__main__':
    main()

