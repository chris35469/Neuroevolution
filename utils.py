# These functions are specific to the drone recorded data.
# In the future, I will make a generic data loading function (and others) with parameters to specify how to load the data
# For now, if you want to load other data in other formats, you will need to create your own data loader function, or use my framework which always loads JSON formatted data.

import torch
import torch.nn as nn
import json
import numpy as np
import csv
import os


def load_json(json_file_path):
    """
    Load data from JSON file and convert to PyTorch tensors.
    
    Args:
        json_file_path: Path to the JSON file containing training data
        
    Returns:
        X: Input tensors (flattened 64x64 matrices) - shape: (n_samples, 4096)
        y_command: Command targets - shape: (n_samples, 3)
        y_continuous: Continuous command targets - shape: (n_samples, 3)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract and flatten matrices (64x64 -> 4096)
    X_list = []
    y_command = []
    y_continuous = []
    
    for data_point in data:
        # Flatten the 64x64 matrix to 4096 features
        matrix = np.array(data_point['matrix'], dtype=np.float32)
        flattened = matrix.flatten()
        X_list.append(flattened)
        
        # Extract command arrays
        y_command.append(data_point['command'])
        y_continuous.append(data_point['continuous_command'])
    
    # Convert to numpy arrays first, then to PyTorch tensors (more efficient)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_command = torch.tensor(np.array(y_command), dtype=torch.float32)
    y_continuous = torch.tensor(np.array(y_continuous), dtype=torch.float32)
    
    return X, y_command, y_continuous


def load_csv(csv_file_path, return_tensor=True, dtype=np.float32):
    """
    Load data from a CSV file and convert to numpy array or PyTorch tensor.
    
    Args:
        csv_file_path: Path to the CSV file to load
        return_tensor: If True, returns PyTorch tensor; if False, returns numpy array (default: True)
        dtype: Data type for the array/tensor (default: np.float32)
        
    Returns:
        Data as numpy array or PyTorch tensor
        
    Example:
        >>> # Load matrix CSV
        >>> matrix_data = load_csv('data/csv_output/matrix.csv')
        >>> print(f"Shape: {matrix_data.shape}")
        >>> 
        >>> # Load command CSV
        >>> command_data = load_csv('data/csv_output/command.csv')
        >>> print(f"Shape: {command_data.shape}")
    """
    data = []
    
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert each row to float and append
            data.append([float(value) for value in row])
    
    # Convert to numpy array
    array = np.array(data, dtype=dtype)
    
    # Return as tensor or array based on parameter
    if return_tensor:
        return torch.tensor(array, dtype=torch.float32)
    else:
        return array


def save_to_csv(data, file_path):
    """
    Save arrays/lists to a CSV file.
    Handles lists, numpy arrays, and PyTorch tensors.
    
    Args:
        data: Data to save (list, numpy array, or PyTorch tensor)
        file_path: Path to save the CSV file (will create directory if needed)
        
    Example:
        >>> fitnesses = [-0.5, -0.3, -0.7, -0.2]
        >>> save_to_csv(fitnesses, 'results/fitnesses.csv')
        >>> 
        >>> # Save 2D array
        >>> predictions = [[1, 2, 3], [4, 5, 6]]
        >>> save_to_csv(predictions, 'results/predictions.csv')
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Convert to numpy array if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Ensure data is 2D (if 1D, reshape to column vector)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Write to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    
    # print(f"Data saved to {file_path}")


def loss(predictions, targets, loss_type='mse'):
    """
    Calculate loss between predictions and targets.
    Useful for evaluating model performance in evolutionary algorithms.
    
    Args:
        predictions: Tensor of model predictions - shape: (n_samples, n_outputs)
        targets: Tensor of target/actual values - shape: (n_samples, n_outputs)
        loss_type: Type of loss to calculate. Options:
                  - 'mse': Mean Squared Error (default)
                  - 'mae': Mean Absolute Error (L1 loss)
                  - 'rmse': Root Mean Squared Error
                  
    Returns:
        Scalar loss value (float)
        
    Example:
        >>> predictions = model.test(X)
        >>> loss_value = loss(predictions, y_command)
        >>> print(f"Model loss: {loss_value}")
    """
    # Ensure inputs are tensors
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    
    # Ensure shapes match
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} != targets {targets.shape}")
    
    # Calculate loss based on type
    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
        loss_value = loss_fn(predictions, targets)
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()  # L1Loss is Mean Absolute Error
        loss_value = loss_fn(predictions, targets)
    elif loss_type.lower() == 'rmse':
        loss_fn = nn.MSELoss()
        mse = loss_fn(predictions, targets)
        loss_value = torch.sqrt(mse)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from 'mse', 'mae', 'rmse'")
    
    # Return as Python float for easy comparison
    return loss_value.item()


def json_to_csv(input_data_path, output_data_path):
    """
    Convert JSON data file to CSV files.
    Creates 3 CSV files containing all data:
    - matrix.csv: All matrices, each matrix flattened to one row (4096 values per row)
    - command.csv: All commands, each command array on one row
    - continuous_command.csv: All continuous commands, each continuous command array on one row
    
    Args:
        input_data_path: Path to the input JSON file
        output_data_path: Path to the directory where CSV files will be saved
        
    Example:
        >>> json_to_csv('data/dummy_data.json', 'data/csv_output')
        >>> # Creates: matrix.csv, command.csv, continuous_command.csv
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    # Load JSON data
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    
    # Open CSV files for writing
    matrix_filename = os.path.join(output_data_path, 'matrix.csv')
    command_filename = os.path.join(output_data_path, 'command.csv')
    continuous_command_filename = os.path.join(output_data_path, 'continuous_command.csv')
    
    # Write all matrices to one CSV file (each matrix flattened to one row)
    with open(matrix_filename, 'w', newline='') as matrix_file:
        matrix_writer = csv.writer(matrix_file)
        for data_point in data:
            matrix = np.array(data_point['matrix'])
            # Flatten the matrix (64x64 -> 4096) and write as one row
            flattened_matrix = matrix.flatten()
            matrix_writer.writerow(flattened_matrix)
    
    # Write all commands to one CSV file
    with open(command_filename, 'w', newline='') as command_file:
        command_writer = csv.writer(command_file)
        for data_point in data:
            command = data_point['command']
            command_writer.writerow(command)
    
    # Write all continuous commands to one CSV file
    with open(continuous_command_filename, 'w', newline='') as continuous_command_file:
        continuous_command_writer = csv.writer(continuous_command_file)
        for data_point in data:
            continuous_command = data_point['continuous_command']
            continuous_command_writer.writerow(continuous_command)
    
    # print(f"Successfully converted {len(data)} elements from {input_data_path}")
    # print(f"CSV files saved to: {output_data_path}")
    # print(f"Created 3 CSV files:")
    # print(f"  - matrix.csv ({len(data)} matrices, {len(data)} rows, 4096 columns per row)")
    # print(f"  - command.csv ({len(data)} commands, {len(data)} rows)")
    # print(f"  - continuous_command.csv ({len(data)} continuous commands, {len(data)} rows)")

