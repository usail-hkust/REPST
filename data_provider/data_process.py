import os
import numpy as np
import pandas as pd
from pydmd import DMD, MrDMD

def load_data(file_path):
    """
    Loads the solar energy data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the solar energy data.
    
    Returns:
        np.ndarray: Data array with shape (num_samples, num_nodes, 1).
    """
    df = pd.read_csv(file_path, index_col='time')
    data = df.values
    return np.expand_dims(np.asarray(data), axis=-1)

def generate_offsets(seq_length_x, seq_length_y):
    """
    Generates the x and y offsets based on the given sequence lengths.
    
    Args:
        seq_length_x (int): Length of the input sequence.
        seq_length_y (int): Length of the output sequence.
    
    Returns:
        tuple: x_offsets, y_offsets arrays.
    """
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, seq_length_y + 1, 1))
    return x_offsets, y_offsets

def fit_dmd_model(data, svd_rank=-1, max_level=2, max_cycles=3):
    """
    Fits a DMD model to the input data.
    
    Args:
        data (np.ndarray): Input data for DMD model fitting.
        svd_rank (int): Rank of the singular value decomposition. Default is -1 for auto-selection.
        max_level (int): Maximum level for MrDMD. Default is 2.
        max_cycles (int): Maximum number of cycles for MrDMD. Default is 3.
    
    Returns:
        np.ndarray: Reconstructed data after DMD fitting.
    """
    base_dmd = DMD(svd_rank=svd_rank)
    dmd = MrDMD(dmd=base_dmd, max_level=max_level, max_cycles=max_cycles)
    dmd.fit(data.T)
    reconstructed = dmd.reconstructed_data.T
    return reconstructed

def prepare_data(data, x_offsets, y_offsets):
    """
    Prepares the input and output sequences from the given data.
    
    Args:
        data (np.ndarray): The input data array.
        x_offsets (np.ndarray): Offsets for the input sequence.
        y_offsets (np.ndarray): Offsets for the output sequence.
    
    Returns:
        tuple: x (input sequences), y (output sequences).
    """
    num_samples = data.shape[0]
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    
    x, y = [], []
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])

    x = np.stack(x, axis=0, dtype='complex64')
    y = np.stack(y, axis=0, dtype='complex64')
    
    return x.transpose(0, 2, 1, 3), y.transpose(0, 2, 1, 3)

def split_data(x, y, train_ratio=0.7, val_ratio=0.2):
    """
    Splits the data into training, validation, and test sets.
    
    Args:
        x (np.ndarray): Input sequences.
        y (np.ndarray): Output sequences.
        train_ratio (float): Ratio of data for training. Default is 0.7.
        val_ratio (float): Ratio of data for validation. Default is 0.2.
    
    Returns:
        tuple: x_train, y_train, x_val, y_val, x_test, y_test
    """
    num_samples = x.shape[0]
    num_train = round(num_samples * train_ratio)
    num_val = round(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def save_data(x, y, x_offsets, y_offsets, save_dir, dataset_type):
    """
    Saves the prepared data as compressed .npz files.
    
    Args:
        x (np.ndarray): Input sequences.
        y (np.ndarray): Output sequences.
        x_offsets (np.ndarray): x_offsets array.
        y_offsets (np.ndarray): y_offsets array.
        save_dir (str): Directory where the data will be saved.
        dataset_type (str): The type of dataset (train/val/test).
    """
    np.savez_compressed(
        os.path.join(save_dir, f"{dataset_type}.npz"),
        x=x,
        y=y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    )

def main():
    # Configuration
    data_file = './Solar-energy_data.csv'
    save_dir = './solar-energy'
    seq_length_x, seq_length_y = 24, 24

    # Data loading and preprocessing
    data = load_data(data_file)
    x_offsets, y_offsets = generate_offsets(seq_length_x, seq_length_y)

    # DMD model fitting
    reconstructed = fit_dmd_model(data)
    
    # Prepare the final data for training
    feature_list = [data, reconstructed, data - reconstructed]
    data = np.concatenate(feature_list, axis=-1)
    
    # Prepare sequences
    x, y = prepare_data(data, x_offsets, y_offsets)
    
    # Split the data into train, val, test sets
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)
    
    # Save the datasets
    for dataset_type, _x, _y in zip(["train", "val", "test"], [x_train, x_val, x_test], [y_train, y_val, y_test]):
        save_data(_x, _y, x_offsets, y_offsets, save_dir, dataset_type)

    print("Data preparation and saving completed!")

if __name__ == "__main__":
    main()
