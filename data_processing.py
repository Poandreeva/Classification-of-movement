import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from utils.file_io import read_data

# Global settings
WAY_FOLDER = 'fencing_signals/'
FS = 100
BODY_PART = ["front arm", "chest", "front leg", "back leg"]

def creation_of_dfs(way_folder, experimental, folder, num):
    """
    Create DataFrames with gyroscope data for each axis.

    Parameters:
        way_folder (str): Path to the folder with data.
        experimental (str): Experimental setup.
        folder (str): Data folder.
        num (int): Number of DataFrames to create.

    Returns:
        list: List of DataFrames.
    """
    folder_path = f"{way_folder}{experimental}{folder}CSV_Export"
    file_names = [
        "Hyro Y_9.csv", "Hyro X_11.csv", "Hyro Z_16.csv", "Hyro X_17.csv",
        "Hyro X_8.csv", "Hyro Z_13.csv", "Hyro X_14.csv", "Hyro Z_19.csv",
        "Hyro Z_10.csv", "Hyro Y_12.csv", "Hyro Y_15.csv", "Hyro Y_18.csv"
    ]

    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    column_names = ["x", "x", "x", "x", "y", "y", "y", "y", "z", "z", "z", "z"]
    dfs = read_data(file_paths, column_names, num)

    dfs[3]['x'] = dfs[3]['x'].apply(np.negative)
    dfs[2]['y'] = dfs[2]['y'].apply(np.negative)
    dfs[3]['y'] = dfs[3]['y'].apply(np.negative)
    return dfs

def find_movement_peaks(dfs, move, num, params):
    """
    Find indices of start and end of forward and backward movements.

    Parameters:
        dfs (list): List of DataFrames.
        move (str): Movement type.
        num (int): Number of DataFrames.
        params (dict): Parameters for finding peaks.

    Returns:
        tuple: Combined DataFrame, start indices, end indices, positive peaks,
               negative peaks, derivative, peaks, forward indices, backward indices.
    """
    dfs[0]['vector'] *= 2
    dfs[1]['vector'] *= 2

    combined_df = pd.concat(dfs[i]['vector'] for i in range(num))
    combined_df = combined_df.groupby(combined_df.index).agg('sum')

    height_threshold = params['height_threshold']
    prominence_value = params['prominence_value']
    distance_value = params['distance_value']

    peaks, _ = find_peaks(combined_df, height=height_threshold, prominence=prominence_value, distance=distance_value)

    derivative = np.gradient(combined_df, combined_df.index)
    positive_peaks, _ = find_peaks(derivative, distance=15)
    negative_peaks, _ = find_peaks(-derivative, distance=15)

    start_indices, end_indices = define_movement_bounds(peaks, derivative, distance_value, combined_df)
    df_forward_index, df_backward_index = separate_movements(start_indices, end_indices)

    return combined_df, start_indices, end_indices, positive_peaks, negative_peaks, derivative, peaks, df_forward_index, df_backward_index

def define_movement_bounds(peaks, derivative, distance_value, combined_df):
    """
    Helper function to define start and end indices of movements.

    Parameters:
        peaks (list): List of peak indices.
        derivative (array): Derivative of the combined DataFrame.
        distance_value (int): Distance value for finding peaks.
        combined_df (DataFrame): Combined DataFrame of vectors.

    Returns:
        tuple: Start indices, end indices.
    """
    midpoints = [(peaks[j - 1] + peaks[j]) // 2 for j in range(1, len(peaks))]
    start_indices = []
    end_indices = []
    window_size = [(peaks[j] - peaks[j - 1]) // 6 for j in range(1, len(peaks))]

    first_peak_idx = peaks[0]
    search_start = max(0, first_peak_idx - np.median(window_size) * 4)
    start_indices.insert(0, combined_df.index[int(search_start)])

    for n, midpoint in enumerate(midpoints):
        midpoint_idx = combined_df.index.get_loc(midpoint)
        search_start = max(0, midpoint_idx - window_size[n])
        search_end = min(len(combined_df) - 1, midpoint_idx + window_size[n])

        for k in range(search_start + 1, search_end):
            if k < len(derivative) - 1 and derivative[k] < 0 and derivative[k - 1] >= 0:
                end_indices.append(combined_df.index[k] + 10)
                break

        for l in range(search_end - 1, search_start, -1):
            if l < len(derivative) - 1 and derivative[l] > 0 and derivative[l + 1] <= 0:
                start_indices.append(combined_df.index[l] - 10)
                break

    last_peak_idx = peaks[-1]
    search_end = min(len(combined_df) - 1, last_peak_idx + np.median(window_size) * 4)
    end_indices.append(combined_df.index[int(search_end)])

    return start_indices, end_indices

def separate_movements(start_indices, end_indices):
    """
    Helper function to create DataFrames with indices of movements,
    separating all movements into forward and backward movements.

    Parameters:
        start_indices (list): List of start indices.
        end_indices (list): List of end indices.

    Returns:
        tuple: DataFrames with forward and backward movement indices.
    """
    start_forward = start_indices[::2]
    end_forward = end_indices[::2]
    start_backward = start_indices[1::2]
    end_backward = end_indices[1::2]

    df_forward = pd.DataFrame({'start': start_forward, 'end': end_forward})
    df_backward = pd.DataFrame({'start': start_backward, 'end': end_backward})

    return df_forward, df_backward

def movement(dfs, start, end):
    """
    Separate signal into individual movements.

    Parameters:
        dfs (list): List of DataFrames.
        start (int): Start index.
        end (int): End index.

    Returns:
        list: List of DataFrames for each movement.
    """
    movement_dfs = []
    for df in dfs:
        movement_dfs.append(df[['x', 'y', 'z', 't']][start:end])
    return movement_dfs
