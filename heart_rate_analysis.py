import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smoothed_df(hr_df):
    """
    Smooth the heart rate data using a polynomial.

    Parameters:
        hr_df (DataFrame): Heart rate data.

    Returns:
        DataFrame: Smoothed heart rate data.
    """
    x = hr_df.index
    y = hr_df['HeartRate'].values

    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)

    smoothed_values = polynomial(x)
    smoothed_df = pd.DataFrame(smoothed_values).reset_index(drop=True)

    return smoothed_df

def plot_filtered_heart_rate(dataframe, start_index, end_index, tol, window_length, polyorder, save=False):
    """
    Plot filtered heart rate data.

    Parameters:
        dataframe (DataFrame): Heart rate data.
        start_index (int): Start index.
        end_index (int): End index.
        tol (int): Tolerance value for identifying plateau.
        window_length (int): Window length for Savitzky-Golay filter.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        save (bool): Whether to save the plot (optional).

    Returns:
        tuple: Filtered DataFrame, recovery time in seconds.
    """
    df_filtered = dataframe.iloc[start_index:end_index].reset_index(drop=True)
    df_filtered_index = df_filtered.index

    filtered_hr = savgol_filter(df_filtered['HeartRate'], window_length=window_length, polyorder=polyorder)
    df_filtered['Filtered_HR'] = filtered_hr

    max_index = df_filtered['Filtered_HR'].idxmax()

    plateau_start = None
    for i in range(max_index + 1, len(df_filtered['Filtered_HR'])):
        if df_filtered['Filtered_HR'][i] + tol <= df_filtered['Filtered_HR'][max_index]:
            plateau_start = i
            break

    recovery_time_seconds = plateau_start - max_index

    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered_index, df_filtered['HeartRate'], label='Original HR Data', color='dodgerblue', linewidth=1)
    plt.plot(df_filtered_index, df_filtered['Filtered_HR'], label='Filtered HR Data', linestyle='--', color='black',
             linewidth=3)

    plt.axvline(x=0, color='green', linestyle='--')
    plt.axvline(x=max_index, color='red', linestyle='--')
    plt.axvline(x=plateau_start, color='green', linestyle='--')

    plt.annotate('', xy=(max_index, df_filtered['Filtered_HR'][max_index] - 5),
                 xytext=(plateau_start, df_filtered['Filtered_HR'][max_index] - 5),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    plt.text(max_index + (plateau_start - max_index) / 2, df_filtered['Filtered_HR'][max_index] - 4,
             f'Recovery Time: {recovery_time_seconds} s', color='black',
             fontsize=12, horizontalalignment='center')

    plt.annotate('', xy=(0, df_filtered['Filtered_HR'][max_index] - 5),
                 xytext=(max_index, df_filtered['Filtered_HR'][max_index] - 5),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    plt.text(max_index / 2, df_filtered['Filtered_HR'][max_index] - 4,
             f'Set Duration: {max_index} s', color='black',
             fontsize=12, horizontalalignment='center')

    plt.text(plateau_start + 2, df_filtered['Filtered_HR'].loc[plateau_start] + 10,
             f'HR: {df_filtered["Filtered_HR"].loc[plateau_start]:.0f} bpm', color='green',
             fontsize=12, horizontalalignment='left')
    plt.text(max_index + 2, df_filtered['Filtered_HR'].loc[plateau_start] + 10,
             f'HR: {df_filtered["Filtered_HR"].loc[max_index]:.0f} bpm', color='red',
             fontsize=12, horizontalalignment='left')

    plt.xlim(-5, 350)
    plt.title('Heart Rate Before and After Filtering', fontsize=14)
    plt.xlabel('Time, s', fontsize=12)
    plt.ylabel('Heart Rate (bpm)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Heart_Rate_Before_and_After_Filtering_{start_index}.png', dpi=400, bbox_inches='tight')
    plt.show()

    return df_filtered, recovery_time_seconds

def median_param(max_par, n):
    """
    Calculate median values for parameters.

    Parameters:
        max_par (list): List of maximum parameter values.
        n (int): Number of movements in each set.

    Returns:
        list: List of smoothed median values.
    """
    smoothed = []
    for i in range(0, len(max_par), n):
        m_value = np.mean(max_par[i:i + n])
        smoothed.extend([m_value] * (n + 1))
    return smoothed

def plot_speed_ampl(smoothed, title, ylabel, n, c, save=False):
    """
    Plot smoothed speed and amplitude values.

    Parameters:
        smoothed (list): List of smoothed values.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        n (int): Number of movements in each set.
        c (float): Correction value for text placement.
        save (bool): Whether to save the plot (optional).
    """
    plt.figure(figsize=(6, 4))

    plt.plot(range(n), smoothed[:n], color='green', linewidth=2, label='set #1')
    plt.text(n / 2, smoothed[0] - c, f'{round(smoothed[0], 1)}', ha='center', va='top', fontsize=10, color='green')

    plt.plot(range(n + 1, len(smoothed)), smoothed[n + 1:], color='red', linewidth=2, label='set #3')
    plt.text(n + (len(smoothed) - n) / 2, smoothed[n + 1] + c, f'{round(smoothed[n + 1], 1)}', ha='center', va='bottom',
             fontsize=10, color='red')

    plt.title(f'Change in {title} with Cumulative Fatigue')
    plt.xlabel('Movement Number')
    plt.ylabel(ylabel)
    plt.legend()
    plt.axvline(x=n, color='black', linestyle='--', linewidth=2)
    plt.grid(True)
    if save:
        plt.savefig(f'Change_in_{title}_with_Fatigue.png', dpi=400, bbox_inches='tight')
    plt.show()
