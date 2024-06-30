import matplotlib.pyplot as plt

def plot_dfs_accel(way_folder, experimental, folder, dfs, part, body_part, row, column, j, start_indices=None,
                   end_indices=None, save=False):
    """
    Plot accelerometer data.

    Parameters:
        way_folder (str): Path to the folder with data.
        experimental (str): Experimental setup.
        folder (str): Data folder.
        dfs (list): List of DataFrames.
        part (str): Part of the body.
        body_part (list): List of body parts.
        row (int): Number of rows in the plot.
        column (int): Number of columns in the plot.
        j (str): Set number.
        start_indices (list): List of start indices (optional).
        end_indices (list): List of end indices (optional).
        save (bool): Whether to save the plot (optional).
    """
    fig = plt.figure(figsize=(20, 10))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(row, column, i + 1)
        ax.plot(df['t'], df['x'], color='red', label='X')
        ax.plot(df['t'], df['y'], color='blue', label='Y')
        ax.plot(df['t'], df['z'], color='darkgreen', label='Z')
        if start_indices is not None and end_indices are not None:
            for start, end in zip(start_indices, end_indices):
                plt.fill_betweenx(y=[-(df['vector'].max()), df['vector'].max()],
                                  x1=df['t'][start], x2=df['t'][end],
                                  color='lightgray', alpha=0.5)
            plt.vlines(x=df['t'][start_indices], ymin=-(df['vector'].max()), ymax=df['vector'].max(), color='gray')
            plt.vlines(x=df['t'][end_indices], ymin=-(df['vector'].max()), ymax=df['vector'].max(), color='gray')
        ax.set_title(f'Sensor attached to {body_part[i]}', fontsize=16)
        ax.set_ylabel('Acceleration, g', fontsize=12)
        ax.grid(which='major', linewidth=0.5, linestyle=':', color='k')
        ax.set_xlabel('Time, s', fontsize=12)
        ax.autoscale(enable=True, axis='both')
        plt.legend()
    fig.suptitle(f'Accelerometer data. Movement: {part} {j}', fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(f'{way_folder}{experimental}{folder}/{part}.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()
