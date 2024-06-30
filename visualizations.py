import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_dfs(way_folder, experimental, folder, dfs, part, body_part, row, column,
             start_indices=None, end_indices=None, save=False):
    """
    Plot data.

    Parameters:
        way_folder (str): Path to the folder with data.
        experimental (str): Experimental setup.
        folder (str): Data folder.
        dfs (list): List of DataFrames.
        part (str): Part of the body.
        body_part (list): List of body parts.
        row (int): Number of rows in the plot.
        column (int): Number of columns in the plot.
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
        ax.set_ylabel('Angle, deg', fontsize=12)
        ax.grid(which='major', linewidth=0.5, linestyle=':', color='k')
        ax.set_xlabel('Time, s', fontsize=12)
        ax.autoscale(enable=True, axis='both')
        plt.legend()
    fig.suptitle(f'Gyroscope data. Movement: {part}', fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(f'{way_folder}{experimental}{folder}/{part}.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_peaks(way_folder, experimental, folder, combined_df, start_indices,
               end_indices, positive_peaks, negative_peaks, derivative, peaks,
               fencing_move, num, save=False):
    """
    Visualize peaks.

    Parameters:
        way_folder (str): Path to the folder with data.
        experimental (str): Experimental setup.
        folder (str): Data folder.
        combined_df (DataFrame): Combined DataFrame of vectors.
        start_indices (list): List of start indices.
        end_indices (list): List of end indices.
        positive_peaks (list): List of positive peaks.
        negative_peaks (list): List of negative peaks.
        derivative (array): Derivative of the combined DataFrame.
        peaks (list): List of peak indices.
        fencing_move (str): Type of fencing move.
        num (int): Number of DataFrames.
        save (bool): Whether to save the plot (optional).
    """
    fig = plt.figure(figsize=(10, 4))
    plt.plot(combined_df.index, combined_df, color='grey', label=f'Sum of vector signals from {num} sensors')
    for start, end in zip(start_indices, end_indices):
        plt.fill_betweenx(y=[combined_df.min(), combined_df.max()],
                          x1=combined_df.index[start], x2=combined_df.index[end], color='lightgray', alpha=0.5)
    plt.vlines(x=combined_df.index[start_indices], ymin=combined_df.min(), ymax=combined_df.max(), color='green')
    plt.vlines(x=combined_df.index[end_indices], ymin=combined_df.min(), ymax=combined_df.max(), color='green')
    plt.plot(combined_df.index[positive_peaks], derivative[positive_peaks], color='orange', label='Derivative')
    plt.plot(combined_df.index[negative_peaks], derivative[negative_peaks], color='orange')
    plt.plot(peaks, combined_df[peaks], "x", ms=10, mew=2, label='Peak')
    plt.ylabel('Angle, deg', fontsize=10)
    plt.xlabel('Indices', fontsize=10)
    plt.grid(which='major', linewidth=0.5, linestyle=':', color='k')
    plt.autoscale(enable=True, axis='both')
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.title(f'Gyroscope data. Movement: {fencing_move}', fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(f'{way_folder}{experimental}{folder}/{fencing_move}.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

def evaluate_preds(true_values, train_true_values, pred_values, train_pred_values, labels, save=False):
    """
    Evaluate model quality and plot predicted and actual values.

    Parameters:
        true_values (array): True values for the test set.
        train_true_values (array): True values for the training set.
        pred_values (array): Predicted values for the test set.
        train_pred_values (array): Predicted values for the training set.
        labels (list): List of movement labels.
        save (bool): Whether to save the plot (optional).
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(labels, labels, color='white', s=1)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Actual Values', fontsize=14)
    plt.title('Actual and Predicted Values', fontsize=16)
    plt.grid(which='major', linewidth=0.5, linestyle=':', color='k')
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.margins(x=0.1, y=0.1)

    point_counts_train = {}
    for x, y in zip(train_pred_values, train_true_values):
        point_counts_train[(x, y)] = point_counts_train.get((x, y), 0) + 1
    for (x, y), count in point_counts_train.items():
        size = 20 * count
        plt.scatter(x, y, color='red', s=size)
        offset = np.sqrt(size) / 4
        plt.annotate(f' ({count})', xy=(x, y), xytext=(offset, -offset),
                     textcoords='offset points', fontsize=8, color='red', ha='left', va='top')

    point_counts_test = {}
    for x, y in zip(pred_values, true_values):
        point_counts_test[(x, y)] = point_counts_test.get((x, y), 0) + 1
    for (x, y), count in point_counts_test.items():
        size = 20 * count
        plt.scatter(x, y, color='green', s=size)
        offset = np.sqrt(size) / 2
        plt.annotate(f' {count}', xy=(x, y), xytext=(offset, offset),
                     textcoords='offset points', fontsize=8, color='green')

    rect = patches.FancyBboxPatch((0.15, 0.705), 0.27, 0.095,
                                  boxstyle="round,pad=0.01",
                                  facecolor='whitesmoke', alpha=0.5,
                                  edgecolor='black',
                                  transform=plt.gcf().transFigure)
    plt.gca().add_patch(rect)
    plt.annotate('Number of Points:',
                 xy=(0.05, 0.95), xytext=(0.025, 0.875), xycoords='axes fraction', fontsize=10, color='black')
    plt.annotate('• 1 - Test Set',
                 xy=(0.05, 0.95), xytext=(0.025, 0.825), xycoords='axes fraction', fontsize=10, color='green')
    plt.annotate('• (1) - Training Set',
                 xy=(0.05, 0.95), xytext=(0.025, 0.775), xycoords='axes fraction', fontsize=10, color='red')

    if save:
        plt.savefig(f'Actual and Predicted Values.png', dpi=400, bbox_inches='tight')
    plt.show()
