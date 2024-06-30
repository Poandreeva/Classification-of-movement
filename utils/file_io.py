import pandas as pd

def read_data(file_names, column_names, num):
    """
    Read recorded data from files.

    Parameters:
        file_names (list): List of file names.
        column_names (list): List of column names.
        num (int): Number of data sets.

    Returns:
        list: List of concatenated DataFrames.
    """
    dfs = []
    for file_name, column_name in zip(file_names, column_names):
        df = pd.read_csv(file_name, header=None, delimiter="\t", decimal=",", comment=";", names=[column_name])
        df_rounded = df.round(2)
        dfs.append(df_rounded)

    dfs_concated = []
    for i in range(num):
        df = pd.concat([dfs[i], dfs[i + num], dfs[i + 2 * num]], axis=1)
        dfs_concated.append(df)
    return dfs_concated
