def split_dataframe(dfs):
    """
    Split the entire signal into two DataFrames corresponding to movements of 1st and 3rd sets.

    Parameters:
        dfs (list): List of DataFrames.

    Returns:
        tuple: DataFrames for the 1st and 3rd sets.
    """
    split_index1 = 6500
    split_index2 = 12300
    split_index3 = 83530
    split_index4 = 89600

    dfs_1 = []
    dfs_2 = []
    for df in dfs:
        dfs_1.append(df.iloc[split_index1:split_index2])
        dfs_2.append(df.iloc[split_index3:split_index4])
    for df in dfs_1:
        df.reset_index(drop=True, inplace=True)
    for df in dfs_2:
        df.reset_index(drop=True, inplace=True)

    return dfs_1, dfs_2
