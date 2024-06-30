import pandas as pd

def features(dfs, part, body_part, experimental, n=None):
    """
    Extract features and output them into a separate DataFrame.

    Parameters:
        dfs (list): List of DataFrames.
        part (str): Part of the body.
        body_part (list): List of body parts.
        experimental (str): Experimental setup.
        n (int): Number of movements (optional).

    Returns:
        DataFrame: DataFrame with extracted features.
    """
    translation_dict = {
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Std',
        'min': 'Min',
        '25%': '25th Percentile',
        '50%': 'Median',
        '75%': '75th Percentile',
        'max': 'Max'
    }

    features_df = []
    for i, df in enumerate(dfs):
        feature_df = ((df.iloc[:, :3].describe()).round(1)).T.rename(columns=translation_dict)
        feature_df['Movement'] = ' '.join(part.split()[:2])
        if n is None:
            feature_df['Number'] = ' '.join(part.split()[2:])
        else:
            feature_df['Number'] = feature_df.apply(lambda row: ' '.join(part.split()[2:]) + f"_{n}", axis=1)
        feature_df['Fencer'] = experimental
        feature_df['Sensor'] = body_part[i]
        features_df.append(feature_df)

    combined_df = pd.concat(features_df)
    pivot_df = combined_df.pivot_table(index=['Fencer', 'Movement', 'Number', 'Sensor'],
                                       columns=combined_df.index,
                                       values=['Count', 'Mean', 'Std', 'Min', '25th Percentile', 'Median',
                                               '75th Percentile', 'Max'])
    pivot_df = pivot_df.drop(columns=[('Count', 'y'), ('Count', 'z')])
    return pivot_df
