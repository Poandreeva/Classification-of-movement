import os
import pandas as pd
from data_processing import creation_of_dfs, find_movement_peaks, movement
from feature_extraction import features
from movement_classification import train_logistic_regression, train_random_forest
from heart_rate_analysis import smoothed_df, plot_filtered_heart_rate, median_param, plot_speed_ampl
from utils.data_split import split_dataframe
from utils.plot_utils import plot_dfs_accel

# Main script
WAY_FOLDER = 'fencing_signals/'
FS = 100
BODY_PART = ["front arm", "chest", "front leg", "back leg"]
SAVE = True

exp = ['Cherkasova_20_1/', 'Cherkasova_20_2/', 'Sugkoeva_17/', 'Zushko_18/']
exp_endurance = ['Cherkasova_20_3/']

settings = {
    'Cherkasova_20_1/': {
        'Arm/': {
            'start_trim': 152,
            'end_trim': 2550,
            'height_threshold': 95,
            'prominence_value': 90,
            'distance_value': 50
        },
        'Step/': {
            'start_trim': 303,
            'end_trim': 3816,
            'height_threshold': 400,
            'prominence_value': 200,
            'distance_value': 70
        },
        'Lunge/': {
            'start_trim': 485,
            'end_trim': 5066,
            'height_threshold': 430,
            'prominence_value': 400,
            'distance_value': 90
        }
    },
    'Cherkasova_20_2/': {
        'Arm/': {
            'start_trim': 197,
            'end_trim': 4309,
            'height_threshold': 95,
            'prominence_value': 90,
            'distance_value': 50
        },
        'Step/': {
            'start_trim': 152,
            'end_trim': 4028,
            'height_threshold': 376,
            'prominence_value': 250,
            'distance_value': 65
        },
        'Lunge/': {
            'start_trim': 212,
            'end_trim': 4307,
            'height_threshold': 750,
            'prominence_value': 400,
            'distance_value': 57
        }
    },
    'Sugkoeva_17/': {
        'Arm/': {
            'start_trim': 180,
            'end_trim': 5069,
            'height_threshold': 90,
            'prominence_value': 90,
            'distance_value': 50
        },
        'Step/': {
            'start_trim': 147,
            'end_trim': 5372,
            'height_threshold': 400,
            'prominence_value': 250,
            'distance_value': 45
        },
        'Lunge/': {
            'start_trim': 284,
            'end_trim': 5660,
            'height_threshold': 500,
            'prominence_value': 380,
            'distance_value': 65
        }
    },
    'Zushko_18/': {
        'Arm/': {
            'start_trim': 167,
            'end_trim': 3809,
            'height_threshold': 95,
            'prominence_value': 90,
            'distance_value': 50
        },
        'Step/': {
            'start_trim': 100,
            'end_trim': 4539,
            'height_threshold': 400,
            'prominence_value': 200,
            'distance_value': 70
        },
        'Lunge/': {
            'start_trim': 162,
            'end_trim': 4918,
            'height_threshold': 500,
            'prominence_value': 380,
            'distance_value': 65
        }
    },
    'Cherkasova_20_3/': {
        'Lunge/': [
            {
                'height_threshold': 450,
                'prominence_value': 50,
                'distance_value': 68
            },
            {
                'height_threshold': 450,
                'prominence_value': 200,
                'distance_value': 80
            }
        ]
    }
}

# Movement classification

moves = ['Arm/', 'Step/', 'Lunge/']
fencing_move = ['with arm', 'with step', 'with lunge']
num = 4

ways = ['attack', 'defense']
all_movement_dfs = []
all_params = []
peaks_executed = False
action_executed = False
row = 2
column = 2

for experimental in exp:
    for i, move in enumerate(moves):
        dfs = creation_of_dfs(WAY_FOLDER, experimental, move, num)

        t = np.arange(len(dfs[0]['x'])) / FS
        for df in dfs:
            df['t'] = t
            df['vector'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)

        if move in settings.get(experimental, {}):
            setting_params = settings[experimental][move]
            start_trim = setting_params['start_trim']
            end_trim = setting_params['end_trim'] if setting_params['end_trim'] is not None else len(dfs[0])
            for b, df in enumerate(dfs):
                dfs[b] = df.iloc[start_trim:end_trim].reset_index(drop=True)

        combined_df, start_indices, end_indices, positive_peaks, negative_peaks, derivative, peaks, df_forward_ind, df_backward_ind = find_movement_peaks(
            dfs, move, num, setting_params)
        if not peaks_executed:
            plot_peaks(WAY_FOLDER, experimental, move, combined_df, start_indices, end_indices, positive_peaks,
                       negative_peaks, derivative, peaks, fencing_move[i], num, save=SAVE)
            print('Number of peaks:', len(peaks))
            plot_dfs(WAY_FOLDER, experimental, move, dfs, fencing_move[i], BODY_PART, row, column, start_indices,
                     end_indices, save=SAVE)
            peaks_executed = True

        for n, df in enumerate([df_forward_ind, df_backward_ind]):
            for a in range(len(df)):
                movement_dfs = movement(dfs, df['start'][a], df['end'][a])
                all_movement_dfs.append(movement_dfs)
                if not action_executed:
                    plot_dfs(WAY_FOLDER, experimental, move, movement_dfs, f'{ways[n]} {fencing_move[i]} No.{a + 1}',
                             BODY_PART, row, column, save=SAVE)
                    action_executed = True

                params = features(movement_dfs, f'{ways[n]} {fencing_move[i]} {a + 1}', BODY_PART, experimental)
                all_params.append(params)

A = pd.concat(all_params)
print('Number of parameters for one movement: ', 4 * A.shape[1] - 3)
A.head(8)

unique_movements = list(A.index.get_level_values('Movement').unique())
print(unique_movements)

All_signs = A.reset_index().pivot_table(index=['Fencer', 'Movement', 'Number'], columns=['Sensor'])
All_signs.columns = [f'{col[0]}_{col[1]}_{col[2]}' for col in All_signs.columns]
All_signs.reset_index(inplace=True)
All_signs = All_signs.drop(['Fencer', 'Number'], axis=1)
print('Number of movements: ', len(All_signs))

y = All_signs['Movement']
X = All_signs.drop('Movement', axis=1)

train_logistic_regression(X, y, save=SAVE)
