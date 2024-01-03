import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_tracking_and_plays(csv_name):
    """
    Return a dataframe containing cleaned tracking data joined with the plays data
    """
    tracking = pd.read_csv(csv_name)
    plays = pd.read_csv("plays.csv")

    tracking.loc[tracking['playDirection'] == 'left', 'x'] = 120 - tracking.loc[tracking['playDirection'] == 'left', 'x']
    tracking.loc[tracking['playDirection'] == 'left', 'y'] = (160/3) - tracking.loc[tracking['playDirection'] == 'left', 'y']
    tracking.loc[tracking['playDirection'] == 'left', 'dir'] += 180
    tracking.loc[tracking['dir'] > 360, 'dir'] -= 360
    tracking.loc[tracking['playDirection'] == 'left', 'o'] += 180
    tracking.loc[tracking['o'] > 360, 'o'] -= 360

    tracking_with_plays = tracking.merge(plays, on=['gameId', 'playId'], how='left')
    tracking_with_plays['is_on_offense'] = tracking_with_plays['club'] == tracking_with_plays['possessionTeam']
    tracking_with_plays['is_on_defense'] = tracking_with_plays['club'] == tracking_with_plays['defensiveTeam']
    tracking_with_plays['is_ballcarrier'] = tracking_with_plays['ballCarrierId'] == tracking_with_plays['nflId']

    return tracking_with_plays

def compute_feature_df(tracking_with_plays):
    """
    Take in a dataframe of joined tracking and plays data. Compute per-frame features
    """
    info = pd.DataFrame()
    info_frames = []

    for gid in tqdm(tracking_with_plays['gameId'].unique()):
        game = tracking_with_plays[tracking_with_plays['gameId'] == gid].copy()
        for pid in game['playId'].unique():
            try:
                play = game[game['playId'] == pid].copy()
                ball_carrier_id = play['ballCarrierId'].iloc[0]
                end_frame_id = play['frameId'].max()
                ball_end_x = play.loc[(play['frameId'] == end_frame_id) & (play['nflId'] == ball_carrier_id), 'x'].iloc[0]

                # # OPTIONAL: Randomly select 40% of frames to keep
                # all_frame_ids = play['frameId'].unique()
                # selected_frame_ids = np.random.choice(all_frame_ids, size=int(len(all_frame_ids) * 0.3), replace=False)
                # play = play[play['frameId'].isin(selected_frame_ids)]

                for fid in play['frameId'].unique():
                    frame = play[play['frameId'] == fid].copy()
                    frame['is_on_offense'] = frame['club'] == frame['possessionTeam'].iloc[0]
                    frame['is_on_defense'] = frame['club'] == frame['defensiveTeam'].iloc[0]
                    frame['is_ball_carrier'] = frame['nflId'] == ball_carrier_id
                    frame['s_x'] = frame['s']*np.cos(np.deg2rad(frame['dir']))
                    frame['s_y'] = frame['s']*np.sin(np.deg2rad(frame['dir']))
                    frame['ball_carrier_s_x'] = frame.loc[frame['nflId'] == ball_carrier_id, 's_x'].iloc[0]
                    frame['ball_carrier_s_y'] = frame.loc[frame['nflId'] == ball_carrier_id, 's_y'].iloc[0]
                    frame['relative_s_x'] = frame['ball_carrier_s_x'] - frame['s_x']
                    frame['relative_s_y'] = frame['ball_carrier_s_y'] - frame['s_y']
                    frame['ball_carrier_x'] = frame.loc[frame['nflId'] == ball_carrier_id, 'x'].iloc[0]
                    frame['ball_carrier_y'] = frame.loc[frame['nflId'] == ball_carrier_id, 'y'].iloc[0]
                    frame['relative_y'] = frame['ball_carrier_y'] - frame['y']
                    frame['yards_remaining'] = ball_end_x - frame['ball_carrier_x']
                    info_frames.append(frame)
            except:
                print('error')


    info = pd.concat(info_frames, ignore_index=True)
    # info.to_csv('info.csv', index=False)
    return info

def create_feature_tensor(tracking_with_plays):
    """
    Convert the input frame_df to a 4D tensor.
        - The first dimension is the frame
        - The second dimension is the index of the current player
        - The third dimension is the index of the relative player
    """
    tensor_shape = (tracking_with_plays.groupby(['gameId', 'playId', 'frameId']).ngroups, 10, 11, 10)
    input_tensor = np.zeros(tensor_shape)
    target_tensor = np.zeros((tracking_with_plays.groupby(['gameId', 'playId', 'frameId']).ngroups))
    reference_tensor = np.zeros((tracking_with_plays.groupby(['gameId', 'playId', 'frameId']).ngroups, 3))
    positional_reference_tensor = np.zeros((tracking_with_plays.groupby(['gameId', 'playId', 'frameId']).ngroups, 11, 10, 5))
    cur_count = 0
    for game_id, game_group in tqdm(tracking_with_plays.groupby('gameId')):
        for play_id, play_group in game_group.groupby('playId'):
            max_frame = play_group['frameId'].max()
            ball_end_x = play_group.loc[(play_group['frameId'] == max_frame) & (play_group['is_ballcarrier']), 'x']
            for frame_id, frame_group in play_group.groupby('frameId'):
                offense = frame_group[(frame_group['is_on_offense']) & (~frame_group['is_ballcarrier'])]
                defense = frame_group[frame_group['is_on_defence']]
                ballcarrier = frame_group[frame_group['is_ballcarrier']]
                ballcarrier_sx = ballcarrier.s * np.cos(np.deg2rad(ballcarrier.dir))
                ballcarrier_sy = ballcarrier.s * np.sin(np.deg2rad(ballcarrier.dir))
                for i, def_player in enumerate(defense.itertuples()):
                    def_player_sx = def_player.s * np.cos(np.deg2rad(def_player.dir))
                    def_player_sy = def_player.s * np.sin(np.deg2rad(def_player.dir))
                    for j, off_player in enumerate(offense.itertuples()):
                        off_player_sx = off_player.s * np.cos(np.deg2rad(off_player.dir))
                        off_player_sy = off_player.s * np.sin(np.deg2rad(off_player.dir))
                        input_tensor[cur_count, 0, i, j] = off_player.x - def_player.x
                        input_tensor[cur_count, 1, i, j] = off_player.y - def_player.y
                        input_tensor[cur_count, 2, i, j] = def_player_sx
                        input_tensor[cur_count, 3, i, j] = def_player_sy
                        input_tensor[cur_count, 4, i, j] = def_player_sx - ballcarrier_sx
                        input_tensor[cur_count, 5, i, j] = def_player_sy - ballcarrier_sy
                        input_tensor[cur_count, 6, i, j] = def_player.x - ballcarrier.x
                        input_tensor[cur_count, 7, i, j] = def_player.y - ballcarrier.y
                        input_tensor[cur_count, 8, i, j] = off_player_sx - def_player_sx
                        input_tensor[cur_count, 9, i, j] = off_player_sy - def_player_sy
                        yards_remaining = ball_end_x.iloc[0] - ballcarrier.x
                        target_tensor[cur_count] = yards_remaining
                        reference_tensor[cur_count, 0] = game_id
                        reference_tensor[cur_count, 1] = play_id
                        reference_tensor[cur_count, 2] = frame_id
                        positional_reference_tensor[cur_count, i, j, 0] = def_player.nflId
                        positional_reference_tensor[cur_count, i, j, 1] = off_player.nflId
                        positional_reference_tensor[cur_count, i, j, 2] = game_id
                        positional_reference_tensor[cur_count, i, j, 3] = play_id
                        positional_reference_tensor[cur_count, i, j, 4] = frame_id
                        # positional_reference_tensor[cur_count, i, j, 5] = yards_remaining
                cur_count += 1
    return input_tensor, target_tensor, reference_tensor, positional_reference_tensor