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
    tracking = tracking[tracking['playId'].isin(tracking[tracking['event'] != 'fumble']['playId'].unique())]
    plays = plays[plays['playNullifiedByPenalty'] == 'N']

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

    bc_coords=tracking_with_plays.loc[tracking_with_plays['is_ballcarrier']]
    bc_coords['bc_x']=bc_coords['x']
    bc_coords['bc_y']=bc_coords['y']
    bc_coords=bc_coords[['gameId', 'playId', 'frameId', 'bc_x', 'bc_y']]
    tracking_with_plays=tracking_with_plays.merge(bc_coords, on=['gameId', 'playId', 'frameId'], how='left')

    end_frame = tracking_with_plays[tracking_with_plays['event'].isin(['tackle', 'out_of_bounds'])].groupby(['gameId', 'playId'])['frameId'].min().reset_index()
    end_frame.rename(columns={'frameId': 'frameId_end'}, inplace=True)

    start_frame = tracking_with_plays[tracking_with_plays['event'].isin(['run', 'lateral', 'run_pass_option', 'handoff', 'pass_arrived'])].groupby(['gameId', 'playId'])['frameId'].min().reset_index()
    start_frame.rename(columns={'frameId': 'frameId_start'}, inplace=True)

    tracking_with_plays = tracking_with_plays.merge(start_frame, on=['gameId', 'playId'], how='left')
    tracking_with_plays = tracking_with_plays.merge(end_frame, on=['gameId', 'playId'], how='left')

    tracking_with_plays = tracking_with_plays[(tracking_with_plays['frameId'] <= tracking_with_plays['frameId_end']) &
                                              (tracking_with_plays['frameId'] >= tracking_with_plays['frameId_start'])]

    return tracking_with_plays


def create_feature_tensor(tracking_with_plays):
    """
    Create the input tensor for the model
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
                defense = frame_group[frame_group['is_on_defense']]
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