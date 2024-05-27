'''
Author: Yi Xu
Preprocess dataset
Transfer football (91) csv files to npz
'''

import pandas as pd
import numpy as np
import pickle
import glob
import os
from tqdm import tqdm

csv_path = 'anywhere/base_datasets/football'
seq_len = 50 # To be same as basketball dataseet
only_middle = False

def read_csv(csv_path, seq_len, sliding=50):
    all_data = []
    file_list = glob.glob(os.path.join(csv_path, '*.csv'))
    print('Total {} csv files.'.format(len(file_list)))
    train_index, test_index = split(len(file_list))
    for k in tqdm(range(len(file_list))):
        csv_data = [] # data of one csv file
        data = pd.read_csv(file_list[k], index_col=['playId'])
        # each 'playID' is a unique sequence
        unique_seq = data.index.unique()
        for i in tqdm(range(len(unique_seq))):
            # for each sequence, get the player list
            df = data.loc[unique_seq[i]].set_index('nflId')
            d_football = df[df.index.isna()] # football data
            d_player = df[~df.index.isna()] # player data
            unique_player = d_player.index.unique()
            if len(unique_player) != 22 or len(d_football) == 0:
                # some sequences don't have 22 players, just disregard
                # some sequences don't have the tracking data of the ball
                continue
            t_list = []
            feat_list = []
            football_feat = d_football[['x','y']].to_numpy() # [T 5]
            if np.any(np.isnan(football_feat)):
                continue
            # append the football feature and the seq length
            feat_list.append(football_feat)
            t_list.append(football_feat.shape[0])
            for n in range(len(unique_player)):
                # player_feat = d_player.loc[[unique_player[n]]][['x','y','s','dis','dir']].to_numpy() # [T 5]
                # x, y, speed, distance, motion angel
                player_feat = d_player.loc[[unique_player[n]]][['x','y']].to_numpy() # [T 2]
                # if np.any(np.isnan(player_feat)):
                #     print("Nan in player feat")
                # append each player feature and the seq length
                feat_list.append(player_feat)
                t_list.append(player_feat.shape[0])
            # if have the same length
            assert len(t_list) == 23
            assert all(x == t_list[0] for x in t_list) == True
            t_standard = min(t_list)
            if t_standard < 50:
                continue
            else:
                if only_middle:
                    start_idx = int((t_standard - seq_len)/2)
                    seq_data = np.zeros((1, seq_len, len(t_list), 2)).astype(np.float32) #[1 50 22+1 2]
                    for n in range(len(feat_list)): # 22+1 
                        seq_data[0,:,n,:] = feat_list[n][start_idx:start_idx+seq_len,:]
                else:
                    seq_num = int((t_standard-seq_len)/sliding)+1
                    seq_data = np.zeros((seq_num, seq_len, len(t_list), 2)).astype(np.float32) #[n_seq 50 22+1 2]
                    for j in range(seq_num):
                        for n in range(len(feat_list)): # 22+1
                            seq_data[j,:,n,:] = feat_list[n][j*sliding:j*sliding+seq_len,:]
            csv_data.append(seq_data)
        csv_data_array = np.concatenate(csv_data, axis=0)
        all_data.append(csv_data_array)
    train_data = [all_data[inx] for inx in train_index]
    test_data = [all_data[inx] for inx in test_index]
    
    train_data_array = np.concatenate(train_data, axis=0) # [N T A 2]
    test_data_array = np.concatenate(test_data, axis=0)
    
    # Remove the NaN sequence
    filtered_data = []
    for i in range(train_data_array.shape[0]):
        if not np.isnan(train_data_array[i]).any():  # if no NaN in [T, A, 2]
            filtered_data.append(train_data_array[i])
    filtered_train_data = np.array(filtered_data)
    
    filtered_data = []
    for i in range(test_data_array.shape[0]):
        if not np.isnan(test_data_array[i]).any():  # if no NaN in [T, A, 2]
            filtered_data.append(test_data_array[i])
    filtered_test_data = np.array(filtered_data)
    
    with open(os.path.join(csv_path, 'train.p'), 'wb') as f:
        pickle.dump(filtered_train_data, f)
    f.close()
    
    with open(os.path.join(csv_path, 'test.p'), 'wb') as f:
        pickle.dump(filtered_test_data, f)
    f.close()

def split(N):
    '''
    total 91 matches, 73 for training, 18 for testing
    data list
    '''
    assert N == 91
    x = np.arange(91)
    np.random.seed(0)
    np.random.shuffle(x)
    train_index = x[0:73]
    test_index = x[73:]
    return train_index, test_index


if __name__ == "__main__":
    read_csv(csv_path, seq_len)