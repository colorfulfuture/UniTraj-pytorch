'''
Author: Yi Xu
Preprocess dataset
Read Soccer datase, convert from bbx to coordinates
https://www.kaggle.com/code/chaozhuang/soccertrack-collective-dynamics-analysis/notebook
'''

import pandas as pd
import numpy as np
import pickle
import glob
import os
from tqdm import tqdm

csv_path = 'anywhere/base_datasets/soccer/top_view/annotations'
seq_len = 50
save_path = 'anywhere/base_datasets/soccer'
sliding = 4

def read_csv(csv_path, seq_len, sliding):
    all_data = []
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files.sort()
    print('Total {} csv files.'.format(len(csv_files)))
    train_index, test_index = split(len(csv_files))
    for k in tqdm(range(len(csv_files))):
        sequences = [] # data of one csv file
        data = pd.read_csv(csv_files[k], header=None)
        # Dropping the original first few rows, and the first column
        data = data.drop([0, 1, 2, 3])
        agent_columns = data.columns[1:]
        num_features = 4 # bbx
        num_agents = int(len(agent_columns)/num_features)
        assert num_agents == 23 # including ball

        for start in range(0, len(data)-seq_len+1, sliding):
            frame_data = data.iloc[start:start+seq_len].astype(float)
            if len(frame_data) == seq_len and not frame_data.isna().any().any():
                reshaped_data = frame_data.iloc[:, 1:].to_numpy().reshape(seq_len, num_agents, num_features) # [T N 4]
                x_coordinates = reshaped_data[:, :, 1] + reshaped_data[:, :, 3] / 2
                y_coordinates = reshaped_data[:, :, 2] + reshaped_data[:, :, 0] / 2
                xy_coordinates = np.stack((x_coordinates, y_coordinates), axis=-1) # [T N 2]
                # Move ball to the first
                agent_indices = np.arange(xy_coordinates.shape[1])
                new_order = np.roll(agent_indices, 1)
                xy_ballfirst = xy_coordinates[:, new_order, :]
                sequences.append(xy_ballfirst)
            else:
                continue

        sequence_array = np.array(sequences)
        all_data.append(sequence_array)
    
    train_data = [all_data[inx] for inx in train_index]
    test_data = [all_data[inx] for inx in test_index]

    train_data_array = np.concatenate(train_data, axis=0) # [N T A 2]
    test_data_array = np.concatenate(test_data, axis=0)


    # Double-check the NaN sequence
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
    
    with open(os.path.join(save_path, 'train.p'), 'wb') as f:
        pickle.dump(filtered_train_data, f)
    f.close()
    
    with open(os.path.join(save_path, 'test.p'), 'wb') as f:
        pickle.dump(filtered_test_data, f)
    f.close()



def split(N):
    '''
    total 60 files, 48 for training, 12 for testing
    data list
    '''
    x = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(x)
    train_index = x[0:48]
    test_index = x[48:]
    return train_index, test_index


if __name__ == "__main__":
    read_csv(csv_path, seq_len, sliding)