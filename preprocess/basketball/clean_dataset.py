'''
Author: Yi Xu
Clean basektball dataset
remove sequence out of the basketball court
'''

import numpy as np
import os
import pickle

DATAPATH = 'anywhere/base_datasets/basketball'
# Feet
LENGTH = 94
WIDTH = 50
N_AGENTS = 10

def fetch_clean(is_train):
    if is_train:
        filename = os.path.join(DATAPATH, 'train.npz')
    else:
        filename = os.path.join(DATAPATH, 'test.npz')
    # Load data
    assert os.path.isfile(filename)
    ori_data = np.load(filename)['data']
    N, seq_len, dim = ori_data.shape
    data = np.reshape(ori_data, (N,
                                 seq_len, 
                                 N_AGENTS + 1, 
                                 dim//(N_AGENTS+1))) # [N 50 11 2]
    # Clean the data
    ballx = data[:,:,0:1,0]
    ballx_lar = np.argwhere(ballx>=LENGTH)
    ballx_sml = np.argwhere(ballx<=0)
    bally = data[:,:,0:1,1]
    bally_lar = np.argwhere(bally>=WIDTH)
    bally_sml = np.argwhere(bally<=0)
    x = data[:,:,1:,0]
    y = data[:,:,1:,1]
    x_lar = np.argwhere(x>=LENGTH)
    x_sml = np.argwhere(x<=0)
    y_lar = np.argwhere(y>=WIDTH)
    y_sml = np.argwhere(y<=0)
    all=  np.concatenate((ballx_lar, ballx_sml, bally_lar, bally_sml, x_lar, x_sml, y_lar, y_sml))
    unique_ind = np.unique(all[:,0])
    clean_data = np.delete(data, unique_ind, axis=0)
    if is_train:
        clean_filename = os.path.join(DATAPATH, 'train_clean.p')
    else:
        clean_filename = os.path.join(DATAPATH, 'test_clean.p')
    with open(clean_filename, 'wb') as f:
        pickle.dump(clean_data, f)
    f.close()
    return clean_data


if __name__ == "__main__":
    train_data = fetch_clean(is_train=True)
    test_date = fetch_clean(is_train=False)