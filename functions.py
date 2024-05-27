'''
Author: Yi Xu <xu.yi@northeastern.edu>
Some functions
'''

import torch
import numpy as np
from numpy import linalg as LA

def get_oob(loc, m, name):
    # Boundary
    if name == 'basketball':
        LENGTH = 94
        WIDTH = 50
    elif name == 'football':
        LENGTH = 120
        WIDTH = 53.5
    elif name == 'soccer':
        LENGTH = 3840
        WIDTH = 2160
    loc2evaluate = loc
    denominator = np.sum(m) # doubled total positions because x/y both counts
    x_out = np.sum(loc2evaluate[:,:,:,0] > LENGTH)
    y_out = np.sum(loc2evaluate[:,:,:,1] > WIDTH)
    oob = (x_out + y_out)/denominator
    return oob

def get_step(loc):
    loc_change = loc[:,1:,:,:] - loc[:,:-1,:,:]
    step_distance = LA.norm(loc_change, axis=-1)
    step_change = np.abs(step_distance[:,1:,:] - step_distance[:,:-1,:])
    return np.mean(step_change)


def get_path_l_d(loc):
    loc_change = loc[:,1:,:,:] - loc[:,:-1,:,:] #[B T-1 N]
    step_distance = LA.norm(loc_change, axis=-1) #[B T-1 N]
    traj_len = np.sum(step_distance, axis=1) #[B N]
    max_traj_len = np.max(traj_len)
    min_traj_len = np.min(traj_len)
    max_min_diff = max_traj_len - min_traj_len
    return np.mean(traj_len), max_min_diff

def get_metrics(x, y, m, name=None):
    '''
    Input: x/y/m (Array) [B T N 2]
           name: basketball/football/

    Return: OOB: average out-of-bound rate
            Step: average step size change
            Path-L: average trajectory length
            Path-D: max-min path difference
    '''
    traj = x
    oob = get_oob(traj, m, name)
    step = get_step(traj)
    path_l, path_d = get_path_l_d(traj)
    metrics = {
        'OOB': oob,
        'Step': step,
        'Path-L': path_l,
        'Path-D': path_d,
    }
    return metrics

def get_trainiong_loss(x, y):
    '''
    Input: x/y (Tensor) [B T N 2]
    Return: average error of each agent of each time step
    '''
    results = torch.norm(y-x, p=2, dim=-1) #[B T N]
    return torch.mean(results)

def get_testing_metric(x, y):
    '''
    Input: x/y (Array) [B T N 2]
    Return: ADE and FDE
    '''
    B, T, N, _ = x.shape
    results = LA.norm((y-x), axis=-1) #[B T N]
    ade_sum = np.sum(results)
    fde_sum = np.sum(results[:,-1,:])
    ade_count = B * T * N
    fde_count = B * N
    return ade_sum, ade_count, fde_sum, fde_count

def evaluate_select_best(y_hat_diverse, y, m):
    '''
    Input: y_hat_diverse (Array) [B T N K 2]
           y (Array) [B T N 2]
           m (Array) [B T N 2]
    Return: y_hat_best [B T N 2] has best ADE
    '''
    K = y_hat_diverse.shape[3]
    valid_mask = 1 - m[:,:,:,0] # 1: for eval
    y_repeat = np.repeat(y[:,:,:,np.newaxis,:], K, axis=3)
    distance_k = LA.norm((y_repeat-y_hat_diverse), axis=-1)
    ade_k = distance_k*valid_mask[:,:,:,np.newaxis] # with only masked locations
    sum_ade_k = np.sum(ade_k, axis=(0,1,2)) # [K]
    index = np.argmin(sum_ade_k)
    y_hat_best = y_hat_diverse[:,:,:,index,:]
    return y_hat_best

def get_masked_min_ade(x, y, m):
    '''
    Input: x (Array) [B T N K 2]
           y (Array) [B T N 2]
           m (Array) [B T N 2]
    Return: ADE Sum and Count
    '''
    K = x.shape[3]
    valid_mask = 1 - m[:,:,:,0] # [B T N] 1: for eval
    y_repeat = np.repeat(y[:,:,:,np.newaxis,:], K, axis=3) # [B T N K 2]
    distance_k = LA.norm((y_repeat-x), axis=-1) # [B T N K]
    ade_k = distance_k*valid_mask[:,:,:,np.newaxis] # [B T N K] with only masked locations
    min_ade_sum = np.min(np.sum(ade_k, axis=(0,1,2))) # [K]
    ade_count = np.count_nonzero(valid_mask)
    return min_ade_sum, ade_count