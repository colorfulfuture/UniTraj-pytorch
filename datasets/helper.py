'''
Author: Yi Xu <xu.yi@northeastern.edu>
Functions for dataloader
'''

import numpy as np
import random

class norm_unnorm():
    def __init__(self, data):
        super().__init__()
        # data: [N T A+1 2]
        self.mean = np.mean(data, axis=(0, 1, 2))
        self.std = np.std(data, axis=(0, 1, 2))

    def normalization(self, x):
        data = np.divide(x-self.mean, self.std)
        return data

    def unnormalization(self, x):
        data = np.multiply(x, self.std) + self.mean
        return data

def prediction_mask(T, A, C, start=None):
    """Generates a mask from one point to the end for prediction task
        split into observation/predcition
        each agent's different
    """
    mask = np.ones((T, A, C))
    for n in range(A):
        if start is None:
            start_random = random.randint(10, 49) # 10-48, at least one time step to prediction
        else:
            start_random = np.random.choice(start)
        mask[start_random:,n,:] = 0
    return mask

def random_consecutive_mask(T, A, C):
    """Generates a random consecutive hole"""
    mask = np.ones((T, A, C))
    for n in range(A):
        N_holes = random.randint(1, 6) # 1,2,3,4,5 holes
        for _ in range(N_holes):
            hole_len = random.randint(3, 6) # hole length 3,4,5
            limx = T - hole_len 
            start = random.randint(0, int(limx))
            mask[start:(start+hole_len),n,:] = 0
    return mask

def random_discrete_mask(T, A, C):
    """Generates a discrete mask"""
    mask = np.ones((T, A, C))
    for n in range(A):
        prob = 0.5 + 0.3 * random.random() # 0.5-0.8
        curr_mask = np.random.random(T)
        # set 0 for masked area, set 1 for unmasked area
        curr_mask = 1.0*(curr_mask >= prob)
        mask[:,n,:] = np.repeat(curr_mask[:, np.newaxis], C, axis=1)
    return mask

def center_consecutive_mask(T, A, C, masked_len=None):
    """Generates a random consecutive hole
        each agent's different
    """
    mask = np.ones((T, A, C))
    center = T//2
    half_length_min = masked_len[0]//2 # 25/2
    half_length_max = masked_len[1]//2 # 40/2
    for n in range(A):
        half_length = random.randint(half_length_min, half_length_max+1)
        mask[center-half_length:center+half_length,n,:] = 0
    return mask

def random_agent_mask(T, A, C, masked_agent=None):
    '''
    mask serveral agents
    '''
    mask = np.ones((T, A, C))
    selected_agent = np.random.choice(A, masked_agent, replace=False)
    for agent in enumerate(selected_agent):
        mask[:, agent, :] = 0
    return mask
