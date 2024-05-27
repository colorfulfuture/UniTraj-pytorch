'''
Author: Yi Xu <xu.yi@northeastern.edu>
Dataloader
'''

import torch
from torch.utils.data import Dataset
import os
import pickle
from datasets.helper import norm_unnorm
from datasets.helper import prediction_mask, random_consecutive_mask, random_discrete_mask, center_consecutive_mask, random_agent_mask
import numpy as np
from random import choices
from tqdm import tqdm

class GenerateDataset(Dataset):
    def __init__(self, is_train, dataset_path, dataset_name, mask_type, mask_weight, norm_cla=None):
        super().__init__()
        self.is_train = is_train
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.type = mask_type
        self.weight = mask_weight
        assert len(self.type) == len(self.weight)
        if self.is_train:
            filename = os.path.join(dataset_path, dataset_name, 'train_clean.p')
            maskname = os.path.join(dataset_path, dataset_name, 'train_mask.p')
        else:
            filename = os.path.join(dataset_path, dataset_name, 'test_clean.p')
            maskname = os.path.join(dataset_path, dataset_name, 'test_mask.p')
        assert os.path.exists(filename) == True
        with open(filename, 'rb') as f:
            data= pickle.load(f)
        f.close()
        # Get the mask
        if os.path.exists(maskname):
            with open(maskname, 'rb') as f:
                mask2read = pickle.load(f)
            f.close()
            self.mask_type_count = mask2read['stats']
            self.mask= mask2read['value']
        else:
            print('No existing mask dataset, now generating...')
            self.mask_type_count = {}
            self.mask = self.generate_mask(data) # [N T A+1 2]
            print('Done!')
            print('Now saving...')
            mask2save= {
                'stats': self.mask_type_count,
                'value': self.mask
                }
            with open(maskname, 'wb') as f:
                pickle.dump(mask2save, f)
            f.close()
            print('Done!')

        if self.is_train:
            self.norm_unnorm = norm_unnorm(data)
        else:
            self.norm_unnorm = norm_cla
        self.data = torch.from_numpy(self.norm_unnorm.normalization(data)).float() # [N T A+1 2]
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
        self.data  # [N T A+1 2]
        self.mask # [N T A+1 2]
        Return: [B T A+1 2]
        '''
        batch_data = self.data[index]
        batch_mask = self.mask[index]

        return batch_data, batch_mask

    def generate_mask(self, data):
        '''
        Input: data: [B T A+1 2]
                self.type: list [1, 2, 3, 4, 5]
                self.weight: list [w1, w2, w3, w4, w5]
        Return: mask: [N T A+1 2]
        '''
        num_seq, T, num_agent, dim_in = data.shape
        mask = np.ones_like(data).astype(np.float32)
        for k in tqdm(range(num_seq)):
            mask_type = choices(self.type, self.weight)[0]
            if mask_type in self.mask_type_count:
                self.mask_type_count[mask_type] += 1
            else:
                self.mask_type_count[mask_type] = 1
            # prediction, mask from a middle point to the end
            if mask_type == 1:
                mask[k,:,:,:] = prediction_mask(T, num_agent, dim_in, start=[25, 30, 35, 40])
            # completion, mask in the middle part
            if mask_type == 2:
                mask[k,:,:,:] = random_consecutive_mask(T, num_agent, dim_in)
            # random discrete mask
            if mask_type == 3:
                mask[k,:,:,:] = random_discrete_mask(T, num_agent, dim_in)
            # random discrete mask
            if mask_type == 4:
                mask[k,:,:,:] = center_consecutive_mask(T, num_agent, dim_in, masked_len=[25, 40])
            # random player mask
            if mask_type == 5:
                mask[k,:,:,:] = random_agent_mask(T, num_agent, dim_in, masked_agent=5)
        return mask