'''
Author: Yi Xu <xu.yi@northeastern.edu>
Runner
'''

import numpy as np
import os
import time
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.generatedataset import GenerateDataset
from models.select_model import select_model
from functions import get_masked_min_ade
from functions import evaluate_select_best
from functions import get_metrics

class Runner():
    def __init__(self, args):
        self.args = args
        self._init_mkdir()
        self.save_args()
        self.summary = SummaryWriter(self.boardx_dir)
        
    def _init_mkdir(self):
        self.start_epoch = 0
        self.global_step = 0
        self.best_ade = 10000
        self.best_epoch_ade = 0
        # self.this_fde = 0
        if self.args.extra_note:
            self.save_dir = os.path.join(self.args.checkpoint_path + '_' + self.args.extra_note,
                                         self.args.dataset_name, 
                                         self.args.model_name)
        else:
            self.save_dir = os.path.join(self.args.checkpoint_path,
                                         self.args.dataset_name, 
                                         self.args.model_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.boardx_dir = os.path.join(self.save_dir, 'tensorboard')
        if not os.path.exists(self.boardx_dir):
            os.makedirs(self.boardx_dir)

    def save_args(self):
        args_dict = vars(self.args)
        with open(os.path.join(self.save_dir, 'args_dict.p'), 'wb') as f:
            pickle.dump(args_dict, f)
        f.close()
        # Also save one copy of txt
        with open(os.path.join(self.save_dir, 'args_dict.txt'), 'w') as f:
             for key, value in args_dict.items():
                f.write(f'{key}: {value}\n')
        f.close()

    def start(self):
        self.load_data()
        self.load_model()
        self.load_optimizer()
        # If resume from some epoch
        if self.args.is_resume:
            self.resume_model()
        print('==========================================')
        print('>>> Begin Training......')
        for self.epoch in range(self.start_epoch, self.args.num_epoch):
            print('==========================================')
            print('>>> Train Epoch: {}'.format(self.epoch))
            self.train_epoch()
            print('------------------------------------------')
            print('<<< Test Epoch: {}'.format(self.epoch))
            self.test_epoch()

        print('==========================================')
        print('Done!')

        print('Best minADE is: {}, of Epoch: {}'
              .format(self.best_ade, self.best_epoch_ade))
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as log_file:
            log_file.write('-'*50)
            log_file.write('Epoch: '+str(self.best_epoch_ade)+'\n')
            log_file.write('Best minADE: '+str(self.best_ade)+'\n')
            log_file.write('\n')
        log_file.close()

    def load_data(self):
        print('==========================================')
        print('>>> Begin Loading Training Data')
        train_dset = GenerateDataset(
                    is_train = True,
                    dataset_path = self.args.dataset_path,
                    dataset_name = self.args.dataset_name,
                    mask_type = self.args.mask_type,
                    mask_weight = self.args.mask_weight,
                    norm_cla = None)
        self.train_loader = DataLoader(
                    train_dset, 
                    batch_size = self.args.train_batch_size, 
                    shuffle = True,
                    num_workers = self.args.num_workers)
        print('>>> Load Training Data, Done!')

        print('>>> Begin Loading Testing Data')
        self.norm_unnorm = train_dset.norm_unnorm
        test_dset = GenerateDataset(
                    is_train = False,
                    dataset_path = self.args.dataset_path,
                    dataset_name = self.args.dataset_name,
                    mask_type = self.args.mask_type,
                    mask_weight = self.args.mask_weight,
                    norm_cla = self.norm_unnorm)
        self.test_loader = DataLoader(
                    test_dset, 
                    batch_size = self.args.test_batch_size, 
                    shuffle = False,
                    num_workers = self.args.num_workers)
        print('>>> Load Testing Data, Done!')

        self.N_Train = train_dset.__len__()
        self.N_Test = test_dset.__len__()
        print('>>> Number of Training Data: {}'
              .format(self.N_Train))
        print('>>> Number of Testing Data: {}'
              .format(self.N_Test))

    def load_model(self):
        print('==========================================')
        print('>>> Begin Loading Model')
        model = select_model(self.args.model_name)(self.args)
        print(">>> Total params: {:.6f}M"
              .format(sum(p.numel() for p in model.parameters()) / 1000000.0))
        if self.args.is_dataparallel:
            self.model = nn.DataParallel(model, device_ids = self.args.device_ids) # multi-GPU
        else:
            self.model = model
        self.model.cuda()

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                            self.model.parameters(),
                            lr = self.args.lr,
                            momentum = 0.9,
                            nesterov = self.args.nesterov,
                            weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                            self.model.parameters(),
                            lr = self.args.lr,
                            weight_decay = self.args.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.epoch % self.args.lr_step == 0:
            self.args.lr = self.args.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
            
    def save_model(self, ade, prefix = 'last'):
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        state = {
                'epoch': self.epoch,
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'ade': ade
                }
        name = os.path.join(self.save_dir, prefix + '_epoch.pth')
        torch.save(state, name)

    def resume_model(self):
        if self.args.resume_epoch == 'last':
            epoch2resume = os.path.join(self.save_dir, 'last_epoch.pth')
        else:
            epoch2resume = os.path.join(self.save_dir, self.args.resume_epoch+'_epoch.pth')
        state = torch.load(epoch2resume)
        self.epoch = state['epoch']
        self.optimizer.load_state_dict(state['optimizer'])
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])
        print('>>> Resume from epoch: {}'.format(self.epoch))

        # Restart epoch
        self.start_epoch = self.epoch + 1

        # Get information from best
        best_epoch_name = os.path.join(self.save_dir, 'best_ade_epoch.pth')
        best_state = torch.load(best_epoch_name)
        self.best_ade = best_state['ade']
        self.best_epoch_ade = best_state['epoch']
        print('>>> Current best testing ade: {} from epoch: {}'
              .format(self.best_ade, self.best_epoch_ade))

    def train_epoch(self):
        time_s = time.time()
        self.model.train()

        if self.args.is_adjust_lr:
            self.adjust_lr()

        epoch_recons_loss = 0
        epoch_kld_loss = 0
        epoch_diverse_loss = 0
        epoch_loss = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, ncols=50, ascii=True)):

            data = batch[0].cuda()
            mask = batch[1].cuda()

            self.global_step += 1
            self.optimizer.zero_grad()
            kld_loss, recons_loss, diverse_loss = self.model(data, mask)
            loss = recons_loss + self.args.lambda_kld * kld_loss + self.args.lambda_diverse * diverse_loss

            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            loss.backward()
            self.optimizer.step()

            # Print
            if batch_idx % 20 == 0 or batch_idx == (len(self.train_loader) - 1):
                tqdm.write('Batch: {}, MSE Loss: {:.4f}, KLD Loss: {:.8f}, Diverse Loss: {:.4f},'
                           'Total Weighted Training Loss: {:.4f} with Weight: {:.4f} and {:.4f}' 
                           .format(batch_idx, recons_loss, kld_loss, diverse_loss, loss, 
                                self.args.lambda_kld, self.args.lambda_diverse))

            # Record
            self.summary.add_scalar(f'Recons_MSE_Loss/Batch', recons_loss.item(), self.global_step)
            self.summary.add_scalar(f'KLD_Loss/Batch', kld_loss.item(), self.global_step)
            self.summary.add_scalar(f'Diverse_loss/Batch', diverse_loss.item(), self.global_step)
            self.summary.add_scalar(f'Total_Weighted_Loss/Batch', loss.item(), self.global_step)

            # Sum losses of all batches
            epoch_diverse_loss += diverse_loss.item()*data.size()[0]
            epoch_recons_loss += recons_loss.item()*data.size()[0]
            epoch_kld_loss += kld_loss.item()*data.size()[0]
            epoch_loss += loss.item()*data.size()[0]

        time_e = time.time()

        # Average of one epoch
        ave_diverse_loss = epoch_diverse_loss/self.N_Train
        ave_recons_loss = epoch_recons_loss/self.N_Train
        ave_kld_loss = epoch_kld_loss/self.N_Train
        ave_epoch_loss = epoch_loss/self.N_Train
        tqdm.write('>>> Train Epoch: {} Finished, Average Recons MSE Loss: {:.4f}, KLD Loss: {:.4f}, Diverse Loss: {:.4f}, Weighted Training Loss: {:.4f}, Time {:.4f}'
              .format(self.epoch, ave_recons_loss, ave_kld_loss, ave_diverse_loss, ave_epoch_loss, (time_e - time_s)))

        # Log file
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as log_file:
            log_file.write('Epoch: '+str(self.epoch)+'\n')
            log_file.write('Recons MSE Loss: '+str(ave_recons_loss)+'\n')
            log_file.write('KLD Loss: '+str(ave_kld_loss)+'\n')
            log_file.write('Diverse WinnerTA Loss: '+str(ave_diverse_loss)+'\n')
            log_file.write('Weighted Training Loss: '+str(ave_epoch_loss)+'\n')
        log_file.close()

    def test_epoch(self):
        time_s = time.time()

        epoch_ade = 0
        epoch_ade_count = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):

                data = batch[0].cuda()
                mask = batch[1].cuda()
                # GT
                y = data

                self.global_step += 1
                self.optimizer.zero_grad()
                out = self.model.inference(data, mask) # [B T N K 2]

                # Unnormalize
                out_un = self.norm_unnorm.unnormalization(out.detach().cpu().numpy())
                gt_un = self.norm_unnorm.unnormalization(y.detach().cpu().numpy())
                mask_np = mask.cpu().numpy()
                
                # For this batch
                ade_sum, ade_count = get_masked_min_ade(out_un, gt_un, mask_np)

                epoch_ade += ade_sum
                epoch_ade_count += ade_count

        # Average
        ade = epoch_ade/epoch_ade_count

        time_e = time.time()
        print('<<< Test Epoch: {} Finished, minADE: {:.4f}, Time {:.4f}'
              .format(self.epoch, ade, (time_e - time_s)))

        # Record
        self.summary.add_scalar(f'minADE/Epoch', ade, self.epoch)

        # Log file
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as log_file:
            log_file.write('Testing Result of minADE: '+str(ade)+'\n')
            log_file.write('\n')
        log_file.close()

        # Save this epoch model
        self.save_model(ade)

        # ADE
        if ade < self.best_ade:
            self.best_ade = ade
            self.best_epoch_ade = self.epoch
            self.save_model(ade, prefix='best_ade')
        
        print('Best minADE is: {}, of Epoch: {}'.format(self.best_ade, self.best_epoch_ade))


    def evaluate(self):
        # Load testing data
        print('==========================================')
        print('>>> Extract Normalization Func from Training Data')
        train_dset = GenerateDataset(
                    is_train = True,
                    dataset_path = self.args.dataset_path,
                    dataset_name = self.args.dataset_name,
                    mask_type = self.args.mask_type,
                    mask_weight = self.args.mask_weight,
                    norm_cla = None)

        print('>>> Begin Loading Testing Data')
        self.norm_unnorm = train_dset.norm_unnorm
        test_dset = GenerateDataset(
                    is_train = False,
                    dataset_path = self.args.dataset_path,
                    dataset_name = self.args.dataset_name,
                    mask_type = self.args.mask_type,
                    mask_weight = self.args.mask_weight,
                    norm_cla = self.norm_unnorm)
        self.test_loader = DataLoader(
                    test_dset, 
                    batch_size = self.args.test_batch_size, 
                    shuffle = False,
                    num_workers = self.args.num_workers)
        print('>>> Load Testing Data, Done!')

        self.N_Test = test_dset.__len__()
        print('>>> Number of Testing Data: {}'
              .format(self.N_Test))

        # Load model and the best epoch for evaluation
        self.load_model()
        best_epoch_name = os.path.join(self.save_dir, 'best_ade_epoch.pth')
        best_state = torch.load(best_epoch_name)
        self.epoch = best_state['epoch'] # which epoch
        # model state dict
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(best_state['model'])
        else:
            self.model.load_state_dict(best_state['model'])
        print('>>> Load from epoch: {}'.format(self.epoch))

        # of which has the best ade
        self.best_ade = best_state['ade']
        self.best_epoch_ade = best_state['epoch']
        print('>>> Best minADE: {}'.format(self.best_ade))

        # Begin evaluate
        time_s = time.time()

        self.model.eval()
        out_all = []
        gt_all = []
        mask_all = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):

                data = batch[0].cuda()
                mask = batch[1].cuda()
                # GT
                y = data

                out_diverse = self.model.inference(data, mask) # [B T N K 2]

                # Unnormalize
                out_un_diverse = self.norm_unnorm.unnormalization(out_diverse.detach().cpu().numpy()) # [B T N K 2]
                gt_un = self.norm_unnorm.unnormalization(y.detach().cpu().numpy()) # [B T N 2]
                mask_np = mask.cpu().numpy() # [B T N 2]

                # Select out the out with the smallest ADE for further evaluation
                out_un = evaluate_select_best(out_un_diverse, gt_un, mask_np)

                out_all.append(out_un)
                gt_all.append(gt_un)
                mask_all.append(mask_np)
        
        out_all_cat = np.concatenate(out_all, axis=0)
        gt_all_cat =  np.concatenate(gt_all, axis=0)
        mask_all_cat = np.concatenate(mask_all, axis=0)

        results = {
            'out': out_all_cat,
            'gt': gt_all_cat,
            'mask': mask_all_cat
        }

        metrics = get_metrics(out_all_cat, gt_all_cat, mask_all_cat, self.args.dataset_name)
        metrics['Epoch'] = self.epoch
        metrics['ADE'] = self.best_ade

        time_e = time.time()

        # Print
        import pprint
        print('<<< Evaluate Epoch: {} Finished, Time {:.4f}'
              .format(self.epoch, (time_e - time_s)))
        pprint.pprint(metrics)

        # Save
        with open(os.path.join(self.save_dir, 'results.p'), 'wb') as f:
            pickle.dump(results, f)
        f.close()

        with open(os.path.join(self.save_dir, 'results.txt'), 'a') as log_file:
            log_file.write(str(metrics) + '\n')
        log_file.close()