'''
Author: Yi Xu <xu.yi@northeastern.edu>
Options for models
'''

import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--evaluate', action='store_true',
                            help='False as default, claim True for evaluate, i.e., other metrics')

        # Multi-modality
        self.parser.add_argument('--k', type=int, default=20)

        # Mask type
        self.parser.add_argument('--mask_type', type=int, default=[1, 2, 3, 4, 5],
                    help='mask type, \
                        1: prediction, \
                        2: compeletion (random_consecutive_mask), \
                        3: random_discrete_mask, \
                        4: center mask, \
                        5: agent mask')
        self.parser.add_argument('--mask_weight', type=int, default=[1, 1, 1, 1, 1],
                            help='corresponding weights for each mask type')

        self.parser.add_argument('--dataset_path', type=str, default= '')
        self.parser.add_argument('--dataset_name', type=str, default='basketball')
        self.parser.add_argument('--num_agent', type=int, default=10)
        self.parser.add_argument('--model_name', type=str, default='unitraj')
        self.parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
        self.parser.add_argument('--extra_note', type=str, default='', help='Extra note for saving etc.')

        # Random seed
        self.parser.add_argument('--is_seed', type=bool, default=True, help='1 is using random seed, 0 is not using')
        self.parser.add_argument('--seed', type=int, default=2024, help='torch seed')

        # Optimizer
        self.parser.add_argument('--optimizer', type=str, default='Adam', 
                                    help='[Adam, SGD]')
        self.parser.add_argument('--nesterov', type=bool, default=True, 
                                    help='Nesterov accelerated gradien in SGD')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, 
                                    help='Weight decay for optimizer')

        # Hyper-paramter
        self.parser.add_argument('--num_epoch', type=int, default=100)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--is_adjust_lr', type=bool, default=True,
                                    help='If adjust the learning rate after each xxx epochs')
        self.parser.add_argument('--lr_step', type=int, default=20, 
                                    help='Step interval for adjust learning rate')
        self.parser.add_argument('--lr_decay', type=float, default=0.9, 
                                    help='Learning rate decay (gamma)')
        self.parser.add_argument('--clip_grad', type=float, default=None,
                                help='Gradient clipping')
        self.parser.add_argument('--train_batch_size', type=int, default=128)
        self.parser.add_argument('--test_batch_size', type=int, default=256)

        # Attention
        self.parser.add_argument('--operator', type=str, default='max', 
                                 help='[mean, max, sum]')
        self.parser.add_argument('--num_heads', type=int, default=8)

        # Model parameters
        self.parser.add_argument('--total_len', type=int, default=50)
        self.parser.add_argument('--input_dim', type=int, default=8,
                                 help='C_in, [x y vx vy mask 0 0 0 etc.]')
        self.parser.add_argument('--output_dim', type=int, default=2,
                                 help='C_out, may be xy or distributions')
        self.parser.add_argument('--delta_dim', type=int, default=2,
                                 help='temporal decay')

        self.parser.add_argument('--learn_prior', action='store_true')
        self.parser.add_argument('--z_dim', type=int, default=128)

        self.parser.add_argument('--bias', type=bool, default=False)
        self.parser.add_argument('--conv_bias', type=bool, default=False)
        self.parser.add_argument('--p_dropout', type=float, default=0.2)


        # Mamba parameters
        # Feature Dimension in Spatial/Temporal/Decoder Mamba
        self.parser.add_argument('--model_dim', type=int, default=64)
        self.parser.add_argument('--state_dim', type=int, default=64)
        self.parser.add_argument('--conv_dim', type=int, default=4)
        self.parser.add_argument('--expand', type=int, default=2)
        self.parser.add_argument('--tem_depth', type=int, default=4)

        # For Resueme
        self.parser.add_argument('--is_resume', action='store_true', 
                                 help='False default (train from scratch), claim for True (resuming training)')
        self.parser.add_argument('--resume_epoch', type=str, default='last')
        self.parser.add_argument('--start_epoch',type=int, default = -1, help = 'start epoch')

        # Torch Setting
        self.parser.add_argument('--is_dataparallel', type=bool, default=False,
                                 help='Use nn.DataParallel or not')
        self.parser.add_argument('--device_ids', nargs='+', type=int,
                                 help='Device ids for nn.DataParallel')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                help='Dataloader number of worker')

    def parse(self):
        args = self.parser.parse_args()
        if args.is_dataparallel and args.device_ids == None:
            args.device_ids = [0, 1, 2, 3]
        return args