'''
Author: Yi Xu <xu.yi@northeastern.edu>
Main function
'''

from options import Options
import torch
import random
import numpy as np
from runner import Runner
from pprint import pprint

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def init_configs(args):
    if args.dataset_name == 'basketball':
        args.num_agent = 10
    elif args.dataset_name == 'football':
        args.num_agent = 22
    elif args.dataset_name == 'soccer':
        args.num_agent = 22

def main(args):
    init_configs(args)
    print("\n================== Arguments =================")
    pprint(vars(args), indent=4)
    print("==========================================\n")
    if args.is_seed:
        init_seed(args.seed)
    engine = Runner(args)
    if args.evaluate:
        engine.evaluate()
    else: # Train
        engine.start()


if __name__ == '__main__':
    args = Options().parse()
    main(args)