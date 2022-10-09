# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/

import os
import numpy as np
from ops.argparser import argparser
if __name__ == "__main__":
    params = argparser()
    print(params)

    if params['mode'] == 0:
        data_path = params['F']
        dataset_name = params['dataset']
        if params['gpu']!="":
            os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        import torch

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # use the 2nd layer conv+attention and then predict: type=1 and attention=0
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        if use_cuda:
            torch.cuda.manual_seed_all(params['seed'])
            print ('Available devices ', torch.cuda.device_count())
            from training.train_coseg import train_coseg

            train_coseg(data_path, params)
        else:
            print("CoSeg requires cuda for efficient training.")
    
