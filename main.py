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
        video_path=os.path.abspath(params['F'])
        frame_path=os.path.abspath(params['M'])#specify the directory to save frames
        dataset_name = params['dataset']
        if dataset_name=="B":
            from data_processing.Gen_Frame import Gen_Breakfast_Frame
            Gen_Breakfast_Frame(video_path,frame_path)
        elif dataset_name=="S":
            from data_processing.Gen_Frame import Gen_Salad_Frame
            Gen_Salad_Frame(video_path,frame_path)
        elif dataset_name=="I":
            from data_processing.Gen_Frame import Gen_INRIA_Frame
            Gen_INRIA_Frame(video_path,frame_path)

    elif params['mode'] == 1:
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

            print ('Available devices ', torch.cuda.device_count())
            from training.train_coseg import train_coseg

            train_coseg(data_path, params)
        else:
            print("CoSeg requires cuda for efficient training.")

    elif params['mode']==2:
        #inference of coseg.
        data_path = params['F']  # the frame directory
        dataset_name = params['dataset']
        if params['gpu'] != "":
            os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        import torch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        if use_cuda:
            from inference.infer_coseg_model import infer_coseg_model
            infer_coseg_model(data_path, params)
        else:
            print("CoSeg requires cuda for inference.")

