import torch.nn as nn
import numpy as np
import os
from ops.os_operation import mkdir
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model.MoCo import MoCo
from model.MaskedVideoModel import MaskedVideoModel
import torch
import random
from inference.infer_utils import init_log_path
from inference.encode_video_feature import encode_video_feature
from data_processing.build_dataloader import build_infer_loader
from data_processing.Feature_Dataset_Mask import Feature_Dataset_Mask
def infer_coseg_model(data_path, params):
    #init path to save output results
    save_path = init_log_path(params)

    #configure random seed
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    #configure dataset
    loader = build_infer_loader(data_path,params)

    #configure feature model
    n_channels = 512
    feature_model = MoCo('resnet18', params['encode_feature'], params['moco_k'], params['moco_m'], params['moco_T'])
    feature_model = feature_model.cuda()
    feature_model = nn.DataParallel(feature_model, device_ids=None)
    if params['resume']:
        model_path = params['M']
        state_dict = torch.load(model_path)
        msg=feature_model.load_state_dict(state_dict['feature_state'])
        print("feature model loading msg:", msg)

    #run feature encodings first to save time for later processes

    feature_model.eval()
    feature_path = os.path.join(save_path, "Contrastive_Feature.npy")
    Video_Refer_Dict = loader.dataset.Refer_Video_Dict

    if not os.path.exists(feature_path):
        Feature_Matrix = encode_video_feature(feature_model, loader, n_channels)
        np.save(feature_path, Feature_Matrix)
    else:
        Feature_Matrix = np.load(feature_path)



    #configure masked model
    Bert_Model = MaskedVideoModel(n_channels,n_channels, params['window_length'], params['num_layers'],
                                  attn_heads=params['num_attention'], dropout=0)
    Bert_Model = Bert_Model.cuda()
    Bert_Model = nn.DataParallel(Bert_Model, device_ids=None)
    if params['resume']:
        msg=Bert_Model.load_state_dict(state_dict['bert_state'])
        print("bert model loading msg:",msg)
    Bert_Model.eval()

    feature_dataset = Feature_Dataset_Mask(Feature_Matrix, Video_Refer_Dict, params['window_length'],
                                               params['mask_length'])
    dataloader = DataLoader(feature_dataset, params['batch_size'], shuffle=False, drop_last=False,
                            num_workers=params['num_workers'])

    mse_path = os.path.join(save_path, 'MSE_window_' + str(params['window_length'])+"_mask_"+str(params['mask_length']))
    mkdir(mse_path)
    listfiles = [x for x in os.listdir(mse_path) if "mse.txt" in x]
    if len(listfiles) != len(Video_Refer_Dict):
        from inference.gen_coseg_mse import gen_coseg_mse
        with torch.no_grad():
            gen_coseg_mse(mse_path, dataloader, Bert_Model,
                         use_cuda, params['batch_size'], n_channels,
                          params, Video_Refer_Dict,params['study_length'])








