import numpy as np
import os
from ops.os_operation import mkdir
from training.train_utils import  init_log_path,init_Logger,save_checkpoint
from training.train_coseg_model import train_coseg_model
from model.MoCo import MoCo
from model.MaskedVideoModel import MaskedVideoModel
import torch.nn as nn
from data_processing.build_dataloader import build_dataloader
from torch import optim
import torch
import random
def train_coseg(data_path, params):
    data_path = os.path.abspath(data_path)

    random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # confim log path
    log_path, result_path = init_log_path(params)

    #configure model settings
    #configure moco self supervised model
    feature_model = MoCo('resnet18', params['encode_feature'], params['moco_k'], params['moco_m'], params['moco_T'])
    #because ResNet18 output channels after avg pool is 512.
    n_channels = 512

    Bert_Model = MaskedVideoModel(n_channels,n_channels, params['window_length'], params['num_layers'],
                                  attn_heads=params['num_attention'], dropout=params['dropout'])


    feature_model = feature_model.cuda()
    feature_model = nn.DataParallel(feature_model, device_ids=None)
    Bert_Model = Bert_Model.cuda()
    Bert_Model = nn.DataParallel(Bert_Model, device_ids=None)

    #configure dataset
    train_dataloader = build_dataloader(data_path, params)

    #configure optimizer
    feature_optimizer = optim.SGD(feature_model.parameters(), lr=params['lr'],
                                  momentum=params['momentum'], weight_decay=params['weight_decay'], nesterov=True)
    bert_optimizer = optim.SGD(Bert_Model.parameters(), lr=params['lr'],
                               momentum=params['momentum'], weight_decay=params['weight_decay'], nesterov=True)
    start_epoch = params['start_epoch']
    if params['resume']:
        model_path=params['M']
        state_dict=torch.load(model_path)
        feature_model.load_state_dict(state_dict['feature_state'])
        Bert_Model.load_state_dict(state_dict['bert_state'])
        bert_optimizer.load_state_dict(state_dict['bert_optimizer'])
        feature_optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
    #configure log settings
    writer = None
    if params['tensorboard']:
        from tensorboardX import SummaryWriter
        log_tensor = os.path.join(log_path, 'Tensorboard')
        writer = SummaryWriter(log_tensor)
        writer.add_text('Text', str(params))
    train_logger, train_batch_logger = init_Logger(log_path)


    best_accu = 0
    best_accu2 = 0

    for epoch in range(start_epoch, params['epochs']):
        train_loss, ctop1,btop1 = train_coseg_model(epoch, feature_model, Bert_Model, feature_optimizer,
                                                  bert_optimizer, train_dataloader,
                                                    params,writer, train_logger,
                                                  train_batch_logger)
        is_best1 = best_accu < ctop1
        best_accu = max(best_accu,ctop1)
        is_best2= best_accu2 < btop1
        best_accu2 = max(best_accu2, btop1)

        state = {
            'epoch': epoch + 1,
            'feature_state': feature_model.state_dict(),
            'optimizer': feature_optimizer.state_dict(),
            'loss': train_loss,
            'max_accu': best_accu
        }
        state['bert_state'] = Bert_Model.state_dict()
        state['bert_optimizer'] = bert_optimizer.state_dict()
        save_checkpoint(state, is_best1, checkpoint=result_path)
        if is_best2:
            save_checkpoint(state, False, checkpoint=result_path,filename='Bbest_model.pth.tar')
