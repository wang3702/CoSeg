import os
from ops.os_operation import  mkdir
import datetime
import time
def init_log_path(params):
    learning_rate = params['lr']
    weight_decay = params['weight_decay']
    log_path= os.path.join(os.getcwd(), params['log_path'])
    mkdir(log_path)
    dataset_name=params['dataset']
    log_path=os.path.join(log_path,dataset_name)
    mkdir(log_path)
    log_path = os.path.join(log_path, 'CoSeg')
    mkdir(log_path)
    log_path = os.path.join(log_path, 'lr_' + str(learning_rate)+"_wd_"+str(weight_decay))
    mkdir(log_path)
    log_path=os.path.join(log_path,"window_length_"+str(params['window_length']))
    mkdir(log_path)
    log_path = os.path.join(log_path, "mask_length_" + str(params['mask_length']))
    mkdir(log_path)
    #to avoid duplicated saveing to overwrite previous model
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    log_path = os.path.join(log_path, formatted_today + now)
    mkdir(log_path)
    result_path = os.path.join(log_path,'model')
    mkdir(result_path)
    return log_path,result_path


def init_Logger(log_path):

    train_logger = Logger(
        os.path.join(log_path, 'train.log'),
        ['epoch', 'loss','closs1','closs2', 'Ctop1','Btop1','lr1','lr2'])

    train_batch_logger = Logger(
        os.path.join(log_path, 'train_batch.log'),
        ['epoch', 'iter','batch',  'loss', 'closs1','closs2','Ctop1','Btop1','lr1','lr2'])

    return train_logger,train_batch_logger


import os
import torch
import csv

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

import numpy as np
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, predicted = torch.max(outputs, 1)
    nb_corrects = float((predicted == targets).sum().data)
    batch_accuracy = nb_corrects / batch_size
    return batch_accuracy
def Calculate_top_accuracy(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_logger(log_path):
    train_logger = Logger(
        os.path.join(log_path, 'train.log'),
        ['epoch', 'loss', 'top1', 'top5', 'lr1'])
    train_batch_logger = Logger(
        os.path.join(log_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'top1', 'top5', 'lr1'])
    Val_logger = Logger(
        os.path.join(log_path, 'val.log'), ['epoch','Closs', 'top1', 'top5',])
    Test_logger = Logger(
        os.path.join(log_path, 'test.log'), ['epoch', 'Closs', 'top1', 'top5'])

    return train_logger, train_batch_logger, Val_logger, Test_logger

import os
import shutil
def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
