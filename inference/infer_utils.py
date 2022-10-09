import os
from ops.os_operation import  mkdir
import datetime
import time

def init_log_path(params):
    save_path = os.path.join(os.getcwd(), 'E2E_CoSeg')
    mkdir(save_path)
    dataset_name = params['dataset']
    save_path = os.path.join(save_path, dataset_name)
    mkdir(save_path)
    save_path=os.path.join(save_path,"window_length_"+str(params['window_length']))
    mkdir(save_path)
    save_path = os.path.join(save_path, "mask_length_" + str(params['mask_length']))
    mkdir(save_path)
    return save_path

