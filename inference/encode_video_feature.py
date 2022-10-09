import torch.nn as nn
import torch
import gc
import time
from training.train_utils import AverageMeter,Calculate_top_accuracy
import numpy as np
def encode_video_feature(feature_model, loader, feature_size):
    data_time = AverageMeter()
    feature_model=feature_model.eval()
    Feature_matrix=np.zeros([len(loader.dataset),feature_size])
    iteration=len(loader)
    with torch.no_grad():
        for batch_idx, read_data in enumerate(loader):
            end_time = time.time()
            choose_frame, frame_index = read_data


            choose_frame = choose_frame.cuda(non_blocking=True)
            choose_frame = choose_frame.float()
            if batch_idx == 0:
                print(choose_frame[0])
                print(frame_index)
            # no last fc kept, only used the feature after pooling
            pred_feature = feature_model.module.encoder_q(choose_frame, use_feature=True)
            pred_feature = pred_feature.detach().cpu().numpy()
            frame_index = frame_index.detach().cpu().numpy()
            Feature_matrix[frame_index] = pred_feature
            data_time.update(time.time() - end_time)
            if batch_idx % 10 == 0:
                print_str = 'Batch: [{0}/{1}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    .format(
                    batch_idx + 1,
                    iteration,
                    batch_time=data_time
                )
                print(print_str)
    return Feature_matrix


