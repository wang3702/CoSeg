import os
import torch.nn as nn
import torch
from training.train_utils import AverageMeter
import time


def gen_coseg_mse(mse_path, dataloader, Bert_Model,
                     default_batch_size, n_channels,params,Video_Refer_Dict,mask_length):
    Bert_Model.eval()
    mse_loss = nn.MSELoss()
    mse_loss = mse_loss.cuda()
    data_time = AverageMeter()
    batch_time=AverageMeter()
    Loss=AverageMeter()
    Ploss = AverageMeter()
    iteration = len(dataloader)
    video_refer_list=list(Video_Refer_Dict.keys())
    for batch_idx,video_data in enumerate(dataloader):
        end_time=time.time()
        video_data,mask_input,mask_start_list= video_data
        batch_size=video_data.size(0)
        seq_len=video_data.size(1)
        video_data = video_data.float().cuda()
        mask_input=mask_input.float().cuda()
        video_data = nn.functional.normalize(video_data, dim=2)
        output = Bert_Model(video_data, mask_input)
        # output shape:  ( batch, seq_len,num_directions * hidden_size)
        output=nn.functional.normalize(output, dim=2)

        record_detail_loss=[]

        loss = 0
        for k in range(batch_size):
            # tmp_mask=np.array(mask_input[k])
            # if batch_idx==0 and k==0:
            #    print (tmp_mask)
            mask_start_point = mask_start_list[k]
            loss += torch.mean(torch.mean((output[k, mask_start_point:mask_start_point + mask_length,:]
                         - video_data[k, mask_start_point:mask_start_point + mask_length,:]) ** 2, dim=1), dim=0)

            tmp_error = torch.mean((output[k, mask_start_point:mask_start_point + mask_length, :] -
                                    video_data[k,mask_start_point:mask_start_point + mask_length,:]) ** 2,
                                   dim=1)
            record_detail_loss.append(tmp_error.detach().cpu().numpy())
        loss /= batch_size
        Loss.update(loss.item(),batch_size)

        batch_time.update(time.time() - end_time)
        print_str = 'Epoch: [{0}/{1}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    ' Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    ' Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                    'Prev MSE {loss0.val:.6f} ({loss0.avg:.6f})\t' \
            .format(
            batch_idx + 1,
            iteration,
            batch_time=batch_time,
            data_time=data_time, loss=Loss, loss0=Ploss,
        )
        print(print_str)
        end_idx = min((batch_idx + 1) * default_batch_size, len(dataloader.dataset))
        for search_index in range(batch_idx*default_batch_size,end_idx):

            video_name=dataloader.dataset.Search_Video_Dict[search_index]
            tmp_mse_list=record_detail_loss[search_index-batch_idx*default_batch_size]
            current_index=dataloader.dataset.Search_Index_Dict[search_index]
            video_name=video_name.replace(".mp4","")
            video_name=video_name.replace(".avi","")
            tmp_path=os.path.join(mse_path,video_name+"_mse.txt")
            with open(tmp_path, 'a+') as file:
                for k, tmp_mse in enumerate(tmp_mse_list):
                    file.write('%d\t%.6f\n' % (current_index+k+1, tmp_mse))
                file.flush()
