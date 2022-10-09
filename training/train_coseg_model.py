
import time
from training.train_utils import AverageMeter,Calculate_top_accuracy
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn
import torch
import gc
import random

def train_coseg_model(epoch, feature_model, bert_model,
                        feature_optimizer,bert_optimizer,
                        dataloader, params, writer,
                        train_logger, train_batch_logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Loss = AverageMeter()
    Closs1 = AverageMeter()
    Closs2 = AverageMeter()
    end_time = time.time()
    CTOP1 = AverageMeter()
    BTOP1 = AverageMeter()
    feature_model.train()
    bert_model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    mse_criterion = nn.MSELoss().cuda()
    mae_criterion = nn.L1Loss().cuda()
    cos_criterion = nn.CosineEmbeddingLoss().cuda()
    study_length = params['window_length']
    mask_length = params['mask_length']
    iteration = len(dataloader)
    kl_criterion=torch.nn.KLDivLoss().cuda()

    for batch_idx, read_data in enumerate(dataloader):
        video_segment_data=read_data
        batch_segment = video_segment_data.size(0)
        Qimage_List=[]#build temporal pairs based on checking
        Kimage_List = []
        for j in range(batch_segment):
            for k in range(study_length):
                choose_frame=video_segment_data[j,k]
                random_index=random.randint(0,study_length-1)
                while random_index==k:
                    random_index = random.randint(0, study_length - 1)
                neighbor_frame=video_segment_data[j,random_index]
                Qimage_List.append(neighbor_frame)
                Kimage_List.append(choose_frame)

        Qimage_List = torch.stack(Qimage_List, dim=0)
        Kimage_List = torch.stack(Kimage_List, dim=0)
        data_time.update(time.time() - end_time)
        if batch_idx == 0 and epoch % 10 == 0:
            print("Qimage size",Qimage_List.size())
            if writer is not None:
                tmp1 = vutils.make_grid(Qimage_List, normalize=True, scale_each=True)
                writer.add_image('Qimage', tmp1, epoch)
                tmp1 = vutils.make_grid(Kimage_List, normalize=True, scale_each=True)
                writer.add_image('Kimage', tmp1, epoch)
        Qimage_List = Qimage_List.cuda(non_blocking=True)
        Kimage_List = Kimage_List.cuda(non_blocking=True)
        K_output,logits, labels = feature_model(Qimage_List,Kimage_List,type=1)
        #contrastive loss applied here
        loss = criterion(logits, labels)
        accu,accu5 = Calculate_top_accuracy(logits, labels, topk=(1, 5))
        Closs1.update(loss.item(),batch_segment*study_length)
        CTOP1.update(accu.item(),batch_segment*study_length)
        #Masked reconstruction model
        mask = torch.ones([batch_segment, study_length])
        Mask_start = []
        for k in range(batch_segment):
            rand_start = random.randint(0, study_length- mask_length)
            mask[rand_start:rand_start+mask_length] = 0
            Mask_start.append(rand_start)
        mask = mask.float().cuda()
        video_feature_data=[]
        for j in range(batch_segment):
            start_point=j*study_length
            end_point=(j+1)*study_length
            video_feature_data.append(K_output[start_point:end_point])
        video_feature_data = torch.stack(video_feature_data,dim=0)
        if batch_idx == 0 and epoch == 0:
            print("Video Feature input:",video_feature_data.size())
        output = bert_model(video_feature_data, mask)
        video_feature_data = nn.functional.normalize(video_feature_data, dim=2)
        output = nn.functional.normalize(output, dim=2)

        #MSE loss for reconstruction
        Query_Frame = []
        Key_Frame = []
        for k in range(batch_segment):
            rand_start = Mask_start[k]
            for j in range(mask_length):
                if j+rand_start<study_length:
                    Query_Frame.append(output[k, rand_start+j])
                    Key_Frame.append(video_feature_data[k, rand_start+j])
        Query_Frame = torch.stack(Query_Frame, dim=0)
        Key_Frame = torch.stack(Key_Frame, dim=0)
        loss2 = mse_criterion(Query_Frame,Key_Frame)

        accu, accu5 = Calculate_top_accuracy(logits, labels, topk=(1, 5))
        Closs2.update(loss2.item(), batch_segment * study_length)
        BTOP1.update(accu.item(), batch_segment * study_length)

        loss = loss + params['beta']*loss2
        Loss.update(loss.item(),batch_segment * study_length)
        #update
        feature_optimizer.zero_grad()

        bert_optimizer.zero_grad()
        loss.backward()
        feature_optimizer.step()
        bert_optimizer.step()
        batch_time.update(time.time() - end_time)
        print_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    ' Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    ' Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    ' CLoss1 {Closs.val:.4f} ({Closs.avg:.4f})\t' \
                    ' CLoss2 {Sloss.val:.4f} ({Sloss.avg:.4f})\t' \
                    ' CTop1 {top1.val:.4f} ({top1.avg:.4f})\t' \
                    ' BTop1 {btop1.val:.4f} ({btop1.avg:.4f})\t' \
            .format(
            epoch,
            batch_idx + 1,
            iteration,
            batch_time=batch_time, Closs=Closs1,
            data_time=data_time, loss=Loss, Sloss=Closs2, top1=CTOP1,btop1=BTOP1,
        )
        print(print_str)
        #'epoch', 'iter', 'batch', 'loss', 'closs1', 'closs2', 'Ctop1', 'Btop1', 'lr1', 'lr2'
        batch_record_dict = {
            'epoch': epoch,
            'batch': batch_idx + 1,
            'iter': (epoch) * iteration + (batch_idx + 1),
            'loss': Loss.val,
            'lr1': feature_optimizer.param_groups[0]['lr'],
            'lr2': bert_optimizer.param_groups[0]['lr'],
            'closs1': Closs1.val,
            'closs2': Closs2.val,
            'Ctop1': CTOP1.val,
            'Btop1': BTOP1.val,
        }
        train_batch_logger.log(batch_record_dict)

        end_time = time.time()
    #'epoch', 'loss','closs1','closs2', 'Ctop1','Btop1','lr1','lr2'
    epoch_record_dict = {
        'epoch': epoch,
        'loss': Loss.avg,
        'lr1': feature_optimizer.param_groups[0]['lr'],
        'lr2': bert_optimizer.param_groups[0]['lr'],
        'closs1': Closs1.avg,
        'closs2': Closs2.avg,
        'Ctop1': CTOP1.avg,
        'Btop1': BTOP1.avg,
    }
    train_logger.log(epoch_record_dict)
    if writer is not None:
        writer.add_scalars('Data/Loss', {'train_loss': Loss.avg}, epoch)
        writer.add_scalars('Data/CLoss1', {'train_loss': Closs1.avg}, epoch)
        writer.add_scalars('Data/CLoss2', {'train_loss': Closs2.avg}, epoch)
        writer.add_scalar('LR/LR1', feature_optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/LR2', bert_optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('Accu/Btop1', {'train': BTOP1.avg}, epoch)
        writer.add_scalars('Accu/Ctop1', {'train': CTOP1.avg}, epoch)
        if epoch % 10 == 0:
            for name, param in feature_model.named_parameters():
                writer.add_histogram('Feature_' + name, param.clone().cpu().data.numpy(), epoch)
            for name, param in bert_model.named_parameters():
                writer.add_histogram("Bert_" + name, param.clone().cpu().data.numpy(), epoch)
    gc.collect()  # release unreferenced memory
    return Loss.avg, CTOP1.avg,BTOP1.avg









