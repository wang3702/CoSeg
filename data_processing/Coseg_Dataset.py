

#/*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/

import torch
import torch.utils.data as data
import numpy as np
import random
import os
from PIL import Image
from torch.utils.data import Dataset

#batch_size*window_size
#prepare snippets of input frames for each video segment.
class Coseg_Dataset(Dataset):
    def __init__(self, dataset_dir,params,transform,study_length):
        listfiles=os.listdir(dataset_dir)
        self.segment_list=[]
        self.end_point=set()
        count=0
        for k,item in enumerate(listfiles):
            tmp_dir=os.path.join(dataset_dir,item)
            listfile_dir=[os.path.join(tmp_dir,x) for x in os.listdir(tmp_dir) if "jpg" in x]
            if len(listfile_dir)<=study_length:
                continue
            listfile_dir = sorted(listfile_dir, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
            self.segment_list.append(listfile_dir)
            current_video_length=len(listfile_dir)
            count+=current_video_length
            self.end_point.add(count)#ending point for filtering possible consecutive list
            print("Currently the total frame size %d"%count)
        print("In total we have %d videos end points"%len(self.end_point))
        self.params=params
        #self.study_length=self.params['study_length']
        self.transform=transform
        self.study_length=study_length
        self.total_length=count
        print("iterative times %d" % int(self.total_length/float(len(self.end_point))))


    def __getitem__(self, index):
        choose_segment=random.randint(0,len(self.segment_list)-1)
        current_frame_list = self.segment_list[choose_segment]
        start_frame = random.randint(0,len(current_frame_list)-self.study_length)
        video_list = []
        for k in range(self.study_length):
            img = Image.open(current_frame_list[k + start_frame])
            img = self.transform(img)
            video_list.append(img)
        video_list = torch.stack(video_list, dim=0)
        return video_list


    def __len__(self):
        return int(self.total_length/float(len(self.end_point)))
