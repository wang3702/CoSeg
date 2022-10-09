
import numpy as np
import random
import os
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
class Path_Mapper(Dataset):
    def __init__(self, dataset_dir,transform=None):

        listfiles=os.listdir(dataset_dir)
        self.segment_list=[]
        Refer_Video_Dict=defaultdict(list)
        Index_T_Dict=defaultdict(list)
        count=0
        for k,item in enumerate(listfiles):
            tmp_dir=os.path.join(dataset_dir,item)
            listfile_dir=[os.path.join(tmp_dir,x) for x in os.listdir(tmp_dir) if "jpg" in x]
            listfile_dir = sorted(listfile_dir, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
            listfile_dir=list(listfile_dir)
            video_length=len(listfile_dir)
            for frame_idx,frame_path in enumerate(listfile_dir):
                self.segment_list.append(frame_path)
                Refer_Video_Dict[item].append(count)
                Index_T_Dict[count]=frame_idx/video_length
                count+=1
            #self.segment_list.append(list(listfile_dir))
        self.Refer_Video_Dict=Refer_Video_Dict
        self.transform=transform
        self.Index_Time_Dict=Index_T_Dict
        self.total_frame=len(self.Index_Time_Dict)
    def __getitem__(self, index):
        choose_frame=self.segment_list[index]
        img = Image.open(choose_frame)
        img=img.convert('RGB')
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
            #img = np.asarray(img, dtype="int32") / 255

        return img,index
    def __len__(self):
        return len(self.segment_list)
