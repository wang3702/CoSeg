
import torch.utils.data as data
import numpy as np




class Feature_Dataset_Mask(data.Dataset):
    def __init__(self, Feature_Matrix, Refer_Dict,input_length, pred_length):
        super(Feature_Dataset_Mask, self).__init__()
        self.Feature_Matrix=Feature_Matrix
        self.Refer_Dict=Refer_Dict
        self.search_list=list(Refer_Dict.keys())
        self.input_length=input_length
        self.pred_length=pred_length

        self.possible_list=[]
        self.mask_list=[]
        common_mask=np.ones(input_length)
        common_mask[0:pred_length]=0
        count=0
        self.Search_Video_Dict={}
        self.Search_Index_Dict={}
        self.Start_List=[]
        self.Start_Point={}
        for key in self.search_list:
            current_frame_list=self.Refer_Dict[key]
            self.Start_Point[key]=count
            end_index=0
            for k in range(0,len(current_frame_list)-self.input_length,self.pred_length):
                self.possible_list.append(current_frame_list[k])
                self.mask_list.append(common_mask)
                self.Search_Video_Dict[count]=key
                self.Search_Index_Dict[count]=k
                self.Start_List.append(0)
                end_index=k
                count+=1
            for k in range(end_index,len(current_frame_list),self.pred_length):
                current_mask=np.ones(input_length)
                start_index=k-end_index
                current_mask[start_index:start_index+self.pred_length]=0
                self.Start_List.append(start_index)
                self.possible_list.append(current_frame_list[end_index])
                self.mask_list.append(current_mask)
                self.Search_Video_Dict[count] = key
                self.Search_Index_Dict[count] = k
                count+=1
    def __len__(self):
        return len(self.possible_list)

    def __getitem__(self, index):
        current_index=self.possible_list[index]
        use_index=current_index+self.input_length
        video_input=self.Feature_Matrix[current_index:use_index]
        mask_input=self.mask_list[index]
        mask_start=self.Start_List[index]
        return video_input,mask_input,mask_start
