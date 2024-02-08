from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
import random
from IPython import embed

class CarlaFPVBEVTCN(BaseDataset):
    #def __init__(self, root_dir, transform=None):
    #    super().__init__(root_dir, transform, action=False, reward=False, terminal=False, goal=False)
    
    def __init__(self, root_dir, pos_distance, transform=None, value=False, episode=True, goal=False, use_lstm=False, truncated=True):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal, use_lstm=use_lstm, truncated=truncated)

        self.pos_distance = pos_distance
        assert (self.pos_distance <= 12)
        print("pos_distance", self.pos_distance)

        #self.sample_next = sample_next



    def __getitem__(self, item):

        img, value, episode = [], [], []
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = item
        else:
            im_ind = item - self.each_len[file_ind-1]

        left_pos = self.pos_distance
        right_pos = self.pos_distance
        #print(len(self.limit_nps[file_ind]), len(self.episode_nps[file_ind]), im_ind, len(self.id_dict))
        while (self.limit_nps[file_ind][im_ind] - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] < 3):
            if self.each_len[file_ind] - self.each_len[file_ind-1] > 0:
                im_ind = random.randint(0, self.each_len[file_ind] - self.each_len[file_ind-1])
            else:
                im_ind = 0



        if self.pos_distance > 0:
            
            #set the right boundary
            if im_ind + right_pos > self.limit_nps[file_ind][im_ind] - 2:
                right_pos = max(self.limit_nps[file_ind][im_ind] - im_ind - 1, 0)

            #set the left boundary
            if im_ind - left_pos < self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] + 2:
                left_pos = max(im_ind - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] - 1, 0)
            

            assert(im_ind + right_pos <= self.limit_nps[file_ind][im_ind])
            assert(im_ind - left_pos >= self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]])


            assert(right_pos >= 0)
            assert(left_pos >= 0)
            assert(right_pos != 0 or left_pos != 0)

            #randomly sample in the right direction
            if left_pos == 0:
                posarr = [None, random.randint(im_ind+1, im_ind+right_pos)]
                ind = 1

            #randomly sample in the left direction
            elif right_pos == 0:
                posarr = [random.randint(im_ind-left_pos, im_ind-1), None]
                ind = 0
            
            else:
                posarr = [random.randint(im_ind-left_pos, im_ind-1), random.randint(im_ind+1, im_ind+right_pos)]
                ind = random.randint(0, 1)

        #if pos_distance is -1 then sample from either side to the limit
        else:
            if abs(im_ind - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]) < 2:
                posarr = [None, random.randint(im_ind+1, self.limit_nps[file_ind][im_ind]-1)]
                ind = 1

            elif abs(im_ind - self.limit_nps[file_ind][im_ind]) < 2:
                posarr = [random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-1), None]
                ind = 0

            else:
                posarr = [random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-1), random.randint(im_ind+1, self.limit_nps[file_ind][im_ind]-1)]
                ind = random.randint(0, 1)


        posind = posarr[ind]
        
        if ind == 0:
            assert(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] < posind)
            negind = random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]], posind-1)
        else:
            assert(posind < self.limit_nps[file_ind][im_ind])
            negind = random.randint(posind+1, self.limit_nps[file_ind][im_ind])


        #find d
        #+d, -d from the im_ind -> sample positive pair 
        #sample smth either from (start, -d) or (d, end)

        assert (abs(negind-im_ind) > abs(posind - im_ind))
        #sample anchor, positive and negative

        fpv = [np.moveaxis(self.obs_nps[file_ind][im_ind].astype(np.float32), -1, 0), np.moveaxis(self.obs_nps[file_ind][posind].astype(np.float32), -1, 0), np.moveaxis(self.obs_nps[file_ind][negind].astype(np.float32), -1, 0)]
        bev = [np.expand_dims(self.bev_nps[file_ind][im_ind][:, :, 0].astype(np.float32), axis=0), np.expand_dims(self.bev_nps[file_ind][posind][:, :, 0].astype(np.float32), axis=0), np.expand_dims(self.bev_nps[file_ind][negind][:, :, 0].astype(np.float32), axis=0)]

        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)


        
        return np.stack(fpv, axis=0), np.stack(bev, axis=0)
