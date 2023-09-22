
from torch.utils import data
import os
import torch
import numpy as np
from einops import rearrange


class Data(data.Dataset):
    def __init__(self, data_location, img_size=256):
        self.data_location = data_location
        with open(os.path.join(data_location, "siegfried_spatiotempo.txt")) as file:
            lines = file.readlines()
            samples_labelled = [line.rstrip() for line in lines if len(line.rstrip()) != 0]

        self.img_size = img_size
        self.samples_labelled = samples_labelled

    def __len__(self):
        return len(self.samples_labelled)

    def __getitem__(self, index):
        ID = self.samples_labelled[index]
        file_names = ID.split(',')[:-1] # remove empty name

        anno_ID_img = np.load(os.path.join(self.data_location, file_names[0]))['arr_0']
        anno_ID_img = np.transpose(anno_ID_img, (2, 0, 1))

        sheet_ID_imgs = [np.expand_dims(np.load(os.path.join(self.data_location, file_names[i]))['arr_0'][:, :, :3], 0) for i in range(1, len(file_names))]

        temporal_ID_imgs = sheet_ID_imgs[:-1]
        original_ID_img = sheet_ID_imgs[-1][:, self.img_size:self.img_size*2, self.img_size:self.img_size*2,...]
        
        temporal_ID_imgs = np.concatenate(temporal_ID_imgs, axis=0)
        temporal_ID_imgs = np.concatenate((original_ID_img, temporal_ID_imgs), axis=0)
        temporal_ID_imgs = np.transpose(temporal_ID_imgs, (0, 3, 1, 2))
        temporal_ID_imgs = temporal_ID_imgs

        spatial_ID_imgs = sheet_ID_imgs[-1][0] # Remember to index at 0 position because of the expand dimension
        spatial_ID_imgs = rearrange(spatial_ID_imgs, '(p1 h) (p2 w) c -> (p1 p2) h w c', p1=3, p2=3, h=self.img_size, w=self.img_size)
        spatial_ID_imgs = np.transpose(spatial_ID_imgs, (0, 3, 1, 2))

        # Image index (4 in center)
        # # # # #
        # 0 1 2 #
        # 3 4 5 #
        # 6 7 8 #
        # # # # #

        spatial_ID_imgs = torch.from_numpy(spatial_ID_imgs).float()
        temporal_ID_imgs = torch.from_numpy(temporal_ID_imgs).float()
        anno_ID_img = torch.from_numpy(anno_ID_img).float()

        return spatial_ID_imgs, temporal_ID_imgs, anno_ID_img