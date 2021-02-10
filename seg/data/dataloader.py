#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 09
# @Author  : Luo jin
# @User    : 22403 
# @File    : dataloader.py
# -----------------------------------------
#
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class Loader(Dataset):
    def __init__(self, DataPath):
        self.DataPath = DataPath
        self.dataList = []
        self.length =self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)

        self.Trans = ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path, label_path=self.dataList[index]
        image = np.array(Image.open(image_path),dtype=np.float32)
        image=image/255.0 #norm to 0 -1
        image=self.Trans(image)

        label_np_array = np.array(Image.open(label_path), dtype=np.int64) - 1
        label_index = torch.from_numpy(np.expand_dims(label_np_array, 0))
        label = torch.zeros((10,256, 256))
        label.scatter_(0, label_index, 1).float()

        return image,label

    def _readTXT(self,txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                image_path, label_path = line.strip().split('\t')
                self.dataList.append((image_path,label_path))
        random.shuffle(self.dataList)
        return len(self.dataList)






