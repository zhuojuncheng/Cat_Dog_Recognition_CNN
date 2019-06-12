import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils import data
from PIL import Image


class dataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.data_set = []
        self.data_set.extend(os.listdir(path))

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        label = torch.Tensor([int(self.data_set[index][0])])
        img_path = os.path.join(self.path, self.data_set[index])
        img = Image.open(img_path)
        img_convert = np.array(img).transpose(2, 0, 1)
        img_data = torch.Tensor(img_convert)/255 - 0.5
        return img_data, label


