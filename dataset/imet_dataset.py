import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

import os


class IMetDataset(data.Dataset):
    def __init__(self, dataroot, datalist, transform=None):
        super(IMetDataset, self).__init__()
        self.dataroot = dataroot
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
        ]) if transform is None else transform

        # datalist: a list like [('img_name', [label1, label2, ...]), ...]
        self.datalist = datalist


    def __len__(self):
        return len(datalist)


    def __getitem__(self, idx):
        img_name, labels = self.datalist[idx]
        target = np.zeros(1102, dtype=np.float32)
        target[labels] = 1
        img = Image.open(os.path.join(self.dataroot, img_name)).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(target)

