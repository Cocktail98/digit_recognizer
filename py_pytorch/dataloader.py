import torch
from torch.utils.data import Dataset
import numpy as np


class DigitDataset(Dataset):
    def __init__(self, df, transform=None):
        # self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        labels = df.label
        labels = np.array(labels)
        images = df.loc[:, 'pixel0':]
        images = np.array(images)
        images = images.astype('float').reshape(images.shape[0], 28, 28) / 255.0
        self.images = torch.from_numpy(images).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        # print(label)
        img = self.images[index]
        sample = {'image': img, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


class DigitTestDataset(Dataset):
    def __init__(self, df, transform=None):
        # self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        images = df.loc[:, 'pixel0':]
        images = np.array(images)
        images = images.astype('float').reshape(images.shape[0], 28, 28) / 255.0
        self.images = torch.from_numpy(images).unsqueeze(1).float()

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        # print(label)
        img = self.images[index]
        sample = {'image': img}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
