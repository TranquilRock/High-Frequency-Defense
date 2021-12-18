import torch
import os
from torch.utils.data import Dataset

# https://github.com/MadryLab/constructed-datasets
def fetch_data(data_path="d_robust_CIFAR", foldername="release_datasets/"):
    data_path = foldername + data_path
    return torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims"))), torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))


class RobustDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        assert imgs.size(0) == labels.size(0)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[index]

    def __len__(self):
        return self.imgs.size(0)
