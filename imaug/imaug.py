# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torchvision.utils as vutils
import numpy as np
from typing import List, Union, Tuple



# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class IMaug(object):
    def __init__(self, datapath:str) -> None:
        super(IMaug, self).__init__()
        
        assert datapath is not None or datapath != '', "A datapath must be provided"
        self.datapath = datapath

        self.rseed = self._random_seed()
        self._load_dataset(self.datapath)

    def _random_seed(self) -> None:
        manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        return manualSeed

    def _load_dataset(self, datapath: str) -> None:
        """Loads a dataset and sets useful parameters

        Parameters
        ----------
        datapath : str
            Path to the image dataset (as ImageFolder requires)
        """
        print('Loading dataset...', end='')

        data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.image_dataset = datasets.ImageFolder(datapath, data_transforms)
        self.dataset_size = len(self.image_dataset)
        self.class_to_idx = self.image_dataset.class_to_idx
        self.idx_to_class = dict((v,k) for k,v in self.class_to_idx.items())
        C,W,H = self.image_dataset.__getitem__(0)[0].shape # (channels, width, height)

        # (n, idxs_n) for nth class
        self.idx_for_classes = dict(
            (i, torch.nonzero(torch.Tensor(self.image_dataset.targets)==i).flatten())
            for i in self.idx_to_class.keys()
        )
        # (n, size_n) for nth class
        self.class_sizes = dict((i, len(self.idx_for_classes[i])) for i in self.idx_to_class.keys())
        max_class, min_class = max(self.class_sizes, key=self.class_sizes.get),\
                               min(self.class_sizes, key=self.class_sizes.get)
        max_size, min_size = self.class_sizes[max_class], self.class_sizes[min_class]
        balance_deg = max_size / min_size
        print('OK')

        imbalance_threshold = self.dataset_size - self.dataset_size + 5
        if np.absolute(max_size - min_size) <= imbalance_threshold:
            print('Classes are already balanced')
            return

        print(f'\nFound the following imbalance between classes:')
        print(f'    Class \'{self.idx_to_class[max_class]}\' is the majority with {max_size} samples')
        print(f'    Class \'{self.idx_to_class[min_class]}\' is the minority with {min_size} samples')
        print(f'\n    The resulting balance degree is {balance_deg}')
        proceed = input('== Proceed with balance? (y/n): ')
        
        if proceed.lower() != 'y':
            print('\n\nBalance operation cancelled by user...\n')
            return

        print('\n\nProceeding with balancing dataset...\n')
        class_min_idxs = torch.nonzero(torch.Tensor(self.image_dataset.targets)==min_class).flatten()
        self.minority_set = Subset(self.image_dataset, class_min_idxs)
        # Create the dataloader
        min_dataloader = torch.utils.data.DataLoader(self.minority_set, batch_size=batch_size,
                                shuffle=True, num_workers=workers)
        return min_dataloader



if __name__ == '__main__':
    PATH = '../data/UTKFace/train'

    imaug = IMaug(PATH)