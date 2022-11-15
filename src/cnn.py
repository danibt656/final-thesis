# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
"""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu


class ConvNet(nn.Module):
    def __init__(self, in_ch=3, out_f=1):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=96, kernel_size=(7,7), stride=(4,4), padding=0, bias=False) # + ReLU
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.drop1 = nn.Dropout(p=0.25)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), padding=0, bias=False) # + ReLU
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3))
        self.drop2 = nn.Dropout(p=0.25)
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), padding=0, bias=False) # + ReLU
        self.bn3 = nn.BatchNorm2d(num_features=384)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3))
        self.drop3 = nn.Dropout(p=0.25)

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), padding=0, bias=False) # + ReLU
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.pool4 = nn.MaxPool2d(kernel_size=(3,3))
        self.drop4 = nn.Dropout(p=0.25)

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=512, out_features=512) # + ReLU
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.drop5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=512, out_features=512) # + ReLU
        self.fc1 = nn.Linear(in_features=512, out_features=out_f) # + ReLU
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        print(f'SHAPE-0: {x.shape}')
        # Layer 1
        x = relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        print(f'SHAPE-1: {x.shape}')
        # Layer 2
        x = relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        print(f'SHAPE-2: {x.shape}')
        # Layer 3
        x = relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        print(f'SHAPE-3: {x.shape}')
        # Layer 4
        x = relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.drop4(x)
        print(f'SHAPE-4 {x.shape}')
        # Fully connected layer
        x = relu(self.fc1(x))
        x = self.bn5(x)
        x = self.drop5(x)
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        return self.softmax(x)