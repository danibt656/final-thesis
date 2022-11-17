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
        super(ConvNet, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=96, kernel_size=(7,7), stride=(4,4), padding=0, bias=False) # + ReLU
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=True)
        self.drop1 = nn.Dropout(p=0.25)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(1,1), padding=0, bias=False) # + ReLU
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), ceil_mode=True)
        self.drop2 = nn.Dropout(p=0.25)
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=0, bias=False) # + ReLU
        self.bn3 = nn.BatchNorm2d(num_features=384)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), ceil_mode=True)
        self.drop3 = nn.Dropout(p=0.25)

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=0, bias=False) # + ReLU
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.pool4 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), ceil_mode=True)
        self.drop4 = nn.Dropout(p=0.25)

        # Fully connected layer 1
        self.fc1 = nn.Linear(in_features=50176, out_features=1024) # + ReLU
        self.bn5 = nn.BatchNorm1d(num_features=1024)
        self.drop5 = nn.Dropout(p=0.5)

        # Fully connected layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=512) # + ReLU
        self.fc3 = nn.Linear(in_features=512, out_features=out_f) # + ReLU
        self.softmax = nn.Softmax(dim=1) # output


    def forward(self, x):
        # print(f'SHAPE-0: {x.shape}')
        # Layer 1
        x = relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        # print(f'SHAPE-1: {x.shape}')
        # Layer 2
        x = relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        # print(f'SHAPE-2: {x.shape}')
        # Layer 3
        x = relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        # print(f'SHAPE-3: {x.shape}')
        # Layer 4
        x = relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.drop4(x)
        # print(f'SHAPE-4: {x.shape}')
        # Fully connected layer 1
        x = x.view(x.size(0), -1) # flatten for FCL
        # print(f'SHAPE-flat: {x.shape}')
        x = relu(self.fc1(x))
        x = self.bn5(x)
        x = self.drop5(x)
        # print(f'SHAPE-FC1: {x.shape}')
        # Fully connected layer 2
        x = relu(self.fc2(x))
        # print(f'SHAPE-FC2: {x.shape}')
        x = relu(self.fc3(x))
        # print(f'SHAPE-FC3: {x.shape}')
        out = self.softmax(x)
        
        return out