# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import copy
import glob
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from skimage.io import imread
from sklearn.model_selection import train_test_split


BATCH_SIZE = 64


dataset_folder_name = '../data/UTKFace/Images'

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 200

dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())


def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It iterates over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()

    return df


class Dataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the images, perform transforms on them,
    and load their corresponding labels.
    """
    
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.transform = transform
        
    def __getitem__(self, idx):
        #print(f'#{idx}...', end='')
        if idx >= self.df.shape[0]:
            idx = self.df.shape[0]-1
        img_path = self.df.iloc[idx]['file']
#         print("img_path:", img_path)
        #print('OK')
        img = imread(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        sample = {
            "image": img,
        }
        sample["gender"] = dataset_dict['gender_alias'][self.df.iloc[idx]["gender"]]
#        sample["id"] = self.df.loc[idx, "id"]
        return sample
    
    def __len__(self):
        try:
            return self.df.shape[0]
        except AttributeError:
            return len(self.images)

#################################################################################################

df = parse_dataset(dataset_folder_name)

train_indices, test_indices = train_test_split(df.index, test_size=0.25)

transform_pipe = transforms.Compose([
    transforms.ToPILImage(), # Convert np array to PILImage

    # Resize image to 224 x 224 as required by most vision models
    transforms.Resize(
        size=(224, 224)
    ),

    # Convert PIL image to tensor with image values in [0, 1]
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = Dataset(
    df=df,
    img_dir="../data/UTKFace/Images/",
    transform=transform_pipe
)

# The training dataset loader will randomly sample from the train samples
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=torch.utils.data.SubsetRandomSampler(
        train_indices
    )
#     shuffle=True,
#     num_workers=8
)

# The testing dataset loader will randomly sample from the test samples
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    sampler=torch.utils.data.SubsetRandomSampler(
        test_indices
    )
#     shuffle=True,
#     num_workers=8
)

dataloaders = {
    "train": train_loader,
    "test": test_loader
}
