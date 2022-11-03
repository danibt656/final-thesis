import numpy as np 
import pandas as pd
from math import floor

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim

################################################################
#       Prepare data
################################################################
DATA_PATH = '../data/age_gender.csv'
data = pd.read_csv(DATA_PATH)

data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype = 'float32'))
data['pixels'] = data['pixels'].apply(lambda x:x/255)

gender_dist = data['gender'].value_counts().rename(index={0:'Male', 1:'Female'})

X = np.array(data['pixels'].tolist())
X = X.reshape(len(data), 48, 48) # reshape each array of length 48x48 into matrix
y = data['gender'].tolist()

train_ratio = 0.7 # 70% train, 30% test
slice = int(floor(len(X) * train_ratio))
batch_size = 64

class DataSetLoader(Dataset):
    def __init__(self, x, y):       
        self.x = x
        self.y = y
        
        self.x_train = torch.tensor(self.x, dtype=torch.float32)
        self.y_train = torch.tensor(self.y)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        img = self.x_train[idx]
        img = img.unsqueeze(0)
        
        return img, self.y_train[idx]
    
trainset = DataSetLoader(X[:slice], y[:slice])
testset = DataSetLoader(X[slice:], y[slice:])

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

classes = ('male', 'female')

################################################################
#       Create network
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6400, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = x.reshape(x.shape[0], -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x    

# Load model
MODEL_PATH = '../notebooks/models/gender_test.pth'
cf = Classifier().to(device=device)
cf.load_state_dict(torch.load(MODEL_PATH))

# 1. How the network performs on the whole dataset

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = cf(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Global Accuracy: {100 * correct // total} %')


# 2. what are the classes that performed well, and the classes that did not perform well?

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = cf(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')