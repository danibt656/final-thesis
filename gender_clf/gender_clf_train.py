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

cf = Classifier().to(device=device)

################################################################
#       Train network (& save weigths)
################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cf.parameters(), lr=0.001, momentum=0.9)

NUM_EPOCHS = 40

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # model is in GPU, thus we send data also to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[Epoch: {epoch + 1}] Loss: {running_loss / 2000}')
            running_loss = 0.0

print('Finished Training')

# Save model
PATH = './models/gender_test.pth'
torch.save(cf.state_dict(), PATH)