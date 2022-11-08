import numpy as np
import pandas as pd
import os
import copy
import glob
import matplotlib.pyplot as plt

import torch
import torchvision

from dataloader import dataloaders

USE_GPU = True
EPOCHS = 5

if __name__ == '__main__':
  model = torchvision.models.resnet50()
  # model = cnnModel()

  # Replace final fully connected layer to suite problem
  model.fc = torch.nn.Sequential(
      torch.nn.Linear(
          in_features=2048,
          out_features=1
      ),
      torch.nn.Softmax(dim=1)
  )

  # Model training
  if USE_GPU:
    model = model.cuda()  # Should be called before instantiating optimizer

  optimizer = torch.optim.Adam(model.parameters())
  criterion = torch.nn.BCELoss()  # For binary classification problem

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for i in range(EPOCHS):
    for phase in ["train", "test"]:
      if phase == "train":
        model.train()
      else:
        model.eval()

      samples = 0
      loss_sum = 0
      correct_sum = 0
      for j, batch in enumerate(dataloaders[phase]):
        X = batch["image"]
        genders = batch["gender"]
        if USE_GPU:
          X = X.cuda()
          genders = genders.cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          y = model(X)
          loss = criterion(
              y,
              genders.view(-1, 1).float()
          )

          if phase == "train":
            loss.backward()
            optimizer.step()

          # We need to multiple by batch size as loss is the mean loss of the samples in the batch
          loss_sum += loss.item() * X.shape[0]
          samples += X.shape[0]
          num_corrects = torch.sum(
            (y >= 0.5).float() == genders.view(-1, 1).float())
          correct_sum += num_corrects

          # Print batch statistics every 50 batches
          # if j % 50 == 49 and phase == "train":
          print("E{}:B{} - loss: {}, acc: {}".format(
              i + 1,
              j + 1,
              float(loss_sum) / float(samples),
              float(correct_sum) / float(samples)
          ))

      # Print epoch statistics
      epoch_acc = float(correct_sum) / float(samples)
      epoch_loss = float(loss_sum) / float(samples)
      print("Epoch: {} - {} loss: {}, {} acc: {}".format(i +
            1, phase, epoch_loss, phase, epoch_acc))

      # Deep copy the model
      if phase == "test" and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "resnet50.pth")
