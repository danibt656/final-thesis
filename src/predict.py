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


if __name__ == '__main__':
  # Reconstruct model from saved weights
  model1 = torchvision.models.resnet50()
  model1.fc = torch.nn.Sequential(
      torch.nn.Linear(
          in_features=2048,
          out_features=1
      ),
      torch.nn.Sigmoid()
  )
  model1.load_state_dict(torch.load("resnet50.pth"))

  # Make predictions
  model1.eval()
  if USE_GPU:
    model1 = model1.cuda()

  ids_all = []
  predictions = []

  for j, batch in enumerate(dataloaders['test']):
    X = batch["image"]
    print(f'Batch[{j}]')
#        ids = batch["id"]
    if USE_GPU:
      X = X.cuda()

#        for _id in ids:
#            ids_all.append(_id)

    with torch.set_grad_enabled(False):
      y_pred = model1(X)
      print(f'Predictions: {y_pred}')
      predictions.append((y_pred >= 0.5).float().cpu().numpy())

  print("Done making predictions!")
