import numpy as np
import pandas as pd
import os
import copy
import glob
import matplotlib.pyplot as plt

import torch
import torchvision


class ResNetModel():

  def __init__(self, n_epochs=3, use_gpu=True):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Sequential(
      torch.nn.Linear(
        in_features=2048,
        out_features=1  # Output is only 1 label
      ),
      torch.nn.Softmax(dim=1)
    )
    if use_gpu:
      model = model.cuda()

    self.model = model
    self.use_gpu = use_gpu
    self.n_epochs = n_epochs
    self.optimizer = torch.optim.Adam(model.parameters())
    self.criterion = torch.nn.BCELoss()

  
  def fit(self, dataloaders):
    """
    Entrena el modelo con los datos dados

    Args:
      dataloaders: Dict con los dataloaders de Train y Test
    """
    for i in range(self.n_epochs):
      # Entrenar el modelo
      self.model.train()

      samples = 0
      loss_sum = 0
      correct_sum = 0
      for j, batch in enumerate(dataloaders['train']):
        X = batch["image"]
        labels = batch["label"]
        if self.use_gpu:
          X = X.cuda()
          labels = labels.cuda()

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          y = self.model(X)
          loss = self.criterion(y,labels.view(-1, 1).float())
          loss.backward()
          self.optimizer.step()

          # Multiplicar por tam de Batch porque loss es la loss media del Batch
          loss_sum += loss.item() * X.shape[0]
          samples += X.shape[0]
          num_corrects = torch.sum((y >= 0.5).float() == labels.view(-1, 1).float())
          correct_sum += num_corrects

          # Print batch statistics every 50 batches
          # if j % 50 == 49 and phase == "train":
          print("E{}:B{} - loss: {}, acc: {}".format(
              i + 1,
              j + 1,
              float(loss_sum) / float(samples),
              float(correct_sum) / float(samples)
          ))

      # Estadisticas de la epoca
      epoch_acc = float(correct_sum) / float(samples)
      epoch_loss = float(loss_sum) / float(samples)
      print(f"Epoch: {i+1} Train loss: {epoch_loss}, Train acc: {epoch_acc}")

      # Guardar el modelo cada vez que mejora el accuracy
      if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(best_model_wts, "resnet50.pth")
  
  
  def predict(self, dataloaders, state_dict_file="resnet50.pth"):
    """
    Realiza predicciones en base al modelo entrenado previamente

    Args:
      dataloaders: Dict con los dataloaders de Train y Test
      state_dict_file: Fichero con los pesos del modelo entrenado (guardados en fit)
    """
    self.model.load_state_dict(torch.load(state_dict_file))

    # Hacer predicciones
    self.model.eval()

    predictions = []

    for j, batch in enumerate(dataloaders['test']):
      X = batch["image"]
      print(f'Batch[{j}]')
      if self.use_gpu:
        X = X.cuda()

      with torch.set_grad_enabled(False):
        y_pred = self.model(X)
        print(f'=> Predictions: {y_pred}')
        predictions.append(y_pred.cpu().numpy())