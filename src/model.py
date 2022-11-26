# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
"""
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision
from cnn import ConvNet


class ResNetModel():

  def __init__(self, in_ch=3, out_f=1, n_epochs=3, use_gpu=True):
    """Construct a ResNetModel
    Args:
      in_ch: Numero de canales de entrada (por defecto 3, RGB)
      out_f: Numero de features de salida
      n_epochs: Numero de epocas de entrenamiento
      use_gpu: Si usar gpu para ejecutar el modelo o no
    """
    model = torchvision.models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(
      in_channels=in_ch,
      out_channels=64,
      kernel_size=(7, 7),
      stride=(2, 2),
      padding=(3, 3),
      bias=False,
    )
    model.fc = torch.nn.Sequential(
      torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=out_f  # Output size
      ),
      torch.nn.Softmax(dim=1)
    )
    # model = ConvNet(in_ch, out_f)
    if use_gpu:
      model = model.cuda()

    self.model = model
    self.use_gpu = use_gpu
    self.n_epochs = n_epochs
    self.optimizer = torch.optim.SGD(
      model.parameters(), lr=0.001, momentum=0.9)
    self.criterion = torch.nn.CrossEntropyLoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    self.scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)

  def fit(self, dataloaders):
    """
    Entrena el modelo con los datos dados

    Args:
      dataloaders: Dict con los dataloaders de Train y Test

    Return:
      Estadisticas del proceso de entrenamiento
    """
    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_acc = 0.0
    batch_epochs = 1
    accuracies = []
    losses = []

    for i in tqdm(range(self.n_epochs), desc="Training Epochs: "):
      # Entrenar el modelo
      self.model.train()

      samples = 0
      loss_sum = 0
      correct_sum = 0
      for j, batch in enumerate(dataloaders['train']):
        X = batch["image"]
        targets = batch["label"]
        targets = targets.type(torch.FloatTensor)
        if self.use_gpu:
          X, targets = X.cuda(), targets.cuda()

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          y = torch.squeeze(self.model(X))
          if self.use_gpu:
            y = y.cuda()
          loss = self.criterion(y, targets)
          loss.backward()
          self.optimizer.step()

          # Multiplicar por tam de Batch porque loss es la loss media del Batch
          loss_sum += loss.item() * X.shape[0]
          samples += X.shape[0]
          num_corrects = torch.sum(
            (y >= 0.5).float() == targets.view(-1, 1).float())
          correct_sum += num_corrects

          # Print batch statistics every 50 batches
          batch_loss = float(loss_sum) / float(samples)
          batch_acc = float(correct_sum) / float(samples)
          batch_epochs += 1
          losses.append(batch_loss)
          accuracies.append(batch_acc)
          if j % 50 == 49:
            print(f"E{i + 1}:B{j + 1} - loss: {batch_loss}, acc: {batch_acc}")

      self.scheduler.step()

      # Estadisticas de la epoca
      epoch_acc = float(correct_sum) / float(samples)
      epoch_loss = float(loss_sum) / float(samples)
      print(f"Epoch: {i+1} Train loss: {epoch_loss}, Train acc: {epoch_acc}")

      # Guardar el modelo cada vez que mejora el accuracy
      if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(best_model_wts, "saves/resnet50.pth")

    return batch_epochs, accuracies, losses

  def predict(self, dataloaders, state_dict_file="saves/resnet50.pth", n_batches=None):
    """
    Realiza predicciones en base al modelo entrenado previamente

    Args:
      dataloaders: Dict con los dataloaders de Train y Test
      state_dict_file: Fichero con los pesos del modelo entrenado (guardados en fit)
      n_batches: Si no es None, determina el numero de lotes a predecir antes de terminar
    """
    self.model.load_state_dict(torch.load(state_dict_file))

    # Hacer predicciones
    self.model.eval()

    predictions = []
    ground_truth = []

    for bi, batch in enumerate(dataloaders['test']):
      X = batch["image"]
      targets = batch["label"]
      ground_truth = np.append(ground_truth, targets.numpy())
      if self.use_gpu:
        X = X.cuda()

      with torch.set_grad_enabled(False):
        y_pred = torch.squeeze(self.model(X))
        predictions = np.append(predictions, y_pred.cpu().numpy())

      if n_batches is not None and bi + 1 == n_batches:
        break

    m_err = self.mean_error(ground_truth, predictions)
    print(f'==> ERROR MEDIO: {m_err}')

  def mean_error(self, datos, pred):
    if len(datos) != len(pred):
      print(
        f'==> Longitudes no coinciden! datos:{len(datos)}, pred:{len(pred)}')
    return len([i for i in range(len(datos)) if datos[i] != pred[i]]) / len(datos)


class DummyModel(object):

  def __init__(self):
    super(DummyModel, self).__init__()

  def fit(self):
    return "Fit"
  
  def predict(self, instance, state_dict_file="saves/resnet50.pth"):
    return np.random.uniform(low=0, high=1)