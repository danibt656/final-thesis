# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
"""
import numpy as np
import threading
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


IMG_SIZE = 224
BATCH_SIZE = 32
DEGREES = 15
ROOT_I = 'Images'
gender_rev = {0: 'male', 1: 'female'}

def get_dataloaders_from_path(path):
  """ Wrapper for dataloader-from-directory logic
  """

  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomSizedCrop(IMG_SIZE),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
      transforms.Scale(IMG_SIZE),
      transforms.CenterCrop(IMG_SIZE),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
  }

  data_dir = path
  image_datasets = {x: datasets.ImageFolder(data_dir+x,data_transforms[x]) for x in ['train', 'val']}

  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],\
                                                batch_size=BATCH_SIZE,\
                                                shuffle=shuf,\
                                                num_workers=4)\
                  for x,shuf in [('train', True), ('val', False)]}

  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  class_names = [c for c in image_datasets['train'].classes if c != ROOT_I]

  return dataloaders, dataset_sizes, class_names


def plot_images_sample(dataloader):
  """ Plot 4 images from desired dataloader (train or val) """
  images, labels = next(iter(dataloader))
  _, axes = plt.subplots(figsize=(16, 4), ncols=4)
  for ii in range(4):
      ax = axes[ii]
      img = images[ii]
      npimg = img.numpy()
      ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
      ax.set_title(gender_rev[int(labels[ii])])


class DummyModel(object):

  def __init__(self, class_pair):
    super(DummyModel, self).__init__()
    self.class_pair = class_pair

  def fit(self):
    print(f"Fit with pair {self.class_pair}", end='')
  
  def predict(self, instance, state_dict_file="saves/resnet50.pth"):
    return np.random.uniform(low=0, high=1)


class FICAR(object):
  """
  Fuzzy Imbalanced Consensus Automated Reductor

  Attrs:
    m: Numero de clases del problema de clasificacion
    class_names: Lista con los nombres de las clases asignadas por fila/columna:
    predictors: Matriz de clasificadores para el esquema de descomposicion
    decisions: Matriz con las decisiones de los clasificadores para una cierta instancia (probabilidades)

    m x m:                  class_names[0]  class_names[1]    ...
            class_names[0]        -             1 vs. 0       ...

            class_names[1]       0 vs. 1           -          ...

            ....                   ...             ...         -
  """

  def __init__(self, n_classes=2, class_names=[]):
    """
    Args:
      n_classes: Numero de clases para clasificar (minimo 2)
      class_names: Lista con los nombres de las clases asignadas por fila/columna
    """
    super(FICAR, self).__init__()

    if len(class_names) != n_classes:
      raise ValueError("El numero de clases no coincide con el parametro `n_classes`")

    self.m = n_classes
    self.class_names = class_names
    self.predictors = np.empty((n_classes, n_classes), dtype=object)
    self.decisions = np.zeros((n_classes, n_classes), dtype=float)

    # m(m-1)/2 clasificadores (triangulo superior SIN diagonal)
    for r in range(self.m):
      for c in range(r + 1, self.m):
        if r == c:
          continue
        self.predictors[r, c] = DummyModel((self.class_names[r], self.class_names[c]))

  def train(self, dataloaders):
    """
    Entrena el modelo con los datos dados

    Args:
      dataloaders: Dict con los dataloaders de Train y Test
    """
    for r in range(self.m):
      for c in range(r + 1, self.m):
        if r == c:
          continue
        print(f'P({r},{c})-> ', end='')
        self.predictors[r, c].fit()
        print('')

  def predict(self, instance):
    """
    Predecir la clase de una instancia a traves de la matriz de descomposicion
      - m(m-1)/2 predicciones (triangulo superior SIN diagonal)
      - El triangulo inferior son las inversas: r_ij = 1 - r_ji

    Args:
      instance: Ejemplo para el que predice la clase

    Return:
      La clase predecida para la instancia
    """
    self.decisions = np.zeros((self.m, self.m))  # Reset a 0 las probabilidades

    for r in range(self.m):
      for c in range(r + 1, self.m):
        if r == c:
          continue
        # r_ij = Probabilidad de Ci frente a Cj
        self.decisions[r, c] = self.predictors[r, c].predict(instance)
        # r_ji = 1 - r_ij (Cj frente a Ci)
        self.decisions[c, r] = 1 - self.decisions[r, c]

    y_pred = self.aggregate(self.decisions)

    return y_pred

  def aggregate(self, decision_mat):
    """
    Agrega las predicciones de una matriz OVO

    Args:
      decision_mat: Matriz de probabilidades

    Return:
      La clase resultado de agregar las probabilidades y obtener la vencedora
    """
    max_i = decision_mat.argmax() // decision_mat.shape[0]
    max_j = decision_mat.argmax() % decision_mat.shape[0]
    
    assert max_i < self.m

    return self.class_names[max_j]
