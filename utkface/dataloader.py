# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
Clases para manejar de forma comoda los datasets
"""

import os
import glob
import pandas as pd

import torch
from torchvision import transforms
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate

#################################################################################################

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
  Usado para extraer info del dataset UTKFace

  Args:
    dataset_path: Directorio con las imagenes en el formato indicado
    ext: Extension (formato) de las imagenes (jpg, png...)

  Return:
    DataFrame con (age, gender, race) de todos los ficheros
  """
  def parse_info_from_file(path):
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

#################################################################################################


class Dataset(torch.utils.data.Dataset):
  """
  Clase del dataset. Carga las imagenes, hace las transformaciones, y carga sus correspondientes etiquetas
  """

  def __init__(self, df, img_dir, transform=None):
    """
    Args:
      df: DataFrame con la info del dataset
      img_dir: Directorio con las imagenes
      transform: Pipeline de transformaciones a aplicar a las imagenes
    """
    self.df = df
    self.img_dir = img_dir
    self.transform = transform
    self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]

  def __getitem__(self, idx):
    # print(f'#{idx}...', end='')
    if idx >= self.df.shape[0]:
      idx = self.df.shape[0] - 1
    img_path = self.df.iloc[idx]['file']
    # print("img_path:", img_path)
    # print('OK')

    # Leemos las imagenes aqui para cargar solo las necesarias en memoria
    img = imread(img_path)

    if self.transform:
      img = self.transform(img)

    sample = {
      "image": img,
    }
    gender = dataset_dict['gender_alias'][self.df.iloc[idx]["gender"]]
    # race = dataset_dict['race_alias'][self.df.iloc[idx]["race"]]
    # Labels del tipo genero-raza
    sample["label"] = gender
    return sample

  def __len__(self):
    try:
      return self.df.shape[0]
    except AttributeError:
      return len(self.images)

#################################################################################################

def bias_collate(batch):
  """
  Funcion para producir sesgos en el dataset
  """
  modified_batch = []
  for item in batch:
    _, label = item
    # if label == 1 or label == 2: # only train in these numbers, but test on all!
    modified_batch.append(item)
  return default_collate(modified_batch)


class UTKFaceDS():
  """
  Clase para el dataset especifico (UTKFace)

  Attrs:
    df: DataFrame
    dataset: Objeto de clase Dataset con la info general
    dataloaders: Dict con los dataloaders de Train y Test
  """

  def __init__(self,
               dataset_folder_name="../data/UTKFace/Images",
               model_res=(224, 224),
               batch_size=64,
               test_size=0.3
               ):
    """
    Args:
      dataset_folder_name: directorio con las imagenes del dataset
      model_res: Resolucion aceptada por el modelo de vision artificial. Por defecto 224x224 px (ResNet)
      batch_size: Tamaño de los chunks que devuelven los dataloaders
      test_size: Porcentaje sobre el total de datos que representan los datos de test, Por defecto 30% = 0.3
    """
    self.df = parse_dataset(dataset_folder_name)
    train_indices, test_indices = train_test_split(
      self.df.index, test_size=test_size)

    transform_pipe = transforms.Compose([
      transforms.ToPILImage(),  # Convertir array de numpy a imagen de PIL
      # Resize a la resolucion requerida por el modelo
      transforms.Resize(
        size=model_res
      ),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
      )
    ])

    self.dataset = Dataset(
      df=self.df,
      img_dir=dataset_folder_name,
      transform=transform_pipe
    )

    # The training dataset loader will randomly sample from the train samples
    train_loader = DataLoader(
      self.dataset,
      batch_size=batch_size,
      sampler=torch.utils.data.SubsetRandomSampler(train_indices),
      num_workers=8,
      collate_fn = bias_collate,
    )

    # The testing dataset loader will randomly sample from the test samples
    test_loader = DataLoader(
      self.dataset,
      batch_size=batch_size,
      sampler=torch.utils.data.SubsetRandomSampler(test_indices),
      num_workers=8,
    )

    self.dataloaders = {
      "train": train_loader,
      "test": test_loader
    }
