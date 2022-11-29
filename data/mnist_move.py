import os
import numpy as np

dir_path = './MNIST/Images/'

labels = [i for i in range(10)]

for l in labels:
  try:
    os.rmdir(f'./MNIST/{l}')
  except OSError:
    print(f'dir ./MNIST/{l} does not exist')

for l in labels:
  try:
    os.mkdir(f'./MNIST/{l}')
  except OSError:
    print(f'dir ./MNIST/{l} already exists')

for path in os.listdir(dir_path):
  if os.path.isfile(os.path.join(dir_path, path)):
    path_info = path.split('_')
    label = path_info[0]
    os.rename(os.path.join(dir_path, path), os.path.join(f'./MNIST/{label}', path))
    print(path)
