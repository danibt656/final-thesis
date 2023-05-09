import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm
from sklearn.metrics.cluster import entropy

def glcm_entropy(data_path, dl, IMGSIZE, ext='JPG'):

    ds = np.zeros((1*IMGSIZE*IMGSIZE))
    for imgs, _ in tqdm(dl):
        ds=np.vstack((ds, imgs.numpy().reshape(imgs.shape[0], imgs.shape[1]*imgs.shape[2]*imgs.shape[3])))
    ds = ds[1:]

    entropies = np.array([])
    for gray_img in tqdm(ds):
        ee = entropy(gray_img.reshape(1,IMGSIZE,IMGSIZE))
        entropies = np.append(entropies, ee)

    print('Mean:', entropies.mean(), '| STD:', entropies.std())

    plt.hist(entropies, bins=np.arange(min(entropies), max(entropies), step=1e-2))
    plt.xticks(np.arange(min(entropies), max(entropies), .5))
    plt.xlabel('GLCM entropy')
    plt.ylabel('Frequency')
    plt.title(f'Frequencies of GLCM for {data_path.split("/")[2]} dataset')
    plt.savefig(f'entropy_hist_{data_path.split("/")[2]}.png')


if __name__ == '__main__':
    _, data_path, imsize, ext = sys.argv
    imsize = int(imsize)

    data_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    image_dataset = datasets.ImageFolder(data_path, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=10, shuffle=True, num_workers=4)

    glcm_entropy(data_path=data_path, dl=dataloader, IMGSIZE=imsize, ext=ext)
