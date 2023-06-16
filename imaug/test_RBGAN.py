import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
nz = 100

netG = torch.load('saves/netG_PV.pt')
netD = torch.load('saves/netD_PV.pt')

from torchvision.utils import save_image

N_to_generate = 1280  # TODO PONER ESTO!!!!!
n_generated = 0
ID=0
SAVE_FOLDER = '../data/PlantVillage/train/tomato_infected/'
noise = torch.randn(64, nz, 1, 1, device=device)

while n_generated < N_to_generate:
    fake = netG(noise).detach().cpu() # 64, 3, 64, 64
    for i, imagen in enumerate(fake):
        ID+=1
        imagen += 1
        imagen = imagen / (imagen.max() - imagen.min())
        if i==1 and n_generated==0:
            plt.imshow(np.transpose(imagen, (1, 2, 0)))
            plt.show()
        nombre_archivo = SAVE_FOLDER + f"_fake_{ID}.jpg"
        save_image(imagen, nombre_archivo)
    n_generated += fake.shape[0]
    print(f'{n_generated}/{N_to_generate} images done')
print(f'Done generating {n_generated} synthetic samples')