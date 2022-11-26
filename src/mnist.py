# -*- coding: utf-8 -*-
"""
@author: Daniel Barahona
"""
from dataloader import MNISTDS, parse_mnist
from model import ResNetModel
import matplotlib.pyplot as plt

data = MNISTDS(dataset_folder_name="../data/MNIST/Images",
             parse_method=parse_mnist,
             model_res=(224, 224),
             batch_size=64,
             test_size=0.3
            )

resnet = ResNetModel(in_ch=1, out_f=1, n_epochs=3, use_gpu=True)

print('####################################################')
print('          FIT')
print('####################################################')

# Con el output podemos hacer plots de loss y accuracy frente al batch de cada epoca
batch_epochs, accuracies, losses = resnet.fit(data.dataloaders)
# Accuracy plot
plt.plot(batch_epochs, accuracies, 'r', label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Batch-Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
# Loss plot
plt.plot(batch_epochs, losses, 'b', label='Loss')
plt.title('Loss')
plt.xlabel('Batch-Epoch')
plt.ylabel('Loss (%)')
plt.legend()
plt.show()

print('####################################################')
print('          PREDICT')
print('####################################################')

resnet.predict(data.dataloaders, n_batches=3)