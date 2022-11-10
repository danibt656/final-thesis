# -*- coding: utf-8 -*-
from dataloader import MNISTDS, parse_mnist
from model import ResNetModel

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

resnet.fit(data.dataloaders)

print('####################################################')
print('          PREDICT')
print('####################################################')

resnet.predict(data.dataloaders, n_batches=3)