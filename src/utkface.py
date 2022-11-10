# -*- coding: utf-8 -*-
from dataloader import UTKFaceDS, parse_utkface
from model import ResNetModel

data = UTKFaceDS(dataset_folder_name="../data/UTKFace/Images",
             parse_method=parse_utkface,
             model_res=(224, 224),
             batch_size=64,
             test_size=0.3
            )

resnet = ResNetModel(in_ch=3, out_f=1, n_epochs=1, use_gpu=True)

print('####################################################')
print('          FIT')
print('####################################################')

resnet.fit(data.dataloaders)

print('####################################################')
print('          PREDICT')
print('####################################################')

resnet.predict(data.dataloaders)