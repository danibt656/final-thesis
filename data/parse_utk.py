#/usr/bin/python3

import os
import argparse
import random


ROOT = "UTKFace"
ROOT_I = "UTKFace/Images"

# [age]_[gender]_[race]_[date&time].jpg
num_imgs = 23708
gender = {'male': 0, 'female': 1}
gender_rev = {'0': 'male', '1': 'female'}
race = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}

def move_to_root():
    os.mkdir(ROOT_I)
    for phase in ['train', 'val']:
        print(f'moving {phase.upper()} images.....', end='')
        for subdir, _, files in os.walk(os.path.join(ROOT, phase)):
            if subdir != ROOT_I and subdir != ROOT:
                for file in files:
                    os.replace(os.path.join(subdir, file), os.path.join(ROOT_I, file))
                os.rmdir(subdir)
        print('Done')
    print(f'\n\tAll images were moved to the root directory {ROOT_I}')
                

def create_mf_subset(val_prob):
    """ Create a Male-Female partition sub-dataset """
    # Create directories
    for phase in ['train', 'val']:
        os.mkdir(os.path.join(ROOT, phase))
        for label in ['male', 'female']:
            os.mkdir(os.path.join(ROOT+'/'+phase, str(gender[label])+'/'))

    # Move images to respective directories
    for filename in os.listdir(ROOT_I):
        f = os.path.join(ROOT_I, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.lower().endswith('.jpg'):
            # decide wether train or validation
            prob = random.uniform(0, 1)
            phase = 'val' if prob <= val_prob else 'train'
            # Move to corresponding folder
            destiny = filename.split('_')[1] # 2nd number is gender
            os.replace(f, os.path.join(ROOT+'/'+phase, destiny+'/'+filename))
            print(f'{filename} ---> {gender_rev[destiny]}')
    os.rmdir(ROOT_I)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tinyGPT')
    parser.add_argument('-r','--root', help='Move all files to root directory "Images/"', required=False, action='store_true')
    parser.add_argument('-mf','--malefemale', help='M-F sub-dataset with Validation percentage', required=False)
    args = vars(parser.parse_args())

    if args['root']:
        move_to_root()
    if args['malefemale']:
        create_mf_subset(int(args['malefemale'])/100)