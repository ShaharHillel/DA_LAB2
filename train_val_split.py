import os
import numpy as np
import shutil
from distutils.dir_util import copy_tree


def split(data_folder):

    # Creating Train / Val folders
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    os.makedirs(f'{data_folder}/train')
    os.makedirs(f'{data_folder}/val')
    for cls in classes:
        os.makedirs(f'{data_folder}/train/{cls}')
        os.makedirs(f'{data_folder}/val/{cls}')

        src = f'data_fixed/train/{cls}'  # Folder to copy images from
        allFileNames = [f'{src}/{f_name}'for f_name in os.listdir(src)]
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames = np.split(np.array(allFileNames),
                                                  [int(len(allFileNames) * 0.8)]
                                                  )

        train_FileNames = train_FileNames.tolist()
        val_FileNames = val_FileNames.tolist()

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, f'{data_folder}/train/{cls}')

        for name in val_FileNames:
            shutil.copy(name, f'{data_folder}/val/{cls}')


def transfer_aug_to_data():
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    if not os.path.exists('data/val'):
        os.makedirs('data/val')

    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    for cls in classes:
        if not os.path.exists(f'data/train/{cls}'):
            os.makedirs(f'data/train/{cls}')
        if not os.path.exists(f'data/val/{cls}'):
            os.makedirs(f'data/val/{cls}')
            copy_tree(f'data_split/val/{cls}', f'data/val/{cls}')

    for aug in os.listdir('augmentations'):
        for cls in classes:
            copy_tree(f'augmentations/{aug}/{cls}', f'data/train/{cls}')
