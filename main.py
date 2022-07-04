import os
import train_val_split
import augmentations
import display_plots


def label_count(dir_path):
    x = [(dirs, len(os.listdir(f'{dir_path}/{dirs}'))) for dirs in sorted(os.listdir(dir_path))]
    print(x)
    print(sum(a[1] for a in x))


def count_process():
    print('original')
    label_count('data_original/train')
    print('fixed')
    label_count('data_fixed/train')
    print('split train')
    label_count('data_split/train')
    print('split val')
    label_count('data_split/val')

    print('augmentations')
    for aug in os.listdir('augmentations'):
        dir_path = f'augmentations/{aug}'
        print(aug)
        label_count(dir_path)

    print('final train')
    label_count('data/train')
    print('final validation')
    label_count('data/val')


if __name__ == '__main__':
    # train_val_split.split('data')

    # augmentations.augment_images(['Rotate'], 'rotate')
    # augmentations.augment_images(['Noise'], 'noise')
    # augmentations.augment_images(['FlipHorizontal'], 'flipH')
    # augmentations.augment_images(['FlipVertical'], 'flipV')
    # augmentations.augment_images(['Distort'], 'distort')
    # augmentations.augment_images(['Auto'], 'auto')
    # augmentations.augment_images(['Move'], 'move')

    # # augmentations.augment_images(['Add'], 'add')

    # train_val_split.transfer_aug_to_data()

    # label_count('data/train')
    # label_count('data/val')

    # for aug in os.listdir('augmentations'):
    #     # train_val_split.split(f'data/{aug}', aug)
    #     dir_path = f'augmentations/{aug}'
    #     print(aug)
    #     label_count(dir_path)

    # count_process()
    display_plots.plot_PCA()
    display_plots.confusion_shit()

