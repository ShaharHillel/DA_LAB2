import os
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, AutoAugment, AutoAugmentPolicy, GaussianBlur, Pad
import random
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class AAug(AutoAugment):
    def __init__(self, policy):
        super().__init__(AutoAugmentPolicy.SVHN)
        self.fill = 255

    def _get_policies(self, policy: AutoAugmentPolicy):
        if policy == AutoAugmentPolicy.SVHN:
            return [
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
            ]


def save_aug_images(new_images, aug_name):
    print(f"Saving augmented images of {aug_name} augmentations")
    if not os.path.exists(f'augmentations/{aug_name}'):
        os.makedirs(f'augmentations/{aug_name}')

    for label, images in new_images.items():
        if not os.path.exists(f'augmentations/{aug_name}/{label}'):
            os.makedirs(f'augmentations/{aug_name}/{label}')
        for image, im_name in images:
            image.save(f'augmentations/{aug_name}/{label}/{im_name}', 'PNG')


def center_image(image):
    image_np = np.array(image)
    mask = np.abs(image_np - 255) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    new_image = PIL.Image.fromarray(out)
    # new_image = PIL.Image.new(image.mode, (out.shape[1] + 20, out.shape[0] + 20), 'WHITE')
    # new_image.paste(PIL.Image.fromarray(out), (10, 10))
    # plt.imshow(new_image)
    # plt.show()
    return new_image


def add_images(image1, image2):
    pad = Pad(50, fill=255)
    size1 = image1.size
    size2 = image2.size
    size = (min(size1[0], size2[0]), min(size1[1], size2[1]))
    image1 = image1.resize(size)
    image2 = image2.resize(size)
    new_image = PIL.Image.new('RGB', (2 * size[0], size[1]), 'WHITE')
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (size[0], 0))
    new_image = pad(new_image)
    # plt.imshow(image1)
    # plt.show()
    # plt.imshow(image2)
    # plt.show()
    # plt.imshow(new_image)
    # plt.show()
    return new_image


# Resize all images to 64x64
def resize_images():
    pass


def append_name(image_path, aug_type):
    return image_path.split('/')[-1][:-4] + '_' + aug_type + '.png'


def general_aug(new_images, images_path_dict, augmentation, aug_type):
    for label, images_paths in tqdm(images_path_dict.items()):
        for image_path in images_paths:
            im_name = append_name(image_path, aug_type)
            with PIL.Image.open(image_path) as image:
                new_images[label].append((augmentation(image.copy()), im_name))


def flip_h_aug(new_images, images_path_dict, aug_type):
    augmentation = RandomHorizontalFlip(p=1.0)
    for label, images_paths in tqdm(images_path_dict.items()):
        if label not in ['ix', 'vii', 'viii']:
            for image_path in images_paths:
                im_name = append_name(image_path, aug_type)
                with PIL.Image.open(image_path) as image:
                    if label in ['i', 'ii', 'iii', 'v', 'x']:
                        new_images[label].append((augmentation(image.copy()), im_name))
                    elif label == 'iv':
                        new_images['vi'].append((augmentation(image.copy()), im_name))
                    elif label == 'vi':
                        new_images['iv'].append((augmentation(image.copy()), im_name))


def flip_v_aug(new_images, images_path_dict, aug_type):
    augmentation = RandomVerticalFlip(p=1.0)
    for label, images_paths in tqdm(images_path_dict.items()):
        if label in ['i', 'ii', 'iii', 'ix', 'x']:
            for image_path in images_paths:
                im_name = append_name(image_path, aug_type)
                with PIL.Image.open(image_path) as image:
                    new_images[label].append((augmentation(image.copy()), im_name))


def rotate_aug(new_images, images_path_dict, aug_type):
    augmentation = RandomRotation(15, fill=255)
    general_aug(new_images, images_path_dict, augmentation, aug_type)


# TODO yeah
def distort_aug(new_images, images_path_dict, aug_type):
    for label, images_paths in tqdm(images_path_dict.items()):
        for image_path in images_paths:
            im_name = append_name(image_path, aug_type)
            with PIL.Image.open(image_path) as image:
                img = np.array(image)
                A = img.shape[0] / 20.0
                w = 2.0 / img.shape[1]

                shift = lambda x: A * np.sin(2.0 * np.pi * x * w)

                for i in range(img.shape[1]):
                    img[:, i] = np.roll(img[:, i], int(shift(i)))
                for i in range(img.shape[0]):
                    img[i, :] = np.roll(img[i, :], int(shift(i)))
                new_image = PIL.Image.fromarray(img)
                new_images[label].append((new_image, im_name))
                # plt.imshow(new_image)
                # plt.show()


def auto_aug(new_images, images_path_dict, aug_type):
    augmentation = AAug(policy=AutoAugmentPolicy.SVHN)
    general_aug(new_images, images_path_dict, augmentation, aug_type)


def noise_aug(new_images, images_path_dict, aug_type):
    augmentation = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    general_aug(new_images, images_path_dict, augmentation, aug_type)


def move_aug(new_images, images_path_dict, aug_type):
    for label, images_paths in tqdm(images_path_dict.items()):
        for image_path in images_paths:
            with PIL.Image.open(image_path) as image:
                w, h = image.size
                new_width = w*2
                new_height = h*2

                center = (w//2, h//2)
                top_left = (0, 0)
                top_right = (w, 0)
                bottom_left = (0, h)
                bottom_right = (w, h)
                # for i, loc in enumerate([center, top_left, top_right, bottom_left, bottom_right]):
                #     im_name = append_name(image_path, aug_type[i])
                #     new_image = PIL.Image.new(image.mode, (new_width, new_height), 'WHITE')
                #     new_image.paste(image, loc)
                #     # plt.imshow(new_image)
                #     # plt.show()
                #     new_images[label].append((new_image, im_name))

                # for i in range(2):
                #     im_name = append_name(image_path, f'move_{i}')
                #     new_image = PIL.Image.new(image.mode, (new_width, new_height), 'WHITE')
                #     loc = (w//2 + random.randrange(-w//2, +w//2), h//2 + random.randrange(-h//2, +h//2))
                #     new_image.paste(image, loc)
                #     new_images[label].append((new_image, im_name))
                im_name = append_name(image_path, 'move')
                new_image = PIL.Image.new(image.mode, (new_width, new_height), 'WHITE')
                loc = (w//2 + random.randrange(-w//2, +w//2), h//2 + random.randrange(-h//2, +h//2))
                new_image.paste(image, loc)
                new_images[label].append((new_image, im_name))


# TODO smarter
def add_aug(new_images, images_path_dict):  # no mixing up with i,ii,iii cause inconsistency
    label_combinations = [(('i', 'v'), 'iv'), (('v', 'i'), 'vi'), (('v', 'ii'), 'vii'),
                          (('v', 'iii'), 'viii'), (('i', 'x'), 'ix')]
    for (l1, l2), out_label in tqdm(label_combinations):
        pairs = [(im1, im2) for im1 in images_path_dict[l1] for im2 in images_path_dict[l2]]
        choice = [random.choice(pairs) for _ in range(300)]
        for image1_path, image2_path in choice:
            with PIL.Image.open(image1_path) as image1:
                with PIL.Image.open(image2_path) as image2:
                    center_image1 = center_image(image1)
                    center_image2 = center_image(image2)
                    new_image = add_images(center_image1, center_image2)
                    im_name = image1_path.split('/')[-1][:4] + '_' + image2_path.split('/')[-1][:4] + '.png'
                    new_images[out_label].append((new_image, im_name))


def augment_images(aug_types, aug_name):
    labels = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    data_dir_path = 'data_split/train/'
    label_dirs = {label_dir: data_dir_path + label_dir for label_dir in os.listdir(data_dir_path)}
    images_path_dict = {}
    for label in labels:
        images_path_dict[label] = []
        for image_file in os.listdir(label_dirs[label]):
            images_path_dict[label].append(label_dirs[label] + "/" + image_file)

    new_images = {label: [] for label in labels}
    print('Augmenting')
    for aug_type in aug_types:
        print(f'Creating Augmentation {aug_type}')
        if 'FlipHorizontal' == aug_type:  # +1500
            flip_h_aug(new_images, images_path_dict, aug_type)
        elif 'FlipVertical' == aug_type:  # +1000
            flip_v_aug(new_images, images_path_dict, aug_type)
        elif 'Rotate' == aug_type:  # +2000
            rotate_aug(new_images, images_path_dict, aug_type)
        elif 'Distort' == aug_type:  # +2000
            distort_aug(new_images, images_path_dict, aug_type)
        elif 'Auto' == aug_type:  # +2000
            auto_aug(new_images, images_path_dict, aug_type)
        elif 'Noise' == aug_type:  # +2000
            noise_aug(new_images, images_path_dict, aug_type)
        elif 'Move' == aug_type:  # +10000 or +2000
            move_aug(new_images, images_path_dict, ['center', 'tl', 'tr', 'bl', 'br'])
        elif 'Add' == aug_type:  # +1500
            add_aug(new_images, images_path_dict)

    save_aug_images(new_images, aug_name)
