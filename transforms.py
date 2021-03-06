import random
import math

from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation
from autoaug.augmentations import apply_augment
from autoaug.archive import fa_reduced_cifar10, fa_reduced_imagenet


class KeepAsepctResize:
    def __init__(self, target_size=(288, 288)):
        self.target_size = target_size
    
    def __call__(self, img):
        width, height = img.size
        long_side = max(width, height)
        
        delta_w = long_side - width
        delta_h = long_side - height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        img = img.resize(self.target_size)
        return img
        
class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


def get_transform(
        target_size=(224,224),
        transform_list='resize, horizontal_flip', # random_crop | keep_aspect
        augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    augments = list()

    for transform_name in transform_list:
        if transform_name == 'random_crop':
            scale = (0.6, 1.0) if is_train else (0.8, 1.0)
            transform.append(RandomResizedCrop(target_size, scale=(0.8, 1.0)))
        elif transform_name == 'resize':
            transform.append(Resize(target_size))
        elif transform_name == 'auto_imagenet':
            transform.append(Augmentation(fa_reduced_imagenet()))
        elif transform_name == 'auto_cifar10':
            transform.append(Augmentation(fa_reduced_cifar10()))
        elif transform_name == 'keep_aspect':
            transform.append(KeepAsepctResize(target_size))
        elif transform_name == 'horizontal_flip':
            augments.append(RandomHorizontalFlip())
        elif transform_name == 'vertical_flip':
            augments.append(RandomVerticalFlihorizontal_flipp())
        elif transform == 'random_grayscale':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomGrayscale(p))
        elif transform_name == 'random_rotate':
            augments.append(RandomRotation(15))
        elif transform_name == 'color_jitter':
            brightness = 0.1 if is_train else 0.05
            contrast = 0.1 if is_train else 0.05
            augments.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
            ))
    transform.append(RandomApply(augments, p=augment_ratio))   
    transform.append(ToTensor())
    # transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transform)