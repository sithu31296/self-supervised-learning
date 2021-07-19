"""torchvision builtin transforms
# shape transform
CenterCrop(size)
Resize(size)
RandomCrop(size, padding=None, pad_if_needed=False, fill=0)
RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33))
RandomRotation(degrees)
Pad(padding, fill=0)

# spatial transform
ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
GaussianBlur(kernel_size, sigma=(0.1, 2.0))
RandomAffine(degrees, translate=None, scale=None, shear=None)
RandomGrayscale(p=0.1)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomPerspective(distortion_scale=0.5, p=0.5)
RandomInvert(p=0.5)
RandomPosterize(bits, p=0.5)
RandomSolarize(threshold, p=0.5)
RandomAdjustSharpness(sharpness_factor, p=0.5)
RandomAutocontrast(p=0.5)

# auto-augment
AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)

# others
RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
RandomApply(transforms, p=0.5)      # apply randomly a list of transformations with a given probability
"""
import random
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T



class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.) -> None:
        self.p = p
        self.radius = random.uniform(radius_min, radius_max)

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(self.radius))
        return img


class Solarization:
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class DINOAug:
    def __init__(self, img_size, crop_scale, local_crops_number) -> None:
        flip_color = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ])
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(img_size, (crop_scale, 1.0), interpolation=Image.BICUBIC),
            flip_color,
            GaussianBlur(1.0),
            normalize
        ])

        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(img_size, (crop_scale, 1.0), interpolation=Image.BICUBIC),
            flip_color,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize
        ])

        self.local_crops_number = local_crops_number
        self.local_transform = T.Compose([
            T.RandomResizedCrop((img_size[0]//2, img_size[1]//2), (0.05, crop_scale), interpolation=Image.BICUBIC),
            flip_color,
            GaussianBlur(0.5),
            normalize
        ])

    def __call__(self, img):
        crops = []
        crops.append(self.global_transform1(img))
        crops.append(self.global_transform2(img))

        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))

        return crops