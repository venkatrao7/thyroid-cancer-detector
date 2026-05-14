from torchvision import transforms
import torch

class ToTensor:
    def __call__(self, image, mask):
        return transforms.ToTensor()(image), transforms.ToTensor()(mask)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask

class Resize:
    def __init__(self, size):
        self.size = size  # tuple like (256, 256)

    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size)
        mask = transforms.functional.resize(mask, self.size)
        return image, mask

class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
