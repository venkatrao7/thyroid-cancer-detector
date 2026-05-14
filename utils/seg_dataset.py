import os
from PIL import Image
from torch.utils.data import Dataset

class ThyroidSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.endswith('.png') or f.endswith('.jpg')]
        self.image_paths.sort()
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(f)) for f in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
