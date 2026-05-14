
import torch
from torchvision import datasets, transforms

def get_classification_loaders(root_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(f"{root_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(f"{root_dir}/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
