import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.seg_dataset import ThyroidSegmentationDataset
from utils.custom_transforms import ToTensor, RandomHorizontalFlip, Resize, Compose

from tqdm import tqdm
import os

# ✅ U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)

# ✅ Paths to your image and mask folders
image_dir = "data/seg/train/image"
mask_dir = "data/seg/train/masks"

# ✅ Transform
# ✅ Transform
transform = Compose([
    Resize((256, 256)),  # <-- new line added
    RandomHorizontalFlip(p=0.5),
    ToTensor()
])

# ✅ Dataset and DataLoader
train_dataset = ThyroidSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ✅ Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

print("✅ Training complete!")
