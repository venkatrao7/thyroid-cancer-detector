# train_cls.py

from utils.cls_dataset import ThyroidClassificationDataset
from models.classifier import build_classifier
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# Updated path to dataset
dataset_path = "data/cls/train"
  # training data path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ThyroidClassificationDataset(root_dir=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = build_classifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
