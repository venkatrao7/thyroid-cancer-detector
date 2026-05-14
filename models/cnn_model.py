# models/cnn_model.py

import torch.nn as nn
import torchvision.models as models

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNClassifier, self).__init__()
        
        # Load ResNet18 model with pretrained weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
