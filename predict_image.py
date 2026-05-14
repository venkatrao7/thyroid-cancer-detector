import torch
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model class (same as the one used during training)
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 16 * 16, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CNNClassifier().to(device)
model.load_state_dict(torch.load("cnn_classifier.pth", map_location=device))
model.eval()

# Define the same image transformations used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Function to predict a single image
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Predict
    with torch.no_grad():
        output = model(image)
        predicted = (output > 0.5).long().item()
    
    return "Malignant" if predicted == 1 else "Benign"

# Test the model on an image
if __name__ == "__main__":
    image_path = "data/cls/val/ID_i/1_i.jpg"  # Update with the path to the image you want to test
    result = predict_image(image_path)
    print(f"Prediction: {result}")
