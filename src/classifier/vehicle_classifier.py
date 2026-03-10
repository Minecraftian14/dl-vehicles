import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -----------------------------
# Class Index Mapping (consistent across all submissions)
# -----------------------------
CLASS_IDX = {
    0: "Bus",
    1: "Truck",
    2: "Car",
    3: "Bike",
    4: "None"
}

# -----------------------------
# Lightweight CNN Model (<5 MB)
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # assuming input resized to 32x32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = SmallCNN(num_classes=len(CLASS_IDX))
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_path: str) -> int:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    classifier = VehicleClassifier(model_path="student_model.pth")  # load your trained weights
    idx = classifier.predict("test_image.jpg")
    print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")