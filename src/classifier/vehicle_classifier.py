import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

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


class SimpleCNN(nn.Sequential):
    def __init__(self,
                 h_conv: list[int] = [16, 32, 32],
                 c_conv: dict[str, int] = {'kernel_size': 3, 'stride': 1, 'padding': 1},
                 h_fc: list[int] = [128],
                 input_shape=(1, 3, 32, 32),
                 out_classes=5,
                 dtype: torch.dtype = torch.float32,
                 add_pools=True,
                 device='cpu',
                 ):

        sp = {'dtype': dtype, 'device': device}

        shape = input_shape
        layers: list[nn.Module] = []
        if len(h_conv) >= 2:
            layers.append(nn.Conv2d(shape[1], h_conv[0], **sp, **c_conv))
            if add_pools: layers.append(nn.MaxPool2d(2, stride=2))
            layers.append(nn.ReLU())
            for h_size in h_conv[1:-1]:
                shape = get_out(input_shape, layers)
                layers.append(nn.Conv2d(shape[1], h_size, **sp, **c_conv))
                if add_pools: layers.append(nn.MaxPool2d(2, stride=2))
                layers.append(nn.ReLU())

        if len(h_conv) >= 1:
            shape = get_out(input_shape, layers)
            layers.append(nn.Conv2d(shape[1], h_conv[-1], **sp, **c_conv))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())
        for h_size in h_fc:
            shape = get_out(input_shape, layers)
            layers.append(nn.Linear(shape[1], h_size, **sp))
            layers.append(nn.ReLU())
        shape = get_out(input_shape, layers)
        layers.append(nn.Linear(shape[1], out_classes, **sp))
        # layers.append(nn.Softmax())  # In most cases pytorch expects raw logits, not the soft max

        super(SimpleCNN, self).__init__(*layers)


class MobileNetCNN(nn.Module):
    def __init__(self,
                 input_shape=(1, 3, 32, 32),
                 out_classes=5,
                 dtype: torch.dtype = torch.float32,
                 device='cpu',
                 ):
        super(MobileNetCNN, self).__init__()
        # Expected input shape is 224,224
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=out_classes)

    def forward(self, x):
        return self.model(x)


class WhatAmIDoingCNN(nn.Module):
    def __init__(self,
                 out_classes=5,
                 device='cpu',
                 ):
        super(WhatAmIDoingCNN, self).__init__()
        # Expected input shape is 224,224
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier = nn.Identity()  # Output is 576
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=256, device=device),
            nn.Hardtanh(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=out_classes, device=device),
        )
        self.discriminator = nn.Linear(in_features=576, out_features=1, device=device)

    def forward(self, x):
        internal_state = self.model.forward(x)
        classification = self.classifier(internal_state)
        discrimination = self.discriminator(internal_state)
        return classification, discrimination


# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path='model_state_space.pth'):
        self.device = torch.device("cpu")
        self.model = WhatAmIDoingCNN(out_classes=4)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path: str) -> int:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            classification, discrimination = self.model(tensor)
            discrimination = discrimination > 1
            return (classification.argmax(axis=1) * discrimination + 4 * (not discrimination)).item()


# -----------------------------
# Utilities
# -----------------------------

def get_out(input_shape: tuple[int], model: list[nn.Module] | nn.Module):
    if not isinstance(model, list): model = [model]
    if len(model) == 0: return input_shape
    with torch.no_grad():

        parameter = None
        for layer in model:
            parameters = layer.parameters()
            for p in parameters: parameter = p
        if parameter is not None:
            device, dtype = parameter.device, parameter.dtype
        else:
            device, dtype = "cpu", torch.float32

        meta = torch.zeros(input_shape, device=device, dtype=dtype)  # device='meta'
        for layer in model:
            meta = layer(meta)
        return meta.shape


def fop(path: str):
    d = path.rsplit(os.sep, 1)[0]
    os.path.exists(d) or os.makedirs(d)
    return path


def measure_size(model):
    file = fop('temp\\model_measure_size.pth')
    torch.save(model.state_dict(), file)
    size = os.path.getsize(file)
    os.remove(file)
    return size / (1024 * 1024)

# # -----------------------------
# # Example Usage
# # -----------------------------
# if __name__ == "__main__":
#     classifier = VehicleClassifier(model_path="student_model.pth")  # load your trained weights
#     idx = classifier.predict("test_image.jpg")
#     print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")
