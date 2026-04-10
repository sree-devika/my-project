import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- 1. THE ARCHITECTURE ---
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 2. LOAD CHECKPOINT ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_simplecnn_plant_disease.pth", map_location=device)

model = CNN(checkpoint['num_classes']).to(device)  # ✅ pass num_classes
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # ✅ set to evaluation mode

class_names = checkpoint['class_names']

# --- 3. PREDICT ON SINGLE IMAGE ---
img_path = r"C:\Users\SREELAKSHMI\Documents\plant disease detection\capped_dataset\val\Apple___Apple_scab\0a14783a-838a-4d4f-a671-ff98011714c6___FREC_Scab 3288.jpg"
if os.path.exists(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    print(f"🌿 Prediction: {class_names[predicted.item()]}")
else:
    print(f"❌ Error: Image not found at {img_path}")