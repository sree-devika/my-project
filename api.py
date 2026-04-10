from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
import json
import os
import torch.nn as nn

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- FastAPI App --------------------
app = FastAPI(title="Plant Disease Detection API")

# -------------------- Paths --------------------
cnn_model_path = "best_simplecnn_plant_disease.pth"
cnn_class_file = "distilbert_plant_model/config.json"
text_model_dir = "distilbert_plant_model"

# -------------------- Load CNN Classes --------------------
if os.path.exists(cnn_class_file):
    with open(cnn_class_file, "r") as f:
        config = json.load(f)
        class_names_cnn = list(config["id2label"].values())
else:
    raise FileNotFoundError(f"CNN class file not found: {cnn_class_file}")

num_classes = len(class_names_cnn)
print("CNN Classes:", num_classes)

# -------------------- CNN MODEL --------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -------------------- Load CNN Model --------------------
cnn_model = CNN(num_classes=num_classes)

checkpoint = torch.load(cnn_model_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)

# Remove last layer weights (avoid mismatch)
state_dict.pop('classifier.3.weight', None)
state_dict.pop('classifier.3.bias', None)

cnn_model.load_state_dict(state_dict, strict=False)

cnn_model.to(device)
cnn_model.eval()

# -------------------- Image Transform --------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------- Text Model --------------------
tokenizer_text = AutoTokenizer.from_pretrained(text_model_dir)
model_text = AutoModelForSequenceClassification.from_pretrained(text_model_dir)

model_text.to(device)
model_text.eval()

# -------------------- Image Prediction --------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only image files supported")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(image_tensor)
            pred_id = torch.argmax(output, dim=1).item()
            pred_class = class_names_cnn[pred_id]

        return {
            "filename": file.filename,
            "predicted_disease": pred_class
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Text Prediction --------------------
@app.post("/predict-text")
async def predict_text(description: str = Form(...)):

    try:
        inputs = tokenizer_text(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model_text(**inputs)
            pred_id = outputs.logits.argmax(dim=-1).item()
            pred_class = model_text.config.id2label[pred_id]

        return {
            "description": description,
            "predicted_disease": pred_class
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Health Check --------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# -------------------- Root --------------------
@app.get("/")
async def root():
    return {
        "message": "Plant Disease Detection API",
        "endpoints": [
            "/predict-image",
            "/predict-text",
            "/health"
        ]
    }