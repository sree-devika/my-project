
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your saved model
save_dir = "./distilbert_plant_model"

# Check if folder exists
if not os.path.exists(save_dir):
    raise FileNotFoundError(f"Model folder not found: {save_dir}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model.eval()

# Prediction function
def predict(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]

if __name__ == "__main__":
    description = input("Enter leaf description: ")
    try:
        result = predict(description)
        print("Predicted Disease:", result)
    except Exception as e:
        print("Error during prediction:", str(e))