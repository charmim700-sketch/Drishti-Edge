from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="Drishti Edge API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Class names
# =========================
classes = [
    "Healthy",
    "Mild DR",
    "Moderate DR",
    "Proliferate DR",
    "Severe DR"
]

# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Load MobileNetV3-Large
# =========================
model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    5
)

# Load trained model
checkpoint = torch.load(
    "models/drishti_mobilenet.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print("Drishti Edge MobileNet model loaded successfully!")
print("Using device:", device)


@app.get("/")
def home():
    return {
        "message": "Drishti Edge Backend is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:
        return {
            "success": False,
            "message": "Invalid image file"
        }

    # =========================
    # Preprocess image
    # =========================
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # =========================
    # AI Prediction
    # =========================
    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, 1
        )

    predicted_class = classes[predicted.item()]
    confidence_value = round(
        confidence.item(),
        2
    )

    return {
        "success": True,
        "filename": file.filename,
        "predicted_condition": predicted_class,
        "confidence": confidence_value,
        "message": "AI screening completed"
    }