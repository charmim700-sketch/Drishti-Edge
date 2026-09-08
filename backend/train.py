import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import MobileNet_V3_Large_Weights

# =========================
# 1. Dataset path
# =========================
DATASET_PATH = r"C:\Users\charmi.M\Downloads\archive"

# =========================
# 2. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 3. Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# 4. Load dataset
# =========================
dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# =========================
# 5. Train / Validation split
# =========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# =========================
# 6. Load pretrained MobileNetV3
# =========================
weights = MobileNet_V3_Large_Weights.DEFAULT

model = models.mobilenet_v3_large(weights=weights)

# Freeze pretrained layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classifier
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    5
)

model = model.to(device)

# =========================
# 7. Loss and optimizer
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# =========================
# 8. Training
# =========================
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

# =========================
# 9. Save trained model
# =========================
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    },
    "models/drishti_mobilenet.pth"
)

print("\nTraining completed!")
print("Model saved at: models/drishti_mobilenet.pth")