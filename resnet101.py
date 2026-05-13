import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

# ======================
# Config
# ======================
BATCH_SIZE = 8
# LR = 0.001
LR = 0.02
COS_LR = False
# EPOCHS = 100
EPOCHS = 50
IMG_SIZE = 224
NUM_CLASSES = 4
DATA_DIR = "augmented_dataset_split"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model
# ======================
class ResNet101_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace FC
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ======================
# Data
# ======================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================
# Setup
# ======================
model = ResNet101_Model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.model.fc.parameters(),
                      lr=LR)
# optimizer = optim.Adam(model.model.fc.parameters(),
#                       lr=LR)

# ✅ Cosine Scheduler
if COS_LR:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )
else:
    scheduler = None

# ======================
# Accuracy
# ======================
def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item()

# ======================
# Training Loop
# ======================
best_val_acc = 0.0

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += compute_accuracy(outputs, labels)
        total += labels.size(0)

    train_acc = train_correct / total

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += compute_accuracy(outputs, labels)
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    # ✅ Step cosine AFTER epoch
    if COS_LR:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet101.pth")

    # ---- Gradual unfreeze (important) ----
    if epoch == 10:
        print("Unfreezing layer4...")

        for param in model.model.layer4.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR * 0.1,
            momentum=0.9
        )

        if COS_LR:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - epoch - 1,
                eta_min=1e-5
            )
        else:
            scheduler = None

print(f"Best Validation Accuracy: {best_val_acc:.4f}")