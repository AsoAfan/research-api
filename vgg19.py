import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

# ======================
# Config
# ======================
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 60
IMG_SIZE = 224
NUM_CLASSES = 4  # change
DATA_DIR = "augmented_dataset_split" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Self-Attention
# ======================
class SeqSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(attn)

        return torch.bmm(attn, V)

# ======================
# Model
# ======================
class VGG19_Attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # First ~14 conv layers
        self.features = nn.Sequential(*list(vgg.features.children())[:28])

        # Freeze early layers
        for param in self.features.parameters():
            param.requires_grad = False

        self.attention = SeqSelfAttention(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # (B,512,7,7)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B,49,512)

        x = self.attention(x)
        x = x.mean(dim=1)

        return self.classifier(x)

# ======================
# Data
# ======================
train_transform = transforms.Compose([
    # transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    # transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================
# Setup
# ======================
model = VGG19_Attention(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ======================
# Accuracy Function
# ======================
def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct

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

    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{epoch}_best_model.pth")
        print(f"weight are saved in {epoch}_best_model.pth")

print(f"Best Validation Accuracy: {best_val_acc:.4f}")