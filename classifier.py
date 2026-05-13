from pathlib import Path
from time import sleep

from ultralytics import YOLO



configs = {
    "epochs": 200,
    # "flipud": 0.5,
    "translate": 0.3,
    "hsv_s": 0,
    "hsv_h": 0,
    # "dropout": 0.4,
    "erasing": 0,
    # "weight_decay": 0.1,
    "optimizer": "SGD",
    "cos_lr": True
    

}

def main():
    model = YOLO("yolo26n-cls")
    results = model.train(data="augmented_dataset", **configs)

if __name__ == "__main__":
    main()
    exit()
# #
# #
#
# model = YOLO("runs/classify/train-22/weights/best.pt")  # load a custom model
#
# # metrics = model.val()  # no arguments needed, dataset and settings remembered
# # print(metrics.top1)  # top1 accuracy
# # print(metrics.top5)  # top5 accuracy
# images = Path("data/for_test/results/protrusion/case 193").glob("*.jpg")
#
# for img_path in images:
#     results = model(img_path)  # predict on an image
#     # print(results[0])
#     # sleep(5)
# #
# # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader

# # device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------
# # Transforms (with augmentation)
# # -------------------------
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # -------------------------
# # Dataset & Loaders
# # -------------------------
# train_dataset = datasets.ImageFolder("data_model_split/train", transform=train_transform)
# val_dataset   = datasets.ImageFolder("data_model_split/val", transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=16)

# # -------------------------
# # Model (ResNet50 pretrained)
# # -------------------------
# model = models.resnet50(pretrained=True)

# num_classes = len(train_dataset.classes)
# model.fc = nn.Linear(model.fc.in_features, num_classes)

# model = model.to(device)

# # -------------------------
# # Optional: Freeze backbone (recommended for stability)
# # -------------------------
# for param in model.parameters():
#     param.requires_grad = False

# for param in model.fc.parameters():
#     param.requires_grad = True

# # -------------------------
# # Loss, Optimizer, Scheduler
# # -------------------------
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # train only final layer first

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# # -------------------------
# # Training Loop
# # -------------------------
# epochs = 30
# best_val_acc = 0.0

# for epoch in range(epochs):
#     # ---- Train ----
#     model.train()
#     total_loss = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     scheduler.step()  # ✅ correct placement

#     avg_loss = total_loss / len(train_loader)

#     # ---- Validation ----
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)

#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#     val_acc = 100 * correct / total

#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

#     # ---- Save Best Model ----
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_model.pth")
#         print(f"✅ Saved best model ({val_acc:.2f}%)")

# # -------------------------
# # Optional: Unfreeze and fine-tune entire model
# # -------------------------
# print("\n🔄 Fine-tuning entire model...\n")

# for param in model.parameters():
#     param.requires_grad = True

# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# for epoch in range(10):  # fine-tune for a few more epochs
#     model.train()
#     total_loss = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     scheduler.step()

#     avg_loss = total_loss / len(train_loader)

#     # validation
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)

#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#     val_acc = 100 * correct / total

#     print(f"[FT] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_model.pth")
#         print(f"✅ Updated best model ({val_acc:.2f}%)")

# print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")