import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix


MODEL_FILE = "best_resnet101.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet101_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet101(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("augmented_dataset_split/val", transform=tf)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

model = ResNet101_Model(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(f"models/resnet101/{MODEL_FILE}", map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in loader:
        preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
