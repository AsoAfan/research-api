import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
MODEL = "models/vgg/2_best_vgg19.pth"

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


class VGG19_Attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg19(weights=None)
        self.features = nn.Sequential(*list(vgg.features.children())[:28])
        self.attention = SeqSelfAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        B, C, _, _ = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.attention(x)
        x = x.mean(dim=1)
        return self.classifier(x)


tf = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder("augmented_dataset_split/val", transform=tf)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

model = VGG19_Attention(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
model.eval()

print(f"Evaluating {MODEL}")

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in loader:
        preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
