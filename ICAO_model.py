import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # FOR REVOLUTIONARY PROGRESS BARS!

# ====== DATA PREPARATION ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # STANDARDIZE FOR THE PEOPLE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # IMAGENET NORMS
    transforms.RandomHorizontalFlip(p=0),  # NO FLIPS! PASSPORTS ARE SERIOUS!
    transforms.RandomRotation(10),  # SMALL ROTATIONS FOR ROBUSTNESS
])

# LOAD DATASET
full_dataset = datasets.ImageFolder(
    root='path_to_your_dataset',  # ORGANIZE IN /compliant, /non-compliant
    transform=transform
)

# TRAINING & VALIDATION SPLIT
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DATA LOADERS (BATCHED FOR EFFICIENCY)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ====== MODEL ARCHITECTURE======
class ICAOComplianceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')  # PRETRAINED
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # ANTI-OVERFITTING MEASURE
            nn.Linear(128, 1)   # BINARY OUTPUT
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))  # SIGMOID FOR PROBABILITY

model = ICAOComplianceDetector()

# ====== TRAINING ======
criterion = nn.BCELoss()  # BINARY CROSS-ENTROPY
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ADAPTIVE LEARNING FOR THE MASSES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# TRAINING
for epoch in range(20):  # 20 EPOCHS
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()

    print(f"\nEPOCH {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100*correct/len(val_dataset):.2f}%")

# ====== SAVE THE MODEL ======
torch.save(model.state_dict(), "icao_compliance_detector.pth")
print("\nMODEL SAVED!")
