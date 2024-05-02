import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 경로
data_dir = '/content/drive/MyDrive/Colab Notebooks/Facenet/database'

# 데이터셋 로드
dataset = ImageFolder(root=data_dir, transform=transform)

# 클래스 이름 확인
class_names = dataset.classes
num_classes = len(class_names)
print(f'클래스 이름: {class_names}, 클래스 수: {num_classes}')

# 데이터로더 생성
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 정의
class FaceSimilarityModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceSimilarityModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

# 모델 초기화 및 손실 함수, 최적화기 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceSimilarityModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


# 학습된 모델 저장
save_path = '/content/drive/MyDrive/face_similarity_model.pth'
torch.save(model.state_dict(), save_path)
