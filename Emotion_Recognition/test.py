# test.py
import torch
from model import EmotionSwin  # Swin 모델 정의
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # RGB 정규화
])

test_dataset = datasets.ImageFolder('Testfilepath', transform=transform)
print(test_dataset[0])
test_loader = DataLoader(test_dataset, batch_size=32)

# 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. 모델 로드
model = EmotionSwin(num_classes=7).to(device)
model.load_state_dict(torch.load('emotion_swin_last.pth', map_location=device)) #last -> 30epoch
model.eval()

# 4. 평가 모드 (정확도 계산)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100. * correct / total
print(f"✅ 테스트 정확도: {accuracy:.2f}%")