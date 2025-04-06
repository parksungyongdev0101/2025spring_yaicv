import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
    
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # RGB 정규화
])

train_dataset = datasets.ImageFolder('RAF dataset/train', transform=transform)
test_dataset = datasets.ImageFolder('RAF dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# train_swin.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🔧 하이퍼파라미터
num_epochs = 30
batch_size = 32
learning_rate = 3e-4
num_classes = 7

# 💻 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 모델 정의
model = EmotionSwin(num_classes=num_classes).to(device)

# 🎯 손실 함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# ⏳ CosineAnnealingWarmRestarts 스케줄러
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# 📈 시각화를 위한 값 저장
loss_history = []
acc_history = []
lr_history = []

# 🔁 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    # 🎯 Epoch 평균 저장
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)
    lr_history.append(scheduler.get_last_lr()[0])

    print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {lr_history[-1]:.6f}")

    # 🔄 스케줄러 업데이트
    scheduler.step()

# 💾 모델 저장
torch.save(model.state_dict(), 'emotion_swin.pth')
print("📦 모델 저장 완료: emotion_swin.pth")

# 📊 학습 곡선 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss', marker='o')
plt.plot(acc_history, label='Accuracy', marker='x')
plt.title('Training Loss & Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lr_history, label='Learning Rate', color='orange')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.grid(True)

plt.tight_layout()
plt.savefig("swin_train_curve.png")
plt.show()