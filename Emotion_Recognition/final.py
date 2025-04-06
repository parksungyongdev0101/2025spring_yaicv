import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from model import EmotionSwin  # Swin 모델 정의

# 감정 라벨 정의
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise",
}

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
model = EmotionSwin(num_classes=7).to(device)
model.load_state_dict(torch.load('emotion_swin_last.pth', map_location=device))
model.eval()

# 이미지 경로 (48x48 grayscale 이미지)
image_path = 'face_gray_48x48.jpg'

# 🔄 전처리: 1채널 → 3채널 복제 → Resize → Tensor → Normalize
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),         # 흑백 → RGB 채널 복제
    transforms.Resize((224, 224)),                       # Swin 입력 크기 맞춤
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)      # [-1, 1] 범위 정규화
])

# 이미지 불러오기 & 전처리
image = Image.open(image_path).convert('L')
image = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# 예측
with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()

# 출력
print(f"🧠 감정 예측 결과: {emotion_labels[predicted]} ({confidence * 100:.2f}%)")