from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance, ImageStat
import torch
from torchvision import transforms

# 얼굴 탐지기 초기화
mtcnn = MTCNN(keep_all=False, device='cpu')  # 단일 얼굴만

def auto_brightness(image, target_mean=200):
    """
    현재 이미지의 밝기 평균을 측정해서,
    target_mean(예: 130)에 맞게 밝기 비율을 조정해주는 함수.
    """
    stat = ImageStat.Stat(image)
    mean = stat.mean[0]  # 흑백 이미지일 때는 채널이 1개

    # 밝기 보정 비율 계산
    brightness_factor = target_mean / (mean + 1e-5)

    # 너무 과한 보정은 방지 (안정화 범위 지정)
    brightness_factor = max(0.7, brightness_factor)

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

# 이미지 경로 (여기에 너의 이미지 경로 넣어줘)
image_path = '/Users/parksungyong/Desktop/Emotion_Recognition/good.png'
image = Image.open(image_path).convert("RGB")

image = auto_brightness(image, target_mean=305)


# 얼굴 crop
face = mtcnn(image)  # 결과: torch.Tensor [3, H, W]

if face is not None:
    # ⬇️ 전처리: Grayscale + Resize(48x48)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224))
    ])

    # 이미지 값 스케일링
    face = (face * 255).clamp(0, 255).byte()

    # Tensor → PIL 이미지로 변환 후 전처리
    face_pil = transforms.ToPILImage()(face)
    face_gray_resized = transform(face_pil)

    # 저장
    face_gray_resized.save("face_gray_48x48.jpg")
    print("✅ 얼굴 crop + 흑백 + 48x48 저장 완료: face_gray_48x48.jpg")

else:
    print("❌ 얼굴을 찾을 수 없습니다.")