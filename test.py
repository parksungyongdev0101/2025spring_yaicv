import cv2
import numpy as np
from deepface import DeepFace

# 1. 이미지를 로드합니다.
img = cv2.imread("messi.jpg")

# 2. DeepFace로 감정 분석 (얼굴이 안 잡힐 수 있으니 enforce_detection=False 고려)
result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
dominant_emotion = result[0]["dominant_emotion"]
print("감정:", dominant_emotion)

# 3. 감정 → 특정 색상 매핑 (마음에 드는 색상으로 변경 가능)
emotion_colors = {
    # B, G, R
    "happy":    (0, 255, 255),  # 노랑
    "sad":      (255, 0, 0),    # 파랑
    "angry":    (0, 0, 255),    # 빨강
    "surprise": (0, 165, 255),  # 주황
    "neutral":  (128, 128, 128),# 회색
    "fear":     (130, 0, 75),   # 임의 보라톤 (원하는 값으로 조정)
    "disgust":  (0, 128, 0)     # 녹색
}


#overlay_color = emotion_colors.get(dominant_emotion, (128, 128, 128))  # 해당하지 않을 경우 회색
overlay_color = emotion_colors["disgust"]
# 4. OpenCV Haar Cascade로 얼굴 영역 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 5. 검출된 얼굴 영역에 색 필터 합성
for (x, y, w, h) in faces:
    # 얼굴만 ROI로 추출
    face_roi = img[y:y+h, x:x+w]

    # ROI 크기만큼 '단색' 이미지 생성
    overlay = np.full_like(face_roi, overlay_color, dtype=np.uint8)

    # 얼굴 ROI와 overlay 이미지를 합성하여 반투명 효과 주기
    alpha = 0.4  # 투명도 (0.0 ~ 1.0)
    blended_face = cv2.addWeighted(face_roi, 1 - alpha, overlay, alpha, 0)

    # 합성된 얼굴 이미지를 원본에 다시 반영
    img[y:y+h, x:x+w] = blended_face

# 6. 결과 저장
cv2.imwrite("output_colored.jpg", img)
print("결과 이미지가 output_colored.jpg로 저장되었습니다.")
