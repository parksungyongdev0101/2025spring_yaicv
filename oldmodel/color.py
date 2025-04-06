import cv2
import numpy as np
import onnxruntime as ort

# 색상 설정
color_map = {
    0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 255),
    3: (0, 255, 0), 4: (255, 0, 0), 5: (255, 0, 255)
}
num = int(input("색상 번호 (0~5): "))
overlay_color = color_map.get(num, (128, 128, 128))

# 모델 로딩
session = ort.InferenceSession("face_parsing.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 이미지 불러오기
image = cv2.imread("ko.jpg")
if image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")
h, w, _ = image.shape

# 입력 전처리
img_input = cv2.resize(image, (512, 512))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img_input = img_input.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# 추론
outputs = session.run(None, {input_name: img_input})
parsing = np.argmax(outputs[0], axis=1)[0]

# ✅ 피부 레이블만 사용
face_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
mask = np.isin(parsing, face_labels).astype(np.uint8) * 255
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# 마스크 후처리
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# 색상 오버레이
overlay = np.full_like(image, overlay_color, dtype=np.uint8)
output = np.where(mask[..., None] > 100,
                  cv2.addWeighted(image, 1 - 0.4, overlay, 0.4, 0),
                  image)

# 결과 출력 및 저장
cv2.imwrite("color.jpg", output)
cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
