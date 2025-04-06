import cv2
import numpy as np
import onnxruntime as ort

# (1) 세그멘테이션 모델 로딩
session = ort.InferenceSession("face_parsing.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# (2) 얼굴 이미지 불러오기
face_image = cv2.imread("son.png")  # 얼굴이 들어있는 원본 이미지
if face_image is None:
    raise FileNotFoundError("파일을 찾을 수 없습니다.")
h, w, _ = face_image.shape

# (3) 전처리
img_input = cv2.resize(face_image, (512, 512))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img_input = img_input.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# (4) 추론
outputs = session.run(None, {input_name: img_input})
parsing = np.argmax(outputs[0], axis=1)[0]

# (5) 얼굴 부분 레이블 정의
#   - 원하는 얼굴 부위만 선택해서 마스크에 포함
#   - 예) [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#   - 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#                13 : 입 14 : 목 16 : 옷
face_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]

# (6) 마스크 만들기
#   - 지정한 레이블(얼굴 부위)일 때만 255, 나머지는 0
mask = np.isin(parsing, face_labels).astype(np.uint8) * 255
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# (7) 마스크 후처리 (옵션: 경계 부드럽게)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.GaussianBlur(mask, (3, 3), 0)

# (8) 새 배경 이미지 로드
bg_image = cv2.imread("nonsan.jpg")  # 합성에 쓸 배경
if bg_image is None:
    raise FileNotFoundError("image.jpg 파일을 찾을 수 없습니다.")
bg_image = cv2.resize(bg_image, (w, h))

# (9) 합성
#   - 얼굴 마스크=1인 부분은 원본 얼굴 픽셀을, 나머지는 새 배경을 사용
mask_3ch = np.dstack([mask]*3)          # (h, w) -> (h, w, 3)
fg_face = face_image * (mask_3ch // 255)  # 얼굴 부분
bg_only = bg_image * (1 - (mask_3ch // 255))  # 배경 부분
result = fg_face + bg_only               # 픽셀 합

# (10) 저장 및 출력
cv2.imwrite("back.jpg", result)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
