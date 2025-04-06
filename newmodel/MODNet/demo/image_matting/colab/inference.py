import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
import torchvision.transforms as transforms


def preprocess(image, ref_size=512):
    """
    전경 이미지를 모델 입력 크기에 맞춰 전처리하고,
    원본 이미지 크기를 반환합니다.
    """
    im = np.asarray(image)

    # 흑백 또는 4채널이면 3채널 RGB로 변환
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, :3]

    im = Image.fromarray(im)
    im_transform = transforms.Compose([
        transforms.ToTensor(),  # (H,W,C) -> (C,H,W), [0,255] -> [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    im_tensor = im_transform(im).unsqueeze(0)  # (1,3,H,W)

    _, _, im_h, im_w = im_tensor.shape
    original_size = (im_h, im_w)

    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh, im_rw = im_h, im_w

    im_rw -= im_rw % 32
    im_rh -= im_rh % 32

    im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode='area')
    im_np = im_tensor.detach().cpu().numpy()
    return im_np, original_size


def postprocess(matte_np, original_size):
    """
    모델 출력(알파 마스크)을 원본 이미지 크기로 복원합니다.
    """
    matte_t = torch.tensor(matte_np)  # (1,1,H,W)
    matte_t = F.interpolate(matte_t, size=original_size, mode='area')
    matte_np = matte_t[0, 0].cpu().numpy()
    matte_np = np.clip(matte_np, 0, 1)
    return matte_np


def alpha_composite(fg_image, matte_np, bg_image):
    """
    알파 마스크(matte_np)를 이용해 전경(fg)과 배경(bg)을 합성합니다.
    - fg, bg: PIL Image (RGB)
    - matte_np: (H, W) 배열, 값 범위 [0,1]
    """
    fg = np.array(fg_image).astype(np.float32)
    bg = np.array(bg_image).astype(np.float32)

    h_fg, w_fg = fg.shape[:2]
    bg = np.array(Image.fromarray(bg.astype(np.uint8)).resize((w_fg, h_fg), Image.BILINEAR))

    if matte_np.shape[0] != h_fg or matte_np.shape[1] != w_fg:
        matte_np = np.array(
            Image.fromarray((matte_np * 255).astype(np.uint8)).resize((w_fg, h_fg), Image.BILINEAR)
        ) / 255.0

    matte_3ch = matte_np[..., None]
    comp = fg * matte_3ch + bg * (1 - matte_3ch)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return Image.fromarray(comp)


def emotion_to_color(emotion_idx):
    """
    감정 코드(0~6)에 따른 RGB 색상 매핑 (심리적 느낌 기반)
      0: Angry     → 어두운 붉은색 (분노, 공격적)
      1: Disgust   → 올리브 그린 (역겨움, 거부감)
      2: Fear      → 다크 슬레이트 블루 (공포, 긴장)
      3: Happy     → 골든 옐로우 (행복, 따뜻함)
      4: Neutral   → 라이트 슬레이트 그레이 (중립, 균형)
      5: Sad       → 스틸 블루 (슬픔, 차분함)
      6: Surprise  → 핫 핑크 (놀람, 활기)
    """
    color_map = {
        0: (183, 28, 28),
        1: (85, 107, 47),
        2: (72, 61, 139),
        3: (255, 215, 0),
        4: (119, 136, 153),
        5: (70, 130, 180),
        6: (255, 105, 180)
    }
    return color_map.get(emotion_idx, (0, 0, 0))

def emotion_explain(num):
    if num == 0:
        emotion = 'angry'
    if num == 1:
        emotion = 'Disgust'
    if num == 2:
        emotion = 'Fear'
    if num == 3:
        emotion = 'Happy'
    if num == 4:
        emotion = 'Neutral'
    if num == 5:
        emotion = 'Sad'
    if num == 6:
        emotion = 'Surprise'
    return emotion
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='전경(인물) 이미지 경로')
    parser.add_argument('--bg-code', type=int, required=True, help='감정에 따른 배경 색상 코드 (0~6)')
    parser.add_argument('--ckpt-path', type=str, required=True, help='ONNX 모델 경로 (modnet.onnx)')
    args = parser.parse_args()

    # 1) 전경 이미지 로딩 + EXIF 자동 회전
    fg_image = Image.open(args.input_path)
    fg_image = ImageOps.exif_transpose(fg_image).convert('RGB')

    # 2) 전처리
    input_np, original_size = preprocess(fg_image)

    # 3) ONNX 모델 로딩 & 추론
    ort_sess = ort.InferenceSession(args.ckpt_path)
    input_name = ort_sess.get_inputs()[0].name
    matte_out = ort_sess.run(None, {input_name: input_np})
    matte_np = matte_out[0]

    # 4) 마스크 후처리 (원본 크기로 복원)
    matte_np = postprocess(matte_np, original_size)

    # 5) 배경 필터 적용: 원본 이미지를 배경으로 사용하고, 감정 색상 필터 오버레이
    bg_color = emotion_to_color(args.bg_code)
    print(f"[Info] 감정 코드 {emotion_explain(args.bg_code)} → 필터 색상 {bg_color}")
    overlay = Image.new("RGB", fg_image.size, color=bg_color)
    filter_strength = 0.4  # 필터 강도 (0: 원본, 1: 완전 단색)
    filtered_bg = Image.blend(fg_image, overlay, alpha=filter_strength)

    # 6) 합성: MODNet 알파 마스크를 활용하여 전경과 필터 적용 배경 합성
    comp_image = alpha_composite(fg_image, matte_np, filtered_bg)

    # 7) 저장: 입력 파일 이름 기반으로 "output" 폴더에 저장
    os.makedirs("output", exist_ok=True)
    input_filename = os.path.basename(args.input_path)
    output_filename = os.path.splitext(input_filename)[0] + "_replaced" + os.path.splitext(input_filename)[1]
    output_full_path = os.path.join("output", output_filename)
    comp_image.save(output_full_path)
    print(f"[Done] saved to {output_full_path}")


if __name__ == '__main__':
    main()
