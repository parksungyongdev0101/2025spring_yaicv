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
    추후 알파 마스크를 원본 크기로 복원하기 위해 (original_height, original_width)도 함께 반환.
    """
    # PIL -> NumPy
    im = np.asarray(image)

    # 만약 흑백/4채널이면 3채널 RGB로 맞춤
    if len(im.shape) == 2:  # H,W
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, :3]

    # 다시 PIL로 변환
    im = Image.fromarray(im)

    # PyTorch transforms
    im_transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1], (H,W,C) -> (C,H,W)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    im_tensor = im_transform(im).unsqueeze(0)  # (1,3,H,W)

    # 이미지 크기
    _, _, im_h, im_w = im_tensor.shape
    original_size = (im_h, im_w)

    # 모델 권장 크기(ref_size=512)에 맞춰 resize(너비높이 비율 유지)
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    # 32 배수로 맞춤 (MODNet 구조상)
    im_rw -= im_rw % 32
    im_rh -= im_rh % 32

    # area interpolation
    im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode='area')

    # NumPy 변환
    im_np = im_tensor.detach().cpu().numpy()
    return im_np, original_size


def postprocess(matte_np, original_size):
    """
    모델 출력(알파 마스크)을 원본 이미지 크기(original_size)로 복원.
    """
    matte_t = torch.tensor(matte_np)  # (1,1,H,W)
    # area 업샘플링으로 원본 크기로
    matte_t = F.interpolate(matte_t, size=original_size, mode='area')
    # shape: (1,1,H_original,W_original)
    matte_np = matte_t[0, 0].cpu().numpy()  # (H_original,W_original)
    # 값 범위 [0,1]로 가정
    matte_np = np.clip(matte_np, 0, 1)
    return matte_np


def alpha_composite(fg_image, matte_np, bg_image):
    """
    알파 마스크(matte_np) 이용해 전경(fg)과 배경(bg)을 합성.
    - fg, bg: PIL Image(RGB)
    - matte_np: shape(H,W), 값 범위 [0,1]
    """
    # 넘파이 변환
    fg = np.array(fg_image).astype(np.float32)
    bg = np.array(bg_image).astype(np.float32)

    # 전경, 배경 크기 맞추기
    h_fg, w_fg = fg.shape[:2]
    bg = np.array(Image.fromarray(bg.astype(np.uint8)).resize((w_fg, h_fg), Image.BILINEAR))

    # 알파 마스크도 전경 크기와 일치 확인
    if matte_np.shape[0] != h_fg or matte_np.shape[1] != w_fg:
        matte_np = np.array(
            Image.fromarray((matte_np*255).astype(np.uint8)).resize((w_fg,h_fg), Image.BILINEAR)
        ) / 255.0

    # 합성
    matte_3ch = matte_np[..., None]  # (H,W,1)
    comp = fg * matte_3ch + bg * (1 - matte_3ch)
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    return Image.fromarray(comp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='전경(인물) 이미지 경로')
    parser.add_argument('--background-path', type=str, required=True, help='배경 이미지 경로')
    parser.add_argument('--ckpt-path', type=str, required=True, help='ONNX 모델 경로 (modnet.onnx)')
    parser.add_argument('--output-path', type=str, required=True, help='최종 합성 결과 경로')
    args = parser.parse_args()

    # 1) 전경 이미지 로딩 + EXIF 자동 회전
    fg_image = Image.open(args.input_path)
    fg_image = ImageOps.exif_transpose(fg_image).convert('RGB')  # 회전 처리

    # 2) 전처리
    input_np, original_size = preprocess(fg_image)

    # 3) ONNX 모델 로딩 & 추론
    ort_sess = ort.InferenceSession(args.ckpt_path)
    # onnxruntime 입력
    input_name = ort_sess.get_inputs()[0].name
    # 추론
    matte_out = ort_sess.run(None, {input_name: input_np})  # list of outputs
    matte_np = matte_out[0]  # shape (1,1,H_infer,W_infer)

    # 4) 마스크 후처리 (원본 크기로 복원)
    matte_np = postprocess(matte_np, original_size)  # shape(H,W), [0,1]

    # 5) 배경 이미지 로딩
    bg_image = Image.open(args.background_path)
    bg_image = ImageOps.exif_transpose(bg_image).convert('RGB')

    # 6) 합성
    comp_image = alpha_composite(fg_image, matte_np, bg_image)

    # 7) 저장
    comp_image.save(args.output_path)
    print(f"[Done] saved to {args.output_path}")


if __name__ == '__main__':
    main()
