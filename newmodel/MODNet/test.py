from PIL import Image
import numpy as np

def replace_background(foreground_path, matte_path, background_path, output_path):
    # 이미지 로딩
    fg = Image.open(foreground_path).convert('RGB')
    matte = Image.open(matte_path).convert('L')  # 흑백 마스크
    bg = Image.open(background_path).convert('RGB')

    # 사이즈 통일
    fg = np.array(fg)
    bg = np.array(bg.resize((fg.shape[1], fg.shape[0])))
    matte = np.array(matte.resize((fg.shape[1], fg.shape[0]))) / 255.0

    # 알파 채널 적용
    matte = matte[:, :, None]  # (H, W) -> (H, W, 1)
    comp = fg * matte + bg * (1 - matte)
    comp = comp.astype(np.uint8)

    # 이미지 저장
    Image.fromarray(comp).save(output_path)

# 예시 사용
replace_background(
    foreground_path='son.jpg',
    matte_path='matte.png',
    background_path='nonsan.jpg',
    output_path='replaced.png'
)
