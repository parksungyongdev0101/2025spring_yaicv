import torch
import torch.nn as nn
import timm  # pip install timm

class EmotionSwin(nn.Module):
    def __init__(self, num_classes=7, model_name='swin_tiny_patch4_window7_224'):
        super(EmotionSwin, self).__init__()

        # Swin Transformer 전체 모델 불러오기 (분류용 head 포함)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)  # 출력 shape: [B, num_classes]