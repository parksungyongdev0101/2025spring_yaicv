import torch
from model import BiSeNet  # face-parsing.PyTorch의 모델 정의

# 모델 초기화
n_classes = 19
net = BiSeNet(n_classes=n_classes)

# CPU에 로드되도록 설정
ckpt_path = '79999_iter.pth'
net.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
net.eval()

# ONNX export
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(net, dummy_input, "face_parsing.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)

print("✅ ONNX export 완료: face_parsing.onnx")
