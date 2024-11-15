import torch
from dotmap import DotMap
from models.simclr import SimCLR

# 예제 encoder_params 설정
encoder_params = DotMap({
    'embedding_dim': 128,
    'pretrained': False,
    'use_norm': True
})

# SimCLR 모델 초기화
temperature = 0.5
margin = 1.0
model = SimCLR(encoder_params=encoder_params, temperature=temperature, margin=margin)

# 예제 입력 생성
batch_size = 2
channels = 3
height = 224
width = 224

img_anchor = torch.randn(batch_size, channels, height, width)
img_positive = torch.randn(batch_size, channels, height, width)
img_negative = torch.randn(batch_size, channels, height, width)

# 모델에 입력을 통과시켜 출력 확인
anchor_proj, positive_proj, negative_proj = model(img_anchor, img_positive, img_negative)
print("Anchor Projection:", anchor_proj)
print("Positive Projection:", positive_proj)
print("Negative Projection:", negative_proj)

# Triplet loss 계산 확인
loss = model.compute_loss(anchor_proj, positive_proj, negative_proj)
print("Triplet Loss:", loss.item())
