

import sys
import os
sys.path.append(os.path.abspath("E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA"))

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dotmap import DotMap
from models.simclr import SimCLR

# Grad-CAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hook_layers()

    def hook_layers(self):
        for layer_name in self.target_layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.save_activation(layer_name))
            layer.register_backward_hook(self.save_gradient(layer_name))

    def save_activation(self, layer_name):
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook_fn

    def save_gradient(self, layer_name):
        def hook_fn(module, grad_in, grad_out):
            self.gradients[layer_name] = grad_out[0].detach()
        return hook_fn

    def generate_cam(self, layer_name, input_image_shape):
        gradients = self.gradients[layer_name].mean(dim=[2, 3], keepdim=True)
        cam = (self.activations[layer_name] * gradients).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.squeeze().cpu().numpy()
        
        # input_image 크기로 cam을 조정합니다.
        cam_resized = cv2.resize(cam, (input_image_shape[1], input_image_shape[0]))
        return cam_resized

    def __call__(self, x, layer_name, input_image_shape):
        output = self.model.forward_single(x)
        self.model.zero_grad()
        class_loss = output[:, output.argmax(dim=1)].sum()
        class_loss.backward()
        return self.generate_cam(layer_name, input_image_shape)

# 모델 설정 및 Grad-CAM 대상 레이어 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SimCLR 모델 초기화 파라미터 설정
encoder_params = DotMap({
    'embedding_dim': 128,
    'pretrained': False,
    'use_norm': True
})
model = SimCLR(encoder_params=encoder_params, temperature=0.5)
model.to(device)

# SimCLR 모델에 단일 입력용 forward 추가
def forward_single(self, img_anchor):
    anchor_features = self.encoder(img_anchor)[0]
    anchor_proj = self.self_attention(self.projector(anchor_features))
    return anchor_proj

SimCLR.forward_single = forward_single

# Grad-CAM 대상 레이어 설정
target_layers = ['encoder.model.7', 'self_attention', 'cross_attention']
grad_cam = GradCAM(model, target_layers)

# 입력 이미지 로드 및 전처리
image_path = "E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA/assets/01.png"
input_image = cv2.imread(image_path)
if input_image is None:
    raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

input_image = cv2.resize(input_image, (224, 224))
input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

# Grad-CAM 생성 및 시각화
for layer_name in target_layers:
    cam = grad_cam(input_tensor, layer_name, input_image.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result = heatmap + np.float32(input_image) / 255
    result = result / np.max(result)
    
    plt.imshow(result)
    plt.title(f"Grad-CAM at {layer_name}")
    plt.axis('off')
    plt.show()
