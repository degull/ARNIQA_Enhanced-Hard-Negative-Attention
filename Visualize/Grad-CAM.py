""" import sys
import os
sys.path.append(os.path.abspath("E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA"))

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dotmap import DotMap
from models.simclr import SimCLR
import torchvision.transforms as T

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
            layer.register_full_backward_hook(self.save_gradient(layer_name))

    def save_activation(self, layer_name):
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook_fn

    def save_gradient(self, layer_name):
        def hook_fn(module, grad_in, grad_out):
            self.gradients[layer_name] = grad_out[0].detach()
        return hook_fn

    def generate_cam(self, layer_name, input_image_shape):
        # 그래디언트의 차원 확인
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]

        print(f"[DEBUG] Gradients shape for layer {layer_name}: {gradients.shape}")
        print(f"[DEBUG] Activations shape for layer {layer_name}: {activations.shape}")

        # 4차원이 아닐 경우 에러 처리
        if gradients.dim() != 4 or activations.dim() != 4:
            raise ValueError(f"Expected 4D tensors, but got gradients {gradients.shape} and activations {activations.shape} for layer {layer_name}")

        # 평균값으로 그래디언트 압축
        gradients = gradients.mean(dim=[2, 3], keepdim=True)

        # Grad-CAM 생성
        cam = (activations * gradients).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # Zero division 방지

        # 입력 이미지 크기로 조정
        cam_resized = cv2.resize(cam.squeeze().cpu().numpy(), (input_image_shape[1], input_image_shape[0]))
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
image_path = "E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA/assets/03.png"
input_image = cv2.imread(image_path)
if input_image is None:
    raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

# 이미지 전처리
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Grad-CAM 생성 및 시각화
for layer_name in target_layers:
    try:
        cam = grad_cam(input_tensor, layer_name, input_image.shape)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 처리하므로 RGB로 변환
        overlay = cv2.addWeighted(input_image.astype(np.float32) / 255, 0.5, heatmap.astype(np.float32) / 255, 0.5, 0)

        plt.imshow(overlay)
        plt.title(f"Grad-CAM at {layer_name}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"[ERROR] Unable to generate Grad-CAM for layer {layer_name}: {e}")
 """


####

import sys
import os
sys.path.append(os.path.abspath("E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA"))

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dotmap import DotMap
from models.simclr import SimCLR
import torchvision.transforms as T

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
            layer.register_full_backward_hook(self.save_gradient(layer_name))

    def save_activation(self, layer_name):
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook_fn

    def save_gradient(self, layer_name):
        def hook_fn(module, grad_in, grad_out):
            self.gradients[layer_name] = grad_out[0].detach()
        return hook_fn

    def generate_cam(self, layer_name, input_image_shape):
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]

        print(f"[DEBUG] Gradients shape for layer {layer_name}: {gradients.shape}")
        print(f"[DEBUG] Activations shape for layer {layer_name}: {activations.shape}")

        if gradients.dim() != 4 or activations.dim() != 4:
            raise ValueError(f"Expected 4D tensors, but got gradients {gradients.shape} and activations {activations.shape} for layer {layer_name}")

        gradients = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (activations * gradients).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_resized = cv2.resize(cam.squeeze().cpu().numpy(), (input_image_shape[1], input_image_shape[0]))
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
model_with_attention = SimCLR(encoder_params=encoder_params, temperature=0.5)
model_with_attention.to(device)

# SimCLR 모델에 단일 입력용 forward 추가
def forward_single(self, img_anchor):
    anchor_features = self.encoder(img_anchor)[0]
    anchor_proj = self.projector(anchor_features)
    if self.self_attention is not None:
        anchor_proj = self.self_attention(anchor_proj)
    return anchor_proj

SimCLR.forward_single = forward_single

# Grad-CAM 생성 및 시각화 함수
def save_and_plot(cam, layer_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(input_image.astype(np.float32) / 255, 0.5, heatmap.astype(np.float32) / 255, 0.5, 0)

    plt.imshow(overlay)
    plt.title(f"Grad-CAM at {layer_name}")
    plt.axis('off')

    save_path = f"E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA/Visualize/results/grad_cam_{layer_name}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved Grad-CAM visualization for {layer_name} at {save_path}")


# 입력 이미지 로드 및 전처리
image_path = "E:/ARNIQA - Enhanced-Hard-Negative-Attention/ARNIQA/assets/03.png"
input_image = cv2.imread(image_path)
if input_image is None:
    raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(input_image).unsqueeze(0).to(device)

# 1. Attention Mechanism이 없는 모델
print("[INFO] Generating Grad-CAM for model without attention...")
model_no_attention = SimCLR(encoder_params=encoder_params, temperature=0.5)
model_no_attention.self_attention = None
model_no_attention.cross_attention = None
model_no_attention.to(device)

grad_cam_no_attention = GradCAM(model_no_attention, target_layers=['encoder.model.7'])
cam_no_attention = grad_cam_no_attention(input_tensor, 'encoder.model.7', input_image.shape)
save_and_plot(cam_no_attention, 'encoder.model.7_no_attention')

# 2. Grad-CAM의 다른 레이어에서의 시각화
print("[INFO] Generating Grad-CAM for different layers...")
grad_cam_with_attention = GradCAM(model_with_attention, target_layers=['encoder.model.0', 'encoder.model.4', 'encoder.model.7'])
for layer in ['encoder.model.0', 'encoder.model.4', 'encoder.model.7']:
    cam = grad_cam_with_attention(input_tensor, layer, input_image.shape)
    save_and_plot(cam, layer)

# 3. Self-Attention vs Cross-Attention
print("[INFO] Generating Grad-CAM for Self-Attention and Cross-Attention...")
grad_cam_attention = GradCAM(model_with_attention, target_layers=['self_attention', 'cross_attention'])
for layer in ['self_attention', 'cross_attention']:
    cam = grad_cam_attention(input_tensor, layer, input_image.shape)
    save_and_plot(cam, layer)
