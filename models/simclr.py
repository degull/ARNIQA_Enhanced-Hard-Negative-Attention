""" import torch
import torch.nn as nn
from dotmap import DotMap
from scipy import stats
import sys
import os

# 필요한 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet
from models.attention_module import DistortionAttention, HardNegativeCrossAttention

class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()

        # ResNet Encoder와 projection layer
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        
        # projector의 첫 번째 Linear 레이어 입력 크기를 2048로 설정
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )

        self.self_attention = DistortionAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.cross_attention = HardNegativeCrossAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.temperature = temperature

        # 초기화 방법 설정 (xavier 초기화)
        self._initialize_weights()

    def _initialize_weights(self):
        # Sequential 내부의 Linear 레이어 각각 초기화
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, img_A, img_B):
        # 입력 차원 확인 [batch_size, num_crops, 3, H, W]
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()

        # num_crops 차원을 평탄화하여 encoder 입력 준비
        img_A_flat = img_A.view(-1, C, H, W)  # [batch_size * num_crops, C, H, W]
        img_B_flat = img_B.view(-1, C, H, W)  # [batch_size * num_crops, C, H, W]

        # encoder 통과
        proj_A = self.encoder(img_A_flat)
        proj_B = self.encoder(img_B_flat)

        # encoder에서 튜플 반환 시 첫 번째 요소만 사용
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]

        # [batch_size, num_crops, embedding_dim] 형식으로 다시 reshape
        proj_A = proj_A.view(batch_size, num_crops, -1)
        proj_B = proj_B.view(batch_size, num_crops, -1)

        # projector 적용하여 attention module 입력 형식 맞춤
        proj_A = self.projector(proj_A).view(batch_size, num_crops, -1)
        proj_B = self.projector(proj_B).view(batch_size, num_crops, -1)

        # Self-attention 적용 전의 평균 값 디버깅
        print(f"Before self-attention: proj_A mean = {proj_A.mean().item()}, proj_B mean = {proj_B.mean().item()}")

        # self-attention 및 cross-attention 적용
        proj_A = self.self_attention(proj_A)
        proj_B = self.self_attention(proj_B)
        
        # Self-attention 적용 후의 평균 값 디버깅
        print(f"After self-attention: proj_A mean = {proj_A.mean().item()}, proj_B mean = {proj_B.mean().item()}")
        
        cross_attn_output = self.cross_attention(proj_A, proj_B)

        # cross_attn_output을 [batch_size, num_crops, embedding_dim] 형식으로 reshape
        actual_batch_size = cross_attn_output[0].shape[0] // num_crops
        try:
            proj_A, proj_B = [x.view(actual_batch_size, num_crops, -1) for x in cross_attn_output[:2]]
        except RuntimeError:
            raise ValueError(f"크기 재구성 오류 발생: cross_attn_output의 실제 크기와 맞지 않음.")

        # 디버깅: attention 후 proj_A와 proj_B 차이 확인
        diff = torch.norm(proj_A - proj_B, dim=-1)
        print(f"proj_A와 proj_B 차이 (평균): {diff.mean().item()}")

        # 최종 출력 반환
        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        # contrastive loss 계산을 위한 차원 일치 확인
        proj_q = proj_q.view(-1, proj_q.shape[-1])
        proj_p = proj_p.view(-1, proj_p.shape[-1])

        assert proj_q.shape == proj_p.shape, \
            f"proj_q and proj_p shapes must match, got {proj_q.shape} and {proj_p.shape}"
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        # NT-Xent (Normalized temperature-scaled cross-entropy) loss 계산
        a_norm = torch.norm(a, dim=1, keepdim=True) + 1e-8  # 작은 값 추가
        a_cap = a / a_norm
        b_norm = torch.norm(b, dim=1, keepdim=True) + 1e-8  # 작은 값 추가
        b_cap = b / b_norm
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t()) / tau
        exp_sim_by_tau = torch.exp(sim)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1) - torch.diag(exp_sim_by_tau)

        # Anchor-positive similarity scores를 위한 slice 조정
        sum_of_rows = sum_of_rows[:a.shape[0]]

        numerators = torch.exp(torch.div(torch.nn.functional.cosine_similarity(a_cap, b_cap), tau))
        num_by_den = numerators / (sum_of_rows + 1e-8)  # 작은 값 추가
        return -torch.mean(torch.log(num_by_den + 1e-8))  # 작은 값 추가


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A_np = proj_A.detach().cpu().numpy().flatten()
    proj_B_np = proj_B.detach().cpu().numpy().flatten()
    srocc, _ = stats.spearmanr(proj_A_np, proj_B_np)
    plcc, _ = stats.pearsonr(proj_A_np, proj_B_np)
    
    print(f"SRCC: {srocc}, PLCC: {plcc}")
    return srocc, plcc


if __name__ == "__main__":
    # 모델 구조 확인을 위한 테스트 코드
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    model = SimCLR(encoder_params)

    # [batch_size, num_crops, C, H, W] 형식의 예시 데이터 생성
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # Forward pass 수행
    proj_A, proj_B = model(img_A, img_B)

    # SRCC, PLCC 계산
    calculate_srcc_plcc(proj_A, proj_B)

    # 출력 차원 확인
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
 """

""" 
# simclr 원본 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from scipy import stats
from models.resnet import ResNet
from models.attention_module import DistortionAttention, HardNegativeCrossAttention

class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.5):
        super(SimCLR, self).__init__()

        # ResNet Encoder와 projection layer
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        
        # projector의 첫 번째 Linear 레이어 입력 크기를 2048로 설정
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )

        self.self_attention = DistortionAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.cross_attention = HardNegativeCrossAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.temperature = temperature

        # 초기화 방법 설정 (xavier 초기화)
        self._initialize_weights()

    def _initialize_weights(self):
        # Sequential 내부의 Linear 레이어 각각 초기화
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, img_A, img_B):
        # 입력 차원 확인 [batch_size, num_crops, 3, H, W]
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()

        # num_crops 차원을 평탄화하여 encoder 입력 준비
        img_A_flat = img_A.view(-1, C, H, W)  # [batch_size * num_crops, C, H, W]
        img_B_flat = img_B.view(-1, C, H, W)  # [batch_size * num_crops, C, H, W]

        # encoder 통과
        proj_A = self.encoder(img_A_flat)
        proj_B = self.encoder(img_B_flat)

        # encoder에서 튜플 반환 시 첫 번째 요소만 사용
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]

        # [batch_size, num_crops, embedding_dim] 형식으로 다시 reshape
        proj_A = proj_A.view(batch_size, num_crops, -1)
        proj_B = proj_B.view(batch_size, num_crops, -1)

        # projector 적용하여 attention module 입력 형식 맞춤
        proj_A = self.projector(proj_A).view(batch_size, num_crops, -1)
        proj_B = self.projector(proj_B).view(batch_size, num_crops, -1)

        # Self-attention 적용 전의 평균 값 디버깅
        print(f"Before self-attention: proj_A mean = {proj_A.mean().item()}, proj_B mean = {proj_B.mean().item()}")

        # self-attention 및 cross-attention 적용
        proj_A = self.self_attention(proj_A)
        proj_B = self.self_attention(proj_B)
        
        # Self-attention 적용 후의 평균 값 디버깅
        print(f"After self-attention: proj_A mean = {proj_A.mean().item()}, proj_B mean = {proj_B.mean().item()}")
        
        cross_attn_output = self.cross_attention(proj_A, proj_B)

        # cross_attn_output을 [batch_size, num_crops, embedding_dim] 형식으로 reshape
        actual_batch_size = cross_attn_output[0].shape[0] // num_crops
        try:
            proj_A, proj_B = [x.view(actual_batch_size, num_crops, -1) for x in cross_attn_output[:2]]
        except RuntimeError:
            raise ValueError(f"크기 재구성 오류 발생: cross_attn_output의 실제 크기와 맞지 않음.")

        # 디버깅: attention 후 proj_A와 proj_B 차이 확인
        diff = torch.norm(proj_A - proj_B, dim=-1)
        print(f"proj_A와 proj_B 차이 (평균): {diff.mean().item()}")

        # 최종 출력 반환
        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        # contrastive loss 계산을 위한 차원 일치 확인
        proj_q = proj_q.view(-1, proj_q.shape[-1])
        proj_p = proj_p.view(-1, proj_p.shape[-1])

        assert proj_q.shape == proj_p.shape, \
            f"proj_q and proj_p shapes must match, got {proj_q.shape} and {proj_p.shape}"
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        # NT-Xent (Normalized temperature-scaled cross-entropy) loss 계산
        a_norm = torch.norm(a, dim=1, keepdim=True) + 1e-8  # 작은 값 추가
        a_cap = a / a_norm
        b_norm = torch.norm(b, dim=1, keepdim=True) + 1e-8  # 작은 값 추가
        b_cap = b / b_norm
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t()) / tau
        exp_sim_by_tau = torch.exp(sim)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1) - torch.diag(exp_sim_by_tau)

        # Anchor-positive similarity scores를 위한 slice 조정
        sum_of_rows = sum_of_rows[:a.shape[0]]

        numerators = torch.exp(torch.div(torch.nn.functional.cosine_similarity(a_cap, b_cap), tau))
        num_by_den = numerators / (sum_of_rows + 1e-8)  # 작은 값 추가
        return -torch.mean(torch.log(num_by_den + 1e-8))  # 작은 값 추가


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A_np = proj_A.detach().cpu().numpy().flatten()
    proj_B_np = proj_B.detach().cpu().numpy().flatten()
    srocc, _ = stats.spearmanr(proj_A_np, proj_B_np)
    plcc, _ = stats.pearsonr(proj_A_np, proj_B_np)
    
    print(f"SRCC: {srocc}, PLCC: {plcc}")
    return srocc, plcc


if __name__ == "__main__":
    # 모델 구조 확인을 위한 테스트 코드
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    model = SimCLR(encoder_params)

    # [batch_size, num_crops, C, H, W] 형식의 예시 데이터 생성
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # Forward pass 수행
    proj_A, proj_B = model(img_A, img_B)

    # SRCC, PLCC 계산
    calculate_srcc_plcc(proj_A, proj_B)

    # 출력 차원 확인
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")

 """


## 수정
""" import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from models.resnet import ResNet
from models.attention_module import DistortionAttention, HardNegativeCrossAttention
from torch.nn import TripletMarginLoss

class ModifiedSimCLR(nn.Module):  # 여기에서 ModifiedSimCLR 이름을 SimCLR로 변경합니다.
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1, margin: float = 1.0):
        super(ModifiedSimCLR, self).__init__()

        # ResNet Encoder와 projection layer
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        
        # projector의 첫 번째 Linear 레이어 입력 크기를 2048로 설정
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )

        self.self_attention = DistortionAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.cross_attention = HardNegativeCrossAttention(embed_dim=encoder_params.embedding_dim, num_heads=4)
        self.temperature = temperature
        self.triplet_loss = TripletMarginLoss(margin=margin)

        # 초기화 방법 설정 (xavier 초기화)
        self._initialize_weights()

    def _initialize_weights(self):
        # Sequential 내부의 Linear 레이어 각각 초기화
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, img_anchor, img_positive, img_negative):
        # Encoder 통과 후 projector를 거쳐 attention 적용
        anchor_features = self.encoder(img_anchor)[0]
        positive_features = self.encoder(img_positive)[0]
        negative_features = self.encoder(img_negative)[0]

        anchor_proj = self.self_attention(self.projector(anchor_features))
        positive_proj = self.self_attention(self.projector(positive_features))
        negative_proj = self.self_attention(self.projector(negative_features))

        return anchor_proj, positive_proj, negative_proj

    def compute_loss(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)
 """


# 이게 simclr 원본
""" 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from models.resnet import ResNet
from models.attention_module import DistortionAttention, HardNegativeCrossAttention
from torch.nn import TripletMarginLoss

class SimCLR(nn.Module):  
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1, margin: float = 1.0):
        super(SimCLR, self).__init__()

        # ResNet Encoder와 projection layer
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        
        # projector의 첫 번째 Linear 레이어 입력 크기를 2048로 설정
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )

        self.self_attention = DistortionAttention(embed_dim=encoder_params.embedding_dim, num_heads=8)
        self.cross_attention = HardNegativeCrossAttention(embed_dim=encoder_params.embedding_dim, num_heads=8)
        self.temperature = temperature
        self.triplet_loss = TripletMarginLoss(margin=margin)

        # 초기화 방법 설정 (xavier 초기화)
        self._initialize_weights()

    def _initialize_weights(self):
        # Sequential 내부의 Linear 레이어 각각 초기화
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, img_anchor, img_positive, img_negative):
        # Encoder 통과 후 projector를 거쳐 attention 적용
        anchor_features = self.encoder(img_anchor)[0]
        positive_features = self.encoder(img_positive)[0]
        negative_features = self.encoder(img_negative)[0]

        anchor_proj = self.self_attention(self.projector(anchor_features))
        positive_proj = self.self_attention(self.projector(positive_features))
        negative_proj = self.self_attention(self.projector(negative_features))

        return anchor_proj, positive_proj, negative_proj

    def compute_loss(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)
     """

# 12/6 수정
import torch
import torch.nn as nn
from models.attention_module import DistortionAttention, HardNegativeCrossAttention
from models.resnet import ResNet


import torch
import torch.nn as nn
from models.attention_module import DistortionAttention, HardNegativeCrossAttention
from models.resnet import ResNet
from utils.nt_xent_loss import NT_Xent_Loss  # NT-Xent Loss 임포트


class SimCLR(nn.Module):
    def __init__(self, encoder_params, temperature: float = 0.1, margin: float = 1.0):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )
        self.self_attention = DistortionAttention(embed_dim=encoder_params.embedding_dim, num_heads=8)
        self.cross_attention = HardNegativeCrossAttention(embed_dim=encoder_params.embedding_dim, num_heads=8)
        self.temperature = temperature
        self.nt_xent_loss = NT_Xent_Loss(temperature=temperature)  # NT-Xent Loss 사용
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):  # BatchNorm 초기화
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, img_anchor, img_positive, img_negative):
        # ResNet encoder
        anchor_features = self.encoder(img_anchor)[0]
        positive_features = self.encoder(img_positive)[0]
        negative_features = self.encoder(img_negative)[0]

        # Self-Attention
        anchor_proj = self.self_attention(self.projector(anchor_features).unsqueeze(1)).squeeze(1)
        positive_proj = self.self_attention(self.projector(positive_features).unsqueeze(1)).squeeze(1)

        # Cross-Attention
        if negative_features is None:
            raise ValueError("[Error] Negative features are None. Check the data preprocessing pipeline.")

        negative_proj = self.cross_attention(
            self.projector(anchor_features),
            self.projector(negative_features)
        )

        return anchor_proj, positive_proj, negative_proj

    def compute_loss(self, anchor, positive, negative):
        if anchor is None or positive is None or negative is None:
            raise ValueError("[Error] One or more inputs to compute_loss are None.")
        return self.nt_xent_loss(anchor, positive)  # NT-Xent Loss 계산





# Attention 관련 파라미터를 추가
""" import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from models.resnet import ResNet
from models.attention_module import DistortionAttention, HardNegativeCrossAttention
from torch.nn import TripletMarginLoss

class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, attention_params: DotMap, temperature: float = 0.1, margin: float = 1.0):
        super(SimCLR, self).__init__()

        # ResNet Encoder와 projection layer
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        
        # projector의 첫 번째 Linear 레이어 입력 크기를 2048로 설정
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_params.embedding_dim)
        )

        # Self-Attention 및 Cross-Attention 초기화
        self.self_attention = DistortionAttention(
            embed_dim=attention_params.embed_dim,
            num_heads=attention_params.num_heads,
            dropout=attention_params.dropout,
            num_layers=attention_params.num_layers
        )
        self.cross_attention = HardNegativeCrossAttention(
            embed_dim=attention_params.embed_dim,
            num_heads=attention_params.num_heads,
            dropout=attention_params.dropout,
            num_layers=attention_params.num_layers
        )

        self.temperature = temperature
        self.triplet_loss = TripletMarginLoss(margin=margin)

        # 초기화 방법 설정 (xavier 초기화)
        self._initialize_weights()

    def _initialize_weights(self):
        # Sequential 내부의 Linear 레이어 각각 초기화
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, img_anchor, img_positive, img_negative):
        # Encoder 통과 후 projector를 거쳐 attention 적용
        anchor_features = self.encoder(img_anchor)[0]
        positive_features = self.encoder(img_positive)[0]
        negative_features = self.encoder(img_negative)[0]

        # Self-Attention 적용
        anchor_proj = self.self_attention(self.projector(anchor_features))
        positive_proj = self.self_attention(self.projector(positive_features))
        negative_proj = self.self_attention(self.projector(negative_features))

        return anchor_proj, positive_proj, negative_proj

    def compute_loss(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)
 """