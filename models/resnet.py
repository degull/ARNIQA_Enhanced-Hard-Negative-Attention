""" import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50


class ResNet(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1  # V1 weights work better than V2
        else:
            weights = None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),  # 첫 번째 Linear 레이어
            nn.ReLU(),                       # 활성화 함수
            nn.Linear(1024, self.embedding_dim)  # 두 번째 Linear 레이어
        )

        

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
        
        
if __name__ == "__main__":
    # 모델 인스턴스 생성
    model = ResNet(embedding_dim=128, pretrained=True, use_norm=True)

    # 포함된 레이어 확인
    print(model)

 """

#이거원본
""" 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
import torch

class ResNet(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        # 사전 학습 가중치 설정
        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = resnet50(weights=weights)

        # 특징 차원 수 가져오기
        self.feat_dim = self.model.fc.in_features
        # FC 레이어 제외
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # `projector` 레이어 정의
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embedding_dim)
        )

    def forward(self, x):
        # 모델의 특징 추출 파트 실행
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        # 정규화 여부에 따라 처리
        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g

if __name__ == "__main__":
    # 모델 인스턴스 생성
    model = ResNet(embedding_dim=128, pretrained=True, use_norm=True)
    
    # `device` 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 모델 전체를 `device`로 이동

    # 모델 구조 출력
    print(model)
 """
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.use_norm = use_norm

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embedding_dim)
        )

    def forward(self, x):
        features = self.model(x)
        features = features.view(-1, self.feat_dim)

        if self.use_norm:
            features = nn.functional.normalize(features, dim=1)

        projection = self.projector(features)
        if self.use_norm:
            projection = nn.functional.normalize(projection, dim=1)

        print(f"[Debug] ResNet features shape: {features.shape}, projection shape: {projection.shape}")
        return features, projection
