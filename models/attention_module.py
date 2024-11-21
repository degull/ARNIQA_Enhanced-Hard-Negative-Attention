""" import torch
import torch.nn as nn
import torch.nn.init as init

class DistortionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(DistortionAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, features):
        attn_output = features
        for layer in self.layers:
            attn_output, _ = layer(attn_output, attn_output, attn_output)
            attn_output = self.dropout(attn_output)
        return attn_output


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(HardNegativeCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, high_res_features, low_res_features):
        attn_output = high_res_features
        for layer in self.layers:
            attn_output, _ = layer(attn_output, low_res_features, low_res_features)
            attn_output = self.dropout(attn_output)
        return attn_output


if __name__ == "__main__":
    # 테스트 데이터 생성
    x = torch.randn(2, 7, 128)  # [batch_size, sequence_length, embed_dim]

    # DistortionAttention 테스트
    distortion_attention = DistortionAttention(embed_dim=128, num_heads=4, dropout=0.2, num_layers=3)
    output = distortion_attention(x)
    print("DistortionAttention output shape:", output.shape)

    # HardNegativeCrossAttention 테스트
    hard_neg_attention = HardNegativeCrossAttention(embed_dim=128, num_heads=4, dropout=0.2, num_layers=3)
    output = hard_neg_attention(x, x)
    print("HardNegativeCrossAttention output shape:", output.shape)
 """
""" 
import torch
import torch.nn as nn
import torch.nn.init as init

class DistortionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(DistortionAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization 추가
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, features):
        attn_output = features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Layer normalization - 입력에 적용
            attn_output, _ = layer(attn_output, attn_output, attn_output)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Layer normalization - 출력에 적용
        return attn_output



class HardNegativeCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(HardNegativeCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization 추가
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, high_res_features, low_res_features):
        attn_output = high_res_features
        for layer in self.layers:
            attn_output, _ = layer(attn_output, low_res_features, low_res_features)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Layer normalization 적용
        return attn_output


if __name__ == "__main__":
    # 테스트 데이터 생성
    x = torch.randn(2, 7, 128)  # [batch_size, sequence_length, embed_dim]

    # DistortionAttention 테스트
    distortion_attention = DistortionAttention(embed_dim=128, num_heads=4, dropout=0.2, num_layers=3)
    output = distortion_attention(x)
    print("DistortionAttention output shape:", output.shape)

    # HardNegativeCrossAttention 테스트
    hard_neg_attention = HardNegativeCrossAttention(embed_dim=128, num_heads=4, dropout=0.2, num_layers=3)
    output = hard_neg_attention(x, x)
    print("HardNegativeCrossAttention output shape:", output.shape)
 """

# 원본
""" import torch
import torch.nn as nn
import torch.nn.init as init

class DistortionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(DistortionAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization 추가
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, features):
        attn_output = features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Layer normalization - 입력에 적용
            attn_output, _ = layer(attn_output, attn_output, attn_output)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Layer normalization - 출력에 적용
        return attn_output

class HardNegativeCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(HardNegativeCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization 추가
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, high_res_features, low_res_features):
        attn_output = high_res_features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Layer normalization - 입력에 적용
            attn_output, _ = layer(attn_output, low_res_features, low_res_features)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Layer normalization - 출력에 적용
        return attn_output
 """

# Positional Encoding과 Conv/2 + ReLU 추가
""" import torch
import torch.nn as nn
import torch.nn.init as init
import math

class DistortionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(DistortionAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization 추가
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, features):
        attn_output = features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Layer normalization - 입력에 적용
            attn_output, _ = layer(attn_output, attn_output, attn_output)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Layer normalization - 출력에 적용
        return attn_output

class HardNegativeCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(HardNegativeCrossAttention, self).__init__()

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)
        
        # Conv/2 + ReLU 레이어
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # Multi-Head Attention Layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, high_res_features, low_res_features):
        # Positional Encoding 추가
        high_res_features = high_res_features + self.positional_encoding
        low_res_features = low_res_features + self.positional_encoding

        # Conv/2 + ReLU 적용
        # Conv2d는 입력을 4차원 (B, C, H, W) 형태로 받기 때문에 채널 차원을 맞춰줘야 함
        high_res_features = high_res_features.unsqueeze(-1).unsqueeze(-1)
        low_res_features = low_res_features.unsqueeze(-1).unsqueeze(-1)

        high_res_features = self.conv1(high_res_features)
        high_res_features = self.relu(high_res_features)
        low_res_features = self.conv2(low_res_features)
        low_res_features = self.relu(low_res_features)

        # Conv2d 연산 후 2차원 (Batch, Embed_dim) 형태로 변환
        high_res_features = high_res_features.squeeze(-1).squeeze(-1)
        low_res_features = low_res_features.squeeze(-1).squeeze(-1)

        # Multi-Head Cross Attention Layers
        attn_output = high_res_features
        for layer in self.layers:
            # Layer Normalization
            attn_output = self.norm(attn_output)
            # Multi-Head Cross Attention
            attn_output, _ = layer(attn_output, low_res_features, low_res_features)
            # Dropout
            attn_output = self.dropout(attn_output)
            # Output에 Layer Normalization 추가
            attn_output = self.norm(attn_output)
        
        return attn_output

 """

## 개선
import torch
import torch.nn as nn
import torch.nn.init as init


class DistortionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(DistortionAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, features):
        attn_output = features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Normalize input
            attn_output, _ = layer(attn_output, attn_output, attn_output)
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Normalize output
        return attn_output


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_layers=2):
        super(HardNegativeCrossAttention, self).__init__()

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

        # Convolutional layers
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Multi-Head Attention Layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, high_res_features, low_res_features):
        # Add positional encoding
        high_res_features = high_res_features + self.positional_encoding
        low_res_features = low_res_features + self.positional_encoding

        # Reshape features for Conv2D layers
        high_res_features = high_res_features.unsqueeze(-1).unsqueeze(-1)  # (Batch, Channels, 1, 1)
        low_res_features = low_res_features.unsqueeze(-1).unsqueeze(-1)

        # Convolutional layers
        high_res_features = self.relu(self.conv1(high_res_features))
        low_res_features = self.relu(self.conv2(low_res_features))

        # Flatten back to 2D (Batch, Embed_dim)
        high_res_features = high_res_features.view(high_res_features.size(0), -1)
        low_res_features = low_res_features.view(low_res_features.size(0), -1)

        # Multi-Head Cross Attention Layers
        attn_output = high_res_features
        for layer in self.layers:
            attn_output = self.norm(attn_output)  # Normalize input
            attn_output, _ = layer(attn_output, low_res_features, low_res_features)  # Attention
            attn_output = self.dropout(attn_output)
            attn_output = self.norm(attn_output)  # Normalize output

        return attn_output
