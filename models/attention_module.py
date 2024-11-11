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
