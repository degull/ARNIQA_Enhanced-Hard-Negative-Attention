import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Cosine similarity 계산
        similarity = F.cosine_similarity(z_i, z_j, dim=1)
        loss = -torch.log(torch.exp(similarity / self.temperature) / torch.sum(torch.exp(similarity / self.temperature), dim=0))
        return loss.mean()
