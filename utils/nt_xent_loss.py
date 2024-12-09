""" import torch
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
 """


import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature

        mask = torch.eye(batch_size * 2, device=z.device).bool()
        positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)])
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)

        loss = -torch.log(torch.exp(positives) / torch.sum(torch.exp(negatives), dim=1))
        return loss.mean()
