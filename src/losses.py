import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha_toxic: float = 0.71, gamma: float = 2.0):
        super().__init__()
        self.alpha_toxic = alpha_toxic
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.long().view(-1)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p[torch.arange(targets.numel(), device=targets.device), targets]
        logpt = logp[torch.arange(targets.numel(), device=targets.device), targets]
        alpha = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha_toxic),
            torch.full_like(pt, 1.0 - self.alpha_toxic),
        )
        return (-alpha * (1.0 - pt).pow(self.gamma) * logpt).mean()
