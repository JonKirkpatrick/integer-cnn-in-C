import torch
from losses.base import IntLoss

class CourseCrossEntropyLoss(IntLoss):
    def __init__(self, scale=96, affine_shift=0):
        self.scale = scale
        self.affine_shift = affine_shift

    def compute(self, logits, targets):
        """
        logits: int8 or int32 tensor [B, C]
        targets: int64 tensor [B]
        returns: int32 error tensor [B, C]
        """
        logits_f = logits.float()
        grad_logits = torch.softmax(logits_f, dim=1)
        grad_logits[torch.arange(targets.size(0)), targets] -= 1.0
        error = torch.round(grad_logits * self.scale + self.affine_shift).to(torch.int32)
        print(f"Max Error: {error.max().item()}, Min Error: {error.min().item()}, Avg Error: {error.float().mean().item()}")
        print(f"Error Std Dev: {error.float().std().item()}, a few error values: {error.view(-1)[:12].tolist()}")
        print(f"True labels: {targets[:12].tolist()}")
        return error