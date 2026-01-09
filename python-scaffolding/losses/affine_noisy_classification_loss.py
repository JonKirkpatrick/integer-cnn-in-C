import torch
from losses.base import IntLoss

"""
    Affine Noisy Classification Loss
    --------------------------------
    This loss function introduces a noisy affine setpoint around which the logits are pulled.
    It applies a bump to the true class and penalizes incorrect predictions with compensation
    for other classes. The noise is uniformly distributed around a randomly chosen centre
    within a specified range and regenerated each forward pass(Batch).

    It is not likely that this loss function is optimal, but it serves as a starting point for
    experimenting with integer-based classification tasks.  It was observed that an integer model
    can be trained to chase a set point, and also one with noise.  This particular loss was the first
    to produce anything even resembling learning in the course grained 8-bit integer landscape.  We
    produced about 5% over random with this loss on a 12 class problem.
"""

class AffineNoisyClassificationLoss(IntLoss):
    def __init__(
        self,
        centre_low=-30,
        centre_high=21,
        radius=12,
        true_class_bump=60,
        rival_class_penalty=110,
        rival_class_compensation=10,
    ):
        self.centre_low = centre_low
        self.centre_high = centre_high
        self.radius = radius
        self.true_class_bump = true_class_bump
        self.rival_class_penalty = rival_class_penalty
        self.rival_class_compensation = rival_class_compensation

    def compute(self, logits, targets):
        """
        logits: int8 or int32 tensor [B, C]
        targets: int64 tensor [B]
        returns: int32 error tensor [B, C]
        """
        device = logits.device
        B, _ = logits.shape

        masked_logits = logits.clone().to(torch.int32)
        masked_logits[torch.arange(B), targets] = -256
        rival = torch.argmax(masked_logits, dim=1)

        # sample centre (per batch)
        centre = torch.randint(
            low=self.centre_low,
            high=self.centre_high + 1,
            size=(1,),
            device=device,
            dtype=torch.int32
        )

        # uniform noise around centre
        rand = torch.randint(
            low=centre - self.radius,
            high=centre + self.radius + 1,
            size=logits.shape,
            device=device,
            dtype=torch.int32
        )
        
        # base error: pull toward noisy affine setpoint while damping the logits
        # error = rand - logits.to(torch.int32) // 2
        error = torch.zeros_like(logits, dtype=torch.int32, device=device)

        # true class bump and wrong class penalty
        error[torch.arange(B, device=device), targets] += 1 #self.true_class_bump
        error[torch.arange(B, device=device), rival] -= 1 #self.rival_class_penalty

        return error