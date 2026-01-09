from optimizers.base import IntOptimizer
import torch

# Dead shift: right shift with zeroing values below threshold and no clamping
def dead_shift(x_i32, shift):
    threshold = 1 << shift
    mask = x_i32.abs() >= threshold
    x = torch.where(mask, x_i32, torch.zeros_like(x_i32)).to(torch.int32)
    x = x >> shift
    return x


class SignedThresholdOptimizer(IntOptimizer):
    def __init__(self, threshold):
        self.threshold = threshold

    def _register(self, model):
        for module in model.modules():
            if hasattr(module, 'dW'):
                sign_acc = torch.zeros_like(module.dW, dtype=torch.int32, device=module.dW.device)
                module.register_buffer('sign_acc', sign_acc)

    def step(self, model):
        for module in model.modules():
            if not (hasattr(module, 'dW')):
                continue

            dW = dead_shift(module.dW, 8)
            #module.weights += dW
            #print(f"A sample of some dW values {dW.view(-1)[:10].tolist()} for module {module.__class__.__name__}")
            module.dW.zero_()