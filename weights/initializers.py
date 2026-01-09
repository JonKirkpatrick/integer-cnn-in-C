import torch

def int_kaiming(module):
    """
    Initializes the weights of the given module using Kaiming initialization
    adapted for integer weights.

    Args:
        module: The neural network module whose weights need to be initialized.
    """
    if hasattr(module, 'weights'):
        if module.weight_type in ["depthwise", "pointwise", "linear"]:
            fan_in = module.in_ch if module.weight_type != "depthwise" else module.kernel_size
            numerator = 6 if module.weight_type == "pointwise" else 3
            bound = max(1, int(((numerator / fan_in) ** 0.5) * 96))
            weight = torch.randint(-bound, bound, module.weights.shape, dtype=torch.int8)
            with torch.no_grad():
                module.weights.copy_(weight)
        else:
            raise ValueError(f"Unknown weight type: {module.weight_type}")
        
def int_uniform(module):
    """
    Initializes the weights of the given module using uniform distribution
    adapted for integer weights.

    Args:
        module: The neural network module whose weights need to be initialized.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.
    """
    if hasattr(module, 'weights'):
        if module.weight_type in ["depthwise", "pointwise", "linear"]:
            weight = torch.randint(-128, 128, module.weights.shape, dtype=torch.int8)
            with torch.no_grad():
                module.weights.copy_(weight)
        else:
            raise ValueError(f"Unknown weight type: {module.weight_type}")