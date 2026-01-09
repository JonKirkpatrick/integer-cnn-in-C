import torch
import torch.nn as nn
import math
from weights.initializers import int_uniform as initialize_weights

"""
Integer-based 1D Separable Convolutional Neural Network with Backpropagation

This module defines integer based modules for building a 1D CNN suitable for tasks such as tactile classification.
The network supports depthwise and pointwise convolutions, ReLU activations, max pooling, and a final linear classifier.
All operations are performed using integer arithmetic to facilitate deployment on resource-constrained devices.

Within the PyTorch framework, we have to use rather inefficient implementations due to a lack of integer support for many operations.
Eventually, we will be migrating to a custom backend that supports integer operations natively, but for now this serves as a proof of concept
and a way to experiment with integer-based architectures and training methods without the development time of building and modifying a custom backend.
"""

"""
The specification tuple consists of:
- Input Channels
- Initial Temporal Length
- Convolutional Blocks: A list of tuples, each defining (kernel_size, out_channels, maxpool_size)
    - There is no upper limit to the number of convolutional block tuples, but hardware constraints may apply.
    - Each convolutional block consists of:
    - Depthwise 1d Convolution:
        - kernel_size: The size of the depthwise 1d convolution kernel.
            - must be odd to preserve symmetry.
            - stride is always 1.
            - padding is always kernel_size // 2 to preserve temporal length.
    - Pointwise 1d Convolution:    
        - out_channels: The number of output channels for the pointwise 1d convolution.
            - it uses the out channels of the previous layer as input channels.
            - it must always be a power of two and at least the same size as the input channels.
            - it can be no larger than 256 due to hardware constraints.
    - ReLU Activation:
        - always applied after the pointwise convolution.
    - Maxpooling:
        - maxpool_size: The size of the maxpooling operation following the pointwise convolution.
            - it reduces the temporal length by this factor.
            - it must be a power of two or None (to skip maxpooling).
            - stride is always equal to maxpool_size.
            - padding is always 0.
- Global Average Pooling:
    - requires no parameters.
    - always applied after the last convolutional block to reduce temporal length to 1.
    - it leverages the power of two temporal lengths to allow efficient averaging by bit-shifting.
- Fully Connected Layer:
    - uses the output channels from the global average pooling as input features.
    - uses the number of output classes as output features.
    - reduces the output from the global average pooling to the number of output classes.
- Number of Output Classes

An example specification:
spec = (13, 128, [(7, 64, 2), (3, 128, None), (13, 256, 4), (3, 256, 2)], 12)
     = ( input channels = 13,
         initial temporal length = 128,
         convolutional blocks = [
             ( depthwise kernel size = 7,
               pointwise out channels = 64,
               maxpool size = 2 ),
             ( depthwise kernel size = 3,
               pointwise out channels = 128,
               maxpool size = None ),
             ( depthwise kernel size = 13,
               pointwise out channels = 256,
               maxpool size = 4 ),
             ( depthwise kernel size = 3,
               pointwise out channels = 256,
               maxpool size = 2 )
         ],
         number of output classes = 12 )

The model builder also requires the batch size to compute appropriate shifts for averaging during backpropagation.
"""

# Utility functions

# Clamp int32 tensor to int8 range after right shift
def shift_clamp_i32_to_i8(x_i32, shift):
    threshold = 1 << shift
    mask = x_i32.abs() >= threshold
    x = torch.where(mask, x_i32, torch.zeros_like(x_i32)).to(torch.int32)
    x = x >> shift
    return torch.clamp(x, -128, 127).to(torch.int8)

# Dead shift: right shift with zeroing values below threshold and no clamping
def dead_shift(x_i32, shift):
    threshold = 1 << shift
    mask = x_i32.abs() >= threshold
    x = torch.where(mask, x_i32, torch.zeros_like(x_i32)).to(torch.int32)
    x = x >> shift
    return x

# Compute final temporal length and required shift for averaging
def compute_final_length_and_shift(L0, block_specs):
    L = L0
    for _, _, p in block_specs:
        if p is not None:
            if L % p != 0:
                raise ValueError(f"Pooling kernel {p} does not divide length {L}")
            L //= p

    if (L & (L - 1)) != 0:
        raise ValueError(f"Final temporal length {L} is not a power of two")

    shift = int(math.log2(L))
    return L, shift

def debug_dx(dX, name, b=0, c=0):
    B, C, L = dX.shape
    print(f"{name} dX[b={b}, c={c}, :10]: {dX[b, c, :10].tolist()}")
    print(f"{name} dX[b={b}, :, L//2][:10]: {dX[b, :, L//2].tolist()}")

def debug_dx_linear(dX, name, b=0, f=0, n=10):
    # dX shape: [B, F] or [F]
    if dX.dim() == 1:
        print(f"{name} dX[:{n}]: {dX[:n].tolist()}")
        return

    B, F = dX.shape

    # Feature trace for a fixed batch element
    print(f"{name} dX[b={b}, :{n}]: {dX[b, :n].tolist()}")

    # Batch trace for a fixed feature
    print(f"{name} dX[:{n}, f={f}]: {dX[:n, f].tolist()}")


# Set of modules for integer-based CNN

# The IntBlock combines depthwise conv, pointwise conv, ReLU, and optional maxpooling as a sequential block
class IntBlock(nn.Module):
    def __init__(self, in_ch, dw_kernel, out_ch, pool_kernel, current_length):
        super().__init__()
        dw_shift = int(math.floor(math.log2(dw_kernel)))
        pw_shift = int(math.log2(in_ch))
        pw_dX_shift = int(math.log2(out_ch))
        self.dw = IntDepthwiseConv1D(
            channels=in_ch,
            kernel_size=dw_kernel,
            shift=dw_shift,
            current_length=current_length
        )

        self.pw = IntPointwiseConv1D(
            in_ch=in_ch,
            out_ch=out_ch,
            shift=pw_shift,
            back_shift=pw_dX_shift,
        )
        self.relu = IntReLU()
        self.pool = IntMaxPool1D(pool_kernel) if pool_kernel else IntIdentity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    def backward(self, dY):
        dX = self.pool.backward(dY)
        dX = self.relu.backward(dX)
        dX = self.pw.backward(dX)
        dX = self.dw.backward(dX)
        return dX

# Integer Depthwise 1D Convolution Module
class IntDepthwiseConv1D(nn.Module):
    def __init__(self, channels, kernel_size, shift, current_length):
        super().__init__()
        self.weight_type = "depthwise"
        self.channels = channels
        self.kernel_size = kernel_size
        self.shift = shift
        self.radius = kernel_size // 2
        self.current_length = current_length

        weight = torch.empty((channels, kernel_size), dtype=torch.int8)
        self.register_buffer('weights', weight)
        self.register_buffer('dW', torch.zeros((channels, kernel_size), dtype=torch.int32))
        initialize_weights(self)
        self.last_x = None
        self.last_w = None

    def forward(self, x):
        B, C, L = x.shape
        out = torch.zeros((B, C, L), dtype=torch.int32, device=x.device)
        weight = self.weights
        self.last_x = x.to(torch.int32)
        self.last_w = weight.to(torch.int32)
        
        for k in range(self.kernel_size):
            ti = k - self.radius
            w_k = weight[:, k][None, :, None]
            if ti < 0:
                out[:, :, -ti:] += (x[:, :, :L+ti] * w_k)
            elif ti > 0:
                out[:, :, :-ti] += (x[:, :, ti:] * w_k)
            else:
                out += (x * w_k)
        return shift_clamp_i32_to_i8(out, self.shift)
    
    def backward(self, dY):
        dOut = dY
        dX = torch.zeros_like(self.last_x, dtype=torch.int32)

        for k in range(self.kernel_size):
            ti = k - self.radius
            w_k = self.last_w[:, k][None, :, None]
            if ti < 0:
                dX[:, :, :dOut.shape[2]+ti] += (dOut[:, :, -ti:] * w_k)
            elif ti > 0:
                dX[:, :, ti:] += (dOut[:, :, :-ti] * w_k)
            else:
                dX += (dOut * w_k)
        for k in range(self.kernel_size):
            ti = k - self.radius
            if ti < 0:
                self.dW[:, k] += (self.last_x[:, :, :dOut.shape[2] + ti] * dOut[:, :, -ti:]).sum(dim=(0, 2))
            elif ti > 0:
                self.dW[:, k] += (self.last_x[:, :, ti:] * dOut[:, :, :-ti]).sum(dim=(0, 2))
            else:
                self.dW[:, k] += (self.last_x * dOut).sum(dim=(0, 2))
        dX = dX // (self.kernel_size - (self.radius * (self.radius + 1)) // self.current_length)
        dX = dead_shift(dX, 5)
        debug_dx(dX, "DepthwiseConv1D", b=0, c=0)
        return dX

# Integer Pointwise 1D Convolution Module
class IntPointwiseConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, shift, back_shift):
        super().__init__()
        self.weight_type = "pointwise"
        self.shift = shift
        self.back_shift = back_shift
        self.in_ch, self.out_ch = in_ch, out_ch
        
        weight = torch.empty((out_ch, in_ch), dtype=torch.int8)
        self.register_buffer('weights', weight)
        initialize_weights(self)
        self.register_buffer('dW', torch.zeros((out_ch, in_ch), dtype=torch.int32, device=weight.device))
        self.last_x = None
        self.last_w = None

    def forward(self, x):
        weight = self.weights
        self.last_x = x.to(torch.int32)
        self.last_w = weight.to(torch.int32)
        acc = (self.last_x.unsqueeze(1) * self.last_w.unsqueeze(0).unsqueeze(3)).sum(dim=2)
        return shift_clamp_i32_to_i8(acc, self.shift)

    def backward(self, dY):
        dAcc = dY
        self.dW += (
            dAcc.unsqueeze(2) *
            self.last_x.unsqueeze(1)
        ).sum(dim=(0, 3))
        dX = (
            dAcc.unsqueeze(2) *
            self.last_w.unsqueeze(0).unsqueeze(3)
        ).sum(dim=1)
        dX = dead_shift(dX, self.back_shift + 2)
        debug_dx(dX, "PointwiseConv1D", b=0, c=0)
        return dX

# Integer ReLU Activation Module
class IntReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_x = None

    def forward(self, x):
        self.last_x = (x > 0)
        return torch.maximum(x, torch.zeros_like(x))

    def backward(self, dY):
        mask = (self.last_x > 0)
        return dY * mask.to(dY.dtype)

# Integer Max Pooling 1D Module
class IntMaxPool1D(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.last_indices = None

    def forward(self, x):
        self.last_indices = None
        B, C, L = x.shape
        L2 = L // self.k
        maxpool = x[:, :, :L2*self.k].view(B, C, L2, self.k).max(dim=3)
        self.last_indices = maxpool.indices
        return maxpool.values
    
    def backward(self, dY):
        B, C, L2 = dY.shape
        L = L2 * self.k
        dX = torch.zeros((B, C, L), dtype=dY.dtype, device=dY.device)

        for i in range(self.k):
            mask = (self.last_indices == i).to(dY.dtype)
            dX[:, :, i::self.k] += dY * mask

        return dX

# Integer Identity Module (for skipping operations)
class IntIdentity(nn.Module):
    def forward(self, x):
        return x

    def backward(self, dY):
        return dY

# Integer Classifier Module, combining global average pooling and final linear layer
class IntClassifier(nn.Module):
    def __init__(self, in_ch, num_classes, final_shift):
        super().__init__()
        fc_shift = int(math.log2(in_ch))
        fc_dX_shift = int(math.log2(num_classes))
        self.final_shift = final_shift
        self.fc = IntLinear(
            in_ch=in_ch, 
            out_ch=num_classes, 
            shift=fc_shift,
            back_shift=fc_dX_shift)
        self.last_x = None

    def forward(self, x):
        self.last_x = x
        acc = x.to(torch.int32).sum(dim=2)
        acc = dead_shift(acc, self.final_shift)
        return self.fc.forward(acc)
    
    def backward(self, dY):
        dAcc = self.fc.backward(dY)
        B, C, L = self.last_x.shape
        dX = dAcc.unsqueeze(-1).expand(B, C, L)
        return dX

# Integer Linear Layer Module (Fully Connected Layer)
class IntLinear(nn.Module):
    def __init__(self, in_ch, out_ch, shift, back_shift):
        super().__init__()
        self.weight_type = "linear"
        self.shift = shift
        self.back_shift = back_shift
        self.in_ch, self.out_ch = in_ch, out_ch
    
        weight = torch.empty((out_ch, in_ch), dtype=torch.int8)
        self.register_buffer('weights', weight)
        initialize_weights(self)
        self.register_buffer('dW', torch.zeros((out_ch, in_ch), dtype=torch.int32, device=weight.device))
        self.last_x = None
        self.last_w = None

    def forward(self, x):
        weight = self.weights
        self.last_x = x.to(torch.int32)
        self.last_w = weight.to(torch.int32)
        acc = (self.last_x.unsqueeze(1) * self.last_w.unsqueeze(0)).sum(dim=2)
        y = shift_clamp_i32_to_i8(acc, self.shift)
        return y
    
    def backward(self, dY):
        dAcc = dY
        self.dW += (dAcc.unsqueeze(2) * self.last_x.unsqueeze(1)).sum(dim=0)
        dX = (dAcc.unsqueeze(2) * self.last_w.unsqueeze(0)).sum(dim=1)
        dX = dead_shift(dX, self.back_shift + 3)
        debug_dx_linear(dX, "Linear", b=0, f=0, n=10)
        return dX

# Integer Model constructed from specification tuple
class IntModelFromSpec(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        C_in, L0, block_specs, num_classes = spec
        L = L0
        self.final_L, self.final_shift = compute_final_length_and_shift(L0, block_specs)

        self.blocks = nn.ModuleList()
        c_in = C_in

        for (k, c_out, p) in block_specs:
            block = IntBlock(
                in_ch=c_in,
                dw_kernel=k,
                out_ch=c_out,
                pool_kernel=p,
                current_length=L
            )
            self.blocks.append(block)
            c_in = c_out
            L = L // p if p is not None else L

        self.classifier = IntClassifier(
            in_ch=c_in,
            num_classes=num_classes,
            final_shift=self.final_shift
        )

        self.permute_input = True

    def forward(self, x):
        if self.permute_input:
            x = x.permute(0, 2, 1).contiguous().to(torch.int8)
        else:
            x = x.to(torch.int8)

        for block in self.blocks:
            x = block(x)
    
        return self.classifier(x)
    
    def backward(self, dY):
        dX = self.classifier.backward(dY)
        for block in reversed(self.blocks):
            dX = block.backward(dX)
        return dX