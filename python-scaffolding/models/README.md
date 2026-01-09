Integer-based 1D Separable Convolutional Neural Network with Backpropagation

This module defines integer based modules for building a 1D CNN suitable for tasks such as tactile classification.
The network supports depthwise and pointwise convolutions, ReLU activations, max pooling, and a final linear classifier.
All operations are performed using integer arithmetic to facilitate deployment on resource-constrained devices.

Within the PyTorch framework, we have to use rather inefficient implementations due to a lack of integer support for many operations.
Eventually, we will be migrating to a custom backend that supports integer operations natively, but for now this serves as a proof of concept
and a way to experiment with integer-based architectures and training methods without the development time of building and modifying a custom backend.

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
