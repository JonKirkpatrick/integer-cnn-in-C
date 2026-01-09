from training.training import train_model
from optimizers.signed_threshold import SignedThresholdOptimizer
from losses.affine_noisy_classification_loss import AffineNoisyClassificationLoss
from losses.course_cross_entropy import CourseCrossEntropyLoss

# Define training parameters

# Total number of epochs to train
number_of_epochs = 1000

# Batch size for training
batch_size = 256

# Fraction of the dataset to use for training
sample_size = 0.1

# Path to the dataset
dataset_path = "data/int8_dataset.npz"

# Enable confusion matrix recording and specify output directory
record_confusion_matrices = True
output_dir = "evolution_frames"

# Initialize the optimizer

"""
This is an area of active research. The model has been fully implemented to operate strictly
with integer arithmetic during both the forward and backward passes. However, the choice of
optimizer for updating the integer weights is still under investigation, along with the loss
function design.
"""

optimizer = SignedThresholdOptimizer(
    threshold=64
)

# Initialize the Loss Function

"""
loss_function = CourseCrossEntropyLoss(scale=96, affine_shift=0)
"""

loss_function = AffineNoisyClassificationLoss(
    centre_low=-4,
    centre_high=8,
    radius=12,
    true_class_bump=128,
    rival_class_penalty=64,
    rival_class_compensation=0,
)

# Define the model specification

"""
The specification tuple consists of:
- Input Channels (13)
- Initial Temporal Length (128)
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
    - always applied after the last convolutional block to reduce temporal length to 1.
    - it leverages the power of two temporal lengths to allow efficient averaging by bit-shifting.
- Fully Connected Layer:
    - reduces the output from the global average pooling to the number of output classes.
- Number of Output Classes (12)
"""

spec = (13, 128, [(7, 64, 2), (3, 128, 2), (13, 256, 2), (3, 256, 2)], 12)

train_model(
    number_of_epochs,
    batch_size,
    sample_size,
    dataset_path,
    spec,
    optimizer,
    loss_function,
    record_confusion_matrices,
    output_dir
)