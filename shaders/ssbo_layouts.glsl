#ifndef SSBO_LAYOUTS_GLSL
#define SSBO_LAYOUTS_GLSL

const uint SSBO_LAYOUT_VERSION = 1;

const uint MAX_CHANNELS  = 256;
const uint MAX_TIMESTEPS = 256;

// Packed int8 activations: 4 values per int
const uint TENSOR_WORDS =
    (MAX_CHANNELS * MAX_TIMESTEPS + 3) / 4;

layout(std430) buffer TensorBuffer
{
    int data[TENSOR_WORDS];
};

layout(std430) buffer ModelSpec
{
    uint channels;
    uint timesteps;
    uint shift;
    uint flags;
};

#endif