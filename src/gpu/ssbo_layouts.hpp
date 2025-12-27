#pragma once
#include <cstdint>

namespace gpu
{
    constexpr uint32_t SSBO_LAYOUT_VERSION = 1;
    constexpr uint32_t MAX_CHANNELS  = 256;
    constexpr uint32_t MAX_TIMESTEPS = 256;

    constexpr uint32_t TENSOR_WORDS =
        (MAX_CHANNELS * MAX_TIMESTEPS + 3) / 4;

    struct TensorSSBO
    {
        int32_t data[TENSOR_WORDS];
    };

    struct ModelSpecSSBO
    {
        uint32_t channels;
        uint32_t timesteps;
        uint32_t shift;
        uint32_t flags;
    };
}