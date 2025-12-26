# Integer-Native CNN Training and Inference

This repository contains an experimental framework for training and running
convolutional neural networks **entirely in integer space**, without floating
point arithmetic, fixed-point scaling, or post-training quantization.

The core idea is simple:

> Train the model in exactly the same numerical domain it will occupy at
> inference time.

Rather than learning in floating point and then attempting to approximate the
result with quantized parameters, this project explores a workflow in which:
- parameters are integer-valued from the outset (e.g. int8),
- accumulation and normalization are performed using shifts and integer math,
- the resulting model can be deployed directly to constrained hardware
  (FPGA, embedded systems, or fixed-function accelerators) without further
  transformation.

---

## Project Goals

The primary goals of this project are:

- **Numerical fidelity**  
  Eliminate the mismatch between training-time and inference-time arithmetic.

- **Hardware alignment**  
  Design the software model as a close analogue of a real hardware
  implementation (FPGA-oriented, instruction-driven, explicit memory layout).

- **Minimal abstraction**  
  Avoid large ML frameworks in favor of a small, inspectable, deterministic
  execution model.

- **Analysability**  
  Enable direct inspection of intermediate representations, feature lifetimes,
  and information flow through the network.

This is not intended to be a general-purpose deep learning framework. It is a
research and engineering tool focused on a specific class of models and
numerical constraints.

---

## Architecture Overview

- **Language:** C++ (C++23)
- **Execution:** OpenGL compute shaders (GLSL 4.3+)
- **Frontend:** SFML (for OpenGL context and optional visualization)
- **Numerics:** Integer-only (no floating point in model execution)
- **Target use case:** 1D sensor time-series data with separable convolutions

The CNN is expressed as a sequence of explicit operations (e.g. depthwise 1D
convolution, pointwise 1D convolution, bias addition, shifts), conceptually
similar to a small instruction set rather than a dynamic computation graph.

Shaders are treated as runtime assets and are loaded relative to the executable,
making the `bin/` directory fully self-contained.

---

## Status

This project is **experimental**.

Key components (training stability, update rules, loss formulation) are still
under active development and evaluation. There are no claims that this approach
outperforms conventional floating-point training or established quantization
methods.

The intent is to explore whether this style of training is:
- viable,
- stable,
- and useful for specific hardware-constrained domains.

Results — positive or negative — are part of the goal.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

You are free to:
- use the code,
- modify it,
- redistribute it,
- publish derivatives,

provided that the terms of the GPLv3 are respected. See the `LICENSE` file for
full details.

Attribution beyond the requirements of the GPL is appreciated but not required.

---

## Notes

If you are coming from a traditional machine learning background, some design
choices here may appear unusual or intentionally restrictive. These choices are
deliberate and motivated by hardware behavior rather than framework conventions.

If you are coming from a hardware or signal-processing background, the model may
look more familiar than expected.

Both perspectives are welcome.
