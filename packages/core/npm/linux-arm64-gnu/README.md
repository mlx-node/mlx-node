# `@mlx-node/core-linux-arm64-gnu`

This is the **aarch64-unknown-linux-gnu** CUDA (`sm_121a`, DGX Spark / GB10) binary for `@mlx-node/core`.

## System Requirements

Loading the native `.node` addon and running MLX NVRTC JIT kernels require:

- **CUDA 13.x runtime libraries & NVIDIA driver**: `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcufft.so.12`, `libcusolver.so.12`, `libnvrtc.so.13` (plus NVRTC builtins), and the NVIDIA driver (`libcuda.so.1`)
- **CUDA 13.x toolkit headers**: `cuda-cudart-dev-13-*` or full `cuda-toolkit-13-*` at `/usr/local/cuda/include` (or point `CUDA_HOME` / `CUDA_PATH` to the CUDA toolkit root) — required by NVRTC JIT alongside the CCCL/CuTe/CUTLASS headers bundled in this package
- **cuDNN 9**: `libcudnn9-cuda-13`
- **BLAS / LAPACK**: `libblas3`, `liblapack3`
