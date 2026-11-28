# `@mlx-node/core-win32-arm64-msvc`

This is the **aarch64-pc-windows-msvc** CUDA binary for `@mlx-node/core`.

## System Requirements

Loading the native `.node` addon and running MLX NVRTC JIT kernels require:

- **Windows 11 ARM64** with an NVIDIA driver (`nvcuda.dll`)
- **CUDA 13.4+ runtime DLLs on `PATH`** (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`, `cufft64_12.dll`, `cusolver64_12.dll`, `cusparse64_12.dll`, `nvJitLink_130_0.dll`, `nvrtc64_130_0.dll`, and NVRTC builtins) plus CUDA toolkit headers (`CUDA_PATH` / `CUDA_HOME`) for NVRTC JIT alongside the CCCL/CuTe/CUTLASS headers bundled in this package
- **cuDNN 9 runtime DLLs on `PATH`** (`cudnn64_9.dll` and companion `cudnn_*.dll`)

`openblas.dll` and the CCCL/CuTe/CUTLASS headers (`include/`) are bundled in this package.
