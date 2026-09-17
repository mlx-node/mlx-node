# Custom Metal kernels

The `.metal.inc` files contain C++ raw strings for `fast::metal_kernel`.
Organize them by their mathematical and storage contracts:

- `common/`: reusable operations. A kernel may specialize a dtype, tile size,
  quantization format, reduction order, or head mapping without belonging to a
  model family. `quantized.h` provides the host-side MLX preambles shared by
  these operations. `native_activations.h` shares the evaluated BF16 SiLU
  lookup while preserving the native sigmoid and multiplication rounding.
- `qwen4/`: checkpoint-specific fusions and dispatch assumptions: 512-way
  top-10 routing, fixed shared-expert packing, 16 key / 48 value heads, and
  four-stream hyper-connection mixing. The dense projection kernels also
  retain mixer epilogues that divide by four.
- Add `qwen3_5/`, `lfm2/`, or `gemma4/` when a shader requires that family's
  semantics. Qwen3.5 currently uses `common/` recurrence and quantized kernels;
  LFM2 and Gemma4 have no family-specific shader includes here.

## Reuse contracts

The C++ family adapters retain shape checks, dispatch policy, and fallbacks.
Moving a shader to `common/` does not enable it for another model automatically.
Before adding a caller, validate its storage layout and exact arithmetic:

| Common operation                                             | Required contract                                                                                                                                                            |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `quantized_expert_*`, `affine_expert_down`, `packed_group32` | Match packed bits, group size, companion dtypes, expert-aligned tiles and output rounding. NAX paths require supported hardware.                                             |
| `route_counts`                                               | Eight SIMD groups per block; ballot variant supports at most 512 experts.                                                                                                    |
| `sorted_expert_shared_combine`                               | Inverse routing permutation, eight-lane reduction order, rounded routed products and a separate sigmoid-gated shared expert.                                                 |
| `gated_delta_step_4vcol_vector`                              | Contiguous inputs, 128 key channels, value width divisible by four, modulo key-head mapping, FP32 state and output. Other recurrence variants keep their own state rounding. |
| `group_rms_norm`                                             | Independent groups with per-group F32 weights; widths divisible by 128, at most 4096, and evenly partitioned SIMD slices.                                                    |
| `rms_norm_rope_256`                                          | 256-channel heads, split-half rotary layout, F32 scale and reduction, explicit low-precision products.                                                                       |
| `rope_split_half`                                            | Split-half rotation with precomputed cosine/sine and separate rounded products.                                                                                              |
| `attention_gate_bf16`                                        | Head-interleaved query/gate storage and a BF16 sigmoid lookup matching the caller's native operation.                                                                        |
| `conv1d_silu_k4`                                             | Four F32 taps, three rows of separately owned history, ordered products and rounded SiLU.                                                                                    |
| `precise_sigmoid`                                            | Precise exponential; this is not interchangeable with the native BF16 lookup tables.                                                                                         |

`build.rs` watches this directory recursively. Source attribution for the
mlx.fast ports is retained in the shaders and [MLXFAST-LICENSE.txt](MLXFAST-LICENSE.txt).
