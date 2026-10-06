# Custom Metal kernels

The `.metal.inc` files contain C++ raw strings for `fast::metal_kernel`. The
`.metal` files are prebuilt into `paged_attn.metallib`.
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
- `kquant/`: the ggml K-quant / IQ kernels (`kquant.h`, `kquant_nax.h`) and
  their instantiation lists (`kquant.metal`, `kquant_nax.metal`). `build.rs`
  compiles the lists with MLX's kernel flags into `paged_attn.metallib`, and
  `mlx_kquant_metal.cpp` loads every kernel from there (no JIT fallback),
  choosing kernels like MLX's affine dispatcher. Its `kquant::kernels` name
  builders must match the lists; `kquant_metallib_names` checks both ways.
  `build.rs` also turns each header into a `quantized_preamble::` function for
  the custom kernels that reuse its decoders. Two in-memory weight layouts
  (`mlx_kquant.h`): the on-disk row-major one, and `Tiled64` (mode suffix
  `@t64`, set at load by `QuantizedLinear::tile_kquant_layout` for every 2-D
  weight with N % 64 == 0 and K % 256 == 0 on a Metal host), where 64 rows
  interleave per 32-code unit and per super-block so a 64-column threadgroup
  streams contiguous runs. Only the
  `_t64` kernels (`qmv_t64`, `qmv_wide_t64`, `qmm_m8_nax_t64`,
  `qmm_t_nax_t64`, `qmm_t_t64`, `qmm_t_splitk_t64`) and the CPU reference
  read it; the dispatcher refuses it everywhere else (`kquant_tiled` tests).
- `affine_mixed/affine_qmv_wide_mixed.metal`: the BF16 x / F32 affine sidecar
  `qmv_wide`, one instantiation per tile width. `segmented_sdpa/sdpa_segmented.metal`:
  the BF16 D=256 segmented SDPA kernels, specialized by function constants at
  pipeline build; `mlx_segmented_sdpa.cpp` reduces their partials with MLX's
  own `sdpa_vector_2pass_2`. `build.rs` prebuilds both into
  `paged_attn.metallib` with the K-quant flags, and their dispatchers load them
  from there (no JIT fallback). The host names must match the dispatchers'
  name builders; `bridge_metallib_names` checks both ways.
- The three kernel families above (`kquant/`, `affine_qmv_wide_mixed`,
  `sdpa_segmented`) are guarded by `#[ignore]` golden-digest gates in
  `crates/mlx-core/tests/*_golden_gate.rs`, run on every MLX pin bump. A change
  here that moves output bits fails them too, and they have no capture mode
  ([docs/mlx-fork.md](../../../../docs/mlx-fork.md)).
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
| `qk_norm_rope`                                               | Bit-identical to `fast::rms_norm` then `fast::rope(traditional=false)`: N_READS=4 row mapping, the two `simd_sum` folds, `precise::rsqrt`, `exp2(-d * log2(base))`, `fast::cos/sin`; the normed row is rounded to T before rotation. D % 128 == 0, int32 `[B]` offsets. |
| `attention_gate_bf16`                                        | Head-interleaved query/gate storage and a BF16 sigmoid lookup matching the caller's native operation.                                                                        |
| `conv1d_silu_k4`                                             | Four F32 taps, three rows of separately owned history, ordered products and rounded SiLU.                                                                                    |
| `precise_sigmoid`                                            | Precise exponential; this is not interchangeable with the native BF16 lookup tables.                                                                                         |

`build.rs` watches this directory recursively. Source attribution for the
mlx.fast ports is retained in the shaders and [MLXFAST-LICENSE.txt](MLXFAST-LICENSE.txt).
