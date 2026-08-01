# Convert & quantize internals

`docs/cli.md` covers how to *invoke* `mlx convert` / `mlx calibrate`. This doc covers what those
commands actually do: the on-disk formats, the per-tensor decision engine, the `config.json`
provenance contract, and the places where a conversion looks right for the wrong reason.

Audience: someone adding a quantization mode, or debugging a checkpoint that converted cleanly and
loads wrong.

## Mental model — three owners, one file layout

Every quantized tensor is a group of 1–3 arrays sharing a base key: `{base}.weight`,
`{base}.scales`, `{base}.biases`. Nine different formats reuse that same shape. Which format a group
actually is comes from `config.json`, not from the file layout — so three stages own three different
things and confusing them is the main source of drift here.

| Stage         | Code                                                                                                                     | Owns                                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **CONVERTER** | `crates/mlx-core/src/convert.rs`, `crates/mlx-core/src/utils/gguf.rs`, `crates/mlx-core/src/convert_gemma_import.rs`      | Picks per-key `{bits, group_size, mode}`, packs the bytes, writes the `quantization` block + per-tensor overrides                        |
| **LOADER**    | `crates/mlx-core/src/engine/persistence.rs`, `crates/mlx-core/src/models/quant_dispatch.rs`, per-family `persistence.rs` | Parses that block, **rebuilds arrays that are not on disk**, fail-loud validates every dtype/shape before dispatch                       |
| **KERNEL**    | `crates/mlx-sys/mlx/mlx/ops.cpp` → `backend/{metal,cpu}`                                                                  | Consumes `(weight, scales, biases, group_size, bits, mode)`; re-validates via `validate_mode_with_type` + `quantization_params_from_mode` |

Three things the loader materializes that are **not bytes on disk**:

1. `.biases` for symmetric ggml imports (Q4_0/Q8_0) — `scales * -Z`, `crates/mlx-core/src/engine/persistence.rs:279`
2. The sym8 `[K,N]` contiguous int8 kernel operand — `crates/mlx-core/src/models/qwen3_5/quantized_linear.rs:440`
3. The bf16 weight for `fp8_e4m3` — reconstructed once at load, `crates/mlx-core/src/quant/fp8_weight.rs:156`

Mode strings round-trip through one pair of inverses: `parse_mode_str`
(`crates/mlx-core/src/models/quant_dispatch.rs:123`) and `mode_to_str` (`:153`).

## Pipeline

```
                       mlx convert -i <in> -o <out> [flags]
                                     │
                     flag validation │ packages/cli/src/commands/convert.ts:210-475
                     (runs for ALL   │ 8 guards live here and ONLY here
                      paths, before  │
                      any dispatch)  │
                                     ▼
                        inputPath.endsWith('.gguf') ?      convert.ts:480
                          │                          │
                         YES                        NO
                          │                          │
   ┌──────────────────────▼──────────┐               │
   │ convertGgufToSafetensors        │               │
   │   utils/gguf.rs:2477            │               │
   │   -m / --q-mtp / sym8 NOT read  │               │
   │   → ONE model.safetensors       │               │
   │     (no index.json)             │               │
   │   + optional 2nd pass for       │               │
   │     --mmproj → vision.safetensors│              │
   └─────────────────────────────────┘               │
                                                     │
                          modelType = -m ?? auto-detect     convert.ts:601-651
                                     │                       (try/catch SWALLOWS a bad config.json)
                          ┌──────────┴───────────┐
                          │                      │
              -m ∈ {pp-lcnet-ori, uvdoc}    everything else
                          │                      │
   ┌──────────────────────▼───────┐  ┌───────────▼──────────────────────────┐
   │ convertForeignWeights        │  │ convertModel → convert_model_inner   │
   │  utils/foreign_weights.rs:50 │  │  crates/mlx-core/src/convert.rs:1884 │
   │  Paddle / PyTorch pickle     │  │                                      │
   │  no quant, no sharding,      │  │  1. weight-file probe (first match)  │
   │  synthesized config.json     │  │  2. expand_symmetric_affine_biases   │
   └──────────────────────────────┘  │  3. recipe.sanitize()   (per family) │
                                     │  4. dtype cast                       │
                                     │  5. AWQ imatrix pre-scale            │
                                     │  6. quantize_weights_inner           │
                                     │  7. save_safetensors_sharded (5 GiB) │
                                     │  8. config.json + quantization block │
                                     └──────────────────────────────────────┘
```

Inside step 6, the per-tensor decision is composed of up to four layers:

```
  key ──▶ apply_mtp_quant_policy          OUTERMOST   convert.rs:2908-2913
             (intercepts every mtp.* key; inner predicate never consulted)
            │
            ▼
          apply_mxfp_upgrade  XOR  apply_nvfp4_upgrade
             (skipped entirely when a FIXED unsloth map was selected)
                                                      convert.rs:2888-2898
            │
            ▼
          recipe predicate                            convert.rs:2883-2887
             build_official_unsloth_recipe(kind)  OR  build_predicate_for_recipe(name)
             (absent ⇒ resolve_legacy_entry ladder, the only sym8-aware path)
            │
            ▼
          should_quantize(key, embed_quantizable)     convert.rs:3317
             (the shared floor — called INSIDE each recipe)
```

Execution (`quantize_weights_inner`, `crates/mlx-core/src/convert.rs:6557`):
`Skip` → leave dense; `Default` → re-check `should_quantize`, then use the top-level triple;
`Custom` → use it verbatim. A per-tensor override lands in `config.json` **only** when the resolved
triple differs from the top-level one (`record_quant_override_if_non_default`,
`crates/mlx-core/src/convert.rs:5601`, called from `:6455`, `:6528`, `:6677`).

## Quantization modes & on-disk formats

Nine formats. Only five are selectable by `--q-mode` (`VALID_QUANT_MODES`,
`crates/mlx-core/src/convert.rs:1967`). `fp8_e4m3` is emitted only by the fixed Unsloth DGX map. The
three ggml K-quants are **consume-only** — `mx.quantize` throws for them by name
(`crates/mlx-sys/mlx/mlx/ops.cpp:5245`).

### Master table

Shapes are for a dense `[N, K]` source weight; stacked experts add a leading `[E, …]` to every array.

| mode                 | `.weight`               | `.scales`               | `.biases`                     | bpw                                   | default bits / gs     | mlx-lm loadable         | direction        |
| -------------------- | ----------------------- | ----------------------- | ----------------------------- | ------------------------------------- | --------------------- | ----------------------- | ---------------- |
| **affine**           | u32 `[N, K·b/32]`       | *wdtype* `[N, K/gs]`    | *wdtype* `[N, K/gs]`          | `b + 2·16/gs` → 4/64 = **4.500**      | **4 / 64**            | yes                     | produce + consume |
| **mxfp4**            | u32 `[N, K/8]`          | u8 `[N, K/32]`          | must be absent                | **4.250**                             | **4 / 32** (pinned)   | yes                     | produce + consume |
| **mxfp8**            | u32 `[N, K/4]`          | u8 `[N, K/32]`          | must be absent                | **8.250**                             | **8 / 32** (pinned)   | yes                     | produce + consume |
| **nvfp4**            | u32 `[N, K/8]`          | u8 (E4M3) `[N, K/16]`   | must be absent                | **4.500**                             | **4 / 16** (pinned)   | yes                     | produce + consume |
| **fp8_e4m3**         | u8 `[N, K]` raw E4M3    | bf16 `[N, 1]`           | must be absent                | `8 + 16/K` = **8.0039** @ K=4096      | 8 / `null`            | no — KeyError + null gs | DGX map only     |
| **sym8**             | **int8** `[N, K]`       | **f32 `[N]`**           | error if present              | `8 + 32/K` = **8.0078** @ K=4096      | 8 / `null`            | no — by design          | produce + consume, **M5+ only** |
| **q6k**              | u32 `[N, K·6/32]`       | **int8** `[N, K/16]`    | f16 `[N, K/256]`              | **6.5625** (= ggml)                   | 6 / 16 (pinned)       | no                      | **consume only** |
| **q4k**              | u32 `[N, K/8]`          | u8 `[N, 2K/32]`         | f16 `[N, 2K/256]`             | **4.6250** (ggml 4.500)               | 4 / 32 (pinned)       | no                      | **consume only** |
| **q5k**              | u32 `[N, 5K/32]`        | u8 `[N, 2K/32]`         | f16 `[N, 2K/256]`             | **5.6250** (ggml 5.500)               | 5 / 32 (pinned)       | no                      | **consume only** |
| **Q4_0 → affine**    | u32 `[N, K/8]`          | f16 `[N, K/32]`         | **omitted, derived `-8·s`**   | **4.500** (= ggml)                    | 4 / 32                | no — missing `.biases`  | GGUF source only |
| **Q8_0 → affine**    | u32 `[N, K/4]`          | f16 `[N, K/32]`         | **omitted, derived `-128·s`** | **8.500** (= ggml)                    | 8 / 32                | no                      | GGUF source only |
| **Q4_1 → affine**    | u32 `[N, K/8]`          | f16 `[N, K/32]`         | f16 `[N, K/32]` (ggml `m`)    | **5.000** (= ggml)                    | 4 / 32                | yes                     | GGUF source only |

Defaults are declared in exactly three consistent places plus MLX itself:
`packages/cli/src/commands/convert.ts:9` (display only — see gotchas),
`crates/mlx-core/src/convert.rs:1995` (SafeTensors), `crates/mlx-core/src/utils/gguf.rs:2884` (GGUF),
`crates/mlx-sys/mlx/mlx/ops.cpp:4586` (`quantization_params_from_mode`, which additionally owns
q6k(16,6) / q4k(32,4) / q5k(32,5)).

### affine

```
decode:  w = scale * q + bias,   q ∈ [0, 2^bits - 1]  (unsigned)
         multiply(w, scales) then add(biases)     ops.cpp:5361
shapes:  wq.back()     = K * bits / 32            ops.cpp:5103
         scales.back() = K / group_size           ops.cpp:5113 (both cast back to w.dtype())
legal:   group_size ∈ {32, 64, 128},  bits ∈ {2,3,4,5,6,8}    ops.cpp:5059
```

Affine is the **only** mode exempt from the `(group_size, bits)` pinning gate at
`crates/mlx-sys/mlx/mlx/ops.cpp:4661`.

bpw = `bits + 2·(scale_dtype_bits / group_size)`. On `[4096, 4096]` = 16,777,216 weights, bf16
companions:

| config | weight bytes                | scales    | biases    | total      | bpw       |
| ------ | --------------------------- | --------- | --------- | ---------- | --------- |
| 4 / 64 | 4096·512·4 = 8,388,608      | 524,288   | 524,288   | 9,437,184  | **4.500** |
| 8 / 64 | 4096·1024·4 = 16,777,216    | 524,288   | 524,288   | 17,825,792 | **8.500** |
| 4 / 32 | 8,388,608                   | 1,048,576 | 1,048,576 | 10,485,760 | **5.000** |
| 3 / 64 | 4096·384·4 = 6,291,456      | 524,288   | 524,288   | 7,340,032  | **3.500** |

`--dtype float32` doubles the companion cost: 4/64 becomes `4 + 2·32/64` = **5.000** bpw.

### mxfp4 / mxfp8 / nvfp4 — float micro-scaling

Two-array modes. `quant_weight_arrays()` returns 1 companion
(`crates/mlx-sys/mlx/mlx/primitives.h:165`) and `validate_mode_with_type` throws
"Biases must be null for quantization mode" at `crates/mlx-sys/mlx/mlx/ops.cpp:4744`.

```
w = scale * decode(code)                                   fp_quantized.h:139
element codec by BITS:    bits==4 → E2M1 4-bit             fp_quantized.h:51
                          bits==8 → E4M3 byte
scale   codec by GROUP_SIZE, not by mode:                  fp_quantized.h:30
                          group_size==16 → E4M3 scale
                          otherwise      → E8M0 scale
  ⇒  mxfp4 = E2M1 × E8M0(gs 32)
     mxfp8 = E4M3 × E8M0(gs 32)
     nvfp4 = E2M1 × E4M3(gs 16)
```

Output dtypes `{uint32, uint8}` are hard-coded in `fp_quantize`
(`crates/mlx-sys/mlx/mlx/ops.cpp:5217`). bpw:

| mode  | bits | gs | weight bpw | scale bpw   | total     |
| ----- | ---- | -- | ---------- | ----------- | --------- |
| mxfp4 | 4    | 32 | 4          | 8/32 = 0.25 | **4.250** |
| mxfp8 | 8    | 32 | 8          | 0.25        | **8.250** |
| nvfp4 | 4    | 16 | 4          | 8/16 = 0.50 | **4.500** |

Checked on `[4096,4096]`: mxfp4 = 8,388,608 + 524,288 = 8,912,896 B ⇒ 4.25 ✓; nvfp4 = 8,388,608 +
1,048,576 = 9,437,184 ⇒ 4.50 ✓.

**No NVFP4 global scale is ever written** — `global_scale` has zero hits in `crates/mlx-core/src/`.
The DGX port does not carry Unsloth's calibrated global scales.

### fp8_e4m3 — plain per-output-channel E4M3

Not a `--q-mode`. Explicitly distinct from MLX `mxfp8` (`crates/mlx-core/src/quant/fp8_weight.rs:1`).

```
encode   dequant_scale = clip( max(|w|, axis=-1, keepdims) / 448 , min = f32::MIN_POSITIVE )
                                                       fp8_weight.rs:79   ← the floor matters:
                                                       an all-zero row gets 1.175e-38, not 0
         encoded = to_fp8( clip( w / dequant_scale, ±448 ) )       → Uint8 [..., N, K]
         scales  = dequant_scale.astype(BFloat16)                  → BF16  [..., N, 1]
decode   w ≈ from_fp8(q) * scale, in bf16, ONCE at load    fp8_weight.rs:156-187
forward  plain x.matmul(weight.T)             qwen3_5/quantized_linear.rs:763
```

`group_size` is the sentinel `-1` (`crates/mlx-core/src/quant/fp8_weight.rs:19`), serialized as JSON
`null` (`crates/mlx-core/src/convert.rs:5589`). `FP8_E4M3_MAX = 448.0`. Load-time rejects include any
decoded magnitude > 448, because MLX's fast decoder maps the reserved `0x7f`/`0xff` E4M3 patterns to
±480 while `to_fp8` saturates at ±448 (`crates/mlx-core/src/quant/fp8_weight.rs:170`).

**Emission gate** — `--q-recipe unsloth --q-mode nvfp4` alone is *not* enough.
`select_official_unsloth_recipe` (`crates/mlx-core/src/convert.rs:4416`) returns `Some(Nvfp4)` only
when `recipe == "unsloth"` **and** `is_qwen35_hybrid` **and** `!quant_mxfp` **and**
`quant_mode == "nvfp4"`. On any non-Qwen-hybrid input the same CLI line returns `None` and
`validate_unsloth_imatrix_after_selection` (`:4396`) either hard-errors or falls through to the
legacy affine predicate — zero `fp8_e4m3` tensors.

### sym8 — per-output-channel symmetric int8

The only mode with an **unpacked** weight and a **1-D** scale.

```
encode   s[n] = max( max_k |w[n,k]| , 1e-12 ) / 127
         q    = clip(round(w/s), ±127) as int8              mlx_na_int8.cpp:1028
decode   w[n,k] ≈ scales[n] * q[n,k]
forward  M ≤ 2 → int8_w8a16_qmv ;  M ≥ 3 → int8_w8a8_matmul
         never touches mlx_quantized_matmul (no sym8 pack)  quantized_linear.rs:662
bpw      8 + 32/K  →  8.0078 @ K=4096
```

**No group_size at all.** `config.json` records top-level `"group_size": null`
(`crates/mlx-core/src/convert.rs:3127`); the in-memory sentinel is `-1`
(`crates/mlx-core/src/models/quant_dispatch.rs:25`). `parse_group_size` accepts `null` **only** for
`sym8`/`fp8_e4m3` and rejects an integer for them (`:621-648`). The `(8, 64)` pair in the defaults
tables is the **affine-fallback** group for the layers sym8 declines, not sym8's own group
(`crates/mlx-core/src/convert.rs:2000`).

Eligibility (`sym8_eligible`, `crates/mlx-core/src/convert.rs:5546`): 2-D `[N,K]` **and**
`K % 16 == 0`. Deliberately excludes the GPU-generation check, which is a runtime property — see
gotchas.

Fallback ladder inside `resolve_legacy_entry` (`crates/mlx-core/src/convert.rs:6233`), in order:

| condition                                                                       | outcome                    |
| ------------------------------------------------------------------------------- | -------------------------- |
| `!should_quantize(key)`                                                          | dense bf16                 |
| gemma4 PLE (`per_layer_*`) / `audio_tower` / `audio_encoder` / `embed_audio`     | dense bf16 (sym8-scoped)   |
| lfm2 packed embedding                                                            | dense bf16                 |
| `is_affine_only_key` (lm_head / router.proj / embed_tokens\* / embedding_projection) | 8-bit **affine**, gs 64 |
| `is_router_gate`                                                                 | 8-bit **affine**, gs 64    |
| 3-D `[E,N,K]` experts, or 2-D with `K % 16 != 0`                                 | 8-bit **affine**, gs 64    |
| everything else                                                                  | **sym8**                   |

`enforce_sym8_group_coherence` (`crates/mlx-core/src/convert.rs:6002`) then re-applies the emission
gates to every co-quantized group (five tables at `:5963`) and forces the whole group dense if any
member would not emit, because the strict loaders are all-or-none. It hard-errors instead of
force-densing when any member is already sidecarred on disk (`:6185`).

`sym8_supported()` is true for `Qwen35Recipe` (**both** dense and MoE,
`crates/mlx-core/src/convert.rs:1170`), `Lfm2Recipe` (`:1513`), `Gemma4Recipe` (`:1736`).

### ggml K-quants — the two-level decode

On-disk contract at `crates/mlx-core/src/utils/gguf_kquant.rs:9`, shape formulas at `:113`, for a row
of `K` values with `K % 256 == 0`.

`.biases` on a K-quant holds ggml's `d` (and `dmin`) — a **SCALE, not an additive bias**. The name is
reused only so the ~25 `.scales`/`.biases` "is quantized" sentinel sites keep working
(`crates/mlx-core/src/utils/gguf_kquant.rs:22`).

Decode (`KQScales::at`, identical in Metal `crates/mlx-sys/mlx/mlx/backend/metal/kernels/kquant.h:627`
and CPU `crates/mlx-sys/mlx/mlx/backend/cpu/quantized.cpp:1132`). There are exactly **two** branches,
selected by `has_min`, which is true for **both** q4k and q5k
(`quant_has_sub_min`, `crates/mlx-sys/mlx/mlx/primitives.h:185`):

```
has_min == true    (q4k AND q5k)          super_ratio = 8
    scale =   d[2*(g >> 3)]     * sc[2*g]
    bias  = -( d[2*(g >> 3) + 1] * sc[2*g + 1] )
    q5k differs from q4k only in the fifth bit plane.

has_min == false   (q6k)                  super_ratio = 16
    scale = d[g >> 4] * (int8) sc[g]
    bias  = -32.0f * scale                 // folds ggml's (q - 32)
```

Why this reuses MLX's affine kernel algebra:

1. A K-quant sub-block is *algebraically* affine — `value = scale*q + bias`, both constant inside a
   group (`kquant.h:5`). `get_pack_factor`, `load_vector`, `qdot`, `qouter`, `dequantize` are byte
   copies of the affine ones and take `(scale, bias)` by value — they never see a K-quant.
2. The **importer**, not the kernel, absorbs ggml's swizzle: `q6k_code`/`q4k_code`/`q5k_code`
   de-interleave ggml's ql/qh/nibble planes (`gguf_kquant.rs:257`) and `BitPacker` re-emits a plain
   LSB-first n-bit stream — bit `j` of code `i` at absolute bit `i*bits + j` (`:184`).
3. Sub-scales are stored **unpacked** rather than in ggml's 6-bit fields, which "keeps the affine
   kernel's per-group pointer walk intact" (`gguf_kquant.rs:432`) and costs exactly +0.125 bpw.
4. Every row is a whole number of 256-value super-blocks, so a flat group index stays aligned as the
   cursor runs off one row into the next (`kquant.h:604`).

It is a **copied kernel family, not the same binary**: `kquant.h`/`kquant.metal` are separate
instantiations with two extra template params (`super_ratio`, `has_min`), and `QuantizedBlockLoader`
generalises the affine loader's `static_assert(BCOLS <= group_size)`
(`crates/mlx-sys/mlx/mlx/backend/metal/kernels/quantized.h:574`) into `group_steps`/`scale_step`
because q6k's group of 16 is narrower than the BK=32 tile (`kquant.h:672`).

bpw = `bits + 8/scales_per_value + 16·per_group/256`:

| mode | weight | scales      | biases          | mlx-node   | bytes/super-block | ggml block | ggml bpw | Δ      |
| ---- | ------ | ----------- | --------------- | ---------- | ----------------- | ---------- | -------- | ------ |
| q6k  | 6      | 8/16 = 0.5  | 16/256 = 0.0625 | **6.5625** | 192+16+2 = **210** | 210 B      | 6.5625   | **0**  |
| q4k  | 4      | 2·8/32 = 0.5 | 2·16/256 = 0.125 | **4.6250** | 128+16+4 = **148** | 144 B      | 4.5000   | +0.125 |
| q5k  | 5      | 0.5         | 0.125           | **5.6250** | 160+16+4 = **180** | 176 B      | 5.5000   | +0.125 |

Exactness vs llama.cpp: decode is float32 on both levels, so **q4k and q5k are bitwise identical**.
**Q6_K is identical up to the sign of zero** — ggml subtracts 32 in integer arithmetic, the contract
folds it into `bias = -32*scale`, and IEEE-754 gives `x + (-x) = +0.0` where ggml writes `-0.0`
(`crates/mlx-sys/mlx/mlx/backend/cpu/quantized.cpp:1091`). A second divergence exists for a
non-finite `d` (whole super-block NaN here vs signed infinities in ggml) but no real GGUF holds one.

Consume-only is enforced by name: `quantize()` throws "can be read but not produced" for any mode
with `quant_super_ratio > 0` (`crates/mlx-sys/mlx/mlx/ops.cpp:5245`); `dequantize()` routes to the
`kq_dequantize` primitive at `crates/mlx-sys/mlx/mlx/ops.cpp:5575`; `kquant.metal:12` has no
`quantize` instantiation.

### ggml symmetric Q4_0 / Q8_0 — the derived-bias scheme

```
ggml     Q4_0:  w = d*(q-8)          Q8_0: w = d*(q-128)   (after the importer's sign-bit flip)
MLX      w = scale*q + bias   ⇒  bias is the CONSTANT -Z*scale, so it need not be stored.

CONVERTER writes  .weight u32, .scales f16, NO .biases,
                  config.json symmetric_zero_point = 8 (Q4_0) / 128 (Q8_0)   gguf.rs:891
LOADER   rebuilds biases = scales.mul_scalar(-Z)                    persistence.rs:279
```

Bitwise exactness rests on two independent facts:

- `mlx_array_mul_scalar` builds the scalar **in the array's own dtype** when the array is floating
  (`crates/mlx-sys/src/mlx_array_ops.cpp:357`), so an f16 `.scales` yields an f16 `.biases` with no
  f32 promotion and no double-rounding.
- `Z ∈ {8 = 2³, 128 = 2⁷}` is a power of two, so `-Z*s` only shifts the exponent field. The mantissa
  is untouched and the product is exactly representable in f16 for every finite non-overflowing `s`,
  including subnormals (`2^-24 · 8 = 2^-21`).

The historical writer is kept in-tree as `derived_symmetric_bias_bits`
(`crates/mlx-core/src/utils/gguf.rs:748`) purely as a test oracle; `symmetric_bias_parity.rs` asserts
raw IEEE-754 bit equality with **no tolerance** over an adversarial scale set.

bpw at K=4096 rows:

| source | weight | scales      | biases            | mlx-node  | ggml block | ggml bpw |
| ------ | ------ | ----------- | ----------------- | --------- | ---------- | -------- |
| Q4_0   | 4      | 16/32 = 0.5 | **0** (derived)   | **4.500** | 18 B / 32  | 4.500    |
| Q8_0   | 8      | 0.5         | **0** (derived)   | **8.500** | 34 B / 32  | 8.500    |
| Q4_1   | 4      | 0.5         | 0.5 (real `m`)    | **5.000** | 20 B / 32  | 5.000    |

Before the change, Q4_0 landed at 5.0 and Q8_0 at 9.0 — larger than the GGUF they came from. 0.5 bpw
= 0.0625 bytes/weight; the 681 MB saving on Gemma-4-12B-QAT (`crates/mlx-core/src/engine/persistence.rs:206`) therefore implies
≈ 1.09 × 10¹⁰ quantized weights, i.e. essentially every matrix in the model.

Legal zero points are exactly `1 << (bits-1)` (`crates/mlx-core/src/models/quant_dispatch.rs:700`).
Any other value is **rejected, not clamped** — it would rebuild every bias at the wrong offset while
the shapes still match.

Related but distinct: `repack_symmetric_to_mlx_affine`
(`crates/mlx-core/src/utils/gemma_quant_repack.rs:17`) maps the same algebra for Google gemma-QAT
weights but **does** write a real `.biases` array (every entry `-2^(bits-1)·s`) and involves no
`symmetric_zero_point`. It is a parallel implementation — changing one does not change the other.

### Fail-loud guard matrix

Because all nine formats share the triplet shape, dtypes are the only discriminator. Every guard runs
**before** dispatching on `plq.mode`.

| guard                                     | trips when                                                                                          | file:line                                              |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `ensure_dense_weight_floating`            | non-float weight reaches a dense `set_weight` (stripped `.scales`)                                   | `crates/mlx-core/src/models/quant_dispatch.rs:224-234` |
| `ensure_int8_storage_resolves_sym8`       | int8 `.weight` but mode ≠ sym8                                                                       | `quant_dispatch.rs:243-261`                            |
| `ensure_plain_fp8_storage_resolves_fp8_e4m3` | u8 weight + **floating** `.scales` but mode ≠ fp8_e4m3                                             | `quant_dispatch.rs:270-295`                            |
| `ensure_kquant_storage_resolves_kquant`   | int8 `.scales` (q6k) **or** u8 `.scales` + f16 `.biases` (q4k/q5k) but mode is not a K-quant          | `quant_dispatch.rs:307-334`                            |
| `ensure_affine_biases_present`            | affine group has `.weight` + `.scales` but no `.biases`                                              | `quant_dispatch.rs:355-378`                            |
| `resolve_kquant_group`                    | K-quant with wrong dtype or rank; `Ok(None)` **only** if `.scales` is absent                          | `quant_dispatch.rs:417-475`                            |
| `validate_mode_with_type` (C++)           | affine w/o biases; K-quant wrong scales dtype / missing f16 biases; float modes with non-null biases   | `crates/mlx-sys/mlx/mlx/ops.cpp:4678-4749`             |
| `quantization_params_from_mode` (C++)     | any non-affine mode whose `(gs, bits)` differ from the pinned pair                                    | `crates/mlx-sys/mlx/mlx/ops.cpp:4661-4671`                                    |

`ensure_affine_biases_present` is live in four families: gemma4 (6 non-test call sites —
`crates/mlx-core/src/models/gemma4/persistence.rs:1361`, `:1390`, `:1586`, `:1850`, `:2140`, `:2205`),
qwen3_5, qwen3_5_moe, lfm2. Without it the failure surfaces as MLX C++'s anonymous "Biases must be
provided for affine quantization".

The converter mirrors these before writing: `validate_existing_quantized_entry` /
`validate_existing_kquant_entry` refuse to relabel already-packed bytes
(`crates/mlx-core/src/convert.rs:5620`). Dtype-cast preservation is content-keyed, not suffix-keyed:
`kquant_biases_to_preserve` freezes an f16 `.biases` only when its sibling `.scales` is int8/u8
(`:5455`), and `sym8_scales_cast_action` normalizes an f16/bf16 `[N]` scale next to an int8 weight
*up* to f32 (`:5485`).

## Recipes & the per-tensor decision engine

```rust
enum QuantDecision { Skip, Default, Custom { bits: i32, group_size: i32, mode: String } }
// crates/mlx-core/src/convert.rs:3909
```

There are **six predicate-builder functions**; `build_predicate_for_recipe`
(`crates/mlx-core/src/convert.rs:5207`) is a dispatcher, not a builder:

| builder                          | line   | serves                                              |
| -------------------------------- | ------ | --------------------------------------------------- |
| `build_recipe_predicate`         | `3949` | all four `mixed_*` (one dispatcher arm at `:5214`)   |
| `build_qwen35_recipe`            | `4049` | `qwen3_5`                                           |
| `build_unsloth_recipe`           | `4172` | `unsloth` (legacy affine, imatrix-gated)            |
| `build_official_unsloth_recipe`  | `4470` | the **fixed** Unsloth class map (Mxfp / Nvfp4)      |
| `build_nvidia_recipe`            | `4637` | `nvidia`                                            |
| `build_privacy_filter_predicate` | `4736` | privacy-filter (never calls `should_quantize`)      |

Seven `--q-recipe` names are accepted (`crates/mlx-core/src/convert.rs:2122`); nine dispatch paths
exist once the fixed map and privacy-filter are counted; `resolve_legacy_entry` (`:6233`) is the tenth
path and the only sym8-aware one.

### Recipe × tensor-class matrix

`B` = `--q-bits`, `G` = `--q-group-size`. Under `--q-mode nvfp4` the recipe builder is handed
`recipe_gs = 64` instead of the top-level 16, because affine only accepts {32,64,128}
(`crates/mlx-core/src/convert.rs:2878`).

| Tensor class                                 | `mixed_L_H`               | `qwen3_5`         | `unsloth` legacy | `unsloth`+`--q-mxfp` (fixed) | `unsloth`+`nvfp4` (fixed) | `nvidia`     | no recipe   |
| -------------------------------------------- | ------------------------- | ----------------- | ---------------- | ---------------------------- | ------------------------- | ------------ | ----------- |
| `embed_tokens` / `embedding.`                | Skip                      | Skip              | `snap(B+2)`/G    | **Skip**                     | **Skip**                  | Skip         | Skip\*      |
| `lm_head`                                    | `H`/G affine              | **Skip**          | `snap(B+3)`/G    | mxfp8 8/32                   | fp8_e4m3                  | mxfp4 4/32   | Skip        |
| `self_attn.q/k/v_proj`                       | `L` (v→`H` in eligible)   | `min(B+2,8)`      | `snap(B+2)`      | mxfp8 8/32                   | fp8_e4m3                  | mxfp8 8/32   | top-level   |
| `self_attn.o_proj`                           | `L`                       | **8/64 affine**   | **8/64 affine**  | mxfp8 8/32                   | fp8_e4m3                  | mxfp8 8/32   | top-level   |
| `linear_attn.in_proj_qkv` / `in_proj_z`      | `L`                       | `min(B+2,8)`      | `snap(B+2)`      | mxfp8                        | fp8_e4m3                  | mxfp8 8/32   | top-level   |
| `linear_attn.out_proj`                       | `L`                       | **8/64 affine**   | **8/64 affine**  | mxfp8                        | fp8_e4m3                  | mxfp8 8/32   | top-level   |
| `linear_attn.in_proj_a` / `in_proj_b`        | `L`                       | **8/64 affine**   | **8/64 affine**  | **Skip (bf16)**              | **Skip (bf16)**           | 8/64 affine  | top-level   |
| `linear_attn.in_proj_ba`                     | Skip                      | Skip              | Skip             | Skip                         | Skip                      | Skip         | Skip        |
| FFN `gate_proj` / `up_proj`                  | `L`                       | `Default`         | `Default`        | layer < N−8 → mxfp4; else mxfp8 | layer < N−8 → nvfp4; else fp8_e4m3 | mxfp4 4/32 | top-level |
| FFN `down_proj`                              | `H` in eligible, else `L` | `min(B+1,8)`      | `snap(B+1)`      | same split                   | same split                | mxfp4 4/32   | top-level   |
| Router gates                                 | 8/64 affine               | 8/64 affine       | 8/64 affine      | **Skip (bf16)**              | **Skip (bf16)**           | 8/64 affine  | 8/64 affine |
| norms / `A_log` / `dt_bias` / `conv1d`       | Skip                      | Skip              | Skip             | Skip                         | Skip                      | Skip         | Skip        |
| vision (`vision_tower`, `visual.`, …)        | Skip                      | Skip              | Skip             | Skip                         | Skip                      | Skip         | Skip        |
| `mtp.*`                                      | Skip                      | Skip              | Skip             | Skip                         | Skip                      | Skip         | Skip        |
| **anything unmatched**                       | `L`                       | `Default`         | `Default`        | **Skip**                     | **Skip**                  | **Skip**     | top-level   |

\* unless `embed_quantizable` (lfm2/lfm2_moe) — but see the note below: no recipe ever receives it.

The last row is the sharpest structural split: `mixed_*` / `qwen3_5` / `unsloth`-legacy are
**allow-by-default**; `build_official_unsloth_recipe` and `build_nvidia_recipe` are
**deny-by-default** (`QuantDecision::Skip` fall-through at `crates/mlx-core/src/convert.rs:4573` and
`:4716`).

Bits arithmetic for `unsloth` (`snap_bits` maps 7 → 8, `crates/mlx-core/src/convert.rs:4176`):

| class                            | formula      | via CLI (`d=3`)  | via direct NAPI (`d=4`) |
| -------------------------------- | ------------ | ---------------- | ----------------------- |
| `gate_proj` / `up_proj`          | `Default`→`d` | 3                | 4                       |
| `down_proj`                      | `snap(d+1)`  | `snap(4)` = 4    | `snap(5)` = 5           |
| `embed_tokens`                   | `snap(d+2)`  | `snap(5)` = 5    | `snap(6)` = 6           |
| `lm_head`                        | `snap(d+3)`  | `snap(6)` = 6    | `snap(7)` = **8**       |
| `q/k/v_proj`, `in_proj_qkv/z`    | `snap(d+2)`  | 5 + AWQ          | 6 + AWQ                 |
| `o_proj`, `out_proj`, `in_proj_a/b` | pinned    | 8-bit affine gs 64 | 8-bit affine gs 64    |

The `d=3` column reproduces the documented ladder (`packages/cli/src/commands/convert.ts:87`). `d=3`
is a **TS-only injection** at `packages/cli/src/commands/convert.ts:383` — see gotchas.

### `should_quantize` — the shared floor

Returns `false` for (`crates/mlx-core/src/convert.rs:3317-3396`):

| #  | test                                                              | line   |
| -- | ----------------------------------------------------------------- | ------ |
| 1  | key does not end in `.weight`                                     | `3319` |
| 2  | `vision_tower` or `visual.`                                       | `3324` |
| 3  | `vision_embedder` (loader has no quantized branch)                | `3331` |
| 4  | `lm_head` — **unconditional**                                     | `3338` |
| 5  | `embed_tokens` / `embedding.` unless `embed_quantizable`          | `3345` |
| 6  | `layernorm` / `rms_norm` / `_norm.`                               | `3350` |
| 7  | `conv1d`                                                          | `3355` |
| 8  | ends with `conv.conv.weight` (LFM2 depthwise short conv)          | `3365` |
| 9  | `A_log` / `dt_bias`                                               | `3370` |
| 10 | `in_proj_ba.` (fused GDN low-rank; split a/b are NOT excluded)    | `3384` |
| 11 | `is_mtp_key`                                                      | `3391` |

Recipes that quantize `lm_head` / `embed_tokens` branch **before** calling it: `mixed_*` (`:3988`),
`unsloth` (`:4198`, `:4205`), fixed-Unsloth (`:4534`), `nvidia` (`:4648`). `qwen3_5` has no such
branch, so `--q-recipe qwen3_5` always leaves both bf16.

**Every recipe passes `embed_quantizable = false` hard-coded** — `:3997`, `:4058`, `:4213`, `:4538`,
`:4658`. The real flag is read only on the no-recipe ladder (`:6246`) and in `quantize_weights_inner`'s
`Default` arm (`:6561`). So lfm2's packed-embedding support is unreachable through any recipe except
`build_unsloth_recipe`'s explicit pre-`should_quantize` branch at `:4198` — which fires for every
family, not just lfm2.

### `--q-mxfp` — two unrelated meanings

**(a) Fixed-map selection.** When `select_official_unsloth_recipe`
(`crates/mlx-core/src/convert.rs:4416`) returns `Some(kind)`, **no upgrade wrapper runs at all**
(`:2889`/`:2891` both require `official_unsloth_kind.is_none()`), and `--q-bits`/`--q-group-size` have
zero effect — `build_official_unsloth_recipe` takes no bits arguments. `is_qwen35_hybrid` requires
three independent agreements (`:4329`): the config's own `model_type` collapses to a qwen3_5 family,
the caller's `--model-type` collapses to the SAME family, and the sanitized keys carry all four of
`self_attn.q_proj` + `linear_attn.in_proj_qkv` + `in_proj_z` + `out_proj` (`:4308`).

The map's only depth rule: `final_eight_start = num_layers.saturating_sub(8)`, with `num_layers`
computed **after** filtering MTP / `vision_tower` / `visual.` / `vision_embedder` keys (`:4476`) —
the only builder that does this filtering.

**(b) Upgrade wrapper.** Branch order is load-bearing (`crates/mlx-core/src/convert.rs:4815-4887`):

```
original = inner(key)
 1. is_affine_only_key(key)                        → return original UNCHANGED     :4820
    (lm_head | router.proj | embed_tokens* | embedding_projection)
 2. is_bitexact_affine_proj(key) AND original == Custom{8, *, "affine"}
                                                    → return original UNCHANGED     :4836
 3. is_router_gate(key)  → Skip stays Skip; anything else → Custom{8, 64, affine}   :4849
 4. match original: Skip              → Skip
                    Default + default_bits==8 → mxfp8 8/32
                    Default + default_bits==4 → mxfp4 4/32
                    Default + ANYTHING ELSE   → Default   (SILENT pass-through)     :4863
                    Custom{bits:8,..}         → mxfp8 8/32
                    Custom{bits:4,..}         → mxfp4 4/32
                    other                     → unchanged
```

`apply_nvfp4_upgrade` (`crates/mlx-core/src/convert.rs:5139`) is **not** the same shape:

| difference                     | `apply_mxfp_upgrade`                 | `apply_nvfp4_upgrade`                            |
| ------------------------------ | ------------------------------------ | ------------------------------------------------ |
| `is_bitexact_affine_proj` arm  | present (`:4836`)                    | **absent** — no equivalent (`:3890` says none needed) |
| `is_affine_only_key` arm       | returns `original` untouched (`:4820`) | **rewrites** `Default` → `Custom{8,64,affine}` (`:5155`) |
| `Default` arm                  | branches on `default_bits` (`:4863`) | promotes **unconditionally** to nvfp4 4/16 (`:5185`) |

### The two decision-scoping traps

**Trap A — `is_bitexact_affine_proj` must be decision-scoped.** It is
(`crates/mlx-core/src/convert.rs:4836`):

```rust
if is_bitexact_affine_proj(key)
    && matches!(&original, QuantDecision::Custom { bits: 8, mode, .. } if mode == "affine")
{ return original; }
```

Key-only would break `mixed_*` + `--q-mxfp`: `mixed_4_6` assigns `o_proj` / `out_proj` /
`in_proj_a` / `in_proj_b` its **low** bits — `Custom{4, gs, affine}` (`:4026`), not 8. A key-only
guard returns that untouched, so `--q-mxfp` becomes a silent partial no-op on exactly four tensor
classes. Regression test: `apply_mxfp_upgrade_promotes_non_pinned_low_bit_projections` (`:8394`),
paired with `apply_mxfp_upgrade_preserves_bitexact_affine_projections` (`:8312`).

It must also **not** be folded into `is_affine_only_key` (`:3886`), because that helper is also
consulted by the no-recipe ladder (`:6311`), where listing `o_proj` would stop a uniform
`--q-mode mxfp8` from upgrading it.

**Trap B — `sym8_eligible` must not be asked about a packed tensor.** It reads the ARRAY:
`ndim == 2 && K % 16 == 0` (`:5546`). A packed affine weight is `Uint32 [N, K/8]` — for `[8, 128]`
that is 2-D with `128 % 16 == 0`, so it answers "eligible" while describing the *packing*.
`resolve_legacy_entry` therefore takes a `for_existing` flag and swaps the question for a dtype
witness (`:6345`):

```rust
let stays_sym8 = match weights.get(key) {
    Some(array) if for_existing => array.dtype()? == DType::Int8,   // stored-format witness
    Some(array) => sym8_eligible(array)?,                            // fresh float weight
    None => false,
};
```

Named mutation at `crates/mlx-core/src/convert.rs:10310`.

### `--q-mtp` policies

CLI accepts `off | cyankiwi | all | split | drafter`
(`packages/cli/src/commands/convert.ts:265`); Rust normalizes `drafter → split` at
`crates/mlx-core/src/convert.rs:1960`. Non-`off`, non-`split` requires `--quantize` **and**
`--q-recipe` (`:1987`).

```
mtp_quant_decision(key)                                   convert.rs:3790
  policy == "off"        → None            (delegate entirely)
  !is_mtp_key(key)       → None            (delegate)
  key lacks ".weight"    → Some(Skip)
  prefix = normalize_mtp_prefix(key minus ".weight")
    is_mtp_layer_quantizable_prefix(prefix)       → Custom{4, 32, "affine"}
    OR (policy == "all" AND prefix == "mtp.fc")   → Custom{4, 32, "affine"}
    else                                          → Some(Skip)
```

| policy               | MTP layer linears        | `mtp.fc`  | norms | storage                                                       |
| -------------------- | ------------------------ | --------- | ----- | ------------------------------------------------------------- |
| `off`                | inner predicate (Skip)   | Skip      | Skip  | —                                                             |
| `cyankiwi`           | affine 4/32              | **Skip**  | Skip  | dense qwen3_5 → `mtp.safetensors`; qwen3_5_moe → inline shards |
| `all`                | affine 4/32              | affine 4/32 | Skip | same split                                                    |
| `split` (= `drafter`) | **not wrapped** (like `off`, `:2909`) | — | —  | `mtp.*` pulled into a bf16 `mtp-drafter/` dir                 |

`MTP_QUANT_BITS = 4`, `MTP_QUANT_GROUP_SIZE = 32` are hard-coded
(`crates/mlx-core/src/convert.rs:3303`). Because `apply_mtp_quant_policy` is the **outermost**
wrapper, MTP linears are never touched by either upgrade wrapper — 4/32 affine even under
`--q-mxfp`. The GGUF entry point never wires the policy at all
(`crates/mlx-core/src/utils/gguf.rs:2974`).

### `mixed_*` layer eligibility — and where it diverges from mlx-lm

```rust
// crates/mlx-core/src/convert.rs:3966
let num_layers     = infer_num_layers(weight_keys);   // max layers.N index + 1, NO filtering
let first_boundary = num_layers / 8;
let last_boundary  = num_layers - num_layers / 8;
use_more_bits[i] = i < first_boundary || i >= last_boundary || (i % 3 == 0);
```

```python
# mlx-lm/mlx_lm/convert.py:61
use_more_bits = (index < num_layers // 8
                 or index >= 7 * num_layers // 8
                 or (index - num_layers // 8) % 3 == 2)
```

`num_layers = 48`:

|          | first | last boundary  | middle band (6 ≤ i < 42)               | count |
| -------- | ----- | -------------- | -------------------------------------- | ----- |
| mlx-node | i<6   | i ≥ 48−6 = 42  | `i%3==0` → {6, 9, …, 39}               | 24    |
| mlx-lm   | i<6   | i ≥ 7·48//8 = 42 | `(i−6)%3==2` ⇒ i≡2 (mod 3) → {8,11,…,41} | 24  |

Same count, **different layers** — the middle-band phase is offset by 2. `num_layers = 36`:
mlx-node's last boundary is 32, mlx-lm's is `7*36//8 = 31`, so layer 31 is high-bit there and low-bit
here.

Three further deltas: mlx-lm also gives high bits to `v_a_proj`/`v_b_proj` which
`key.contains("v_proj")` does not match (`:4016`); mlx-node adds a router-gate 8/64 pin mlx-lm has no
equivalent of (`:4002`); and `infer_num_layers` (`:3932`) does **no** vision/MTP filtering, so on a
VLM whose vision tower is deeper than the LM every boundary is wrong. The doc comment at `:3945`
calls this "Logic (from mlx-lm)" — it is a paraphrase, not a port.

## GGUF import

Reachable only through `#[napi] convert_gguf_to_safetensors`
(`crates/mlx-core/src/utils/gguf.rs:2477`), plus the K-quant repacker
(`crates/mlx-core/src/utils/gguf_kquant.rs`).

```
1. HEADER, no payload read
   parse_gguf  gguf.rs:459   magic 0x46554747, version >= 3 (no upper bound)
     alignment = general.alignment ?? 32, .max(1)                 :504
     tensor infos — UNKNOWN TYPE ⇒ HARD REJECT, whole file        :530
     data_offset = align(stream_pos)                              :550

2. HEADER-ONLY GUARDS  (all run before the destructive File::create)
     :2591  --gguf-kquant + requantize     (only if the file HOLDS K-quants)
     :2633  --quantize over Q4_0/Q8_0
     :2678  secondary output carries symmetric or K-quant tensors
     :2733  gemma4 K=V detected + synthesized config + head_count_kv can't state both

3. PAYLOAD   load_gguf_tensors  :1058
     k_quant_format().is_some()  ─┬─ import_k_quants → load_kquant_repack   :970 (4 MiB chunks)
                                  ├─ off + Q6_K + arch=="gemma4"
                                  │    + name=="token_embd.weight"
                                  │        → load_q6k_tensor_bf16 (DEQUANT) :633
                                  └─ else → reject                          :1101
     is_mlx_affine_quantized()   → load_quantized_tensor  :759  (WHOLE tensor into RAM)
     otherwise                   → load_unquantized_tensor :578

4. POST, in this exact order
     remap_keys :1365 (collision = HARD ERROR) → fixup_gemma4_mmproj_layout :1408
     → fixup_shapes :1477 → fixup_qwen35_linear_attn :1565 → dtype cast :2804
     → AWQ imatrix :2842 → optional re-quantize :2941 → vlm_key_prefix :3061

5. WRITE
     save_safetensors :3096   ONE file, no index.json   ◀ DESTRUCTIVE from here
     if is_primary_model (filename == "model.safetensors"):
        config.json  (authoritative or extract_config :1753)
        gemma4 attention_k_eq_v / layer_types :3133
        quantization block :3155 / preserved_source_quantization :3166
        12-file runtime asset allowlist :3193
```

Guards 2 are hoisted deliberately: `File::create` truncates an existing checkpoint the instant it
runs (`crates/mlx-core/src/utils/gguf.rs:2651`). **This is not universal** — three refusals still run
after the save: `--config-dir` without a `config.json` (`:3109`), the source-quant profile collision
inside `preserved_source_quantization` (only called from `:3166`/`:3182`), and the asset-copy /
config-write `?` at `:3215` / `:3178`.

### Source-type routing

Nine types are recognized (`GgufTensorType`, `crates/mlx-core/src/utils/gguf.rs:42`). Anything else is
a **hard error at header parse** — the whole file is refused even if one tensor uses an unlisted type.

| ggml type (id) | block | `type_size` | route                                  | mlx-node bytes / 4096-col row | ggml bytes | Δ    |
| -------------- | ----- | ----------- | -------------------------------------- | ----------------------------- | ---------- | ---- |
| F32 (0)        | 1     | 4           | dense (`:597`)                         | —                             | —          | —    |
| F16 (1)        | 1     | 2           | dense (`:604`)                         | —                             | —          | —    |
| BF16 (30)      | 1     | 2           | dense (`:611`)                         | —                             | —          | —    |
| Q4_0 (2)       | 32    | 18          | affine repack (`:808`)                 | 512·4 + 128·2 = **2304**      | 2304       | **0** |
| Q4_1 (3)       | 32    | 20          | affine repack (`:831`)                 | 2048+256+256 = **2560**       | 2560       | **0** |
| Q8_0 (8)       | 32    | 34          | affine repack (`:854`)                 | 1024·4 + 256 = **4352**       | 4352       | **0** |
| Q4_K (12)      | 256   | 144         | K-quant repack, needs `--gguf-kquant`  | 2048+256+64 = **2368**        | 2304       | +64  |
| Q5_K (13)      | 256   | 176         | K-quant repack, needs `--gguf-kquant`  | 2560+256+64 = **2880**        | 2816       | +64  |
| Q6_K (14)      | 256   | 210         | K-quant repack **or** BF16 dequant     | 3072+256+32 = **3360**        | 3360       | **0** |
| everything else | —    | —           | **rejected at `:530`**                 | —                             | —          | —    |

`block_size()` (`:99`), `k_quant_format()` (`:127`) and `SourceQuantProfile::for_gguf_type` (`:1902`)
are all `_`-free exhaustive matches, so adding a type forces every routing site to be updated.

Memory behaviour differs per route: K-quant streams in 4 MiB chunks (`:903`); the affine repack reads
the **whole tensor into a `Vec<u8>` at once** (`:769`); the Q6_K→BF16 fallback streams one 210-byte
block but allocates the destination in full.

### Name maps

Dispatch is by metadata only (`gguf_name_to_hf_for_metadata`,
`crates/mlx-core/src/utils/gguf.rs:1252`):

| condition                                                                          | map                             | can return `None`?          |
| ---------------------------------------------------------------------------------- | ------------------------------- | --------------------------- |
| `general.architecture == "gemma4"`                                                 | `gemma4_name_to_hf` (`:1178`)   | **yes** — `rope_freqs.weight` only |
| `arch == "clip"` and (`clip.vision.projector_type == "gemma4uv"` or `clip.audio.projector_type == "gemma4ua"`) | `gemma4_mmproj_name_to_hf` (`:1219`) | no |
| otherwise                                                                          | `gguf_name_to_hf` (`:1266`)     | no                          |

Where the two LLM maps disagree — the whole reason the gemma4 map exists:

| GGUF infix                    | generic (`:1322`)               | gemma4 (`:1199`)                     |
| ----------------------------- | ------------------------------- | ------------------------------------ |
| `.attn_norm.`                 | `.input_layernorm.`             | `.input_layernorm.`                  |
| `.ffn_norm.`                  | `.post_attention_layernorm.`    | **`.pre_feedforward_layernorm.`**    |
| `.post_attention_norm.`       | `.post_attention_layernorm.`    | `.post_attention_layernorm.`         |
| `.post_ffw_norm.`             | *(unmapped, passes through)*    | `.post_feedforward_layernorm.`       |
| `.layer_output_scale.weight`  | *(unmapped)*                    | `.layer_scalar`                      |

Under the generic map `ffn_norm` and `post_attention_norm` both land on `post_attention_layernorm` —
a silent one-of-two loss when collected into a `HashMap`. `remap_keys` (`:1365`) now makes that a
**hard error** ("GGUF key remap collision").

Generic-only Qwen3.5 GDN rules the gemma4 map does not carry (`:1328`) — note every destination
carries the `.linear_attn.` segment, which is load-bearing because `fixup_qwen35_linear_attn`
triggers on `k.contains("linear_attn.")` (`:1570`):

```
.attn_qkv.   → .linear_attn.in_proj_qkv.      .ssm_conv1d. → .linear_attn.conv1d.
.attn_gate.  → .linear_attn.in_proj_z.        .ssm_norm.   → .linear_attn.norm.
.ssm_beta.   → .linear_attn.in_proj_b.        .ssm_dt.bias → .dt_bias
.ssm_alpha.  → .linear_attn.in_proj_a.        .ssm_a       → .A_log
.ssm_out.    → .linear_attn.out_proj.
```

**Global-tensor quant-group rename.** Per-layer rules are infix `replace()`, so they ride along on
any suffix. The three global tensors are whole-string matches and need
`rename_global_quant_group` (`:1167`), which requires the remainder to be exactly one of
`QUANT_GROUP_SUFFIXES = [".weight", ".scales", ".biases"]` (`:1154`):

| from                                    | to                                                  |
| --------------------------------------- | --------------------------------------------------- |
| `token_embd{.weight,.scales,.biases}`   | `model.embed_tokens{…}`                             |
| `output_norm.weight`                    | `model.norm.weight` (exact — norms are never quantized) |
| `output{.weight,.scales,.biases}`       | `lm_head{…}`                                        |
| `mm.input_projection{…}`                | `model.embed_vision.embedding_projection{…}`        |
| `mm.a.input_projection{…}`              | `model.embed_audio.embedding_projection{…}`         |

Moving only `.weight` was the original bug: loaders probe `embed_tokens.scales` as the "this tensor
is quantized" sentinel, so a stranded sidecar silently degrades the group to a bare packed weight
(`:1147`). Pinned by `global_renames_do_not_over_generalize` (`:5940`).

### What is dropped

Exactly one tensor, and only for gemma4:

```rust
// crates/mlx-core/src/utils/gguf.rs:1179  (gemma4_name_to_hf)
if name == "rope_freqs.weight" { return None; }   // RoPE freqs are derived from config at runtime
```

The **generic map has no such rule**: `gguf_name_to_hf("rope_freqs.weight")` matches nothing and
returns the string unchanged (`:1299`), so a non-gemma4 GGUF carries the precomputed RoPE table into
`model.safetensors` under its raw GGUF name.

Is the dropped data recoverable?

| family              | RoPE source at load                                                        | recoverable from GGUF metadata? |
| ------------------- | --------------------------------------------------------------------------- | ------------------------------- |
| qwen3_5 / lfm2 / …  | flat top-level `rope_theta` (serde, `crates/mlx-core/src/models/qwen3_5/config.rs:88`)          | **yes** — `extract_config` writes it from `{arch}.rope.freq_base` (`gguf.rs:1783`) |
| gemma4              | **nested** `rope_parameters.full_attention.{rope_theta, partial_rotary_factor}` + `rope_parameters.sliding_attention.rope_theta` (`crates/mlx-core/src/models/gemma4/persistence.rs:192`, `:441`) | **no** — nothing writes `rope_parameters` |

So for gemma4 with a synthesized config, `parse_rope_parameters(None)` returns the hard-coded triple
`(1_000_000.0, 10_000.0, 0.25)` (`crates/mlx-core/src/models/gemma4/persistence.rs:446`), and the
flat `rope_theta` in the emitted `config.json` is **dead**. `partial_rotary_factor` has no GGUF source
at all.

### Config synthesis vs `--config-dir`

```
asset_dir  = config_source_dir  OR  input_path.parent()  OR  "."      gguf.rs:2711
src_config = asset_dir/config.json
synthesized_config = !src_config.exists()

--config-dir given, no config.json inside  → HARD ERROR   :3109  (AFTER the save)
--config-dir absent, no config.json beside → SILENT fall-through to extract_config  :3117
```

`extract_config` (`:1753`) can produce at most **12 flat keys**: `_name_or_path`, `model_type`
(the GGUF arch string verbatim — `qwen3`, never `qwen3_5`), `hidden_size`, `num_hidden_layers`,
`intermediate_size`, `num_attention_heads`, `num_key_value_heads` (**scalar only**),
`max_position_embeddings`, `rope_theta`, `rms_norm_eps`, `vocab_size`, `head_dim`.

It **cannot** produce: `vision_config`, `audio_config`, `rope_parameters`, `partial_rotary_factor`,
`tie_word_embeddings`, `layer_types`, `sliding_window`, PLE fields, `architectures`, or any tokenizer
artifact. Its value match at `:1790` has a `_ => {}` arm, so **any array-valued metadata is silently
dropped** — which is exactly why gemma4's array-spelled `head_count_kv` needs
`apply_gemma4_attention_geometry` (`:2257`), and why that helper runs **only when
`synthesized_config`** (`:3150`).

The runtime-asset allowlist is exactly 12 filenames (`:3193`): `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json`,
`chat_template.jinja`, `generation_config.json`, `preprocessor_config.json`,
`video_preprocessor_config.json`, `processor_config.json`, `viterbi_calibration.json`. Copy semantics
are **asymmetric** (`:3207`): with `--config-dir` a failed `fs::copy` is a hard error; without it the
same failure is only a `warn!`.

### Family inference from tensor presence

Gemma4's global layers reuse the key projection as V, so llama.cpp never writes `blk.N.attn_v`. No
metadata key records this — the tensor list is the only witness
(`gemma4_layer_types_from_missing_v`, `crates/mlx-core/src/utils/gguf.rs:2064`):

```
collect blk.N indices                                              :2065
missing_v = layers with no blk.N.attn_v*                           :2075
  ├─ empty OR == all layers        → Ok(None)   NOT an error       :2084  ◀ load-bearing guard
  ├─ blocks not 0..N contiguous, or gemma4.block_count disagrees → Err     :2104
  ├─ head_count_kv NOT spelled as an array → Ok(Some(types))       :2130  (silence ≠ contradiction)
  ├─ array len != layer count → Err                                :2135
  ├─ argmin(head_count_kv) set != missing_v set → Err              :2158
  └─ else → Ok(Some(types))
```

The all-present/all-absent guard matters: a truncated download is also missing `attn_v`, and reading
that as K=V suppresses the loader's own missing-weight check and feeds attention the keys as values
(`:2031`). An absent header is tolerated (`:2048`) only because the caller closes the hole it
creates — the `:2733` refusal. Results are written **non-destructively** (`:3133`): `attention_k_eq_v`
and `layer_types` are set only if the config does not already state them, probed via
`gemma4_config_states` (`:2188`) which checks `text_config` first then top level.

Qwen3.5 hybrid detection (`is_qwen35_hybrid_gguf`, `:1738`) requires two witnesses: architecture ∈
{`qwen35`, `qwen35moe`, `qwen3`} **and** `has_qwen35_hybrid_weight_shape`
(`crates/mlx-core/src/convert.rs:4308`). Ordinary Qwen3 fails the shape check.

`fixup_qwen35_linear_attn` (`:1565`) is architecture-independent — it triggers purely on the presence
of any `linear_attn.` key. Geometry comes from metadata with **hard-coded fallbacks**:
`{arch}.ssm.state_size` defaults to 128, `{arch}.ssm.inner_size` to 4096 (`:1584`), `qk_dim` to 4096
(`:1598`). If `n_value_heads < 2` the whole fixup silently returns `Ok(())` (`:1613`). `A_log` also
becomes `log(-x)` because GGUF stores `-exp(A_log)` (`:1626`).

### dtype policy

`GgufConversionOptions.dtype` accepts `float32|f32`, `float16|f16`, `bfloat16|bf16`
(`crates/mlx-core/src/utils/gguf.rs:2805`). The napi doc says *"default: keep original"* (`:2404`) —
but the CLI never passes `None`: `const dtype = args.dtype || 'bfloat16'`
(`packages/cli/src/commands/convert.ts:512`).

The cast loop (`:2817`) skips a key if any of four tests hit:

| # | test                                | line   | covers                                                  |
| - | ----------------------------------- | ------ | -------------------------------------------------------- |
| 1 | `preserve_dtype_keys.contains(key)` | `2819` | every K-quant-sourced output, named explicitly            |
| 2 | `key.ends_with(".scales")`          | `2823` | affine f16 scales, K-quant int8/uint8 sub-scales          |
| 3 | `key.ends_with(".biases")`          | `2823` | Q4_1 f16 minima, K-quant f16 `d`/`dmin`                   |
| 4 | `arr.dtype() == DType::Uint32`      | `2827` | every packed weight plane                                 |

`preserve_dtype_keys` (`:2761`) is built from the source tensor list mapped through the **same**
`gguf_name_to_hf_for_metadata` that `remap_keys` uses, then expanded by `k_quant_output_keys`
(`:947`). Tests 2-4 only *happen* to cover those keys today, which is why the producer and the
do-not-cast set are named through one helper and cannot drift (`:2749`).

## SafeTensors inputs, families, foreign formats

### Weight-file discovery — first match wins

```
model.safetensors             exists? → single lazy load            convert.rs:2361
weights.safetensors           exists? → single lazy load                     :2380
model.safetensors.index.json  exists? → sharded                              :2394
else → Err "No model weights found"                                          :2441
```

The index is used **only to derive the set of shard filenames**
(`let shard_files: HashSet<String> = index.weight_map.values().cloned().collect();`, `:2403`), and
shards merge with `all_tensors.extend(...)` (`:2427`). Tensor *names* in `weight_map` are never
cross-checked in either direction.

Output sharding is fixed and does not depend on the input layout
(`crates/mlx-core/src/utils/safetensors.rs:862`):

```
MAX_SHARD_SIZE = 5 << 30 = 5 × 1,073,741,824 = 5,368,709,120 bytes = 5 GiB   safetensors.rs:330
1. sort tensor names lexicographically                (deterministic; mlx-lm uses insertion order)
2. byte_size = array.size() * dtype.byte_size()
3. greedy bin-pack, close when current + next > 5 GiB, guarded by !current_shard.is_empty()
   (that guard prevents mlx-lm's empty leading shard when tensor 0 exceeds the budget)
naming: 1 shard → model.safetensors ; N>1 → model-{i+1:05}-of-{N:05}.safetensors
model.safetensors.index.json is written ALWAYS, even for one shard        :967
```

### Family resolution — two independent stages

```
config.json ──▶ TS auto-detect ──▶ modelType: string | undefined
                convert.ts:602-651        │
                                          ├─ pp-lcnet-ori | uvdoc → convert_foreign_weights()
                                          │
                                          └─ convertModel({modelType, …})
                                                 recipe_for(model_type)     convert.rs:1760
                                                   Some(recipe)  → recipe.sanitize(...)
                                                   None + Some(mt) → Err "Unknown model type"  :2688
                                                   None (mt ABSENT) → tensors pass through
                                                                       UNSANITIZED, no error   :2699
```

TS auto-detect arms (`packages/cli/src/commands/convert.ts:602-651`):
`paddleocr_vl` → `paddleocr-vl`; `qwen3_5_moe`/`qwen3_5` passthrough; `gemma4`/`gemma4_text` →
`gemma4`; `gemma4_unified` **kept raw**; `architectures[]` containing
`Gemma4UnifiedForConditionalGeneration` → `gemma4_unified`; `lfm2_moe`/`lfm2` passthrough;
`openai_privacy_filter` → `privacy-filter`. There is **no arm** for `qianfan-ocr` (its HF
`model_type` is `internvl_chat`, `crates/mlx-core/src/models/qianfan_ocr/config.rs:109`),
`pp-lcnet-ori`, or `uvdoc`.

`gemma4_unified` is deliberately not collapsed: the E2B prequantized importer gates on the exact
string `"gemma4"` (`crates/mlx-core/src/convert.rs:2266`), so collapsing would route an
audio-carrying unified QAT into an importer that drops audio.

Native registry: 9 `model_type` strings → 6 recipe impls (`crates/mlx-core/src/convert.rs:1745`).

| `model_type`               | impl                            | struct line |
| -------------------------- | ------------------------------- | ----------- |
| `qwen3_5`, `qwen3_5_moe`   | `Qwen35Recipe { is_moe }`       | `219`       |
| `lfm2`, `lfm2_moe`         | `Lfm2Recipe`                    | `1189`      |
| `paddleocr-vl`             | `PaddleOcrVlRecipe`             | `1520`      |
| `qianfan-ocr`              | `QianfanOcrRecipe`              | `1540`      |
| `privacy-filter`           | `PrivacyFilterRecipe`           | `1562`      |
| `gemma4`, `gemma4_unified` | `Gemma4Recipe`                  | `1593`      |

### Recipe asymmetry flags

Five behaviour flags on the trait (`crates/mlx-core/src/convert.rs:153`), all defaulting to the
conservative value:

| flag (default)                     | qwen3_5 | qwen3_5_moe | lfm2 / lfm2_moe | paddleocr-vl | qianfan-ocr | privacy-filter | gemma4\* |
| ---------------------------------- | ------- | ----------- | --------------- | ------------ | ----------- | -------------- | -------- |
| `owns_dtype_cast` (false)          | **true** (`1166`) | **true** | **true** (`1505`) | false  | false       | false          | false    |
| `embed_quantizable` (false)        | false   | false       | **true** (`1509`) | false      | false       | false          | false    |
| `sym8_supported` (false)           | **true** (`1170`) | **true** | **true** (`1513`) | false  | false       | false          | **true** (`1736`) |
| `quant_managed_by_sanitizer` (false) | false | false       | false           | false        | false       | **true** (`1584`) | false  |
| `has_mtp` (None)                   | **Sidecar** (`1179`) | **Inline** | None      | None         | None        | None           | None     |

`owns_dtype_cast = true` bypasses the generic dtype loop entirely (`:2585`), so the hard
`Err("Unsupported target dtype")` at `:2669` is **unreachable** for qwen3_5 / qwen3_5_moe / lfm2 /
lfm2_moe — those families only `warn!` and default to bfloat16 (`:507`, `:1208`).

`fn model_types()` (`:161`) is `#[allow(dead_code)]` — it exists only for the registry-consistency
test.

### Per-family sanitization, in brief

**qwen3_5 / qwen3_5_moe** (`crates/mlx-core/src/convert.rs:486-1165`). Two preflights on the
**untouched** source map first (`:499`): `qwen_vision_quantization_preflight` (`:218`) — the vision
runtime is dense-only, so only one uniform packed mode is allowed — and
`reject_prequantized_qwen_individual_experts` (`:442`). Then five steps: key remap (`:542`) →
dequantize pre-quantized vision groups (`:616`) → **FP8 E4M3 dequant** (`:695`) → expert stacking
(`:846`) → mlx-vlm sanitize (`:1008`: 5-D `patch_embed.proj.weight` transpose, 3-D `conv1d.weight`
transpose when `dim2 <= 16`, RMSNorm `+1.0` on five suffixes).

**lfm2 / lfm2_moe** (`:1196-1503`). Drop tied `lm_head`; `*.conv.conv.weight` 3-D transpose when
`shape[2] > shape[1]`; `w1/w2/w3` → `gate_proj/down_proj/up_proj` renaming **all three** suffixes;
reject any `feed_forward.experts.*` sidecar (`:1306`); stack experts (`:1323`); float-only cast
**excluding `.expert_bias`** (`:1396`) — the repo's analogue of mlx-lm's `cast_predicate`
(`mlx-lm/mlx_lm/models/lfm2_moe.py:389` excludes exactly `expert_bias`); final invariant pass
(`:1454`).

**gemma4 / gemma4_unified** (`:1596-1734`). Longest-first prefix strip (`:1613`); drop `rotary_emb`;
drop `.input_max`/`.input_min`/`.output_max`/`.output_min` **only for language keys** (`:1628`) since
mlx-vlm's `ClippableLinear` needs them on the multimodal side; drop tied `lm_head`; multimodal keys
keep the bare stripped key with two conv transposes (`:1649`); `.experts.gate_up_proj` split on
**axis 1** at `shape[1]/2` → `switch_glu.{gate,up}_proj` (`:1688`); everything else re-prefixed
`language_model.model.{stripped}` (`:1721`).

**paddleocr-vl** (`crates/mlx-core/src/models/paddleocr_vl/persistence.rs:88`). Identity-detect first
(any `language_model.` prefix ⇒ already MLX); otherwise merge visual q/k/v into a single `.qkv.` via
`concatenate_many` on axis 0 (`:114`), `patch_embedding.weight` 4-D transpose, key rewrites (`:14`).

**qianfan-ocr** (`crates/mlx-core/src/models/qianfan_ocr/persistence.rs:111`). Pure key rename plus
one Conv2d transpose gated on the heuristic `shape[1] < shape[2]` (`:85`). Unmatched keys fall
through unchanged.

**privacy-filter** (`crates/mlx-core/src/convert.rs:1567`). **Identity pass** — `Ok(weights)`.
Quantization is owned by a dedicated predicate block (`:2813`) that re-derives a *complete* per-layer
override map from the resulting `.scales` keys.

### Foreign weight formats

`convert_foreign_weights` (`crates/mlx-core/src/utils/foreign_weights.rs:49`) is a **separate NAPI
export**. It never touches `convert_model`, `ConversionRecipe`, quantization, or sharding. It always
writes exactly `model.safetensors` plus a **synthesized** `config.json` — the source directory's
config is not read or copied.

```
model_type dispatch                          foreign_weights.rs:71
  "pp-lcnet-ori" → convert_pp_lcnet_ori      :118
  "uvdoc"        → convert_uvdoc             :168
  other          → Err "Unknown foreign model type"

per-type auto-detect
  dir + inference.pdiparams + inference.json → load_paddle_inference_params  :849
  dir fallback                               → *.pdparams / *.pkl|*.pt|*.pth
  file .pdiparams                            → needs sibling .json, else Err
  file else                                  → pytorch ZIP-pickle           :363
```

`.pdiparams` stream layout per tensor (`:879-987`):
`u32 version` → `u64 lod_count` → LoD levels → `u32 desc version` → `u32 proto_size` →
TensorDesc protobuf → raw payload of `prod(dims) * elem_size` bytes.

Name↔tensor pairing (`:992`) is purely **positional**:

```
names  = extract_param_names_from_json()   // ops with "#":"p", name at A[3]     :1014
ONLY CHECK: names.len() == tensors.len()                                          :992
names.sort_by_key(strip_deepcopy_suffix)                                          :1002
zip(names, tensors_ordered)                //  dims DISCARDED: `(_dims, array)`   :1005
```

Dtype table (`paddle_dtype_to_str`, `:1192`): `2→i32, 3→i64, 4→f16, 5→f32, 6→f64`, **`_ → "f32"`**.
Paddle's `VarType` also defines `BOOL=0, INT16=1, UINT8=20, INT8=21, BF16=22` — none mapped.
`numpy_dtype_size` also defaults to 4 (`:776`), and `parse_tensor_desc` defaults `dtype = 5` when the
field is absent (`:1095`).

### MoE — two source formats, one canonical output

```
Format A (individual)                     Format B (pre-stacked fused)
 …experts.{i}.gate_proj.weight            …experts.gate_up_proj   [E, 2*I, H]
 …experts.{i}.up_proj.weight              …experts.down_proj      [E, H, I]
 …experts.{i}.down_proj.weight
         │ mx.stack(axis=0)                        │ slice axis 1 at I
         ▼                                         ▼
     switch_mlp.{gate,up,down}_proj.weight  [E, out, in]   (qwen, lfm2)
     switch_glu.{gate,up,down}_proj.weight  [E, out, in]   (gemma4)
```

Detection is by key probe and **A wins if both are present**
(`crates/mlx-core/src/convert.rs:858`, warning at `:866`).

**Why FP8 dequant must run BEFORE stacking.** Inside `Qwen35Recipe::sanitize`, Step 2 (FP8 dequant,
`:695`) consumes every `*weight_scale_inv*` pair; Step 3 (expert stacking, `:846`) consumes only
`*.weight`. `dequant_fp8` (`:6801`) is a **2-D** block-wise op — it reads `shape[0]`/`shape[1]`, pads
both to 128, reshapes to `[m/128, 128, n/128, 128]`, and broadcasts
`scale_inv[m_blocks, 1, n_blocks, 1]`. A `[E, N, K]` stacked tensor has no valid interpretation under
that reshape. Stacking first would also orphan every per-expert `weight_scale_inv`, because Step 3's
cleanup only deletes `.weight` keys (`:947`). The per-expert case is additionally rejected outright
before any mutation (`:442`), and lfm2 rejects the same shape by name **and** by dtype
(`:1306`, `:1345`).

3-D stacked experts are NOT excluded on the recipe paths — `quant_entry_emits` only requires
`ndim >= 2` (`:5899`), and `quantize_with_optional_tiling` splits axis 0 into 32-expert tiles once
`E >= 32` to dodge the ~5 s macOS GPU watchdog (`:5235`). They are forced to 8-bit affine **only
under `--q-mode sym8`**, via `sym8_eligible`'s `ndim != 2 → false`.

### VLM — vision stays inline on the SafeTensors path

`convert_model` never writes `vision.safetensors`. Vision/audio tensors stay in the main sharded
output under their family prefix.

| family                     | prefix kept                                     | quantized by the generic pass?                                     |
| -------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| qwen3_5 / qwen3_5_moe      | `vision_tower.`                                 | no — `should_quantize` skips it (`:3323`)                           |
| paddleocr-vl               | `visual.`                                       | no (`:3324`)                                                        |
| gemma4 (SigLIP)            | `vision_tower.`, `multi_modal_projector.`       | no for `vision_tower`; `multi_modal_projector` is **not** excluded  |
| gemma4_unified             | `vision_embedder.`, `embed_vision.`             | `vision_embedder` skipped (`:3328`); `embed_vision.embedding_projection` forced to 8-bit affine by `is_affine_only_key` (`:3865`) |
| gemma4 audio               | `audio_tower.`, `audio_encoder.`, `embed_audio.` | **yes** for `audio_tower`/`audio_encoder` outside sym8 — see gotchas |
| qianfan-ocr                | `vision.`                                       | **yes** — no `vision.` arm in `should_quantize`                     |

The `vision.safetensors` sidecar is a **GGUF-path artifact**, written only by the `--mmproj` second
pass (`packages/cli/src/commands/convert.ts:567`). At load it is **never required**:

```
append_vision_safetensors(dir, load_vision, params)     engine/persistence.rs:176
   if !load_vision           → Ok(())        :181
   if !vision_path.exists()  → Ok(())        :185   ◀ SILENTLY OPTIONAL
   else → params.extend(load_safetensors_lazy(...))
```

It is appended **after** `expand_symmetric_affine_biases` in both branches
(`crates/mlx-core/src/engine/persistence.rs:120`, `:162`) deliberately, so a Q4_0 body + Q8_0 mmproj
cannot inherit each other's zero point.

### MTP

```
is_mtp_key(k)         = strip_wrapper_prefix(k).starts_with("mtp."|"mtp_") || k.contains(".mtp.")
                        crates/mlx-core/src/convert.rs:3399
normalize_mtp_prefix  → models::mtp_drafter::strip_wrapper_prefix (shared longest-first chain)  :3416
```

The delegation is load-bearing: a triple-wrapped `model.language_model.model.mtp.…` must collapse
before the bare-prefix test, or it falls into the language-model remap branch (`:530`).

Carry policy → emission:

```
MtpPolicy::Sidecar (dense qwen3_5)  → extract to mtp.safetensors    convert.rs:2979, :3255
MtpPolicy::Inline  (qwen3_5_moe)    → quantized in place, main shards
MtpPolicy::None    (all others)     → nothing
quant_mtp == "split"                → mtp-drafter/ dir, EXCLUDED from Sidecar     :2977
```

`write_mtp_drafter_dir` (`:3552`) emits `model.safetensors`, a `config.json` with
`model_type: "qwen3_5_mtp"`, `block_size = mtp_num_hidden_layers + 2` (`:3624`),
`tie_word_embeddings` defaulting to **true** when absent (`:3645`), and copies tokenizer assets from
the **source** dir. It first removes stale legacy sidecars (`:3050`), because the dense loader probes
`mtp.safetensors` *before* `mtp-drafter/`.

MTP sanitize is deliberately split from the body: the main `+1.0` norm loop skips `mtp.*`, but an
**independent** probe (`:1069-1083`) samples the *mean* of `mtp.layers.0.input_layernorm.weight` and
shifts the seven MTP norms when `mean < 0.5` (`:1146`). The comment at `:1141` records that a
previous revision skipped `mtp.*` entirely and produced zero MTP acceptance.

Pre-quantized MTP sources are refused when `do_quantize && has_custom_sanitizer`
(`crates/mlx-core/src/convert.rs:2511`, predicate `is_pre_quantized_mtp_key` at `:3409`):
re-quantizing the body rewrites the global `quantization` block, through which `mtp.rs::apply_weights`
resolves missing per-tensor overrides, so an affine-8 MTP head would silently load as NVFP4-4/g16.

### gemma-QAT prequantized import

```
is_gemma_qat_source = config.quantization_config.quant_method == "gemma"     convert.rs:2258
is_gemma_qat_family = nvidia_recipe_family(model_type) == Some("gemma4")
                      && is_gemma_qat_source                                          :2264
   │
   ├─ is_gemma_qat_family + any of -q / --q-recipe / --imatrix-path / --q-mtp → Err    :2267
   ├─ is_gemma_qat_source + unified (model_type or architectures[0])          → Err    :2284
   └─ is_gemma_e2b_import (EXACT --model-type "gemma4")                                :2266
          validate_e2b_qat_schedule(&config)                                           :2306
          import_gemma_prequantized(tensors, &config, dtype)                           :2560
```

All three gates are **pure config reads**, evaluated before the process-wide convert mutex and before
`CpuConvertGuard::enter_cpu()` (`:2252`) — a bad source is rejected without touching MLX.
`validate_e2b_qat_schedule` (`crates/mlx-core/src/convert_gemma_import.rs:92`) compares
`module_quant_configs` against a hard-coded 6-entry regex→bits table (`:77`); nothing is derived from
the checkpoint, so any other gemma4 QAT variant is rejected rather than mis-repacked.

Routing (`crates/mlx-core/src/convert_gemma_import.rs:483`):

| source class                                    | detector          | output                              | override                   |
| ----------------------------------------------- | ----------------- | ----------------------------------- | -------------------------- |
| `embed_tokens_per_layer` (PLE)                  | name (`:585`)     | affine triplet, group 128           | `{4, 128, affine}`         |
| I8 modules (per-layer gates, vision tower)      | `dtype == Int8` (`:633`) | **dequant to target dtype**  | none (dense at runtime)    |
| 2/4-bit U8 linears / `lm_head` / `embed_tokens` | `dtype == Uint8` (`:554`) | affine triplet, group 128    | `{bits, 128, affine}`      |
| floats (norms, projections, conv)               | fallthrough (`:567`) | cast + gemma4 conv transposes    | none                       |

Bit routing: `lm_head`/`embed_tokens` → 2; attention q/k/v/o → 4; MLP → 4 for `layer <= 14`, else 2
(`:292`). Dropped (`:235`): `.input_activation_scale`, `.output_activation_scale`, `.k_cache_scale`,
`.v_cache_scale` — the **entire a8o8 half of "wNa8o8"**. Skipped (`:243`): all audio,
`relative_k_proj`, `.per_dim_scale`, `rotary_emb`.

The lossless mapping (`crates/mlx-core/src/utils/gemma_quant_repack.rs:1`):

```
Google:  w[o,c] = (q_unsigned[o,c] - 2^(bits-1)) * weight_scale[o]
MLX:     w[o,c] =  q_unsigned[o,c] * scales[o,g] + biases[o,g],  g = c/group_size
⇒  pack the RAW nibble/crumb (no subtraction)
   scales[o,g] = weight_scale[o]
   biases[o,g] = -(2^(bits-1)) * weight_scale[o]
```

Size cost at `LINEAR_GROUP_SIZE = 128`, `in = 2048`:

|                              | 4-bit                     | 2-bit                     |
| ---------------------------- | ------------------------- | ------------------------- |
| Google source (per-row f32)  | 4 + 32/2048 = **4.0156**  | 2 + 32/2048 = **2.0156**  |
| MLX affine @ gs 128          | 4 + 64/128 = **4.5**      | 2 + 64/128 = **2.5**      |
| MLX affine @ gs 64 (rejected) | 4 + 64/64 = **5.0**      | 2 + 1.0 = **3.0**         |

Lossless but ~+0.48 bpw of pure redundancy, since every group in a row carries the identical source
scale. 128 is MLX affine's largest legal group, which halves that overhead versus 64.

A convert-time tripwire, `verify_override_coverage`
(`crates/mlx-core/src/convert_gemma_import.rs:268`), **fails the conversion** if any `.scales`-bearing
output lacks a per-layer override. It exists only here — see gotchas.

## Provenance: `config.json` and the loader contract

### The two aliases

Every quantizing converter writes the same object into two keys as byte-identical clones:

```rust
output_config["quantization"]        = quant_obj.clone();
output_config["quantization_config"] = quant_obj;
```

| writer                                       | site                                        | `skip_mtp` |
| -------------------------------------------- | ------------------------------------------- | ---------- |
| SafeTensors convert                          | `crates/mlx-core/src/convert.rs:3131`       | `true`     |
| GGUF → SafeTensors with `--quantize`         | `crates/mlx-core/src/utils/gguf.rs:3155`    | `false`    |
| GGUF → SafeTensors, source-preserved         | `crates/mlx-core/src/utils/gguf.rs:3166`    | n/a        |

Built by one function so the aliases and frontends cannot diverge
(`build_quantization_object`, `crates/mlx-core/src/convert.rs:6726`):

```json
{ "group_size": <int|null>, "bits": <int>, "mode": "<str>",
  "<normalized per-tensor key>": { "bits": …, "group_size": …, "mode": "…" }, … }
```

Per-tensor keys go through `normalize_override_key` (`crates/mlx-core/src/utils/mod.rs:27`), which
forces the mlx-lm/mlx-vlm `language_model.model.*` prefix — except for privacy-filter, which keeps raw
keys (`crates/mlx-core/src/convert.rs:6740`). The loader re-strips wrappers with
`normalize_per_layer_key` (`crates/mlx-core/src/models/quant_dispatch.rs:528`).

Top-level `group_size` is JSON `null` for `sym8` (`crates/mlx-core/src/convert.rs:3132`). It can never
be `fp8_e4m3` — that mode is not in `VALID_QUANT_MODES` (`:1967`), so it only ever appears in
per-tensor children via `serialized_quant_override` (`:5589`).

**MTP is deliberately excluded** (`skip_mtp = true`). Quantized MTP linears are described instead by a
separate `mtplx_mtp_quantization` object pinned to `{4, 32, affine}` (`:3155`).

### When does EVERY quantized tensor get an entry?

There is **no `mixed` field in any emitted config** — `grep -rn '"mixed"' --include=*.rs crates/`
returns nothing. `mixed` is a local `bool` in the GGUF importer.

| emitter                  | rule                                                                             | coverage           |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------ |
| generic quantize         | entry only if `bits != default \|\| group_size != default \|\| mode != default` (`convert.rs:5601`) | **SPARSE by design** |
| privacy-filter           | iterate every `*.scales` key, insert unconditionally (`convert.rs:2843`)          | 100 %              |
| GGUF source-preserved    | modal profile wins top level; entry if `mixed \|\| profile.requires_explicit_entry()` (`gguf.rs:2344`) | see below |
| gemma-QAT import         | `verify_override_coverage` **fails the conversion** on a gap (`convert_gemma_import.rs:268`) | 100 %, enforced |

GGUF modal derivation (`crates/mlx-core/src/utils/gguf.rs:2344`):

```
default_profile = argmax over profiles by (count, profile)   // modal, ties by Ord
mixed           = any tensor profile != default_profile
requires_explicit_entry() = self.mode != "affine"            // ⇒ every K-quant tensor is named
```

`SourceQuantProfile` identity is the full 4-tuple `(bits, group_size, mode, symmetric_zero_point)`
(`:1872`) — deliberately not bits alone, because Q4_0 and Q4_K are both 4-bit through different
kernels, and Q4_0 vs Q4_1 differ only in derived-vs-stored offset.

The gemma-QAT importer is the **only** path that derives an honest top-level block from its own
overrides (`crates/mlx-core/src/convert_gemma_import.rs:139`): it takes the modal `bits` and
hard-errors if the overrides mix `mode` or `group_size` ("no honest top-level `quantization.mode`
exists").

### Loader order of operations

```
load_with_thread(path)                              crates/mlx-core/src/engine/persistence.rs
  └─ load_all_safetensors(dir, load_vision)         :98
       single-file branch                           :102-123
         1. load_safetensors_lazy                   :112
         2. expand_symmetric_affine_biases   ◀ FIRST :120
         3. append_vision_safetensors               :121
       sharded branch                               :125-167
         1. load every shard, extend                :155
         2. expand_symmetric_affine_biases   ◀ SAME ORDER, deliberate  :164
         3. append_vision_safetensors               :165
  ├─ prewarm_checkpoint_pages
  ├─ dequant_fp8_weights
  ├─ sanitize_weights
  ├─ validate_required_weights
  ├─ MoE gate/up fusion
  └─ apply_weights  ◀ LAYER CONSTRUCTION; ensure_affine_biases_present fires here
```

Bias expansion must precede the sidecar join (`:113`): `SymmetricZeroPoints::for_key` falls back to
the top-level default for any key with no entry (`quant_dispatch.rs:725`), so a Q4_0 main model paired
with a Q8_0 mmproj would otherwise derive every vision bias at the wrong offset. "Running afterwards
turns a loud missing-`.biases` failure into silent corruption."

### Loud vs silent

| condition                                                          | behaviour                                                      | site                              |
| ------------------------------------------------------------------ | -------------------------------------------------------------- | --------------------------------- |
| `.biases` present AND config declares `symmetric_zero_point` for it | **Err** "derived or stored, never both"                        | `crates/mlx-core/src/engine/persistence.rs:257`              |
| `.scales` non-floating under a symmetric declaration                | **Err** "an affine scale must be floating"                     | `crates/mlx-core/src/engine/persistence.rs:272`              |
| both aliases present but **different**                             | **Err** "Conflicting quantization aliases … must be identical" | `quant_dispatch.rs:576`           |
| either alias present but not an object                             | **Err** "Invalid {name} alias: expected an object"             | `quant_dispatch.rs:567`           |
| `symmetric_zero_point != 1 << (bits-1)`                            | **Err** "a symmetric group subtracts {expected}"               | `quant_dispatch.rs:701`           |
| `symmetric_zero_point` on a non-affine mode                        | **Err**                                                        | `quant_dispatch.rs:693`           |
| K-quant group missing `.weight` / `.biases` / wrong dtype          | **Err**, names the layer                                       | `quant_dispatch.rs:432`           |
| int8 weight resolving to non-sym8                                  | **Err** "config drift / stale quantization metadata"           | `quant_dispatch.rs:255`           |
| u8 weight + float scales resolving to non-fp8_e4m3                 | **Err** "config drift / missing fp8_e4m3 override"             | `quant_dispatch.rs:288`           |
| per-layer object naming mode/group_size but **no** `bits`          | **Err** — fails the whole load                                 | `quant_dispatch.rs:833`           |
| per-layer object with **none** of bits/group_size/mode/input_amax/symmetric_zero_point | silent `continue`                          | `quant_dispatch.rs:824`           |
| plain-Qwen3 loader given ANY non-empty quantization block          | **Err**, names the mode                                        | `crates/mlx-core/src/models/qwen3/persistence.rs:342`        |
| **`config.json` unreadable / unparseable**                         | **silent** `Ok(0)` — no bias rebuild                           | `crates/mlx-core/src/engine/persistence.rs:227`, `:237`      |
| **config text lacks the literal `symmetric_zero_point`**           | **silent** `Ok(0)`, block never strictly parsed                | `crates/mlx-core/src/engine/persistence.rs:234`              |
| **`config.json` missing/unparseable at quant-settings load**       | **silent** caller defaults + EMPTY override map                | `quant_dispatch.rs:977`           |
| `vision.safetensors` absent when `load_vision = true`              | silent `Ok(())`                                                | `crates/mlx-core/src/engine/persistence.rs:186`              |

The missing-weight message is deliberately bare and uninterpolated (`crates/mlx-core/src/models/gemma4/persistence.rs:502-807`):
`Missing required weight: layers.7.self_attn.q_proj.weight`. Rationale at `:500`: without it "the
branch found no `.weight` to load, and the model silently kept its constructor-RANDOM weights."

### Round-trip (out1 → out2)

There is **no dequantize/requantize path**. Re-converting keeps the packed bytes, so pass 2 must both
prove the bytes match the new request **and** re-emit any override that moves the tensor off the
top-level triple. Phase 1 of `quantize_weights_inner` is read-only against the pristine input map
(`crates/mlx-core/src/convert.rs:6404`):

```
for {base}.weight whose {base}.scales or {base}.weight_scale_inv already exists     :6418
  ── recipe active ────────────────────────────────────────────────
     Skip    → Err "recipe requires this tensor dense"
     Default → !should_quantize ⇒ Err ; else default entry
     Custom  → entry from the recipe decision
     then validate_existing_quantized_entry  :6454
          record_quant_override_if_non_default  :6455
  ── no recipe, has_scales ────────────────────────────────────────
     Uint8 weight + float scales (plain FP8) → Err                 :6475
     else resolve_legacy_entry(for_existing = true)                :6499
          None ⇒ Err "default predicate does not quantize"
     validate (unless deferred to sym8 group coherence)
     record_quant_override_if_non_default                          :6528
```

Validation by resolved mode (`validate_existing_quantized_entry`, `:5620`): affine needs u32 weight +
float scales + **mandatory** float `.biases`; mxfp/nvfp needs u32 + u8 scales and `.biases` **absent**;
sym8 needs int8 + float scales; fp8_e4m3 needs rank 2/3 and no `.biases`; K-quants get a full ggml
geometry check.

The *recording* half is the regression, spelled out at `:6493`:

> "Record: a key the ladder moves off the top-level triple (every router gate, pinned to 8-bit
> affine) needs its per-layer override re-emitted. Without it the config writer stamps the top-level
> triple over 8-bit bytes and the same nameless decode failure comes back — validating alone would
> only have proved the bytes were fine."

The nameless failure (`:6488`): group-32 data relabelled group-64, or K-quant bytes relabelled
affine, survive the write and surface at first decode as `null handle returned: quantized_matmul`,
naming neither the layer nor the shape.

**Second round-trip hazard: the derived-bias claim.** `expand_symmetric_affine_biases` runs at the
converter's *input* boundary (`:2447`), so the output is a *stored*-bias checkpoint.
`strip_symmetric_zero_point` (`:1930`) therefore removes `symmetric_zero_point` from both aliases and
from every child object before writing. Without it, a plain `mlx convert` of an imported model
produces an unloadable model — the loader hits the "derived or stored, never both" contradiction.

### Auditing an artifact — set difference, never the top-level block

```
A = { prefix : "{prefix}.scales" ∈ safetensors index }        # what is actually packed
B = { normalize_per_layer_key(k) : k ∈ config.quantization ,  # what is described
                                   value is an object }
A \ B must be ∅.  Anything in A \ B decodes with the top-level triple —
which for several recipes describes zero tensors on disk.
```

## imatrix / AWQ and FP8 activation calibration

### imatrix format

A standard GGUF v3 whose tensors come in pairs, all F32 — any other type is a hard error
(`crates/mlx-core/src/utils/imatrix.rs:243`):

```
{name}.in_sum2   [input_channels]   sum of squared activations over calibration
{name}.counts    [1]                number of calibration tokens
importance = in_sum2 / counts
```

Keys are mapped `blk.N.ffn_gate.weight → model.layers.N.mlp.gate_proj.weight` via
`gguf_name_to_hf` — imported from `utils/gguf.rs` (`crates/mlx-core/src/utils/imatrix.rs:20`), so name
rules are shared even though the two modules have independent GGUF header parsers.

### AWQ pre-scaling

```
s_j = max(importance_j, 1e-8) ^ ratio                    convert.rs:7187
s   = s / max( sqrt(max(s) * min(s)), 1e-8 )
ratio = 0.5, hardcoded at BOTH entry points:
  crates/mlx-core/src/convert.rs:2724          (SafeTensors)
  crates/mlx-core/src/utils/gguf.rs:2472       (GGUF, reached from gguf.rs:2842)
```

Fold groups (`crates/mlx-core/src/convert.rs:6885`, impl `:6920`):

| group | multiply by `s` (input cols)                                        | divide by `s`                                                        | missing target |
| ----- | -------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------- |
| A     | `mlp.gate_proj`, `mlp.up_proj`                                       | **`pre_feedforward_layernorm` if present, else `post_attention_layernorm`** | **silent**  |
| B     | `mlp.down_proj` (cols)                                               | `mlp.up_proj` (rows)                                                 | n/a            |
| C     | `self_attn.{q,k,v}_proj`                                             | `input_layernorm`                                                    | warns          |
| D     | `linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b}`         | `input_layernorm`                                                    | warns          |

`self_attn.o_proj` and `linear_attn.out_proj` are **not covered by design** — their inputs come from
attention/GDN compute, not a norm (`:6891`).

**Which norm is correct — the code says `pre_feedforward_layernorm` for gemma4.** `:6932` probes
`{prefix}.pre_feedforward_layernorm.weight` first and only falls back. Rationale at `:6926`: gemma4
sandwich layers feed the MLP from `pre_feedforward_layernorm`; their `post_attention_layernorm`
normalizes the attention output into the residual, so folding `1/s` there rescales the attention
branch and leaves gate/up columns uncompensated. Verified against the model: gemma4's decoder layer is
`h = x + post_attention_layernorm(r); r = mlp(pre_feedforward_layernorm(h))`
(`crates/mlx-core/src/models/gemma4/decoder_layer.rs:79`, forward at `:421`). Two-norm families
(qwen3_5, lfm2) have no `pre_feedforward_layernorm` and keep `post_attention_layernorm`.

Group A takes the element-wise max of gate/up importance and accepts a **single** present key
(`:7096`); Groups C/D require ALL keys or skip with a warning, because "partial AWQ correction is
worse than none" (`:7126`).

Recipe acceptance:

| recipe                                                          | imatrix                | gate                                          |
| --------------------------------------------------------------- | ---------------------- | --------------------------------------------- |
| `unsloth` legacy affine (no `--q-mxfp`, mode ≠ nvfp4)           | **required**           | `validate_unsloth_imatrix_selector`, `:4374`  |
| `unsloth` + `--q-mxfp` or `--q-mode nvfp4`                      | optional (warns)       | `:4365`, `:2803`                              |
| `unsloth` no-imatrix that failed family/shape validation        | **hard error**         | `validate_unsloth_imatrix_after_selection`, `:4396` |
| `nvidia`                                                        | **rejected outright**  | `:5069` — "a data-free port; an imatrix would trigger AWQ pre-scaling that silently alters weights" |
| `mixed_*`, `qwen3_5`, no recipe                                 | accepted, no gate      | `:2719`                                       |
| gemma-QAT source / `--gguf-kquant`                              | **rejected**           | `:2267` / `packages/cli/src/commands/convert.ts:503` |

Pre-quantized bodies are rejected before any mutation (`reject_awq_for_prequantized_body`, `:6861`):
"Packed weights cannot be safely AWQ-scaled and this converter has no dequantize/requantize path;
refusing before mutating tensors."

### `mlx calibrate` — FP8 activation amax

State layout (`crates/mlx-core/src/calibration/activation_amax.rs:12`) is split on purpose:

```
CALIBRATING  thread_local Cell<bool>                   :36   arm flag, per model thread
AMAX         LazyLock<Mutex<HashMap<String,f32>>>      :47   running max, process-global
calib_guard  tokio Mutex, try_lock                 napi.rs:40   serializes RUNS (2nd run fails fast)
```

Arming is RAII (`CalibrationArmGuard`, `:131`) so every exit path disarms. `record` (`:91`) folds a
**non-finite-preserving** running max: once a key sees inf/NaN it stays non-finite, because plain
`f32::max` ignores NaN and a later finite sample would erase the evidence.

Exactly six calibrated sites (`is_activation_fp8_site`, `:162`):
`.self_attn.{q,k,v,o}_proj`, `.linear_attn.in_proj_qkvz`, `.linear_attn.out_proj`. `in_proj_ba`
(affine 8/64) is deliberately absent. The forward tap additionally requires `self.mode == MXFP8_MODE`
(`crates/mlx-core/src/models/qwen3_5/quantized_linear.rs:723`).

```
fake-quant           crates/mlx-core/src/quant/fp8_activation.rs:14
  s_in = 448 / amax ;  s_out = amax / 448
  xq   = from_fp8( to_fp8( x.astype(f32) * s_in ) , f32 ) * s_out   → astype(x.dtype)
  amax <= 0.0 ⇒ return x unchanged
  (the f32 upcast before the pre-scale is required for bit-exactness with modelopt)

gate at use          quantized_linear.rs:746
  requires input_amax > 0 AND mode == MXFP8_MODE,
  and is SUPPRESSED entirely while calibrating (so re-calibration measures raw bf16)
```

Pipeline and failure ladder (`crates/mlx-core/src/calibration/napi.rs:230`):

```
readCalibTexts(dataset, calibSize)        calibrate.ts:33   first N rows IN FILE ORDER, no shuffle
        │  0 rows → throw, never reaches native
        ▼
calibrateActivationAmaxRaw(modelPath, rows, calibSeq)
  1. calib_guard().try_lock()                                        :242
  2. read_model_type(config.json) — qwen3_5 | qwen3_5_moe ONLY       :251 / Err at :293
  3. ActivationAmaxCollector::take()   (clear residue from a panic)  :257
  4. per row: tokenize raw text → truncate → PREFILL ONLY
       (no chat template, no generated token; caches reset per row)
  5. rows_prefilled == 0    → Err              config UNTOUCHED      :104
  6. amax map empty         → Ok(0)            config BYTE-UNTOUCHED :154
  7. any amax non-finite    → Err              config UNTOUCHED      :163
  8. write_amax_into_config()                  activation_amax.rs:201
  9. written < expected     → Err                                    :176
```

`write_amax_into_config` iterates **config entries**, not the collected map, because config keys are
raw/wrapped and store the GDN input projection **split** (`in_proj_qkv`, `in_proj_z`) while the
collector uses stripped keys with it **merged** (`in_proj_qkvz`); the merged amax fans out to both
(`:343`). Only per-layer object entries with a parseable integer `bits` gain `input_amax` (`:332`),
mirroring the loader's materialization gate. The write is atomic: serialize to
`.{name}.tmp.{pid}.{nanos}` in the same directory, then `rename(2)` (`:264`). If `homed < expected` it
returns early **without writing** (`:257`).

Stated three times in code plus `docs/cli.md:257-259`: Apple GPUs have no FP8 matmul hardware, so this
is **numeric parity only, not a speedup** (`crates/mlx-core/src/quant/fp8_activation.rs:7`,
`crates/mlx-core/src/models/qwen3_5/quantized_linear.rs:744`).

## Gotchas & footguns

The highest-value section. Everything here is confirmed in code.

### Silent defaults that substitute for missing data

| # | Trap                                                                                                                                                                                                                 | Where                                                                                     |
| - | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1 | **An undetectable or unreadable `config.json` silently disables ALL sanitization.** The TS auto-detect wraps the config read in `try { … } catch {}`, leaving `modelType` undefined; the native driver then takes `None => converted_tensors` — pure dtype conversion, no key remap, no norm shift, no expert stacking, no error, no warning. Only an *explicit* unknown `-m` errors. A `model_type: "qwen3_5_text"` root config (accepted by `qwen35_recipe_family` but absent from the CLI's exact-match list) converts to a checkpoint that loads with raw ~0.0 RMSNorm weights. | `packages/cli/src/commands/convert.ts:648`, `crates/mlx-core/src/convert.rs:2699`, `:2688` |
| 2 | **`load_quant_settings_from_disk` returns the caller's defaults + an EMPTY override map** when `config.json` is missing or is not valid JSON. No warning. An affine-4/64 fallback over bytes that were really affine-4/32 passes every storage guard and decodes garbage. | `crates/mlx-core/src/models/quant_dispatch.rs:977`                                          |
| 3 | **`expand_symmetric_affine_biases` returns `Ok(0)` silently** on an unreadable config, an unparseable config, *and* when the raw config TEXT does not literally contain `symmetric_zero_point`. The last is a deliberate compatibility shim, but it means a corrupt config skips the bias rebuild. Caught loudly only in gemma4 / qwen3_5 / qwen3_5_moe / lfm2. | `crates/mlx-core/src/engine/persistence.rs:227`, `:234`, `:237`                            |
| 4 | **`read_meta_array`'s unsupported-element arm yields an EMPTY `ArrayU32`**, so "unreadable" is indistinguishable from "empty". Only Uint32/Int32/Float32/String arrays decode; Bool, U8, I8, U16, I16, U64, I64, F64 all hit `_ => { … Ok(ArrayU32(Vec::new())) }`. Live victim: `gemma4.attention.sliding_window_pattern` is a bool array, which is why the K=V inference falls back to `head_count_kv`. | `crates/mlx-core/src/utils/gguf.rs:447`, `:2023`                                            |
| 5 | **`extract_config`'s value match has `_ => {}`, dropping every array-valued metadata field.** Gemma4 spells `attention.head_count_kv` as an array, so `num_key_value_heads` is silently dropped — the exact reason `apply_gemma4_attention_geometry` exists. Any other architecture spelling a mapped field as an array gets the same drop, with no rescue helper. | `crates/mlx-core/src/utils/gguf.rs:1790`, `:2242`                                            |
| 6 | **`fixup_qwen35_linear_attn` uses hard-coded geometry fallbacks**: `ssm.state_size` → 128, `ssm.inner_size` → 4096, `qk_dim` → 4096, and silently `Ok(())`s when `n_value_heads < 2`.                                                                                        | `crates/mlx-core/src/utils/gguf.rs:1584`, `:1598`, `:1613`                                  |
| 7 | **Unknown Paddle dtypes silently become f32 and desync the stream parse.** `paddle_dtype_to_str` maps only 2/3/4/5/6; `numpy_dtype_size` also defaults to 4; `parse_tensor_desc` defaults `dtype = 5` when absent. The `.pdiparams` reader is a sequential byte walk, so a wrong `elem_size` lands the offset mid-tensor and every later tensor is garbage. | `crates/mlx-core/src/utils/foreign_weights.rs:1199`, `:776`, `:1095`                        |
| 8 | **Missing MoE config fields fall back to magic numbers with only a `warn!`** — `num_experts` → 256, `num_hidden_layers` → 40. The failure then surfaces as "Missing expert weight: …" naming an expert index, not "num_experts not found".                                     | `crates/mlx-core/src/convert.rs:518`, `:872`                                                |
| 9 | **The `mlx_dequantize` FFI shim substitutes `"affine"` for a null/empty mode string** (`mlx_advanced_ops.cpp:872`) and treats `group_size <= 0` / `bits <= 0` as "use the mode default" (`:869-870`). The `-1` sentinel convention collides with sym8/fp8_e4m3's own `-1` group-size sentinel, so a `-1` arriving here is silently reinterpreted. | `crates/mlx-sys/src/mlx_advanced_ops.cpp:869`, `:872`                                       |

### Silent losses

| # | Trap                                                                                                                                                                                                                            | Where                                                                            |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 10 | **SafeTensors weight-file discovery is first-match-wins**, so a stray single-file `model.safetensors` beside a shard index silently masks every shard. There is no cross-check that the index's `weight_map` is covered — the map is used only to derive the shard *filename* set — and shards merge with `HashMap::extend` over an unordered `HashSet`, so a name in two shards resolves nondeterministically. | `crates/mlx-core/src/convert.rs:2355`, `:2403`, `:2427`                          |
| 12 | **`.pdiparams` name↔tensor pairing is positional with only a count check.** Names are sorted by de-`_deepcopy_N`'d name and zipped against binary order; the decoded `dims` are explicitly discarded (`for (name, (_dims, array)) in …`). If Paddle's on-disk order ever diverges from alphabetical, every tensor gets the wrong name and the conversion reports success. | `crates/mlx-core/src/utils/foreign_weights.rs:992`, `:1002`, `:1005`             |
| 13 | **No multi-part / split GGUF support, and no error when one is passed.** The parser reads no `split.no` / `split.count` / `split.tensors.count` keys. `model-00001-of-00003.gguf` imports only that shard's tensors and exits cleanly with a plausible `numTensors`. | `crates/mlx-core/src/utils/gguf.rs:459`                                          |
| 14 | **`rope_freqs.weight` is dropped only by the gemma4 map**; the generic map writes it into the output under its raw GGUF name, as dead weight the loader does not expect. It is the only tensor any map ever drops. | `crates/mlx-core/src/utils/gguf.rs:1179`, `:1299`, `:1372`                       |
| 15 | **For gemma4, the `rope_theta` extracted from GGUF metadata is DEAD**, and `partial_rotary_factor` has no GGUF source at all. The gemma4 loader reads only a nested `rope_parameters` object that nothing in the converter writes, so a gemma4 GGUF converted without `--config-dir` silently gets `(1e6, 1e4, 0.25)` while the extracted flat field sits in config.json looking authoritative. Other families read it flat and are unaffected — which is what makes this easy to miss. | `crates/mlx-core/src/models/gemma4/persistence.rs:441`, `crates/mlx-core/src/utils/gguf.rs:1783` |
| 16 | **The gemma-QAT importer drops the entire activation half of "wNa8o8"** — `.input_activation_scale`, `.output_activation_scale`, `.k_cache_scale`, `.v_cache_scale`. The imported model is weight-only. | `crates/mlx-core/src/convert_gemma_import.rs:235`                                |
| 18 | **`--config-dir`'s asset copy fails hard, but the implicit alongside-GGUF copy only warns.** Same loop, two branches — without `--config-dir`, a permission error or full disk while copying `tokenizer.json` gives a warning line and a `✓ Converted` banner, leaving an unusable directory. | `crates/mlx-core/src/utils/gguf.rs:3214`                                         |
| 19 | **The `vision.safetensors` media sidecar is silently optional at load.** `append_vision_safetensors` returns `Ok(())` when the file is absent even though `should_load_media_sidecar` returned true. Only a few individual weights have explicit presence checks. | `crates/mlx-core/src/engine/persistence.rs:185`, `crates/mlx-core/src/models/gemma4/persistence.rs:392` |
| 20 | **On the GGUF path an AWQ whole-model no-match is completely silent.** `apply_gguf_awq_prescaling` discards the returned `modified` count, so the SafeTensors path's `warn!("modified == 0")` has no GGUF counterpart. Layer-prefix detection recognizes only `language_model.model.layers.` and `model.layers.`, so a third wrapper spelling gets zero AWQ. | `crates/mlx-core/src/utils/gguf.rs:2466`, `crates/mlx-core/src/convert.rs:2725`, `:6911` |
| 21 | **An imatrix tensor whose `.counts` is ≤ 0 or absent is silently `continue`d** and never enters the importance map. Group A may still fire on the surviving partner, producing a scale derived from half the intended evidence. | `crates/mlx-core/src/utils/imatrix.rs:294`, `crates/mlx-core/src/convert.rs:7103` |

### Flags that do less than they look like

| # | Trap                                                                                                                                                                                                                                  | Where                                                                       |
| - | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 22 | **`--q-mxfp` is a complete no-op on `mixed_2_6` and `mixed_3_6`, and a partial no-op on `mixed_3_4`.** The upgrade arms match only `bits: 8` and `bits: 4`; `mixed_2_6` emits 2- and 6-bit, `mixed_3_6` emits 3- and 6-bit. The CLI prints `--q-mxfp: eligible 8b->mxfp8/4b->mxfp4` with no warning that the eligible set is empty. | `crates/mlx-core/src/convert.rs:4876`, `:3955`, `packages/cli/src/commands/convert.ts:404` |
| 23 | **`--q-bits` is silently ignored by all four `mixed_*` recipes.** `build_predicate_for_recipe` forwards only `default_group_size` to `build_recipe_predicate`, which has no bits parameter — but `--q-bits 8` still changes the recorded top-level `bits`, and therefore which decisions count as overrides. | `crates/mlx-core/src/convert.rs:5214`, `:3949`                              |
| 24 | **`--model-type` is parsed but never read at all on the GGUF path** — not forwarded, not validated, not auto-detected. The GGUF branch returns before the auto-detect block, and `GgufConversionOptions` has no `model_type` field. This is load-bearing for the nvidia gate: GGUF passes `None` for both `config_family` and `requested_model_type`, which is what makes `--q-recipe nvidia` on GGUF reject wholesale. | `packages/cli/src/commands/convert.ts:591`, `crates/mlx-core/src/utils/gguf.rs:2529` |
| 25 | **`--config-dir`, `--mmproj`, and `--imatrix-path` are fully validated by the CLI and then silently discarded on the paths where they do not apply.** `ConversionOptions` has no `config_source_dir` and no mmproj field at all; the foreign path forwards no imatrix. The user sees a successful conversion with no warning. | `packages/cli/src/commands/convert.ts:440-475`, `:669`, `:706`               |
| 26 | **`--q-mtp cyankiwi`/`all` silently no-ops on any non-Qwen model type AND still writes a false `mtplx_mtp_quantization: {prequantized: true}` block.** The "no mtp.* tensors" safety errors are scoped to `MtpPolicy::Sidecar` / `Inline` / `is_split`; for `MtpPolicy::None` nothing rejects it, but the config writer is gated only on `do_quantize && quant_mtp != "off" && !is_split`. | `crates/mlx-core/src/convert.rs:2978`, `:3148`                              |
| 27 | **`--q-recipe` without `--quantize` is silently ignored on the GGUF path** — the guard exists in `convert.rs:2110` and has no GGUF counterpart. Except `unsloth` and `nvidia`, whose validators run *unconditionally* and still hard-error. Inconsistent coverage of one flag. | `crates/mlx-core/src/convert.rs:2110`, `crates/mlx-core/src/utils/gguf.rs:2503`, `:2941` |
| 28 | **`apply_mxfp_upgrade`'s `Default` arm falls through to `Default` for any `default_bits` outside {4, 8}.** Under `--q-recipe unsloth --q-mxfp` on a checkpoint that FAILS official-map validation, any key the legacy predicate leaves `Default` stays 3-bit affine gs 64 while the request said mxfp. The recorded metadata is *correct* (it matches the top-level triple) but the checkpoint is not what the flag promised. | `crates/mlx-core/src/convert.rs:4863`                                       |
| 29 | **The `--gguf-kquant` + re-quantization reject uses two different predicates.** TS rejects on flags alone, unconditionally; Rust rejects only when the source file actually contains K-quant tensors. `importKQuants: true, quantize: true` on a BF16 GGUF is accepted natively and refused by the CLI. | `packages/cli/src/commands/convert.ts:502`, `crates/mlx-core/src/utils/gguf.rs:2591` |
| 30 | **`--mmproj` silently rewrites the MAIN model's tensor keys to the `language_model.*` VLM namespace**, which `--help` does not mention — it describes `--mmproj` purely as "Converts and merges vision weights". The mmproj sub-conversion also hardcodes `dtype: 'bfloat16'`, ignoring `--dtype`. | `packages/cli/src/commands/convert.ts:42`, `:550`, `:569`, `crates/mlx-core/src/utils/gguf.rs:3061` |
| 31 | **The CLI's `-d` default makes the napi field's documented "keep original" behaviour unreachable.** `const dtype = args.dtype \|\| 'bfloat16'` is forwarded unconditionally, so an F32 GGUF is always downcast to bf16 unless `-d float32` is given. | `packages/cli/src/commands/convert.ts:512`, `crates/mlx-core/src/utils/gguf.rs:2404` |
| 32 | **The `mlx calibrate` per-5% progress logging is dead code.** `calibrate()` runs one blocking native call then calls `onProgress?.(rows.length, rows.length)`, so the handler's throttling can only ever emit a single `calibrated N/N rows` line. A 1024×512 calibration prints nothing for its whole runtime, which reads as a hang. | `packages/cli/src/commands/calibrate.ts:91`, `:193`                         |

### Validation that fires too late, or not at all

| # | Trap                                                                                                                                                                                              | Where                                                                       |
| - | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 33 | **Neither layer validates `--q-bits` against MLX's {2,3,4,5,6,8}, nor `--q-group-size` against affine's {32,64,128}.** TS checks `/^[1-9]\d*$/`; Rust checks `> 0` (and even that is missing on the GGUF path). `--q-bits 7 --q-mode affine` or `--q-group-size 48` reaches `mlx_quantize` mid-conversion. `convert.rs:2871` states the affine constraint in a comment while validating nothing. | `packages/cli/src/commands/convert.ts:220`, `crates/mlx-core/src/convert.rs:2019`, `:2871`, `:5320` |
| 34 | **`build_qwen35_recipe` can emit `bits = 7`**, which the repo's own comment says MLX cannot express. `high_bits = (default_bits + 2).min(8)` with NO `snap_bits`. `--q-bits 5` ⇒ attention/GDN at 7; `--q-bits 6` ⇒ `down_proj` at 7. | `crates/mlx-core/src/convert.rs:4053`, `:4176`                              |
| 35 | **`--q-mode nvfp4` without `--q-recipe` is rejected only in Rust, and only after the entire checkpoint is loaded and sanitized.** On a 100 GB MoE that is many minutes of work before a purely static flag error. Same for `--q-mxfp` without a recipe and bits ∉ {4,8}. | `crates/mlx-core/src/convert.rs:2930`, `:2944`, `crates/mlx-core/src/utils/gguf.rs:3012` |
| 36 | **`--config-dir` pointing at a directory with no `config.json` is not detected until AFTER `model.safetensors` has been written.** The TS check only verifies the directory exists. The output is left with valid weights and no config. | `packages/cli/src/commands/convert.ts:466`, `crates/mlx-core/src/utils/gguf.rs:3096`, `:3109` |
| 37 | **The affine GGUF repack performs no divisibility validation**, unlike the K-quant path which rejects a last dim that is not a positive multiple of `QK_K`. `load_quantized_tensor` uses integer division throughout; a 2-row tensor with last dim 48 indexes `scales[2]` out of bounds — a panic across the napi boundary, not a named error. | `crates/mlx-core/src/utils/gguf.rs:776`, `:978`                             |
| 38 | **`Vec::with_capacity` on `tensor_count` and `n_dims` is not covered by `MAX_GGUF_ALLOC`.** The 256 MiB cap applies to string lengths and array element *counts* only (256 M `String` headers ≈ 6 GB). A malformed header can abort the process before a single tensor is parsed. | `crates/mlx-core/src/utils/gguf.rs:511`, `:517`, `:362`                     |
| 39 | **The gemma-QAT "already quantized" reject is narrower than it looks.** It is gated on `is_gemma_qat_family` = `nvidia_recipe_family(model_type) == Some("gemma4") && is_gemma_qat_source`, so a gemma-QAT source converted with a mismatched `-m` (e.g. `-m qwen3_5 -q`) escapes it entirely and falls through to the generic quantizer, which would re-quantize already-quantized weights. Only the *unified* reject hangs off bare `is_gemma_qat_source`. | `crates/mlx-core/src/convert.rs:2264`, `:2267`, `:2284`                     |
| 40 | **`--dtype` is validated inconsistently by family.** For `owns_dtype_cast` families the generic loop is bypassed, so `Err("Unsupported target dtype")` is unreachable and a `warn!` is the only handler. `mlx convert -m qwen3_5 -d float64` prints `Dtype: float64`, writes bf16, exits 0. The same flag on `-m gemma4` hard-errors. | `crates/mlx-core/src/convert.rs:507`, `:2585`, `:2669`                      |
| 42 | **`verify_override_coverage` — the set-difference audit — exists ONLY for the gemma-prequant import.** Its doc explains why: the generic paths' override maps are intentionally sparse, so a `.scales` tensor without an override is normal there. No automated guard will catch a coverage hole on any other path. | `crates/mlx-core/src/convert_gemma_import.rs:262`                           |
| 43 | **A sym8 checkpoint refuses to load on any GPU below Apple gen 17 (M5).** `try_build_sym8_quantized_linear` hard-errors with "sym8 checkpoints require an M5+ GPU". `sym8_eligible` deliberately OMITS this check because it is a runtime property, so conversion succeeds on an M1–M4 box and produces a checkpoint that same box cannot load. Neither `docs/cli.md:144` nor the `--q-mode` help mentions it. | `crates/mlx-core/src/models/qwen3_5/quantized_linear.rs:389`, `crates/mlx-core/src/convert.rs:5546` |

### Guard coverage: TS vs Rust

8 guards live **only** in `packages/cli/src/commands/convert.ts` and **15** live only in Rust. Of the
TS-only ones, only two are genuinely reachable by a direct NAPI caller (`--mmproj` and
`--imatrix-path` path/extension checks) — the rest are structurally impossible or fail later anyway.
Of the Rust-only ones the load-bearing one is the **nvidia family gate**
(`crates/mlx-core/src/convert.rs:5027`, `:5042`, `:5054`): it reads the *input config.json's* own
`model_type`, not `-m`, precisely because the CLI forwards an explicit `-m` verbatim and skips
auto-detection.

Because all three NAPI entry points (`convertGgufToSafetensors`, `convertForeignWeights`,
`convertModel`) are public exports of `@mlx-node/core` (`packages/core/index.cjs:784`), every TS-only
guard is bypassable.

Two consequences worth naming:

- **`--q-recipe unsloth`'s documented 3-bit base is a TS-only default**
  (`packages/cli/src/commands/convert.ts:383`). Rust has none — `default_bits` comes from `--q-mode`
  alone. A direct `convertModel({quantRecipe:'unsloth', quantize:true, imatrixPath:'…'})` NAPI call
  silently produces the `4/5/6/8/6` ladder instead of the documented `3/4/5/6/5`. (Without
  `imatrixPath` the call hard-errors at `crates/mlx-core/src/convert.rs:2148` — it does not silently
  produce anything.)
- The **nvfp4 arm** of `effectiveQuantBits` is *not* redundant. It is a ternary chain:
  `quantMode === 'nvfp4' ? 4 : quantRecipe === 'unsloth' ? 3 : undefined`. For the documented
  `--q-recipe unsloth --q-mode nvfp4` pair the nvfp4 arm **preempts** the unsloth 3-bit arm; delete
  it and the TS nvfp4 invariant check at `convert.ts:430` rejects the invocation with "nvfp4 requires
  bits=4 and group_size=16". The code comment at `convert.ts:379` says this verbatim.

### `config.json` provenance traps

| # | Trap                                                                                                                                                                                                                                               | Where                                                                       |
| - | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 44 | **`--q-recipe unsloth --q-mxfp` writes a top-level `{bits:3, group_size:64, mode:"affine"}` block that describes ZERO tensors.** The recipe branch never updates `quant_mode_effective` / `quant_group_size_effective`; only the no-recipe branch does. It is safe *only* because the fixed official map returns `Custom`/`Skip` for every key and never `Default`, so 100 % of quantized tensors carry an explicit override and the fallback is never taken. On this path the map emits only mxfp8 8/32, mxfp4 4/32, and `Skip` (bf16) — the 8-bit-affine router-gate class belongs to the *upgrade wrapper*, which does not run here. | `crates/mlx-core/src/convert.rs:2914` vs `:2967`; `:3132`; `packages/cli/src/commands/convert.ts:383` |
| 45 | **The same bug class was recognized and fixed for `--q-mode nvfp4` and never for `--q-mxfp`.** The CLI comment: "the unsloth 3-bit default would otherwise produce an inconsistent checkpoint: top-level bits=3 but per-layer overrides at bits=4 … with no failure surface." The nvfp4 top-level `nvfp4/4/16` is genuinely load-bearing — the early-FFN class equals it and correctly gets **no** override. | `packages/cli/src/commands/convert.ts:379`, `crates/mlx-core/src/convert.rs:9889` |
| 46 | **The top-level block is misleading for EVERY recipe artifact.** Auditing by reading `quantization.mode` gives the wrong answer; the only correct audit is the set difference of `.scales` keys against the override map. There is one exception in the other direction: the gemma-QAT import (`convert.rs:2750`) is the sole path that rewrites all *four* effective values, deriving them honestly from its own override map. | `crates/mlx-core/src/convert.rs:2740`, `:2750`, `:2967`, `:6726`             |
| 47 | **There is no `mixed` field in any emitted config.** Any consumer looking for `"mixed": true` to decide whether the top-level triple is trustworthy finds nothing. `mixed` is a local `bool` in the GGUF importer. | `crates/mlx-core/src/utils/gguf.rs:2349`, `:2371`                            |
| 48 | **MTP quantization is excluded from the `quantization` block** and described by a non-standard `mtplx_mtp_quantization` key hard-coded to `{4, 32, affine}` regardless of what the MTP tensors actually are. Any standard loader reading only `quantization` treats them as unquantized. The `cyankiwi` description string also says "Load **calibrated** CyanKiwi MTP layer linears" — nothing is calibrated; `cyankiwi` and `all` differ by exactly one tensor (`mtp.fc`). | `crates/mlx-core/src/convert.rs:3138`, `:3155`, `:3156`, `:3798`            |
| 49 | **The emitted `config.json` is a verbatim clone of the source with only `_name_or_path` removed and quantization injected.** `torch_dtype` is never read or rewritten — an f32 source converted to bf16 emits `torch_dtype: "float32"` beside bf16 tensors, while mlx-lm uses `config["torch_dtype"]` as its default conversion dtype. The source `model_type` is likewise preserved, so a `gemma4_text` input keeps `gemma4_text` even though it was sanitized as `gemma4`. | `crates/mlx-core/src/convert.rs:3122`, `mlx-lm/mlx_lm/convert.py:131`        |

### Naming and layout traps

| # | Trap                                                                                                                                                                                                                        | Where                                                                       |
| - | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 50 | **`.biases` on a K-quant holds ggml's `d` (and `dmin`) — a SCALE, not an additive bias.** Any code assuming it is additive (a bias-folding pass, a dtype-following cast rule, a re-quantizer) silently corrupts K-quant tensors. Its dtype must stay exactly `float16`. This is why the converter needs a dedicated content-keyed exemption rather than the ordinary "float follows `--dtype`" rule. | `crates/mlx-core/src/utils/gguf_kquant.rs:22`, `crates/mlx-core/src/models/quant_dispatch.rs:455`, `crates/mlx-core/src/convert.rs:5455` |
| 51 | **`kquant_biases_to_preserve` is skipped entirely when the family owns its own dtype cast** (`let kquant_biases_keys = if has_custom_sanitizer { HashSet::new() } else { … }`). `owns_dtype_cast()` is true for qwen3_5, qwen3_5_moe, lfm2, lfm2_moe. Latent today (the GGUF K-quant import writes its own output and re-quantization is refused), but it is a whole-family opt-out, not a per-tensor one. | `crates/mlx-core/src/convert.rs:2546`, `:1166`                              |
| 52 | **`nn::Linear`'s quantized backend hardcodes `mode = "affine"`** at both the forward (`mlx_quantized_matmul`) and the load-time dequant. There is no mode parameter on `Linear::load_quantized`. This is the concrete reason `is_affine_only_key` exists — emitting mxfp4/mxfp8/nvfp4 at lm_head / router.proj / embed_tokens\* / embedding_projection would be silently mis-dequantized as affine, no error, just wrong numbers. | `crates/mlx-core/src/nn/linear.rs:62`, `:149`, `crates/mlx-core/src/convert.rs:3865` |
| 53 | **`is_affine_only_key` short-circuits `is_router_gate` inside both upgrade wrappers, and the two disagree.** For `.router.proj`, `apply_mxfp_upgrade` returns the inner decision unchanged (including a bare `Default`, which resolves to the global affine default), whereas `apply_nvfp4_upgrade` rewrites a `Default` into an explicit `Custom{8, 64, affine}`. Latent — no shipped recipe returns `Default` there. | `crates/mlx-core/src/convert.rs:4820` vs `:5153`                            |
| 54 | **PaddleOCR-VL's key transform uses `String::replace`, which rewrites ALL occurrences**: `result.replace("model.", "language_model.model.")` guarded only by `!result.contains("visual")`. Any key containing `model.` more than once is rewritten at every position, and `transform_key` is the identity for unmatched keys, so the damage is silent. | `crates/mlx-core/src/models/paddleocr_vl/persistence.rs:31`                  |
| 55 | **Three families each guess PyTorch-vs-MLX conv layout from shape alone.** Qianfan-OCR uses `shape[1] < shape[2]`; Qwen uses `dim2 <= 16`; paddleocr uses `t == 3 \|\| (out >= k_h && out >= k_w && k_h == k_w)`. A conv whose in_channels equals its kernel height (or a kernel > 16) is silently left in the wrong layout — both layouts have the same rank and element count, so nothing downstream can fail. | `crates/mlx-core/src/models/qianfan_ocr/persistence.rs:88`, `crates/mlx-core/src/convert.rs:1121`, `crates/mlx-core/src/models/paddleocr_vl/persistence.rs:60` |
| 56 | **The `already_sanitized` probe reads a single f32 scalar and, when true, skips the whole of Step 4** — including both conv transposes, not just the norm shift. It samples element 0 of the first non-MTP `.input_layernorm.weight` and treats `> 0.5` as "already MLX format", then sets `keys = Vec::new()`. Norm-shift state and conv-layout state are independent properties decided by one number. | `crates/mlx-core/src/convert.rs:1019-1032`, `:1084-1089`                    |
| 57 | **The GGUF path writes ONE unsharded `model.safetensors` with no `index.json`**; the SafeTensors path shards at 5 GiB and always writes an index. A 60 GB GGUF import produces one 60 GB file — a different on-disk shape from every other conversion, with no flag to change it. | `crates/mlx-core/src/utils/gguf.rs:3096`, `crates/mlx-core/src/utils/safetensors.rs:833` |
| 58 | **There is no inverse of llama.cpp's Q/K head permutation anywhere in the importer.** The only head-order fixup is `fixup_qwen35_linear_attn`, and it touches GDN tensors only. `self_attn.q_proj`/`k_proj` are copied through unreordered. mlx-lm has an export-direction `permute_weights` (`mlx-lm/mlx_lm/gguf.py:133`); nothing here undoes it. Shapes are identical, so nothing downstream can detect it. | `crates/mlx-core/src/utils/gguf.rs:1570`                                    |

### Dead / defensive code you should not read as capability

| # | Fact                                                                                                                                                                                                                  | Where                                                                       |
| - | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 59 | **The K-quant arms in the SafeTensors converter are unreachable from any production path.** `VALID_QUANT_MODES` is `[affine, mxfp4, mxfp8, nvfp4, sym8]`, no recipe constructs a `Custom{mode:"q6k"}`, and the GGUF frontend restricts its modes to affine/mxfp8/mxfp4/nvfp4. They are defensive and unit-tested — but reading them as evidence that convert can emit or round-trip K-quants is wrong. | `crates/mlx-core/src/convert.rs:5910`, `:5711`, `:1967`, `crates/mlx-sys/mlx/mlx/ops.cpp:5245` |
| 60 | **`resolve_legacy_entry`'s `lm_head` clause in arm 4 is dead by construction** — `should_quantize` already excluded it at arm 1. The code says so: "kept for defense-in-depth". | `crates/mlx-core/src/convert.rs:6297`                                       |
| 61 | **`VALID_MTP_QUANT_POLICIES` still lists `"drafter"`**, but the alias is normalized to `"split"` fourteen lines earlier, so that entry can never match. Two independent copies of the list (TS and Rust) can drift. | `crates/mlx-core/src/convert.rs:1960`, `:1977`, `packages/cli/src/commands/convert.ts:265` |
| 62 | **`derived_symmetric_bias_bits` is `pub` production code with zero production callers** — it exists purely as the parity gate's test oracle. It looks like dead code to a linter, and deleting it removes the only check that the load-time reconstruction still reproduces the historical bytes. | `crates/mlx-core/src/utils/gguf.rs:743`                                     |
| 63 | **`fn model_types()` on `ConversionRecipe` is `#[allow(dead_code)]`** — no runtime dispatch role; it exists only for the registry-consistency test. | `crates/mlx-core/src/convert.rs:161`                                        |
| 64 | **`utils/gemma_quant_repack.rs` is NOT on the GGUF path.** Its module doc says it "mirrors the GGUF Q4_0 → MLX affine repack in `super::gguf`", which makes it easy to mistake for shared code. Its only production caller is `convert_gemma_import.rs:52`, and unlike the GGUF path it still **materializes** the derived biases array. Changing one does not change the other. | `crates/mlx-core/src/utils/gemma_quant_repack.rs:31`                        |
| 65 | **There are TWO independent GGUF header parsers in the crate.** `utils/gguf.rs:34` and `utils/imatrix.rs:23` each declare `GGUF_MAGIC` and their own reader helpers; the imatrix copy has no `MAX_GGUF_ALLOC` equivalent. They *do* share the name mapper (`imatrix.rs:20` imports `gguf_name_to_hf`), so a rename rule fixed in one reaches the other — but a spec/hardening fix does not. | `crates/mlx-core/src/utils/imatrix.rs:20`, `:23`                            |

### Documentation vs code — code wins

| claim                                                                                                                                        | where                                                | reality                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--q-recipe nvidia` "is supported **only** for `qwen3_5` / `qwen3_5_moe` … Other families (e.g. `gemma4`) need their own recipe."             | `docs/cli.md:97-100`                                 | Stale. `nvidia_recipe_family` accepts `gemma4` / `gemma4_unified` / `gemma4_text` (`crates/mlx-core/src/convert.rs:4952`) and `validate_nvidia_recipe_options` passes them for **dense** gemma4 (`:5027`). Only gemma4 **MoE** is rejected. The CLI's own `--help` documents it correctly (`convert.ts:112`). |
| Auto-detected families list                                                                                                                  | `docs/cli.md:229-232`                                | No arm exists for `qianfan-ocr` / `pp-lcnet-ori` / `uvdoc` (all need explicit `-m`), and the list **omits** `lfm2`, `lfm2_moe`, `privacy-filter`, which *are* detected (`convert.ts:641`, `:644`). |
| `--model-type` options in `mlx convert --help`                                                                                               | `packages/cli/src/commands/convert.ts:37`            | Omits `gemma4` and `gemma4_unified`, both accepted and auto-detected (`crates/mlx-core/src/convert.rs:1753`).                                                                                    |
| sym8 allowlist is "dense qwen3_5, lfm2/lfm2_moe, gemma4 — **NOT** qwen3_5_moe"                                                                | `crates/mlx-core/src/convert.rs:194-201`, `convert.ts:51` | Contradicted by the impl 970 lines below: `Qwen35Recipe::sym8_supported()` returns `true` **unconditionally** for both dense and MoE, and its own inline comment says "Both dense qwen3_5 and qwen3_5_moe dispatch sym8" (`:1170`). `ConversionOptions`'s doc (`:1801`) has it right. `-m qwen3_5_moe -q --q-mode sym8` **is** accepted today — and MoE is exactly where the 3-D `switch_mlp` experts divert to forced 8-bit affine, so the result is a sym8/affine mixture. |
| `--gguf-kquant` "keeps the source … byte size" / "the same bits at ggml's byte size"                                                          | `convert.ts:138`, `crates/mlx-core/src/utils/gguf_kquant.rs:323` | True **only** for Q6_K. Q4_K grows 144→148 B/super-block (+2.8 %), Q5_K 176→180 (+2.3 %). The same source file contradicts itself 110 lines later (`gguf_kquant.rs:432`), and `docs/cli.md:184` carries the correct +0.125 bpw table. |
| `fixup_shapes` doc: "Norm weights: GGUF stores delta from 1.0 → add +1.0"                                                                     | `crates/mlx-core/src/utils/gguf.rs:1471`             | The body forty lines later says the exact opposite ("GGUF stores actual trained norm weights … We do NOT apply it here"), and no `+1.0` appears in the function. The NOTE's escape hatch — "handled by persistence.rs sanitize_weights" — cannot fire for GGUF input, because qwen3_5's detector is `conv1d.weight` shape[-1] != 1 and `fixup_shapes` has already reshaped it to `[C,K,1]`. GGUF norms are used verbatim, always. |
| AWQ inline comment `// post_attention_layernorm.weight /= scales`                                                                             | `crates/mlx-core/src/convert.rs:6952`                | Stale — the selection three lines above probes `pre_feedforward_layernorm` first (`:6932`). Any note asserting "fold post_attention_layernorm, NOT pre_feedforward" contradicts both the code and the gemma4 decoder layer. |
| `convert_gemma_import.rs` module doc: "does **NOT** wire into the `mlx convert` CLI driver"                                                   | `crates/mlx-core/src/convert_gemma_import.rs:7`      | Contradicted 36 lines later in the **same** doc comment (`:43`) and by the call site at `crates/mlx-core/src/convert.rs:2560`.                                                                   |
| `Gemma4Recipe`'s sanitize "is the real transform (set via [`set_gemma4_sanitize`])"                                                           | `crates/mlx-core/src/convert.rs:1589`                | `set_gemma4_sanitize` does not exist — the only occurrence in the file is that doc link. Broken intra-doc link; the body is inline at `:1596`.                                                    |
| "GDN `a_log` … stays f32 and is cast on-the-fly inside `compute_g`. Casting it would diverge from mlx-lm semantics."                          | `crates/mlx-core/src/models/qwen3_5_moe/paged_forward.rs:1066` | `set_a_log` unconditionally casts A_log to `dt_bias`'s dtype (`crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs:486`) and is the only such function in the tree. mlx-node genuinely diverges from mlx-lm's `cast_predicate` here (`mlx-lm/mlx_lm/models/qwen3_5.py:386` excludes `A_log`); the divergence is a deliberate perf choice, but this comment denies it. |
| Recipe list                                                                                                                                  | `docs/perf.md:457-475`                               | Omits `nvidia` entirely, and describes the no-recipe default as "router gates → 8-bit; everything else → 4-bit" without noting the 4 is really the per-mode default (affine=4, mxfp8=8, sym8=8) nor the affine-only-key force. |
| `docs/cli.md` convert flag table                                                                                                             | `docs/cli.md:139-149`                                | Omits `--config-dir`, `-m`/`--model-type`, `--q-bits`, `--q-group-size`, `--gguf-kquant`, `-h`.                                                                                                  |
| foreign formats "Paddle `.pdiparams`, PyTorch `.pkl`" / module header lists `.pdparams` only                                                  | `docs/cli.md:236`, `crates/mlx-core/src/utils/foreign_weights.rs:6` | Code supports `.pdiparams` (+ mandatory sibling `.json`), `.pdparams`, `.pt`, `.pkl`, `.pth`, and directory auto-detect. Neither doc is complete; the module header omits `.pdiparams`, which is a completely different (non-pickle) code path. |
| GGUF source-type support in `--help`                                                                                                         | `packages/cli/src/commands/convert.ts:156`           | Says only "BF16, F16, F32, Q4_0, Q4_1, Q8_0". Omits that Q6_K imports **without** `--gguf-kquant` when the tensor is gemma4's `token_embd.weight`, dequantized to BF16 at 16 bpw — a 2.44× size expansion over the 6.5625 bpw source that no doc mentions. |
| `--dtype` default                                                                                                                            | `convert.ts:32` (bfloat16), `crates/mlx-core/src/convert.rs:1782` (float32), `crates/mlx-core/src/utils/gguf.rs:2404` ("keep original") | All three are accurate **for their own layer** — three different defaults for one option name. `docs/cli.md:141` states none of them. |

Asymmetries where the code is right but the *quality rationale* does not survive scrutiny:

- **`linear_attn.out_proj` is protected asymmetrically.** `validate_nvfp4_recipe`
  (`crates/mlx-core/src/convert.rs:4930`) rejects `--q-mode nvfp4 --q-recipe mixed_*` precisely
  because `out_proj` (KLD ~6.0, "worst tensor") would be promoted to a 4-bit float format with no
  affine fallback. But `--q-mxfp --q-recipe mixed_4_6` promotes the same tensor to mxfp4 4/32 with no
  gate, and the behaviour is locked in by a passing test (`:8394`).
- **`build_official_unsloth_recipe` fails silently in two opposite directions** where `mixed_*` fails
  loudly: FFN tensors with no parseable layer index go to bf16 (`None => Skip`, `:4566`), and
  `num_layers == 0` makes `final_eight_start = 0.saturating_sub(8) = 0`, so EVERY layer is "final
  eight" and the whole model gets the high format (`:4488`). `build_recipe_predicate` errors when it
  cannot infer `num_layers` (`:3966`).
- **`mlx calibrate` supports only `qwen3_5` / `qwen3_5_moe`**, but `mlx convert --q-recipe nvidia`
  now also produces `gemma4` / `gemma4_unified` checkpoints — whose mxfp8 attention sites therefore
  keep bf16 activations permanently, with no path to modelopt W8A8 parity. Neither the CLI help nor
  `docs/cli.md` mentions the restriction
  (`crates/mlx-core/src/calibration/napi.rs:259`, `crates/mlx-core/src/convert.rs:4952`).
- **Pre-K-quant Q6_K fallback is still live and narrowly scoped.** With `--gguf-kquant` OFF, a Q6_K
  tensor is accepted only if `general.architecture == "gemma4"` **and** the tensor is exactly
  `token_embd.weight`, in which case it is dequantized to dense BF16. Every other Q6_K position, and
  all Q4_K/Q5_K, hard-error. The same source tensor therefore lands in two completely different
  on-disk formats depending on one flag — 6.5625 bpw packed vs 16 bpw dense. A size or quality
  comparison that does not pin the flag is meaningless
  (`crates/mlx-core/src/utils/gguf.rs:1085`, `:633`).
