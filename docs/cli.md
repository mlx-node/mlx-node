# CLI (`@mlx-node/cli`)

The `mlx` binary is built from `packages/cli/` and exposes the top-level commands `download`, `convert`, `calibrate`, `redact`, `serve`, `launch`, and `agent`.

## `mlx download`

### Models

```bash
mlx download model --model Qwen/Qwen3-0.6B
```

The command resolves the repo's current revision on HuggingFace and pins the
whole download to that commit. A successful download writes a
`.mlx-download-complete.json` marker into the output directory (the same
marker the dashboard reads and writes) recording the repo, the pinned
revision, the downloaded file list, and whether the selection was full or
partial. Glob-filtered runs stay marked partial so neither the CLI nor the
dashboard can mistake one selected shard for a complete model. Re-running the
command then syncs instead of skipping blindly:

- marker matches the current upstream revision and every recorded file is on
  disk → "Model already up to date", nothing is touched;
- upstream revision changed, or the marker is missing → every local file is
  verified against upstream by content hash; changed files are re-downloaded,
  unchanged files are kept, and files the upstream repo no longer has (only
  ones the old marker recorded) are removed;
- `--force` runs that hash-verify sync even when the marker looks current.

Full syncs list the remote tree recursively so nested files already recorded
by a CLI/dashboard marker are verified or removed before the revision advances.
Fresh downloads keep the historical root-only default selection, avoiding
unrelated checkpoints under directories such as `original/`.

There is no need to `rm -rf` a model directory to pick up an upstream update —
re-running the command (or `--force`) is enough. When the revision cannot be
resolved (offline, missing auth on a gated repo), the command warns and falls
back to the previous local-only completeness checks.

An output directory carrying a marker for another HuggingFace repo is refused;
choose a distinct path with `--output`. For marker-less directories created by
older CLIs, a successful full sync removes only superseded standard top-level
SafeTensors layouts (for example, an old `model.safetensors` replaced by
shards) before publishing the new marker.

Note that `--force` without `--glob` on a GGUF directory holding only some
quantization variants downloads all remaining variants, and that the marker
makes the directory dashboard-managed — a dashboard install into it may
replace the contents wholesale.

| Flag             | Default                | Purpose                                                |
| ---------------- | ---------------------- | ------------------------------------------------------ |
| `-m`, `--model`  | `Qwen/Qwen3-0.6B`      | HuggingFace model id                                   |
| `-g`, `--glob`   | —                      | Filename pattern filter (download only matching files) |
| `--force`        | `false`                | Re-verify every file against upstream by content hash  |
| `--cache-dir`    | `~/.cache/huggingface` | HuggingFace cache directory                            |
| `--set-token`    | —                      | Store HuggingFace credentials                          |
| `-o`, `--output` | —                      | Output directory                                       |

### Datasets

```bash
mlx download dataset
```

Default dataset: `openai/gsm8k`. Parquet inputs are automatically converted to JSONL via `convertParquetToJsonl()`.

| Flag               | Default        | Purpose                |
| ------------------ | -------------- | ---------------------- |
| `-d`, `--dataset`  | `openai/gsm8k` | HuggingFace dataset id |
| `-r`, `--revision` | —              | Dataset revision       |
| `-o`, `--output`   | —              | Output directory       |

## `mlx convert`

The convert command uses `--input` / `--output` (not `--model`).

### Dtype conversion

```bash
mlx convert --input ./model --output ./model-bf16 --dtype bf16
```

### Quantization (affine, default)

```bash
mlx convert --input ./model --output ./model-q --quantize --q-recipe mixed_4_6
```

### Unsloth MXFP and DGX tensor-class recipes for Qwen3.5 and SafeTensors Gemma4 MoE

For verified dense and MoE Qwen3.5/Qwen3.6-family checkpoints and the exact
SafeTensors Gemma-4-26B-A4B MoE shape, the fixed
[Unsloth class map](https://unsloth.ai/docs/models/qwen3.6#nvfp4) is
available in two forms. The Apple map translates FP8-class weights to MXFP8;
the DGX map retains NVFP4 weight storage and stores plain E4M3 FP8 weights with
one scale per output channel.
Neither map accepts `--imatrix-path`. AWQ pre-scaling divides weight columns by up to ~56x,
which drives an NVFP4 block scale below the E4M3 minimum and annihilates the block, so the
converter refuses the combination. Plain affine Unsloth still requires an imatrix.
These fixed maps need no replacement: their encoders are data-free and choose
their own block scales. See [Data-free encoder tuning](#data-free-encoder-tuning).

```bash
# Apple MXFP variant: replace NVFP4 with MXFP4
mlx convert -m qwen3_5_moe -q --q-recipe unsloth --q-mxfp \
  -i ./qwen3.5-35b-a3b -o ./qwen3.5-35b-a3b-unsloth-mxfp4-mlx

# DGX weight variant: retain NVFP4
mlx convert -m qwen3_5_moe -q --q-mode nvfp4 --q-recipe unsloth \
  -i ./qwen3.5-35b-a3b -o ./qwen3.5-35b-a3b-unsloth-nvfp4-mlx

# Gemma4 MoE Apple MXFP variant
mlx convert -m gemma4 -q --q-recipe unsloth --q-mxfp \
  -i ./gemma-4-26b-a4b-it -o ./gemma-4-26b-a4b-it-unsloth-mxfp4-mlx

# Gemma4 MoE DGX weight variant
mlx convert -m gemma4 -q --q-mode nvfp4 --q-recipe unsloth \
  -i ./gemma-4-26b-a4b-it -o ./gemma-4-26b-a4b-it-unsloth-nvfp4-mlx
```

For Qwen, early FFNs use the low class; the final eight FFNs, attention
q/k/v/o, GDN qkv/z/out, and `lm_head` use the high class. For Gemma4 MoE, every
dense and expert FFN uses the low class, attention q/k/v/o uses the high class,
and there is no final-eight or `lm_head` exception. The low class is MXFP4 4/32
with `--q-mxfp`, or NVFP4 4/16 with `--q-mode nvfp4`; the high class is MXFP8
8/32 on Apple or `fp8_e4m3` (raw E4M3 `[N,K]` weights + BF16 `[N,1]` dequant
scales) in the DGX artifact. Runtime activations remain A16 for both DGX weight classes:
NVFP4 uses standard MLX weight-only quantized matmul, while plain FP8 weights
are reconstructed to BF16 once at load. This is a data-free tensor-class and
serialized-weight-format port when no imatrix is supplied. It does not include
Unsloth's calibrated NVFP4 global scales, W4A4/W8A8 activation execution, or
calibrated FP8 KV-cache scales, and it does not claim upstream numerical or
performance parity. Embeddings, routers, GDN a/b, vision/audio, MTP,
norms, recurrent parameters, and other unmatched tensors remain BF16. Gemma
expert imatrix pre-scaling is not inferred from flat GGUF statistics. Plain affine Unsloth alone keeps
the legacy Dynamic 2.0 recipe.

### Data-free encoder tuning

All three float block formats leave quality on the table in the encoder itself,
with no calibration data involved. The converter closes all three in its own
encoders: there is no flag and no untuned mode. Tuning changes what the
checkpoint stores, never what the model computes — every rule below is exact in
weight space or strictly closer to the source weights, and none of them touches
an already-quantized body.

Which rule you get follows from the **format**, not from a flag, so it does not
matter how a tensor arrived at that format: any tensor emitted as MXFP4 or MXFP8
goes through the in-tree encoders — via `--q-mxfp`, via `--q-mode mxfp4` /
`--q-mode mxfp8` directly, or via `--q-recipe nvidia`, whose fixed map emits both
without `--q-mxfp`. Any dense SwiGLU FFN whose whole `gate`/`up`/`down` trio is
emitted as NVFP4 is preceded by the power-of-two lift — the trio is the unit,
because the three exponents are solved together. One gap: the lift is a
SafeTensors-lane pass, so `mlx convert` from a **GGUF** source emits unlifted
NVFP4 (its MXFP4/MXFP8 tensors are encoder-tuned like any other).

The resulting checkpoints stay byte-compatible with mlx-lm — same codec, same
group size, same sidecars — but their block scales are not the bytes
`mlx_quantize` would have written for the same weights.

**NVFP4 — power-of-two lift.** A block's scale is `amax / 6` stored in E4M3,
whose smallest normal value is `2^-6`. Real FFN weights put essentially every
block scale below that, so the scale carries three bits instead of four and the
smallest blocks round to the zero code and dequantize to nothing. Each dense
SwiGLU FFN is rescaled by a power of two and the inverse folded into the norm
the MLP reads, moving every block scale into the normal band.

The fold is exact, not approximate. RMSNorm computes its reciprocal from the
norm's *input*, so scaling the norm *weight* scales its output by exactly that
factor; a power of two only shifts an exponent field, so no mantissa and no
rounding decision changes anywhere downstream. silu is not homogeneous, which
pins `gate_proj`'s factor to the inverse of the norm's; the elementwise product
then passes an `up_proj`-side factor straight to `down_proj`'s input. So
`gate == up + down` is forced — three tensors, two degrees of freedom — and the
three exponents are solved together, never greedily per tensor.

A layer is lifted only when its whole `gate`/`up`/`down` trio is bound for NVFP4,
because the three exponents are solved together. A trio with a non-NVFP4 member
has no solution to apply, so it is skipped — counted and warned about, like the
cases below. This is not hypothetical: `--q-mode nvfp4 --q-recipe qwen3_5` hits
it on **every** layer, because that recipe gives `down_proj` one bit more than
the default and the NVFP4 upgrade promotes only 4-bit decisions, leaving
`gate`/`up` NVFP4 and `down` 5-bit affine. Use `--q-recipe unsloth` for a
uniformly NVFP4 FFN.

MoE expert FFNs are **skipped, not lifted** — the skip is counted, and the count
is warned about at the end of the pass, so the gap is visible in the convert and
not only here. On `qwen3_5_moe` the norm the experts read also drives the softmax
router and the sigmoid shared-expert gate, and neither is scale-invariant, so
folding the inverse there would rescale routing while weight-space error
improved. A Gemma4 MoE layer is different — its router reads the raw residual and
its experts read their own `pre_feedforward_layernorm_2`
(`crates/mlx-core/src/models/gemma4/decoder_layer.rs:442-458`) — but its expert
FFNs are skipped too for now. Each Gemma4 MoE layer also carries a dense `mlp`
whose absorber norm nothing else reads, and that dense FFN **is** lifted.

`--dtype bfloat16` (or `float32`) is what gets the full lift. Under `float16` a
layer whose folded norm would fall into the subnormal band — where the fold stops
being exact — is skipped and warned about, with the rest of the model lifted
normally.

**MXFP4 — E8M0 exponent search.** MLX rounds `log2(amax / 6)` to nearest, which
leaves the block maximum above E2M1's top code on roughly three blocks in five
and clips it. Rounding the other way never clips but spends a binade of
resolution on the other thirty-one values. The encoder tries both and keeps the
lower squared error. E8M0 has no mantissa, so those two exponents are the entire
candidate set. This needs an in-tree encoder: `mlx_quantize` takes its scales as
output parameters only, so the tuned encoder — not MLX's — writes every MXFP4 and
MXFP8 tensor the converter emits. MLX's own rounded encoder survives as a
`#[cfg(test)]` reference, where two bit-identity tests pin it byte-for-byte
against `mlx_quantize` and a third pins the production dispatch to the tuned one.

**MXFP8 — E8M0 exponent ceiling.** The same defect, the larger blast radius: on
the fixed MXFP map MXFP8 carries 233 of the 401 quantized tensors — every
attention `q/k/v/o`, every GDN `in_proj_qkv`/`in_proj_z`/`out_proj`, the final
eight FFNs and `lm_head` — against MXFP4's 168.

Rounding lands the exponent within a factor of sqrt(2) either side of
`amax / 448`, so exactly half of all blocks get one below their own maximum, and
E4M3 then saturates that maximum by as much as 1.41x. The ceiling cannot: it is
the smallest exponent whose power of two is at or above `amax / 448`. Measured
on real Qwen3.5 and Qwen3.8 attention, GDN and FFN weights, that moves the
relative error from 6.6-8.1% down to a flat 2.66% — a 2.5-3x cut, and 2.66% is
E4M3's own element-grid floor, so what is left is the format rather than the
scale.

MXFP8 does not get MXFP4's two-candidate search, because at eight bits there is
nothing for it to find. E4M3 spans `2^-9` to 448, about `2^17.8` of dynamic
range, so the binade of headroom the ceiling spends costs a block of 32 nothing;
E2M1 spans 0.5 to 6 and the same binade is a quarter of the format, which is why
MXFP4's clip-versus-resolution trade is genuinely balanced and MXFP8's is not.
Run against the per-block optimum on real weights, the ceiling ties it to four
decimals on every tensor while the search costs 55% more encode time
(19.6s versus 30.4s over Qwen3.8-27B's 10.6 G MXFP8 elements). NVIDIA modelopt,
vLLM and CUTLASS all take the ceiling for E8M0, and Blackwell's
`cvt.rp.satfinite.ue8m0x2.f32` does it in hardware; MLX itself already ceils on
its CUDA backend and rounds to nearest only on Metal and CPU.

The NVFP4 lift and the two MX rules are mutually exclusive by construction. The
lift is a provable no-op on MXFP4 and MXFP8 — E8M0 spans `2^±127` with no
subnormal band, so rescaling by `2^k` shifts every stored exponent by exactly
`k` and leaves every packed element alone.

```bash
# NVFP4 map: the power-of-two lift, applied automatically.
mlx convert -q --q-recipe unsloth --q-mode nvfp4 \
  -i ./Qwen3.8-27B -o ./qwen3.8-27b-unsloth-nvfp4-mlx

# MXFP map: the MXFP4 search over 168 tensors and the MXFP8 ceiling over 233.
mlx convert -q --q-recipe unsloth --q-mxfp \
  -i ./Qwen3.8-27B -o ./qwen3.8-27b-unsloth-mxfp4-mlx
```

### NVIDIA modelopt recipe (data-free MXFP4 port)

`--q-recipe nvidia` ports NVIDIA modelopt's `w4a16_nvfp4-fp8_attn-kv_fp8_cast`
recipe with MXFP4 in place of NVFP4, for both dense `qwen3_5` and MoE
`qwen3_5_moe`. It is a fixed per-layer format map (ignores `--q-bits` /
`--q-group-size`), runs under `--q-mode affine`, and needs no imatrix: FFN +
`lm_head` → mxfp4 4/32, attention q/k/v/o + GDN `in_proj_qkv`/`in_proj_z`/
`out_proj` → mxfp8 8/32, GDN `in_proj_a`/`in_proj_b` + router gates → 8-bit
affine, everything else bf16.

It is supported **only** for `qwen3_5` / `qwen3_5_moe` (the port targets the
Qwen3.5/3.6 hybrid modelopt recipe); passing it with any other `--model-type`,
an omitted one, or a GGUF input is rejected upfront. Other families (e.g.
`gemma4`) need their own recipe.

```bash
# dense
mlx convert -m qwen3_5 -q --q-recipe nvidia \
  -i ./qwen3.6-27b -o ./qwen3.6-27b-nvidia-mxfp4-mlx
# MoE
mlx convert -m qwen3_5_moe -q --q-recipe nvidia \
  -i ./qwen3.6-35b-a3b -o ./qwen3.6-35b-a3b-nvidia-mxfp4-mlx
```

### modelopt NVFP4 ingest (nemotron_h)

`nemotron_h` is an ingest, not a recipe run: the source is NVIDIA's modelopt
checkpoint `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, which is
**already quantized** (experts, shared experts, and `lm_head` in NVFP4; the
Mamba-2 `in_proj`/`out_proj` in FP8). Ingest preserves NVFP4 byte-for-byte — the
fp4 E2M1 codes and the per-group E4M3 `weight_scale` bytes are both carried
verbatim, and the checkpoint's `weight_scale_2` is carried **separately** as a
Float32 `.global_scale` key (an `[E]` vector for the stacked experts, since it
varies per expert) applied as a scalar on the projection output at runtime.
Folding it into the E4M3 group scales, as an earlier revision did, cost ~8% mean
relative error because the product lands in E4M3's subnormal band; **checkpoints
converted before this change must be regenerated** — the loader rejects one that
has no `.global_scale`. Ingest also re-quantizes the FP8 Mamba-2 projections to
**affine 8-bit group-32** with the checkpoint's static `input_scale` threaded as
`input_amax`. (These were mxfp8 8/32 until the quantization-accuracy pass: MLX
rounds the E8M0 block exponent to NEAREST rather than ceil, which costs 6.1%
relative RMS against a per-tensor-E4M3 source versus 0.64% for affine 8/32 —
a 9.6x error reduction on the whole sequence-mixing backbone. Checkpoints
converted before that pass are rejected at load with a regenerate hint.) No
re-quantization flags apply (`-q`/`--q-recipe` are rejected on this
already-quantized source); the convert is a format/repack pass, not a recipe.
One consequence: the output is no longer loadable by mlx-lm as plain nvfp4.

```bash
mlx convert -m nemotron_h \
  -i .cache/models/nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 \
  -o .cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx
```

### Qwen MTP quantization conversion

```bash
mlx convert \
  --input .cache/models/qwen3.6-27b \
  --output .cache/models/qwen3.6-27b-unsloth-nvfp4-mtplx-sidecar \
  --model-type qwen3_5 \
  --quantize --q-mode nvfp4 --q-recipe unsloth \
  --q-mtp cyankiwi
```

`--q-mtp cyankiwi` keeps `mtp.fc` and MTP norms BF16 and packs the MTP layer
linears as 4-bit affine group-size 32 tensors with MTPLX-compatible metadata.
Where those quantized tensors land depends on the model family:

- Dense `qwen3_5` — emitted into a separate `mtp.safetensors` sidecar.
- MoE `qwen3_5_moe` — there is **no sidecar**; the MTP tensors are quantized in
  place and stored inline in the main safetensors shards.

`--q-mtp all` additionally quantizes `mtp.fc` (same dense-sidecar / MoE-inline
split). `--q-mtp split` (alias `drafter`) emits a body checkpoint with **no
`mtp.*` tensors** plus a separate `mtp-drafter/` directory in mlx-vlm's
`qwen3_5_mtp` format (bare-keyed, BF16 MTP head); it does not require
`--quantize`/`--q-recipe` and the body may be BF16 or already-quantized.

| Flag               | Purpose                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `-i`, `--input`    | Source model directory (required)                                                             |
| `-o`, `--output`   | Output directory (required)                                                                   |
| `-d`, `--dtype`    | Target dtype: `float32` / `float16` / `bfloat16`                                              |
| `-q`, `--quantize` | Enable quantization                                                                           |
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth`, `nvidia`     |
| `--q-mode`         | `affine` (default), `mxfp4`, `mxfp8`, `nvfp4`, or `sym8`. `mxfp4`/`mxfp8` always use the in-tree encoders, never MLX's; `nvfp4` also applies the power-of-two lift to dense FFNs (SafeTensors sources only) |
| `--q-mxfp`         | Select Unsloth's fixed MXFP tensor-class map, or upgrade eligible decisions for other recipes; MX block exponents are always encoder-tuned |
| `--q-mtp`          | Qwen MTP-quant policy: `off`, `cyankiwi`, `all`, or `split` (alias `drafter`)                 |
| `--imatrix-path`   | Path to imatrix file for AWQ pre-scaling                                                      |
| `--mmproj`         | Vision-encoder conversion path                                                                |
| `-v`, `--verbose`  | Verbose logging                                                                               |

### GGUF → SafeTensors

```bash
mlx convert --input ./model.gguf --output ./model-mlx
```

Auto-detected by the `.gguf` extension. Supports BF16, F16, F32, Q4_0, Q4_1 and
Q8_0 source types directly, plus the ggml K-quants Q6_K, Q4_K and Q5_K behind
`--gguf-kquant`.

#### K-quants (Q6_K, Q4_K, Q5_K)

```bash
mlx convert --input ./model-UD-Q6_K_XL.gguf --output ./model-mlx --gguf-kquant
```

Imports llama.cpp / Unsloth-Dynamic K-quant tensors with weights **bit-identical
to llama.cpp's**, at ggml byte size, rather than requantizing into MLX's affine
format — which is lossy, and where it is exact is _larger_ than the source (Q6_K
→ affine needs `group_size=16`, costing an fp16 scale **and** bias per 16 weights
= 8.0 bpw against ggml's 6.5625).

This works because K-quants are algebraically affine per sub-block, so the kernel
is MLX's affine kernel with the scalar `(scale, bias)` load replaced by a
two-level decode:

```
Q4_K/Q5_K   y = d*sc[j]*q - dmin*m[j]   ->  scale = d*sc[j]   bias = -dmin*m[j]
Q6_K        y = d*sc[j]*(q-32)          ->  scale = d*sc[j]   bias = -32*d*sc[j]
```

| source | mlx-node   | ggml   | note                           |
| ------ | ---------- | ------ | ------------------------------ |
| Q6_K   | 6.5625 bpw | 6.5625 | exact parity                   |
| Q4_K   | 4.6250 bpw | 4.5000 | +0.125 for unpacked sub-scales |
| Q5_K   | 5.6250 bpw | 5.5000 | +0.125, same reason            |

The sub-scales are stored unpacked rather than in ggml's 6-bit packing: packing
would preserve the exact 4.5 bpw but breaks the affine pointer-walk contract and
puts a divergent branch in the innermost loop of the matvec kernel.

`--gguf-kquant` cannot be combined with `--quantize`, `--q-recipe`, `--q-mxfp` or
`--imatrix-path` — the blocks are imported bit-for-bit and never dequantized, so
there is nothing for a re-quantizer to act on. The combination is rejected
upfront rather than silently ignored.

Producing K-quants is not supported; they are consume-only. IQ4_XS is a 16-entry
non-uniform codebook rather than a scale/bias grid, does not share the kernel
shape, and is not supported.

#### Symmetric formats (Q4_0, Q8_0)

ggml stores these as `w = d * (q - Z)` — one f16 scale per 32 weights, with the
offset derived rather than stored. MLX's affine format is `w = scale * q + bias`,
so the import used to write a `.biases` array whose every entry was `-Z * scale`:
0.5 bpw of pure redundancy, 681 MB on Gemma-4-12B-QAT.

The converter now records `symmetric_zero_point` in `config.json` and leaves the
companion off disk; the loader rebuilds it before any layer is constructed. The
reconstruction is bitwise equal to what was stored — `Z` is a power of two, so
the f16 product is exact — and the output lands at ggml's own density:

| source | before     | after      | ggml                                |
| ------ | ---------- | ---------- | ----------------------------------- |
| Q4_0   | 5.0000 bpw | 4.5000 bpw | 4.5000                              |
| Q8_0   | 9.0000 bpw | 8.5000 bpw | 8.5000                              |
| Q4_1   | 5.0000 bpw | unchanged  | — (stores a real per-block minimum) |

These outputs are **not mlx-lm-loadable**, since mlx-lm requires a stored
`.biases` for affine groups — the same trade-off `--q-mode sym8` already makes.
Q4_1 imports keep their biases and stay portable. Reading a symmetric checkpoint
on an mlx-node build that predates the field fails loudly on first forward
("Biases must be provided for affine quantization"), not silently.

### Model-type auto-detection

The converter auto-detects model families and applies family-specific sanitization passes:

- `qwen3_5`, `qwen3_5_moe`
- `gemma4`, `gemma4_unified`
- `paddleocr-vl`, `qianfan-ocr`
- `pp-lcnet-ori`, `uvdoc`

Sharded models are also supported (parses `model.safetensors.index.json`).

Foreign weight formats: Paddle `.pdiparams`, PyTorch `.pkl`.

## `mlx calibrate`

Calibrate per-tensor **FP8 (E4M3) activation `amax`** for an `--q-recipe nvidia`
model so a later inference run reproduces NVIDIA modelopt's
`w4a16_nvfp4-fp8_attn-kv_fp8_cast` **activation** math (W8A8 on the mxfp8
attention/GDN projections).

`mlx convert --q-recipe nvidia` only quantizes **weights**; activations stay
bf16 until calibrated. This command runs the model over the NVIDIA calibration
mix, records each attention/GDN mxfp8 projection's running `max|activation|`
(modelopt `MaxCalibrator` semantics), and writes `input_amax` into the model's
`config.json` **in place** — under the `quantization` block (plus the legacy
`quantization_config` alias when a source config carries one). At load time each
of those projections then
fake-quantizes its input to E4M3 (`from_fp8(to_fp8(x·448/amax))·amax/448`) before
the matmul. Only the mxfp8 attn/GDN sites (`self_attn.{q,k,v,o}_proj`, GDN
`in_proj_qkv`/`in_proj_z`/`out_proj`) are calibrated; the mxfp4 FFN keeps bf16
activations (modelopt is W4A16 there), and the affine a/b, gates, lm_head, and
embeddings are untouched.

> Apple GPUs have no FP8 matmul hardware — this is **fake-quant for numeric
> parity, not speed**. Expect no throughput change, only a small activation
> quantization error matching modelopt.

```bash
mlx calibrate \
  -i ./qwen3.6-27b-nvidia-mxfp4-mlx \
  --dataset ~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl \
  --calib-size 1024 --calib-seq 512
```

| Flag            | Purpose                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `-i`, `--input` | Model directory to calibrate in place (an `--q-recipe nvidia` model, required) |
| `--dataset`     | Calibration JSONL of `{"text": "..."}` rows (required)                         |
| `--calib-size`  | Number of dataset rows to run (default `1024`, matching modelopt `hf_ptq`)     |
| `--calib-seq`   | Approximate prefill length per row in tokens (default `512`)                   |

The default calibration mix is `cnn_dailymail` + Nemotron-Post-Training-v2,
1024 samples at seq-len 512 (modelopt `hf_ptq` defaults); a 1024-row subset ships
at `~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl`. Running on a non-nvidia
(no mxfp8 attn/GDN) model calibrates 0 projections and leaves `config.json`
unchanged.

## `mlx eval`

Measures a converted checkpoint's **output quality** against a bf16 reference:
teacher-forced NLL and perplexity, top-1 agreement, and KL divergence. Every
other quality number in this repo — quantization error, AWQ deltas, recipe
comparisons — is **weight-space** error, which says how far a dequantized tensor
sits from the original and nothing about what the model emits.

Two steps. The bf16 reference runs **once**; every candidate is then scored
against what it wrote.

```bash
mlx eval cache --teacher .cache/models/qwen3.8-27b \
  --dataset eval.jsonl --cache /tmp/teacher-27b --rows 64 --seq 512

mlx eval score --model .cache/models/qwen3.8-27b-unsloth-nvfp4-mlx \
  --cache /tmp/teacher-27b
```

```text
model            .cache/models/qwen3.5-0.8b-q4
teacher          .cache/models/qwen3.5-0.8b
rows/positions   4 / 322
nll              3.6660   (teacher 3.5166, +0.1494)
perplexity       39.095   (teacher 33.669, +16.11%)
kl_topk          0.13782  (K=512, teacher tail mass 0.04045)
top1_agreement   77.64%
```

| Flag             | Purpose                                                                     |
| ---------------- | --------------------------------------------------------------------------- |
| `--teacher`      | Reference checkpoint, normally bf16 (`cache` mode, required)                |
| `--model`, `-m`  | Candidate checkpoint to score (`score` mode, required)                      |
| `--dataset`      | Eval JSONL of `{"text": "..."}` rows (`cache` mode, required)               |
| `--cache`        | Teacher cache directory (required in both modes)                            |
| `--rows`         | Dataset rows to capture (default `64`)                                      |
| `--seq`          | Tokens kept per row (default `512`, minimum `2`)                            |
| `--top-k`        | Retained support per position (default `1024`, clamped to the vocabulary)   |
| `--logit-chunk`  | Positions per head projection (default `64`)                                |
| `--json`         | Emit the report as one JSON object (`score` mode), for A/B scripting        |

**Reading the numbers.** `nll`, `perplexity` and `top1_agreement` are exact over
the full vocabulary. `kl_topk` is a KL over a `K+1`-way partition: one term per
cached top-`K` token, plus a single aggregated bucket for everything outside that
support. Both sides keep their own true full-vocabulary normaliser, and
`teacher_tail_mass` is the teacher's share of that last bucket. The bucket is not
cosmetic — the top-`K` terms **alone** are not a divergence and go negative
whenever the candidate holds more mass on the teacher's support than the teacher
does, which would rank it better than the teacher's own zero. It is still coarse
outside the support, so a KL is only comparable across checkpoints while
`teacher_tail_mass` stays small; raise `--top-k` if it does not.

**One capture per cache directory.** Give each capture its own `--cache` path.
Rows are written under fixed names, so two `mlx eval cache` runs against the
same directory at the same time interleave their rows, and the one that finishes
last publishes metadata describing a mixture. Nothing detects that afterwards. A
`score` that overlaps a capture IS detected and refused — the capture clears the
metadata before its first row and stamps a new one after its last, and `score`
re-checks it after reading the rows.

**What score refuses.** A candidate must be able to answer for the cached
rows, so `score` requires the same `model_type`, the same `tokenizer.json`
(by digest) and the same vocabulary width as the capture. Width alone is not
enough: score reads its token ids from the cache and indexes the candidate's
logits with the teacher's cached vocabulary indices, so a different tokenizer
would report a finite, plausible number measured on the wrong text. `mlx convert`
copies `tokenizer.json` verbatim, so a quantized checkpoint still matches the
bf16 teacher it came from. The cache is also re-checked after scoring: if a
capture replaced it mid-run, the score is refused rather than reported.

**What the teacher is.** `--teacher` is normally the bf16 model, and every
number is a divergence *from it*. A quantized checkpoint is accepted rather than
refused — anchoring on a released reference, or A/B-ing two recipes against a
shared one, is a real comparison — but the cache records that it was quantized
and `score` prints the fact beside the teacher path. Read a report carrying that
marker as "distance from this checkpoint", never as "distance from bf16".

**Effective versus requested.** The first token of a row primes the forward and
has no target of its own, so a row under 2 tokens scores nothing: `--seq 1` is
raised to 2. `--top-k` is clamped to the teacher's vocabulary, which degrades to
an exact full-vocab KL rather than erroring. The cache records what the rows
actually hold in both cases, so a report's `K=` is the real support width.

**What it does not tell you.** The eval runs the reference AR prefill lane, so it
says nothing about paged KV, MTP or speculative decoding. It also produces no
threshold — whether `+0.15` NLL is acceptable is a judgement about the
checkpoint, not something the tool decides.

Score mode reads its token ids **from the cache** and never re-tokenizes, so a
dataset or tokenizer edit cannot silently produce a comparison against different
text; a candidate whose vocabulary differs from the cache's is refused. Use a
**held-out** eval set — scoring an imatrix-driven or activation-calibrated
checkpoint on its own calibration data is train-on-test and flatters exactly the
recipes a comparison is meant to separate.

`qwen3_5` (dense) and `qwen3_5_moe`, dispatched on `model_type` the same way
`mlx calibrate` is. Other families are refused, not silently approximated.

## `mlx serve`

Runs the shared inference host (`@mlx-node/server/host`) in the foreground: discovers every model under the models dir, binds an Anthropic/OpenAI-compatible HTTP endpoint, and lazily loads a model on the first request that names one. At most one model stays resident; requesting another swaps it.

```bash
mlx serve                                        # auto-picked free port
mlx serve --port 8080
mlx serve --port 0                               # ephemeral port, printed on startup
mlx serve --port 8080 --model qwen3.5-9b         # pin the default model
mlx serve --host 0.0.0.0 --auth-token "$(openssl rand -hex 16)"
mlx serve --verbose                              # capture every HTTP turn to a log dir
```

| Flag                 | Meaning                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--port <n>`         | Port to bind. Omitted ⇒ a free port is picked. `0` ⇒ ephemeral, real port printed                                 |
| `--host <h>`         | Bind address (default `127.0.0.1`). Non-loopback exposes local models — pair with `--auth-token`                  |
| `--models-dir <dir>` | Discovery root (default `~/.mlx-node/models`; also `MLX_MODELS_DIR`, `~/.mlx-node/config.json`)                   |
| `--model <name>`     | Default model. Falls back to `ANTHROPIC_MODEL`, then the first discovered model                                   |
| `--auth-token <tok>` | Required on every route except `/health`, as `x-api-key` or `Authorization: Bearer`. Also `MLX_SERVER_AUTH_TOKEN` |
| `-v, --verbose`      | Write every request/response to a log dir                                                                         |
| `--log-dir <dir>`    | Override the log directory (implies `--verbose`)                                                                  |

`GET /health` is unauthenticated and reports readiness (`ok` / `loading` / `degraded` / `error`), uptime, pid, resident models, and the last load's outcome — the same body `ServerInstance.health()` returns.

Like `mlx launch claude`, it applies the launcher engine policy (`MLX_PAGED_PREFILL_CHUNK_SIZE=2048`) unless the variable is already set in the environment.

## `mlx launch claude`

Launches the same inference host as `mlx serve` and spawns Claude Code against it — the entry point for using MLX-Node as a Claude Code backend. Use `mlx serve` when you want the server without a Claude Code child (for example to point another client at it, or to reproduce a wedged sidecar in a terminal).

## `mlx agent`

A fully-local coding agent — MLX-Node's first all-in-one local agent. It embeds the [pi coding agent](https://www.npmjs.com/org/earendil-works) (`@earendil-works/*`) and serves every model turn through in-process `@mlx-node/lm` inference. There is no HTTP server, no external process, and no API keys: prompts, tools, and weights all stay on the machine. Requires Node.js ≥ 22.19.

```bash
mlx agent                       # interactive session (first run: setup wizard)
mlx agent -c                    # resume the most recent session
mlx agent -p 'summarize this repo' --no-session   # headless / print mode
mlx agent --mode json           # newline-delimited JSON events (for scripting)
mlx agent --models-dir ./models # use a specific local models directory
```

Almost every flag belongs to pi and is forwarded verbatim; `mlx agent` only handles the options below before handing off.

### Model selection and first-run wizard

`mlx agent` discovers local models under the resolved models directory (`--models-dir <dir>`, else `MLX_MODELS_DIR`, else `modelsDir` in `~/.mlx-node/config.json`, else `~/.mlx-node/models`). A dash-leading path must use the `--models-dir=<dir>` form so it is not mistaken for another flag. Dense Qwen3.5/Qwen3.8 `Q<number>_K_XL.gguf` targets are also discovered when placed directly in that directory or one level inside a downloaded GGUF repository; each appears under its filename stem. Other GGUF variants and companion files such as imatrix, mmproj, draft, or DFlash2-only checkpoints are not advertised as agent models.

To pair an XL GGUF target with DFlash2 automatically, use the mlx-node agent's embedded `draft/` convention:

```text
~/.mlx-node/models/qwen38-q4xl/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── Qwen3.8-27B-UD-Q4_K_XL.gguf
└── draft/
    ├── config.json
    └── model.safetensors
```

The root config and tokenizer files belong to the Qwen3.8 target. `draft/config.json` must declare `DFlash2DraftModel` in `architectures`; the draft weights stay in that directory exactly as published by z-lab. `draft/` may be a symlink to an existing local Hugging Face checkout. The agent advertises only `mlx/Qwen3.8-27B-UD-Q4_K_XL`, then passes `draft/` to the loader when that target becomes resident. mlx-vlm itself has no equivalent combined layout: it receives the same two paths explicitly through `--model` and `--draft-model`.

On a fresh run (no explicit `--model`/`--provider`/session flag), it injects the first discovered local model — honoring a persisted `/model` pick when that model is still present — so ambient cloud credentials (e.g. a stray `GROQ_API_KEY`) never win over the local model this command promises.

When no local model exists, an interactive terminal shows a first-run wizard over a curated catalog and downloads the choice via `mlx download model`. In a non-interactive shell it prints the equivalent `mlx download model` commands instead. The catalog:

| Model                 | HuggingFace repo                                 | Size   | Notes                        |
| --------------------- | ------------------------------------------------ | ------ | ---------------------------- |
| Qwen3.8-27B (default) | `Brooooooklyn/Qwen3.8-27B-MXFP4-mlx`                 | ~23 GB | Best tool use — recommended  |
| Qwen-AgentWorld-35B   | `Brooooooklyn/Qwen-AgentWorld-35B-A3B-mxfp4-mlx`     | ~23 GB | Agent-tuned MoE, fast decode |
| Gemma-4-26B-A4B       | `Brooooooklyn/Gemma-4-26B-A4B-Unsloth-MXFP4-mlx`     | ~16 GB | MoE, fast decode             |

The slugs above are what the wizard offers on Apple Silicon. On Linux + NVIDIA
CUDA it offers the `nvfp4` build of the same model instead — see `catalogRepo`
in `packages/agent/src/catalog.ts` for why.

A more compact Gemma-4-12B entry (mxfp4 MLP + mxfp8 attention, ~9 GB, for smaller machines) is coming and will appear in the wizard once it is published.

### Config home

pi's config home is `~/.mlx-node/agent` (override with `PI_CODING_AGENT_DIR`) — it holds `settings.json`, saved sessions, extensions, skills, prompts, and themes. A project-local `.pi/` directory still works for per-repo overrides. `mlx agent` also seeds `PI_SKIP_VERSION_CHECK=1` and `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` (both only when unset) so long prompts keep bounded time-to-first-token on the default paged path.

Gemma4 also uses paged autoregressive decoding by default, even when the checkpoint contains an embedded `draft/`: the agent's temporary config overlay hides that directory without modifying the checkpoint. Set `MLX_AGENT_ENABLE_GEMMA_DRAFT=1` to explicitly use the embedded DSpark/assistant draft instead; that opt-in currently uses flat KV cache and may be slower for quantized agent workloads.

### Permission gate

pi has no permission system of its own, so `mlx agent` installs a safety gate: every `bash`, `write`, and `edit` tool call must be approved before it runs. In an interactive session it prompts (`Yes` / `Always (this session)` / `No`). Without an attached UI — headless print or `--mode json` runs — it blocks those tools unless you opt in with `MLX_AGENT_AUTO_APPROVE=1`:

```bash
MLX_AGENT_AUTO_APPROVE=1 mlx agent -p 'run the test suite and report failures' --no-session
```

### Extensions and skills

The leading positional commands pass through to pi and manage what lives under the agent config home:

```bash
mlx agent install <source>   # add a pi extension / theme / skill
mlx agent remove <name>      # remove one (alias: uninstall)
mlx agent list               # list installed
mlx agent config             # edit which are enabled
```

`mlx agent update` is intentionally blocked (it maps to pi's npm self-update, which would fight the installed `@mlx-node/cli`); update `@mlx-node/cli` through your package manager instead. `mlx agent -h`/`--help` prints the mlx options above and then pi's full flag list. `mlx agent --version`/`-v` and `mlx agent --export <session>` are answered by pi directly — no local model needed, so the first-run wizard stays out — and `--version` prints the embedded pi version, not `@mlx-node/cli`'s (`mlx --version`).

## Dashboard

`mlx dashboard` was **removed**. The dashboard is now the Control Panel window of the mlx-node
desktop app, opened from the tray — it is served over `app://` and a MessagePort, so
there is no port to bind and no unauthenticated HTTP surface to guard. See
[docs/dashboard.md](dashboard.md).
