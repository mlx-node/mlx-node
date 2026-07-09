# CLI (`@mlx-node/cli`)

The `mlx` binary is built from `packages/cli/` and exposes three top-level commands: `download`, `convert`, and `launch`.

## `mlx download`

### Models

```bash
mlx download model --model Qwen/Qwen3-0.6B
```

| Flag             | Default           | Purpose                                                |
| ---------------- | ----------------- | ------------------------------------------------------ |
| `-m`, `--model`  | `Qwen/Qwen3-0.6B` | HuggingFace model id                                   |
| `-g`, `--glob`   | —                 | Filename pattern filter (download only matching files) |
| `--set-token`    | —                 | Store HuggingFace credentials                          |
| `-o`, `--output` | —                 | Output directory                                       |

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

### Qwen MTP quantization conversion

```bash
mlx convert \
  --input .cache/models/qwen3.6-27b \
  --output .cache/models/qwen3.6-27b-unsloth-nvfp4-mtplx-sidecar \
  --model-type qwen3_5 \
  --quantize --q-mode nvfp4 --q-recipe unsloth \
  --imatrix-path ./imatrix.gguf \
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

| Flag               | Purpose                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| `-i`, `--input`    | Source model directory (required)                                               |
| `-o`, `--output`   | Output directory (required)                                                     |
| `-d`, `--dtype`    | Target dtype: `float32` / `float16` / `bfloat16`                                |
| `-q`, `--quantize` | Enable quantization                                                             |
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth`, `nvidia` |
| `--q-mode`         | `affine` (default) or `mxfp8`                                                   |
| `--q-mtp`          | Qwen MTP-quant policy: `off`, `cyankiwi`, `all`, or `split` (alias `drafter`)   |
| `--imatrix-path`   | Path to imatrix file for AWQ pre-scaling                                        |
| `--mmproj`         | Vision-encoder conversion path                                                  |
| `-v`, `--verbose`  | Verbose logging                                                                 |

### GGUF → SafeTensors

```bash
mlx convert --input ./model.gguf --output ./model-mlx
```

Auto-detected by the `.gguf` extension. Supports BF16, F16, F32, Q4_0, Q4_1, Q8_0 source quantization types.

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
`config.json` **in place** — under both the `quantization` and
`quantization_config` blocks. At load time each of those projections then
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

| Flag           | Purpose                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| `-i`, `--input` | Model directory to calibrate in place (an `--q-recipe nvidia` model, required) |
| `--dataset`    | Calibration JSONL of `{"text": "..."}` rows (required)                         |
| `--calib-size` | Number of dataset rows to run (default `1024`, matching modelopt `hf_ptq`)     |
| `--calib-seq`  | Approximate prefill length per row in tokens (default `512`)                   |

The default calibration mix is `cnn_dailymail` + Nemotron-Post-Training-v2,
1024 samples at seq-len 512 (modelopt `hf_ptq` defaults); a 1024-row subset ships
at `~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl`. Running on a non-nvidia
(no mxfp8 attn/GDN) model calibrates 0 projections and leaves `config.json`
unchanged.

## `mlx launch claude`

Launches the local `@mlx-node/server` and spawns Claude Code against it — the entry point for using MLX-Node as a Claude Code backend. The "serve" terminology in commit messages refers to internal server components only; there is no `mlx serve` command.
