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
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth` |
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
- `gemma4`
- `paddleocr-vl`, `qianfan-ocr`
- `pp-lcnet-ori`, `uvdoc`

Sharded models are also supported (parses `model.safetensors.index.json`).

Foreign weight formats: Paddle `.pdiparams`, PyTorch `.pkl`.

## `mlx launch claude`

Launches the local `@mlx-node/server` and spawns Claude Code against it — the entry point for using MLX-Node as a Claude Code backend. The "serve" terminology in commit messages refers to internal server components only; there is no `mlx serve` command.
