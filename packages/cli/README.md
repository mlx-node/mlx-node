# @mlx-node/cli

Command-line tool for downloading models and datasets from HuggingFace Hub and converting model weights for use with `@mlx-node/*` packages.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## Installation

```bash
npm install -g @mlx-node/cli
```

Or run via your package manager:

```bash
npx @mlx-node/cli download model --model Qwen/Qwen3-0.6B
```

## Commands

### Download Model

Download model weights and tokenizer files from HuggingFace Hub:

```bash
mlx download model --model Qwen/Qwen3-0.6B
```

Downloads to `.cache/models/<model-slug>` by default. Skips if already downloaded.

#### Options

| Flag          | Short | Default                | Description                               |
| ------------- | ----- | ---------------------- | ----------------------------------------- |
| `--model`     | `-m`  | `Qwen/Qwen3-0.6B`      | HuggingFace model name                    |
| `--output`    | `-o`  | `.cache/models/<slug>` | Output directory                          |
| `--glob`      | `-g`  | (all supported files)  | Filter files by glob pattern (repeatable) |
| `--set-token` |       |                        | Set up HuggingFace authentication         |

#### Authentication

For gated or private models, set up your HuggingFace token:

```bash
mlx download model --set-token
```

This validates the token against the HuggingFace API and stores it securely in your OS keychain. The token is automatically used for subsequent downloads.

#### File Filtering

By default, downloads config files, tokenizer files, and weight files (`.safetensors`, `.json`, `.pdiparams`, `.yml`). Use `--glob` to filter specific files:

```bash
# Download only bf16 safetensors shards
mlx download model --model Qwen/Qwen3-7B --glob "*.bf16*.safetensors"

# Download specific files
mlx download model --model org/model --glob "*.safetensors" --glob "*.json"
```

Core config and tokenizer files are always included regardless of glob filters.

### Download Dataset

Download datasets from HuggingFace Hub with automatic Parquet-to-JSONL conversion:

```bash
mlx download dataset
```

#### Options

| Flag         | Short | Default        | Env Override       | Description              |
| ------------ | ----- | -------------- | ------------------ | ------------------------ |
| `--dataset`  | `-d`  | `openai/gsm8k` | `GSM8K_DATASET`    | HuggingFace dataset name |
| `--revision` | `-r`  | `main`         | `GSM8K_REVISION`   | Dataset revision/branch  |
| `--output`   | `-o`  | `data/gsm8k`   | `GSM8K_OUTPUT_DIR` | Output directory         |

Produces `train.jsonl` and `test.jsonl` in the output directory. Automatically converts Parquet files to JSONL if the dataset doesn't include JSONL directly.

### Convert Weights

Convert model weights between formats with optional quantization:

```bash
# Dtype conversion (SafeTensors)
mlx convert --input ./model --output ./model-bf16 --dtype bfloat16

# Quantization
mlx convert --input ./model --output ./model-q4 --quantize --q-bits 4

# Mixed-precision quantization recipe
mlx convert --input ./model --output ./model-mixed --quantize --q-recipe mixed_4_6

# MXFP8 quantization
mlx convert --input ./model --output ./model-fp8 --quantize --q-mode mxfp8

# GGUF to SafeTensors
mlx convert --input ./model.gguf --output ./model-safetensors

# GGUF with vision encoder
mlx convert --input ./model.gguf --output ./model-vlm --mmproj ./mmproj.gguf

# imatrix AWQ pre-scaling with unsloth dynamic quantization
mlx convert --input ./model --output ./model-unsloth --quantize --q-recipe unsloth --imatrix-path ./imatrix.gguf

# Qwen3.5 family: official Unsloth map translated to MXFP4/MXFP8
mlx convert --input ./model --output ./model-unsloth-mxfp4 --quantize \
  --q-recipe unsloth --q-mxfp --imatrix-path ./imatrix.gguf

# Qwen3.6 with MTPLX-style MTP sidecar
mlx convert \
  --input .cache/models/qwen3.6-27b \
  --output .cache/models/qwen3.6-27b-unsloth-nvfp4-mtplx-sidecar \
  --model-type qwen3_5 \
  --quantize --q-mode nvfp4 --q-recipe unsloth \
  --imatrix-path ./imatrix.gguf \
  --q-mtp cyankiwi
```

#### Options

| Flag             | Short | Default       | Description                                                            |
| ---------------- | ----- | ------------- | ---------------------------------------------------------------------- |
| `--input`        | `-i`  | _required_    | Input model directory or `.gguf` file                                  |
| `--output`       | `-o`  | _required_    | Output directory                                                       |
| `--dtype`        | `-d`  | `bfloat16`    | Target dtype: `float32`, `float16`, `bfloat16`                         |
| `--model-type`   | `-m`  | auto-detected | Model type override                                                    |
| `--verbose`      | `-v`  | `false`       | Verbose logging                                                        |
| `--quantize`     | `-q`  | `false`       | Enable quantization                                                    |
| `--q-bits`       |       | `4`           | Quantization bits (4 or 8)                                             |
| `--q-group-size` |       | `64`          | Quantization group size                                                |
| `--q-mode`       |       | `affine`      | Mode: `affine` or `mxfp8`                                              |
| `--q-recipe`     |       |               | Per-layer mixed-bit recipe                                             |
| `--q-mtp`        |       | `off`         | Qwen MTP-quant policy: `cyankiwi`, `all`, or `split` (alias `drafter`) |
| `--imatrix-path` |       |               | imatrix GGUF for AWQ pre-scaling                                       |
| `--mmproj`       |       |               | mmproj GGUF for vision encoder weights                                 |

#### Model Types

Auto-detected from `config.json` when not specified:

| Type           | Description                                             |
| -------------- | ------------------------------------------------------- |
| (default)      | Standard SafeTensors dtype conversion                   |
| `qwen3_5`      | Qwen3.5 Dense with FP8 dequant and key remapping        |
| `qwen3_5_moe`  | Qwen3.5 MoE with expert stacking                        |
| `paddleocr-vl` | PaddleOCR-VL weight sanitization                        |
| `pp-lcnet-ori` | PP-LCNet orientation classifier (Paddle to SafeTensors) |
| `uvdoc`        | UVDoc dewarping model (Paddle/PyTorch to SafeTensors)   |

#### Quantization Recipes

| Recipe      | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `mixed_2_6` | 2-bit base, 6-bit sensitive layers                                     |
| `mixed_3_4` | 3-bit base, 4-bit sensitive layers                                     |
| `mixed_3_6` | 3-bit base, 6-bit sensitive layers                                     |
| `mixed_4_6` | 4-bit base, 6-bit sensitive layers                                     |
| `qwen3_5`   | Optimized for Qwen3.5 hybrid architecture                              |
| `unsloth`   | Legacy affine, official MXFP/DGX maps with `--q-mxfp`/`--q-mode nvfp4` |

#### Unsloth Recipe

MLX affine equivalent of [Unsloth Dynamic 2.0](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) (UD) GGUF quantization. Designed for Qwen3.5's hybrid GatedDeltaNet + full attention architecture. Requires imatrix for AWQ pre-scaling of attention/SSM weights.

```bash
# UD-Q3_K_XL equivalent (~17 GB for 35B-A3B)
mlx convert -i ./model -o ./model-q3 -q --q-bits 3 --q-recipe unsloth --imatrix-path ./imatrix.gguf

# UD-Q4_K_XL equivalent (~20 GB for 35B-A3B)
mlx convert -i ./model -o ./model-q4 -q --q-bits 4 --q-recipe unsloth --imatrix-path ./imatrix.gguf
```

For verified Qwen3.5/Qwen3.6-family checkpoints, select the fixed [official
Unsloth class map](https://unsloth.ai/docs/models/qwen3.6#nvfp4) with either
`--q-mxfp` (NVFP4 → MXFP4, FP8 → MXFP8) or `--q-mode nvfp4` (the official DGX
map, retaining NVFP4 and using MXFP8 for FP8 classes). AWQ imatrix pre-scaling
is the same for both. Plain affine Unsloth alone keeps legacy Dynamic 2.0.

```bash
mlx convert -i ./model -o ./model-unsloth-mxfp4 -q \
  --q-recipe unsloth --q-mxfp --imatrix-path ./imatrix.gguf

mlx convert -i ./model -o ./model-unsloth-nvfp4 -q \
  --q-recipe unsloth --q-mode nvfp4 --imatrix-path ./imatrix.gguf
```

The two official maps share the high-precision and BF16 classes:

| Weight class                                                                     | `--q-mxfp` | `--q-mode nvfp4` |
| -------------------------------------------------------------------------------- | ---------- | ---------------- |
| FFN `gate_proj` / `up_proj` / `down_proj`, except the final 8 transformer layers | MXFP4 4/32 | NVFP4 4/16       |
| Final 8 FFNs; attention q/k/v/o; GDN qkv/z/out; `lm_head`                        | MXFP8 8/32 | MXFP8 8/32       |
| Embeddings; routers; GDN a/b; vision; MTP; norms; recurrent parameters           | BF16       | BF16             |

Per-tensor bit assignments (N = `--q-bits`):

| Weight                      | Bits     | Rationale                                                  |
| --------------------------- | -------- | ---------------------------------------------------------- |
| `gate_proj`, `up_proj`      | N        | Bulk of MoE expert params, safe at low bits                |
| `down_proj`                 | N+1      | Slightly more sensitive than other FFN weights             |
| `embed_tokens`              | N+2      | Very low KLD sensitivity (~0.15)                           |
| `self_attn.q/k/v_proj`      | N+2      | AWQ-correctable via input_layernorm                        |
| `linear_attn.in_proj_qkv/z` | N+2      | AWQ-correctable via input_layernorm                        |
| `lm_head`                   | N+3      | Safest tensor (KLD ~0.05)                                  |
| Router gates                | 8        | Standard for MoE routing accuracy                          |
| `self_attn.o_proj`          | 8 affine | No preceding norm (not AWQ) — kept 8-bit for MTP/AR parity |
| `linear_attn.out_proj`      | 8 affine | Worst tensor (KLD ~6.0) — kept 8-bit for MTP/AR parity     |
| `linear_attn.in_proj_a/b`   | 8 affine | Split GDN low-rank projs — kept 8-bit for MTP/AR parity    |
| GDN params (`A_log`, etc.)  | bf16     | Recurrent state params, errors compound over time          |

## Examples

### Full Workflow

```bash
# 1. Set up authentication (one-time)
mlx download model --set-token

# 2. Download a model
mlx download model --model Qwen/Qwen3-0.6B

# 3. Download training data
mlx download dataset --dataset openai/gsm8k

# 4. Quantize the model
mlx convert \
  --input .cache/models/Qwen-Qwen3-0.6B \
  --output .cache/models/Qwen3-0.6B-q4 \
  --quantize --q-bits 4

# 5. Use in your application
# import { loadModel } from '@mlx-node/lm';
# const model = await loadModel('.cache/models/Qwen3-0.6B-q4');
```

### GGUF Conversion

```bash
# Download a GGUF model
mlx download model --model user/model-gguf --glob "*.gguf"

# Convert to SafeTensors
mlx convert --input .cache/models/model-gguf/model.gguf --output ./model-converted
```

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
