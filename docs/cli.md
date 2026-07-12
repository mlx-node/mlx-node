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

### Official Unsloth MXFP and DGX recipes for Qwen3.5

For verified dense and MoE Qwen3.5/Qwen3.6-family checkpoints, the fixed
[official Unsloth class map](https://unsloth.ai/docs/models/qwen3.6#nvfp4) is
available in two forms. Both keep FP8-class weights as MXFP8 and use the same
AWQ imatrix pre-scaling:

```bash
# Apple MXFP variant: replace NVFP4 with MXFP4
mlx convert -m qwen3_5_moe -q --q-recipe unsloth --q-mxfp \
  --imatrix-path ./imatrix_unsloth.gguf_file \
  -i ./qwen3.5-35b-a3b -o ./qwen3.5-35b-a3b-unsloth-mxfp4-mlx

# Official DGX variant: retain NVFP4
mlx convert -m qwen3_5_moe -q --q-mode nvfp4 --q-recipe unsloth \
  --imatrix-path ./imatrix_unsloth.gguf_file \
  -i ./qwen3.5-35b-a3b -o ./qwen3.5-35b-a3b-unsloth-nvfp4-mlx
```

Early FFNs use MXFP4 4/32 with `--q-mxfp`, or NVFP4 4/16 with
`--q-mode nvfp4`. The final eight FFNs, attention q/k/v/o, GDN qkv/z/out, and
`lm_head` use MXFP8 8/32 in both. Embeddings, routers, GDN a/b, vision, MTP,
norms, and recurrent parameters remain BF16. Plain affine Unsloth alone keeps
the legacy Dynamic 2.0 recipe.

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

| Flag               | Purpose                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `-i`, `--input`    | Source model directory (required)                                                         |
| `-o`, `--output`   | Output directory (required)                                                               |
| `-d`, `--dtype`    | Target dtype: `float32` / `float16` / `bfloat16`                                          |
| `-q`, `--quantize` | Enable quantization                                                                       |
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth`, `nvidia` |
| `--q-mode`         | `affine` (default), `mxfp4`, `mxfp8`, `nvfp4`, or `sym8`                                  |
| `--q-mxfp`         | Select Unsloth's fixed MXFP map, or upgrade eligible decisions for other recipes          |
| `--q-mtp`          | Qwen MTP-quant policy: `off`, `cyankiwi`, `all`, or `split` (alias `drafter`)             |
| `--imatrix-path`   | Path to imatrix file for AWQ pre-scaling                                                  |
| `--mmproj`         | Vision-encoder conversion path                                                            |
| `-v`, `--verbose`  | Verbose logging                                                                           |

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

## `mlx launch claude`

Launches the local `@mlx-node/server` and spawns Claude Code against it — the entry point for using MLX-Node as a Claude Code backend. The "serve" terminology in commit messages refers to internal server components only; there is no `mlx serve` command.

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

`mlx agent` discovers local models under the resolved models directory (`--models-dir <dir>`, else `MLX_MODELS_DIR`, else `modelsDir` in `~/.mlx-node/config.json`, else `~/.mlx-node/models`). A dash-leading path must use the `--models-dir=<dir>` form so it is not mistaken for another flag.

On a fresh run (no explicit `--model`/`--provider`/session flag), it injects the first discovered local model — honoring a persisted `/model` pick when that model is still present — so ambient cloud credentials (e.g. a stray `GROQ_API_KEY`) never win over the local model this command promises.

When no local model exists, an interactive terminal shows a first-run wizard over a curated catalog and downloads the choice via `mlx download model`. In a non-interactive shell it prints the equivalent `mlx download model` commands instead. The catalog:

| Model                 | HuggingFace repo                                 | Size   | Notes                        |
| --------------------- | ------------------------------------------------ | ------ | ---------------------------- |
| Qwen3.6-27B (default) | `Brooooooklyn/Qwen3.6-27B-NVFP4-mlx`             | ~22 GB | Best tool use — recommended  |
| Qwen-AgentWorld-35B   | `Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx` | ~23 GB | Agent-tuned MoE, fast decode |
| Gemma-4-26B-A4B       | `Brooooooklyn/Gemma-4-26B-A4B-NVFP4-mlx`         | ~19 GB | MoE, fast decode             |

A more compact Gemma-4-12B entry (mxfp4 MLP + mxfp8 attention, ~9 GB, for smaller machines) is coming and will appear in the wizard once it is published.

### Config home

pi's config home is `~/.mlx-node/agent` (override with `PI_CODING_AGENT_DIR`) — it holds `settings.json`, saved sessions, extensions, skills, prompts, and themes. A project-local `.pi/` directory still works for per-repo overrides. `mlx agent` also seeds `PI_SKIP_VERSION_CHECK=1` and `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` (both only when unset) so long prompts keep bounded time-to-first-token on the default paged path.

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
