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
Both use the same AWQ imatrix pre-scaling when calibration is provided. The imatrix is optional
for these fixed maps: without it, AWQ pre-scaling is skipped and quality may be
lower, while the class map remains unchanged. Plain affine Unsloth still
requires an imatrix. Matching calibration remains preferred when available;
add `--imatrix-path ./imatrix_unsloth.gguf_file` to either command below.

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

| Flag               | Purpose                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `-i`, `--input`    | Source model directory (required)                                                             |
| `-o`, `--output`   | Output directory (required)                                                                   |
| `-d`, `--dtype`    | Target dtype: `float32` / `float16` / `bfloat16`                                              |
| `-q`, `--quantize` | Enable quantization                                                                           |
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth`, `nvidia`     |
| `--q-mode`         | `affine` (default), `mxfp4`, `mxfp8`, `nvfp4`, or `sym8`                                      |
| `--q-mxfp`         | Select Unsloth's fixed MXFP tensor-class map, or upgrade eligible decisions for other recipes |
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
