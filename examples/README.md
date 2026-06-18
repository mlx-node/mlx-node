# MLX-Node Examples

This directory contains example scripts demonstrating MLX-Node capabilities.

## Generation Speed Benchmark (`lm.ts`)

`lm.ts` runs token generation through mlx-node and reports tokens/second.

### Prerequisites

```bash
# Build the project (from project root)
yarn install && yarn build
```

### Model Setup

Convert a Qwen model to MLX float32 format:

```bash
# Using mlx-lm
python -m mlx_lm.convert \
    --hf-path Qwen/Qwen2.5-0.5B-Instruct \
    --mlx-path .cache/models/qwen3-0.6b-mlx-f32 \
    --dtype float32
```

### Running

```bash
node examples/lm.ts
# Or with oxnode
npx oxnode examples/lm.ts
```

### Interpreting Results

`lm.ts`:

- Uses the model at `.cache/models/qwen3-0.6b-mlx-f32`
- Generates with `temperature=0.7, topP=0.9`
- Displays tokens/second for performance

Expected output format:

```
Generated (42 tokens, 850ms, 49.41 tokens/s):
[generated text...]
```

### Performance Notes

- First run may be slower due to model loading and Metal shader compilation
- Subsequent runs typically show consistent performance
- tokens/s metric excludes model loading time, only measures generation
- Uses Apple Metal GPU acceleration
