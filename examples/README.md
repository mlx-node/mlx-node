# MLX-Node Examples

This directory contains example scripts demonstrating MLX-Node capabilities.

## Generation Speed Benchmark

Compare token generation speed between Node.js (mlx-node) and Python (mlx-lm).

### Prerequisites

**For Node.js:**

```bash
# Build the project (from project root)
yarn install && yarn build
```

**For Python:**

```bash
# Install mlx-lm
pip install mlx-lm
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

### Running Benchmarks

**Node.js (mlx-node):**

```bash
node examples/test-converted-model.ts
# Or with tsx
npx tsx examples/test-converted-model.ts
```

**Python (mlx-lm):**

```bash
python examples/test-mlx-lm-speed.py
```

### Interpreting Results

Both scripts:

- Use the same model (`.cache/models/qwen3-0.6b-mlx-f32`)
- Test with identical prompts
- Use identical generation parameters:
  - Node.js: `temperature=0.7, topP=0.9`
  - Python: `temp=0.7, top_p=0.9, top_k=50` (via sampler)
- Display tokens/second for performance comparison

Expected output format:

```
Generated (42 tokens, 850ms, 49.41 tokens/s):
[generated text...]
```

### Performance Notes

- First run may be slower due to model loading and Metal shader compilation
- Subsequent runs typically show consistent performance
- tokens/s metric excludes model loading time, only measures generation
- Both implementations use Apple Metal GPU acceleration
