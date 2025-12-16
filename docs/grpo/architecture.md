# Qwen3 Model Architecture Reference

This document compares our MLX-Node Qwen3 implementation with the official MLX-LM Python reference.

## Reference Implementations

- **MLX-LM Python**: `./mlx-lm/mlx_lm/models/qwen3.py`
- **Our Implementation**: `./src/grpo/models/qwen3-model.ts`

---

## Architecture Overview

### Qwen3 Key Features (from MLX-LM)

1. **QK Normalization**: Always enabled - core feature of Qwen3
2. **Grouped Query Attention (GQA)**: Different num_heads and num_kv_heads
3. **RoPE**: Rotary Position Embeddings (non-traditional mode)
4. **SwiGLU MLP**: Gated activation with 3 projections
5. **Pre-norm Architecture**: Normalization before attention and FFN

---

## Component Comparison

### 1. Attention Layer

#### MLX-LM Reference (`qwen3.py`)

```python
class Attention(nn.Module):
    def __init__(self, args):
        # Projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # QK Normalization (ALWAYS ENABLED for Qwen3)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # RoPE
        self.rope = initialize_rope(head_dim, base=args.rope_theta, traditional=False)

    def __call__(self, x, mask=None, cache=None):
        # 1. Project Q, K, V
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. Reshape and apply QK normalization
        queries = self.q_norm(queries.reshape(B, L, n_heads, head_dim))
        keys = self.k_norm(keys.reshape(B, L, n_kv_heads, head_dim))

        # 3. Transpose to (B, n_heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

        # 4. Apply RoPE AFTER QK norm
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)

        # 5. Update cache and perform attention
        keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(queries, keys, values)

        # 6. Output projection
        return self.o_proj(output)
```

#### Our Implementation Status

✅ **Implemented**: Using existing `Attention` class from mlx-node core

- Separate Q/K/V projections: ✅
- QK normalization support: ✅ (via `useQkNorm` flag)
- RoPE support: ✅
- KV caching: ✅
- GQA support: ✅

⚠️ **Note**: Our core Attention class should match this flow exactly when `useQkNorm=true`

---

### 2. MLP Layer

#### MLX-LM Reference

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        # SwiGLU: down(silu(gate) * up)
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
```

#### Our Implementation Status

✅ **Implemented**: Using existing `MLP` class from mlx-node core

- Three projections (gate, up, down): ✅
- SwiGLU activation: ✅
- No bias: ✅

---

### 3. Transformer Block

#### MLX-LM Reference

```python
class TransformerBlock(nn.Module):
    def __init__(self, args):
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        # Pre-norm attention with residual
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)

        # Pre-norm FFN with residual
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
```

#### Our Implementation Status

✅ **Implemented**: Using existing `TransformerBlock` class from mlx-node core

- Pre-norm architecture: ✅
- Two RMSNorm layers: ✅
- Residual connections: ✅
- Attention + MLP composition: ✅

---

### 4. Full Model Structure

#### MLX-LM Reference

```python
class Qwen3Model(nn.Module):
    def __init__(self, args):
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

class Model(nn.Module):
    def __init__(self, args):
        self.model = Qwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)  # Weight tying
        else:
            out = self.lm_head(out)
        return out
```

#### Our Implementation Status

✅ **Implemented**: `MLXCausalLM` class

- Embedding layer: ✅
- Transformer layers: ✅
- Final normalization: ✅
- LM head: ✅
- Weight tying support: ⚠️ (needs verification)

---

## Configuration Mapping

### HuggingFace Config → Our Config

| HF Config                 | Our Config              | Qwen3-0.6B Value |
| ------------------------- | ----------------------- | ---------------- |
| `vocab_size`              | `vocabSize`             | 151,936          |
| `hidden_size`             | `hiddenSize`            | 1,024            |
| `num_hidden_layers`       | `numLayers`             | 28               |
| `num_attention_heads`     | `numHeads`              | 16               |
| `num_key_value_heads`     | `numKvHeads`            | 8                |
| `intermediate_size`       | `intermediateSize`      | 3,072            |
| `rms_norm_eps`            | `rmsNormEps`            | 1e-6             |
| `rope_theta`              | `ropeTheta`             | 1,000,000        |
| `max_position_embeddings` | `maxPositionEmbeddings` | 40,960           |
| `tie_word_embeddings`     | `tieWordEmbeddings`     | true             |

### Qwen3-Specific Features (Not in HF config)

- `useQkNorm`: **Always true** for Qwen3 (architectural feature)
- `qkNormEps`: Same as `rms_norm_eps` (1e-6)

---

## Generation Pipeline

### MLX-LM Generation Flow

From `mlx-lm/mlx_lm/generate.py` and `sample_utils.py`:

```python
def generate_step(model, prompt, temp=1.0, top_p=1.0):
    # 1. Get logits from model
    logits = model(prompt, cache=cache)[:, -1, :]

    # 2. Apply temperature
    logits = logits / temp

    # 3. Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # 4. Apply top-p (nucleus) sampling
    if top_p < 1.0:
        probs = top_p_sampling(probs, top_p)

    # 5. Sample from distribution
    token = categorical_sampling(probs)

    return token
```

#### Our Implementation Status

- Basic generation: ✅ (greedy decoding works)
- Temperature scaling: ✅ (implemented in `node/src/sampling.rs`)
- Top-p (nucleus) sampling: ✅ (implemented with `top_p` parameter)
- Top-k sampling: ✅ (implemented with `top_k` parameter)
- Min-p sampling: ✅ (implemented with `min_p` parameter)
- Categorical sampling: ✅ (implemented with `categorical()` method)
- XTC sampling: ✅ (eXclude Top Choices diversity sampling)
- Repetition penalty: ✅ (asymmetric penalty algorithm)

---

## Action Items

### Completed ✅

1. ✅ **QK Normalization**: Implemented and enabled for Qwen3
2. ✅ **Attention Implementation**: Core `Attention` properly applies QK norm before RoPE
3. ✅ **Categorical Sampling**: Implemented with `categorical()` method
4. ✅ **Top-P Filtering**: Nucleus sampling fully implemented
5. ✅ **Top-K Filtering**: Top-k sampling fully implemented
6. ✅ **Min-P Filtering**: Min-p sampling fully implemented
7. ✅ **XTC Sampling**: eXclude Top Choices diversity sampling implemented
8. ✅ **Repetition Penalty**: Asymmetric penalty algorithm implemented
9. ✅ **Tokenizer**: HuggingFace tokenizers integrated (151K vocab)
10. ✅ **Weight Loading**: SafeTensors and MLX format loaders working
11. ✅ **GRPO Training**: Full production-ready implementation
12. ✅ **BatchKVCache**: Variable-length batch support with left-padding
13. ✅ **Entropy Filtering**: Selective training on high-uncertainty tokens

### Future Work

14. ⏳ **Qwen3-MoE Support**: Implement MoE variant (~700 lines estimated)
15. ⏳ **Autograd Backward Pass**: Phase 6 - automatic differentiation (not critical for GRPO)
16. ⏳ **Performance Optimization**: Profile and optimize generation speed further

---

## Testing Checklist

- [x] Model configuration matches MLX-LM
- [x] Model instantiation works
- [x] Forward pass works without cache
- [x] Forward pass works with KV cache
- [x] QK normalization is enabled
- [x] Sampling methods work correctly (temperature, top-k, top-p, min-p, XTC, repetition penalty)
- [x] Weight loading from SafeTensors format works
- [x] Weight loading from MLX format works
- [x] GRPO training infrastructure complete (1,178 tests passing)
- [x] BatchKVCache supports variable-length batches
- [x] Entropy filtering for selective training
- [x] HuggingFace tokenizer integration (151K vocab)
- [ ] Generation output quality validation against MLX-LM reference
- [ ] Performance benchmarking vs MLX-LM baseline

---

## References

- **MLX-LM Qwen3**: `./mlx-lm/mlx_lm/models/qwen3.py`
- **MLX-LM Generation**: `./mlx-lm/mlx_lm/generate.py`
- **MLX-LM Sampling**: `./mlx-lm/mlx_lm/sample_utils.py`
- **HuggingFace Qwen3**: https://huggingface.co/Qwen/Qwen3-0.6B
