# Causal Mask Bug Fix

**Date**: January 2025  
**Impact**: Critical - Model attended to future tokens, breaking autoregressive generation

## The Bug

MLX and PyTorch have **opposite boolean mask semantics**:

| Library | `True`            | `False`               |
| ------- | ----------------- | --------------------- |
| PyTorch | Mask out (`-inf`) | Keep                  |
| **MLX** | **Keep**          | **Mask out (`-inf`)** |

Our implementation used PyTorch semantics (`linds < rinds`), creating an **upper triangular** mask that:

- ✅ Kept future positions (WRONG in MLX)
- ❌ Masked past positions (WRONG in MLX)

## The Fix

**File**: `node/src/array/mask.rs:59`

```rust
// Before (WRONG - PyTorch semantics)
let mask = linds.less(&rinds)?;  // Upper triangular

// After (CORRECT - MLX semantics)
let mask = linds.greater_equal(&rinds)?;  // Lower triangular + diagonal
```

**Result**: Lower triangular + diagonal mask (True = keep past/self, False = mask future)

## Missing Mask in Cached Forward

**File**: `node/src/models/qwen3/model.rs:284-296`

Initial multi-token prompts need causal masking even with KV cache:

```rust
// Create causal mask if sequence length > 1
let mask = if seq_len > 1 {
    Some(create_causal_mask(seq_len, None, None)?)
} else {
    None  // Single token: no mask needed
};
```

## Verification

**Predictions now match HuggingFace transformers**:

- Test: "Once upon a time" → Token 11 (',')
  - MLX: logit 19.0968 ✅
  - Transformers: logit 19.0000 ✅
  - Diff: 0.0968 (bfloat16 vs float32)

**Tests**: 14 causal mask tests from mlx-lm, all passing

## References

- MLX semantics: `mlx/python/tests/test_fast_sdpa.py`
- Test implementation: `__test__/core/causal-mask-mlx-lm.test.ts`
- Root cause analysis: `docs/causal-mask-root-cause.md`
