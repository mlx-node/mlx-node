# Causal Mask Bug: Root Cause Analysis

## Why This Happened

**Fundamental Issue**: Assumed PyTorch mask semantics would work in MLX (they're opposite)

**Contributing Factor**: Only tested additive masks (`-inf`/`0.0`), never boolean masks

## The Confusion

MLX supports two mask types with different semantics:

### Additive Masks (What We Tested)

```python
mask[i,j] = 0.0    # Keep (add 0)
mask[i,j] = -inf   # Mask (add -inf)
```

### Boolean Masks (What We Got Wrong)

```python
# MLX: True = keep, False = mask
# PyTorch: True = mask, False = keep  ← OPPOSITE!
```

## Why Tests Passed

Our tests used **additive masks only**:

```typescript
// Additive mask - same semantics across libraries
mask_data[i * N + j] = j > i ? -Infinity : 0.0;
```

Additive masks bypass boolean semantics entirely, so bugs weren't caught.

## How MLX-LM Would Have Caught This

MLX-LM has explicit boolean mask tests:

```python
def test_causal_mask():
    mask = create_causal_mask(4)
    # Verifies: mask[i,j] = True iff i >= j (lower triangular)
```

**Key difference**: Tests the **shape and values** of boolean masks, not just end-to-end behavior.

## Prevention

1. **Test boolean masks explicitly** - Don't just test attention outputs
2. **Read reference tests** - mlx-lm and MLX have definitive examples
3. **Never assume API semantics** - Always verify with source code
4. **Document surprising behavior** - Boolean masks are counterintuitive

## Lessons

| What Went Wrong            | Prevention                 |
| -------------------------- | -------------------------- |
| Wrong boolean semantics    | Test mask values directly  |
| Only tested additive masks | Test both mask types       |
| Assumed PyTorch semantics  | Read MLX source/tests      |
| No mask shape validation   | Add unit tests like mlx-lm |

**Bottom line**: Testing the interface (mask values) catches bugs that testing outcomes (attention results) might miss.

## References

- MLX boolean semantics: `mlx/python/tests/test_fast_sdpa.py`
- mlx-lm mask tests: `mlx-lm/tests/test_models.py`
- Our implementation: `node/src/array/mask.rs`
- Our tests: `__test__/core/causal-mask-mlx-lm.test.ts`
