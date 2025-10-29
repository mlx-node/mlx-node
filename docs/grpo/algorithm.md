# GRPO Algorithm Reference

This document captures the core GRPO (Group Relative Policy Optimization) algorithm as implemented in HuggingFace TRL, serving as a reference for our MLX-Node implementation.

## Overview

GRPO is a reinforcement learning algorithm for training language models, introduced in the DeepSeek-Math paper. It's essentially PPO with group-based advantage normalization instead of using a value function.

**Key Differences from PPO**:
- **No value function**: Advantages computed relative to group mean
- **Group-based sampling**: Generate G completions per prompt, normalize within groups
- **Simpler**: No separate critic network to train

## Core Algorithm (Pseudocode)

```
for each training step:
    1. Sample batch of prompts: P = [p1, p2, ..., pB/G]
    2. Generate G completions per prompt: C = [[c1_1, ..., c1_G], ..., [cB/G_1, ..., cB/G_G]]
    3. Compute rewards: R = reward_func(P, C)  # Shape: (B,)
    4. Compute advantages:
        mean_R_group = mean(R, dim=group)  # Shape: (B/G,)
        A = R - repeat(mean_R_group, G)    # Shape: (B,)
        A = A / (std(R, dim=group) + eps)  # Optional scaling

    5. Compute old log-probs: log π_old(C | P)  # From generation

    6. For μ iterations (default: 1):
        7. Compute new log-probs: log π_θ(C | P)
        8. Compute importance ratio: r = exp(log π_θ - log π_old)
        9. Compute clipped loss:
            L1 = r * A
            L2 = clip(r, 1-ε, 1+ε) * A
            L = -min(L1, L2)  # Per-token loss
        10. Optional: Add KL penalty: L = L + β * KL(π_θ || π_ref)
        11. Aggregate loss (see Loss Variants below)
        12. Update θ with gradient descent
```

## 1. Sampling and Generation

**Configuration** (from `GRPOConfig`):
- `num_generations` (G): 8 completions per prompt
- `temperature`: 1.0 (default, standard sampling)
- `top_p`: 1.0 (nucleus sampling threshold)
- `top_k`: None (top-k filtering disabled by default)
- `min_p`: None (minimum probability filtering)
- `max_completion_length`: 256 tokens

**Implementation** (from `mlx-lm/mlx_lm/sample_utils.py`):

### Top-K Sampling
```python
def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Keep only top-k tokens by probability."""
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf")), axis=-1
    )
    return masked_logprobs
```

### Top-P (Nucleus) Sampling
```python
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Keep tokens with cumulative probability < top_p."""
    probs = mx.exp(logprobs)
    # Sort in ascending order
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Cumulative sum
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1]),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # Mask tokens above threshold
    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )
```

### Min-P Sampling
```python
def apply_min_p(logprobs: mx.array, min_p: float, min_tokens_to_keep: int = 1) -> mx.array:
    """Keep tokens with prob > min_p * max_prob."""
    # Sort descending
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    # Get top probability
    top_logprobs = sorted_logprobs[:, 0:1]

    # Calculate threshold: log(min_p) + log(p_max)
    scaled_min_p = top_logprobs + math.log(min_p)

    # Mask tokens below threshold (keep at least min_tokens_to_keep)
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Rearrange back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1]),
        axis=-1,
    )
    return mx.take_along_axis(selected_logprobs, inverse_indices, axis=-1)
```

### Categorical Sampling
```python
def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    """Sample from categorical distribution."""
    return mx.random.categorical(logits * (1 / temp))
```

### Complete Sampling Pipeline
```python
def sample_token(logits: mx.array, temp: float = 1.0, top_p: float = 1.0,
                 top_k: int = None, min_p: float = None) -> mx.array:
    """Full sampling pipeline."""
    # Start with log-probabilities
    logprobs = logits  # Assume already log-softmax

    # Apply filters in order
    if top_k is not None and top_k > 0:
        logprobs = apply_top_k(logprobs, top_k)

    if min_p is not None and min_p > 0:
        logprobs = apply_min_p(logprobs, min_p)

    if top_p < 1.0:
        logprobs = apply_top_p(logprobs, top_p)

    # Sample with temperature
    return categorical_sampling(logprobs, temp)
```

## 2. Advantage Computation

**Reference**: `trl/trl/trainer/grpo_trainer.py` lines 1567-1588

```python
def compute_advantages(rewards: torch.Tensor, num_generations: int,
                       scale_rewards: str = "group") -> torch.Tensor:
    """
    Compute group-relative advantages.

    Args:
        rewards: Reward tensor, shape (B,) where B = num_prompts * num_generations
        num_generations: Number of completions per prompt (G)
        scale_rewards: "group" (default), "batch", or "none"

    Returns:
        advantages: Shape (B,), advantages for each completion
    """
    # Reshape to (num_prompts, num_generations)
    grouped_rewards = rewards.view(-1, num_generations)

    # Compute mean reward per group
    mean_grouped_rewards = grouped_rewards.mean(dim=1)  # Shape: (num_prompts,)

    # Repeat to match original shape
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)  # (B,)

    # Compute advantages: A = R - mean(R_group)
    advantages = rewards - mean_grouped_rewards

    # Optional: scale to unit variance
    if scale_rewards == "group":
        # Normalize by group standard deviation
        std_rewards = grouped_rewards.std(dim=1)  # Shape: (num_prompts,)
        std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)  # (B,)
        advantages = advantages / (std_rewards + 1e-4)
    elif scale_rewards == "batch":
        # Normalize by global standard deviation
        std_rewards = rewards.std()
        advantages = advantages / (std_rewards + 1e-4)
    # else: scale_rewards == "none", no scaling

    return advantages
```

**Key Points**:
1. Advantages are **zero-mean within each group**
2. Group-based normalization prevents cross-prompt comparison
3. Standard deviation scaling optional but recommended for stability

## 3. GRPO Loss Computation

**Reference**: `trl/trl/trainer/grpo_trainer.py` lines 1730-1858

### Core Loss Formula

```python
def compute_grpo_loss(
    per_token_logps: torch.Tensor,      # Current policy log-probs, shape (B, T)
    old_per_token_logps: torch.Tensor,  # Old policy log-probs, shape (B, T)
    advantages: torch.Tensor,            # Advantages, shape (B,)
    completion_mask: torch.Tensor,       # Mask for valid tokens, shape (B, T)
    epsilon: float = 0.2,                # Clipping parameter
    epsilon_high: float = None,          # Upper clipping (default: same as epsilon)
    beta: float = 0.0,                   # KL penalty coefficient
    ref_per_token_logps: torch.Tensor = None,  # Reference model log-probs
    loss_type: str = "dapo",             # Loss aggregation method
    importance_sampling_level: str = "token",  # "token" or "sequence"
    max_completion_length: int = 256,
    num_items_in_batch: int = None,
) -> torch.Tensor:
    """
    Compute GRPO loss with clipped surrogate objective.

    Args:
        per_token_logps: Log-probs from current policy π_θ
        old_per_token_logps: Log-probs from policy at generation time π_old
        advantages: Advantage values per sequence
        completion_mask: Binary mask for valid completion tokens
        epsilon: Lower clipping bound (1-ε)
        epsilon_high: Upper clipping bound (1+ε), defaults to epsilon
        beta: KL divergence penalty coefficient (0 = no penalty)
        ref_per_token_logps: Log-probs from reference model π_ref (if beta > 0)
        loss_type: "grpo", "dapo", "dr_grpo", or "bnpo"
        importance_sampling_level: "token" or "sequence"
        max_completion_length: Maximum completion length (for dr_grpo)
        num_items_in_batch: Total items across all processes (for dapo)

    Returns:
        loss: Scalar loss value
    """
    epsilon_high = epsilon_high if epsilon_high is not None else epsilon
    B, T = per_token_logps.shape

    # 1. Compute importance sampling weights
    log_ratio = per_token_logps - old_per_token_logps  # Shape: (B, T)

    if importance_sampling_level == "token":
        # Token-level: one weight per token
        log_importance_weights = log_ratio  # Shape: (B, T)
    elif importance_sampling_level == "sequence":
        # Sequence-level: single weight per sequence
        log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)  # Shape: (B, 1)

    # 2. Compute clipped surrogate objective
    # Probability ratio: r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
    coef_1 = torch.exp(log_importance_weights)  # r_t
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon_high)  # clip(r_t, 1-ε, 1+ε)

    # Expand advantages: (B,) → (B, 1) or (B, T)
    adv_expanded = advantages.unsqueeze(1)

    # L1 = r_t * A_t  (unclipped)
    # L2 = clip(r_t, 1-ε, 1+ε) * A_t  (clipped)
    per_token_loss1 = coef_1 * adv_expanded
    per_token_loss2 = coef_2 * adv_expanded

    # Take minimum (PPO clipping): maximize min(L1, L2) = minimize -min(L1, L2)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # Shape: (B, T) or (B, 1)

    # 3. Optional: Add KL divergence penalty
    if beta > 0.0:
        # KL(π_ref || π_θ) = E[exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        per_token_loss = per_token_loss + beta * per_token_kl

    # 4. Aggregate loss based on loss_type
    if loss_type == "grpo":
        # Original GRPO: normalize per sequence
        # Problem: biased towards shorter sequences with positive A, longer with negative A
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

    elif loss_type == "bnpo":
        # Batch normalization: normalize by local batch active tokens
        # Note: results vary with local batch size
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    elif loss_type == "dr_grpo":
        # Dr. GRPO: normalize by global constant (max_completion_length)
        # Eliminates length bias
        loss = (per_token_loss * completion_mask).sum() / (B * max_completion_length)

    elif loss_type == "dapo":
        # DAPO (recommended): normalize by global batch active tokens
        # Eliminates length bias, stable across batch sizes
        loss = (per_token_loss * completion_mask).sum() / num_items_in_batch

    return loss
```

### Loss Type Comparison

| Loss Type | Normalization | Length Bias | Notes |
|-----------|---------------|-------------|-------|
| **grpo** | Per-sequence: `sum(loss) / seq_len` | ⚠️ Yes | Prefers short seqs with +A, long with -A |
| **bnpo** | Local batch: `sum(loss) / sum(mask)` | ✅ No | Varies with local batch size |
| **dr_grpo** | Global constant: `sum(loss) / (B * max_len)` | ✅ No | Fixed normalization |
| **dapo** | Global batch: `sum(loss) / total_tokens` | ✅ No | **Recommended** (default) |

### Why DAPO is Recommended

1. **No length bias**: Normalizes by actual number of active tokens
2. **Batch-size invariant**: Same result regardless of batch size
3. **Stable training**: Better convergence properties

## 4. Implementation Requirements for MLX-Node

### Phase 1: Sampling (Urgent)
- [x] **Temperature scaling**: Already works
- [ ] **Top-K sampling**: Implement using argpartition + put_along_axis
- [ ] **Top-P sampling**: Implement sort + cumsum + rearrange
- [ ] **Min-P sampling**: Scale by max token probability
- [ ] **Categorical sampling**: Replace argmax with mx.random.categorical

### Phase 2: GRPO Core Logic
- [ ] **Advantage computation**: Group-based normalization
- [ ] **Log-prob computation**: Selective log-softmax per token
- [ ] **GRPO loss**: Clipped surrogate objective
- [ ] **Loss variants**: Support grpo, dapo, dr_grpo, bnpo

### Phase 3: Training Infrastructure
- [ ] **Generation loop**: Batch generation with KV caching
- [ ] **Reward computation**: Support function-based rewards
- [ ] **Optimizer integration**: Adam/AdamW with gradient clipping
- [ ] **Gradient accumulation**: Multi-step accumulation

## 5. Key Configuration Parameters

**From TRL Default Config**:
```typescript
{
  // Model and training
  learningRate: 1e-6,
  numEpochs: 3,
  batchSize: 4,
  gradientAccumulationSteps: 4,

  // GRPO specific
  numGenerations: 8,           // G completions per prompt
  clipEpsilon: 0.2,             // ε for clipping
  klCoef: 0.0,                  // β (no KL penalty by default)
  scaleRewards: "group",        // "group", "batch", or "none"
  lossType: "dapo",             // "dapo", "grpo", "dr_grpo", "bnpo"
  importanceSamplingLevel: "token",  // "token" or "sequence"

  // Generation
  maxNewTokens: 256,
  temperature: 1.0,
  topP: 1.0,
  topK: null,
  minP: null,

  // Data
  maxPromptLength: 512,
  shuffleDataset: true,

  // Other
  gradientCheckpointing: true,
  bf16: true,
}
```

## 6. Mathematical Notation

- **π_θ**: Current policy (model being trained)
- **π_old**: Policy at generation time (frozen)
- **π_ref**: Reference policy (optional, for KL penalty)
- **P**: Prompts
- **C**: Completions
- **R**: Rewards
- **A**: Advantages
- **G**: Number of generations per prompt
- **ε**: Clipping parameter (epsilon)
- **β**: KL penalty coefficient (beta)
- **r_t**: Importance sampling ratio at token t

## 7. References

- **Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- **TRL GRPO Trainer**: `./trl/trl/trainer/grpo_trainer.py`
- **TRL GRPO Config**: `./trl/trl/trainer/grpo_config.py`
- **MLX-LM Sampling**: `./mlx-lm/mlx_lm/sample_utils.py`
- **DAPO Paper**: [DAPO: Direct Alignment via Preferences Optimization](https://huggingface.co/papers/2503.14476)
- **Dr. GRPO Paper**: [Dr. GRPO](https://huggingface.co/papers/2503.20783)

---

*This document was created by analyzing the TRL and MLX-LM reference implementations to serve as a blueprint for our MLX-Node GRPO implementation.*
