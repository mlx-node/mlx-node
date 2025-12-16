# Implementation Review: MLX-Node vs TRL & MLX-LM

> **⚠️ HISTORICAL DOCUMENT**: This review was written during early development (January 2025).
> **Current Status**: ALL features described as missing have now been implemented.
> See [FEATURE_ALIGNMENT_SESSION.md](../FEATURE_ALIGNMENT_SESSION.md) for latest status.

**Original Review Date**: January 2025
**Original Status**: Architecture ✅ Solid | Generation ❌ Critical Gaps
**Current Status (Updated)**: ✅ ALL FEATURES IMPLEMENTED - Production Ready

---

## Executive Summary (Historical - January 2025)

~~Our Qwen3 implementation has **correct architecture** matching MLX-LM perfectly, but is **missing critical generation features** required for GRPO training:~~

**UPDATE**: All gaps have been addressed:

- ✅ **Architecture**: Matches MLX-LM reference exactly (QK norm, GQA, RoPE, SwiGLU)
- ✅ **Logprobs Tracking**: IMPLEMENTED (production-ready)
- ✅ **Sampling Methods**: IMPLEMENTED (categorical, top-k, top-p, min-p, XTC, repetition penalty)
- ✅ **Batch Generation**: IMPLEMENTED (BatchKVCache with left-padding support)
- ✅ **Generation Metadata**: IMPLEMENTED (EOS detection, logprobs, token tracking)
- ✅ **GRPO Training**: IMPLEMENTED (full production infrastructure, 1,178 tests passing)

~~**Immediate Action Required**: Implement logprobs tracking and categorical sampling before proceeding with GRPO training loop.~~

**Current Status**: Production-ready for GRPO training at scale.

---

---

> **📝 NOTE FOR HISTORICAL SECTIONS BELOW**:
> The comparisons and gap analyses below reflect the state of the project during early development.
> All features marked as "missing" or "TODO" have since been implemented.
> This document is preserved for historical reference and to document the development journey.

---

## 1. Architecture Comparison

### ✅ What We Implemented Correctly

| Feature                 | MLX-LM                     | Our Implementation               | Status   |
| ----------------------- | -------------------------- | -------------------------------- | -------- |
| QK Normalization        | ✅ Always enabled          | ✅ `useQkNorm: true`             | ✅ MATCH |
| Grouped Query Attention | ✅ Configurable ratios     | ✅ `numHeads: 16, numKvHeads: 8` | ✅ MATCH |
| RoPE Theta              | ✅ 1,000,000               | ✅ `ropeTheta: 1000000.0`        | ✅ MATCH |
| SwiGLU MLP              | ✅ gate/up/down            | ✅ TransformerBlock with MLP     | ✅ MATCH |
| Pre-norm Architecture   | ✅ RMSNorm before attn/mlp | ✅ `norm(x)` pattern             | ✅ MATCH |
| Tied Embeddings         | ✅ Configurable            | ✅ `tieWordEmbeddings`           | ✅ MATCH |
| KV Caching              | ✅ Incremental generation  | ✅ KVCache per layer             | ✅ MATCH |

**Verdict**: Architecture is **production-ready** and matches MLX-LM reference.

### ⚠️ Minor Differences (Non-blocking)

| Feature                  | MLX-LM                       | Our Implementation       | Impact                      |
| ------------------------ | ---------------------------- | ------------------------ | --------------------------- |
| Attention Mask Creation  | ✅ `create_attention_mask()` | ❌ Pass `mask?: MxArray` | Low - can create externally |
| Input Embeddings Support | ✅ `input_embeddings` param  | ❌ Token IDs only        | Low - not needed for GRPO   |
| Model Type Detection     | ✅ Detect Qwen3 vs Qwen3-MoE | ❌ Manual config         | Low - we focus on Qwen3     |

---

## 2. Generation Methods Comparison

### Our Current Implementation

```typescript
// src/grpo/models/qwen3-model.ts

✅ forward(inputIds, mask?, useCache): MxArray
   - Returns logits: (batch, seq_len, vocab_size)
   - Supports KV caching
   - No logprobs returned

⚠️ generate(inputIds, maxNewTokens, temperature): MxArray
   - Uses argmax (greedy decoding only)
   - Returns token IDs only
   - No logprobs tracking
   - Single completion only

❌ generateSample(inputIds, maxNewTokens, temperature, topP, topK): MxArray
   - Placeholder implementation
   - top_k and top_p are TODO comments
   - Uses argmax instead of sampling
   - No actual categorical sampling
```

### MLX-LM Reference Features

```python
# mlx-lm/mlx_lm/generate.py

✅ generate(model, tokenizer, prompt, ...):
   - Categorical sampling with mx.random.categorical()
   - Top-k filtering with argpartition
   - Top-p (nucleus) sampling with sort + cumsum
   - Min-p filtering
   - Temperature scaling
   - Repetition penalty
   - Logit bias
   - Returns tokens AND logprobs

✅ stream_generate(model, tokenizer, prompt, ...):
   - Yields GenerationResponse(token, text, logprobs)
   - Streaming output token-by-token

✅ BatchGenerator:
   - Efficient batch generation
   - Handles variable-length sequences
   - Padding and attention masks
```

### TRL GRPO Requirements

From `trl/trl/trainer/grpo_trainer.py` (lines 1366-1402):

```python
def _generate(self, prompts: list):
    # CRITICAL: Must return logprobs for policy gradient!
    prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

    # Track completion metadata
    completion_lengths = [len(ids) for ids in completion_ids]
    is_truncated = [ids[-1] not in eos_and_pad for ids in completion_ids]

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,  # ⚠️ REQUIRED for GRPO loss!
        "is_truncated": is_truncated,
    }
```

**What GRPO Needs**:

1. ❌ **Logprobs per token**: `log P(token_t | context)` for policy gradient
2. ❌ **Multiple completions**: `num_generations=8` per prompt
3. ⚠️ **Sampling methods**: top-k, top-p for diversity
4. ⚠️ **Completion metadata**: EOS vs truncated tracking

---

## 3. Critical Missing Features

### 🔴 CRITICAL (Blocking GRPO)

#### 3.1 Logprobs Tracking

**Status**: ❌ **NOT IMPLEMENTED**

**Required**: Track `log P(token | context)` for each generated token.

**Reference** (TRL, lines 820-891):

```python
def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep):
    logits = model(input_ids, attention_mask).logits[:, :-1, :]  # Exclude last
    logits = logits[:, -logits_to_keep:, :]  # Keep completion tokens only

    # Compute log P(completion_token | context)
    completion_ids = input_ids[:, -logits_to_keep:]
    logps = selective_log_softmax(logits, completion_ids)  # Shape: (B, T)

    return logps
```

**What We Need**:

```typescript
interface GenerationResult {
  tokens: Int32Array;           // Generated token IDs
  logprobs: Float32Array;       // log P(token_i | context)
  finished: boolean;             // Hit EOS or truncated?
  length: number;                // Completion length
}

// In qwen3-model.ts:
generateWithLogprobs(
  inputIds: MxArray,
  maxNewTokens: number,
  temperature: number
): GenerationResult
```

**Implementation Steps**:

1. After each token generation, extract logits for that position
2. Apply softmax to get probabilities
3. Take log to get log-probabilities
4. Store log P(generated_token | context)
5. Return accumulated logprobs array

#### 3.2 Categorical Sampling

**Status**: ❌ **NOT IMPLEMENTED** (using argmax only)

**Required**: Sample from probability distribution, not just take argmax.

**Reference** (MLX-LM, sample_utils.py line 275-276):

```python
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))
```

**What We Need**:

```rust
// In node/src/array.rs:
#[napi]
impl MxArray {
    /// Sample from categorical distribution
    pub fn categorical(&self, temperature: f32) -> Result<MxArray> {
        let scaled = self.div_scalar(temperature)?;
        Ok(MxArray::from_mlx(mx::random::categorical(&scaled.inner, None)?))
    }
}
```

**Importance**: Without this, we can't:

- Explore diverse completions (stuck with greedy)
- Implement proper GRPO (needs diverse samples for advantage computation)
- Use temperature/top-p/top-k effectively

#### 3.3 Batch Generation

**Status**: ❌ **NOT IMPLEMENTED** (single completion only)

**Required**: Generate `num_generations` completions per prompt.

**Reference** (TRL, lines 1404-1430):

```python
def _generate_and_score_completions(self, prompts: list):
    # Generate G completions per prompt
    all_prompts = [p for p in prompts for _ in range(self.num_generations)]

    # Generate all at once (B*G completions)
    prompt_ids, completion_ids, logprobs = self._generate(all_prompts)

    # Reshape to (B, G) groups
    prompt_ids = reshape_to_groups(prompt_ids, self.num_generations)
    completion_ids = reshape_to_groups(completion_ids, self.num_generations)

    return {
        "prompt_ids": prompt_ids,      # Shape: (B, G, P)
        "completion_ids": completion_ids,  # Shape: (B, G, C)
        "logprobs": logprobs,          # Shape: (B, G, C)
    }
```

**What We Need**:

```typescript
generateBatch(
  prompts: MxArray[],           // B prompts
  numCompletions: number,       // G completions per prompt
  maxNewTokens: number,
  temperature: number
): GenerationResult[]           // B * G results
```

**Why It's Critical**:

- GRPO requires multiple completions per prompt for advantage computation
- Advantages are computed relative to group mean: `A = R - mean(R_group)`
- Can't compute group statistics with only 1 completion

---

### 🟡 HIGH Priority (Needed Soon)

#### 3.4 Top-K Sampling

**Status**: ⚠️ **PLACEHOLDER** (TODO comment)

**Reference** (MLX-LM, sample_utils.py lines 111-133):

```python
def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    vocab_size = logprobs.shape[-1]
    # Use argpartition to find top-k indices
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    # Mask out non-top-k tokens
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf")), axis=-1
    )
    return masked_logprobs
```

**What We Need**:

```rust
// In node/src/sampling.rs:
#[napi]
pub fn apply_top_k(logprobs: &MxArray, top_k: i32) -> Result<MxArray>
```

#### 3.5 Top-P (Nucleus) Sampling

**Status**: ⚠️ **PLACEHOLDER** (TODO comment)

**Reference** (MLX-LM, sample_utils.py lines 201-234):

```python
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
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

    # Mask tokens with cumulative prob > 1 - top_p
    return mx.where(cumulative_probs > 1 - top_p, logprobs, -float("inf"))
```

**What We Need**:

```rust
#[napi]
pub fn apply_top_p(logprobs: &MxArray, top_p: f32) -> Result<MxArray>
```

#### 3.6 Generation Metadata

**Status**: ⚠️ **PARTIAL** (detect EOS but don't return metadata)

**What TRL Tracks** (lines 1390-1400):

```python
# Track completion statistics
is_truncated = [ids[-1] not in eos_and_pad for ids in completion_ids]
completion_lengths = [len(ids) for ids in completion_ids]

# Log metrics
clipped_ratio = is_truncated.float().mean()  # % truncated
mean_terminated_length = lengths[~is_truncated].mean()
```

**What We Need**:

```typescript
interface GenerationResult {
  tokens: Int32Array;
  logprobs: Float32Array;
  finishedReason: 'eos' | 'max_length';  // Why generation stopped
  length: number;
}
```

---

### 🟢 MEDIUM Priority (Nice to Have)

#### 3.7 Min-P Sampling

**Reference** (MLX-LM, sample_utils.py lines 136-198):

```python
def apply_min_p(logprobs, min_p, min_tokens_to_keep=1):
    # Keep tokens with prob > min_p * max_prob
    sorted_logprobs = mx.argsort(-logprobs, axis=-1)
    top_logprobs = sorted_logprobs[:, 0:1]
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False
    # ... mask and rearrange
```

#### 3.8 Repetition Penalty

**Reference** (MLX-LM, sample_utils.py lines 279-309):

```python
def make_repetition_penalty(penalty: float, context_size: int = 20):
    def repetition_penalty_processor(tokens, logits):
        tokens = tokens[-context_size:]
        selected_logits = logits[:, tokens]
        selected_logits = mx.where(
            selected_logits < 0,
            selected_logits * penalty,
            selected_logits / penalty,
        )
        logits[:, tokens] = selected_logits
        return logits
    return repetition_penalty_processor
```

#### 3.9 Streaming Generation

**Status**: ❌ Not implemented

**Use Case**: Interactive applications, monitoring generation

```typescript
async* streamGenerate(
  inputIds: MxArray,
  maxNewTokens: number
): AsyncGenerator<GenerationStep> {
  for (let i = 0; i < maxNewTokens; i++) {
    const token = await this.generateNextToken(...);
    yield {
      token,
      text: this.tokenizer.decode([token]),
      logprob: ...,
    };
  }
}
```

---

## 4. Test Coverage Comparison

### Our Current Tests (14 tests)

```typescript
// src/grpo/__test__/qwen3-model.test.ts

describe('Qwen3 Model', () => {
  ✅ Model Configuration (3 tests)
     - Default configs exist
     - Config values correct
     - Model instantiation

  ✅ Model Instantiation (3 tests)
     - From string config
     - From custom config
     - Error on unknown config

  ✅ Forward Pass (2 tests)
     - Without cache
     - With KV cache

  ✅ Generation (2 tests)
     - Greedy decoding
     - Sampling (placeholder)

  ✅ Model Components (2 tests)
     - Access layers/embedding/lmhead
     - Error on invalid index

  ✅ Qwen3 Specific Features (3 tests)
     - QK normalization enabled
     - GQA configuration
     - High RoPE theta
});

describe('GRPO Integration', () => {
  ✅ Loss Computation (1 test)
     - Compute cross-entropy loss
});
```

**Coverage**: Architecture validation ✅ | Generation features ❌

### MLX-LM Tests (~30 tests)

```python
# tests/test_models.py

def test_qwen3():
    # Basic forward pass

def model_test_runner(model, config):
    ✅ Float32 dtype
    ✅ Float16 dtype
    ✅ Batch size > 1
    ✅ Cache functionality
    ✅ Model pickling
    ✅ Model quantization

# tests/test_generate.py

✅ test_generate() - Basic generation
✅ test_generate_with_logit_bias()
✅ test_stream_generate_max_tokens()
✅ test_stream_generate_eos()
✅ test_stream_generate_repetition_penalty()
✅ test_generate_with_processor()
✅ test_batch_matches_single()
✅ test_many_batches()
✅ test_batch_unique_max_toks()
✅ test_batch_sliding_window()
✅ test_batch_eos()
```

### TRL Tests (~50+ tests)

```python
# tests/test_grpo_trainer.py

✅ test_grpo_trainer() - Basic training
✅ test_grpo_trainer_with_multiple_reward_functions()
✅ test_grpo_trainer_with_loss_type_grpo()
✅ test_grpo_trainer_with_loss_type_dapo()
✅ test_grpo_trainer_with_loss_type_dr_grpo()
✅ test_grpo_trainer_with_loss_type_bnpo()
✅ test_grpo_trainer_with_entropy_filtering()
✅ test_grpo_trainer_with_importance_sampling()
✅ test_grpo_trainer_with_vllm()
✅ test_grpo_trainer_with_vision_language_model()
✅ test_grpo_trainer_with_peft()
✅ test_grpo_trainer_with_guided_decoding()
✅ test_grpo_trainer_mask_truncated_completions()
✅ test_grpo_trainer_scale_rewards_group()
✅ test_grpo_trainer_scale_rewards_batch()
✅ test_grpo_trainer_scale_rewards_none()
```

---

## 5. Missing Test Cases

### 🔴 CRITICAL Tests to Add

#### Test 1: Logprobs Correctness

```typescript
it('should compute correct logprobs for generated tokens', () => {
  const model = new MLXCausalLM('qwen3-0.6b');
  const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));

  const result = model.generateWithLogprobs(inputIds, 5);

  // Manually verify logprobs
  const logits = model.forward(inputIds);
  const probs = Activations.softmax(logits, -1);
  const expectedLogprobs = probs.log();

  // Check that returned logprobs match manual calculation
  expect(result.logprobs).toBeClose(expectedLogprobs);
});
```

#### Test 2: Categorical Sampling Distribution

```typescript
it('should sample from correct distribution', () => {
  const logits = MxArray.fromFloat32(
    float32(0.1, 0.7, 0.2),  // Probs after softmax
    shape(1, 3)
  );

  // Sample 1000 times
  const samples = [];
  for (let i = 0; i < 1000; i++) {
    samples.push(logits.categorical(1.0).toInt32()[0]);
  }

  // Check distribution matches expected probabilities
  const counts = [0, 0, 0];
  samples.forEach(s => counts[s]++);

  expect(counts[0]).toBeCloseTo(100, 20);  // ~10%
  expect(counts[1]).toBeCloseTo(700, 50);  // ~70%
  expect(counts[2]).toBeCloseTo(200, 30);  // ~20%
});
```

#### Test 3: Batch Generation

```typescript
it('should generate multiple completions per prompt', () => {
  const model = new MLXCausalLM('qwen3-0.6b');
  const prompts = [
    MxArray.fromInt32(int32(1, 2, 3), shape(1, 3)),
    MxArray.fromInt32(int32(4, 5, 6), shape(1, 3)),
  ];

  const results = model.generateBatch(
    prompts,
    numCompletions: 4,  // 4 completions per prompt
    maxNewTokens: 10
  );

  expect(results.length).toBe(8);  // 2 prompts × 4 completions

  // Check that results for same prompt are different (sampling diversity)
  const group1 = results.slice(0, 4);
  expect(group1[0].tokens).not.toEqual(group1[1].tokens);
});
```

### 🟡 HIGH Priority Tests

#### Test 4: Top-K Filtering

```typescript
it('should keep only top k tokens', () => {
  const logprobs = MxArray.randomNormal(shape(1, 100), 0, 1);
  const topK = 10;

  const filtered = Sampling.applyTopK(logprobs, topK);

  // Count non -inf values
  const valid = filtered.toFloat32().filter(x => x > -Infinity);
  expect(valid.length).toBe(topK);

  // Verify top k tokens are the highest probability ones
  const sortedOriginal = logprobs.toFloat32().sort((a, b) => b - a);
  const topKOriginal = sortedOriginal.slice(0, topK);
  expect(valid.sort()).toEqual(topKOriginal.sort());
});
```

#### Test 5: Top-P (Nucleus) Filtering

```typescript
it('should keep tokens with cumulative prob < top_p', () => {
  const logprobs = MxArray.fromFloat32(
    float32(-1, -2, -3, -4, -5),  // Log probs
    shape(1, 5)
  );
  const topP = 0.7;

  const filtered = Sampling.applyTopP(logprobs, topP);

  // Convert to probs and check cumulative sum
  const probs = filtered.exp();
  const cumsum = probs.cumsum(0);

  // All kept tokens should have cumsum ≤ top_p
  expect(cumsum.toFloat32().every(x => x <= topP + 0.01)).toBe(true);
});
```

#### Test 6: EOS vs Truncated Detection

```typescript
it('should detect EOS vs max_length termination', () => {
  const model = new MLXCausalLM('qwen3-0.6b');
  const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));

  // Generate with max_length that forces truncation
  const result1 = model.generateWithLogprobs(inputIds, 5);
  expect(result1.finishedReason).toBe('max_length');

  // Generate with high max_length, should hit EOS
  const result2 = model.generateWithLogprobs(inputIds, 1000);
  expect(result2.finishedReason).toBe('eos');
});
```

### 🟢 MEDIUM Priority Tests

#### Test 7: Temperature Scaling

```typescript
it('should increase randomness with higher temperature', () => {
  const model = new MLXCausalLM('qwen3-0.6b');
  const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));

  // Generate 10 completions with temp=0.1 (low)
  const lowTempResults = [];
  for (let i = 0; i < 10; i++) {
    lowTempResults.push(model.generateSample(inputIds, 10, 0.1));
  }

  // Generate 10 completions with temp=2.0 (high)
  const highTempResults = [];
  for (let i = 0; i < 10; i++) {
    highTempResults.push(model.generateSample(inputIds, 10, 2.0));
  }

  // High temp should have more unique completions
  const uniqueLow = new Set(lowTempResults.map(r => r.tokens.join(',')));
  const uniqueHigh = new Set(highTempResults.map(r => r.tokens.join(',')));

  expect(uniqueHigh.size).toBeGreaterThan(uniqueLow.size);
});
```

#### Test 8: Repetition Penalty

```typescript
it('should reduce repetitive tokens', () => {
  const model = new MLXCausalLM('qwen3-0.6b');
  const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));

  // Generate without penalty
  const noPenalty = model.generate(inputIds, 50, 1.0);

  // Generate with repetition penalty
  const withPenalty = model.generateWithRepetitionPenalty(
    inputIds, 50, 1.0, penalty=1.5
  );

  // Count repeated tokens
  const countRepeats = (tokens: Int32Array) => {
    const counts = {};
    tokens.forEach(t => counts[t] = (counts[t] || 0) + 1);
    return Object.values(counts).filter(c => c > 1).length;
  };

  expect(countRepeats(withPenalty)).toBeLessThan(countRepeats(noPenalty));
});
```

---

## 6. Implementation Priority

### Phase 1: Generation Essentials (Week 1-2) 🔴 CRITICAL

**Goal**: Enable basic GRPO training with logprobs and sampling

1. **Add categorical sampling** (Rust)

   ```rust
   // node/src/array.rs
   pub fn categorical(&self, temperature: f32) -> Result<MxArray>
   ```

2. **Implement logprobs tracking** (TypeScript)

   ```typescript
   // src/grpo/models/qwen3-model.ts
   generateWithLogprobs(...): GenerationResult
   ```

3. **Add GenerationResult interface** (TypeScript)

   ```typescript
   interface GenerationResult {
     tokens: Int32Array;
     logprobs: Float32Array;
     finishedReason: 'eos' | 'max_length';
     length: number;
   }
   ```

4. **Tests**:
   - ✅ Logprobs correctness
   - ✅ Categorical sampling distribution
   - ✅ EOS vs truncated detection

### Phase 2: Batch Generation (Week 2-3) 🟡 HIGH

**Goal**: Generate multiple completions per prompt for GRPO

1. **Implement batch generation**

   ```typescript
   generateBatch(prompts: MxArray[], numCompletions: number, ...): GenerationResult[]
   ```

2. **Tests**:
   - ✅ Batch generation produces B×G results
   - ✅ Different completions for same prompt

### Phase 3: Advanced Sampling (Week 3-4) 🟡 HIGH

**Goal**: Complete sampling pipeline with all filters

1. **Implement sampling filters** (Rust)
   - Top-k: `apply_top_k()`
   - Top-p: `apply_top_p()`
   - Min-p: `apply_min_p()`

2. **Create sampling pipeline** (TypeScript)

   ```typescript
   class Sampler {
     sample(logits: MxArray, config: SamplingConfig): MxArray
   }
   ```

3. **Tests**:
   - ✅ Top-k filtering
   - ✅ Top-p (nucleus) filtering
   - ✅ Min-p filtering
   - ✅ Temperature scaling
   - ✅ Complete pipeline

### Phase 4: Logits Processors (Week 4) 🟢 MEDIUM

**Goal**: Additional generation quality improvements

1. **Repetition penalty**
2. **Logit bias**
3. **Tests**: Verify reduced repetition

---

## 7. Recommendations

### Immediate Actions (This Week)

1. **Expose `mx::random::categorical()` in Rust**
   - File: `node/src/array.rs`
   - Add method: `pub fn categorical(&self, temperature: f32) -> Result<MxArray>`
   - Test: Verify distribution matches input probabilities

2. **Add logprobs tracking to generation**
   - File: `src/grpo/models/qwen3-model.ts`
   - Modify: `generate()` and `generateSample()`
   - Return: `GenerationResult` with logprobs array

3. **Write critical tests**
   - Logprobs correctness (compare with manual calculation)
   - Categorical sampling distribution (1000 samples)
   - EOS detection (finishedReason)

### Next Week

4. **Implement batch generation**
   - Support multiple completions per prompt
   - Handle padding for variable-length sequences

5. **Add top-k and top-p sampling**
   - Rust implementations in `node/src/sampling.rs`
   - TypeScript wrappers

### Success Criteria

- ✅ Can generate with logprobs
- ✅ Can sample from distribution (not just argmax)
- ✅ Can generate multiple completions per prompt
- ✅ All critical tests passing
- ✅ Ready for GRPO loss integration

---

## 8. Conclusion

**Architecture**: ✅ **Production-Ready**
Our Qwen3 implementation perfectly matches MLX-LM reference with all key features (QK norm, GQA, RoPE, SwiGLU).

**Generation**: ❌ **Needs Work**
Missing critical features for GRPO:

1. Logprobs tracking (CRITICAL)
2. Categorical sampling (CRITICAL)
3. Batch generation (HIGH)
4. Sampling filters (HIGH)

**Path Forward**:

- Week 1-2: Add logprobs + categorical sampling
- Week 2-3: Batch generation
- Week 3-4: Complete sampling pipeline
- Week 4+: GRPO loss integration

Once generation features are complete, we can proceed with GRPO training loop implementation.

---

**Last Updated**: January 2025
**Next Review**: After Phase 1 completion
