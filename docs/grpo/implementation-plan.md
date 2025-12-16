# GRPO Implementation Plan for MLX-Node

> **📋 COMPLETED DOCUMENT**: This implementation plan was created in early January 2025.
> **Current Status**: ✅ **ALL PHASES COMPLETE** - Project is production-ready!
> This document is preserved for historical reference.

## Executive Summary

Based on analysis of HuggingFace TRL and MLX-LM reference implementations, this document outlined a phased approach to implementing GRPO training for Qwen3 models in MLX-Node.

**Original Timeline**: 12 weeks
**Actual Completion**: January 2025 (All phases complete)
**Focus**: Qwen3 models with full GRPO training
**Scope**: Single-GPU training with function-based rewards

## Implementation Status Overview

| Phase                               | Original Timeline | Status      | Completion Date |
| ----------------------------------- | ----------------- | ----------- | --------------- |
| Phase 1: Text Generation & Sampling | 2 weeks           | ✅ COMPLETE | January 2025    |
| Phase 2: Logprobs & Metadata        | 2 weeks           | ✅ COMPLETE | January 2025    |
| Phase 3: Batch Generation           | 2 weeks           | ✅ COMPLETE | January 2025    |
| Phase 4: GRPO Loss & Advantages     | 2 weeks           | ✅ COMPLETE | January 2025    |
| Phase 5: Training Loop              | 2 weeks           | ✅ COMPLETE | January 2025    |
| Phase 6: Optimization & Testing     | 2 weeks           | ✅ COMPLETE | January 2025    |
| **BONUS**: Autograd Integration     | N/A               | ✅ COMPLETE | January 2025    |

## Phase 1: Text Generation & Sampling ✅ COMPLETE

### Goal

Complete the text generation pipeline with proper sampling methods to replace current argmax-based generation.

### Implementation Status

- ✅ Basic model architecture (Qwen3)
- ✅ Forward pass with KV caching
- ✅ Temperature scaling
- ✅ Greedy decoding (argmax)
- ✅ Top-k sampling
- ✅ Top-p (nucleus) sampling
- ✅ Min-p sampling
- ✅ Categorical sampling
- ✅ **BONUS**: XTC sampling (eXclude Top Choices)
- ✅ **BONUS**: Repetition penalty

### Tasks

#### 1.1 Implement Sampling Methods in Rust (Week 1)

**File**: `node/src/sampling.rs` (new)

```rust
// Top-k sampling
#[napi]
pub fn apply_top_k(logprobs: &MxArray, top_k: i32) -> Result<MxArray>

// Top-p (nucleus) sampling
#[napi]
pub fn apply_top_p(logprobs: &MxArray, top_p: f32) -> Result<MxArray>

// Min-p sampling
#[napi]
pub fn apply_min_p(logprobs: &MxArray, min_p: f32, min_tokens_to_keep: i32) -> Result<MxArray>

// Categorical sampling
#[napi]
pub fn categorical_sample(logits: &MxArray, temperature: f32) -> Result<MxArray>
```

**Reference**: `./mlx-lm/mlx_lm/sample_utils.py` (lines 112-310)

**Key Operations**:

- `mx::random::categorical()`: Sample from probability distribution
- `mx::argsort()`: Sort indices
- `mx::argpartition()`: Partition for top-k
- `mx::cumsum()`: Cumulative sum for top-p
- `mx::put_along_axis()`: Scatter values
- `mx::take_along_axis()`: Gather values

#### 1.2 Integrate Sampling into Generation (Week 1-2)

**Files**:

- `src/grpo/models/qwen3-model.ts` (update)
- `src/grpo/sampling/sampler.ts` (new)

**Updates**:

```typescript
// Replace argmax in generateSample() with proper sampling
const nextToken = categoricalSample(filteredProbs, temperature);

// Add complete sampling pipeline
class Sampler {
  constructor(config: SamplingConfig);
  sample(logits: MxArray): MxArray;
}
```

#### 1.3 Write Comprehensive Tests (Week 2)

**File**: `src/grpo/__test__/sampling.test.ts`

Tests:

- ✅ Top-k filters to k tokens
- ✅ Top-p maintains cumulative probability
- ✅ Min-p scales by max probability
- ✅ Categorical sampling produces valid distribution
- ✅ Temperature affects randomness
- ✅ Complete pipeline integrates all filters

### Deliverables

- [ ] Rust sampling functions in `node/src/sampling.rs`
- [ ] TypeScript Sampler class
- [ ] Updated generation methods in MLXCausalLM
- [ ] 30+ tests for sampling methods
- [ ] Documentation for sampling API

### Success Criteria

- All sampling methods produce valid probability distributions
- Generation with sampling is non-deterministic (temp > 0)
- Performance: < 5ms overhead per token vs argmax

---

## Phase 2: GRPO Loss Computation (2 weeks)

### Goal

Implement the core GRPO loss computation with clipped surrogate objective and all loss variants.

### Tasks

#### 2.1 Implement Core Loss in Rust (Week 3)

**File**: `node/src/losses.rs` (update)

```rust
/// GRPO loss with clipped surrogate objective
#[napi(object)]
pub struct GRPOLossConfig {
    pub epsilon: f32,           // 0.2
    pub epsilon_high: Option<f32>,  // None = same as epsilon
    pub beta: f32,              // 0.0 (no KL penalty)
    pub loss_type: String,      // "dapo", "grpo", "dr_grpo", "bnpo"
    pub importance_sampling_level: String,  // "token" or "sequence"
}

#[napi]
pub fn grpo_loss(
    per_token_logps: &MxArray,
    old_per_token_logps: &MxArray,
    advantages: &MxArray,
    completion_mask: &MxArray,
    config: GRPOLossConfig,
    ref_per_token_logps: Option<&MxArray>,
    max_completion_length: Option<i32>,
    num_items_in_batch: Option<i32>,
) -> Result<MxArray>
```

**Reference**: `./trl/trl/trainer/grpo_trainer.py` (lines 1730-1858)

**Key Operations**:

- Importance sampling ratio: `exp(log π_θ - log π_old)`
- Clipped objective: `min(r*A, clip(r,1-ε,1+ε)*A)`
- KL penalty: `exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1`
- Loss aggregation: GRPO, DAPO, Dr. GRPO, BNPO

#### 2.2 Implement Advantage Computation (Week 3-4)

**File**: `src/grpo/training/advantages.ts`

```typescript
export function computeAdvantages(
  rewards: MxArray,
  numGenerations: number,
  scaleRewards: 'group' | 'batch' | 'none'
): MxArray {
  // 1. Group rewards: reshape to (B/G, G)
  // 2. Compute mean per group
  // 3. Compute advantages: A = R - mean(R_group)
  // 4. Optional: scale by std
  return advantages;
}
```

**Reference**: `./trl/trl/trainer/grpo_trainer.py` (lines 1567-1588)

#### 2.3 Add Log-Probability Computation (Week 4)

**File**: `node/src/nn.rs` (update)

```rust
/// Compute per-token log probabilities
#[napi]
pub fn selective_log_softmax(
    logits: &MxArray,
    target_ids: &MxArray,
) -> Result<MxArray>
```

This computes `log π_θ(token_i | context)` for each token in the sequence.

#### 2.4 Write Loss Tests (Week 4)

**File**: `src/grpo/__test__/grpo-loss.test.ts`

Tests:

- ✅ Clipping works correctly (r < 1-ε, r > 1+ε)
- ✅ Loss increases with negative advantages
- ✅ Loss decreases with positive advantages
- ✅ All loss types produce valid gradients
- ✅ KL penalty reduces divergence

### Deliverables

- [ ] GRPO loss in Rust
- [ ] Advantage computation in TypeScript
- [ ] Selective log-softmax function
- [ ] 40+ tests for loss computation
- [ ] Documentation for loss API

### Success Criteria

- Loss computation matches TRL reference (within 1e-3)
- All 4 loss types implemented and tested
- Gradients flow correctly through loss

---

## Phase 3: Tokenization (1 week)

### Goal

Integrate tiktoken tokenizer for Qwen3 to enable end-to-end text generation.

### Tasks

#### 3.1 Add Tiktoken Dependency (Week 5)

```bash
npm install tiktoken
```

#### 3.2 Create Tokenizer Wrapper (Week 5)

**File**: `src/grpo/tokenization/qwen3-tokenizer.ts`

```typescript
import { Tiktoken } from 'tiktoken';

export class Qwen3Tokenizer {
  private tokenizer: Tiktoken;

  static async fromPretrained(modelPath: string): Promise<Qwen3Tokenizer>;

  encode(text: string, addSpecialTokens: boolean = true): Int32Array;
  decode(tokenIds: Int32Array, skipSpecialTokens: boolean = true): string;
  batchEncode(texts: string[]): Int32Array[];
  batchDecode(tokenIdsList: Int32Array[]): string[];

  get vocabSize(): number;
  get padTokenId(): number;
  get eosTokenId(): number;
  get bosTokenId(): number;
}
```

#### 3.3 Integrate with Model (Week 5)

Update `MLXCausalLM` to accept tokenizer:

```typescript
class MLXCausalLM {
  private tokenizer?: Qwen3Tokenizer;

  setTokenizer(tokenizer: Qwen3Tokenizer): void;

  generateText(
    prompt: string,
    maxNewTokens: number = 100,
    temperature: number = 0.8,
    topP: number = 0.95,
    topK?: number
  ): string;
}
```

### Deliverables

- [ ] Qwen3Tokenizer class
- [ ] Integration with MLXCausalLM
- [ ] Text-to-text generation API
- [ ] Tokenizer tests

### Success Criteria

- Can tokenize and detokenize text correctly
- Matches HuggingFace tokenizer output
- End-to-end text generation works

---

## Phase 4: Generation & Scoring Pipeline (2 weeks)

### Goal

Build complete pipeline for batch generation with group sampling and reward computation.

### Tasks

#### 4.1 Implement Batch Generation (Week 6)

**File**: `src/grpo/training/generator.ts`

```typescript
export class Generator {
  constructor(
    model: MLXCausalLM,
    tokenizer: Qwen3Tokenizer,
    config: GenerationConfig
  );

  async generateBatch(
    prompts: string[],
    numGenerations: number
  ): Promise<GenerationOutput> {
    // Output:
    // - promptIds: number[][]
    // - completionIds: number[][]
    // - perTokenLogProbs: number[][]
    // - completionTexts: string[]
  }
}
```

**Features**:

- Group-based generation: G completions per prompt
- KV cache management
- Token-level log-probability tracking
- Batch processing

#### 4.2 Implement Reward Interface (Week 6-7)

**File**: `src/grpo/rewards/reward-function.ts`

```typescript
export type RewardFunction = (
  prompts: string[],
  completions: string[],
  metadata?: Record<string, any>
) => Float32Array;

export class RewardManager {
  constructor(rewardFuncs: RewardFunction[], weights?: number[]);

  async computeRewards(
    prompts: string[],
    completions: string[]
  ): Promise<Float32Array>;
}
```

**Built-in Rewards**:

```typescript
// Length reward: prefer longer completions
export function lengthReward(prompts: string[], completions: string[]): Float32Array;

// Diversity reward: unique tokens
export function diversityReward(prompts: string[], completions: string[]): Float32Array;

// Format reward: follows pattern (e.g., JSON)
export function formatReward(pattern: RegExp): RewardFunction;
```

#### 4.3 Build Complete Pipeline (Week 7)

**File**: `src/grpo/training/pipeline.ts`

```typescript
export class GRPOPipeline {
  async generateAndScore(
    prompts: string[]
  ): Promise<{
    promptIds: number[][];
    completionIds: number[][];
    completionMask: number[][];
    rewards: Float32Array;
    advantages: Float32Array;
    oldPerTokenLogProbs: number[][];
  }>;
}
```

### Deliverables

- [ ] Batch generation with group sampling
- [ ] Reward function interface
- [ ] Built-in reward functions
- [ ] Complete pipeline
- [ ] Pipeline tests

### Success Criteria

- Can generate multiple completions per prompt
- Reward computation works correctly
- Advantages are zero-mean within groups

---

## Phase 5: Training Loop (3 weeks)

### Goal

Implement complete GRPO training loop with gradient accumulation.

### Tasks

#### 5.1 Implement Training Step (Week 8)

**File**: `src/grpo/training/trainer.ts`

```typescript
export class GRPOTrainer {
  async trainingStep(prompts: string[]): Promise<TrainingMetrics> {
    // 1. Generate and score
    const batch = await this.pipeline.generateAndScore(prompts);

    // 2. Forward pass to get current log-probs
    const perTokenLogProbs = this.model.computeLogProbs(
      batch.promptIds,
      batch.completionIds
    );

    // 3. Compute loss
    const loss = Losses.grpoLoss(
      perTokenLogProbs,
      batch.oldPerTokenLogProbs,
      batch.advantages,
      batch.completionMask,
      this.config
    );

    // 4. Backward pass
    const grads = Gradients.compute(loss, this.model.parameters());

    // 5. Optimizer step
    this.optimizer.step(grads);

    return metrics;
  }
}
```

#### 5.2 Add Gradient Accumulation (Week 8-9)

```typescript
async train(dataset: Dataset): Promise<void> {
  for (const epoch of range(this.config.numEpochs)) {
    for (const batch of dataset.batches(this.config.batchSize)) {
      // Accumulate gradients over multiple steps
      for (let step = 0; step < this.config.gradientAccumulationSteps; step++) {
        const subBatch = batch.slice(step);
        await this.trainingStep(subBatch);
      }

      // Update after accumulation
      this.optimizer.update();
      this.optimizer.zeroGrad();
    }
  }
}
```

#### 5.3 Add Multiple Iterations Support (Week 9)

**Reference**: `num_iterations` parameter in TRL (default: 1)

Support multiple gradient steps per generation batch:

```typescript
for (let iter = 0; iter < this.config.numIterations; iter++) {
  // Use same generation but update model multiple times
  const loss = this.computeLoss(cachedBatch);
  this.optimizer.step(loss);
}
```

#### 5.4 Implement Dataset Handling (Week 9-10)

**File**: `src/grpo/data/dataset.ts`

```typescript
export class PromptDataset {
  static async fromJSON(path: string): Promise<PromptDataset>;

  batches(batchSize: number): AsyncIterable<string[]>;
  shuffle(): void;
  get length(): number;
}
```

**Format**:

```json
[
  {"prompt": "What is 2+2?"},
  {"prompt": "Explain quantum computing."},
  ...
]
```

### Deliverables

- [ ] Complete training loop
- [ ] Gradient accumulation
- [ ] Multiple iterations support
- [ ] Dataset loading
- [ ] Training tests

### Success Criteria

- Training loop completes without errors
- Loss decreases over time
- Metrics are logged correctly

---

## Phase 6: Monitoring & CLI (2 weeks)

### Goal

Add comprehensive logging, monitoring, and a CLI for easy training.

### Tasks

#### 6.1 Implement Metrics Logging (Week 10-11)

**File**: `src/grpo/training/metrics.ts`

```typescript
export class MetricsLogger {
  log(metrics: TrainingMetrics): void;
  logToConsole(): void;
  logToFile(path: string): void;
  logToWandB?(config: WandBConfig): void;
}
```

**Metrics to Track**:

- Loss (total, policy, KL)
- Rewards (mean, std, per function)
- Advantages (mean, std)
- Clipping ratio
- Entropy
- Completion lengths
- Tokens per second
- GPU memory usage

#### 6.2 Create CLI Interface (Week 11)

**File**: `src/grpo/cli/train.ts`

```typescript
#!/usr/bin/env node
import { Command } from 'commander';

const program = new Command();

program
  .name('mlx-grpo')
  .description('Train Qwen3 models with GRPO')
  .option('-m, --model <path>', 'Model path or name')
  .option('-d, --dataset <path>', 'Dataset JSON file')
  .option('-o, --output <path>', 'Output directory')
  .option('--lr <number>', 'Learning rate', '1e-6')
  .option('--epochs <number>', 'Number of epochs', '3')
  .option('--batch-size <number>', 'Batch size', '4')
  .option('--num-generations <number>', 'Completions per prompt', '8')
  .action(async (options) => {
    const trainer = new GRPOTrainer(options);
    await trainer.train();
  });

program.parse();
```

**Usage**:

```bash
mlx-grpo train \
  --model qwen3-0.6b \
  --dataset data/math.json \
  --output checkpoints/qwen3-math \
  --lr 1e-6 \
  --epochs 3
```

#### 6.3 Add Checkpointing (Week 11-12)

```typescript
async saveCheckpoint(path: string): Promise<void> {
  await ModelLoader.saveModel(this.model, path);
  // Save optimizer state
  // Save training state (epoch, step, etc.)
}

async loadCheckpoint(path: string): Promise<void> {
  this.model = await ModelLoader.loadPretrained(path);
  // Load optimizer state
  // Load training state
}
```

### Deliverables

- [ ] Metrics logging system
- [ ] CLI for training
- [ ] Checkpointing
- [ ] Progress bars
- [ ] Documentation

### Success Criteria

- Can train from CLI with simple command
- Metrics are logged clearly
- Can resume from checkpoints

---

## Phase 7: Testing & Documentation (2 weeks)

### Goal

Comprehensive tests and documentation for production use.

### Tasks

#### 7.1 Integration Tests (Week 12)

**File**: `src/grpo/__test__/integration.test.ts`

```typescript
describe('GRPO Integration', () => {
  it('should train for 10 steps without errors', async () => {
    const trainer = new GRPOTrainer({
      modelConfig: 'qwen3-0.6b',
      learningRate: 1e-6,
      // ...
    });

    const dataset = await PromptDataset.fromJSON('test-data.json');
    await trainer.train(dataset);

    expect(trainer.getMetrics().loss).toBeLessThan(initialLoss);
  });

  it('should generate improved completions after training', async () => {
    // Compare completions before and after training
  });
});
```

#### 7.2 Write Documentation (Week 12)

**Files**:

- `src/grpo/README.md`: Overview and quick start
- `src/grpo/docs/api.md`: API reference
- `src/grpo/docs/training.md`: Training guide
- `src/grpo/docs/rewards.md`: Custom reward functions
- `src/grpo/examples/`: Example scripts

### Deliverables

- [ ] 100+ integration tests
- [ ] Complete API documentation
- [ ] Training tutorials
- [ ] Example scripts
- [ ] Troubleshooting guide

### Success Criteria

- All tests pass
- Documentation is clear and comprehensive
- Examples work out of the box

---

## Success Metrics

### Performance

- **Training speed**: > 100 tokens/sec on M1 Max
- **Memory usage**: < 16GB for Qwen3-0.6B
- **Loss convergence**: Similar to TRL reference

### Quality

- **Test coverage**: > 90%
- **Type safety**: Full TypeScript types
- **Documentation**: All public APIs documented

### Functionality

- ✅ Supports Qwen3-0.6B and Qwen3-1.8B
- ✅ All 4 loss types (GRPO, DAPO, Dr. GRPO, BNPO)
- ✅ All sampling methods (top-k, top-p, min-p)
- ✅ Custom reward functions
- ✅ Gradient accumulation
- ✅ Multiple iterations
- ✅ Checkpointing and resuming

---

## Next Steps

**Immediate Next Action**: Start Phase 1 (Text Generation & Sampling)

1. Create `node/src/sampling.rs`
2. Implement `apply_top_k()` function
3. Add tests for top-k sampling
4. Continue with top-p, min-p, categorical

**Weekly Checkpoints**:

- End of Week 2: All sampling methods working
- End of Week 4: GRPO loss computation complete
- End of Week 5: Tokenization integrated
- End of Week 7: Generation pipeline complete
- End of Week 10: Training loop working
- End of Week 12: Production-ready release

---

## References

- **Algorithm Reference**: `./src/grpo/GRPO_ALGORITHM.md`
- **Architecture Reference**: `./src/grpo/ARCHITECTURE.md`
- **TRL Source**: `./trl/trl/trainer/grpo_trainer.py`
- **MLX-LM Source**: `./mlx-lm/mlx_lm/sample_utils.py`
- **Project Status**: `./CLAUDE.md`
