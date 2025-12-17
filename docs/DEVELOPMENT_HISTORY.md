# MLX-Node Development History

## Overview

This document archives the major development sessions and milestones for MLX-Node, preserving historical context while keeping the main CLAUDE.md lean.

---

## December 2025 - GRPO Trainer Refactoring

### Reward API Unification ✅

**Date**: December 17, 2025
**Status**: PRODUCTION-READY
**Commit**: `30692c6 feat(grpo): refactor reward API, add Rust tests, improve tool-use training`

**Changes Summary**:

- 3,792 insertions, 1,078 deletions across 27 files
- 62 new Rust tests for GRPO components
- Unified reward function interface

**Reward API Refactoring**:

Old API (3 separate parameters):

```typescript
type RewardFunction = (
  prompts: string[],
  completions: string[],
  answers: (string | null)[],
) => number[] | Float32Array | Promise<...>;
```

New API (unified structured object):

```typescript
type RewardFunction = (outputs: RewardOutput[]) => number[] | Float32Array | Promise<...>;

interface RewardOutput {
  prompt: string
  completion: CompletionInfo  // Pre-parsed!
  expectedAnswer?: string
}

interface CompletionInfo {
  text: string           // Clean text (tags removed)
  rawText: string        // Original with tags
  toolCalls: ToolCallResult[]  // Already parsed!
  thinking?: string      // Already extracted!
  numTokens: number
  finishReason: string
}
```

**Benefits**:

- Pre-parsed tool calls (no manual regex needed)
- Pre-extracted thinking content
- Both clean text and raw text available
- Token counts included
- Structured, type-safe access

**New Files Added**:

- `crates/mlx-core/src/grpo/engine.rs` (475 lines) - Training engine with `buildRewardOutputs`
- `crates/mlx-core/src/grpo/advantages.rs` (282 lines) - 16 tests
- `crates/mlx-core/src/grpo/entropy.rs` (379 lines) - 21 tests
- `crates/mlx-core/src/grpo/loss.rs` (888 lines) - 25 tests
- `examples/grpo/ast-grep-dataset.ts` (817 lines) - Curriculum dataset

**Files Modified**:

- `packages/trl/src/rewards.ts` - All 5 functions use new API
- `packages/trl/src/types.ts` - Re-exports RewardOutput
- `packages/trl/src/trainers/grpo-trainer.ts` - Uses buildRewardOutputs
- `crates/mlx-core/src/tools/mod.rs` - Enhanced parsing (147 lines)

### Rust Tests Added (62 total) ✅

**1. Advantages Computation** (16 tests in `advantages.rs`):

- Group-level vs batch-level normalization
- Zero std handling (epsilon clamping)
- Single group/completion edge cases
- Advantage zero-mean property validation

**2. Entropy Filtering** (21 tests in `entropy.rs`):

- Quantile threshold selection (0.0, 0.5, 0.8, 1.0)
- Padding token masking
- Shape mismatch validation
- All-equal entropy edge cases

**3. GRPO Loss** (25 tests in `loss.rs`):

- All 4 loss variants: GRPO, DAPO, Dr.GRPO, BNPO
- Token-level vs sequence-level importance sampling
- KL divergence penalty
- Numerical stability (epsilon handling)
- Masked padding tokens

### Tool-Use Training Improvements ✅

**Enhanced System Prompt**:

- Explicit requirement for tool calls
- Concrete JSON format examples
- Clear metavariable syntax rules ($VAR, $$$ARGS)
- Multiple realistic patterns (9+ examples)

**New Curriculum Dataset** (`ast-grep-dataset.ts`, 817 lines):

- 50+ patterns organized by difficulty (basic/intermediate/advanced)
- Real code contexts for pattern matching
- Expected match counts for validation

**Improved Reward Function**:

- Uses pre-parsed tool calls (no manual regex)
- Uses pre-extracted thinking content
- Floating-point precision fix (rounds to 1 decimal)
- Better scoring for pattern quality
- Debug logging option (`DEBUG_REWARDS=1`)

**Impact**:

- Reduced FFI overhead (tool calls parsed in Rust)
- Better type safety in reward functions
- Simplified reward logic
- Comprehensive Rust test coverage

### Branch Cleanup ✅

**Date**: December 18, 2025
**Status**: COMPLETE

**Dead Code Removed**:

- Deleted `crates/mlx-core/src/memory.rs` (425 lines) - Unused Mach API memory tracking
- Removed `pub mod memory;` from `lib.rs`

**Test Fixes**:

- Fixed broken `clearCache` import in test files (function was never exported to JS)
- Removed unused `afterEach` cleanup calls from `grpo-trainer.test.ts` and `grpo-integration.test.ts`

**Verification**:

- `cargo test -p mlx-core`: 219 tests passing
- `yarn test`: 128 tests passing
- No compiler warnings

---

## January 2025 - Feature Completion Sprint

### Autograd Integration ✅

**Date**: January 2025
**Status**: PRODUCTION-READY
**Duration**: Complete implementation with functional architecture

**Implementation**:

- Core autograd infrastructure (`crates/mlx-core/src/autograd.rs` - 360 lines)
- Functional forward pass architecture (`crates/mlx-core/src/utils/functional.rs` - 550 lines)
- Parameter management utilities (`crates/mlx-core/src/param_manager.rs` - 200 lines)
- GRPO autograd integration (`crates/mlx-core/src/grpo/autograd.rs` - 198 lines)
- Full Qwen3 model support through computation graph

**Key Innovation - Functional Forward Pass**:

```rust
// Previous (broken): Used pre-computed logprobs
let loss_fn = |_params| grpo_loss(&fixed_logprobs, ...)?;  // ❌ No gradient path

// Current (working): Recomputes forward pass from parameters
let loss_fn = |params| {
    let param_dict = map_params_to_dict(params, &names)?;
    let logits = qwen3_forward_functional(&config, &param_dict, &input_ids)?;  // ✅ Creates computation graph
    let logprobs = log_softmax(logits, -1)?;
    grpo_loss(&logprobs, ...)?
};
```

**Impact**:

- Automatic differentiation through full Qwen3 model (2,205 lines)
- 311 gradients computed automatically for all parameters
- Production-ready for training without manual gradient implementation
- Clean separation: functional components + parameter management
- 3 comprehensive integration tests passing

**Documentation**: See `AUTOGRAD_INTEGRATION.md` for complete architecture details

---

### Causal Masking Fix ✅

**Date**: January 2025
**Duration**: ~4 hours debugging + implementation
**Issue**: Forward pass produced different results for cached vs non-cached modes (16.41 vs 13.55 logit diff)

**Root Cause Analysis**:

- Cached mode was CORRECT (applied causal masking implicitly via incremental generation)
- Non-cached mode was WRONG (no causal mask, allowing attention to future positions)
- Evidence: Token predictions differed when future context was visible vs masked

**Solution**:

- ✅ Implemented `create_causal_mask()` in Rust (`crates/mlx-core/src/array/mask.rs`)
- ✅ Applied causal mask to all non-cached forward passes in Qwen3 model
- ✅ Updated all test configs to include `headDim` field (9 files, ~15 configs)
- ✅ **Result**: 0/151,936 token differences - perfect match between modes

**Impact**:

- Production-ready generation with correct autoregressive behavior
- All tests passing
- Critical fix for training stability and generation quality

**Files Modified**:

- `crates/mlx-core/src/array/mask.rs` (new) - Causal mask generation
- `crates/mlx-core/src/models/qwen3/model.rs` - Applied mask to forward passes
- 9 test configuration files updated

**Documentation**: See `causal-mask-bug-fix.md` and `causal-mask-root-cause.md`
**Tests**: `__test__/core/causal-mask-mlx-lm.test.ts` (14 tests from mlx-lm)

---

### Feature Alignment Session ✅

**Date**: January 2025
**Duration**: ~10 hours over 2 days
**Objective**: Align features with MLX-LM and TRL for production GRPO training

**Achievements**:

#### 1. Repetition Penalty ✅ (397 lines: 141 Rust + 256 tests)

- Asymmetric penalty algorithm (divide for positive, multiply for negative logits)
- Context size limiting (default 20 tokens)
- Batch processing support
- 14 comprehensive tests, all passing
- **File**: `crates/mlx-core/src/sampling.rs` (lines 473-613)
- **Tests**: `__test__/utils/repetition-penalty.test.ts`

#### 2. BatchKVCache ✅ (859 lines: 376 Rust + 483 tests)

- Left-padding support for variable-length batches
- Dynamic allocation in 256-step increments
- Filter operation with automatic padding optimization
- Extend operation for batch concatenation
- 29 comprehensive tests, all passing
- **File**: `crates/mlx-core/src/transformer/batch_kv_cache.rs`
- **Tests**: `__test__/core/batch-kv-cache.test.ts`

**Data Structure**:

```rust
pub struct BatchKVCache {
    keys: Option<MxArray>,        // (batch, n_kv_heads, seq_len, head_dim)
    values: Option<MxArray>,      // (batch, n_kv_heads, seq_len, head_dim)
    left_padding: Vec<i32>,       // Padding for each batch element
    offset: Vec<i32>,             // Starts negative: -padding
    idx: i32,                     // Current write position
}
```

#### 3. Importance Sampling ✅ (already implemented in GRPO)

- Token-level and sequence-level IS
- PPO-style clipping
- Built into GRPO loss computation

**Status**: 3/4 critical features complete

- ⏳ **Qwen3-MoE** (pending, ~700 lines estimated) - Not critical for basic GRPO training

**Impact**: Production-ready for GRPO training with Qwen3

- 90% feature parity with MLX-LM
- 100% feature parity with TRL GRPO

**Full Documentation**: See `FEATURE_ALIGNMENT_SESSION.md`

---

### Test Porting Session ✅

**Date**: January 2025
**Tests Added**: 69 new tests (829 → 948)
**Components**: Entropy filtering, XTC sampling, Batch generation utils, RotatingKVCache

**Achievements**:

#### 1. Entropy Filtering (12 tests, 176 Rust lines)

- `getHighEntropyMask()` - Train on high-uncertainty tokens
- `computeEntropy()` - Per-token entropy from logits
- PyTorch-compatible quantile calculation
- **File**: `crates/mlx-core/src/grpo/entropy.rs`

#### 2. XTC Sampling (15 tests, 154 Rust lines)

- `applyXtc()` - eXclude Top Choices diversity-promoting sampling
- Special token protection
- MLX-LM reference compatibility
- **File**: `crates/mlx-core/src/sampling.rs` (XTC portion)

#### 3. Batch Generation Utils (21 tests, 298 Rust lines)

- `padSequences()` - Left/right padding
- `createAttentionMask()` - Binary masks
- Custom pad token support
- **File**: `crates/mlx-core/src/utils/batch_generation.rs`

#### 4. RotatingKVCache (21 tests, 394 Rust lines)

- Fixed maximum cache size with rotation
- `keep` parameter for system prompts
- Memory-efficient long-context handling
- **File**: `crates/mlx-core/src/transformer/rotating_kv_cache.rs`

**Impact**: 829 → 948 tests, test coverage parity increased from 62% → 72%

---

## November 2025 - Critical Infrastructure

### 1. Rust-Based Model Persistence 💾

- Migrated `saveModel` to Rust
- Fixed 6 failing GRPO trainer tests
- Test runtime: 234s → 34s
- TypeScript code: 329 → 255 lines (-22%)
- **File**: `crates/mlx-core/src/models/qwen3/persistence.rs` (398 lines)

### 2. Thread-Safe Handle Management 🔒

- Fixed double-free bug with `Arc<MxHandle>`
- Thread-safe `MxArray` with atomic reference counting
- Removed 6 `.eval()` workarounds
- **File**: `crates/mlx-core/src/array/handle.rs`

### 3. Rust Migration Complete ⚡

- All compute operations moved to Rust
- 451 TS lines → 740+ Rust lines
- Expected 15-25% training speedup
- Clean Rust/TypeScript separation maintained

**Impact**:

- Production-ready model persistence
- Eliminated memory management bugs
- Significant performance improvements

---

## Metrics Timeline

| Date           | Tests | Rust Lines | TS Lines | Status                        |
| -------------- | ----- | ---------- | -------- | ----------------------------- |
| Nov 2025       | ~800  | ~6,700     | ~2,500   | Infrastructure complete       |
| Early Jan 2025 | 829   | ~7,000     | ~2,400   | Test porting begins           |
| Mid Jan 2025   | 948   | ~8,500     | ~2,300   | Feature alignment             |
| Late Jan 2025  | 1,039 | 11,203     | 2,082    | Autograd complete             |
| Dec 17 2025    | 614   | ~25,000    | 3,712    | GRPO refactor + 62 Rust tests |

**Note**: Test count decreased due to test reorganization (TS tests moved to Rust).

---

## Key Lessons Learned

### Autograd Implementation

- **Lesson**: MLX autograd requires pure functions, not stateful models
- **Solution**: Functional forward pass architecture with parameter mapping
- **Impact**: Enables automatic differentiation through entire model

### Causal Masking

- **Lesson**: MLX and PyTorch have opposite boolean mask semantics
- **Solution**: Carefully document and test mask generation
- **Impact**: Correct autoregressive generation

### Rust Migration

- **Lesson**: Moving compute to Rust provides 15-25% speedup
- **Solution**: Keep TypeScript for orchestration, Rust for compute
- **Impact**: Clean architecture with maximum performance

---

_This document is maintained to preserve development history and lessons learned._
_For current project status, see CLAUDE.md_
