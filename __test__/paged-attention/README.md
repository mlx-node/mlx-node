# Paged Attention Test Suite

This directory contains comprehensive tests for the PagedAttention system in MLX-Node.

## Overview

PagedAttention is a memory-efficient KV cache management system that uses block-based allocation to reduce memory waste from 60-80% to <4%. The implementation is based on the [PagedAttention paper](https://arxiv.org/abs/2309.06180) and uses Metal kernel acceleration for high performance.

## Test Files

### `paged-kv-cache.test.ts`

Unit tests for the `PagedKVCache` class, covering:

**Constructor and Configuration (5 tests)**

- Creating cache with valid config
- Validating block size (must be 8, 16, or 32)
- Validating head size (must be Metal kernel-compatible)
- Calculating number of blocks from memory budget
- FP8 cache support (2x memory efficiency)

**Sequence Management (5 tests)**

- Adding and removing sequences
- Allocating correct number of blocks
- Handling insufficient memory gracefully
- Supporting multiple concurrent sequences
- Checking allocation before adding sequences

**Block Management (3 tests)**

- Getting slot mappings for sequences
- Building block tables for batches
- Extending sequences with new tokens

**Memory Statistics (3 tests)**

- Tracking total blocks correctly
- Updating used/free blocks when allocating
- Restoring free blocks when removing sequences

**Edge Cases (4 tests)**

- Handling single-token sequences
- Handling sequences exactly matching block size
- Handling very long sequences
- Error handling for invalid sequence IDs

**Total: 20 test cases**

### `paged-attention-qwen3.test.ts`

Integration tests for PagedAttention with Qwen3Model, covering:

**API Availability (3 tests)**

- `hasPagedAttention()` method exposure
- Default paged attention state
- `getPagedMemoryStats()` method exposure

**Configuration (3 tests)**

- Enabling paged attention via config option
- Using default paged attention config
- Validating paged attention config

**Memory Statistics (3 tests)**

- Returning memory stats when enabled
- Updating stats during generation
- Returning null when not enabled

**Generation with Paged Attention (3 tests)**

- Generating text with paged attention
- Handling multiple sequential generations
- Comparing output with non-paged attention

**Memory Efficiency (3 tests)**

- Reducing memory waste vs traditional cache
- Efficiently handling variable-length sequences
- Tracking maximum memory usage

**Error Handling (2 tests)**

- Handling out-of-memory gracefully
- Clearing cache on error

**FP8 Quantization (2 tests)**

- Supporting FP8 cache for memory efficiency
- Maintaining quality with FP8 cache

**Total: 19 test cases**

## Current Status

⚠️ **All tests are currently skipped** because:

1. **PagedKVCache is not exposed via NAPI bindings**
   - The `PagedKVCache` and `PagedAttentionConfig` types exist in the `mlx-paged-attn` Rust crate
   - They are not yet exported to TypeScript via `#[napi]` annotations
   - Once bindings are added, tests in `paged-kv-cache.test.ts` can be enabled

2. **Integration tests require model files**
   - Tests in `paged-attention-qwen3.test.ts` require a Qwen3 model to be available
   - The `loadTestModel` utility is also not currently available

## Enabling Tests

### Step 1: Add NAPI Bindings

Add NAPI exports to `crates/mlx-core/src/transformer/paged_attention.rs`:

```rust
#[napi]
pub struct PagedKVCache {
    inner: mlx_paged_attn::PagedKVCache,
}

#[napi(object)]
pub struct PagedAttentionConfig {
    pub block_size: u32,
    pub gpu_memory_mb: u32,
    pub head_size: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub use_fp8_cache: Option<bool>,
    pub max_seq_len: Option<u32>,
    pub max_batch_size: Option<u32>,
}

#[napi]
impl PagedKVCache {
    #[napi(constructor)]
    pub fn new(config: PagedAttentionConfig) -> Result<Self> {
        // ...
    }

    #[napi]
    pub fn add_sequence(&mut self, prompt_len: u32) -> Result<u32> {
        // ...
    }

    #[napi]
    pub fn remove_sequence(&mut self, seq_id: u32) -> Result<()> {
        // ...
    }

    // ... other methods
}
```

### Step 2: Export from TypeScript

Add to `packages/core/src/index.ts`:

```typescript
export { PagedKVCache, PagedAttentionConfig } from './index.cjs';
```

### Step 3: Rebuild

```bash
yarn build:native
```

### Step 4: Enable Tests

Remove `.skip` from test descriptions:

```diff
- describe.skip('PagedKVCache', () => {
+ describe('PagedKVCache', () => {
```

And uncomment test code.

## API Documentation

### PagedKVCache

```typescript
interface PagedAttentionConfig {
  blockSize: 8 | 16 | 32; // Block size in tokens
  gpuMemoryMb: number; // GPU memory budget in MB
  headSize: number; // Head dimension (must be Metal-compatible)
  numKvHeads: number; // Number of KV heads
  numLayers: number; // Number of transformer layers
  useFp8Cache?: boolean; // Enable FP8 quantization (default: false)
  maxSeqLen?: number; // Maximum sequence length (default: 8192)
  maxBatchSize?: number; // Maximum batch size (default: 256)
}

class PagedKVCache {
  constructor(config: PagedAttentionConfig);

  // Sequence management
  addSequence(promptLen: number): number;
  removeSequence(seqId: number): void;
  extendSequence(seqId: number, numTokens: number): void;

  // Memory management
  canAllocate(blocksNeeded: number): boolean;
  getMemoryStats(): MemoryStats;

  // Block operations
  getSlotMapping(seqIds: number[]): Int32Array;
  buildBlockTables(seqIds: number[]): Int32Array;
}

interface MemoryStats {
  totalBlocks: number;
  numFreeBlocks: number;
  numUsedBlocks: number;
}
```

### Qwen3Model Integration

```typescript
interface Qwen3Config {
  // ... existing config
  usePagedAttention?: boolean;
  pagedAttentionConfig?: PagedAttentionConfig;
}

class Qwen3Model {
  // Check if paged attention is enabled
  hasPagedAttention(): boolean;

  // Get memory statistics (null if not enabled)
  getPagedMemoryStats(): MemoryStats | null;
}
```

## Test Patterns

Tests follow the existing patterns from `batch-kv-cache.test.ts`:

1. **Use TypedArray helpers** from `test-utils.ts` (`int32`, `shape`, `assertShape`)
2. **Test positive and negative cases** (valid config, invalid config)
3. **Verify memory management** (allocation, deallocation, statistics)
4. **Test edge cases** (single token, exact block size, very long sequences)
5. **Document expected behavior** in comments

## References

- **PagedAttention Paper**: https://arxiv.org/abs/2309.06180
- **Rust Implementation**: `crates/mlx-paged-attn/`
- **Metal Kernels**: `crates/mlx-paged-attn/metal/`
- **Similar Tests**: `__test__/core/batch-kv-cache.test.ts`

## Running Tests

```bash
# Run all paged attention tests (currently all skip)
yarn vitest run __test__/paged-attention/

# Run specific test file
yarn vitest run __test__/paged-attention/paged-kv-cache.test.ts

# Run with verbose output
yarn vitest run __test__/paged-attention/ --reporter=verbose

# Once enabled, run without skipping
yarn vitest run __test__/paged-attention/ --run
```

## Future Enhancements

1. **Continuous Batching Scheduler Tests**
   - Add tests for `ContinuousBatchingScheduler`
   - Test dynamic batch composition
   - Test prefill/decode balancing

2. **Performance Benchmarks**
   - Memory efficiency vs traditional cache
   - Throughput comparison
   - Latency impact

3. **Metal Kernel Tests**
   - Direct kernel dispatch tests
   - Correctness verification
   - Performance profiling

---

**Status**: Test suite ready, waiting for NAPI bindings
**Last Updated**: January 26, 2026
**Total Test Cases**: 39 (20 unit + 19 integration)
