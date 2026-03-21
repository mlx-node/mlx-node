# Prompt Cache Reuse Across Multi-Turn Chat

## Problem

Every `chat()` / `chatStream()` call re-tokenizes the full conversation history and re-prefills from token 0. For a multi-turn tool-calling loop where scraped content grows the conversation to 10K+ tokens, turn N re-processes all N*10K tokens. This wastes compute that scales linearly with conversation length.

## Solution

Return an opaque `PromptCache` object in `ChatResult`. When passed back to the next `chat()` call, it enables incremental prefill — only the new tokens (since the last turn) are processed.

## API Design

### TypeScript Interface

```typescript
// PromptCache — opaque handle to KV cache state (native class)
interface PromptCache {
  readonly tokenCount: number;  // total tokens in this cache
  dispose(): void;              // explicit cleanup (also freed on GC)
}

// ChatConfig additions
interface ChatConfig {
  // ... existing fields ...
  cache?: PromptCache;      // cache from a previous chat() call
  reuseCache?: boolean;     // save cache in response (default: true)
}

// ChatResult additions
interface ChatResult {
  // ... existing fields ...
  cache: PromptCache | null;  // null when reuseCache is false
}

// ChatStreamFinal additions (final event in chatStream)
interface ChatStreamFinal {
  // ... existing fields ...
  cache: PromptCache | null;
}
```

### Usage

```typescript
const r1 = await model.chat(messages, { tools })
// r1.cache holds KV state for the entire conversation + response

messages.push({ role: 'assistant', content: r1.rawText })
messages.push({ role: 'user', content: formatToolResponse(result) })

const r2 = await model.chat(messages, { tools, cache: r1.cache })
// Only prefills the new tokens (assistant reply + tool response + gen prompt)
// r1.cache is consumed — don't reuse it after this

// Opt out of caching
const r3 = await model.chat(messages, { reuseCache: false })
// r3.cache is null, no extra GPU memory held
```

### Behavior Matrix

| `cache` param | `reuseCache` | Behavior |
|---|---|---|
| None | true (default) | Full prefill, return new cache |
| None | false | Full prefill, return null cache |
| Provided | true (default) | Incremental prefill if prefix matches, return updated cache |
| Provided | false | Incremental prefill if prefix matches, return null cache |

## Internal Design

### PromptCache Struct (Rust/NAPI)

```rust
#[napi]
pub struct PromptCache {
    /// Per-layer KV cache states (KVCache for full-attn, ArraysCache for GDN)
    pub(crate) caches: Vec<Qwen3_5LayerCache>,
    /// Full token sequence that produced this cache state
    /// (template tokens + generated tokens, excluding final EOS)
    pub(crate) token_history: Vec<u32>,
    /// Model identifier to prevent cross-model cache misuse
    pub(crate) model_type: String, // "qwen3_5" | "qwen3_5_moe"
}
```

### Prefix Verification Algorithm

When `cache` is provided:

```
1. all_tokens = apply_chat_template(messages)  // full conversation
2. cached_tokens = cache.token_history
3. cached_len = cached_tokens.len()

4. if all_tokens.len() < cached_len:
     // Conversation was shortened — full re-prefill
     GOTO full_prefill

5. if all_tokens[..cached_len] != cached_tokens:
     // Prefix mismatch (user edited earlier messages) — full re-prefill
     GOTO full_prefill

6. // Prefix matches — incremental prefill
   new_tokens = all_tokens[cached_len..]
   restore model caches from cache.caches
   prefill(new_tokens)    // only the delta
   GOTO decode

full_prefill:
   reset all caches
   prefill(all_tokens)    // from scratch

decode:
   // ... normal decode loop ...
```

### Why Prefix Matching Works

In ChatML format, each message is independently formatted:
```
<|im_start|>role\ncontent<|im_end|>\n
```

Turn 1 cache contains: `[system_tokens, user1_tokens, gen_prompt_tokens, generated_tokens]`

Turn 2 tokenization: `[system_tokens, user1_tokens, assistant1_tokens, tool_response_tokens, gen_prompt_tokens_2]`

For the prefix to match, `assistant1_tokens` must equal `gen_prompt_tokens + generated_tokens + separator`. This holds when:
- The assistant message content (`rawText`) is passed back unchanged
- The chat template reconstructs the same token sequence from `rawText`

The Qwen3.5 template splits thinking content on `</think>` and reconstructs with `<think>\n` + reasoning + `\n</think>\n\n` + content. Minor whitespace normalization (`.rstrip('\n')`, `.lstrip('\n')`) can cause mismatches in edge cases. The fallback to full re-prefill handles this safely.

### Saving Cache After Generation

After the decode loop completes:

```
token_history = all_tokens            // template-applied input tokens
             ++ generated_tokens       // model output (including <|im_end|>)

// For Qwen3.5 ChatML, also append the newline separator
// that the template places after <|im_end|>
token_history.push(newline_token_id)

cache = PromptCache {
    caches: extract_current_caches(),
    token_history,
    model_type: "qwen3_5_moe",
}
```

### Compiled C++ Path Integration

The compiled C++ decode path stores caches in process-wide globals (`g_moe_caches`). For cache reuse:

**Saving** (after decode, before `MoeResetGuard` drops):
1. Add `mlx_qwen35_moe_export_caches()` — copies `g_moe_caches` to output array pointers
2. Call this before `MoeResetGuard` drops to extract cache state
3. Wrap extracted arrays in `Qwen3_5LayerCache` objects for the `PromptCache`

**Restoring** (on next call with `cache` provided):
1. Place `cache.caches` into the model's `caches_guard`
2. Run incremental prefill through `forward_inner()` (Rust path, not compiled)
3. Call `init_from_prefill()` as usual — it copies updated caches to C++ globals
4. `max_kv_len` = `cache.tokenCount + new_prefill_len + max_new_tokens` (rounded to 256)
5. Compiled decode loop runs normally

The compiled graph may re-compile if `max_kv_len` changes shape, but this is a one-time ~100ms cost per new allocation size.

### Thread Safety

- `PromptCache` holds `MxArray` values (via `Qwen3_5LayerCache`), which are `Send` but not `Sync`
- The compiled mutex (`MOE_COMPILED_MUTEX`) serializes all compiled-path operations
- A `PromptCache` should not be used concurrently from multiple `chat()` calls — document that passing a cache to `chat()` **consumes** it (the cache fields are moved out)
- After `chat()` returns, the old cache is empty — only the new `result.cache` is valid

### Model Type Validation

```rust
if cache.model_type != self.model_type() {
    return Err(Error::from_reason(
        format!("Cache type '{}' doesn't match model type '{}'",
                cache.model_type, self.model_type())
    ));
}
```

### Memory Considerations

For Qwen3.5-35B-A3B MoE (40 layers, 10 full-attention, 30 GDN):
- KV cache per full-attn layer at 4K tokens: ~4 MB (2 heads * 256 dim * 4096 * 2 bytes * 2)
- 10 layers: ~40 MB
- GDN recurrent states: ~3 MB per layer, 30 layers: ~90 MB
- Token history (u32 array): ~40 KB at 10K tokens
- **Total per PromptCache: ~130 MB** at 4K conversation tokens

This grows linearly with conversation length but is small compared to model weights (~65 GB).

## Scope

### In Scope
- `PromptCache` NAPI class with `tokenCount` and `dispose()`
- `cache` and `reuseCache` fields on `ChatConfig`
- `cache` field on `ChatResult` and `ChatStreamFinal`
- Prefix verification with safe fallback
- Compiled C++ path cache export
- Qwen3_5Model (dense) and Qwen3_5MoeModel (MoE)

### Out of Scope (Future)
- `generate()` method cache reuse (lower-level API, less common)
- Qwen3Model cache reuse (older model, different architecture)
- Cache serialization to disk (save/load)
- LRU cache pool for server-style prefix matching
- Context shifting (pruning old messages when context fills)
