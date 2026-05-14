# Privacy-Filter (`openai/privacy-filter`) — Design Spec

**Date**: 2026-05-13
**Branch**: `worktree-privacy-filter`
**Status**: Draft, pending user review
**Model**: <https://huggingface.co/openai/privacy-filter>

## 1. Goal

Bring OpenAI's `privacy-filter` token-classification model to MLX-Node so that JavaScript/TypeScript users on Apple Silicon can run on-device PII detection and redaction. Ship:

1. A `@mlx-node/privacy` library exposing `PrivacyFilter.load()` with both raw (`classify`) and high-level (`redact`) APIs.
2. A new `mlx redact <file>` CLI subcommand.
3. A shared MoE module under `crates/mlx-core/src/moe/` consumed by both privacy-filter and the existing Qwen3.5 MoE.
4. A custom Metal kernel for bidirectional banded attention with attention sinks.

## 2. Model facts (ground truth from downloaded weights)

Source: `.cache/models/privacy-filter/config.json` and `model.safetensors` (140 tensors, bf16).

| Field                                              | Value                                              |
| -------------------------------------------------- | -------------------------------------------------- |
| `model_type`                                       | `openai_privacy_filter`                            |
| Architecture string                                | `OpenAIPrivacyFilterForTokenClassification`        |
| `hidden_size`                                      | 640                                                |
| `head_dim`                                         | 64                                                 |
| `num_attention_heads`                              | 14 (Q heads)                                       |
| `num_key_value_heads`                              | 2 (KV heads, GQA group=7)                          |
| `num_hidden_layers`                                | 8                                                  |
| `num_local_experts`                                | 128                                                |
| `num_experts_per_tok`                              | 4                                                  |
| `intermediate_size`                                | 640 (per-expert FFN)                               |
| `hidden_act`                                       | `silu` (→ SwiGLU)                                  |
| `attention_bias`                                   | `true` (Q/K/V/O all have biases)                   |
| `sliding_window`                                   | 128 (band size; ±128 → 257-token effective window) |
| `rope_parameters.rope_type`                        | `yarn`                                             |
| `rope_parameters.factor`                           | 32.0                                               |
| `rope_parameters.original_max_position_embeddings` | 4096                                               |
| `rope_parameters.rope_theta`                       | 150000.0                                           |
| `rope_parameters.beta_fast` / `beta_slow`          | 32.0 / 1.0                                         |
| `max_position_embeddings`                          | 131072                                             |
| `vocab_size`                                       | 200064 (o200k-harmony / gpt-oss tokenizer)         |
| `tie_word_embeddings`                              | `false`                                            |
| Number of output classes                           | 33 (1 background + 8 labels × 4 BIOES tags)        |
| Weight dtype                                       | bf16 (sinks are f32)                               |

**Per-layer weight shapes** (verified):

```
self_attn.q_proj.weight   (896, 640)   self_attn.q_proj.bias   (896,)
self_attn.k_proj.weight   (128, 640)   self_attn.k_proj.bias   (128,)
self_attn.v_proj.weight   (128, 640)   self_attn.v_proj.bias   (128,)
self_attn.o_proj.weight   (640, 896)   self_attn.o_proj.bias   (640,)
self_attn.sinks           (14,) f32
input_layernorm.weight           (640,)
post_attention_layernorm.weight  (640,)
mlp.router.weight                (128, 640)    mlp.router.bias        (128,)
mlp.experts.gate_up_proj         (128, 640, 1280)   gate_up_proj_bias (128, 1280)
mlp.experts.down_proj            (128, 640, 640)    down_proj_bias    (128, 640)
```

Top-level: `model.embed_tokens.weight (200064, 640)`, `model.norm.weight (640,)`, `score.weight (33, 640)`, `score.bias (33,)`.

**Architectural notes that diverge from anything else in the repo:**

- **Attention sinks** — per-head learned scalar (`sinks[h]`) appended to the softmax denominator only. No new K/V row; the sink contributes to normalization but its V-projection is zero. This is gpt-oss-specific.
- **Bidirectional banded attention** — query _i_ attends to keys in `[max(0, i-128), min(T, i+128)]`. Not causal. Long context (128K) works natively since per-token attention is O(band) = O(257).
- **Fused `gate_up_proj`** — gate and up are a single `(E, in, 2·hidden)` tensor; split to two halves after matmul. Different layout from Qwen3.5 MoE.
- **No KV cache** — single forward pass per input. Bypasses all paged-attention machinery.
- **YaRN RoPE** — `factor=32, original_max=4096, theta=150000`. Same family as Qwen-style YaRN but parameters differ; needs an mlx-core RoPE path that already supports YaRN (Qwen3.5 uses YaRN; can reuse with new params).

## 3. Architecture & data flow

### 3.1 Forward pass

```
input_ids [B, T]
   │
   ├── embed_tokens                         → hidden [B, T, 640]
   │
   ├── for each of 8 layers:
   │     ├── input_layernorm (RMSNorm, eps=1e-5)
   │     ├── Banded attention with sinks:
   │     │     Q = (h @ q_proj.W^T) + q_proj.b   → [B, 14, T, 64]
   │     │     K = (h @ k_proj.W^T) + k_proj.b   → [B,  2, T, 64]
   │     │     V = (h @ v_proj.W^T) + v_proj.b   → [B,  2, T, 64]
   │     │     (Q, K) = yarn_rope(Q, K, factor=32, beta=(1,32),
   │     │                        orig_max=4096, theta=150000)
   │     │     # bidirectional banded mask: position i sees [i-128, i+128]
   │     │     # softmax denominator includes exp(sinks[h])
   │     │     attn = banded_softmax_with_sinks(Q K^T / √d, band=128, sinks)
   │     │     out  = (attn @ V_broadcast) → o_proj                 → [B, T, 640]
   │     ├── residual add
   │     ├── post_attention_layernorm
   │     ├── Sparse MoE (fused gate-up layout):
   │     │     logits = h @ router.W^T + router.b                   → [B, T, 128]
   │     │     (w, idx) = top4_softmax(logits)         # softmax-then-top4 (verify
   │     │                                              #  against HF impl)
   │     │     for each chosen expert e in idx:
   │     │         gu = (h @ gate_up_proj[e].T) + gate_up_proj_bias[e]   → [..., 1280]
   │     │         gate, up = gu[..., :640], gu[..., 640:]
   │     │         h_e = silu(gate) * up
   │     │         o_e = (h_e @ down_proj[e].T) + down_proj_bias[e]      → [..., 640]
   │     │     out = Σ_e w_e * o_e                                       → [B, T, 640]
   │     └── residual add
   │
   ├── model.norm (RMSNorm)
   ├── score: Linear(640 → 33, bias=true)             → logits [B, T, 33]
   │
   └── Viterbi(transitions + calibration) → tag_ids [B, T]
       → span extraction → entities[]
```

### 3.2 Banded attention with sinks (kernel)

**Semantics:** for query token _q_, key index _k_:

```
mask(q, k) = 0       if |q - k| ≤ band     (band = 128)
           = -inf    otherwise

w_kv      = softmax_k( q·k_T / √d, masked )
w_sink    = exp(sink_h) / Σ
attn_qk   = w_kv (sums to 1 - w_sink)
out_q     = Σ_k w_kv(q,k) · V_k                       # sink contributes 0
```

**Kernel:** `crates/mlx-paged-attn/metal/banded_attention.metal`. Signature:

```c
banded_attention_with_sinks(
    Q[B, H, T, D],
    K[B, Hkv, T, D],
    V[B, Hkv, T, D],
    sinks[H],            // f32
    band: i32
) -> O[B, H, T, D]
```

Each threadgroup handles one `(batch, head, query_block)`. For each query _q_, iterate `k ∈ [q-band, q+band] ∩ [0, T)`. GQA: `kv_head = h / group` (group=7). Two-pass softmax inside the kernel (max-subtract → exp+sum → normalize).

**CPU fallback (correctness oracle):** Build a `[T, T]` band mask plus an additional sink column with logit `sinks[h]`; call existing `scaled_dot_product_attention` and add the sink contribution analytically. Gated by env var `MLX_BANDED_ATTN_FALLBACK=1` for tests.

### 3.3 Shared MoE module

New crate-internal module: `crates/mlx-core/src/moe/`.

```rust
// crates/mlx-core/src/moe/mod.rs
pub mod router;
pub mod dispatch;

pub use router::{TopKRouter, RouterConfig};
pub use dispatch::{dispatch_tokens, scatter_aggregate};
```

`TopKRouter` owns `(weight, bias, top_k)` and exposes `route(h) -> (weights, indices)`. `dispatch_tokens` groups input tokens by their chosen experts (returns gather indices per expert). `scatter_aggregate` combines per-expert outputs back into `[B, T, hidden]` weighted by router scores.

**Expert MLP stays model-specific** — Qwen3.5 keeps `Qwen35SwitchExpert` (separate gate/up, optional shared expert, MXFP8 via `gather_qmm`). Privacy-filter implements `GptOssFusedExpert` (fused gate_up, bf16, with biases).

**Qwen3.5 migration (parity-gated):** Qwen3.5 MoE's existing `moe.rs` is rewritten to consume `moe::TopKRouter` + dispatch helpers; its expert MLP code stays put. Acceptance gate:

1. Existing Qwen3.5 MoE tests pass byte-for-byte.
2. `cargo test -p mlx-core` shows no new failures.
3. End-to-end tokens/sec benchmark on `Qwen3.5-MoE` is within ±1% of pre-refactor baseline (recorded in spec PR).

If the parity gate fails, the refactor PR is held; privacy-filter ships with `moe::TopKRouter` consumed only by it, and a separate follow-up PR retries the Qwen3.5 migration.

### 3.4 Viterbi BIOES decoder

Pure Rust, no MLX. Lives at `crates/mlx-core/src/models/privacy_filter/viterbi.rs`.

**Transition matrix `T[33, 33]` built at load time:**

- BIOES legality: `O → {O, B-*, S-*}`, `B-X → {I-X, E-X}`, `I-X → {I-X, E-X}`, `E-X → {O, B-*, S-*}`, `S-X → {O, B-*, S-*}`. All other entries = `-inf`.
- Calibration biases applied to allowed transitions:

| Bias key                              | Applies to                      |
| ------------------------------------- | ------------------------------- |
| `transition_bias_background_stay`     | `O → O`                         |
| `transition_bias_background_to_start` | `O → {B-*, S-*}`                |
| `transition_bias_end_to_background`   | `{E-*, S-*} → O`                |
| `transition_bias_end_to_start`        | `{E-*, S-*} → {B-*, S-*}`       |
| `transition_bias_inside_to_continue`  | `{B-*, I-*} → I-*` (same class) |
| `transition_bias_inside_to_end`       | `{B-*, I-*} → E-*` (same class) |

Default values come from `viterbi_calibration.json` (`operating_points.default.biases`). Per-call override exposed via `pf.classify(text, { calibration: {...} })` and `pf.redact(text, { calibration: {...} })`.

**Viterbi step:** standard `forward[t, j] = max_i (forward[t-1, i] + T[i, j]) + emit[t, j]`. Backtrack to get best path. O(T·33²); cheap.

After tagging, span extraction:

- A span starts at `B-X` or `S-X` and ends at `E-X` (B…E) or is a singleton (`S-X`).
- Span score = mean of per-token logit-softmax probabilities over the span.
- Output: `{ start, end, label, score, text }` where `start`/`end` are **character** offsets (computed from tokenizer offset mapping).

### 3.5 Tokenizer

The model ships `tokenizer.json` (HF tokenizers format) with the gpt-oss `o200k_harmony` BPE. The existing `Qwen3Tokenizer::from_file` is a thin wrapper around `tokenizers-rs` — load via that path. We need character-offset spans, so use `Encoding::offsets()` directly (not the chat-template path).

**Risk:** the existing wrapper may not expose offsets through NAPI. If so, add a `tokenize_with_offsets(text) -> { ids: Uint32Array, offsets: Array<[number, number]> }` method to the tokenizer NAPI surface. Spec item, not a blocker.

## 4. Public API

### 4.1 TS package (`packages/privacy/`)

```typescript
import { PrivacyFilter } from '@mlx-node/privacy';

const pf = await PrivacyFilter.load('./privacy-filter');

// Raw classification
const result = await pf.classify('Hi, I'm Alice <alice@x.com>.', {
  threshold: 0.5,              // optional, default 0.5
  calibration: { /* ... */ },  // optional override of viterbi biases
});
// result: {
//   entities: [
//     { start: 8,  end: 13, label: 'private_person', score: 0.999, text: 'Alice' },
//     { start: 15, end: 26, label: 'private_email',  score: 0.999, text: 'alice@x.com' },
//   ],
//   tokens: [ { id, text, offsetStart, offsetEnd, tag } ],   // optional, off by default
// }

// High-level redaction
const { redacted, entities } = await pf.redact(text, {
  replacement: '[redacted]'                        // string, OR
            | 'label'                              // → '[private_email]'
            | ((e: Entity) => string),             // custom function
  labels: ['private_email', 'private_phone'],      // optional allow-list filter
  threshold: 0.7,
  calibration: { /* ... */ },
});
```

**Types** (`packages/privacy/src/types.ts`):

```typescript
export type PrivacyLabel =
  | 'account_number'
  | 'private_address'
  | 'private_date'
  | 'private_email'
  | 'private_person'
  | 'private_phone'
  | 'private_url'
  | 'secret';

export interface Entity {
  start: number; // inclusive char offset
  end: number; // exclusive char offset
  label: PrivacyLabel;
  score: number;
  text: string;
}

export interface ClassifyOptions {
  threshold?: number;
  calibration?: Partial<ViterbiCalibration>;
  returnTokens?: boolean;
}

export interface RedactOptions extends ClassifyOptions {
  replacement?: string | 'label' | ((entity: Entity) => string);
  labels?: PrivacyLabel[];
}

export interface ViterbiCalibration {
  transition_bias_background_stay: number;
  transition_bias_background_to_start: number;
  transition_bias_end_to_background: number;
  transition_bias_end_to_start: number;
  transition_bias_inside_to_continue: number;
  transition_bias_inside_to_end: number;
}
```

### 4.2 NAPI surface (`crates/mlx-core` exports)

```rust
#[napi(js_name = "PrivacyFilterModel")]
pub struct PrivacyFilterModel { /* ... */ }

#[napi]
impl PrivacyFilterModel {
    #[napi] pub async fn load(model_path: String) -> Result<Self>;
    #[napi] pub async fn classify(
        &self,
        text: String,
        opts: Option<ClassifyOpts>,
    ) -> Result<ClassifyResult>;
}
```

The TS-side `redact()` is pure JS on top of `classify()`.

### 4.3 Model loader registration

**Decision: do not extend `packages/lm/src/models/model-loader.ts`.** The existing loader returns shapes (`TrainableModel | LoadableModel`) built around generation — privacy-filter is a token classifier and doesn't fit. `PrivacyFilter.load(path)` is the only entry point. The CLI command knows it's loading a privacy filter explicitly; no auto-dispatch is needed.

Alternative considered: a generic `loadClassifier(path)` polymorphic entry point. Rejected — only one classifier exists; YAGNI until a second token classifier lands.

### 4.4 CLI: `mlx redact`

New file: `packages/cli/src/commands/redact.ts`.

```bash
mlx redact --model ~/.mlx-node/models/privacy-filter \
           --input ./email.txt \
           [--output ./email.redacted.txt]   \
           [--replacement '[REDACTED]' | --replacement label] \
           [--labels private_email,private_phone] \
           [--threshold 0.7] \
           [--json]            # also emit entities[] as JSON to stdout/sidecar
```

Reads stdin if `--input` is omitted. Writes redacted text to `--output` or stdout. With `--json`, emits the entity list to a sidecar (`<output>.entities.json`) or to stdout in `--json --output -` mode.

## 5. Module layout

```
crates/mlx-core/src/
├── moe/                                  # NEW
│   ├── mod.rs
│   ├── router.rs                         # TopKRouter
│   └── dispatch.rs                       # token gather/scatter
└── models/
    ├── qwen3_5/
    │   └── moe.rs                        # MIGRATED to consume moe::*
    └── privacy_filter/                   # NEW
        ├── mod.rs                        # NAPI #[napi] PrivacyFilterModel
        ├── config.rs                     # parse config.json
        ├── persistence.rs                # safetensors load + tokenizer + viterbi calib
        ├── attention.rs                  # banded attn with sinks (calls kernel)
        ├── experts.rs                    # GptOssFusedExpert
        ├── transformer.rs                # block (norm → attn → norm → moe)
        ├── classifier.rs                 # score head
        ├── viterbi.rs                    # BIOES decoder
        ├── spans.rs                      # span extraction + scoring
        └── forward.rs                    # full forward orchestration

crates/mlx-paged-attn/
└── metal/
    └── banded_attention.metal            # NEW kernel
    └── ... binding & rust wrapper in src/

packages/privacy/                         # NEW @mlx-node/privacy
├── src/
│   ├── index.ts
│   ├── classifier.ts
│   ├── redactor.ts
│   └── types.ts
├── __test__/
│   ├── classify.test.ts
│   ├── redact.test.ts
│   └── fixtures/
├── package.json
└── tsconfig.json

packages/cli/src/commands/redact.ts       # NEW
```

## 6. Testing strategy

**Rust:**

- `cargo test -p mlx-core moe::router` — router top-k correctness + dispatch parity vs. naive implementation.
- `cargo test -p mlx-core moe::dispatch` — scatter-aggregate matches dense reference.
- `cargo test -p mlx-core models::privacy_filter::viterbi` — BIOES decoder: legality enforced, calibration biases applied, known traces match expected paths.
- `cargo test -p mlx-paged-attn banded_attention` — kernel output vs. CPU fallback on random inputs across shapes (T ∈ {64, 257, 1024, 8192}, several head configs).
- `cargo test -p mlx-core models::qwen3_5::moe` — **all pre-existing tests pass byte-identical** post-refactor.

**TS:**

- `packages/privacy/__test__/classify.test.ts` — end-to-end on small fixtures: known sentences with hand-labeled spans, assert ≥X recall (X TBD after first benchmark, will set the threshold in the implementation plan).
- `packages/privacy/__test__/redact.test.ts` — replacement strategies, label filter, calibration override.
- `packages/privacy/__test__/parity.test.ts` (slow, gated by env): compare a handful of outputs against the HuggingFace `pipeline('token-classification', 'openai/privacy-filter')` reference saved as a JSON fixture. Drives confidence that our forward pass is correct.

**CLI:** smoke test invoking `mlx redact` against a fixture file.

## 7. Performance targets

This is a 50M-active-param model with banded attention; on M3 Max it should be dramatically faster than the 47–52 tok/s Qwen3.5 MoE baseline. Concrete targets to record in the implementation plan after first working version:

- Single-pass classification on a 512-token input: < 100 ms (target; calibrate against Python reference).
- 8K-token input: < 800 ms.
- Memory: O(T · band) attention, peak < 4 GB for 128K input.

These are starting hypotheses, not gates. The implementation plan will pin real numbers after the first pure-Rust forward pass works.

## 8. Risks & open questions

1. **Tokenizer offsets** — `Qwen3Tokenizer` wrapper may need a new method exposing `Encoding::offsets()`. Verify early in implementation.
2. **YaRN params** — existing YaRN code uses Qwen3.5 params; need to plumb config-driven `beta_fast/slow/factor/original_max/theta`. Check whether existing `fast::rope` supports this or needs extension.
3. **Router softmax order** — gpt-oss applies softmax **before** top-k (renormalize after); some implementations apply softmax only over top-k. The HF reference is the ground truth; verify when reading the modeling code or by output diff against `transformers`.
4. **Attention sinks numerics** — sinks are f32 but Q/K/V are bf16; need to compute softmax in f32 to avoid sink underflow. Pattern matches existing attention paths.
5. **Qwen3.5 parity gate** — if shared-MoE refactor regresses Qwen3.5, ship privacy-filter without the migration and follow up. Plan must include a clear rollback path.
6. **Long-context memory** — 128K input × 640 hidden × bf16 × multiple layers' KV ≈ tens of GB. Banded attention means we don't store all-to-all QK, but we still hold full Q/K/V per layer. Spec needs an explicit "streaming chunks" path for very long inputs in a future PR; for v1, document the practical input ceiling.
7. **`output_router_logits: false`** in config — confirms training-time auxiliary loss is off at inference; no implications for us beyond skipping any "return router logits" path.

## 9. Non-goals (v1)

- Training / fine-tuning support. Inference only.
- Quantization. Ship bf16; quantization is a follow-up.
- Multi-process / GPU sharding. Single-process, single-device.
- Streaming output per token (it's a classifier, not a generator — return all entities at once).
- Generic "TokenClassifier" abstraction. Privacy-filter ships its own surface; abstraction comes when a second token classifier arrives.
- Compiled C++ forward for privacy-filter in v1. The user asked for "custom Metal kernels too," which is satisfied by the banded-attention kernel. A `mlx_privacy_filter.cpp` compiled graph is a perf follow-up.

## 10. Acceptance criteria

- [ ] `await PrivacyFilter.load('.cache/models/privacy-filter')` succeeds.
- [ ] `classify("Hi, I'm Harry Potter and harry@hogwarts.edu")` returns `private_person` + `private_email` spans matching the HF reference output character-for-character.
- [ ] `redact()` with `replacement: 'label'` produces `"Hi, I'm [private_person] and [private_email]"`.
- [ ] `mlx redact --input fixture.txt` writes the expected redacted file + entities JSON.
- [ ] Qwen3.5 MoE: all pre-existing tests pass byte-identical; tok/s within ±1% of pre-refactor baseline (recorded in PR).
- [ ] `cargo test -p mlx-core`, `cargo test -p mlx-paged-attn`, `yarn vite run test` all green.
- [ ] `cargo clippy --all -- -D warnings` and `cargo fmt --check` clean.
- [ ] `yarn vite fmt` and `yarn vite lint --type-aware --type-check` clean.
- [ ] Implementation plan recorded tokens/sec on M3 Max for a 512-token and 8K-token input.
