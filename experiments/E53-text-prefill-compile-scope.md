# E53 — Scope: C++ text-prefill compile-cache port

Status: scope-only, no code yet.
Expected gain: 1–3% at 1024-prompt single-chunk (per autoresearch.ideas /
post-arc backlog notes).

## Headline question

Most of the per-experiment landscape has saturated at 1024-prompt
single-chunk:

- E37+E38+E39+E40+E47+E51 composed: **-16.4%** same-binary A/B
- Register-blocking (E47/E48/E50): saturated at 4-vcol
- TG_Y geometry (E45/E49): TG_Y=4 optimal across all variants
- Q+K+V projection stack (E41 revisited as E52-investigate): rejected,
  duplicates the slice→reshape copy hazard

The remaining big lever for prefill is moving the per-chunk forward into
a single FFI call backed by the existing `qwen35_common` helpers, so the
Rust→MLX graph-build per-op is replaced by C++→MLX graph-build inside
one call. The decode path already does this via `qwen35_decode_fn`;
prefill on the text path still uses the Rust `forward_inner` loop.

## What already exists

`crates/mlx-sys/src/mlx_qwen35_common.h` ships:

- `gdn_prefill_fn(x: [B,T,hidden], layer_idx, cfg)` — full-sequence
  GDN forward (in_proj → conv1d → split/reshape → RMSNorm → per-step
  Metal kernel → output projection). Handles `T>1`.
- `attn_prefill_fn(x, layer_idx, position_ids, cfg, mrope_section)` —
  attention forward; takes M-RoPE position IDs.
- All the inner compile-cached primitives (`compiled_silu`,
  `compiled_silu_mul`, `swiglu`, `fused_compute_g`, `gated_delta_kernel_call`,
  `rms_norm_no_weight`, `linear_proj`).

`crates/mlx-sys/src/mlx_qwen35_vlm.cpp` ships `mlx_qwen35_vlm_prefill`:
a complete compiled prefill that wires the helpers above into a single
FFI entry point. It accepts pre-merged `inputs_embeds` and a 3-D
`position_ids` (for M-RoPE) and returns the last token's logits plus
populated `g_vlm_caches`. The caller (`generate_with_vision` in
`crates/mlx-core/src/models/qwen3_5/model.rs:~8175`) hands the caches
off to `mlx_qwen35_compiled_init_from_prefill` so the standard compiled
decode path can pick up from where prefill left off.

## The text variant — delta from VLM

For text-only Qwen3.5 prefill, the deltas from VLM prefill are
small but real:

1. **Embedding lookup inside C++.** The VLM path takes
   `inputs_embeds` because vision tokens are merged in upstream. The
   text path can take `input_ids` and do
   `take(get_weight("embedding"), input_ids, axis=0)` at the entry of
   the forward.
2. **Scalar RoPE offset, not M-RoPE.** `attn_prefill_fn` takes
   `position_ids` + `mrope_section`. We need a parallel
   `attn_prefill_fn_scalar` that takes an int offset and uses
   `fast::rope` with `offset = offset + arange(T)`. Most of the body is
   reusable.
3. **No `rope_deltas`.** Drop the M-RoPE-specific argument; the scalar
   offset already encodes the prefill start position.
4. **Cache transfer for chunked prefill.** For multi-chunk prefill
   (prompts > 1024), each chunk's output state needs to feed the
   next chunk's input state. The decode-path globals
   (`g_compiled_caches`, `g_offset_int`) are the destination; the
   prefill helper can write directly to them (no separate `g_vlm_caches`
   roundtrip) since text prefill doesn't need a sidecar.

## Minimal viable slice for benching (single-chunk only)

For the 1024-prompt bench specifically (single chunk, no caches feeding
back into anything beyond the bench's own next-token sample):

1. New FFI: `mlx_qwen35_text_prefill(input_ids, ..., output_logits)`.
   Same shape as VLM prefill minus M-RoPE / inputs_embeds / mrope
   section.
2. New helper: `attn_prefill_fn_scalar(x, layer_idx, offset, cfg)`.
   Subset of `attn_prefill_fn`; uses scalar RoPE instead of M-RoPE.
3. Layer loop inside `mlx_qwen35_text_prefill` mirrors VLM's loop
   structure (lines 83–127 of `mlx_qwen35_vlm.cpp`).
4. Rust dispatch in `forward_inner` (model.rs:~7345) — if
   `use_compiled` is set AND `cache` is None (or empty/fresh), route
   the prefill through the FFI; otherwise fall through to the existing
   Rust path.
5. Env-toggle `MLX_DISABLE_E53_COMPILED_TEXT_PREFILL=1` reverts to
   the legacy Rust path for same-binary A/B.

### Out of scope for the first cut

- Multi-chunk prefill (chunks beyond PREFILL_STEP_SIZE=1024)
- Cache transfer to the compiled decode path (`mlx_qwen35_compiled_init_from_prefill`)
- Paged-attention variant (the flat-cache version is enough for the bench)
- Quantized variants (the bench uses bf16 dense)

These can land in a follow-up if the first-cut bench shows a win.

## Estimated effort

| Step | Estimate |
|------|----------|
| Write `attn_prefill_fn_scalar` helper | 30 min |
| Write `mlx_qwen35_text_prefill` FFI + globals | 1 h |
| Wire FFI signature into `mlx-sys/src/lib.rs` | 15 min |
| Rust dispatch in `forward_inner` | 45 min |
| Build, parity test, fix issues | 1 h |
| Cold-state A/B (3–5 batches of 4 runs each) | 30 min |
| **Total** | **~4 h** |

Multi-session scope. Recommend splitting into two sessions:

- **Session A**: helpers + FFI + parity test.
- **Session B**: Rust dispatch + bench + iterate on issues.

## Risks

1. **Bit-exact correctness.** The Rust path and C++ path must compute
   the same logits to bf16 precision. The VLM path proved this is
   achievable for `gdn_prefill_fn`; the new piece is
   `attn_prefill_fn_scalar`'s match against `Qwen3_5Attention::forward`.
   Specifically: the Q-gating reshape, partial-RoPE on `rope_dims`,
   Q/K layernorm order — all need to match exactly.
2. **Cache lifecycle.** The Rust dispatch must guarantee the per-layer
   caches are populated equivalently to the legacy path. If we punt
   that for the first cut (return only logits, no cache write-back),
   the bench is still valid for measuring prefill *throughput*, but
   subsequent decode would fail. The MLX_DISABLE_E53… toggle should
   default to legacy until cache lifecycle is solved.
3. **Compile-graph rebuilds.** mlx::core::compile caches by
   (function-pointer, shape-fingerprint). The shapeless inner helpers
   (`compiled_silu`, `compiled_swiglu`, `fused_compute_g`) handle
   varying shapes via shapeless compile, but the per-step Metal kernel
   takes T as a runtime scalar. Cold-start cost is paid once; warm
   runs hit the cache.
4. **Mutex contention.** The compiled-path globals (`g_compile_config`,
   weight table) are guarded by a process-wide RwLock (per `CLAUDE.md`).
   Concurrent inference (e.g. server with multiple sessions) is
   serialized through compiled paths anyway, so this isn't worse than
   the current decode path.

## Decision

Defer implementation to a dedicated session. The bench's 0.3% noise
floor in cold thermal window means the 1–3% expected gain is
detectable, but the implementation effort is multi-hour and warrants
a clean session start with the full code context loaded. Do not
attempt to start the implementation in a tail-end loop iteration.

Recommended next loop action: long-delay wakeup to either continue
autonomous exploration of smaller ideas, or hand off to a focused
session if a human is available.
