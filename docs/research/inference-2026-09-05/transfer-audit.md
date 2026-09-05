# Inference control-flow and transfer audit

Reviewed 2026-09-06, after structural revision `2b6cb786`. Scope: native request
entry, all seven chat-family runners, flat and paged prefill/decode, MTP and
DFlash/DSpark, batching, media inputs, output extraction and SSD cache lifecycle.
The scan also covered OCR/ASR inference entry points and the C++ bridge. Conversion,
training, diagnostics and test-only reads were classified separately from serving.
This is a source/call-path audit with targeted runtime checks, not a claim that
an Instruments trace covered every model, Metal device or fallback.

The audit found avoidable transfers in normal execution, not just theoretical
API opportunities. Five changes are implemented:

| Finding                                                 | Previous work                                                                                                          | Change                                                                                                                                                   |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3 chunked/cache-hit prefill                         | Read full context K/V from private pool into shared staging, CPU unpack, then copy into MLX arrays for every layer     | Use the existing graph-native KV gather, retaining pending writes and the same BF16/FP16 layout and causal mask                                          |
| Generic f32/i32/u32 export                              | Add a full zero tensor and evaluate before flatten/cast/read, even for completed contiguous arrays                     | Evaluate the actual array; materialize contiguous storage only when needed; copy requested output once                                                   |
| Gemma DSpark/assistant and Muse DFlash sampled drafting | Export a full f32 vocabulary row for every inverse-CDF draw while retaining the original distribution for verification | Scan the completed shared allocation directly; preserve sequential f64 sums, one Rust RNG draw, invalid-mass behavior and retained verifier distribution |
| Paged KV-write bounds validation                        | For every multi-token layer write, launch and wait for a max reduction of host-created slot metadata                   | Scan available metadata directly; retain the lazy-array reduction fallback and authoritative runtime guard                                               |
| PaddleOCR batched decode                                | Read sampled IDs, build a CPU vector, create a new MLX input, then read the same IDs again for history/EOS             | Reshape sampled IDs for embeddings; read once for output, penalties and EOS                                                                              |

Implementation: [Qwen block](../../../crates/mlx-core/src/transformer/block.rs),
[array export bridge](../../../crates/mlx-sys/src/mlx_common.h),
[shared draft sampler](../../../crates/mlx-core/src/sampling/dense_draw.rs),
[native probability reads](../../../crates/mlx-sys/src/mlx_nn_ops.cpp),
[KV factory](../../../crates/mlx-sys/src/mlx_paged_ops.cpp),
[PaddleOCR runner](../../../crates/mlx-core/src/models/paddleocr_vl/model.rs).

## What constitutes a copy here

Apple Silicon and MLX share physical memory between CPU and GPU. This permits a
CPU scan of a completed MLX array without a transfer to a separate device address
space. It does not make a CPU vector export, a private-buffer staging blit, or a
layout conversion free. Completion still precedes host reads. See MLX's
[unified-memory model](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
and [contiguous operation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.contiguous.html).

The pinned MLX implementation supplies the exact local semantics:

- `MxArray::clone` increments the handle's `Arc`. MLX `copy` shares storage after
  evaluation; it is not a deep copy. Slice/reshape may share storage, depending on
  strides. `deep_copy` explicitly materializes independent storage to avoid
  retaining a large parent allocation.
- Host-pointer array constructors copy into MLX-owned storage. The BF16/F16
  constructors avoid dtype expansion but are not zero-copy loaders.
- `item_at_*` waits through `ensure_readable` if necessary, then reads one element
  and casts on CPU. It does not launch a GPU cast or copy a vocabulary tensor.
- MLX `Contiguous` can copy a small view solely to release a large parent. Export
  now checks row-contiguity first to avoid that unrelated retention heuristic.
- `eval` and stream synchronization are completion boundaries, not data copies.
  Removing them without replacing dependency/lifetime guarantees is incorrect.

These claims were checked against vendored MLX `6d45ab90`, including
`mlx/array.h`, `backend/metal/allocator.cpp`, `backend/gpu/primitives.cpp` and
`array.cpp`, plus our array/FFI implementation.

## Phase-by-phase findings

| Phase                         | Data crossing the boundary                                                | Assessment                                                                                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JS request and model thread   | Text, options, owner IDs and typed command handles                        | Queueing does not clone model weights or KV tensors. Tokenization is CPU work. Initial token/media arrays require owned input storage.                                                                                                 |
| Flat prefill                  | Prompt IDs and media tensors enter MLX; K/V remain arrays                 | Chunk evaluation limits graph/memory growth. Normal transformer/recurrent arithmetic stays in the graph. No full logits or hidden-state export is required.                                                                            |
| Paged prefill                 | CPU block tables, slot IDs and lengths become small MLX metadata arrays   | Metadata is shared/cached across layers where the adapter supports it. The six-field fixed-shape `build_paged_attention_inputs` helper currently has test callers, so its padded uploads are not a serving bottleneck.                 |
| Cache-hit prefill             | Paged pool to flat SDPA layout or direct varlen PagedAttention            | Qwen3's normal host route is removed. Dense/MoE Qwen3.5, LFM2, Nemotron, Gemma and Muse use graph gathers/varlen routing; some retain explicit host fallbacks after graph-construction failure.                                        |
| Paged decode                  | GPU Q/K/V and private resident pool; compact per-request metadata         | Native MLX primitives and retained Metal-buffer views avoid attention-output host round trips. Legacy `PagedAttentionOutput::to_mlx_array` has no production caller; the adapter uses `to_mlx_array_view`.                             |
| Standard AR output            | One token per request, plus requested log probabilities                   | Required for detokenization, stopping, penalties and protocol guards. The device token remains usable for the next graph. Scheduled batches upload compact current-token/row metadata; no full-vocabulary export in ordinary sampling. |
| MTP verification              | Small proposal IDs, acceptance scalar probabilities/argmax IDs            | Target/draft distributions, recurrent tapes and KV stay as arrays. Token/history reads support transaction decisions. Rejected writes do not become committed history.                                                                 |
| DSpark/DFlash                 | Proposal IDs, dense or compact sparse distributions                       | Dense draws now avoid vocabulary copies on macOS. CPU f64 scanning and GPU completion remain; this is not GPU sampling. DFlash2 exports top-k candidate/edge tables for its CPU path selector, not full target logits.                 |
| Scheduler and owner switching | CPU admission/frontier metadata; array handles and necessary row gathers  | No weight copies per request. Batch compaction and recurrent gather/scatter are device work; copies preserving independent owner state are intentional. Speculation remains an ordered barrier.                                        |
| SSD capture                   | Completed private KV blocks → shared staging → owned bytes → writer       | Real bounded copies, required by the current private-pool/owned-writer contract. One submission reads all layers of a block. Recurrent/sliding sidecars carry exact boundary state.                                                    |
| SSD restore                   | Validated bytes → shared staging → reserved private pool slots            | Publish only after upload completion and validation. CPU I/O can proceed asynchronously; GPU visibility and reservation release remain explicit. No permanent second RAM cache tier is added.                                          |
| OCR/ASR and media             | Input image/audio arrays; final tokens, boxes, masks or feature summaries | Host postprocessing has legitimate output reads. PaddleOCR's duplicate decode-ID trip is removed. Dynamic vision shape/mask counts still require scalar decisions.                                                                     |

Relevant code boundaries:
[decode](../../../crates/mlx-core/src/engine/decode.rs),
[scheduler](../../../crates/mlx-core/src/engine/hybrid_scheduler.rs),
[MTP](../../../crates/mlx-core/src/engine/mtp_turn.rs),
[DSpark](../../../crates/mlx-core/src/engine/dspark_turn.rs),
[paged adapter](../../../crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs),
[pool staging](../../../crates/mlx-paged-attn/src/layer_kv_pool.rs),
[cold cache](../../../crates/mlx-paged-attn/src/cold_cache.rs).

## Costs deliberately retained

Host fallback attention remains expensive when native graph construction fails.
It is a recovery route with diagnostics, not evidence that the common path is
copy-free under every configuration. The Qwen3 replacement propagates gather
errors instead of silently returning to a host round trip.

SSD snapshots still use owned staging data. Gemma grouped sliding capture also
retains bounded array snapshots before sidecar serialization. Replacing these
representations with leased buffers or Metal IO could reduce staging copies, but
must preserve immutable capture bytes, checksum/identity validation, committed
frontiers and upload publication. No SSD throughput improvement is claimed from
this change. The existing bounded capture/restore protocol was preserved and
retested.

The old uncached fused transformer helper contains SDPA completion points added
for timeout prevention. Cached chat uses the separate cache-aware path. Those
waits and diagnostic top-2/logit reads were not blindly removed. Likewise, the
new shared sampler still does O(vocabulary) CPU work; a GPU inverse-CDF redesign
would need its own numerical/RNG validation because Metal does not provide the
same sequential f64 accumulation used here.

## Validation

Results and reproducible measurements are recorded in [validation.md](validation.md).
The prefix workload removed 875 MiB of host KV staging across five traced turns;
its paired median continuation latency fell from 154.738 to 56.300 ms.
New checks cover broadcast/strided exports, exact integer values, signed zero,
asynchronous completion, invalid draft mass without RNG consumption, and seeded
draft draw parity. The slot guard checks valid/sentinel/overflow cases with both
available multi-token metadata and lazy inputs. Existing chunk/prefix parity,
owner isolation, speculative frontier and SSD failure/publication tests remain
part of validation.
