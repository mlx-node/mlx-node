# Inference architecture

This document describes the model-extension boundary for chat inference. The
design keeps request orchestration model-neutral while leaving tensor programs,
hybrid cache state, and model-specific prompt/media preparation with the model
that owns them.

## Control flow

```text
config.json
  -> TypeScript MODEL_FAMILY_REGISTRY
  -> native model instance
  -> immutable ExecutionPlan

messages / delta + ChatConfig
  -> extract media and render prompt
  -> resolve one TurnPlan
  -> exactly one whole-turn executor
       multimodal | paged | speculative | generic AR
  -> shared decode / stream / finalize machinery
```

The two planning stages have different responsibilities:

- `packages/lm/src/models/model-loader.ts` is the ordered registry for model
  detection, aliases, loader selection, model kind, and family-specific load
  options. Architecture probes and the legacy missing-`model_type` Qwen3
  fallback are explicit registry policies. The unified-Gemma architecture is
  authoritative even when `model_type` is explicit or malformed, matching the
  native loader; without a recognized architecture probe, an explicit unknown
  type fails closed. `ModelType`, `LoadableModel`, and `TrainableModel` are
  derived from the registry rather than maintained as separate family lists.
- `crates/mlx-core/src/engine/plan.rs` resolves the capabilities of the loaded
  model against one request. `ExecutionPlan` is immutable load-time data;
  `TurnPlan` is compact request-time data and is computed before cache mutation.

The decode loop does not probe model capabilities. The session selects exactly
one executor, and a declared feature without an implementation fails loudly
instead of falling through to a different route.

## Independent execution dimensions

Paged attention, media, and speculative decoding are dimensions, not mutually
exclusive model modes:

| Dimension       | Plan data                                                                                             | Owner after dispatch                                    |
| --------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Media           | truthful availability, validation routing, current input, live-session context, and raw `MediaInputs` | model-specific preparation and embedding merge          |
| Attention cache | paged availability and delta admission                                                                | `PagedBackend` plus the model's non-paged hybrid state  |
| Decoder         | autoregressive, native MTP, or external draft                                                         | `MtpBackend`, `DsparkBackend`, or the target AR stepper |

This matters for hybrid models. A paged plan describes only the attention
layers represented by the adapter. Convolutional, recurrent, sliding-window,
cross-attention, and other state remains model-owned. Do not add a global
`CacheMode` enum that implies all layer caches have the same topology.

`MediaPlan` distinguishes media that this loaded instance can execute from
`backend_validated` input. The latter is admitted only so an existing family
handler can produce a precise compatibility error; it is never reported as a
working encoder or used to admit speculation.

Speculation decorates target execution. Each speculative implementation
declares separately which current-turn media it can consume, which media may
already be represented in the live session, and whether it can operate on
paged target state. This matters for a text delta over image-derived KV: an
empty current input is not a text-only context. An unsupported combination
keeps the exact target-model path and disables speculation for that turn; it
never discards media or the request.

## Current routing contracts

| Family        | Media                                                        | Paged attention | Chat speculation                                                                  | Important constraint                                                    |
| ------------- | ------------------------------------------------------------ | --------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Qwen3         | none                                                         | fresh and delta | none                                                                              | all paged turns use the shared paged executor                           |
| LFM2          | none                                                         | fresh and delta | none                                                                              | short-conv state is never paged: every paged turn (fresh or delta) rebuilds it from the full token stream via conv Pass-1 |
| Qwen3.5 dense | images when encoder, processor, and paged adapter are loaded | fresh and delta | native MTP, including paged target state and supported image-context continuation | multimedia turns keep the model's VLM preparation path                  |
| Qwen3.5 MoE   | images when encoder, processor, and paged adapter are loaded | fresh and delta | native MTP on flat target state                                                   | paged target execution takes precedence and falls back to target AR     |
| Gemma4        | image/audio components that have a paged adapter             | fresh and delta | external draft on flat text-only state                                            | missing components retain family-specific validation errors             |

The table is a conformance description, not dispatch code. The source of truth
is each model's `execution_plan()` plus its executor implementations.

## Extension boundary

Adding a conventional chat model should require these scoped pieces:

1. Add one descriptor to `MODEL_FAMILY_REGISTRY`: canonical type, exact raw
   aliases or a narrow architecture probe, kind, loader, native trainer-class
   metadata when trainable, and supported load options.
2. Implement `ChatBackend` for prompt rendering, cache lifecycle, target
   forward/prefill, and model-specific defaults.
3. Return an `ExecutionPlan` built only from immutable load-time state.
4. Opt into the narrow traits that apply:
   - `PagedBackend` for shared block-paged attention lifecycle;
   - `MtpBackend` for native MTP proposal/verification;
   - `DsparkBackend` for the current external-draft implementation;
   - a multimodal executor for processor/encoder output and embedding merge.
5. Add planner conformance tests for every advertised combination, then real
   model parity tests for flat/paged, sync/stream, fresh/delta, and enabled/
   disabled speculation as applicable.

### Model optimizations

An optimization belongs at the narrowest layer that understands its invariant:

- request-level compatibility (paged attention, MTP/draft, media) belongs in
  `ExecutionPlan` and `TurnPlan`;
- attention KV allocation/lifecycle belongs in `PagedBackend` and
  `mlx-paged-attn`;
- speculative proposal, verification, rollback, and commit belong in the MTP
  or draft backend;
- checkpoint storage dispatch belongs in the family-neutral
  `models/quant_dispatch.rs` helpers;
- tensor graph and compiled-kernel choices remain inside the model forward
  implementation, where layer shape and cache topology are known.

Do not add a generic optimization flag that the engine cannot validate. Add a
plan field only when it changes request admission or engine orchestration;
otherwise expose the optimization behind the relevant narrow backend.

One-shot OCR pipelines and embedding-only models are intentionally not forced
through `ChatBackend`. They use the same registry for loading but retain APIs
that match their lifecycle.

## Cache and speculative safety

Every cache topology must define reset, prefix verification, prefill, decode,
and save semantics. Speculative implementations additionally need an explicit
commit boundary: accepted tokens advance durable state; rejected proposals do
not. Today those semantics are implemented by the existing MTP/draft backends
and family cache code. A future generalized `SequenceCache` transaction should
be introduced only when it can represent recurrent and convolutional snapshots
as well as attention KV blocks.

Do not infer cache compatibility from a model name or from the presence of an
adapter alone. Admission belongs in `ExecutionPlan`; state transitions belong
in the backend that owns the state.

## Deliberate migration boundary

Qianfan-OCR still owns specialized session/decode loops. Its prompt rendering
depends on image tiling and preprocessing, and its stream/default semantics do
not yet have characterization parity with the chat engine. Migrate it only
after introducing a prepared-multimodal-input seam and tests that lock those
behaviors. PaddleOCR one-shot stages and Harrier embeddings should remain
outside the chat-turn executor.

## Reference research snapshot

The architecture was checked against freshly fetched remote branches on
2026-07-10:

| Project    | Remote revision                            | Borrowed principle                                                                                        |
| ---------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| vLLM       | `88e5e2c57be8ce6e25510c1249352a23b8a85ec4` | compositional multimodal processing, heterogeneous KV specifications, speculation around target execution |
| mistral.rs | `b7956205fae9`                             | logical cache manager separated from physical cache engine; speculative reserve/commit/rollback           |
| mlx-vlm    | `05440cc57c62`                             | media features converge at an input-embedding preparation seam                                            |
| mlx-lm     | `a790972f0f84`                             | small module/loader/cache protocols                                                                       |
| mlx-rs     | `f4aa309c79b6`                             | structural module traversal and optimization passes                                                       |
| MTPLX      | `510ac8c9224c`                             | declarative backend admission and recurrent-state snapshots                                               |

The design intentionally does not copy vLLM's CUDA-oriented execution engine or
MTPLX's family-specific conditional dispatch. The reusable part is the
separation of declarations, planning, state ownership, and execution.
