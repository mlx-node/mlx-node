# Models

## Language models

All language wrappers share a uniform `ChatSession<M>` surface (`send` / `sendStream` / `sendToolResult` / `reset`) driven by the native `chatSessionStart` / `chatSessionContinue` / `chatSessionContinueTool` NAPI entry points. The legacy `model.chat()` / `model.chatStream()` methods are removed from every generative model.

| Model             | `generate()` | Session API |  Training  | Notes                                                                            |
| ----------------- | :----------: | :---------: | :--------: | -------------------------------------------------------------------------------- |
| **Qwen3**         |     yes      |     yes     | GRPO + SFT | Speculative decoding; paged attention                                            |
| **Qwen3.5 Dense** |     yes      |     yes     | GRPO + SFT | Compiled C++ forward (see [ffi-cpp.md](ffi-cpp.md)); VLM variant                 |
| **Qwen3.5 MoE**   |     yes      |     yes     | GRPO + SFT | Compiled C++ forward with expert routing; VLM variant                            |
| **Gemma4**        |     yes      |     yes     |     —      | Hybrid sliding/global attention + MoE/PLE; DSpark + assistant-MTP spec. decoding |
| **Muse-Glimmer**  |     yes      |     yes     |     —      | Text decoder; Q4_K import; DFlash; hybrid paged AR                              |
| **LFM2.5**        |     yes      |     yes     |     —      | Hybrid conv + attention                                                          |
| **Nemotron 3.5 Lightning** |     yes      |     yes     |     —      | Hybrid Mamba-2 + MoE + attention; native MTP; inference-only                     |

`Qwen3Model | Qwen35Model | Qwen35MoeModel` is the public `TrainableModel` union in `@mlx-node/lm` — Gemma4, Muse-Glimmer, LFM2.5, and Nemotron 3.5 Lightning are inference-only.

**Nemotron 3.5 Lightning reasoning note.** On the NVFP4 checkpoint, greedy (T=0) turns with thinking enabled can loop inside the `<think>` block for hundreds of tokens on simple prompts before closing it (characteristic 4-bit expert noise on this heavy reasoning checkpoint; the same prompts answer crisply once thinking ends). The runtime behaves as designed — an unclosed `<think>` at the token budget is redacted to `text=""` by `finalize.rs`. For deterministic short answers prefer `reasoningEffort: 'none'`; for reasoning output allow a generous `maxNewTokens` or set `thinkingTokenBudget` to force the block closed.

**MTP & flat/paged numerics note.** Synchronous MTP turns run the flat target path (`run_mtp_whole_turn`) even on paged models, while `enableMtp: false` runs the paged executor; every verify token is forwarded as a sequential `[1,1]` decode so the MTP committed stream is bit-identical to the flat-path greedy stream (real-checkpoint `real_mtp_t0_lossless_gate`). The draft head is **stateful and NoPE**: it keeps a per-turn KV cache of its own (its own `k_proj`/`v_proj`, never the backbone's), seeded over the whole prompt during prefill with vLLM's EAGLE token shift and rewound on rejection by a cursor trim to the committed length. Depth is clamped to 1. That seed costs one extra MTP-layer pass over the prompt per turn — real TTFT, so measure it on the checkpoint rather than assuming it is free. On quantized checkpoints the flat and paged paths themselves are not greedy-identical — kernel ULP noise amplified by the Mamba-2 recurrence — so MTP-on vs paged-AR cross-path `T=0` text equality is not guaranteed, and the flat path is more loop-prone than the paged path on the NVFP4 checkpoint (see the reasoning note above). Reconciling flat/paged recurrence numerics is the open follow-up.

## Embedding model

| Model       | Purpose                                                          |
| ----------- | ---------------------------------------------------------------- |
| **Harrier** | Embedding model (inference-only). Loaded through `@mlx-node/lm`. |

## Vision-language models

| Model               | Backbone                              | Purpose                                                                                                       |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Qwen3.5 VLM**     | Qwen3.5 dense or MoE + vision encoder | General VLM; integrated with paged attention (text-only turns); LRU image-feature cache keyed by content hash |
| **PaddleOCR-VL**    | ERNIE language model + vision encoder | OCR-first VLM; single-turn `VLModel.chat()` entry point (intentionally outside the session API)               |
| **QianfanOCRModel** | InternVL-based                        | Newer OCR/document VLM, exported from `@mlx-node/vlm`                                                         |

## Document processing pipeline

| Model                 | Purpose                                                              |
| --------------------- | -------------------------------------------------------------------- |
| **PP-DocLayoutV3**    | Document layout analysis (RT-DETR + HGNetV2 backbone, 25 categories) |
| **PP-TextDet**        | Text-line detection (DBNet with PPHGNetV2 backbone)                  |
| **PP-TextRec**        | Text recognition (SVTR neck + CTC head, character dictionary)        |
| **PP-DocOrientation** | 4-class orientation classifier (0 / 90 / 180 / 270 degrees)          |
| **PP-DocUnwarp**      | Document dewarping via 2D displacement field (UVDocNet)              |

The pipeline is exposed as `StructureV3Pipeline` from `@mlx-node/vlm`:

```typescript
import { StructureV3Pipeline } from "@mlx-node/vlm";
const pipeline = await StructureV3Pipeline.load(modelDir);
const result = await pipeline.analyze(imageBuffer);
```

## ChatSession

`ChatSession<M>` (`packages/lm/src/chat-session.ts`) is the cross-model chat wrapper. It holds a `SessionCapableModel` and exposes:

- `send(message)` / `sendStream(message)` — chat turn rendered from the checkpoint template
- `sendToolResult(...)` / `sendToolResultStream(...)` — feed back a structured tool result through the same template path
- `reset()` — clear conversation
- `primeHistory(history)` / `startFromHistory(history)` / `startFromHistoryStream(history)` — server-side cold-start replay
- `applyChatTemplate(history)` — apply tokenizer chat template (e.g. for token counting)
- `hasBlockPagedCache?()` — paged-cache routing hint

Every role-aware turn sends the full structured history to native code. The checkpoint-provided chat template renders that history, and native KV reuse is allowed only when the rendered token sequence exactly extends the committed cache. A template mismatch safely falls back to full prefill; Rust never manufactures user/tool wire-format strings.

All generative wrappers (Qwen3, Qwen3.5 Dense, Qwen3.5 MoE, Gemma4,
Muse-Glimmer, LFM2.5, Nemotron 3.5 Lightning, and the VLM
`QianfanOCRModel`) structurally satisfy `SessionCapableModel` — any of
them can be passed to `new ChatSession(model)`.

## Streaming

```typescript
import { loadSession } from "@mlx-node/lm";

const session = await loadSession("./models/Qwen3.5-0.8B");
for await (const event of session.sendStream("Hello!")) {
  if (!event.done) process.stdout.write(event.text);
}
```

The streaming bridge is implemented in `packages/lm/src/stream.ts`: native callback-based methods are captured at module load and re-exposed as `AsyncGenerator` via `_runChatStream`.

## Muse-Glimmer Q4_K and DFlash

Convert Meta's GGUF target, vision projector, and DFlash companion together.
`--gguf-kquant` imports Q4_K, Q5_K, and Q6_K blocks without dequantizing or
requantizing them:

```bash
yarn mlx convert \
  -i ./Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
  -o ./muse-glimmer-30b-q4k \
  --config-dir ./Muse-Glimmer-30B \
  --mmproj ./mmproj-kquant.gguf \
  --draft ./dflash-kquant.gguf \
  --gguf-kquant
```

The output contains `model.safetensors`, `vision.safetensors`, and
`draft.safetensors`. The current inference path is text-only; the converter
preserves the vision sidecar, but Muse image/video execution is not wired yet.

When `draft.safetensors` is present, `hasMtpWeights()` and
`autoEnablesMtp()` return true and `ChatSession` enables DFlash unless a call
sets `enableMtp: false`. Muse DFlash uses the checkpoint's 16-token parallel
infill block by default. An explicit `mtpDepth` clamps to `[1, 16]`; leaving
both depth knobs unset enables the measured target-AR fallback guard.

Ordinary AR requests use the hybrid full/sliding paged cache and continuous
batching. The default 2 GiB cache admits up to eight sequences for the release
geometry. DFlash uses owner-isolated flat target and draft caches and is a
scheduler barrier, so simultaneous DFlash turns are serialized; set
`enableMtp: false` for request pools where batched concurrency is preferred.

Paged AR prefixes can also survive hot-cache eviction and process restart via
the SSD cold tier. Enable it with `persist_paged_cache: true` in the model
config or `MLX_PERSIST_PAGED_CACHE=1`; `mlx agent` enables persistence by
default for supported families. Muse persists the 13 full-attention layers as
the authoritative block chain and all 39 sliding layers as a companion sidecar
at the same token boundary. A missing, corrupt, or mismatched sidecar discards
the full-group hit and recomputes from the start. DFlash's flat caches are not
persisted, and an unfinished in-flight AR turn uses recompute preemption rather
than publishing partial state.

## Speculative decoding: Gemma4 drafts (DSpark + assistant MTP)

Gemma4 supports two external-draft speculative decoding variants behind one load surface — the loader picks the variant from the draft checkpoint's `config.json`:

- **DSpark** (`deepseek-ai/dspark_gemma4_12b_block7`): a 5-layer draft that proposes a block of up to 7 tokens per cycle from mask tokens, conditioned on tapped target hidden states.
- **Assistant MTP** (`google/gemma-4-{12B,26B-A4B,31B}-it-assistant`, apache-2.0, ~800 MB): Google's official 4-layer draft. Its attention layers are Q-only and read the **target's own KV caches** (last non-KV-shared sliding/full layers) — the draft keeps no KV cache and never prefills. Tokens are drafted one at a time, chained through a hidden-state feedback loop, `mtpDepth` per cycle (default 3). The E2B/E4B assistants (centroid sparse lm_head) are not yet supported and are rejected at load.

Pass `draftModelPath` when loading an external draft:

```typescript
import { loadSession } from "@mlx-node/lm";

const session = await loadSession("./models/gemma-4-12b-it", {
  draftModelPath: "./models/dspark_gemma4_12b_block7",
});
// The attached draft flips hasMtpWeights(), so ChatSession auto-enables the
// speculative path; pass `enableMtp: false` per call to opt out.
const result = await session.send("Give a simple recipe for pancakes.", {
  config: { temperature: 0 },
});
console.log(
  result.performance?.mtpCycles,
  result.performance?.mtpMeanAcceptedTokensTotal,
);
```

For a self-contained checkpoint, place the draft's `config.json` and
`model.safetensors` under `<model>/draft/`. Gemma4 discovers that directory
automatically, so `loadSession('./models/gemma-4-12b-it')` enables the same
speculative path without a load option. An explicit `draftModelPath` overrides
the embedded directory. `mlx agent` is the exception: it uses a paged AR
overlay by default and enables the embedded draft only when
`MLX_AGENT_ENABLE_GEMMA_DRAFT=1` is set.

- **Lossless at T=0** — every committed token is verified by the target model, so greedy output matches the plain autoregressive run (up to inherent bf16 near-ties; see the oracle suites in `crates/mlx-core/tests/gemma4_dspark.rs` and `crates/mlx-core/tests/gemma4_assistant.rs`).
- **Stats** — `ChatResult.performance` reports `mtpCycles` (actual draft+verify cycles executed) and `mtpMeanAcceptedTokensTotal` (mean committed tokens per speculative cycle, including the always-verified token). DSpark's target-only calibration/fallback cycles remain part of the overall token/decode-speed metrics but are intentionally excluded from these MTP acceptance fields.
- **Knobs** — DSpark: with both knobs unset, full draft blocks (7 tokens on the v1 draft) are guarded by a short, per-turn target-AR/DSpark throughput calibration; if speculation loses on the current host and context, the remainder of that turn uses exact target-only decoding. The guard activates only when the generation budget can contain the AR probe plus two full-depth speculative probes; shorter generations preserve the fixed-block schedule. An explicit `mtpDepth` caps and pins the block, disabling the guard unless `mtpAdaptiveDepth: true` opts it back in; `mtpAdaptiveDepth: false` always disables it. Assistant: unset `mtpDepth` defaults to 3 drafts per cycle, explicit values clamp to [1, 8], and `mtpAdaptiveDepth` remains ignored.
- **Memory** — the draft loads alongside the target (~6.9 GB extra for the bf16 DSpark 12B draft; ~0.8 GB for an assistant). Both variants run on the flat KV-cache path; a target config that explicitly enables `use_block_paged_cache` is rejected at load.
- `draftModelPath` is gemma4-only: `loadModel` / `loadSession` reject it for every other family.

## Server-side sessions

The HTTP endpoints `/v1/responses` and `/v1/messages` live in `@mlx-node/server` (`packages/server/src/endpoints/`). Both route through a per-model `SessionRegistry` (`packages/server/src/session-registry.ts`) that owns the `ChatSession` lifetimes — clients pass `previous_response_id` and the registry handles resume vs. cold-start replay internally.
