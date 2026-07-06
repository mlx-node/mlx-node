# Models

## Language models

All language wrappers share a uniform `ChatSession<M>` surface (`send` / `sendStream` / `sendToolResult` / `reset`) driven by the native `chatSessionStart` / `chatSessionContinue` / `chatSessionContinueTool` NAPI entry points. The legacy `model.chat()` / `model.chatStream()` methods are removed from every generative model.

| Model             | `generate()` | Session API |  Training  | Notes                                                            |
| ----------------- | :----------: | :---------: | :--------: | ---------------------------------------------------------------- |
| **Qwen3**         |     yes      |     yes     | GRPO + SFT | Speculative decoding; paged attention                            |
| **Qwen3.5 Dense** |     yes      |     yes     | GRPO + SFT | Compiled C++ forward (see [ffi-cpp.md](ffi-cpp.md)); VLM variant |
| **Qwen3.5 MoE**   |     yes      |     yes     | GRPO + SFT | Compiled C++ forward with expert routing; VLM variant            |
| **Gemma4**        |     yes      |     yes     |     —      | Hybrid sliding/global attention + MoE/PLE; DSpark spec. decoding |
| **LFM2.5**        |     yes      |     yes     |     —      | Hybrid conv + attention                                          |

`Qwen3Model | Qwen35Model | Qwen35MoeModel` is the public `TrainableModel` union in `@mlx-node/lm` — Gemma4 and LFM2.5 are inference-only.

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
import { StructureV3Pipeline } from '@mlx-node/vlm';
const pipeline = await StructureV3Pipeline.load(modelDir);
const result = await pipeline.analyze(imageBuffer);
```

## ChatSession

`ChatSession<M>` (`packages/lm/src/chat-session.ts`) is the cross-model chat wrapper. It holds a `SessionCapableModel` and exposes:

- `send(message)` / `sendStream(message)` — chat turn (delta path when KV is reusable)
- `sendToolResult(...)` / `sendToolResultStream(...)` — feed back a tool result; always uses `chatSessionContinueTool`
- `reset()` — clear conversation
- `primeHistory(history)` / `startFromHistory(history)` / `startFromHistoryStream(history)` — server-side cold-start replay
- `applyChatTemplate(history)` — apply tokenizer chat template (e.g. for token counting)
- `hasBlockPagedCache?()` — paged-cache routing hint

Turn 0 (and any turn whose image set changed) dispatches through `chatSessionStart` with the full rebuilt history. Later turns take the cheap `chatSessionContinue` delta path that reuses the live KV cache. Tool-result turns always use `chatSessionContinueTool`.

All generative wrappers (Qwen3, Qwen3.5 Dense, Qwen3.5 MoE, Gemma4, LFM2.5, and the VLM `QianfanOCRModel`) structurally satisfy `SessionCapableModel` — any of them can be passed to `new ChatSession(model)`.

## Streaming

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./models/Qwen3.5-0.8B');
for await (const event of session.sendStream('Hello!')) {
  if (!event.done) process.stdout.write(event.text);
}
```

The streaming bridge is implemented in `packages/lm/src/stream.ts`: native callback-based methods are captured at module load and re-exposed as `AsyncGenerator` via `_runChatStream`.

## Speculative decoding: Gemma4 + DSpark

Gemma4 supports DSpark speculative decoding: an external draft model proposes a block of tokens per cycle and the target model verifies the whole block in one forward, committing the accepted prefix. Pass `draftModelPath` (a DSpark draft checkpoint directory, e.g. `deepseek-ai/dspark_gemma4_12b_block7`) when loading:

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./models/gemma-4-12b-it', {
  draftModelPath: './models/dspark_gemma4_12b_block7',
});
// The attached draft flips hasMtpWeights(), so ChatSession auto-enables the
// speculative path; pass `enableMtp: false` per call to opt out.
const result = await session.send('Give a simple recipe for pancakes.', { config: { temperature: 0 } });
console.log(result.performance?.mtpCycles, result.performance?.mtpMeanAcceptedTokensTotal);
```

- **Lossless at T=0** — every committed token is verified by the target model, so greedy output matches the plain autoregressive run (up to inherent bf16 near-ties; see the oracle suite in `crates/mlx-core/tests/gemma4_dspark.rs`).
- **Stats** — `ChatResult.performance` reports `mtpCycles` (draft+verify cycles executed) and `mtpMeanAcceptedTokensTotal` (mean committed tokens per cycle, including the always-verified token).
- **Knobs** — an unset `mtpDepth` runs full draft blocks (7 tokens on the v1 draft); an explicit `mtpDepth` caps the block. `mtpAdaptiveDepth` is ignored.
- **Memory** — the draft loads alongside the target (~6.9 GB extra for the bf16 12B draft). DSpark runs on the flat KV-cache path; a target config that explicitly enables `use_block_paged_cache` is rejected at load.
- `draftModelPath` is gemma4-only: `loadModel` / `loadSession` reject it for every other family.

## Server-side sessions

The HTTP endpoints `/v1/responses` and `/v1/messages` live in `@mlx-node/server` (`packages/server/src/endpoints/`). Both route through a per-model `SessionRegistry` (`packages/server/src/session-registry.ts`) that owns the `ChatSession` lifetimes — clients pass `previous_response_id` and the registry handles resume vs. cold-start replay internally.
