# General-purpose reranking

`@mlx-node/rerank` ranks retrieval candidates for any caller: RAG, search, memory services, or batch processing. It is independent of Hindsight. Jina receives all candidates in each block, so its scores and document features depend on the query, candidate order and block budget. Internal document vectors are deliberately not exposed as reusable embeddings.

BGE-v2-M3 uses an independent sentence-pair encoder with sigmoid scoring. See [BGE support and migration](rerank-bge.md). The following block and prompt rules describe Jina.

## Supported checkpoints

| Profile            | Attention                                                     | Weight formats                      |
| ------------------ | ------------------------------------------------------------- | ----------------------------------- |
| Jina Reranker v3   | Global causal attention                                       | Dense BF16, MLX affine 4-bit, MXFP4 |
| Jina Reranker v3.5 | Configured full and causal sliding layers; final query marker | Dense BF16, MLX affine 4-bit, MXFP4 |

Identification uses `JinaForRanking`, attention configuration, and optional versioned metadata, never a directory/repository name. Converted checkpoints carry `reranker: { version: 1, profile: "jina-v3" | "jina-v3.5" }`. Conflicting metadata, unknown layer layouts, scaled RoPE, bias-bearing projections and other unsupported structures fail loading.

Single-file and indexed SafeTensors checkpoints are supported, with a separate or merged projector. Known projector tensor aliases are normalized at the loader boundary. Duplicate tensors fail; missing weights never leave randomly initialized parameters. Quantized linear and embedding weights remain packed. Mixed dense/packed modules and quantized projectors are supported when their explicit metadata and sidecars match. A repository name containing “Q4” is not evidence of a supported format. No repository Python is executed by the runtime or converter.

## TypeScript

```ts
import { loadReranker } from '@mlx-node/rerank';

const reranker = await loadReranker('/models/jina', {
  policy: { maxBlockTokens: 32768, prefillChunkSize: 512 },
});
try {
  const ranked = await reranker.rerank({
    query: 'What is unified memory?',
    documents: ['CPU and GPU share a memory pool.', { text: 'A city park.' }],
    topN: 1,
    returnDocuments: true,
    signal: AbortSignal.timeout(30_000),
  });
  console.log(ranked.results[0]); // { index: 0, relevanceScore: ..., document: { text: ... } }
} finally {
  await reranker.dispose();
}
```

The SDK accepts string documents or `{ text }` objects, preserves original indices, sorts by descending score and uses index order for exact ties. Omitting `topN` returns everything; a positive value larger than the candidate count is clamped. Empty candidates return an empty result; blank queries are rejected. Empty and duplicate documents remain candidates. Returned text is the original caller input, including when the model sees a truncated version.

`model.info` reports the profile, effective context/policy, packed-weight flag and weight formats. `/catalog`, `/types` and `/protocol` are native-free import paths. `dispose()` is idempotent. Abort requests reach the native worker and settle only after worker completion/synchronization. Direct SDK callers should bound their own concurrency; each model serializes native work.

## HTTP

Start a host whose models directory contains one or more model subdirectories:

```sh
mlx serve --models-dir /models --model jina
curl http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"jina","query":"What is unified memory?","documents":["A shared CPU/GPU memory pool.",{"text":"A city park."}],"top_n":1,"return_documents":true}'
```

Use the URL printed by `mlx serve` (or `--port 8080` for the example). When authentication is configured, include the same bearer token used by other inference routes.

```json
{
  "model": "jina",
  "results": [{ "index": 0, "relevance_score": 0.7, "document": { "text": "A shared CPU/GPU memory pool." } }],
  "usage": { "input_tokens": 18, "prompt_tokens": 190 },
  "meta": {
    "profile": "jina-v3.5",
    "blocks": 1,
    "processed_documents": 2,
    "max_block_tokens": 32768,
    "max_documents_per_block": 125,
    "truncated_query": false,
    "truncated_document_indices": []
  }
}
```

The numbers above illustrate the schema. Actual cosine scores are checkpoint/input dependent and may be negative. They are not calibrated probabilities. `input_tokens` counts sanitized input before optional text truncation; `prompt_tokens` counts actual model tokens, including templates and repeated query text in each block.

The accepted fields are exactly `model`, `query`, `documents`, `top_n`, `return_documents`, `truncate`. Unsupported extension fields are rejected. This implements common text-rerank fields, not every Jina/Cohere API extension. `/v1/models` advertises `task: "rerank"` and capabilities. Known incompatible models fail before loading; unknown rerank names do not use chat's alias fallback. Authentication and CORS use the host's existing middleware.

| Error                                                                        | HTTP status |
| ---------------------------------------------------------------------------- | ----------- |
| `invalid_request`, `model_capability_mismatch`                               | 400         |
| `model_not_found`                                                            | 404         |
| `input_too_large`                                                            | 413         |
| `queue_full`                                                                 | 429         |
| `server_shutdown`, `model_unavailable`, `closed`                             | 503         |
| `deadline_exceeded`                                                          | 504         |
| `invalid_model`, `numerical_error`, `worker_failed`, other internal failures | 500         |

Client disconnects cancel work; the server does not release its inference lease until native work stops. A deadline is a cancellation deadline, not permission to abandon a still-running GPU task. Consequently the final timeout response can follow the deadline by the time needed for a safe cancellation boundary.

## Limits and ranking semantics

| Policy                       | Default                                |
| ---------------------------- | -------------------------------------- |
| `maxBlockTokens`             | min(32768, effective model context)    |
| `maxDocumentsPerBlock`       | 125                                    |
| `prefillChunkSize`           | 512                                    |
| `maxQueryTokens`             | v3: 512; v3.5: 2048                    |
| `maxDocumentTokens`          | v3: 2048; v3.5: 8192                   |
| `maxDocuments`               | 1024                                   |
| `maxInputTokens`             | 1,000,000                              |
| HTTP `maxBodyBytes`          | 16 MiB                                 |
| HTTP `timeoutMs`             | 300,000                                |
| HTTP `maxConcurrentRequests` | 17 active/body-reading/queued requests |
| Per-model waiting queue      | Existing host setting; default 16      |

Load options can override model policy. A server can configure `InferenceHostOptions.reranker.policy` and `InferenceHostOptions.rerank` (HTTP limits). `mlx serve --rerank-config file.json` accepts:

```json
{
  "policy": { "maxBlockTokens": 16384, "maxDocuments": 512, "maxInputTokens": 500000 },
  "service": { "maxBodyBytes": 8388608, "maxDocuments": 512, "timeoutMs": 120000, "maxConcurrentRequests": 17 }
}
```

HTTP callers cannot raise these limits. Query/document overflow is an error unless `truncate: true`; total-input/candidate limits remain errors even with truncation. Truncation never drops candidates. A single document plus query/template that cannot fit the configured block budget fails; increase the budget or reduce the text limits explicitly. Reserved ranking markers are stripped from input to prevent feature-position injection.

Candidate blocks retain original order and use exact fully formatted token counts. Each block computes contextual document/query vectors. The unnormalized query vectors are weighted by each block's maximum `(cosine + 1) / 2`, fused in FP32, then used to score every document before `topN`. Nonfinite values, zero norms and invalid fusion weights fail explicitly. Block budgets and document count caps affect ranking semantics; internal prefill chunk size only changes execution. The effective token budget, document cap per block and actual block count are returned.

## CLI and conversion

```sh
mlx download model --model jinaai/jina-reranker-v3.5-mlx --output /models/jina-bf16
mlx rerank --model /models/jina-bf16 --input request.json
cat request.json | mlx rerank --model jina-bf16 --input -
mlx convert --input /models/jina-bf16 --output /models/jina-affine4 -d bf16 -q --q-bits 4 --q-group-size 64
mlx convert --input /models/jina-bf16 --output /models/jina-mxfp4 -d bf16 -q --q-bits 4 --q-group-size 32 --q-mode mxfp4
```

`request.json` contains the HTTP request fields except `model`, which comes from `--model`. stdout is JSON; diagnostics go to stderr. Local names resolve through the normal models-directory setting. Conversion chooses the ranking recipe from architecture, writes version metadata, preserves tokenizer/attribution files and keeps the projector dense by default. Convert from an unquantized source into a new or empty directory. Already-quantized checkpoints can be loaded directly; re-quantization and additional community formats are not supported by this recipe.

## Architecture and operations

The native `jina_reranker` module owns configuration, strict weight loading, tokenization, block planning, a separate inference-only transformer, projector and scoring. It reuses `LinearProj`, packed embedding, normalization, RoPE, attention, request-private KV caches and `ModelThread`. Full layers use normal caches; sliding layers use rotating caches with separate absolute RoPE and local mask offsets. No LM logits, chat fast-path changes or new Metal kernels are involved. Each reranker registers its packed weight bytes with the existing process-wide cache-budget coordinator and removes that contribution on disposal. Existing `MLX_CACHE_LIMIT_GB` / `MLX_GPU_HEADROOM_GB` settings continue to apply. Ordinary Qwen3's quantized-weight rejection remains intact.

The SDK owns the public contract and cancellation bridge. `RerankRegistry` owns identity-based admission/FIFO queues; `RerankService` owns HTTP validation, deadlines and error mapping. The host's capability coordinator composes chat, ASR, TTS and rerank adapters. By default one model of each capability may remain resident; `--model-residency single` makes switching unload every other capability under the existing writer admission. Load/unload use exclusive permits; inference uses shared permits. Chat unregistering/session disposal retains the existing chat lifecycle; its native model allocation can remain until JavaScript GC releases the last model handle. Rerank and audio have explicit native disposal. Cold admissions transfer atomically into resident queues. Shutdown closes admission, cancels/drains work and admitted loads, then releases residents.

Logs contain model/profile/weight formats, counts, tokens, truncation counts, status and phase timings. Rerank query/document text and returned documents are excluded from request logs. GPU allocator cache retention is distinct from active model/request allocations; validation measures both. Co-resident inference shares hardware and can increase latency; see the measured report rather than assuming zero overhead.

## Licensing and validation

Official model cards declare CC BY-NC 4.0. That is separate from mlx-node's MIT runtime license. Preserve the checkpoint's model card/license when distributing converted weights. See [package notice](../packages/rerank/NOTICE).

[Validation procedure](../scripts/rerank/README.md) pins model revisions, Python versions and frozen score thresholds. [Validation report](validation/rerank.md) records measured results and remaining limits. Real-model numerical, lifecycle and existing regression gates must pass before merging; this feature does not auto-merge or publish anything.
