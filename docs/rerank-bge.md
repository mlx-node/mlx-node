# BGE-v2-M3 reranking

`@mlx-node/rerank` supports XLM-R single-output cross-encoders through the
`xlm_roberta_reranker` family and `xlmr-cross-encoder-v1` profile. The reference
checkpoint is `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Architecture recognition uses config
fields and validates the encoder, tokenizer and weights; directory names do not
select the backend. Ordinary XLM-R embedding models are not rerankers.

This branch has passed numerical and lifecycle checks but has **not passed all
acceptance gates**. The M4 batching throughput target failed, and an initial Jina
latency regression did not reproduce consistently. See the [acceptance report](validation/bge-rerank.md)
for evidence and remaining limits before considering a merge or release.

BGE returns **sigmoid scores between 0 and 1**. Sigmoid runs once in the native
backend. SDK, HTTP and CLI share those scores; there is no raw-logit option.
Scores need not sum to one across candidates and are not calibrated relevance
probabilities. Jina continues returning cosine scores under its existing protocol.

## Usage

```ts
import { loadReranker } from '@mlx-node/rerank';

const model = await loadReranker('/models/bge-v2-m3', {
  policy: { maxPairTokens: 8192, maxBatchSize: 8, maxBatchTokens: 8192 },
});
try {
  const result = await model.rerank({
    query: '中国的首都是哪里？',
    documents: ['北京是中国的首都。', { text: '巴黎是法国的首都。' }],
    returnDocuments: true,
    signal: AbortSignal.timeout(30_000),
  });
  console.log(result.results);
} finally {
  await model.dispose();
}
```

```sh
mlx download model --model BAAI/bge-reranker-v2-m3 --output /models/bge-v2-m3
mlx convert --input /models/bge-v2-m3 --output /models/bge-v2-m3-fp16 --dtype float16
mlx rerank --model /models/bge-v2-m3 --input request.json
mlx serve --port 8080 --models-dir /models --rerank-config rerank-config.json
```

`request.json`:

```json
{ "query": "capital of China", "documents": ["Beijing", "Paris"], "top_n": 1, "return_documents": true }
```

```sh
curl http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-v2-m3","query":"capital of China","documents":["Beijing","Paris"],"top_n":1}'
```

The response uses `results[].index`, `relevance_score`, optional `document.text`,
`usage.input_tokens`, `usage.prompt_tokens` and family-specific `meta`.
`/v1/models` reports the actual family and `task: "rerank"`. Existing authentication,
CORS, body limits, queue limits, deadlines and residency policies apply.

## Policy and migration

| BGE setting         | Default                                               |
| ------------------- | ----------------------------------------------------- |
| `maxPairTokens`     | `min(8192, effective context)`                        |
| `maxQueryTokens`    | 512, bounded by effective context minus pair overhead |
| `maxDocumentTokens` | 8192                                                  |
| `maxBatchSize`      | 8                                                     |
| `maxBatchTokens`    | 8192, including padding                               |
| `maxDocuments`      | 1024                                                  |
| `maxInputTokens`    | 1,000,000, before truncation                          |

A sentence pair includes four template tokens. Explicit pair and document limits cannot exceed
context; the query limit must leave room for those tokens. Batch token budget
must accommodate the maximum pair length. A shorter pair policy may also require
a smaller query limit. Unsupported family policy keys are errors.

The tokenizer builds `<s> query </s></s> document </s>`, including empty documents.
Query tokenization is reused. Truncation only operates on token IDs and requires
`truncate: true`; candidate count and original token limits always apply. Query
truncates first, then each document to its own limit and remaining pair budget.
Empty and duplicate documents retain their indices. No document windows or
cross-request batching are used.

`usage.inputTokens` counts the raw query once plus raw documents;
`promptTokens` counts all actual sentence pairs, including repeated queries and
special tokens, excluding padding. `meta.paddedTokens` counts the full padded
batch workload; `paddingTokens` is its excess over prompt tokens. Batches are
stable length-sorted and satisfy both count and padded-token budgets.

Server configuration selects a policy per family:

```json
{
  "families": {
    "jina_reranker": { "policy": { "maxBlockTokens": 32768 } },
    "xlm_roberta_reranker": { "policy": { "maxPairTokens": 8192, "maxBatchTokens": 8192, "maxBatchSize": 8 } }
  },
  "service": { "maxDocuments": 1024, "timeoutMs": 300000 }
}
```

The old top-level `policy` remains a Jina-only compatibility alias. Defining both
that alias and `families.jina_reranker` is rejected before startup. HTTP callers
cannot override these load policies or raise server limits.

Consumers of model-specific metadata should narrow the union:

```ts
if (result.meta.profile === 'xlmr-cross-encoder-v1') {
  console.log(result.meta.batches, result.meta.paddingTokens);
} else {
  console.log(result.meta.blocks, result.meta.maxBlockTokens);
}
if (model.info.family === 'xlm_roberta_reranker') {
  console.log(model.info.scoreType, model.info.policy.maxPairTokens);
}
```

`/types`, `/protocol`, `/catalog` and `/families` imports do not load the addon.
`/families` contains pure descriptors and is safe for browser-side catalogs.

## Implementation and formats

The native backend uses a bidirectional post-LayerNorm XLM-R encoder, absolute
positions, exact GELU and a CLS → dense → tanh → single-output classifier.
Position IDs treat the pad token specially, while attention masks come from the
encoding and actual right padding. Literal `<pad>` in text remains attended.
Fused SDPA receives a broadcast key mask. Each layer is evaluated before checking
cancellation; the shared runtime synchronizes completed GPU work before releasing
the request, permit or model.

Implemented storage contracts are FP32, FP16, BF16, MLX affine 4-bit and MXFP4.
Packed word embeddings gather and dequantize only requested rows. Conversion
quantizes word embeddings and encoder matrices; classifier, position/type
embeddings, LayerNorm and additive biases remain dense. Quantized classifiers
can also load. Affine `.biases` and additive linear `.bias` are distinct.

| Format                  | Numerical verification                  | Fixed-subset 4-bit quality gate |
| ----------------------- | --------------------------------------- | ------------------------------- |
| FP32                    | Pass against official Transformers      | Reference baseline              |
| FP16                    | Pass against same-checkpoint Python MLX | Not required by this gate       |
| BF16                    | Pass against same-checkpoint Python MLX | Not required by this gate       |
| affine 4-bit / group 64 | Pass against same-checkpoint Python MLX | Pass on both datasets           |
| MXFP4 / group 32        | Pass against same-checkpoint Python MLX | Pass on both datasets           |

These columns describe individual checks. Overall acceptance remains blocked by
the failed batching target and unresolved initial Jina performance regression;
the full performance matrix is incomplete. Exact quality changes, including MRR
decreases, are recorded in the [acceptance report](validation/bge-rerank.md).

Dense modules use one compute dtype. Packed embedding rows convert to that dtype;
affine projections restore it at each projection boundary even when sidecars
use higher precision. Conflicting compute metadata, unsupported dtypes, missing
biases, invalid shapes, orphan sidecars and duplicate tensors fail loading.
Model-specific rules do not change other families' quantization decisions.

Conversion requires a fresh output directory and retains tokenizer and upstream
attribution. It records the profile, compute dtype, quantization metadata and
source revision when known. Unknown source revision remains null. The pinned BGE
model card declares Apache-2.0; the runtime's project license does not replace
checkpoint licenses.

See [validation commands](../scripts/rerank/bge/README.md) and the
[acceptance report](validation/bge-rerank.md) for tested formats and actual gates.
