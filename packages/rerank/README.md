# @mlx-node/rerank

Native text reranking on Apple Silicon, independently of chat, audio, or any retrieval application. Jina v3/v3.5 use listwise cosine scoring; BGE-v2-M3 uses a bidirectional XLM-R cross-encoder and automatically returns sigmoid scores. See the [BGE configuration and migration guide](../../docs/rerank-bge.md) for its precision formats and policy.

```ts
import { loadReranker } from '@mlx-node/rerank';

const model = await loadReranker('/models/reranker');
try {
  const response = await model.rerank({
    query: 'What is unified memory?',
    documents: ['A shared memory pool for CPU and GPU.', { text: 'A city park.' }],
    topN: 1,
    returnDocuments: true,
    signal: AbortSignal.timeout(30_000),
  });
  console.log(response.results); // original index, relevanceScore, optional document
} finally {
  await model.dispose();
}
```

`@mlx-node/rerank/catalog`, `/types` and `/protocol` do not import the native addon. Cancellation waits for the native worker to stop; `dispose()` is idempotent and drains work. A model instance serializes its requests. Callers using the SDK directly should bound concurrency; HTTP serving provides bounded admission.

See [configuration, HTTP, CLI and compatibility](../../docs/rerank.md) and [reproducible validation](../../scripts/rerank/README.md).

The runtime package is MIT licensed. Model weights and their training prompt protocols retain their own licenses; the supported official Jina checkpoints are CC BY-NC 4.0. See [NOTICE](./NOTICE).
