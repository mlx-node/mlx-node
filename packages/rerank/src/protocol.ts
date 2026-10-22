import type { RerankerFamily, RerankerFamilyOptions, RerankerPolicy, RerankRequest, RerankResponse } from './types.js';

export class RerankError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status = 400,
  ) {
    super(message);
    this.name = 'RerankError';
  }
}
const commonPolicyKeys = ['maxQueryTokens', 'maxDocumentTokens', 'maxDocuments', 'maxInputTokens'];
const familyPolicyKeys: Record<RerankerFamily, ReadonlySet<string>> = {
  jina_reranker: new Set([...commonPolicyKeys, 'maxBlockTokens', 'maxDocumentsPerBlock', 'prefillChunkSize']),
  xlm_roberta_reranker: new Set([...commonPolicyKeys, 'maxPairTokens', 'maxBatchSize', 'maxBatchTokens']),
};
/** Native-free validation shared by SDK and service startup. Defaults are resolved by the model profile. */
export function validateRerankerPolicy(
  value: unknown,
  family: RerankerFamily = 'jina_reranker',
): Partial<RerankerPolicy> {
  if (value === undefined) return {};
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new RerankError('invalid_request', 'policy must be an object');
  for (const [key, limit] of Object.entries(value)) {
    if (!familyPolicyKeys[family].has(key)) throw new RerankError('invalid_request', `Unknown rerank policy: ${key}`);
    if (typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1)
      throw new RerankError('invalid_request', `${key} must be a positive integer`);
    // Match the backend's shape/index range before startup can evict a resident model.
    if (family === 'xlm_roberta_reranker' && limit > 2 ** 31 - 1)
      throw new RerankError('invalid_request', `${key} exceeds the XLM-R 32-bit policy range`);
  }
  return { ...value };
}
/** Validate all startup policies before any model mutation or loading. */
export function validateRerankerFamilies(value: RerankerFamilyOptions = {}): RerankerFamilyOptions {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new RerankError('invalid_request', 'Invalid reranker configuration');
  for (const key of Object.keys(value))
    if (!['policy', 'families'].includes(key))
      throw new RerankError('invalid_request', `Unknown reranker configuration: ${key}`);
  validateRerankerPolicy(value.policy, 'jina_reranker');
  if (value.families !== undefined) {
    if (!value.families || typeof value.families !== 'object' || Array.isArray(value.families))
      throw new RerankError('invalid_request', 'families must be an object');
    if (value.policy !== undefined && value.families.jina_reranker !== undefined)
      throw new RerankError('invalid_request', 'Jina policy specified twice');
    for (const [family, options] of Object.entries(value.families)) {
      if (!Object.hasOwn(familyPolicyKeys, family) || !options || typeof options !== 'object' || Array.isArray(options))
        throw new RerankError('invalid_request', `Invalid reranker family: ${family}`);
      for (const key of Object.keys(options))
        if (key !== 'policy') throw new RerankError('invalid_request', `Unknown family option: ${key}`);
      validateRerankerPolicy(options.policy, family as RerankerFamily);
    }
  }
  return value;
}
export function familyLoadOptions(value: RerankerFamilyOptions | undefined, family: RerankerFamily) {
  return value?.families?.[family] ?? (family === 'jina_reranker' ? { policy: value?.policy } : {});
}
export function normalizeRequest(request: RerankRequest): {
  query: string;
  documents: string[];
  topN?: number;
  truncate: boolean;
} {
  if (!request || typeof request.query !== 'string' || !request.query.trim())
    throw new RerankError('invalid_request', 'query must be a nonempty string');
  if (!Array.isArray(request.documents)) throw new RerankError('invalid_request', 'documents must be an array');
  const documents = request.documents.map((doc: unknown) => {
    const text = typeof doc === 'string' ? doc : doc && typeof doc === 'object' && 'text' in doc ? doc.text : undefined;
    if (typeof text !== 'string')
      throw new RerankError('invalid_request', 'Every document must be a string or { text: string }');
    return text;
  });
  if (request.topN !== undefined && (!Number.isSafeInteger(request.topN) || request.topN < 1))
    throw new RerankError('invalid_request', 'topN must be a positive integer');
  for (const key of ['returnDocuments', 'truncate'] as const)
    if (request[key] !== undefined && typeof request[key] !== 'boolean')
      throw new RerankError('invalid_request', `${key} must be boolean`);
  return { query: request.query, documents, topN: request.topN, truncate: request.truncate ?? false };
}
export function parseRerankBody(value: unknown): { model: string; request: RerankRequest } {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new RerankError('invalid_request', 'Expected a JSON object');
  const body = value as Record<string, unknown>;
  for (const key of Object.keys(body))
    if (!['model', 'query', 'documents', 'top_n', 'return_documents', 'truncate'].includes(key))
      throw new RerankError('invalid_request', `Unsupported parameter: ${key}`);
  if (typeof body.model !== 'string' || !body.model.trim())
    throw new RerankError('invalid_request', 'model is required');
  const request = {
    query: body.query,
    documents: body.documents,
    topN: body.top_n,
    returnDocuments: body.return_documents,
    truncate: body.truncate,
  } as RerankRequest;
  normalizeRequest(request);
  return { model: body.model, request };
}
export function rerankWireResponse(model: string, value: RerankResponse) {
  return {
    model,
    results: value.results.map(({ index, relevanceScore, document }) => ({
      index,
      relevance_score: relevanceScore,
      ...(document ? { document } : {}),
    })),
    usage: { input_tokens: value.usage.inputTokens, prompt_tokens: value.usage.promptTokens },
    meta: {
      processed_documents: value.meta.processedDocuments,
      ...(value.meta.profile === 'xlmr-cross-encoder-v1'
        ? {
            profile: value.meta.profile,
            batches: value.meta.batches,
            max_pair_tokens: value.meta.maxPairTokens,
            max_batch_size: value.meta.maxBatchSize,
            max_batch_tokens: value.meta.maxBatchTokens,
            padded_tokens: value.meta.paddedTokens,
            padding_tokens: value.meta.paddingTokens,
          }
        : {
            profile: value.meta.profile,
            blocks: value.meta.blocks,
            max_block_tokens: value.meta.maxBlockTokens,
            max_documents_per_block: value.meta.maxDocumentsPerBlock,
          }),
      truncated_query: value.meta.truncatedQuery,
      truncated_document_indices: value.meta.truncatedDocumentIndices,
    },
  };
}
export function rerankError(error: unknown): RerankError {
  if (error instanceof RerankError) return error;
  const message = error instanceof Error ? error.message : String(error);
  const match = /rerank_([a-z_]+):\s*(.*)/s.exec(message);
  const code = match?.[1] ?? 'internal_error';
  const status =
    ({ invalid_request: 400, input_too_large: 413, closed: 503, cancelled: 499 } as Record<string, number>)[code] ??
    500;
  return new RerankError(code, match?.[2] ?? message, status);
}
