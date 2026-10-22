export type RerankerFamily = 'jina_reranker' | 'xlm_roberta_reranker';
interface CommonPolicy {
  maxQueryTokens: number;
  maxDocumentTokens: number;
  maxDocuments: number;
  maxInputTokens: number;
}
export interface JinaRerankerPolicy extends CommonPolicy {
  maxBlockTokens: number;
  maxDocumentsPerBlock: number;
  prefillChunkSize: number;
}
export interface XlmRobertaRerankerPolicy extends CommonPolicy {
  maxPairTokens: number;
  maxBatchSize: number;
  maxBatchTokens: number;
}
export type RerankerPolicy = JinaRerankerPolicy | XlmRobertaRerankerPolicy;
interface CommonInfo {
  contextWindow: number;
  quantized: boolean;
  weightFormats: readonly ('dense' | 'affine-4bit' | 'mxfp4')[];
}
export type RerankerInfo = CommonInfo &
  (
    | {
        family: 'jina_reranker';
        profile: 'jina-v3' | 'jina-v3.5';
        execution: 'listwise';
        scoreType: 'cosine';
        policy: Readonly<JinaRerankerPolicy>;
      }
    | {
        family: 'xlm_roberta_reranker';
        profile: 'xlmr-cross-encoder-v1';
        execution: 'cross-encoder';
        scoreType: 'sigmoid';
        policy: Readonly<XlmRobertaRerankerPolicy>;
      }
  );
export type RerankDocument = string | { text: string };
export interface RerankRequest {
  query: string;
  documents: readonly RerankDocument[];
  topN?: number;
  returnDocuments?: boolean;
  truncate?: boolean;
  signal?: AbortSignal;
}
interface CommonMeta {
  processedDocuments: number;
  truncatedQuery: boolean;
  truncatedDocumentIndices: number[];
}
export type RerankMeta = CommonMeta &
  (
    | { profile: 'jina-v3' | 'jina-v3.5'; blocks: number; maxBlockTokens: number; maxDocumentsPerBlock: number }
    | {
        profile: 'xlmr-cross-encoder-v1';
        batches: number;
        maxPairTokens: number;
        maxBatchSize: number;
        maxBatchTokens: number;
        paddedTokens: number;
        paddingTokens: number;
      }
  );
export interface RerankResponse {
  results: Array<{ index: number; relevanceScore: number; document?: { text: string } }>;
  usage: { inputTokens: number; promptTokens: number };
  meta: RerankMeta;
}
export interface RerankerModel {
  readonly info: RerankerInfo;
  rerank(request: RerankRequest): Promise<RerankResponse>;
  dispose(): Promise<void>;
}
export interface LoadRerankerOptions {
  policy?: Partial<RerankerPolicy>;
}
/** Server policies are resolved per discovered family before loading. */
export interface RerankerFamilyOptions {
  /** Compatibility alias for families.jina_reranker.policy. */
  policy?: Partial<JinaRerankerPolicy>;
  families?: {
    jina_reranker?: { policy?: Partial<JinaRerankerPolicy> };
    xlm_roberta_reranker?: { policy?: Partial<XlmRobertaRerankerPolicy> };
  };
}
