import type { RerankerFamily } from './types.js';

/** Structural dispatch only; loaders perform full config and tensor validation. */
export const rerankerFamilies = Object.freeze([
  Object.freeze({ family: 'jina_reranker', architecture: 'JinaForRanking' }),
  Object.freeze({ family: 'xlm_roberta_reranker', architecture: 'XLMRobertaForSequenceClassification' }),
] as const);
export function rerankerFamily(config: unknown): RerankerFamily | undefined {
  if (!config || typeof config !== 'object' || !('architectures' in config) || !Array.isArray(config.architectures))
    return undefined;
  const architectures = config.architectures;
  const matches = rerankerFamilies.filter((f) => architectures.includes(f.architecture));
  return matches.length === 1 ? matches[0].family : undefined;
}
