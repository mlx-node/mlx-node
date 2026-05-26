// Mock AttentionRun source used by the Attention chapter while the real
// backend hook is being built in parallel. The shape MUST validate against
// inspector-types.ts so that swap-in to the real worker call is mechanical.

import type {
  AttentionLayer,
  AttentionRun,
  ModelMeta,
  TokenInfo,
} from "../../src/inspector-types";

// A short, deterministic example sentence the chapter visualizes.
// We pretend the tokenizer split it like Qwen3's BPE would: leading-space
// markers ("Ġ") are preserved verbatim so TokenStrip can render them.
const MOCK_TOKENS: TokenInfo[] = [
  { id: 9876, text: "The" },
  { id: 1402, text: " cat" },
  { id: 4513, text: " sat" },
  { id: 1119, text: " on" },
  { id: 1170, text: " the" },
  { id: 1576, text: " mat" },
  { id: 13, text: "." },
  { id: 1163, text: " It" },
  { id: 6960, text: " purred" },
  { id: 2407, text: " softly" },
  { id: 11, text: "," },
  { id: 2614, text: " then" },
];

const MOCK_GENERATED_TOKEN: TokenInfo = { id: 13130, text: " slept" };

const SEQ_LEN = MOCK_TOKENS.length;
const NUM_HEADS = 4;

// Deterministic pseudo-random number for layer/head/i/j — good enough to give
// each head a visibly different pattern. We do not want Math.random because
// the mock should look identical on every run.
function hash(layer: number, head: number, i: number, j: number): number {
  let x = layer * 9176 + head * 3137 + i * 197 + j * 13 + 17;
  x = (x ^ (x << 13)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ (x << 5)) >>> 0;
  return (x % 10000) / 10000;
}

/** Build a synthetic but plausibly attention-shaped score row for query i.
 *
 *  Properties:
 *  - causal: only positions j <= i get non-zero pre-softmax mass
 *  - per-head "personality":
 *      head 0: diagonal-biased (each token looks mostly at itself)
 *      head 1: previous-token-biased (looks at the token just before)
 *      head 2: BOS-biased (looks heavily at token 0)
 *      head 3: broad / noisy
 *  - rows sum to 1 (softmax invariant)
 */
function buildRow(
  layer: number,
  head: number,
  i: number,
  seqLen: number,
): Float32Array {
  const row = new Float32Array(seqLen);
  for (let j = 0; j <= i; j++) {
    let logit = 0;
    const noise = (hash(layer, head, i, j) - 0.5) * 0.6;
    switch (head % 4) {
      case 0:
        // Diagonal-biased
        logit = j === i ? 3.0 : -Math.abs(i - j) * 0.6;
        break;
      case 1:
        // Previous-token-biased (with a soft skew further back)
        logit = j === i - 1 ? 3.0 : j === i ? 1.0 : -Math.abs(i - 1 - j) * 0.4;
        break;
      case 2:
        // BOS / first-token-biased
        logit = j === 0 ? 2.5 : j === i ? 1.5 : -j * 0.05;
        break;
      default:
        // Broad / noisy
        logit = 0.3 - Math.abs(i - j) * 0.15;
        break;
    }
    // Stir in some layer-dependent variation so switching layers visibly
    // changes the pattern.
    logit += (layer === 0 ? 0.2 : -0.2) * (j === i ? 1 : 0);
    row[j] = logit + noise;
  }

  // Softmax over the causal prefix (j <= i). Upper triangle (j > i) stays 0.
  let max = -Infinity;
  for (let j = 0; j <= i; j++) {
    if (row[j]! > max) max = row[j]!;
  }
  let sum = 0;
  for (let j = 0; j <= i; j++) {
    const v = Math.exp(row[j]! - max);
    row[j] = v;
    sum += v;
  }
  if (sum > 0) {
    for (let j = 0; j <= i; j++) row[j] = row[j]! / sum;
  }
  return row;
}

function buildLayer(layerIndex: number, seqLen: number): AttentionLayer {
  const scores = new Float32Array(NUM_HEADS * seqLen * seqLen);
  for (let h = 0; h < NUM_HEADS; h++) {
    for (let i = 0; i < seqLen; i++) {
      const row = buildRow(layerIndex, h, i, seqLen);
      scores.set(row, h * seqLen * seqLen + i * seqLen);
    }
  }
  return {
    layerIndex,
    kind: "full",
    numHeads: NUM_HEADS,
    // GQA-ish: 2 KV heads for the 4 query heads.
    numKvHeads: 2,
    scores,
  };
}

// Two "full" attention layers exposed in the mock. Real Qwen3.5-0.8B has many
// more total layers but only a subset are full-attention; we mirror that idea
// at small scale.
const MOCK_MODEL_META: ModelMeta = {
  name: "Qwen3.5-0.8B (mock)",
  numLayers: 8,
  fullAttentionLayerIndices: [3, 7],
};

/**
 * Returns a deterministic AttentionRun for the Attention chapter.
 *
 * The `prompt` argument is accepted for API parity with the real
 * `runForInspector` call but is currently ignored — the mock always returns
 * the same fixed example. When the real backend lands, this function is
 * replaced by a call into `mlxWorker` (or removed entirely).
 */
export function makeMockAttentionRun(prompt: string): AttentionRun {
  const attention: AttentionLayer[] = MOCK_MODEL_META.fullAttentionLayerIndices.map(
    (idx) => buildLayer(idx, SEQ_LEN),
  );
  return {
    // We echo back whatever the user typed so the UI feels alive even though
    // the heatmap itself is the fixed example.
    prompt: prompt || "The cat sat on the mat. It purred softly, then",
    tokens: MOCK_TOKENS,
    generatedToken: MOCK_GENERATED_TOKEN,
    attention,
    modelMeta: MOCK_MODEL_META,
  };
}
