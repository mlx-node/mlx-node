import * as React from 'react';

import { Button } from '../../components/ui/button';
import { embed } from '../../lib/embed-client';
import { tokenize } from '../../lib/tokenizer-client';
import { DemoCallout } from '../inspector/DemoCallout';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { CosineSimilarityTool } from '../widgets/CosineSimilarityTool';
import { ShapeProblem } from '../widgets/ShapeProblem';

/**
 * Chapter 2 — Embeddings.
 *
 * Prose explains how tokens become vectors. The interactive widget tokenizes
 * a curated word list, looks up each word's first-token embedding through the
 * loaded model, runs PCA on the resulting `[N, hidden_dim]` matrix, and
 * scatters the projected coordinates colored by hand-labelled category. A
 * nearest-neighbour browser computes cosine similarity in the full
 * hidden-dim space (NOT the PCA-projected 2D space — PCA throws away most
 * of the structure).
 *
 * Console logs use the `[embeddings-demo]` prefix.
 */

// -----------------------------------------------------------------------------
// Curated word list
// -----------------------------------------------------------------------------
//
// Each word is prefixed with a leading space because Qwen3's BPE tokenizer
// usually emits a single token for `" word"` and 2+ tokens for the bare
// `"word"` (chapter 1 covers this). Picking the first token id gives us the
// closest single-vector representation for each word.

type Category = 'animal' | 'number' | 'color' | 'country' | 'verb' | 'food';

type CuratedWord = { word: string; category: Category };

const WORDS: CuratedWord[] = [
  // Animals (24)
  ...[
    'dog',
    'cat',
    'fish',
    'bird',
    'lion',
    'tiger',
    'elephant',
    'horse',
    'cow',
    'sheep',
    'pig',
    'rabbit',
    'wolf',
    'fox',
    'bear',
    'deer',
    'mouse',
    'rat',
    'snake',
    'frog',
    'shark',
    'whale',
    'eagle',
    'owl',
  ].map((word) => ({ word, category: 'animal' as const })),
  // Numbers (16) — written out plus a few digits
  ...[
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'twenty',
    'fifty',
    'hundred',
    'thousand',
  ].map((word) => ({ word, category: 'number' as const })),
  // Colors (14)
  ...[
    'red',
    'blue',
    'green',
    'yellow',
    'orange',
    'purple',
    'pink',
    'black',
    'white',
    'brown',
    'gray',
    'violet',
    'crimson',
    'magenta',
  ].map((word) => ({ word, category: 'color' as const })),
  // Countries (18)
  ...[
    'Japan',
    'France',
    'Brazil',
    'Germany',
    'China',
    'India',
    'Mexico',
    'Canada',
    'Italy',
    'Spain',
    'Russia',
    'Egypt',
    'Nigeria',
    'Kenya',
    'Australia',
    'Argentina',
    'Sweden',
    'Norway',
  ].map((word) => ({ word, category: 'country' as const })),
  // Verbs (20)
  ...[
    'run',
    'walk',
    'jump',
    'sing',
    'dance',
    'swim',
    'eat',
    'drink',
    'sleep',
    'read',
    'write',
    'talk',
    'laugh',
    'cry',
    'drive',
    'fly',
    'build',
    'break',
    'make',
    'buy',
  ].map((word) => ({ word, category: 'verb' as const })),
  // Food (18)
  ...[
    'apple',
    'pizza',
    'bread',
    'rice',
    'pasta',
    'cheese',
    'butter',
    'egg',
    'soup',
    'salad',
    'burger',
    'cake',
    'cookie',
    'chocolate',
    'coffee',
    'tea',
    'milk',
    'sugar',
  ].map((word) => ({ word, category: 'food' as const })),
];

const CATEGORY_COLORS: Record<Category, string> = {
  animal: '#ef4444', // red-500
  number: '#3b82f6', // blue-500
  color: '#a855f7', // purple-500
  country: '#10b981', // emerald-500
  verb: '#f59e0b', // amber-500
  food: '#ec4899', // pink-500
};

const CATEGORY_LABELS: Record<Category, string> = {
  animal: 'Animals',
  number: 'Numbers',
  color: 'Colors',
  country: 'Countries',
  verb: 'Verbs',
  food: 'Food',
};

const ALL_CATEGORIES: Category[] = ['animal', 'number', 'color', 'country', 'verb', 'food'];

// -----------------------------------------------------------------------------
// PCA helper
// -----------------------------------------------------------------------------
//
// We avoid materializing the `numCols x numCols` covariance matrix (4 MB for
// 1024 dims) and run power iteration directly on the centered data matrix X.
// Each iteration computes `(X^T X) v = X^T (X v)` in O(numRows * numCols) ops,
// then deflation removes the top component before extracting the second.

function pca2D(
  data: Float32Array,
  numRows: number,
  numCols: number,
): {
  coords: Array<[number, number]>;
  explainedVariance: [number, number];
  means: Float32Array;
  v1: Float32Array;
  v2: Float32Array;
} {
  if (numRows === 0 || numCols === 0) {
    return {
      coords: [],
      explainedVariance: [0, 0],
      means: new Float32Array(numCols),
      v1: new Float32Array(numCols),
      v2: new Float32Array(numCols),
    };
  }

  // Center: subtract column means in place on a copy of `data`.
  const centered = new Float32Array(data.length);
  const means = new Float32Array(numCols);
  for (let r = 0; r < numRows; r++) {
    const off = r * numCols;
    for (let c = 0; c < numCols; c++) means[c] += data[off + c]!;
  }
  for (let c = 0; c < numCols; c++) means[c] /= numRows;
  for (let r = 0; r < numRows; r++) {
    const off = r * numCols;
    for (let c = 0; c < numCols; c++) centered[off + c] = data[off + c]! - means[c]!;
  }

  const v1 = powerIteration(centered, numRows, numCols, null);
  const lambda1 = quadraticForm(centered, numRows, numCols, v1);
  // Deflate: subtract the rank-1 contribution of v1 from centered rows.
  // For row r, the projection onto v1 is dot(row, v1); subtract that times v1.
  for (let r = 0; r < numRows; r++) {
    const off = r * numCols;
    let proj = 0;
    for (let c = 0; c < numCols; c++) proj += centered[off + c]! * v1[c]!;
    for (let c = 0; c < numCols; c++) centered[off + c] -= proj * v1[c]!;
  }
  const v2 = powerIteration(centered, numRows, numCols, v1);
  const lambda2 = quadraticForm(centered, numRows, numCols, v2);

  // Project the ORIGINAL centered data onto (v1, v2). Use the still-centered
  // (pre-deflation) means rather than recomputing — `centered` has been
  // mutated by deflation above, so recompute the centered rows from `data`.
  const coords = new Array<[number, number]>(numRows);
  for (let r = 0; r < numRows; r++) {
    const off = r * numCols;
    let a = 0;
    let b = 0;
    for (let c = 0; c < numCols; c++) {
      const x = data[off + c]! - means[c]!;
      a += x * v1[c]!;
      b += x * v2[c]!;
    }
    coords[r] = [a, b];
  }
  return { coords, explainedVariance: [lambda1, lambda2], means, v1, v2 };
}

// Project a single hidden-dim embedding row into the (v1, v2) basis computed
// by `pca2D` over the curated word set. Used to plot user-added custom words
// on the SAME chart without re-running PCA.
function projectIntoBasis(
  row: Float32Array,
  basis: { means: Float32Array; v1: Float32Array; v2: Float32Array },
): [number, number] {
  let x = 0;
  let y = 0;
  const d = row.length;
  for (let i = 0; i < d; i++) {
    const centered = row[i]! - basis.means[i]!;
    x += centered * basis.v1[i]!;
    y += centered * basis.v2[i]!;
  }
  return [x, y];
}

// Power iteration on X^T X via the implicit `X^T (X v)` product. The optional
// `orthogonalTo` vector is used during the second-component extraction to
// keep v2 orthogonal to v1 in the face of f32 roundoff during deflation.
function powerIteration(
  X: Float32Array,
  numRows: number,
  numCols: number,
  orthogonalTo: Float32Array | null,
): Float32Array {
  // Seed with a deterministic non-uniform vector (uniform vectors converge
  // poorly on data whose dominant eigenvector is balanced across dims).
  const v = new Float32Array(numCols);
  for (let i = 0; i < numCols; i++) v[i] = Math.sin(i * 1.31 + 0.7);
  normalize(v);

  const Xv = new Float32Array(numRows);
  const next = new Float32Array(numCols);
  const MAX_ITER = 50;
  let prev = -Infinity;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    // Xv = X @ v
    for (let r = 0; r < numRows; r++) {
      const off = r * numCols;
      let s = 0;
      for (let c = 0; c < numCols; c++) s += X[off + c]! * v[c]!;
      Xv[r] = s;
    }
    // next = X^T @ Xv
    next.fill(0);
    for (let r = 0; r < numRows; r++) {
      const off = r * numCols;
      const xv = Xv[r]!;
      for (let c = 0; c < numCols; c++) next[c] += X[off + c]! * xv;
    }
    if (orthogonalTo) {
      // Project out the orthogonalTo direction to keep eigenvectors orthogonal.
      let proj = 0;
      for (let c = 0; c < numCols; c++) proj += next[c]! * orthogonalTo[c]!;
      for (let c = 0; c < numCols; c++) next[c] -= proj * orthogonalTo[c]!;
    }
    const norm = vectorNorm(next);
    if (norm < 1e-12) break;
    for (let c = 0; c < numCols; c++) v[c] = next[c]! / norm;
    // Rayleigh-quotient-style convergence check.
    if (Math.abs(norm - prev) / (norm + 1e-12) < 1e-5) break;
    prev = norm;
  }
  return v;
}

function quadraticForm(X: Float32Array, numRows: number, numCols: number, v: Float32Array): number {
  // Returns v^T X^T X v = ||X v||^2 — the eigenvalue under the power method.
  let s = 0;
  for (let r = 0; r < numRows; r++) {
    const off = r * numCols;
    let dot = 0;
    for (let c = 0; c < numCols; c++) dot += X[off + c]! * v[c]!;
    s += dot * dot;
  }
  return s;
}

function vectorNorm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return Math.sqrt(s);
}

function normalize(v: Float32Array): void {
  const n = vectorNorm(v);
  if (n < 1e-12) return;
  for (let i = 0; i < v.length; i++) v[i] = v[i]! / n;
}

// -----------------------------------------------------------------------------
// Nearest-neighbour: cosine similarity in the FULL hidden-dim space.
// -----------------------------------------------------------------------------

type PointRow = {
  word: string;
  category: Category;
  tokenId: number;
  norm: number;
};

function cosineSimilarity(
  embeddings: Float32Array,
  hiddenDim: number,
  a: number,
  b: number,
  normA: number,
  normB: number,
): number {
  if (normA === 0 || normB === 0) return 0;
  const offA = a * hiddenDim;
  const offB = b * hiddenDim;
  let dot = 0;
  for (let c = 0; c < hiddenDim; c++) {
    dot += embeddings[offA + c]! * embeddings[offB + c]!;
  }
  return dot / (normA * normB);
}

function topNeighbors(
  embeddings: Float32Array,
  hiddenDim: number,
  rows: PointRow[],
  selectedIdx: number,
  k: number,
): Array<{ idx: number; sim: number }> {
  const normSel = rows[selectedIdx]?.norm ?? 0;
  const scores: Array<{ idx: number; sim: number }> = [];
  for (let i = 0; i < rows.length; i++) {
    if (i === selectedIdx) continue;
    const sim = cosineSimilarity(embeddings, hiddenDim, selectedIdx, i, normSel, rows[i]!.norm);
    scores.push({ idx: i, sim });
  }
  scores.sort((a, b) => b.sim - a.sim);
  return scores.slice(0, k);
}

// =============================================================================
// Chapter body
// =============================================================================

/**
 * Scaffolding metadata for chapter 2 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[1].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'embeddings',
  objective:
    'Explain how an integer token id becomes a high-dimensional vector and why nearby vectors mean related tokens.',
  problem:
    'The transformer needs a continuous representation of each token so it can express similarity and gradients can flow.',
  minutes: 7,
  glossary: [
    {
      term: 'embedding',
      definition:
        'The vector a token id is mapped to — a row of the embedding matrix that the rest of the network operates on.',
    },
    {
      term: 'embedding matrix',
      definition:
        'Shape [vocab_size, hidden_dim]. For Qwen3.5-0.8B that is roughly 151,936 by 1,024 — about 156M parameters.',
    },
    {
      term: 'hidden_dim',
      definition: "The width of every token's vector as it flows through the model. 1,024 for Qwen3.5-0.8B.",
    },
    {
      term: 'tied embeddings',
      definition:
        'Sharing the input embedding matrix with the output lm_head so the same vector space reads the prompt and writes the next-token logit.',
    },
    {
      term: 'PCA',
      definition:
        'Principal component analysis — projects high-dimensional points onto the two directions of greatest variance to make a 2D plot.',
    },
    {
      term: 'cosine similarity',
      definition:
        'Dot product of two vectors divided by their norms. Measures direction agreement and ignores magnitude; near 1 means very similar.',
    },
  ],
  takeaways: [
    "The embedding matrix is roughly a fifth of a small model's parameters — it is not a free lookup table.",
    'PCA is good for spotting clusters but its distances lie, so compute similarity in the full hidden_dim space.',
    'Tied input/output embeddings force one matrix to do double duty, which constrains what the geometry can encode.',
  ],
  exercise: {
    prompt:
      "Click Load embeddings, then click an animal like 'tiger'. Then add the custom word 'banana' with the input box. Which existing category does its top-5 nearest-neighbour list lean toward, and why?",
    answer:
      "Banana's neighbours skew toward the food cluster (apple, bread, cheese show up high) rather than colors or animals, because the embedding encodes 'edible noun' more strongly than 'yellow object' even though banana is both.",
  },
  quiz: [
    {
      id: 'q1-why-vectors',
      prompt: 'Why does the model embed tokens as vectors instead of just feeding the integer id forward?',
      options: [
        {
          id: 'a',
          label: "Integers can't be passed through a GPU matmul.",
        },
        {
          id: 'b',
          label:
            "Integers carry no notion of similarity — adjacent ids aren't related, and the model needs a space where 'close' means 'related' so gradients can move things around.",
        },
        {
          id: 'c',
          label: 'Vectors save memory compared to the original id.',
        },
      ],
      correctId: 'b',
      explanation:
        'The whole point of an embedding is to turn discrete tokens into a continuous space the model can do geometry on. Memory actually goes up, not down.',
    },
    {
      id: 'q2-pca-distances',
      prompt:
        'Why does the demo compute nearest neighbours in the full 1024-dim space instead of the 2D PCA coordinates?',
      options: [
        {
          id: 'a',
          label:
            'The top two components only capture a few percent of the variance, so 2D distances drop most of the structure that defines similarity.',
        },
        {
          id: 'b',
          label: 'PCA distances are slower to compute than cosine in 1024 dims.',
        },
        {
          id: 'c',
          label: 'PCA only works on integer ids, not floating-point vectors.',
        },
      ],
      correctId: 'a',
      explanation:
        'PCA projects to the directions of greatest variance, but those two directions explain only a small slice of a 1024-dim cloud. The visual is useful for clusters; the distances mislead.',
    },
    {
      id: 'q3-tied-embeddings',
      prompt: "What does it mean for Qwen3.5 to 'tie' its input embeddings with the output lm_head?",
      options: [
        {
          id: 'a',
          label:
            'Both layers share the same [vocab_size, hidden_dim] matrix — one matrix reads the prompt and the same matrix turns final hidden states back into logits.',
        },
        {
          id: 'b',
          label: 'The model trains them with the same learning rate.',
        },
        {
          id: 'c',
          label: "It forces every token's embedding to have unit L2 norm.",
        },
      ],
      correctId: 'a',
      explanation:
        'Tying = literally one shared weight tensor for both directions. Saves parameters and forces a single coherent representation per token.',
    },
  ],
};

export function EmbeddingsChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>Embeddings: turning tokens into vectors</h1>
        <p>
          Tokenization gave the model a sequence of integers — one id per token. But integers carry no meaning the model
          can compute with: id <code>1234</code> isn't bigger, smaller, or more <em>similar</em> to <code>1235</code> in
          any useful way. The next step is to map each id into a continuous vector the rest of the network can do math
          on. That vector is the token's <strong>embedding</strong>.
        </p>

        <h2>Why vectors?</h2>
        <p>
          A continuous representation lets the model express degrees of similarity. The classic word2vec demo —{' '}
          <code>king − man + woman ≈ queen</code> — showed that learned embeddings can capture relationships as
          directions in the vector space. Modern LLM embeddings have <em>much</em> more entangled structure than
          word2vec did (one model serves dozens of tasks across many languages and code styles), so the clean analogy
          arithmetic is weaker — but the underlying idea is the same:{' '}
          <strong>nearby vectors mean related tokens</strong>, and the transformer layers spend their entire forward
          pass moving those vectors around in contextually useful ways.
        </p>

        <h2>The embedding table</h2>
        <p>
          The embedding lookup is a single matrix multiply (or, equivalently, a row gather): given an integer id, take
          row <code>id</code> of the embedding matrix. The matrix has shape <code>[vocab_size, hidden_dim]</code>. For
          Qwen3.5-0.8B that's about <code>151,936 × 1,024 ≈ 156 million</code> parameters in this one table — roughly
          one-fifth of the whole model.
        </p>
        <p>
          Modern LLMs (Qwen3.5 included) often <strong>tie</strong> the input embedding to the output unembedding (the{' '}
          <code>lm_head</code> that turns the final hidden state back into logits over the vocabulary). Tying these two
          matrices saves parameters, and forces the representations to be useful in both directions: the same vector
          space that <em>reads</em> the prompt also <em>writes</em> the next token's logit. The widget on the right
          plots rows from this exact shared matrix.
        </p>

        <ShapeProblem
          question="How big is Qwen3's embedding matrix? And how much does tied embeddings save?"
          formula="params = vocab_size × hidden_dim"
          values={[
            { label: 'vocab_size', value: '152,064' },
            { label: 'hidden_dim', value: '1024' },
          ]}
          answer="152,064 × 1024 ≈ 155.7M parameters — about 20% of Qwen3-0.8B's total 800M. Tied embeddings reuse this exact matrix for the output projection (unembed), saving another 155.7M. Without tying, the embedding+unembed alone would be roughly the size of a small standalone model."
        />

        <h2>PCA: a 2D window into a 1024-dim space</h2>
        <p>
          We can't see 1024 dimensions, so we project. <strong>Principal component analysis</strong> finds the
          directions of greatest variance in the data and projects onto the top two (or three). The result is the best
          linear flattening possible: the projected points preserve as much of the original spread as a 2D picture can.
        </p>
        <p>
          The catch is that "as much as possible" is still very little. The top two components of a 1024-dim cloud
          typically explain just a few percent of the total variance — the rest is in the directions we threw away. So
          PCA is great for spotting <em>clusters</em> (tokens with similar overall direction in the embedding space land
          near each other), but the <em>distances</em> between points in the picture are misleading. That's why the
          nearest-neighbour browser computes cosine similarity in the full 1024-dim space, not the projected 2D
          coordinates.
        </p>

        <h2>What you should see</h2>
        <p>
          Click <strong>Load embeddings</strong>. The widget tokenizes ~110 words from six categories (animals, numbers,
          colors, countries, verbs, food), looks up each word's first-token embedding through the model worker, and runs
          PCA in your browser. Expect to see:
        </p>
        <ul>
          <li>
            <strong>Clusters by category.</strong> Animals near animals, numbers near numbers, etc. The clusters are
            usually clear despite living in a tiny 2D slice of the 1024-dim space.
          </li>
          <li>
            <strong>Sub-structure within clusters.</strong> Numbers often arrange themselves along a rough gradient.
            Colors split into warm/cool sub-regions. Food separates savoury from sweet.
          </li>
          <li>
            <strong>Nearest neighbours that look semantic.</strong> Click any point: the top-5 neighbours in the full
            hidden-dim space appear highlighted, and they're usually category-mates plus a few surprising near-misses
            across categories.
          </li>
        </ul>

        <h2>Why cosine, not Euclidean?</h2>
        <p>
          Cosine similarity measures the <em>angle</em> between two vectors, which is what encodes semantic{' '}
          <em>direction</em> in embedding space. Euclidean distance measures the full vector difference, which conflates
          direction with magnitude. The magnitudes that come out of training depend on optimization dynamics and
          aren&apos;t semantically meaningful — two synonyms can have very different norms but nearly the same
          direction. That&apos;s why cosine is the standard for embedding similarity. Some retrieval systems do use L2
          on pre-normalised vectors, which is mathematically equivalent.
        </p>

        <h2>The scatter is for orientation, not measurement</h2>
        <p>
          The PCA scatter projects 1024-dim vectors onto just two axes, so it throws away 1022 dimensions of structure.
          Two points that look close on screen might be far apart in the real space, and vice versa. The colour groups
          still cluster (the dominant axes of variance often pick those up), but don&apos;t use the literal pixel
          distance for similarity — use the cosine number from the panel below. Treat the scatter as a navigation aid
          for finding clusters, not a ruler.
        </p>

        <h2>Cosine similarity, side by side</h2>
        <p>
          The scatter on the right is a 2D projection. The panel below skips the projection entirely: each pair&apos;s
          cosine similarity is measured in the model&apos;s full 1024-dim embedding space and shown as a bar. The
          ordering teaches the lesson — identical &gt; synonym &gt; related &gt; antonym &gt; cross-language &gt;
          unrelated.
        </p>

        <CosineSimilarityTool />

        <h2>Why so many dimensions?</h2>
        <p>
          If we tried to do this in 3D, almost every concept would collide. There simply isn&apos;t room for "animals",
          "countries", "colors", "verbs", "syntactic role", "register", "language", and a dozen other axes of meaning to
          vary independently in a three-axis space. Qwen3.5-0.8B uses <strong>1,024</strong> embedding dimensions, which
          gives the model a generous amount of "room" — each independent feature can occupy its own direction without
          crowding the others. The PCA scatter above is a 2D projection of that 1024-dim space; the clusters survive the
          projection because they really are clusters, but most of the geometry is in the axes we threw away.
        </p>

        <p className="mt-6 text-muted-foreground">
          Up next (<em>Self-attention</em>) we'll watch how these vectors actually <em>move</em> as the transformer
          pulls information between positions — the operation that gives "the cat sat on the mat" a different meaning
          from "the mat sat on the cat".
        </p>
      </Prose>
    </ChapterFrame>
  );
}

// =============================================================================
// Interactive widget
// =============================================================================

export type EmbeddingsDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus = { kind: 'idle' } | { kind: 'ok' } | { kind: 'error'; error: string } | { kind: 'aborted' };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

type LoadedScatter = {
  points: PointRow[];
  coords: Array<[number, number]>;
  explainedVariance: [number, number];
  hiddenDim: number;
  embeddings: Float32Array;
  pcaBasis: { means: Float32Array; v1: Float32Array; v2: Float32Array };
};

type CustomPoint = {
  word: string;
  tokenCount: number;
  embedding: Float32Array;
  coords: [number, number];
};

type Selected = { kind: 'builtin'; index: number } | { kind: 'custom'; index: number };

export function EmbeddingsDemo({ workerRef, abortRef }: EmbeddingsDemoProps) {
  const [scatter, setScatter] = React.useState<LoadedScatter | null>(null);
  const [status, setStatus] = React.useState<RunStatus>({ kind: 'idle' });
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState<string>('');
  const [selected, setSelected] = React.useState<Selected | null>(null);
  const [hoverIdx, setHoverIdx] = React.useState<number | null>(null);
  const [activeCategories, setActiveCategories] = React.useState<Set<Category>>(() => new Set(ALL_CATEGORIES));
  const [customPoints, setCustomPoints] = React.useState<CustomPoint[]>([]);
  const [customInput, setCustomInput] = React.useState('');
  const [customNote, setCustomNote] = React.useState<string | null>(null);
  const [customError, setCustomError] = React.useState<string | null>(null);
  const [adding, setAdding] = React.useState(false);

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);

  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
    },
    [],
  );

  async function handleLoad() {
    const myGen = ++runGenRef.current;
    setRunning(true);
    setSelected(null);
    setCustomPoints([]);
    setCustomInput('');
    setCustomNote(null);
    setCustomError(null);

    const worker = workerRef.current;
    if (!worker) {
      console.error('[embeddings-demo] worker is unavailable — chapter view should have been gated on modelReady');
      setStatus({ kind: 'error', error: 'Worker is unavailable. Reload the page.' });
      setRunning(false);
      return;
    }

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;

    const appSignal = abortRef.current?.signal;
    if (appSignal?.aborted) ctrl.abort();
    const onAppAbort = () => ctrl.abort();
    if (appSignal && !appSignal.aborted) {
      appSignal.addEventListener('abort', onAppAbort, { once: true });
    }

    try {
      setProgress(`Tokenizing ${WORDS.length} words...`);
      // Tokenize each word with a leading space (most words land as a single
      // token under Qwen3 BPE when space-prefixed). We pick the FIRST token id
      // for each word — multi-token words still get a sensible vector since
      // their first piece typically carries the semantic seed.
      const wordTokenIds: Array<number | null> = new Array(WORDS.length).fill(null);
      for (let i = 0; i < WORDS.length; i++) {
        if (runGenRef.current !== myGen) return;
        const tokens = await tokenize(worker, ' ' + WORDS[i]!.word, {
          signal: ctrl.signal,
        });
        if (tokens.length === 0) continue;
        const first = tokens[0]!.id;
        if (first < 0) continue;
        wordTokenIds[i] = first;
      }
      if (runGenRef.current !== myGen) return;

      // Filter out any words that failed to tokenize. Build a parallel
      // index map so we can stitch back the original metadata.
      const keptWords: CuratedWord[] = [];
      const keptTokenIds: number[] = [];
      for (let i = 0; i < WORDS.length; i++) {
        const tid = wordTokenIds[i];
        if (tid == null) continue;
        keptWords.push(WORDS[i]!);
        keptTokenIds.push(tid);
      }
      if (keptTokenIds.length === 0) {
        throw new Error('No tokens produced for any word');
      }

      setProgress(`Fetching ${keptTokenIds.length} embeddings from the model...`);
      const { embeddings, hiddenDim } = await embed(worker, keptTokenIds, {
        signal: ctrl.signal,
      });
      if (runGenRef.current !== myGen) return;

      setProgress(`Running PCA on [${keptWords.length} × ${hiddenDim}] matrix...`);
      // Yield to the event loop so the progress text renders before we
      // start the (short) PCA computation.
      await new Promise((r) => setTimeout(r, 0));
      const { coords, explainedVariance, means, v1, v2 } = pca2D(embeddings, keptWords.length, hiddenDim);

      // Precompute per-row L2 norms once so the neighbour browser doesn't
      // recompute them on every selection.
      const points: PointRow[] = keptWords.map((w, i) => {
        const off = i * hiddenDim;
        let s = 0;
        for (let c = 0; c < hiddenDim; c++) s += embeddings[off + c]! ** 2;
        return {
          word: w.word,
          category: w.category,
          tokenId: keptTokenIds[i]!,
          norm: Math.sqrt(s),
        };
      });

      console.log('[embeddings-demo] embed ok', {
        words: points.length,
        hiddenDim,
        explainedVariance,
      });
      setScatter({
        points,
        coords,
        explainedVariance,
        hiddenDim,
        embeddings,
        pcaBasis: { means, v1, v2 },
      });
      setStatus({ kind: 'ok' });
      setProgress('');
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info('[embeddings-demo] aborted (worker terminated or superseded)');
        if (appSignal?.aborted) setStatus({ kind: 'aborted' });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error('[embeddings-demo] embed lookup failed', err);
        setStatus({ kind: 'error', error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      if (runGenRef.current === myGen) {
        setRunning(false);
        setProgress('');
      }
    }
  }

  // Auto-fire Load on mount — the app gates chapter rendering on the model
  // being ready, so the worker is alive by this point and the user lands
  // on a populated PCA scatter instead of an empty "Click Load embeddings"
  // placeholder.
  const didAutoLoadRef = React.useRef(false);
  React.useEffect(() => {
    if (didAutoLoadRef.current) return;
    didAutoLoadRef.current = true;
    void handleLoad();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Per-custom-point L2 norms in the full hidden-dim space, recomputed when
  // the custom-points list changes. Built-in norms are precomputed on load.
  const customNorms = React.useMemo(() => {
    return customPoints.map((p) => {
      let s = 0;
      const emb = p.embedding;
      for (let i = 0; i < emb.length; i++) s += emb[i]! * emb[i]!;
      return Math.sqrt(s);
    });
  }, [customPoints]);

  // Neighbour search runs over the UNION of built-in + custom points in the
  // full 1024-dim space. Returns refs into either set via the `kind` tag.
  type NeighborRef = { kind: 'builtin'; index: number; sim: number } | { kind: 'custom'; index: number; sim: number };
  const neighbors = React.useMemo<NeighborRef[]>(() => {
    if (!scatter || selected == null) return [];
    const hiddenDim = scatter.hiddenDim;
    // Source vector + its norm.
    let srcOffset = -1;
    let srcEmbedding: Float32Array | null = null;
    let srcNorm = 0;
    if (selected.kind === 'builtin') {
      const row = scatter.points[selected.index];
      if (!row) return [];
      srcOffset = selected.index * hiddenDim;
      srcEmbedding = scatter.embeddings;
      srcNorm = row.norm;
    } else {
      const p = customPoints[selected.index];
      if (!p) return [];
      srcEmbedding = p.embedding;
      srcOffset = 0;
      srcNorm = customNorms[selected.index] ?? 0;
    }
    if (srcNorm === 0 || srcEmbedding == null) return [];

    const scores: NeighborRef[] = [];
    // Compare against built-in points.
    for (let i = 0; i < scatter.points.length; i++) {
      if (selected.kind === 'builtin' && selected.index === i) continue;
      const normB = scatter.points[i]!.norm;
      if (normB === 0) continue;
      const offB = i * hiddenDim;
      let dot = 0;
      for (let c = 0; c < hiddenDim; c++) {
        dot += srcEmbedding[srcOffset + c]! * scatter.embeddings[offB + c]!;
      }
      scores.push({
        kind: 'builtin',
        index: i,
        sim: dot / (srcNorm * normB),
      });
    }
    // Compare against custom points.
    for (let i = 0; i < customPoints.length; i++) {
      if (selected.kind === 'custom' && selected.index === i) continue;
      const normB = customNorms[i] ?? 0;
      if (normB === 0) continue;
      const emb = customPoints[i]!.embedding;
      let dot = 0;
      for (let c = 0; c < hiddenDim; c++) {
        dot += srcEmbedding[srcOffset + c]! * emb[c]!;
      }
      scores.push({
        kind: 'custom',
        index: i,
        sim: dot / (srcNorm * normB),
      });
    }
    scores.sort((a, b) => b.sim - a.sim);
    return scores.slice(0, 5);
  }, [scatter, selected, customPoints, customNorms]);

  // For visual highlighting of neighbour points in the SVG. Use a string-key
  // set so we can tag custom vs built-in without index collisions.
  const neighborKeySet = React.useMemo(() => new Set(neighbors.map((n) => `${n.kind}:${n.index}`)), [neighbors]);

  function toggleCategory(cat: Category) {
    setActiveCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  async function handleAddCustom() {
    const trimmed = customInput.trim();
    if (trimmed.length === 0) return;
    if (!scatter) return;

    // Duplicate check (case-insensitive against existing custom points). If
    // the word is already plotted, just re-select it instead of double-adding.
    const lowered = trimmed.toLowerCase();
    const existingIdx = customPoints.findIndex((p) => p.word.toLowerCase() === lowered);
    if (existingIdx >= 0) {
      setSelected({ kind: 'custom', index: existingIdx });
      return;
    }

    const worker = workerRef.current;
    if (!worker) {
      setCustomError('Worker is unavailable. Reload the page.');
      return;
    }

    setAdding(true);
    setCustomError(null);
    setCustomNote(null);

    try {
      const tokens = await tokenize(worker, ' ' + trimmed);
      if (tokens.length === 0) {
        setCustomError("Couldn't tokenize that input.");
        return;
      }
      if (tokens.length > 1) {
        setCustomNote(`"${trimmed}" tokenizes as ${tokens.length} pieces — plotting the first.`);
      } else {
        setCustomNote(null);
      }

      let embedResult: { embeddings: Float32Array; hiddenDim: number };
      try {
        embedResult = await embed(worker, [tokens[0]!.id]);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setCustomError(message);
        return;
      }

      const row = embedResult.embeddings.slice(0, embedResult.hiddenDim);
      const coords = projectIntoBasis(row, scatter.pcaBasis);

      setCustomPoints((prev) => [
        ...prev,
        {
          word: trimmed,
          tokenCount: tokens.length,
          embedding: row,
          coords,
        },
      ]);
      setCustomInput('');
    } finally {
      setAdding(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          onClick={handleLoad}
          disabled={running}
          title="Tokenizes ~110 words, looks up embeddings, runs PCA in browser"
        >
          {running ? 'Loading…' : scatter ? 'Reload embeddings' : 'Load embeddings'}
        </Button>
        <span className="text-xs text-muted-foreground">
          {progress
            ? progress
            : scatter
              ? `${scatter.points.length} words · ${scatter.hiddenDim}-dim embeddings`
              : "Tokenize a curated word list, fetch the model's embeddings, project to 2D with PCA."}
        </span>
      </div>

      {status.kind === 'error' ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Embed lookup failed.</strong> {status.error}
        </div>
      ) : null}
      {status.kind === 'aborted' ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Request cancelled — model was reloaded. Click <strong>Reload embeddings</strong> to retry.
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {ALL_CATEGORIES.map((cat) => {
          const active = activeCategories.has(cat);
          return (
            <button
              key={cat}
              type="button"
              onClick={() => toggleCategory(cat)}
              className={[
                'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
                active
                  ? 'border-foreground/20 bg-background'
                  : 'border-dashed border-muted-foreground/40 bg-muted/40 text-muted-foreground line-through',
              ].join(' ')}
            >
              <span
                aria-hidden="true"
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: CATEGORY_COLORS[cat] }}
              />
              {CATEGORY_LABELS[cat]}
            </button>
          );
        })}
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Try your own word — e.g. 'banana'"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                void handleAddCustom();
              }
            }}
            disabled={adding || !scatter}
            className="flex h-8 w-full rounded-md border border-input bg-transparent px-3 py-1 text-xs font-mono shadow-xs outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-input/30"
          />
          <Button size="sm" onClick={() => void handleAddCustom()} disabled={adding || !customInput.trim() || !scatter}>
            {adding ? 'Adding…' : 'Add'}
          </Button>
          {customPoints.length > 0 ? (
            <button
              type="button"
              onClick={() => {
                setCustomPoints([]);
                setSelected(null);
              }}
              className="text-xs text-muted-foreground hover:text-foreground underline-offset-2 hover:underline"
            >
              Clear ({customPoints.length})
            </button>
          ) : null}
        </div>
        {customNote ? <p className="text-xs text-muted-foreground">{customNote}</p> : null}
        {customError ? <p className="text-xs text-destructive">{customError}</p> : null}
      </div>

      {scatter ? (
        <ScatterPlot
          scatter={scatter}
          selected={selected}
          hoverIdx={hoverIdx}
          neighborKeySet={neighborKeySet}
          activeCategories={activeCategories}
          customPoints={customPoints}
          onSelect={(sel) => setSelected(sel)}
          onHover={(idx) => setHoverIdx(idx)}
        />
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Load embeddings</strong> to project Qwen3.5's embedding matrix to 2D.
        </div>
      )}

      {scatter && selected != null ? (
        <NeighborList scatter={scatter} selected={selected} customPoints={customPoints} neighbors={neighbors} />
      ) : scatter ? (
        <div className="rounded-md bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          Click a point in the scatter to see its top-5 nearest neighbours (cosine similarity in the full{' '}
          {scatter.hiddenDim}-dim space).
        </div>
      ) : null}

      <DemoCallout
        items={[
          'Click a dot in the scatter to pin it and reveal its top-5 nearest neighbours by cosine similarity.',
          "Tokens from the same colour group tend to cluster together — that's the embedding space organising itself.",
          'Add a custom word with the input field above; it will land near the cluster that fits its meaning.',
        ]}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
// Scatter plot (SVG)
// -----------------------------------------------------------------------------

const SCATTER_WIDTH = 560;
const SCATTER_HEIGHT = 380;
const SCATTER_PADDING = 28;

function ScatterPlot({
  scatter,
  selected,
  hoverIdx,
  neighborKeySet,
  activeCategories,
  customPoints,
  onSelect,
  onHover,
}: {
  scatter: LoadedScatter;
  selected: Selected | null;
  hoverIdx: number | null;
  neighborKeySet: Set<string>;
  activeCategories: Set<Category>;
  customPoints: CustomPoint[];
  onSelect: (sel: Selected) => void;
  onHover: (idx: number | null) => void;
}) {
  const { coords, points } = scatter;
  const selectedBuiltinIdx = selected?.kind === 'builtin' ? selected.index : null;
  const bounds = React.useMemo(() => {
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const [x, y] of coords) {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    // Custom points must fit too — extend the bounds to include them so a
    // user-typed outlier doesn't get clipped off the SVG.
    for (const p of customPoints) {
      const [x, y] = p.coords;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    // Degenerate cases (1 point or all colinear) — pad so we don't divide by 0.
    if (xMax - xMin < 1e-6) {
      xMin -= 0.5;
      xMax += 0.5;
    }
    if (yMax - yMin < 1e-6) {
      yMin -= 0.5;
      yMax += 0.5;
    }
    return { xMin, xMax, yMin, yMax };
  }, [coords, customPoints]);

  function project(x: number, y: number): [number, number] {
    const w = SCATTER_WIDTH - 2 * SCATTER_PADDING;
    const h = SCATTER_HEIGHT - 2 * SCATTER_PADDING;
    const px = SCATTER_PADDING + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * w;
    // Flip y so positive PC2 points up in the visual.
    const py = SCATTER_PADDING + h - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * h;
    return [px, py];
  }

  const hasFocus = hoverIdx != null || selected != null;

  return (
    <div className="rounded-md border border-border bg-background p-2">
      <svg
        role="img"
        aria-label="PCA scatter of token embeddings"
        viewBox={`0 0 ${SCATTER_WIDTH} ${SCATTER_HEIGHT}`}
        className="block w-full h-auto"
      >
        {/* Axes — neutral cross at the origin if it falls inside the bounds. */}
        {bounds.xMin <= 0 && bounds.xMax >= 0
          ? (() => {
              const [x, _y] = project(0, bounds.yMin);
              const [_x2, y2] = project(0, bounds.yMax);
              return (
                <line
                  x1={x}
                  x2={x}
                  y1={SCATTER_PADDING}
                  y2={SCATTER_HEIGHT - SCATTER_PADDING}
                  stroke="currentColor"
                  strokeOpacity="0.08"
                  strokeDasharray="2 4"
                />
              );
            })()
          : null}
        {bounds.yMin <= 0 && bounds.yMax >= 0
          ? (() => {
              const [_x, y] = project(bounds.xMin, 0);
              return (
                <line
                  y1={y}
                  y2={y}
                  x1={SCATTER_PADDING}
                  x2={SCATTER_WIDTH - SCATTER_PADDING}
                  stroke="currentColor"
                  strokeOpacity="0.08"
                  strokeDasharray="2 4"
                />
              );
            })()
          : null}

        {/* Axis labels */}
        <text
          x={SCATTER_WIDTH - SCATTER_PADDING}
          y={SCATTER_HEIGHT - 8}
          textAnchor="end"
          className="fill-muted-foreground"
          fontSize="10"
        >
          PC1
        </text>
        <text x={8} y={SCATTER_PADDING - 6} className="fill-muted-foreground" fontSize="10">
          PC2
        </text>

        {coords.map(([x, y], i) => {
          const p = points[i]!;
          const visible = activeCategories.has(p.category);
          if (!visible) return null;
          const [px, py] = project(x, y);
          const isSelected = i === selectedBuiltinIdx;
          const isHover = i === hoverIdx;
          const isNeighbor = neighborKeySet.has(`builtin:${i}`);
          const r = isSelected ? 6 : isHover ? 5 : isNeighbor ? 4.5 : 3;
          const stroke = isSelected ? 'var(--foreground)' : isNeighbor ? 'var(--foreground)' : 'none';
          const strokeWidth = isSelected ? 2 : isNeighbor ? 1.5 : 0;
          return (
            <g key={i}>
              <circle
                cx={px}
                cy={py}
                r={r}
                fill={CATEGORY_COLORS[p.category]}
                fillOpacity={hasFocus && !isSelected && !isNeighbor && !isHover ? 0.4 : 0.9}
                stroke={stroke}
                strokeWidth={strokeWidth}
                onClick={() => onSelect({ kind: 'builtin', index: i })}
                onMouseEnter={() => onHover(i)}
                onMouseLeave={() => onHover(null)}
                style={{ cursor: 'pointer' }}
              >
                <title>
                  {p.word} · {CATEGORY_LABELS[p.category]}
                  {p.tokenId >= 0 ? ` · id ${p.tokenId}` : ''}
                </title>
              </circle>
              {isSelected || isHover || isNeighbor ? (
                <text x={px + 8} y={py + 3} fontSize="10" className="pointer-events-none fill-foreground">
                  {p.word}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* Custom user-added points (rendered on top so they don't get
            occluded by built-in clusters). */}
        {customPoints.map((p, i) => {
          const [px, py] = project(p.coords[0], p.coords[1]);
          const isSelected = selected?.kind === 'custom' && selected.index === i;
          const isNeighbor = neighborKeySet.has(`custom:${i}`);
          return (
            <g key={`custom-${i}`} onClick={() => onSelect({ kind: 'custom', index: i })} className="cursor-pointer">
              {/* outer halo ring */}
              <circle
                cx={px}
                cy={py}
                r={9}
                fill="none"
                stroke="var(--foreground)"
                strokeOpacity={isSelected || isNeighbor ? 0.6 : 0.35}
                strokeWidth={1}
              />
              {/* main marker — neutral fill to distinguish from category colors */}
              <circle
                cx={px}
                cy={py}
                r={5}
                fill="var(--background)"
                stroke="var(--foreground)"
                strokeWidth={isSelected ? 2.5 : 1.5}
              >
                <title>
                  {p.word}
                  {p.tokenCount > 1 ? ` · ${p.tokenCount} tokens (plotting first)` : ''}
                </title>
              </circle>
              <text x={px + 8} y={py + 4} fontSize="11" className="pointer-events-none select-none fill-foreground">
                {p.word}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex items-center justify-between gap-2 px-2 pb-1 font-mono text-[10px] text-muted-foreground">
        <span>
          PC1={scatter.explainedVariance[0].toExponential(2)}, PC2=
          {scatter.explainedVariance[1].toExponential(2)} (eigenvalues)
        </span>
        <span>click a point for nearest neighbours</span>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Neighbor list
// -----------------------------------------------------------------------------

function NeighborList({
  scatter,
  selected,
  customPoints,
  neighbors,
}: {
  scatter: LoadedScatter;
  selected: Selected;
  customPoints: CustomPoint[];
  neighbors: Array<{ kind: 'builtin'; index: number; sim: number } | { kind: 'custom'; index: number; sim: number }>;
}) {
  // Resolve the selected point's display metadata. Built-in points carry a
  // category and tokenId; custom points are neutral and tagged "(custom)".
  let selWord: string;
  let selSwatch: string;
  let selSuffix: string;
  if (selected.kind === 'builtin') {
    const sel = scatter.points[selected.index];
    if (!sel) return null;
    selWord = sel.word;
    selSwatch = CATEGORY_COLORS[sel.category];
    selSuffix = `${CATEGORY_LABELS[sel.category]}${sel.tokenId >= 0 ? ` · id ${sel.tokenId}` : ''}`;
  } else {
    const sel = customPoints[selected.index];
    if (!sel) return null;
    selWord = sel.word;
    selSwatch = 'var(--foreground)';
    selSuffix = sel.tokenCount > 1 ? `custom · ${sel.tokenCount} tokens (first plotted)` : 'custom';
  }
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 flex items-center gap-2 text-xs">
        <span
          aria-hidden="true"
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: selSwatch }}
        />
        <span className="font-mono text-foreground">{selWord}</span>
        <span className="text-muted-foreground">({selSuffix})</span>
        <span className="ml-auto text-muted-foreground">top-5 nearest in {scatter.hiddenDim}-dim cosine</span>
      </div>
      <ol className="space-y-1">
        {neighbors.map((n, rank) => {
          let word: string;
          let swatch: string;
          let suffix: string;
          let key: string;
          if (n.kind === 'builtin') {
            const p = scatter.points[n.index]!;
            word = p.word;
            swatch = CATEGORY_COLORS[p.category];
            suffix = CATEGORY_LABELS[p.category];
            key = `builtin-${n.index}`;
          } else {
            const p = customPoints[n.index]!;
            word = p.word;
            swatch = 'var(--foreground)';
            suffix = 'custom';
            key = `custom-${n.index}`;
          }
          return (
            <li key={key} className="grid grid-cols-[1.5rem_auto_minmax(0,1fr)_4rem] items-center gap-2 text-xs">
              <span className="font-mono text-muted-foreground">#{rank + 1}</span>
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: swatch }}
              />
              <span className="truncate">
                <span className="font-mono">{word}</span>
                <span className="ml-2 text-muted-foreground">{suffix}</span>
              </span>
              <span className="text-right font-mono text-foreground/80">{n.sim.toFixed(3)}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
