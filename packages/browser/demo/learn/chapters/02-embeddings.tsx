import * as React from "react";

import { Button } from "../../components/ui/button";
import { embed } from "../../lib/embed-client";
import { tokenize } from "../../lib/tokenizer-client";
import { Prose } from "../Prose";

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
 * When the model isn't loaded yet, a mock scatter renders with pre-tuned
 * positions so the lesson is still legible. Console logs use the
 * `[embeddings-demo]` prefix.
 */

// -----------------------------------------------------------------------------
// Curated word list
// -----------------------------------------------------------------------------
//
// Each word is prefixed with a leading space because Qwen3's BPE tokenizer
// usually emits a single token for `" word"` and 2+ tokens for the bare
// `"word"` (chapter 1 covers this). Picking the first token id gives us the
// closest single-vector representation for each word.

type Category =
  | "animal"
  | "number"
  | "color"
  | "country"
  | "verb"
  | "food";

type CuratedWord = { word: string; category: Category };

const WORDS: CuratedWord[] = [
  // Animals (24)
  ...[
    "dog",
    "cat",
    "fish",
    "bird",
    "lion",
    "tiger",
    "elephant",
    "horse",
    "cow",
    "sheep",
    "pig",
    "rabbit",
    "wolf",
    "fox",
    "bear",
    "deer",
    "mouse",
    "rat",
    "snake",
    "frog",
    "shark",
    "whale",
    "eagle",
    "owl",
  ].map((word) => ({ word, category: "animal" as const })),
  // Numbers (16) — written out plus a few digits
  ...[
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "twenty",
    "fifty",
    "hundred",
    "thousand",
  ].map((word) => ({ word, category: "number" as const })),
  // Colors (14)
  ...[
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "brown",
    "gray",
    "violet",
    "crimson",
    "magenta",
  ].map((word) => ({ word, category: "color" as const })),
  // Countries (18)
  ...[
    "Japan",
    "France",
    "Brazil",
    "Germany",
    "China",
    "India",
    "Mexico",
    "Canada",
    "Italy",
    "Spain",
    "Russia",
    "Egypt",
    "Nigeria",
    "Kenya",
    "Australia",
    "Argentina",
    "Sweden",
    "Norway",
  ].map((word) => ({ word, category: "country" as const })),
  // Verbs (20)
  ...[
    "run",
    "walk",
    "jump",
    "sing",
    "dance",
    "swim",
    "eat",
    "drink",
    "sleep",
    "read",
    "write",
    "talk",
    "laugh",
    "cry",
    "drive",
    "fly",
    "build",
    "break",
    "make",
    "buy",
  ].map((word) => ({ word, category: "verb" as const })),
  // Food (18)
  ...[
    "apple",
    "pizza",
    "bread",
    "rice",
    "pasta",
    "cheese",
    "butter",
    "egg",
    "soup",
    "salad",
    "burger",
    "cake",
    "cookie",
    "chocolate",
    "coffee",
    "tea",
    "milk",
    "sugar",
  ].map((word) => ({ word, category: "food" as const })),
];

const CATEGORY_COLORS: Record<Category, string> = {
  animal: "#ef4444", // red-500
  number: "#3b82f6", // blue-500
  color: "#a855f7", // purple-500
  country: "#10b981", // emerald-500
  verb: "#f59e0b", // amber-500
  food: "#ec4899", // pink-500
};

const CATEGORY_LABELS: Record<Category, string> = {
  animal: "Animals",
  number: "Numbers",
  color: "Colors",
  country: "Countries",
  verb: "Verbs",
  food: "Food",
};

const ALL_CATEGORIES: Category[] = [
  "animal",
  "number",
  "color",
  "country",
  "verb",
  "food",
];

// -----------------------------------------------------------------------------
// Mock data — pre-tuned 2D positions clustered by category for the fallback
// scatter when the model isn't loaded. The numeric structure mirrors what a
// real PCA projection tends to look like: clear inter-category separation,
// some intra-category spread.
// -----------------------------------------------------------------------------

type MockPoint = { word: string; category: Category; x: number; y: number };

const MOCK_SCATTER: MockPoint[] = [
  // Animals cluster (top-left)
  { word: "dog", category: "animal", x: -2.3, y: 1.4 },
  { word: "cat", category: "animal", x: -2.1, y: 1.6 },
  { word: "fish", category: "animal", x: -2.6, y: 1.0 },
  { word: "bird", category: "animal", x: -1.9, y: 1.8 },
  { word: "lion", category: "animal", x: -2.4, y: 1.2 },
  { word: "horse", category: "animal", x: -2.0, y: 1.5 },
  // Numbers cluster (top-right)
  { word: "one", category: "number", x: 2.0, y: 1.8 },
  { word: "two", category: "number", x: 2.2, y: 1.6 },
  { word: "three", category: "number", x: 2.1, y: 1.7 },
  { word: "ten", category: "number", x: 2.3, y: 1.5 },
  { word: "hundred", category: "number", x: 2.5, y: 1.3 },
  // Colors cluster (right)
  { word: "red", category: "color", x: 1.8, y: -0.2 },
  { word: "blue", category: "color", x: 2.0, y: 0.0 },
  { word: "green", category: "color", x: 1.7, y: -0.4 },
  { word: "yellow", category: "color", x: 2.1, y: -0.1 },
  { word: "purple", category: "color", x: 1.9, y: 0.1 },
  // Countries cluster (bottom-right)
  { word: "Japan", category: "country", x: 1.4, y: -1.6 },
  { word: "France", category: "country", x: 1.6, y: -1.8 },
  { word: "Brazil", category: "country", x: 1.2, y: -1.5 },
  { word: "China", category: "country", x: 1.5, y: -1.7 },
  // Verbs cluster (bottom-left)
  { word: "run", category: "verb", x: -1.6, y: -1.5 },
  { word: "walk", category: "verb", x: -1.4, y: -1.7 },
  { word: "jump", category: "verb", x: -1.8, y: -1.3 },
  { word: "sing", category: "verb", x: -1.5, y: -1.9 },
  // Food cluster (left)
  { word: "apple", category: "food", x: -2.4, y: -0.3 },
  { word: "pizza", category: "food", x: -2.2, y: -0.5 },
  { word: "bread", category: "food", x: -2.5, y: -0.1 },
  { word: "rice", category: "food", x: -2.0, y: -0.4 },
  { word: "cake", category: "food", x: -2.3, y: -0.6 },
  { word: "coffee", category: "food", x: -2.1, y: -0.2 },
];

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
} {
  if (numRows === 0 || numCols === 0) {
    return { coords: [], explainedVariance: [0, 0] };
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
  return { coords, explainedVariance: [lambda1, lambda2] };
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

function quadraticForm(
  X: Float32Array,
  numRows: number,
  numCols: number,
  v: Float32Array,
): number {
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
    const sim = cosineSimilarity(
      embeddings,
      hiddenDim,
      selectedIdx,
      i,
      normSel,
      rows[i]!.norm,
    );
    scores.push({ idx: i, sim });
  }
  scores.sort((a, b) => b.sim - a.sim);
  return scores.slice(0, k);
}

// =============================================================================
// Chapter body
// =============================================================================

export function EmbeddingsChapterBody() {
  return (
    <Prose>
      <h1>Embeddings: turning tokens into vectors</h1>
      <p>
        Tokenization gave the model a sequence of integers — one id per
        token. But integers carry no meaning the model can compute with: id{" "}
        <code>1234</code> isn't bigger, smaller, or more <em>similar</em> to{" "}
        <code>1235</code> in any useful way. The next step is to map each id
        into a continuous vector the rest of the network can do math on.
        That vector is the token's <strong>embedding</strong>.
      </p>

      <h2>Why vectors?</h2>
      <p>
        A continuous representation lets the model express degrees of
        similarity. The classic word2vec demo —{" "}
        <code>king − man + woman ≈ queen</code> — showed that learned
        embeddings can capture relationships as directions in the vector
        space. Modern LLM embeddings have <em>much</em> more entangled
        structure than word2vec did (one model serves dozens of tasks across
        many languages and code styles), so the clean analogy arithmetic is
        weaker — but the underlying idea is the same: <strong>nearby
        vectors mean related tokens</strong>, and the transformer layers
        spend their entire forward pass moving those vectors around in
        contextually useful ways.
      </p>

      <h2>The embedding table</h2>
      <p>
        The embedding lookup is a single matrix multiply (or, equivalently, a
        row gather): given an integer id, take row <code>id</code> of the
        embedding matrix. The matrix has shape{" "}
        <code>[vocab_size, hidden_dim]</code>. For Qwen3.5-0.8B that's about{" "}
        <code>151,936 × 1,024 ≈ 156 million</code> parameters in this one
        table — roughly one-fifth of the whole model.
      </p>
      <p>
        Modern LLMs (Qwen3.5 included) often <strong>tie</strong> the input
        embedding to the output unembedding (the <code>lm_head</code> that
        turns the final hidden state back into logits over the vocabulary).
        Tying these two matrices saves parameters, and forces the
        representations to be useful in both directions: the same vector
        space that <em>reads</em> the prompt also <em>writes</em> the next
        token's logit. The widget on the right plots rows from this exact
        shared matrix.
      </p>

      <h2>PCA: a 2D window into a 1024-dim space</h2>
      <p>
        We can't see 1024 dimensions, so we project. <strong>Principal
        component analysis</strong> finds the directions of greatest variance
        in the data and projects onto the top two (or three). The result is
        the best linear flattening possible: the projected points preserve
        as much of the original spread as a 2D picture can.
      </p>
      <p>
        The catch is that "as much as possible" is still very little. The
        top two components of a 1024-dim cloud typically explain just a few
        percent of the total variance — the rest is in the directions we
        threw away. So PCA is great for spotting <em>clusters</em> (tokens
        with similar overall direction in the embedding space land near each
        other), but the <em>distances</em> between points in the picture
        are misleading. That's why the nearest-neighbour browser computes
        cosine similarity in the full 1024-dim space, not the projected 2D
        coordinates.
      </p>

      <h2>What you should see</h2>
      <p>
        Click <strong>Load embeddings</strong>. The widget tokenizes ~110
        words from six categories (animals, numbers, colors, countries,
        verbs, food), looks up each word's first-token embedding through the
        model worker, and runs PCA in your browser. Expect to see:
      </p>
      <ul>
        <li>
          <strong>Clusters by category.</strong> Animals near animals,
          numbers near numbers, etc. The clusters are usually clear despite
          living in a tiny 2D slice of the 1024-dim space.
        </li>
        <li>
          <strong>Sub-structure within clusters.</strong> Numbers often
          arrange themselves along a rough gradient. Colors split into
          warm/cool sub-regions. Food separates savoury from sweet.
        </li>
        <li>
          <strong>Nearest neighbours that look semantic.</strong> Click any
          point: the top-5 neighbours in the full hidden-dim space appear
          highlighted, and they're usually category-mates plus a few
          surprising near-misses across categories.
        </li>
      </ul>

      <p className="mt-6 text-muted-foreground">
        Up next (<em>Self-attention</em>) we'll watch how these vectors
        actually <em>move</em> as the transformer pulls information between
        positions — the operation that gives "the cat sat on the mat" a
        different meaning from "the mat sat on the cat".
      </p>
    </Prose>
  );
}

// =============================================================================
// Interactive widget
// =============================================================================

export type EmbeddingsDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus =
  | { kind: "idle" }
  | { kind: "ok" }
  | { kind: "mock-no-worker" }
  | { kind: "mock-error"; error: string }
  | { kind: "aborted" };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

type LoadedScatter = {
  points: PointRow[];
  coords: Array<[number, number]>;
  explainedVariance: [number, number];
  hiddenDim: number;
  embeddings: Float32Array;
  isMock: boolean;
};

function makeMockScatter(): LoadedScatter {
  const points: PointRow[] = MOCK_SCATTER.map((p, i) => ({
    word: p.word,
    category: p.category,
    tokenId: -1 - i,
    norm: 1,
  }));
  // Fake embeddings derived from the mock 2D positions so the
  // nearest-neighbour computation has *something* sensible to do under the
  // mock fallback. We tile each (x, y) across a small hidden dim and add a
  // tiny per-category offset so neighbours cluster by category.
  const hiddenDim = 16;
  const embeddings = new Float32Array(points.length * hiddenDim);
  for (let i = 0; i < MOCK_SCATTER.length; i++) {
    const { x, y, category } = MOCK_SCATTER[i]!;
    const off = i * hiddenDim;
    const catSeed = ALL_CATEGORIES.indexOf(category);
    for (let c = 0; c < hiddenDim; c++) {
      const phase = (c * 0.7 + catSeed) % 6.28;
      embeddings[off + c] = x * Math.cos(phase) + y * Math.sin(phase);
    }
  }
  // Recompute norms.
  for (let i = 0; i < points.length; i++) {
    const off = i * hiddenDim;
    let s = 0;
    for (let c = 0; c < hiddenDim; c++) s += embeddings[off + c]! ** 2;
    points[i]!.norm = Math.sqrt(s);
  }
  const coords = MOCK_SCATTER.map(
    (p): [number, number] => [p.x, p.y],
  );
  return {
    points,
    coords,
    explainedVariance: [1, 1],
    hiddenDim,
    embeddings,
    isMock: true,
  };
}

export function EmbeddingsDemo({ workerRef, abortRef }: EmbeddingsDemoProps) {
  const [scatter, setScatter] = React.useState<LoadedScatter | null>(null);
  const [status, setStatus] = React.useState<RunStatus>({ kind: "idle" });
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState<string>("");
  const [selectedIdx, setSelectedIdx] = React.useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = React.useState<number | null>(null);
  const [activeCategories, setActiveCategories] = React.useState<Set<Category>>(
    () => new Set(ALL_CATEGORIES),
  );

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);

  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
    },
    [],
  );

  function applyMockFallback(reason: RunStatus, logMessage: string) {
    const mock = makeMockScatter();
    console.log(logMessage, { points: mock.points.length });
    setScatter(mock);
    setStatus(reason);
    setSelectedIdx(null);
  }

  async function handleLoad() {
    const myGen = ++runGenRef.current;
    setRunning(true);
    setSelectedIdx(null);

    const worker = workerRef.current;
    if (!worker) {
      applyMockFallback(
        { kind: "mock-no-worker" },
        "[embeddings-demo] using mock scatter (model not loaded)",
      );
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
      appSignal.addEventListener("abort", onAppAbort, { once: true });
    }

    try {
      setProgress(`Tokenizing ${WORDS.length} words...`);
      // Tokenize each word with a leading space (most words land as a single
      // token under Qwen3 BPE when space-prefixed). We pick the FIRST token id
      // for each word — multi-token words still get a sensible vector since
      // their first piece typically carries the semantic seed.
      const wordTokenIds: Array<number | null> = new Array(WORDS.length).fill(
        null,
      );
      for (let i = 0; i < WORDS.length; i++) {
        if (runGenRef.current !== myGen) return;
        const tokens = await tokenize(worker, " " + WORDS[i]!.word, {
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
        throw new Error("No tokens produced for any word");
      }

      setProgress(
        `Fetching ${keptTokenIds.length} embeddings from the model...`,
      );
      const { embeddings, hiddenDim } = await embed(worker, keptTokenIds, {
        signal: ctrl.signal,
      });
      if (runGenRef.current !== myGen) return;

      setProgress(
        `Running PCA on [${keptWords.length} × ${hiddenDim}] matrix...`,
      );
      // Yield to the event loop so the progress text renders before we
      // start the (short) PCA computation.
      await new Promise((r) => setTimeout(r, 0));
      const { coords, explainedVariance } = pca2D(
        embeddings,
        keptWords.length,
        hiddenDim,
      );

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

      console.log("[embeddings-demo] embed ok", {
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
        isMock: false,
      });
      setStatus({ kind: "ok" });
      setProgress("");
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info(
          "[embeddings-demo] aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) setStatus({ kind: "aborted" });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error(
          "[embeddings-demo] failed; falling back to mock",
          err,
        );
        applyMockFallback(
          { kind: "mock-error", error: message },
          "[embeddings-demo] using mock scatter (backend error)",
        );
      }
    } finally {
      if (appSignal) appSignal.removeEventListener("abort", onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      if (runGenRef.current === myGen) {
        setRunning(false);
        setProgress("");
      }
    }
  }

  const neighbors = React.useMemo(() => {
    if (!scatter || selectedIdx == null) return [];
    return topNeighbors(
      scatter.embeddings,
      scatter.hiddenDim,
      scatter.points,
      selectedIdx,
      5,
    );
  }, [scatter, selectedIdx]);

  const neighborSet = React.useMemo(
    () => new Set(neighbors.map((n) => n.idx)),
    [neighbors],
  );

  function toggleCategory(cat: Category) {
    setActiveCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          onClick={handleLoad}
          disabled={running}
          title="Tokenizes ~110 words, looks up embeddings, runs PCA in browser"
        >
          {running ? "Loading…" : scatter ? "Reload embeddings" : "Load embeddings"}
        </Button>
        <span className="text-xs text-muted-foreground">
          {progress
            ? progress
            : scatter
              ? scatter.isMock
                ? "Demo data (model not loaded)"
                : `${scatter.points.length} words · ${scatter.hiddenDim}-dim embeddings`
              : "Tokenize a curated word list, fetch the model's embeddings, project to 2D with PCA."}
        </span>
      </div>

      {status.kind === "mock-no-worker" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Showing a hand-tuned demo scatter — load the model from the chat tab
          first to see the real PCA projection of Qwen3.5's embedding matrix.
        </div>
      ) : null}
      {status.kind === "mock-error" ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Embed lookup failed.</strong> Showing demo scatter instead.{" "}
          {status.error}
        </div>
      ) : null}
      {status.kind === "aborted" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Request cancelled — model was reloaded. Click{" "}
          <strong>Reload embeddings</strong> to retry.
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
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
                active
                  ? "border-foreground/20 bg-background"
                  : "border-dashed border-muted-foreground/40 bg-muted/40 text-muted-foreground line-through",
              ].join(" ")}
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

      {scatter ? (
        <ScatterPlot
          scatter={scatter}
          selectedIdx={selectedIdx}
          hoverIdx={hoverIdx}
          neighborSet={neighborSet}
          activeCategories={activeCategories}
          onSelect={(idx) => setSelectedIdx(idx)}
          onHover={(idx) => setHoverIdx(idx)}
        />
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Load embeddings</strong> to project Qwen3.5's
          embedding matrix to 2D.
        </div>
      )}

      {scatter && selectedIdx != null ? (
        <NeighborList
          scatter={scatter}
          selectedIdx={selectedIdx}
          neighbors={neighbors}
        />
      ) : scatter ? (
        <div className="rounded-md bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          Click a point in the scatter to see its top-5 nearest neighbours
          (cosine similarity in the full {scatter.hiddenDim}-dim space).
        </div>
      ) : null}
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
  selectedIdx,
  hoverIdx,
  neighborSet,
  activeCategories,
  onSelect,
  onHover,
}: {
  scatter: LoadedScatter;
  selectedIdx: number | null;
  hoverIdx: number | null;
  neighborSet: Set<number>;
  activeCategories: Set<Category>;
  onSelect: (idx: number) => void;
  onHover: (idx: number | null) => void;
}) {
  const { coords, points } = scatter;
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
  }, [coords]);

  function project(x: number, y: number): [number, number] {
    const w = SCATTER_WIDTH - 2 * SCATTER_PADDING;
    const h = SCATTER_HEIGHT - 2 * SCATTER_PADDING;
    const px =
      SCATTER_PADDING +
      ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * w;
    // Flip y so positive PC2 points up in the visual.
    const py =
      SCATTER_PADDING +
      h -
      ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * h;
    return [px, py];
  }

  const hoveredOrSelected = hoverIdx ?? selectedIdx;

  return (
    <div className="rounded-md border border-border bg-background p-2">
      <svg
        role="img"
        aria-label="PCA scatter of token embeddings"
        viewBox={`0 0 ${SCATTER_WIDTH} ${SCATTER_HEIGHT}`}
        className="block w-full h-auto"
      >
        {/* Axes — neutral cross at the origin if it falls inside the bounds. */}
        {bounds.xMin <= 0 && bounds.xMax >= 0 ? (
          (() => {
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
        ) : null}
        {bounds.yMin <= 0 && bounds.yMax >= 0 ? (
          (() => {
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
        ) : null}

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
        <text
          x={8}
          y={SCATTER_PADDING - 6}
          className="fill-muted-foreground"
          fontSize="10"
        >
          PC2
        </text>

        {coords.map(([x, y], i) => {
          const p = points[i]!;
          const visible = activeCategories.has(p.category);
          if (!visible) return null;
          const [px, py] = project(x, y);
          const isSelected = i === selectedIdx;
          const isHover = i === hoverIdx;
          const isNeighbor = neighborSet.has(i);
          const r = isSelected ? 6 : isHover ? 5 : isNeighbor ? 4.5 : 3;
          const stroke = isSelected
            ? "var(--foreground)"
            : isNeighbor
              ? "var(--foreground)"
              : "none";
          const strokeWidth = isSelected ? 2 : isNeighbor ? 1.5 : 0;
          return (
            <g key={i}>
              <circle
                cx={px}
                cy={py}
                r={r}
                fill={CATEGORY_COLORS[p.category]}
                fillOpacity={
                  hoveredOrSelected != null &&
                  !isSelected &&
                  !isNeighbor &&
                  !isHover
                    ? 0.4
                    : 0.9
                }
                stroke={stroke}
                strokeWidth={strokeWidth}
                onClick={() => onSelect(i)}
                onMouseEnter={() => onHover(i)}
                onMouseLeave={() => onHover(null)}
                style={{ cursor: "pointer" }}
              >
                <title>
                  {p.word} · {CATEGORY_LABELS[p.category]}
                  {p.tokenId >= 0 ? ` · id ${p.tokenId}` : ""}
                </title>
              </circle>
              {isSelected || isHover || isNeighbor ? (
                <text
                  x={px + 8}
                  y={py + 3}
                  fontSize="10"
                  className="pointer-events-none fill-foreground"
                >
                  {p.word}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex items-center justify-between gap-2 px-2 pb-1 font-mono text-[10px] text-muted-foreground">
        <span>
          {scatter.isMock
            ? "mock 2D coords"
            : `PC1=${scatter.explainedVariance[0].toExponential(2)}, PC2=${scatter.explainedVariance[1].toExponential(2)} (eigenvalues)`}
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
  selectedIdx,
  neighbors,
}: {
  scatter: LoadedScatter;
  selectedIdx: number;
  neighbors: Array<{ idx: number; sim: number }>;
}) {
  const sel = scatter.points[selectedIdx]!;
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 flex items-center gap-2 text-xs">
        <span
          aria-hidden="true"
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: CATEGORY_COLORS[sel.category] }}
        />
        <span className="font-mono text-foreground">{sel.word}</span>
        <span className="text-muted-foreground">
          ({CATEGORY_LABELS[sel.category]}
          {sel.tokenId >= 0 ? ` · id ${sel.tokenId}` : ""})
        </span>
        <span className="ml-auto text-muted-foreground">
          top-5 nearest in {scatter.hiddenDim}-dim cosine
        </span>
      </div>
      <ol className="space-y-1">
        {neighbors.map(({ idx, sim }, rank) => {
          const p = scatter.points[idx]!;
          return (
            <li
              key={idx}
              className="grid grid-cols-[1.5rem_auto_minmax(0,1fr)_4rem] items-center gap-2 text-xs"
            >
              <span className="font-mono text-muted-foreground">
                #{rank + 1}
              </span>
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: CATEGORY_COLORS[p.category] }}
              />
              <span className="truncate">
                <span className="font-mono">{p.word}</span>
                <span className="ml-2 text-muted-foreground">
                  {CATEGORY_LABELS[p.category]}
                </span>
              </span>
              <span className="text-right font-mono text-foreground/80">
                {sim.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
