import * as React from 'react';

import { Button } from '../../components/ui/button';

/**
 * A hand-worked 2-token attention example. All numbers are computed at module
 * load from fixed inputs so the user sees the exact same values every visit.
 * Steps are rendered as stacked cards; a stepper highlights one card at a
 * time so a learner can walk through Q, K, V, scores, scaling, masking,
 * softmax, and the final output in order.
 */

// Tokens
const TOKEN_NAMES = ['The', 'cat'];

// Hand-picked embeddings, weights, etc. Kept in 2-d so the matrix algebra
// stays printable. Q and K use identity projections — Q[The] = x[The], etc. —
// and V uses a small non-identity matrix so the final output mixing is
// visible rather than degenerate.
const X = [
  [1.0, 0.0], // The
  [0.0, 1.0], // cat
];

const W_Q = [
  [1.0, 0.0],
  [0.0, 1.0],
];
const W_K = [
  [1.0, 0.0],
  [0.0, 1.0],
];
const W_V = [
  [0.5, -0.3],
  [0.2, 0.7],
];

const D_K = 2;
const SQRT_D_K = Math.sqrt(D_K);

function matVec(M: number[][], v: number[]): number[] {
  // Row-vector × matrix → row-vector. v has length |cols(M)| treated as
  // (1, n) and M is (n, n). Returns length-n.
  const n = v.length;
  const out = new Array<number>(n).fill(0);
  for (let c = 0; c < n; c++) {
    let s = 0;
    for (let r = 0; r < n; r++) {
      s += v[r]! * M[r]![c]!;
    }
    out[c] = s;
  }
  return out;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return n > 0 ? '∞' : '-∞';
  return n.toFixed(2);
}

// Per-token projections.
const Q = X.map((x) => matVec(W_Q, x));
const K = X.map((x) => matVec(W_K, x));
const V = X.map((x) => matVec(W_V, x));

// Raw scores: scores[i][j] = Q[i] · K[j].
const SCORES = Q.map((q) => K.map((k) => dot(q, k)));
const SCALED = SCORES.map((row) => row.map((s) => s / SQRT_D_K));

// Causal-masked scaled scores: upper triangle (j > i) becomes -Infinity.
const MASKED = SCALED.map((row, i) => row.map((v, j) => (j > i ? Number.NEGATIVE_INFINITY : v)));

function softmaxRow(row: number[]): number[] {
  const finite = row.filter((v) => Number.isFinite(v));
  const max = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 0));
}

const ATTN = MASKED.map(softmaxRow);

// Output: out[i] = sum_j attn[i][j] * V[j].
const OUT = ATTN.map((row) => {
  const o = new Array<number>(V[0]!.length).fill(0);
  for (let j = 0; j < row.length; j++) {
    const w = row[j]!;
    for (let d = 0; d < V[j]!.length; d++) {
      o[d]! += w * V[j]![d]!;
    }
  }
  return o;
});

type Card = {
  title: string;
  body: React.ReactNode;
};

const CARDS: Card[] = [
  {
    title: '1. Embeddings',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">Two tokens, each a tiny 2-d vector.</p>
        <pre className="rounded bg-muted p-2 font-mono text-[11px] leading-5">
          <code>
            {`x[The] = [${fmt(X[0]![0]!)}, ${fmt(X[0]![1]!)}]
x[cat] = [${fmt(X[1]![0]!)}, ${fmt(X[1]![1]!)}]`}
          </code>
        </pre>
      </div>
    ),
  },
  {
    title: '2. Q = x · W_Q, K = x · W_K, V = x · W_V',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">
          W_Q and W_K are identity; W_V is non-identity so the final mix is visible.
        </p>
        <pre className="rounded bg-muted p-2 font-mono text-[11px] leading-5">
          <code>
            {`Q[The] = [${fmt(Q[0]![0]!)}, ${fmt(Q[0]![1]!)}]   Q[cat] = [${fmt(Q[1]![0]!)}, ${fmt(Q[1]![1]!)}]
K[The] = [${fmt(K[0]![0]!)}, ${fmt(K[0]![1]!)}]   K[cat] = [${fmt(K[1]![0]!)}, ${fmt(K[1]![1]!)}]
V[The] = [${fmt(V[0]![0]!)}, ${fmt(V[0]![1]!)}]   V[cat] = [${fmt(V[1]![0]!)}, ${fmt(V[1]![1]!)}]`}
          </code>
        </pre>
      </div>
    ),
  },
  {
    title: '3. Scores = Q · K^T',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">Each cell (i, j) is the dot product of Q[i] with K[j].</p>
        <Matrix2x2 rows={SCORES} rowLabels={TOKEN_NAMES} colLabels={TOKEN_NAMES} />
      </div>
    ),
  },
  {
    title: '4. Scale by sqrt(d_k)',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">
          d_k = {D_K}, sqrt(d_k) = {SQRT_D_K.toFixed(3)}. Divide every cell.
        </p>
        <Matrix2x2 rows={SCALED} rowLabels={TOKEN_NAMES} colLabels={TOKEN_NAMES} />
      </div>
    ),
  },
  {
    title: '5. Mask (causal)',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">
          Token &quot;The&quot; cannot see &quot;cat&quot; (it comes later) — set the upper-right cell to -inf so it
          disappears after softmax.
        </p>
        <Matrix2x2 rows={MASKED} rowLabels={TOKEN_NAMES} colLabels={TOKEN_NAMES} />
      </div>
    ),
  },
  {
    title: '6. Softmax per row',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">
          Each row becomes a probability distribution. Row &quot;The&quot; has only one valid key, so it attends 100% to
          itself.
        </p>
        <Matrix2x2 rows={ATTN} rowLabels={TOKEN_NAMES} colLabels={TOKEN_NAMES} highlight />
      </div>
    ),
  },
  {
    title: '7. Output = attn · V',
    body: (
      <div className="space-y-1">
        <p className="text-[12px] text-foreground/85">
          Mix the value vectors using the attention weights. &quot;The&quot; ends up exactly V[The]; &quot;cat&quot; is
          a weighted blend of V[The] and V[cat].
        </p>
        <pre className="rounded bg-muted p-2 font-mono text-[11px] leading-5">
          <code>
            {`out[The] = [${fmt(OUT[0]![0]!)}, ${fmt(OUT[0]![1]!)}]
out[cat] = [${fmt(OUT[1]![0]!)}, ${fmt(OUT[1]![1]!)}]`}
          </code>
        </pre>
      </div>
    ),
  },
];

function Matrix2x2({
  rows,
  rowLabels,
  colLabels,
  highlight = false,
}: {
  rows: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlight?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse font-mono text-[11px]">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-muted-foreground" />
            {colLabels.map((c, i) => (
              <th key={`col-${i}`} className="px-2 py-1 text-right text-muted-foreground">
                K[{c}]
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`row-${i}`}>
              <th className="border-t border-border/60 px-2 py-1 text-left text-muted-foreground">Q[{rowLabels[i]}]</th>
              {row.map((v, j) => {
                const isMasked = !Number.isFinite(v);
                const intensity = highlight && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
                return (
                  <td
                    key={`cell-${i}-${j}`}
                    className="border-t border-border/60 px-2 py-1 text-right"
                    style={
                      highlight && !isMasked
                        ? {
                            backgroundColor: `rgba(56, 189, 248, ${intensity * 0.35})`,
                          }
                        : undefined
                    }
                  >
                    {isMasked ? <span className="text-muted-foreground/60">-∞</span> : fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ToyAttentionExample() {
  const [stepIdx, setStepIdx] = React.useState(0);
  const safeIdx = Math.max(0, Math.min(stepIdx, CARDS.length - 1));

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Toy 2-token attention · hand-worked numbers
        </div>
        <div className="text-[11px] text-muted-foreground">
          Step {safeIdx + 1} of {CARDS.length}
        </div>
      </div>

      <div className="space-y-2">
        {CARDS.map((card, i) => {
          const isActive = i === safeIdx;
          return (
            <div
              key={`card-${i}`}
              className={[
                'rounded-md border p-3 transition-opacity',
                isActive ? 'border-primary/40 bg-primary/5 opacity-100' : 'border-border/60 bg-muted/20 opacity-60',
              ].join(' ')}
            >
              <div className="mb-1 text-[12px] font-semibold text-foreground">{card.title}</div>
              {card.body}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => setStepIdx(Math.max(0, safeIdx - 1))}
          disabled={safeIdx === 0}
        >
          Prev
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setStepIdx(Math.min(CARDS.length - 1, safeIdx + 1))}
          disabled={safeIdx >= CARDS.length - 1}
        >
          Next
        </Button>
        <Button size="sm" variant="outline" onClick={() => setStepIdx(0)} disabled={safeIdx === 0}>
          Reset
        </Button>
        <span className="ml-2 text-[11px] text-muted-foreground">
          Each row of step 6 sums to 1 — that&apos;s softmax.
        </span>
      </div>
    </div>
  );
}
