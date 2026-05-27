import * as React from "react";

/**
 * Three illustrative 6x6 attention heatmaps showing common archetypes of
 * what a head can learn to do. No model call — these are hand-built CSS
 * grids meant to convey the *shape* of each pattern rather than measured
 * data from any particular layer/head.
 */

const SIZE = 6;
const TOKEN_LABELS = ["t0", "t1", "t2", "t3", "t4", "t5"];

type Pattern = {
  label: string;
  caption: string;
  // weights[i][j] = attention from row i (query) to col j (key); 0..1.
  weights: number[][];
};

// Helper to construct a square zero matrix.
function zeroes(n: number): number[][] {
  return Array.from({ length: n }, () => Array(n).fill(0));
}

// Diagonal (positional) head — each row mostly attends to itself, with a
// small leak to immediate neighbours. Row 0 has no causal history so it
// fully self-attends.
function buildPositional(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![i] = 0.7;
    m[i]![i - 1] = 0.2;
    if (i - 2 >= 0) m[i]![i - 2] = 0.1;
  }
  return m;
}

// Recency head — heavy weight on the immediately preceding token. Row 0 has
// nothing to look back at, so it self-attends.
function buildRecency(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![i - 1] = 0.7;
    m[i]![i] = 0.2;
    if (i - 2 >= 0) m[i]![i - 2] = 0.1;
  }
  return m;
}

// Syntactic head — late tokens look back at the determiner at position 0.
function buildSyntactic(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![0] = 0.6;
    m[i]![i] = 0.25;
    if (i - 1 >= 0) m[i]![i - 1] = 0.15;
  }
  return m;
}

const PATTERNS: Pattern[] = [
  {
    label: "Positional head",
    caption:
      "Detector: position. Each token mostly attends to itself; minor leak to neighbours.",
    weights: buildPositional(),
  },
  {
    label: "Recency head",
    caption:
      "Detector: recency. Strong attention to the immediately-prior token.",
    weights: buildRecency(),
  },
  {
    label: "Syntactic head",
    caption:
      'Detector: syntactic head. Late tokens look back at the determiner ("t0").',
    weights: buildSyntactic(),
  },
];

function cellColor(v: number): string {
  // 0 → near-transparent gray; 1 → saturated blue.
  const clamped = Math.max(0, Math.min(1, v));
  const alpha = clamped * 0.85 + (clamped > 0 ? 0.05 : 0);
  return `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
}

function HeatGrid({ weights }: { weights: number[][] }) {
  return (
    <div
      role="img"
      aria-label="6 by 6 attention heatmap"
      className="grid gap-0.5"
      style={{ gridTemplateColumns: `auto repeat(${SIZE}, minmax(0, 1fr))` }}
    >
      <div />
      {TOKEN_LABELS.map((c) => (
        <div
          key={`col-${c}`}
          className="text-center font-mono text-[9px] text-muted-foreground"
        >
          {c}
        </div>
      ))}
      {weights.map((row, i) => (
        <React.Fragment key={`row-${i}`}>
          <div className="pr-1 text-right font-mono text-[9px] text-muted-foreground self-center">
            {TOKEN_LABELS[i]}
          </div>
          {row.map((v, j) => (
            <div
              key={`cell-${i}-${j}`}
              className="aspect-square rounded-[2px] border border-border/30"
              style={{ backgroundColor: cellColor(v) }}
              title={`(${i},${j}) = ${v.toFixed(2)}`}
            />
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}

export function PatternDetectorPanel() {
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          What heads learn · three archetypes
        </div>
      </div>
      <p className="text-[12px] text-foreground/85">
        Different heads learn to detect different things. You can&apos;t tell
        what a head detects from its weights alone — you have to see the
        patterns it produces on real inputs. Here are three common archetypes
        a typical mid-size LLM contains, drawn as illustrative heatmaps.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {PATTERNS.map((p, i) => (
          <div
            key={`pattern-${i}`}
            className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3"
          >
            <div className="text-[12px] font-semibold text-foreground">
              {p.label}
            </div>
            <HeatGrid weights={p.weights} />
            <p className="text-[11px] text-muted-foreground">{p.caption}</p>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-muted-foreground">
        These are hand-built diagrams, not measured from any specific model.
        Real heads are messier and often mix multiple of these archetypes.
      </p>
    </div>
  );
}
