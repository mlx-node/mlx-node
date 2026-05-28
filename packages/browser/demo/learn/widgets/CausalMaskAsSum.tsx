import * as React from 'react';

/**
 * Chapter 3 supplement — show the causal mask as an explicit matrix sum.
 *
 *   scores  +  mask  =  masked_scores  → softmax → attention weights
 *
 * This makes the "set the upper triangle to -inf" sentence concrete: the
 * mask is literally a matrix of 0s and -inf's that we add element-wise
 * before softmax. Inspired by the `decoder-s2/s3` panels in
 * An-Jhon/Hand-Drawn-Transformer, redrawn in SVG with our token set.
 *
 * Numbers are illustrative (not from the real model) — the point is the
 * *operation*, not the values. Picking fixed scores in the [-2, 2] range
 * keeps the softmax outputs visually meaningful too.
 */

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];

// Hand-picked symmetric-ish raw scores so the masked softmax has a clear
// per-row peak (helps the pedagogy: "look, the brightest cell shifts down
// the diagonal as we mask more"). These are NOT model values.
// prettier-ignore
const RAW_SCORES: number[][] = [
  [ 1.2,  0.3, -0.8,  0.5,  1.1],
  [ 0.4,  1.6,  0.2, -0.3,  0.8],
  [-0.5,  0.9,  1.7,  0.1,  0.0],
  [ 0.6,  0.0,  0.4,  1.4,  0.7],
  [ 1.0,  0.5, -0.2,  0.8,  1.8],
];

function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  return text;
}

function softmaxRow(row: number[]): number[] {
  const finite = row.filter(Number.isFinite);
  const max = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0) return row.map(() => 0);
  return exps.map((e) => e / sum);
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '−∞';
  if (Math.abs(v) < 0.005) return '0.00';
  return v.toFixed(2);
}

function fmtProb(v: number): string {
  if (v < 0.005) return '0.00';
  return v.toFixed(2);
}

function MatrixHeader({ label }: { label: string }) {
  return <div className="mb-1 text-center text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>;
}

function CornerLabel() {
  return <th className="px-1 py-0.5" aria-hidden="true" />;
}

function ColHeaders() {
  return (
    <tr>
      <CornerLabel />
      {TOKENS.map((t, j) => (
        <th key={j} className="px-1 py-0.5 text-[10px] font-mono text-muted-foreground">
          {renderToken(t)}
        </th>
      ))}
    </tr>
  );
}

function ScoreCell({ v, mute }: { v: number; mute?: boolean }) {
  return (
    <td
      className={[
        'border border-border/60 px-1.5 py-0.5 text-center font-mono text-[11px]',
        mute ? 'text-muted-foreground/60' : 'text-foreground/85',
      ].join(' ')}
    >
      {fmt(v)}
    </td>
  );
}

function MaskCell({ allowed }: { allowed: boolean }) {
  return (
    <td
      className={[
        'border border-border/60 px-1.5 py-0.5 text-center font-mono text-[11px]',
        allowed ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : 'bg-muted/60 text-muted-foreground',
      ].join(' ')}
    >
      {allowed ? '0' : '−∞'}
    </td>
  );
}

function MaskedCell({ v, allowed }: { v: number; allowed: boolean }) {
  return (
    <td
      className={[
        'border border-border/60 px-1.5 py-0.5 text-center font-mono text-[11px]',
        allowed ? 'text-foreground/90' : 'bg-muted/40 text-muted-foreground',
      ].join(' ')}
    >
      {allowed ? fmt(v) : '−∞'}
    </td>
  );
}

function ProbCell({ v, allowed }: { v: number; allowed: boolean }) {
  if (!allowed) {
    return (
      <td className="border border-border/60 bg-muted/40 px-1.5 py-0.5 text-center font-mono text-[11px] text-muted-foreground">
        0
      </td>
    );
  }
  // Translucent fill keyed off the probability — readable text stays on top.
  const opacity = Math.min(0.85, Math.max(0.05, v));
  return (
    <td
      className="border border-border/60 px-1.5 py-0.5 text-center font-mono text-[11px] text-foreground/95"
      style={{ backgroundColor: `oklch(0.85 0.12 25 / ${opacity})` }}
    >
      {fmtProb(v)}
    </td>
  );
}

function Glyph({ symbol }: { symbol: string }) {
  return (
    <div
      className="flex items-center justify-center px-1 text-2xl font-light text-amber-600 dark:text-amber-400"
      aria-hidden="true"
    >
      {symbol}
    </div>
  );
}

export function CausalMaskAsSum() {
  // Compute masked scores (lower-tri kept, upper-tri set to -inf) and the
  // per-row softmax. This is the actual operation; the visual just tiles it.
  const masked = RAW_SCORES.map((row, i) => row.map((v, j) => (j <= i ? v : Number.NEGATIVE_INFINITY)));
  const probs = masked.map(softmaxRow);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Causal mask as a matrix sum</div>
        <div className="text-[11px] text-muted-foreground">scores + mask = masked scores → softmax → weights</div>
      </div>

      <p className="text-[12px] text-foreground/85">
        "Set the upper triangle to <code>-∞</code>" reads more cleanly as an addition: the mask is literally a matrix of{' '}
        <code>0</code> on/below the diagonal and <code>-∞</code> above. After softmax, the <code>-∞</code> cells become
        exactly <code>0</code>, so every row is a valid probability distribution over earlier positions only.
      </p>

      <div className="flex flex-wrap items-stretch justify-center gap-2 overflow-x-auto">
        {/* Raw scores */}
        <div>
          <MatrixHeader label={'Q·Kᵀ / √d'} />
          <table className="border-collapse">
            <thead>
              <ColHeaders />
            </thead>
            <tbody>
              {TOKENS.map((row, i) => (
                <tr key={i}>
                  <th className="px-1 py-0.5 text-right text-[10px] font-mono text-muted-foreground">
                    {renderToken(row)}
                  </th>
                  {RAW_SCORES[i]!.map((v, j) => (
                    <ScoreCell key={j} v={v} mute={j > i} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Glyph symbol="+" />

        {/* Causal mask */}
        <div>
          <MatrixHeader label="causal mask" />
          <table className="border-collapse">
            <thead>
              <ColHeaders />
            </thead>
            <tbody>
              {TOKENS.map((row, i) => (
                <tr key={i}>
                  <th className="px-1 py-0.5 text-right text-[10px] font-mono text-muted-foreground">
                    {renderToken(row)}
                  </th>
                  {TOKENS.map((_, j) => (
                    <MaskCell key={j} allowed={j <= i} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Glyph symbol="=" />

        {/* Masked scores */}
        <div>
          <MatrixHeader label="masked scores" />
          <table className="border-collapse">
            <thead>
              <ColHeaders />
            </thead>
            <tbody>
              {TOKENS.map((row, i) => (
                <tr key={i}>
                  <th className="px-1 py-0.5 text-right text-[10px] font-mono text-muted-foreground">
                    {renderToken(row)}
                  </th>
                  {masked[i]!.map((v, j) => (
                    <MaskedCell key={j} v={v} allowed={j <= i} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Glyph symbol="→" />

        {/* Post-softmax probabilities */}
        <div>
          <MatrixHeader label="softmax per row" />
          <table className="border-collapse">
            <thead>
              <ColHeaders />
            </thead>
            <tbody>
              {TOKENS.map((row, i) => (
                <tr key={i}>
                  <th className="px-1 py-0.5 text-right text-[10px] font-mono text-muted-foreground">
                    {renderToken(row)}
                  </th>
                  {probs[i]!.map((v, j) => (
                    <ProbCell key={j} v={v} allowed={j <= i} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground">
        Numbers in the leftmost matrix are illustrative, not from the model. The point is the operation, not the values
        — every row of the rightmost matrix sums to <span className="font-mono">1.00</span>, and every cell above the
        diagonal is exactly <span className="font-mono">0</span>.
      </p>
    </div>
  );
}
