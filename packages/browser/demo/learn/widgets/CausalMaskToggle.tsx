import * as React from 'react';

/**
 * Chapter 3 supplement — show the *consequence* of the causal mask by toggling
 * it on and off side-by-side. `CausalMaskAsSum` shows the mechanic ("mask is
 * an additive matrix of 0/-inf"); this widget shows why we bother.
 *
 * With mask ON: row i is a probability distribution over positions 0..i. Row
 *   0 ("The") has nowhere to look except itself → 1.0 on the diagonal. The
 *   model has to predict what follows "The" using only "The".
 *
 * With mask OFF: row 0 can see the entire future of the sequence. The model
 *   is now trying to predict what follows "The" while being shown that the
 *   next token is " cat". This is the "cheating during training" the feedback
 *   asks us to make explicit.
 *
 * Numbers are illustrative, not from the model — same fixed RAW_SCORES as
 * CausalMaskAsSum so the two widgets line up.
 */

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];

// Same scores as CausalMaskAsSum — keeps the two widgets visually consistent
// for a reader scrolling between them. Picked to give a clear diagonal peak
// after masked softmax and a clear "future leakage" pattern without it.
// prettier-ignore
const RAW_SCORES: number[][] = [
  [ 1.2,  0.3, -0.8,  0.5,  1.1],
  [ 0.4,  1.6,  0.2, -0.3,  0.8],
  [-0.5,  0.9,  1.7,  0.1,  0.0],
  [ 0.6,  0.0,  0.4,  1.4,  0.7],
  [ 1.0,  0.5, -0.2,  0.8,  1.8],
];

function renderToken(text: string): string {
  return text.startsWith(' ') ? '·' + text.slice(1) : text;
}

function softmaxRow(row: number[]): number[] {
  const finite = row.filter(Number.isFinite);
  const max = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0) return row.map(() => 0);
  return exps.map((e) => e / sum);
}

function fmtProb(v: number): string {
  if (v < 0.005) return '0.00';
  return v.toFixed(2);
}

function ProbCell({ v, dim, highlight }: { v: number; dim?: boolean; highlight?: boolean }) {
  const opacity = Math.min(0.9, Math.max(0.04, v));
  return (
    <td
      className={[
        'border px-1.5 py-0.5 text-center font-mono text-[11px] transition-colors duration-500',
        dim ? 'border-border/40 text-muted-foreground/70' : 'border-border/70 text-foreground/95',
        highlight ? 'ring-2 ring-amber-500/60 ring-offset-1 ring-offset-background' : '',
      ].join(' ')}
      style={{ backgroundColor: `oklch(0.85 0.12 25 / ${opacity})` }}
    >
      {fmtProb(v)}
    </td>
  );
}

function TableHeader({ label, highlight }: { label: string; highlight?: boolean }) {
  return (
    <div
      className={[
        'mb-1 text-center text-[10px] uppercase tracking-wider transition-colors duration-300',
        highlight ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground',
      ].join(' ')}
    >
      {label}
    </div>
  );
}

function ColHeaders() {
  return (
    <tr>
      <th className="px-1 py-0.5" aria-hidden="true" />
      {TOKENS.map((t, j) => (
        <th key={j} className="px-1 py-0.5 text-[10px] font-mono text-muted-foreground">
          {renderToken(t)}
        </th>
      ))}
    </tr>
  );
}

export function CausalMaskToggle() {
  const [maskOn, setMaskOn] = React.useState(true);

  // With mask: upper triangle set to -inf, then softmax. Without: softmax
  // over the full row. Computed every render — N=5 so cost is trivial.
  const masked = React.useMemo(
    () => RAW_SCORES.map((row, i) => row.map((v, j) => (maskOn && j > i ? Number.NEGATIVE_INFINITY : v))),
    [maskOn],
  );
  const probs = React.useMemo(() => masked.map(softmaxRow), [masked]);

  // The pedagogical row: row 0 ("The") sees nothing useful with the mask on,
  // sees the entire future without it. Highlight it so the eye lands there.
  const focusRow = 0;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Why we mask — flip it off and watch the model cheat
        </div>
        <div className="inline-flex items-center gap-2">
          <span className="text-[11px] text-muted-foreground">Causal mask</span>
          <button
            type="button"
            onClick={() => setMaskOn((v) => !v)}
            className={[
              'relative inline-flex h-6 w-11 items-center rounded-full border transition-colors duration-200',
              maskOn ? 'border-primary/50 bg-primary/30' : 'border-amber-500/50 bg-amber-500/30',
            ].join(' ')}
            role="switch"
            aria-checked={maskOn}
            aria-label="Toggle causal mask"
          >
            <span
              className={[
                'inline-block size-4 rounded-full bg-background shadow transition-transform duration-200',
                maskOn ? 'translate-x-6' : 'translate-x-1',
              ].join(' ')}
            />
          </button>
          <span
            className={[
              'min-w-[3.5rem] text-center font-mono text-[11px] transition-colors duration-200',
              maskOn ? 'text-primary' : 'text-amber-600 dark:text-amber-400',
            ].join(' ')}
          >
            {maskOn ? 'ON' : 'OFF'}
          </span>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The matrix below is <code>softmax(QKᵀ / √d)</code> for the same five fixed scores as the widget above. Toggle
        the mask off and watch the highlighted row — the query for <code>"The"</code> — start attending to tokens that
        haven't happened yet.
      </p>

      <div className="overflow-x-auto">
        <table className="mx-auto border-collapse">
          <thead>
            <TableHeaderCaption maskOn={maskOn} />
            <ColHeaders />
          </thead>
          <tbody>
            {TOKENS.map((row, i) => (
              <tr key={i}>
                <th className="px-1 py-0.5 text-right text-[10px] font-mono text-muted-foreground">
                  {renderToken(row)}
                </th>
                {probs[i]!.map((v, j) => (
                  <ProbCell
                    key={j}
                    v={v}
                    dim={maskOn && j > i}
                    highlight={i === focusRow && (maskOn ? j === 0 : j > 0)}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        className={[
          'rounded-md border px-3 py-2 text-[12px] transition-colors duration-300',
          maskOn
            ? 'border-primary/30 bg-primary/5 text-foreground/90'
            : 'border-amber-500/40 bg-amber-500/10 text-foreground/95',
        ].join(' ')}
      >
        {maskOn ? (
          <>
            <strong>Mask ON.</strong> Row 0 (the query for <code>"The"</code>) has only one position to look at —
            itself. Its prediction must rely on what <code>"The"</code> alone tells the model. That's the regime the
            model trains in: every next-token prediction is made <em>before</em> the answer is visible.
          </>
        ) : (
          <>
            <strong>Mask OFF — the model is cheating.</strong> Row 0 now attends to <code>" cat"</code>,{' '}
            <code>" sat"</code>, <code>" on"</code>, <code>" the"</code> — every future token. During training, the loss
            is computed against
            <em> the same future tokens</em> the model is now allowed to see. It would learn to copy position{' '}
            <em>i+1</em> and score ~100% on the training set while learning nothing about language. At inference time
            there <em>is</em> no future to copy and it falls apart.
          </>
        )}
      </div>

      <p className="text-[11px] text-muted-foreground">
        That single triangular mask is the only thing keeping training honest. It costs nothing at inference (the upper
        triangle is never even computed for decode steps) and everything pedagogically — it's <em>the</em> structural
        difference between encoder and decoder transformers.
      </p>
    </div>
  );
}

function TableHeaderCaption({ maskOn }: { maskOn: boolean }) {
  // A single full-width row above the column headers labelling the table.
  // Rendered inside <thead> with a colSpan so it scrolls with the matrix.
  return (
    <tr>
      <td colSpan={TOKENS.length + 1} className="pb-1">
        <TableHeader
          label={maskOn ? 'attention weights — causal mask applied' : 'attention weights — no mask (cheating)'}
          highlight={!maskOn}
        />
      </td>
    </tr>
  );
}
