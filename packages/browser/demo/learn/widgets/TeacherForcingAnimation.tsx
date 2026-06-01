import * as React from 'react';

/**
 * Chapter 12 (Training) supplement — animate teacher forcing.
 *
 * The whole rest of the course is inference: feed a prompt in, sample one
 * token, repeat. Training inverts the loop: feed the whole sequence in at
 * once, predict the next token at *every* position in parallel, take the
 * cross-entropy loss against the actual next token at each position.
 *
 * This widget plays one training step on a fixed 6-token sequence:
 *   "The cat sat on the mat"
 *
 * Step 1: feed all 6 tokens in (one row of cells).
 * Step 2: causal mask in place, model produces 6 predicted-next-token
 *         distributions in parallel (drawn as mini bar-stacks under each
 *         input position).
 * Step 3: the "true next token" at each position is highlighted in the
 *         predictions; the cross-entropy contribution -log(p_target) is
 *         shown as a stacked-bar.
 * Step 4: loss = mean of per-position contributions, drawn as a single bar.
 *
 * Numbers are illustrative — the point is the *shape* of the training step,
 * not real model probabilities.
 */

const SEQ = ['The', ' cat', ' sat', ' on', ' the', ' mat'];

// For each position, a synthetic predicted distribution over a small "alt
// vocab" — the actual next token at index TARGET_IDX, plus 3 confounders.
// We assign probability mass that gets ROUGHLY right (high p_target) for
// easier positions and uncertain (low p_target) for harder ones, so the
// cross-entropy bars have visible variance.
type Position = {
  /** Probabilities over [cat, sat, on, the, mat, dog] in that fixed order. */
  probs: number[];
  /** Which index in `probs` is the true next token. */
  targetIdx: number;
  /** Text labels for each prob bucket (display only). */
  labels: string[];
};

// Hand-picked so the loss visualization shows variety: position 0 ("The")
// is hard (low confidence in " cat"), position 4 (" the") is easy (high in
// " mat" since "on the mat" is a strong continuation).
const POSITIONS: Position[] = [
  // Position 0: input "The" → predict " cat" (low p, hard)
  { probs: [0.28, 0.14, 0.05, 0.18, 0.05, 0.3], targetIdx: 0, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 1: input "The cat" → predict " sat" (medium)
  { probs: [0.05, 0.45, 0.05, 0.1, 0.05, 0.3], targetIdx: 1, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 2: input "The cat sat" → predict " on" (medium-high)
  { probs: [0.02, 0.05, 0.6, 0.18, 0.05, 0.1], targetIdx: 2, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 3: input "The cat sat on" → predict " the" (high)
  { probs: [0.02, 0.02, 0.05, 0.78, 0.08, 0.05], targetIdx: 3, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 4: input "The cat sat on the" → predict " mat" (high)
  { probs: [0.05, 0.02, 0.02, 0.05, 0.78, 0.08], targetIdx: 4, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 5: input "The cat sat on the mat" → predict <end> (no target — last position not trained)
  { probs: [0.15, 0.15, 0.15, 0.15, 0.15, 0.25], targetIdx: -1, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
];

function renderToken(t: string): string {
  return t.startsWith(' ') ? '·' + t.slice(1) : t;
}

export function TeacherForcingAnimation() {
  const [step, setStep] = React.useState<0 | 1 | 2 | 3>(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setStep((s) => ((s + 1) % 4) as 0 | 1 | 2 | 3);
    }, 2200);
    return () => window.clearInterval(t);
  }, [playing]);

  const stepLabels = [
    'Step 1 — Feed the whole sequence in (one parallel forward pass).',
    'Step 2 — At every position, predict a distribution over the next token. Causal mask keeps each prediction honest.',
    'Step 3 — Compare each prediction to the actual next token. Per-position loss = −log p(target).',
    'Step 4 — Mean across positions = the training loss. Backprop adjusts every parameter to push p(target) up.',
  ];

  // Per-position cross-entropy: -log p(target).
  const losses = POSITIONS.map((p) => (p.targetIdx < 0 ? null : -Math.log(Math.max(1e-9, p.probs[p.targetIdx]!))));
  const trainedLosses = losses.filter((l): l is number => l !== null);
  const meanLoss = trainedLosses.reduce((a, b) => a + b, 0) / Math.max(1, trainedLosses.length);

  return (
    <div className="space-y-4 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Teacher forcing — one training step on six tokens
        </div>
        <div className="inline-flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
            aria-pressed={playing}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <span className="font-mono text-[11px] text-muted-foreground">step {step + 1}/4</span>
        </div>
      </div>

      <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-[12px] text-foreground/95">
        {stepLabels[step]}
      </div>

      {/* Row 1 — input tokens */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">input sequence</div>
        <div className="flex gap-1.5">
          {SEQ.map((t, i) => (
            <div
              key={`in-${i}`}
              className={[
                'flex h-9 flex-1 items-center justify-center rounded border font-mono text-[12px] transition-all duration-300',
                step >= 0
                  ? 'border-primary/40 bg-primary/10 text-foreground/95'
                  : 'border-border/40 bg-muted/40 text-muted-foreground/50',
              ].join(' ')}
            >
              {renderToken(t)}
            </div>
          ))}
        </div>
      </div>

      {/* Row 2 — per-position predicted distributions */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          per-position predictions (after softmax)
        </div>
        <div className="flex gap-1.5">
          {POSITIONS.map((pos, i) => {
            const visible = step >= 1;
            const targetVisible = step >= 2;
            return (
              <div
                key={`pos-${i}`}
                className={[
                  'flex flex-1 flex-col items-stretch gap-0.5 rounded border p-1 transition-all duration-500',
                  visible ? 'border-border/60 bg-background' : 'border-border/30 bg-muted/20 opacity-40',
                ].join(' ')}
              >
                {pos.probs.map((p, k) => {
                  const isTarget = k === pos.targetIdx;
                  return (
                    <div key={`b-${i}-${k}`} className="flex items-center gap-1">
                      <span
                        className={[
                          'w-7 shrink-0 truncate font-mono text-[9px]',
                          isTarget && targetVisible
                            ? 'text-emerald-600 dark:text-emerald-300'
                            : 'text-muted-foreground/70',
                        ].join(' ')}
                      >
                        {pos.labels[k]}
                      </span>
                      <div className="relative flex-1">
                        <div
                          className={[
                            'h-2.5 rounded-sm transition-all duration-500',
                            isTarget && targetVisible
                              ? 'bg-emerald-500/70 outline outline-1 outline-emerald-400'
                              : 'bg-primary/40',
                          ].join(' ')}
                          style={{ width: `${visible ? Math.max(2, p * 100) : 0}%` }}
                        />
                      </div>
                      <span className="w-7 shrink-0 text-right font-mono text-[9px] text-muted-foreground">
                        {visible ? p.toFixed(2) : ''}
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Row 3 — per-position loss = -log p(target) */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          per-position cross-entropy = −log p(target)
        </div>
        <div className="flex gap-1.5">
          {POSITIONS.map((pos, i) => {
            const visible = step >= 2;
            const l = losses[i];
            if (l === null) {
              return (
                <div
                  key={`l-${i}`}
                  className={[
                    'flex h-12 flex-1 items-center justify-center rounded border text-[9px] text-muted-foreground/70',
                    visible ? 'border-border/40 bg-muted/30' : 'border-border/20 bg-muted/10 opacity-30',
                  ].join(' ')}
                >
                  no target
                </div>
              );
            }
            const maxLossForScale = 3.5;
            const heightPct = Math.min(100, (l / maxLossForScale) * 100);
            return (
              <div
                key={`l-${i}`}
                className={[
                  'flex h-12 flex-1 flex-col items-stretch justify-end rounded border p-1 transition-all duration-500',
                  visible ? 'border-amber-500/40 bg-amber-500/5' : 'border-border/30 bg-muted/20 opacity-30',
                ].join(' ')}
              >
                <div
                  className="w-full rounded-sm bg-amber-500/70 transition-all duration-500"
                  style={{ height: `${visible ? heightPct : 0}%` }}
                />
                <span className="mt-0.5 text-center font-mono text-[9px] text-amber-700 dark:text-amber-400">
                  {visible ? l.toFixed(2) : ''}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Row 4 — mean = the training loss */}
      <div
        className={[
          'flex items-center justify-between rounded-md border px-3 py-2 transition-colors duration-500',
          step >= 3 ? 'border-amber-500/50 bg-amber-500/10' : 'border-border/40 bg-muted/30 opacity-50',
        ].join(' ')}
      >
        <div className="font-mono text-[12px] text-foreground/95">
          loss = mean(−log p(target_i)) over 5 trained positions
        </div>
        <div className="font-mono text-[14px] text-amber-700 dark:text-amber-400">
          = {step >= 3 ? meanLoss.toFixed(3) : '— —'}
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground">
        Two crucial observations. <strong>One</strong>: every position is trained simultaneously — the causal mask is
        the only thing that keeps it valid. <strong>Two</strong>: the input fed to position <em>i+1</em> is the{' '}
        <em>true</em> token from position <em>i</em>, not the model's prediction. That's "teacher forcing": during
        training, the model never has to recover from its own mistakes. (At inference, of course, it does — which is the
        small but real reason long-form generation sometimes drifts.)
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — the per-position probabilities here are hand-picked to show the shape of the loss, not live
        output from the model.
      </p>
    </div>
  );
}
