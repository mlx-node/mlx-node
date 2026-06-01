import * as React from 'react';

import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 13 (Training) supplement — parallel-vs-sequential structure.
 *
 * This is distinct from the chapter's other two widgets:
 *   - TeacherForcingAnimation shows ONE training step's per-position CE bars.
 *   - ExposureBias shows DRIFT across inference steps.
 *   - This one isolates the STRUCTURAL difference between the two regimes over
 *     the same short sequence: training scores ALL positions in a single
 *     parallel forward pass (teacher-forced), inference walks the sequence ONE
 *     token at a time, feeding each freshly produced token back in.
 *
 * A single lockstep clock drives both panels. The "Training" panel lights up
 * every position at once (one step) and scores each against the gold next
 * token (green true-token highlight, matching TeacherForcingAnimation). The
 * "Inference" panel fills positions left to right, one per step.
 *
 * Synthetic / scripted — no model, no worker, no WASM.
 */

// Input tokens fed to the model (positions 0..N-1).
const SEQ = ['The', ' cat', ' sat', ' on', ' the', ' mat'] as const;
// The gold "next token" each position predicts (shift-by-one). The last input
// position has no target — the sequence ended — so it is null.
const TARGETS: (string | null)[] = [' cat', ' sat', ' on', ' the', ' mat', null];

const N = SEQ.length;
// Inference walks N steps (one token per step); the lockstep clock cycles
// through those steps plus one settle frame before looping.
const TOTAL_STEPS = N;
const STEP_MS = 1400;

function renderToken(t: string): string {
  return t.startsWith(' ') ? '·' + t.slice(1) : t;
}

export function TrainInferenceParallel() {
  // `step` runs 0..TOTAL_STEPS. For inference it counts revealed tokens; the
  // final frame (step === TOTAL_STEPS) holds the full sequence before looping.
  const [step, setStep] = React.useState(0);
  const [playing, setPlaying] = React.useState(() =>
    typeof window !== 'undefined' ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches : true,
  );

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setStep((s) => (s + 1) % (TOTAL_STEPS + 1)), STEP_MS);
    return () => window.clearInterval(t);
  }, [playing]);

  // Training: a single parallel pass. Every position is "lit" from the first
  // tick onward — there is no left-to-right reveal, that's the whole point.
  const trainingActive = step >= 1;
  // Inference: number of tokens produced so far (left-to-right reveal).
  const producedCount = step;
  // The position currently being produced flashes; -1 once the loop completes.
  const producingIndex = step >= 1 && step <= N ? step - 1 : -1;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          One parallel pass vs a sequential loop
        </div>
        <div className="inline-flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            aria-pressed={playing}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            {playing ? '❚❚ Pause' : '▶ Play'}
          </button>
          <span className="font-mono text-[11px] text-muted-foreground">
            step {Math.min(step, TOTAL_STEPS)}/{TOTAL_STEPS}
          </span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {/* Training — all N positions scored in ONE forward pass. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-medium text-foreground/90">Training — one parallel pass</span>
            <span className="text-[10px] text-muted-foreground">all {N} positions at once</span>
          </div>
          <div className="space-y-1">
            {SEQ.map((tok, i) => {
              const target = TARGETS[i];
              return (
                <div
                  key={`train-${i}`}
                  className={[
                    'flex items-center gap-2 rounded border px-2 py-1 transition-all duration-500',
                    trainingActive
                      ? 'border-primary/40 bg-primary/10 opacity-100'
                      : 'border-border/30 bg-muted/20 opacity-40',
                  ].join(' ')}
                >
                  <span className="w-14 shrink-0 truncate font-mono text-[11px] text-foreground/95">
                    {renderToken(tok)}
                  </span>
                  <span className="text-[10px] text-muted-foreground" aria-hidden>
                    →
                  </span>
                  {target === null ? (
                    <span className="font-mono text-[10px] text-muted-foreground/70">no target</span>
                  ) : (
                    <span
                      className={[
                        'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                        trainingActive
                          ? 'bg-emerald-500/70 text-white outline outline-1 outline-emerald-400'
                          : 'bg-muted/40 text-muted-foreground',
                      ].join(' ')}
                    >
                      {renderToken(target)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-muted-foreground">
            Teacher-forced: every position&apos;s input is the true token, and each is scored against the green gold{' '}
            <em>next</em> token in one parallel forward pass. The last position has no next token inside this window, so{' '}
            {N - 1} of the {N} positions contribute a loss.
          </p>
        </div>

        {/* Inference — one token per step, N steps. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-medium text-foreground/90">Inference — sequential loop</span>
            <span className="text-[10px] text-muted-foreground">one token per step, {N} steps</span>
          </div>
          <div className="space-y-1">
            {SEQ.map((tok, i) => {
              const produced = i < producedCount;
              const flashing = i === producingIndex;
              return (
                <div
                  key={`infer-${i}`}
                  className={[
                    'flex items-center gap-2 rounded border px-2 py-1 transition-all duration-300',
                    flashing
                      ? 'border-primary/60 bg-primary/20 opacity-100 ring-1 ring-primary/40'
                      : produced
                        ? 'border-primary/40 bg-primary/10 opacity-100'
                        : 'border-border/30 bg-muted/20 opacity-40',
                  ].join(' ')}
                >
                  <span className="w-6 shrink-0 text-right font-mono text-[10px] text-muted-foreground">
                    {produced ? i + 1 : ''}
                  </span>
                  <span className="text-[10px] text-muted-foreground" aria-hidden>
                    {produced ? '↻' : '·'}
                  </span>
                  <span
                    className={[
                      'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                      produced ? 'bg-primary/10 text-foreground/90' : 'text-muted-foreground/50',
                    ].join(' ')}
                  >
                    {renderToken(tok)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-muted-foreground">
            Each step produces one token, then feeds it back in as input for the next step — {N} forward passes for {N}{' '}
            tokens.
          </p>
        </div>
      </div>

      {/* Shared training loss — mean cross-entropy over positions. */}
      <MathDisplay latex={String.raw`\mathcal{L}=-\frac{1}{N}\sum_{i} \log p_\theta(x_{i+1}\mid x_{\le i})`} />

      <p className="text-[10px] text-muted-foreground">Illustrative — scripted, not live output from the model.</p>

      <p className="text-[12px] text-foreground/85">
        Same weights, same forward function. The <em>only</em> difference is the loop: training runs every position in{' '}
        <strong>parallel</strong> (teacher-forced, scored in one pass against the gold next token), while inference runs{' '}
        <strong>sequentially</strong> (autoregressive, one freshly sampled token fed back in per step). There is no
        separate &quot;training-mode&quot; network — only parallel-vs-sequential scheduling of the same computation.
      </p>
    </div>
  );
}
