import * as React from 'react';

/**
 * Chapter 13 (Training) supplement — exposure bias, the reason inference drifts.
 *
 * TeacherForcingAnimation shows ONE training step: per-position distributions
 * and the cross-entropy bars. This widget shows the OTHER half of the story —
 * what happens ACROSS steps at inference time, and why it differs from training.
 *
 * Two lanes advance in lockstep, one token per step:
 *   Lane A "Teacher forcing (training)": the next input is always the GOLD
 *     token (chapter: "the input at position i+1 is the TRUE token from position
 *     i"). The lane can never leave the reference sentence — every step lands on
 *     green, regardless of what the model would have predicted.
 *   Lane B "Free-running (inference)": the model appends its OWN prediction. It
 *     tracks the reference until step 5, where it predicts " couch" instead of
 *     the gold " mat". From then on the context is off-reference, so the lane
 *     keeps drifting (" couch", " and", " knocked") — shown in a drift color.
 *
 * The point is exposure bias: training never lets the model consume its own
 * mistakes, so at inference small errors compound. Synthetic / scripted — no
 * model, no worker, no WASM.
 */

// The shared gold reference both lanes are scored against.
const REFERENCE = ['The', ' cat', ' sat', ' on', ' the', ' mat'] as const;

// Free-running inference output. It matches REFERENCE up to DIVERGE_STEP, then
// the model picks its own (plausible but different) continuation and drifts.
const FREE_RUN = ['The', ' cat', ' sat', ' on', ' the', ' couch', ' and', ' knocked'] as const;

// Index of the first token where the free-running lane leaves the reference.
// At this step the model predicts " couch"; teacher forcing would force-feed " mat".
const DIVERGE_STEP = 5;

const TOTAL_STEPS = FREE_RUN.length;
const STEP_MS = 1500;

function renderToken(t: string): string {
  return t.startsWith(' ') ? '·' + t.slice(1) : t;
}

type Cell = {
  text: string;
  /** 'gold' = on the reference, 'drift' = the model has left the reference. */
  kind: 'gold' | 'drift';
};

/** Build the free-running lane up to (and including) `count` revealed tokens. */
function freeRunCells(count: number): Cell[] {
  return FREE_RUN.slice(0, count).map((text, i) => ({
    text,
    kind: i >= DIVERGE_STEP ? 'drift' : 'gold',
  }));
}

export function ExposureBias() {
  const [step, setStep] = React.useState(0);
  const [playing, setPlaying] = React.useState(() =>
    typeof window !== 'undefined' ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches : true,
  );

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setStep((s) => (s + 1) % (TOTAL_STEPS + 1)), STEP_MS);
    return () => window.clearInterval(t);
  }, [playing]);

  // `step` runs 0..TOTAL_STEPS; the number of revealed tokens is `step`, so the
  // final frame (step === TOTAL_STEPS) shows the whole sequence before looping.
  const revealed = step;
  const goldCells = REFERENCE.slice(0, revealed);
  const freeCells = freeRunCells(revealed);
  const diverged = revealed > DIVERGE_STEP;
  const atDivergence = revealed === DIVERGE_STEP + 1;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Exposure bias — why inference drifts
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
            step {Math.min(revealed, TOTAL_STEPS)}/{TOTAL_STEPS}
          </span>
        </div>
      </div>

      {/* Lane A — teacher forcing: always force-fed the gold token. */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Teacher forcing (training)</span>
          <span className="text-[10px] text-muted-foreground">next input = true token</span>
        </div>
        <div className="flex min-h-9 flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
          {goldCells.map((t, i) => (
            <span
              key={`gold-${i}`}
              className="rounded bg-emerald-500/70 px-1.5 py-0.5 font-mono text-[11px] text-white outline outline-1 outline-emerald-400"
            >
              {renderToken(t)}
            </span>
          ))}
          {revealed === 0 ? <span className="font-mono text-[11px] text-muted-foreground/60">(start)</span> : null}
          <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
            ▮
          </span>
        </div>
      </div>

      {/* Lane B — free-running inference: feeds its own predictions back in. */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Free-running (inference)</span>
          <span className="text-[10px] text-muted-foreground">next input = own prediction</span>
        </div>
        <div className="flex min-h-9 flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
          {freeCells.map((c, i) => (
            <span
              key={`free-${i}`}
              className={[
                'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                c.kind === 'gold'
                  ? 'bg-emerald-500/70 text-white outline outline-1 outline-emerald-400'
                  : 'bg-rose-500/70 text-white outline outline-1 outline-rose-400',
              ].join(' ')}
            >
              {renderToken(c.text)}
            </span>
          ))}
          {revealed === 0 ? <span className="font-mono text-[11px] text-muted-foreground/60">(start)</span> : null}
          <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
            ▮
          </span>
        </div>
      </div>

      {/* Divergence note — appears the moment the lanes split, then persists. */}
      <div
        className={[
          'rounded-md border px-3 py-2 text-[12px] transition-colors duration-300',
          diverged
            ? 'border-rose-500/40 bg-rose-500/5 text-foreground/90'
            : 'border-border/40 bg-muted/20 text-muted-foreground/70',
        ].join(' ')}
      >
        {diverged ? (
          <>
            <span className={atDivergence ? 'font-semibold text-rose-700 dark:text-rose-300' : 'text-foreground/90'}>
              Divergence:
            </span>{' '}
            model predicted{' '}
            <span className="font-mono text-rose-700 dark:text-rose-300">{renderToken(FREE_RUN[DIVERGE_STEP]!)}</span> —
            training would have force-fed{' '}
            <span className="font-mono text-emerald-700 dark:text-emerald-300">
              {renderToken(REFERENCE[DIVERGE_STEP]!)}
            </span>
            . Now off-reference, the next tokens drift further.
          </>
        ) : (
          'Both lanes track the reference so far — the model has not yet had to recover from one of its own choices.'
        )}
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — the divergence is scripted, not live output from the model.
      </p>

      <p className="text-[12px] text-foreground/85">
        Teacher forcing means training <em>never</em> lets the model see its own mistakes: at every position the input
        is the true previous token. But at inference the model must consume its <em>own</em> outputs, so one off-gold
        choice shifts the context onto a path it was never trained on, and small errors compound into drift. That gap
        between the teacher-forced training distribution and the free-running inference distribution is{' '}
        <strong>exposure bias</strong>.
      </p>
    </div>
  );
}
