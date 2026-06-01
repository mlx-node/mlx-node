import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';
import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 11 (sampling) supplement — softmax-with-temperature, one stage at a
 * time. This widget slows ONE temperature down into its four mechanical
 * phases so the learner sees the machine turn, not just the final result:
 *
 *   1. raw logits        z
 *   2. divide by T       z / T
 *   3. exponentiate      e^{z/T}
 *   4. normalize         e^{z/T} / Σ e^{z/T}   ← a probability distribution
 *
 * It loops with a play/pause button (same pattern as the GenerationLoop
 * animation), highlighting which token is "winning" — the current argmax —
 * at each phase. The widget needs every intermediate column (not just the
 * final distribution), so the staged math lives here (self-contained, no
 * model / worker).
 *
 * On `prefers-reduced-motion: reduce` the auto-advance is suppressed and the
 * widget parks on the final (normalized) phase, the meaningful end state.
 */

// Eight synthetic tokens with hand-picked logits — a plausible-looking
// next-token shortlist after "The weather today is".
const TOKENS = [' sunny', ' cloudy', ' warm', ' cold', ' nice', ' rainy', ' mild', ' grey'];
const LOGITS = [3.1, 2.4, 1.9, 1.2, 0.9, 0.4, -0.2, -0.8];
const TEMPERATURE = 0.7; // a common production default

type Phase = 0 | 1 | 2 | 3;
const PHASES: { key: Phase; label: string; caption: string }[] = [
  { key: 0, label: 'raw logits', caption: 'z — the model’s unbounded scores, straight from the LM head.' },
  {
    key: 1,
    label: 'divide by T',
    caption: `z / T — scale by the temperature (T = ${TEMPERATURE}). T < 1 spreads the gaps apart.`,
  },
  {
    key: 2,
    label: 'exponentiate',
    caption: 'e^{z/T} — all positive now, and the gaps are stretched multiplicatively.',
  },
  {
    key: 3,
    label: 'normalize',
    caption: 'e^{z/T} / Σ — divide by the sum so the eight values add to 1: a distribution.',
  },
];

const FRAME_MS = 1600;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

export function SoftmaxTemperatureSteps() {
  const reducedMotion = usePrefersReducedMotion();
  // Park on the final phase when motion is reduced; otherwise start at raw.
  const [phase, setPhase] = React.useState<Phase>(reducedMotion ? 3 : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setPhase((p) => ((p + 1) % 4) as Phase), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  // Staged math. Each column is derived from the previous one so the four
  // phases line up exactly with the formula.
  const scaled = React.useMemo(() => LOGITS.map((z) => z / TEMPERATURE), []);
  // True e^{z/T}. These logits are small (max z/T ≈ 4.4) so there is no
  // overflow risk — we show the textbook value directly, with NO max-subtraction
  // trick. That keeps the displayed "exponentiate" numbers exactly e^{z/T} (so
  // they match the phase label), and makes the Σ row below the true normalizing
  // sum that the normalize step divides by.
  const exps = React.useMemo(() => scaled.map((s) => Math.exp(s)), [scaled]);
  const expSum = React.useMemo(() => exps.reduce((s, e) => s + e, 0), [exps]);
  const probs = React.useMemo(() => exps.map((e) => (expSum > 0 ? e / expSum : 0)), [exps, expSum]);

  // Bars are shown ONLY for the two phases whose values are non-negative and so
  // can be drawn honestly as widths AND labelled with their true value: the
  // exponentiate phase (true e^{z/T}) and the normalize phase (true
  // probabilities). Raw logits and z/T can be negative, so for those phases we
  // show the numeric column only and tell the learner the bars arrive once we
  // exponentiate — that way a token never displays two different numbers at once.
  const showBars = phase === 2 || phase === 3;
  const barValues = phase === 3 ? probs : exps; // only read when showBars is true

  // The true numeric value per token for the current phase (shown in the
  // numeric column; the exp/normalize bars reuse these same true values).
  const numericValues = phase === 0 ? LOGITS : phase === 1 ? scaled : phase === 2 ? exps : probs;

  // "Winning" token = current argmax. argmax is invariant across all four
  // phases (monotonic transforms), but we recompute per phase so the label is
  // honest about what it's reading.
  const winnerIndex = React.useMemo(() => {
    let best = 0;
    for (let i = 1; i < numericValues.length; i++) if (numericValues[i]! > numericValues[best]!) best = i;
    return best;
  }, [numericValues]);

  const ids = TOKENS.map((_, i) => i);
  const current = PHASES[phase]!;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Softmax with temperature, phase by phase
        </div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-pressed={playing}
          disabled={reducedMotion}
          className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
        >
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
      </div>

      <MathDisplay latex={String.raw`p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \qquad T = ${TEMPERATURE}`} />

      {/* Phase rail — four steps; the active one lights up. */}
      <div className="flex flex-wrap items-center gap-1.5">
        {PHASES.map((p, i) => (
          <React.Fragment key={p.key}>
            <button
              type="button"
              onClick={() => {
                setPlaying(false);
                setPhase(p.key);
              }}
              aria-pressed={phase === p.key}
              className={[
                'rounded px-2 py-1 font-mono text-[11px] transition-colors',
                phase === p.key ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              {i + 1}. {p.label}
            </button>
            {i < PHASES.length - 1 ? (
              <span aria-hidden className="text-muted-foreground/50">
                →
              </span>
            ) : null}
          </React.Fragment>
        ))}
      </div>

      <p className="min-h-[2.5em] text-[12px] text-foreground/85">{current.caption}</p>

      {/* Numeric column + bars. The numeric value is always the TRUE phase
          value. Bars are shown only once the values are non-negative (the
          exponentiate and normalize phases) and are labelled with that same
          true value, so a token never shows two different numbers at once. */}
      <div className="grid gap-3 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]">
        <div className="space-y-0.5 rounded-md border border-border/60 bg-muted/20 p-2 font-mono text-[11px]">
          <div className="mb-1 flex items-center justify-between text-muted-foreground">
            <span>token</span>
            <span>{current.label}</span>
          </div>
          {TOKENS.map((tok, i) => (
            <div
              key={tok}
              className={[
                'flex items-center justify-between rounded px-1 py-0.5',
                i === winnerIndex ? 'bg-primary/10 text-primary' : 'text-foreground/80',
              ].join(' ')}
            >
              <span className="truncate">{tok.trim()}</span>
              <span>{phase === 3 ? numericValues[i]!.toFixed(3) : numericValues[i]!.toFixed(2)}</span>
            </div>
          ))}
          {phase === 2 || phase === 3 ? (
            <div className="mt-1 flex items-center justify-between border-t border-border/60 pt-1 text-muted-foreground">
              <span>Σ</span>
              <span>{phase === 2 ? expSum.toFixed(2) : probs.reduce((s, p) => s + p, 0).toFixed(3)}</span>
            </div>
          ) : null}
        </div>

        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">
            Winning token: <span className="font-mono text-primary">{TOKENS[winnerIndex]!.trim()}</span>
            {phase < 3 ? ' (argmax — unchanged by these monotonic steps)' : ' (highest probability)'}
          </div>
          {showBars ? (
            <TopKBars ids={ids} probs={barValues} texts={TOKENS} sampledTokenId={winnerIndex} runKey={phase} />
          ) : (
            <div className="flex min-h-[8rem] items-center justify-center rounded-md border border-dashed border-border/60 px-3 text-center text-[11px] text-muted-foreground">
              {phase === 0
                ? 'Raw logits can be negative, so there is nothing to draw as bars yet — exponentiating (step 3) makes every value positive.'
                : 'After ÷ T these are still raw scores and can be negative; the bars appear once we exponentiate in step 3.'}
            </div>
          )}
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative eight-token logits, scripted to show the four phases — not live output from the model. Bars appear
        once the values are non-negative (after exponentiating); every number shown is the true per-phase value.
      </p>
    </div>
  );
}
