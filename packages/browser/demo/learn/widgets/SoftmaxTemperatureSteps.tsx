import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars } from '../inspector/TopKBars';
import { DiagramFrame, useStepPlayer } from '../motion';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 11 (sampling) supplement — softmax-with-temperature, one stage at a
 * time. This widget slows one temperature down into its four mechanical
 * phases — with a T toggle (0.2 / 0.7 / 2.0) so the same eight logits can be
 * watched sharpening and flattening — so the learner sees the machine turn,
 * not just the final result:
 *
 *   1. raw logits        z
 *   2. divide by T       z / T
 *   3. exponentiate      e^{z/T}
 *   4. normalize         e^{z/T} / Σ e^{z/T}   ← a probability distribution
 *
 * It loops, highlighting which token is "winning" — the current argmax — at
 * each phase. The widget needs every intermediate column (not just the final
 * distribution), so the staged math lives here (self-contained, no model /
 * worker).
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval`, private
 * `usePrefersReducedMotion`, header row, play button and caption paragraph.
 * There is no `<svg>` here — the "diagram" is a numeric column plus bars — so
 * this is a kit-only adoption: the frame, the player and the caption slot, and
 * none of the SVG skin tokens (there is nothing for them to apply to).
 *
 * On `prefers-reduced-motion: reduce` the auto-advance is suppressed and the
 * widget parks on the final (normalized) phase, the meaningful end state —
 * `useStepPlayer`'s default `restFrame` (last frame) is exactly that, so the
 * behaviour carries over unchanged.
 *
 * The T picker and the phase rail stay widget-owned: `DiagramFrame` renders
 * play/step only, so the rail is also what tells the reader which of the four
 * phases is on screen (the frame has no counter slot).
 */

// Eight synthetic tokens with hand-picked logits — a plausible-looking
// next-token shortlist after "The weather today is".
const TOKENS = [' sunny', ' cloudy', ' warm', ' cold', ' nice', ' rainy', ' mild', ' grey'];
const LOGITS = [3.1, 2.4, 1.9, 1.2, 0.9, 0.4, -0.2, -0.8];
// Three selectable presets over the SAME eight logits: sharpen (0.2), the
// common production default (0.7), and flatten (2.0).
const TEMPERATURES = [0.2, 0.7, 2.0] as const;
const DEFAULT_TEMPERATURE = 0.7;

type Phase = 0 | 1 | 2 | 3;
type PhaseSpec = { key: Phase; label: string; caption: (T: number) => React.ReactNode };
const PHASES: PhaseSpec[] = [
  { key: 0, label: 'raw logits', caption: () => 'z — the model’s unbounded scores, straight from the LM head.' },
  {
    key: 1,
    label: 'divide by T',
    caption: (T) =>
      `z / T — scale by the temperature (T = ${T}). ${
        T < 1 ? 'T < 1 spreads the gaps apart.' : 'T > 1 squashes the gaps together.'
      }`,
  },
  {
    key: 2,
    label: 'exponentiate',
    caption: () => (
      <>
        e<sup>z/T</sup> — all positive now, and the gaps are stretched multiplicatively.
      </>
    ),
  },
  {
    key: 3,
    label: 'normalize',
    caption: () => (
      <>
        e<sup>z/T</sup> / Σ — divide by the sum so the eight values add to 1: a distribution.
      </>
    ),
  },
];

const ZH_PHASES: PhaseSpec[] = [
  { key: 0, label: '原始 logits', caption: () => 'z——模型的无界分数，直接来自 LM head。' },
  {
    key: 1,
    label: '除以 T',
    caption: (T) => `z / T——按温度缩放（T = ${T}）。${T < 1 ? 'T < 1 把差距拉大。' : 'T > 1 把差距压扁。'}`,
  },
  {
    key: 2,
    label: '取指数',
    caption: () => (
      <>
        e<sup>z/T</sup>——现在全为正，差距被乘法式地拉开。
      </>
    ),
  },
  {
    key: 3,
    label: '归一化',
    caption: () => (
      <>
        e<sup>z/T</sup> / Σ——除以总和，八个值加起来等于 1：一个概率分布。
      </>
    ),
  },
];

// Play / Pause are NOT here any more: those verbs now belong to the shared
// `StepControls` that `DiagramFrame` renders, which carries its own en/zh pair
// so every animated widget in the course says the same word.
const COPY = {
  en: {
    phases: PHASES,
    header: 'Softmax with temperature, phase by phase',
    temperatureLabel: 'Temperature',
    temperatureAria: 'Temperature preset',
    sharpens: 'sharpens — closer to greedy',
    flattens: 'flattens — more diverse',
    expBarsAria: 'e^(z/T) per token, scaled to the largest value',
    tokenColumnLabel: 'token',
    winningLabel: 'Winning token: ',
    argmaxNote: ' (argmax — unchanged by these monotonic steps)',
    highestProbNote: ' (highest probability)',
    rawEmpty:
      'Raw logits can be negative, so there is nothing to draw as bars yet — exponentiating (step 3) makes every value positive.',
    scaledEmpty: 'After ÷ T these are still raw scores and can be negative; the bars appear once we exponentiate in step 3.',
    footnote:
      'Illustrative eight-token logits, scripted to show the four phases — not live output from the model. Bars appear once the values are non-negative (after exponentiating); every number shown is the true per-phase value. Switch T and re-watch the normalize phase: at 0.2 nearly all mass piles onto the top token, at 2.0 it spreads across all eight.',
  },
  zh: {
    phases: ZH_PHASES,
    header: '带温度的 softmax，逐阶段拆解',
    temperatureLabel: 'Temperature',
    temperatureAria: 'Temperature 预设',
    sharpens: '更尖——更接近贪心',
    flattens: '更平——更多样',
    expBarsAria: '每个 token 的 e^(z/T)，按最大值缩放',
    tokenColumnLabel: 'token',
    winningLabel: '胜出 token：',
    argmaxNote: '（argmax——不受这些单调变换影响）',
    highestProbNote: '（概率最高）',
    rawEmpty: '原始 logits 可能为负，所以还画不出柱子——第 3 步取指数后，所有值都会变成正数。',
    scaledEmpty: '除以 T 之后这些仍是可能为负的原始分数；等到第 3 步取指数后柱子才会出现。',
    footnote:
      '示意用的八个 token 的 logits，为展示四个阶段而写定——并非模型的实时输出。柱子在数值变为非负之后（取指数后）才出现；显示的每个数字都是该阶段的真实值。切换 T 再看一遍归一化阶段：0.2 时几乎所有概率质量都压在最上面的 token 上，2.0 时则摊到全部八个上。',
  },
} as const;

// Per-phase honest formatting. At T = 0.2 the exponentiate column reaches
// e^{15.5} ≈ 5.39 million — still far from float overflow, but too wide to
// print with toFixed, so very large values switch to exponential notation.
function fmtVal(v: number, phase: Phase): string {
  if (phase === 3) return v.toFixed(3);
  if (Math.abs(v) >= 100_000) return v.toExponential(1);
  return v.toFixed(2);
}

const FRAME_MS = 1600;

/**
 * Exponentiate-phase bars: true e^{z/T} values, widths scaled by the max.
 * TopKBars is reserved for the normalize phase — its value label prints
 * `toFixed(3)`, which can't show T = 0.2's e^{z/T} ≈ 5.4e6 readably. Here the
 * label uses the same per-phase formatter as the numeric column, so a token
 * never shows two different numbers at once.
 */
function ExpBars({ values, winnerIndex }: { values: number[]; winnerIndex: number }) {
  const copy = COPY[useLocale()];
  const max = Math.max(1e-9, ...values);
  return (
    <div
      role="list"
      aria-label={copy.expBarsAria}
      className="space-y-1 rounded-md border border-border bg-background p-3"
    >
      {TOKENS.map((tok, i) => {
        const isWinner = i === winnerIndex;
        const pct = (values[i]! / max) * 100;
        return (
          <div
            key={tok}
            role="listitem"
            className={[
              'grid grid-cols-[8rem_minmax(0,1fr)_4rem] items-center gap-2 rounded px-1.5 py-1 text-[12px]',
              isWinner ? 'bg-primary/10' : '',
            ].join(' ')}
          >
            <span
              className={['truncate font-mono', isWinner ? 'font-semibold text-primary' : 'text-foreground/80'].join(
                ' ',
              )}
            >
              {tok.trim()}
            </span>
            <div className="relative h-4 w-full overflow-hidden rounded bg-muted">
              <div
                className={['h-full', isWinner ? 'bg-primary' : 'bg-foreground/40'].join(' ')}
                style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
              />
            </div>
            <span className="text-right font-mono text-muted-foreground">{fmtVal(values[i]!, 2)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function SoftmaxTemperatureSteps() {
  const locale = useLocale();
  const copy = COPY[locale];
  // The sweep length IS the phase count, so `player.frame` always addresses a
  // real phase spec — read `key` off the spec instead of casting number→Phase.
  const player = useStepPlayer(PHASES.length, { frameMs: FRAME_MS });
  const current = copy.phases[player.frame]!;
  const phase = current.key;
  const [temperature, setTemperature] = React.useState<number>(DEFAULT_TEMPERATURE);

  // Staged math. Each column is derived from the previous one so the four
  // phases line up exactly with the formula.
  const scaled = React.useMemo(() => LOGITS.map((z) => z / temperature), [temperature]);
  // True e^{z/T}. Even at T = 0.2 the largest value is e^{15.5} ≈ 5.4e6 — no
  // overflow risk — so we show the textbook value directly, with NO max-subtraction
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

  // Every caption this widget can show — FOUR phases x THREE presets, not four.
  // Phase 1's sentence interpolates the preset AND branches on `T < 1`, so a
  // list built from the current temperature alone would let the slot resize
  // when the reader switches T. Which branch wraps longest also differs by
  // locale, so the whole product is passed and the grid measures it.
  const captions: React.ReactNode[] = TEMPERATURES.flatMap((t) => copy.phases.map((p) => p.caption(t)));

  return (
    <DiagramFrame
      title={copy.header}
      player={player}
      locale={locale}
      caption={current.caption(temperature)}
      captions={captions}
      note={copy.footnote}
    >
      <div className="space-y-3">
        {/* Temperature presets — same eight logits, three temperatures. Not the
            frame's business: it owns play/step only, so widget controls sit
            here, above the body. */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.temperatureLabel}</span>
          <SegmentedToggle
            value={temperature}
            onChange={setTemperature}
            ariaLabel={copy.temperatureAria}
            options={TEMPERATURES.map((t) => ({ value: t, label: `T = ${t}` }))}
          />
          <span className="text-[11px] text-muted-foreground">
            {temperature < 1 ? copy.sharpens : copy.flattens}
          </span>
        </div>

        <MathDisplay latex={String.raw`p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \qquad T = ${temperature}`} />

        {/* Phase rail — four steps; the active one lights up. Also the widget's
            "step 3 of 4" readout: StepControls has no counter slot, and these
            captions never say which phase they are. `goTo` pauses first, which
            is what the hand-rolled onClick did with its own setPlaying(false). */}
        <div className="flex flex-wrap items-center gap-1.5">
          {copy.phases.map((p, i) => (
            <React.Fragment key={p.key}>
              <button
                type="button"
                onClick={() => player.goTo(p.key)}
                aria-pressed={phase === p.key}
                className={[
                  'rounded px-2 py-1 font-mono text-[11px] transition-colors',
                  phase === p.key ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {i + 1}. {p.label}
              </button>
              {i < copy.phases.length - 1 ? (
                <span aria-hidden className="text-muted-foreground/50">
                  →
                </span>
              ) : null}
            </React.Fragment>
          ))}
        </div>

        {/* Numeric column + bars. The numeric value is always the TRUE phase
            value. Bars are shown only once the values are non-negative (the
            exponentiate and normalize phases) and are labelled with that same
            true value, so a token never shows two different numbers at once. */}
        <div className="grid gap-3 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]">
          <div className="space-y-0.5 rounded-md border border-border/60 bg-muted/20 p-2 font-mono text-[11px]">
            <div className="mb-1 flex items-center justify-between text-muted-foreground">
              <span>{copy.tokenColumnLabel}</span>
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
                <span>{fmtVal(numericValues[i]!, phase)}</span>
              </div>
            ))}
            {phase === 2 || phase === 3 ? (
              <div className="mt-1 flex items-center justify-between border-t border-border/60 pt-1 text-muted-foreground">
                <span>Σ</span>
                <span>{phase === 2 ? fmtVal(expSum, 2) : probs.reduce((s, p) => s + p, 0).toFixed(3)}</span>
              </div>
            ) : null}
          </div>

          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">
              {copy.winningLabel}
              <span className="font-mono text-primary">{TOKENS[winnerIndex]!.trim()}</span>
              {phase < 3 ? copy.argmaxNote : copy.highestProbNote}
            </div>
            {phase === 3 ? (
              <TopKBars
                ids={ids}
                probs={probs}
                texts={TOKENS}
                sampledTokenId={winnerIndex}
                // Changes on every phase OR temperature change so the grow-in
                // animation re-fires when the distribution actually moves.
                runKey={phase * 100 + Math.round(temperature * 10)}
              />
            ) : phase === 2 ? (
              <ExpBars values={exps} winnerIndex={winnerIndex} />
            ) : (
              <div className="flex min-h-[8rem] items-center justify-center rounded-md border border-dashed border-border/60 px-3 text-center text-[11px] text-muted-foreground">
                {phase === 0 ? copy.rawEmpty : copy.scaledEmpty}
              </div>
            )}
          </div>
        </div>
      </div>
    </DiagramFrame>
  );
}
