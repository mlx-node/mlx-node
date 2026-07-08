import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars, renderTokenDisplay } from '../inspector/TopKBars';

/**
 * Chapter 11 (sampling) supplement — top-p (nucleus) sampling, made visual.
 *
 * Ten synthetic tokens with probabilities sorted descending. A slider sets the
 * cumulative cutoff p; we walk the sorted list accumulating probability until
 * the running total reaches p, and everything past that point is the "tail".
 * The LEFT panel shows the original distribution with the tail dimmed and a
 * cutoff marker; the RIGHT panel shows the surviving nucleus RENORMALIZED so
 * it sums to 1 again (reusing the shared TopKBars primitive). Watching the two
 * side by side is the whole lesson: top-p keeps the smallest set whose mass
 * reaches p, then rescales it back to a probability distribution.
 *
 * Self-contained: NO model / worker. The probabilities are a fixed scripted
 * vector. (The chapter defines a private `SliderRow`; this widget inlines an
 * equivalent `accent-primary` range input to avoid importing a chapter-local
 * helper into a reusable widget.)
 */

// Ten synthetic tokens, probabilities already sorted descending and summing to
// exactly 1 — a plausible-looking next-token distribution after "I went to the".
// They MUST sum to 1 so that top-p = 1.0 keeps every token and renormalizes by
// 1.0 (an identity) — i.e. the no-truncation baseline leaves the distribution
// unchanged, as real top-p does.
const TOKENS = [' store', ' park', ' beach', ' gym', ' office', ' movies', ' airport', ' doctor', ' bank', ' moon'];
const PROBS = [0.33, 0.22, 0.14, 0.11, 0.07, 0.055, 0.035, 0.025, 0.01, 0.005];

const COPY = {
  en: {
    header: 'Nucleus (top-p) cutoff',
    nucleusCount: (keptCount: number, total: number) => (
      <>
        Nucleus: <strong className="font-mono text-primary">{keptCount}</strong> of {total} tokens
      </>
    ),
    massKept: 'Mass kept before renormalizing: ',
    cutoffNote: (p: string) => `(cutoff: smallest set with cumulative ≥ ${p})`,
    originalLabel: 'Original — sorted, tail dimmed past the cutoff',
    originalAria: 'Original probabilities with nucleus cutoff',
    rowTitle: (tok: string, prob: string, cumulative: string) => `${tok} · p=${prob} · cumulative=${cumulative}`,
    dashedLineNote: 'The dashed line is the cutoff: everything below it is the discarded tail.',
    nucleusLabel: 'Nucleus — renormalized to sum to 1, then sampled',
    rescaleNote: (mass: string) =>
      `Survivors rescaled by ÷ ${mass} so they sum to 1 — that rescale is why the kept bars jump up.`,
    footnote: 'Illustrative ten-token distribution, sorted and scripted — not live output from the model.',
  },
  zh: {
    header: '核（top-p）截断',
    nucleusCount: (keptCount: number, total: number) => (
      <>
        核：{total} 个 token 中保留 <strong className="font-mono text-primary">{keptCount}</strong> 个
      </>
    ),
    massKept: '重新归一化前保留的概率质量：',
    cutoffNote: (p: string) => `（截断：累计概率 ≥ ${p} 的最小集合）`,
    originalLabel: '原始分布——已排序，截断之后的尾部变暗',
    originalAria: '带核截断标记的原始概率',
    rowTitle: (tok: string, prob: string, cumulative: string) => `${tok} · p=${prob} · 累计=${cumulative}`,
    dashedLineNote: '虚线就是截断线：它下方的一切都是被丢弃的尾部。',
    nucleusLabel: '核——重新归一化到总和为 1，再从中采样',
    rescaleNote: (mass: string) =>
      `幸存者按 ÷ ${mass} 重缩放，使总和重新等于 1——这次重缩放正是被保留的柱子集体跳高的原因。`,
    footnote: '示意用的十个 token 的分布，已排序、预先写定——并非模型的实时输出。',
  },
} as const;

export function NucleusSlider() {
  const copy = COPY[useLocale()];
  const [p, setP] = React.useState(0.9);

  // Walk the (already descending) list accumulating probability; keep the
  // smallest prefix whose cumulative sum reaches p. At least one token is
  // always kept (the top token), even for very small p.
  const { keptCount, keptMass, cumulative } = React.useMemo(() => {
    const cum: number[] = [];
    let acc = 0;
    let count = 0;
    let reached = false;
    for (let i = 0; i < PROBS.length; i++) {
      acc += PROBS[i]!;
      cum.push(acc);
      if (!reached) {
        count = i + 1;
        if (acc >= p) reached = true;
      }
    }
    if (count === 0) count = 1;
    let mass = 0;
    for (let i = 0; i < count; i++) mass += PROBS[i]!;
    return { keptCount: count, keptMass: mass, cumulative: cum };
  }, [p]);

  const ids = TOKENS.map((_, i) => i);

  // Renormalized nucleus for the right panel + the shared TopKBars: kept
  // tokens divided by their mass (so they sum to 1), tail tokens set to 0 so
  // TopKBars dims them with its built-in `prob === 0` styling.
  const renormProbs = React.useMemo(
    () => PROBS.map((prob, i) => (i < keptCount && keptMass > 0 ? prob / keptMass : 0)),
    [keptCount, keptMass],
  );

  const maxProb = Math.max(...PROBS);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <label htmlFor="nucleus-slider-p" className="uppercase tracking-wider">
            p
          </label>
          <input
            id="nucleus-slider-p"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={p}
            onChange={(e) => setP(Number.parseFloat(e.target.value))}
            className="h-2 w-40 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
          />
          <span className="w-10 font-mono text-foreground/80">{p.toFixed(2)}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-foreground/85">
        <span>{copy.nucleusCount(keptCount, TOKENS.length)}</span>
        <span>
          {copy.massKept}
          <strong className="font-mono">{keptMass.toFixed(3)}</strong>
        </span>
        <span className="text-muted-foreground">{copy.cutoffNote(p.toFixed(2))}</span>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* LEFT: original sorted distribution, tail dimmed, cutoff annotated. */}
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{copy.originalLabel}</div>
          <div role="list" aria-label={copy.originalAria} className="space-y-1">
            {TOKENS.map((tok, i) => {
              const prob = PROBS[i]!;
              const inNucleus = i < keptCount;
              const pct = (prob / maxProb) * 100;
              const isLastKept = i === keptCount - 1;
              return (
                <div
                  key={tok}
                  role="listitem"
                  className={[
                    'grid grid-cols-[6rem_minmax(0,1fr)_5.5rem] items-center gap-2 rounded px-1.5 py-1 text-[12px] transition-opacity',
                    inNucleus ? '' : 'opacity-40',
                    isLastKept ? 'border-b border-dashed border-primary/50 pb-1.5' : '',
                  ].join(' ')}
                  title={copy.rowTitle(renderTokenDisplay(tok), prob.toFixed(3), cumulative[i]!.toFixed(3))}
                >
                  <span
                    className={['truncate font-mono', inNucleus ? 'text-foreground/80' : 'text-muted-foreground'].join(
                      ' ',
                    )}
                  >
                    {renderTokenDisplay(tok)}
                  </span>
                  <div className="relative h-4 w-full overflow-hidden rounded bg-muted">
                    <div
                      className={['h-full origin-left', inNucleus ? 'bg-primary' : 'bg-foreground/40'].join(' ')}
                      style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                    />
                  </div>
                  <span className="text-right font-mono text-muted-foreground">
                    {prob.toFixed(3)}
                    <span className="ml-1 text-muted-foreground/60">Σ{cumulative[i]!.toFixed(2)}</span>
                  </span>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] text-muted-foreground">{copy.dashedLineNote}</div>
        </div>

        {/* RIGHT: renormalized nucleus via the shared TopKBars. Tail tokens are
            passed prob=0 so TopKBars dims them automatically. */}
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{copy.nucleusLabel}</div>
          <TopKBars ids={ids} probs={renormProbs} texts={TOKENS} sampledTokenId={-1} runKey={keptCount} />
          <div className="text-[10px] text-muted-foreground">{copy.rescaleNote(keptMass.toFixed(3))}</div>
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
