import { cn } from '@/lib/utils';
import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 7 supplement — WHY RMSNorm bothers to pin the per-token magnitude.
 *
 * This widget is the "felt problem" behind the chapter's opening claim: if a
 * token's activation magnitude drifts upward across layers, the downstream
 * softmax (think attention scores) saturates — it collapses onto a single
 * option and stops distinguishing the rest. RMSNorm holds the per-token
 * magnitude near √hidden_dim ≈ 32, which keeps that softmax informative.
 *
 * Distinct from `ScaledSoftmax` (chapter 3): that widget shows the ÷√d_k fix
 * for the *head-dimension* growth. This one shows the magnitude→saturation
 * relationship that RMSNorm exists to prevent.
 *
 * Mechanism (all formula-derived — no model, no worker):
 *   • A fixed base pattern of 6 "scores" (one per token A…F). The pattern is
 *     constant; only its overall MAGNITUDE varies.
 *   • The magnitude slider sets the per-token L2 `M`. We anchor M = 32
 *     (√1024, the healthy RMSNorm regime) to a gentle scale and let it drift
 *     up to ~256. The logit scale applied to the pattern is `s = M / 32`, so
 *     larger magnitude ⇒ larger logits ⇒ sharper softmax.
 *   • softmax(s · base) is a correct normalized softmax: scaling the logits by
 *     a larger factor provably sharpens it toward one-hot.
 *
 * Schematic illustration of the principle — the real saturation happens in
 * attention scores, not in a literal six-way readout.
 */

// Fixed score pattern: the *shape* of the per-token scores, distinct values,
// roughly unit-scale. Magnitude is what the slider varies, not this pattern.
const BASE_SCORES = [1.0, 0.6, 0.3, 0.1, -0.2, -0.6];
const TOKEN_LABELS = ['token A', 'token B', 'token C', 'token D', 'token E', 'token F'];
const TOKEN_IDS = BASE_SCORES.map((_, i) => i);

// Per-token magnitude (L2) the slider sweeps. √1024 ≈ 32 is the regime RMSNorm
// pins; drift carries it upward until the softmax collapses.
const HEALTHY_M = 32; // √hidden_dim for Qwen3.5-0.8B's 1024-dim hidden state
const MIN_M = 16;
const MAX_M = 256;
const DRIFTED_M = 192; // a clearly-saturated "magnitude drifted up" stop

// A distribution counts as "saturated" once one option dominates this hard.
const SATURATION_THRESHOLD = 0.9;

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 0));
}

export function SaturatedSoftmax() {
  const [magnitude, setMagnitude] = React.useState(HEALTHY_M);

  // Logit scale: gentle at the healthy magnitude, growing linearly with drift.
  // s = M / 32, so M = 32 ⇒ s = 1 (informative); M = 256 ⇒ s = 8 (one-hot-ish).
  const scale = magnitude / HEALTHY_M;
  const logits = BASE_SCORES.map((v) => v * scale);
  const probs = softmax(logits);
  const top = Math.max(...probs);
  const saturated = top >= SATURATION_THRESHOLD;

  // Remount the bars when the magnitude changes so the grow-in animation re-fires.
  const runKey = Math.round(magnitude);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          When magnitude drifts, softmax saturates
        </div>
        <div className="text-[11px] text-muted-foreground">Same score pattern, swept by overall magnitude</div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Six fixed scores (one per token). Their <em>pattern</em> never changes — only the overall{' '}
        <strong>magnitude</strong> does. Drag it up and the downstream{' '}
        <code className="text-foreground/80">softmax</code> stops spreading attention and collapses onto a single token.
      </p>

      <div className="space-y-2">
        <SegmentedToggle
          value={magnitude}
          onChange={setMagnitude}
          ariaLabel="Per-token magnitude preset"
          options={[
            { value: HEALTHY_M, label: 'Healthy ≈ √1024 ≈ 32' },
            { value: DRIFTED_M, label: 'Drifted / saturated' },
          ]}
        />
        <div className="space-y-1">
          <div className="flex items-baseline justify-between text-xs text-muted-foreground">
            <label htmlFor="saturated-softmax-mag" className="uppercase tracking-wider">
              per-token magnitude (L2)
            </label>
            <span className="font-mono text-foreground/80">
              {Math.round(magnitude)} (logit scale {scale.toFixed(1)}×)
            </span>
          </div>
          <input
            id="saturated-softmax-mag"
            type="range"
            min={MIN_M}
            max={MAX_M}
            step={1}
            value={magnitude}
            onChange={(e) => setMagnitude(Number(e.target.value))}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
            aria-valuetext={`per-token magnitude ${Math.round(magnitude)}`}
          />
          <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
            <span className="text-primary">√1024 ≈ 32 · informative</span>
            <span className="text-amber-600 dark:text-amber-500">drifted · saturated</span>
          </div>
        </div>
      </div>

      <div className="space-y-1">
        <div className="flex items-baseline justify-between text-[11px]">
          <span className={cn('font-medium', saturated ? 'text-amber-600 dark:text-amber-500' : 'text-primary')}>
            {saturated ? 'saturated — one token dominates' : 'informative — attention is spread'}
          </span>
          <span className="font-mono text-muted-foreground">top = {top.toFixed(3)}</span>
        </div>
        <TopKBars ids={TOKEN_IDS} probs={probs} texts={TOKEN_LABELS} sampledTokenId={-1} runKey={runKey} />
      </div>

      <p className="text-[12px] text-foreground/85">
        Bigger activation magnitude scales every score up by the same factor, which makes{' '}
        <code className="text-foreground/80">softmax</code> spike onto the single largest option and ignore the rest —
        the distribution <strong>saturates</strong>. A near-one-hot distribution carries almost no information about how
        the other options compare, and the gradient into them all but vanishes. <strong>RMSNorm</strong> pins each
        token's magnitude near <span className="font-mono">√hidden_dim ≈ 32</span> (here{' '}
        <span className="font-mono">√1024</span> for Qwen3.5-0.8B), keeping the downstream distribution informative
        instead of letting drift collapse it.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — a schematic six-way softmax over fixed synthetic scores, not a literal model readout. The real
        saturation this prevents happens in the attention scores. The softmax itself is exact: scaling the scores by a
        larger factor genuinely sharpens it toward one-hot.
      </p>
    </div>
  );
}
