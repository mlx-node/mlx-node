import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';
import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 4 supplement — WHY attention divides QKᵀ by √d_k.
 *
 * For random Q, K with ~unit-variance components, a dot product q·k is a sum of
 * d_k independent products, so its standard deviation grows like √d_k. As the
 * head dimension widens, raw scores blow up and softmax saturates toward a
 * one-hot distribution — the top key heads toward 1.0 and gradients into the
 * other keys vanish. Dividing by √d_k cancels exactly that growth, keeping the
 * score variance ~1 so the distribution stays informative.
 *
 * To make the point crisply we fix a set of per-key "unit" alignments `U`
 * (the d=1 scores) and reconstruct a d-dependent raw score as `raw = U · √d`.
 * Then:
 *   • unscaled = softmax(raw)            → grows peaky as d↑ (≈ one-hot at 256)
 *   • scaled   = softmax(raw / √d)
 *              = softmax(U)              → INVARIANT to d (top ≈ 0.523 always)
 *
 * All numbers are formula-derived — no model, no worker.
 */

// Fixed per-key "unit scores": the alignments q·k would have at d_k = 1. These
// are the seed the d-sweep scales up by √d. Six synthetic keys, descending.
const UNIT_SCORES = [2.0, 1.2, 0.5, 0.0, -0.7, -1.3];

// Discrete head-dim stops. 256 is Qwen3.5-0.8B's actual head_dim (√256 = 16).
const D_STOPS = [4, 16, 64, 256];

// Stand-in token ids/labels so we can reuse the TopKBars primitive. These are
// "keys" (positions a query could attend to), not vocabulary tokens.
const KEY_IDS = UNIT_SCORES.map((_, i) => i);
const KEY_TEXTS = UNIT_SCORES.map((_, i) => `key ${i}`);

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 0));
}

export function ScaledSoftmax() {
  const [dIdx, setDIdx] = React.useState(D_STOPS.length - 1); // default 256
  const d = D_STOPS[dIdx]!;
  const sqrtD = Math.sqrt(d);

  // raw_i = U_i · √d  → grows with the head dimension.
  const raw = UNIT_SCORES.map((u) => u * sqrtD);
  const unscaled = softmax(raw);
  // softmax(raw / √d) === softmax(U): invariant to d.
  const scaled = softmax(raw.map((r) => r / sqrtD));

  const unscaledTop = Math.max(...unscaled);
  const scaledTop = Math.max(...scaled);

  // Remount the bars when d changes so the grow-in animation re-fires.
  const runKey = dIdx + 1;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Why divide by √d_k</div>
        <div className="text-[11px] text-muted-foreground">Same key alignments, swept across head_dim</div>
      </div>

      <MathDisplay latex={String.raw`\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)`} />

      <div className="space-y-1">
        <label htmlFor="scaled-softmax-d" className="block text-xs text-muted-foreground">
          head_dim <span className="font-mono text-foreground/85">d_k = {d}</span> (
          <span className="font-mono">√d_k = {sqrtD}</span>)
        </label>
        <input
          id="scaled-softmax-d"
          type="range"
          min={0}
          max={D_STOPS.length - 1}
          step={1}
          value={dIdx}
          onChange={(e) => setDIdx(Number(e.target.value))}
          className="w-full accent-primary"
          list="scaled-softmax-stops"
          aria-valuetext={`head dimension ${d}`}
        />
        <datalist id="scaled-softmax-stops">
          {D_STOPS.map((stop, i) => (
            <option key={stop} value={i} label={`${stop}`} />
          ))}
        </datalist>
        <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
          {D_STOPS.map((stop) => (
            <span key={stop}>{stop}</span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <div className="flex items-baseline justify-between text-[11px]">
            <span className="font-medium text-foreground/85">without ÷√d</span>
            <span className="font-mono text-muted-foreground">top = {unscaledTop.toFixed(3)}</span>
          </div>
          <TopKBars ids={KEY_IDS} probs={unscaled} texts={KEY_TEXTS} sampledTokenId={-1} runKey={runKey} />
          <p className="text-[11px] text-muted-foreground">
            softmax(<span className="font-mono">U · √d</span>) — peaks higher as <span className="font-mono">d</span>{' '}
            grows; near one-hot at 256.
          </p>
        </div>
        <div className="space-y-1">
          <div className="flex items-baseline justify-between text-[11px]">
            <span className="font-medium text-foreground/85">with ÷√d</span>
            <span className="font-mono text-muted-foreground">top = {scaledTop.toFixed(3)}</span>
          </div>
          <TopKBars ids={KEY_IDS} probs={scaled} texts={KEY_TEXTS} sampledTokenId={-1} runKey={runKey} />
          <p className="text-[11px] text-muted-foreground">
            softmax(<span className="font-mono">U · √d / √d</span>) = softmax(<span className="font-mono">U</span>) —{' '}
            <strong>identical at every</strong> <span className="font-mono">d</span>.
          </p>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        As <span className="font-mono">d_k</span> grows, a random dot product&apos;s spread grows like{' '}
        <span className="font-mono">√d_k</span>, so the <strong>unscaled</strong> scores blow up and softmax collapses
        toward one key — vanishing gradients. Dividing by <span className="font-mono">√d_k</span> cancels exactly that
        growth, so the <strong>scaled</strong> distribution is unchanged by the head width.{' '}
        <strong>Qwen3.5-0.8B uses head_dim = 256</strong>, so it divides every score by{' '}
        <span className="font-mono">√256 = 16</span>.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — the six key alignments are fixed synthetic numbers scaled by √d, not live output from the model.
      </p>
    </div>
  );
}
