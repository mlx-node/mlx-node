import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';
import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 1 (overview) supplement — demystify "248k logits" by shrinking the
 * vocabulary to FIVE friendly fake tokens and showing the whole softmax
 * pipeline on numbers you can read: 5 raw logits → exp → divide by the sum →
 * 5 probabilities that add to 1.
 *
 * Self-contained: NO model / worker / WASM. The logits are a fixed scripted
 * vector; one "sharpness" slider scales them (multiply by 0.3…3) so the
 * learner can watch the same five numbers go from nearly-flat to spiky and see
 * the bars (reused TopKBars) recompute live. There is no "sampled" token here —
 * this widget is purely about the logits → probabilities transform — so the
 * bars are rendered with `sampledTokenId={-1}` (no row highlighted), exactly
 * like the forward-pass phase of the GenerationLoop animation.
 */

// Five friendly fake tokens with hand-picked raw logits. Order is the natural
// reading order; the bars sort visually by height via their widths, not by
// reordering rows, so the token labels stay put as the slider moves.
const TOKENS = [' cat', ' dog', ' ran', ' the', '.'];
const BASE_LOGITS = [2.4, 1.6, 0.7, -0.3, -1.1];

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const sum = exps.reduce((s, e) => s + e, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 0));
}

export function LogitsToSoftmax() {
  // Sharpness factor: <1 flattens the distribution, >1 sharpens it. We scale
  // the raw logits by this factor before softmax — mathematically the same
  // lever as an inverse temperature (sharpness = 1/T).
  const [sharpness, setSharpness] = React.useState(1);

  const scaledLogits = React.useMemo(() => BASE_LOGITS.map((l) => l * sharpness), [sharpness]);

  // The three pipeline columns: raw logit, its exp, and the final probability.
  // `expDenom` is the shared denominator (Σ exp) every probability divides by.
  const { exps, expDenom, probs } = React.useMemo(() => {
    // Compute exp WITHOUT the max-subtraction here so the displayed numbers
    // match the textbook formula e^{z_i} / Σ e^{z_j}. With these small logits
    // there's no overflow risk; `softmax()` below does the stable version for
    // the bars, and the two agree to display precision.
    const rawExps = scaledLogits.map((l) => Math.exp(l));
    const denom = rawExps.reduce((s, e) => s + e, 0);
    return { exps: rawExps, expDenom: denom, probs: softmax(scaledLogits) };
  }, [scaledLogits]);

  // Bump on every recompute so TopKBars remounts its fills and re-fires the
  // grow-in animation as the slider moves.
  const runKey = React.useMemo(() => Math.round(sharpness * 100), [sharpness]);

  const ids = TOKENS.map((_, i) => i);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Softmax on a 5-token toy vocabulary
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <label htmlFor="logits-softmax-sharpness" className="uppercase tracking-wider">
            Sharpen ↔ flatten
          </label>
          <input
            id="logits-softmax-sharpness"
            type="range"
            min={0.3}
            max={3}
            step={0.05}
            value={sharpness}
            onChange={(e) => setSharpness(Number.parseFloat(e.target.value))}
            className="h-2 w-32 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
          />
          <span className="w-10 font-mono text-foreground/80">×{sharpness.toFixed(2)}</span>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The real model emits <strong>248,320</strong> logits — one per vocabulary token. The transform that turns them
        into probabilities is the same no matter how many there are, so here it is on just five. Drag the slider to
        scale the raw logits: a bigger factor sharpens the distribution onto the top token, a smaller one flattens it.
      </p>

      <MathDisplay latex={String.raw`p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}`} />

      {/* The pipeline table: raw logit z_i → e^{z_i} → p_i. */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-muted-foreground">token</th>
              <th className="px-2 py-1 text-right text-muted-foreground">
                z<sub>i</sub> <span className="normal-case">(logit)</span>
              </th>
              <th className="px-2 py-1 text-right text-muted-foreground">
                exp(z<sub>i</sub>)
              </th>
              <th className="px-2 py-1 text-right text-muted-foreground">÷ Σ</th>
              <th className="px-2 py-1 text-right text-muted-foreground">
                p<sub>i</sub>
              </th>
            </tr>
          </thead>
          <tbody>
            {TOKENS.map((tok, i) => (
              <tr key={tok} className="border-t border-border/60">
                <td className="px-2 py-1 text-left text-foreground/90">{tok === '.' ? '"."' : `"${tok.trim()}"`}</td>
                <td className="px-2 py-1 text-right text-foreground/85">{scaledLogits[i]!.toFixed(2)}</td>
                <td className="px-2 py-1 text-right text-foreground/85">{exps[i]!.toFixed(3)}</td>
                <td className="px-2 py-1 text-right text-muted-foreground">÷ {expDenom.toFixed(3)}</td>
                <td className="px-2 py-1 text-right font-semibold text-foreground">{probs[i]!.toFixed(3)}</td>
              </tr>
            ))}
            <tr className="border-t border-border">
              <td className="px-2 py-1 text-left text-muted-foreground">Σ</td>
              <td className="px-2 py-1 text-right text-muted-foreground">—</td>
              <td className="px-2 py-1 text-right text-muted-foreground">{expDenom.toFixed(3)}</td>
              <td className="px-2 py-1 text-right text-muted-foreground" />
              <td className="px-2 py-1 text-right font-semibold text-foreground">
                {probs.reduce((s, p) => s + p, 0).toFixed(3)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Probability bars — reuse the shared TopKBars primitive. No token is
          "sampled" in this transform-only demo, so nothing is highlighted. */}
      <div className="space-y-1">
        <div className="text-[11px] text-muted-foreground">Probabilities (the last column, as bars)</div>
        <TopKBars ids={ids} probs={probs} texts={TOKENS} sampledTokenId={-1} runKey={runKey} />
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative five-token vocabulary with hand-picked logits — not live output from the model.
      </p>
    </div>
  );
}
