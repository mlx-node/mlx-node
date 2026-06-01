import * as React from 'react';

import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 7 supplement — RMSNorm's only learned parameter is the per-feature
 * gain γ (the gain `g`). After dividing by RMS, γ re-weights every feature:
 * some amplified (γ>1), some suppressed (γ<1). Toggle the gain off and every
 * bar snaps to the γ=1.0 baseline — the init value — showing that without it
 * RMSNorm would scale all 1024 features identically.
 *
 * Data is a DETERMINISTIC synthetic sample of 48 features (a fixed seeded draw,
 * γ ≈ 1 + 0.35·gaussian, clamped to [0.3, 2.2], centred on 1.0). No model, no
 * worker, no runtime randomness — the LCG below runs once at module scope so
 * the render is stable across reloads. The shape (most near 1, a few clearly
 * amplified or suppressed) mirrors real norms; the exact values do not.
 */

const FEATURE_COUNT = 48;
const GAIN_SEED = 1;
const GAIN_SPREAD = 0.35;
const GAIN_MIN = 0.3;
const GAIN_MAX = 2.2;
const BASELINE = 1.0; // γ init value (RMSNorm gain starts at 1.0)
const HIDDEN_DIM = 1024;

const EASE = 'cubic-bezier(0.4, 0, 0.2, 1)';

/** Seeded Gaussian via a small LCG + Box-Muller (ported from 06-rms-norm). */
function seededGaussianVector(dim: number, seed: number): Float32Array {
  const v = new Float32Array(dim);
  let s = seed >>> 0 || 1;
  function nextU(): number {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s & 0x7fffffff) + 1) / 0x80000001;
  }
  for (let i = 0; i < dim; i++) {
    const u1 = Math.max(nextU(), 1e-9);
    const u2 = nextU();
    v[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return v;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

// Build the synthetic γ sample once, at module scope, so it is identical on
// every render and every reload. γ ≈ 1 + 0.35·gaussian, clamped, centred on 1.
const GAINS: readonly number[] = (() => {
  const noise = seededGaussianVector(FEATURE_COUNT, GAIN_SEED);
  const out: number[] = [];
  for (let i = 0; i < FEATURE_COUNT; i++) {
    out.push(clamp(BASELINE + GAIN_SPREAD * noise[i]!, GAIN_MIN, GAIN_MAX));
  }
  return out;
})();

// Diverging-bar geometry: bars grow from a fixed baseline row. The half above
// the baseline holds amplified features (γ>1), the half below holds suppressed
// ones (γ<1). Scale to the largest deviation so the tallest bar fills its half.
const MAX_DEVIATION = GAINS.reduce((m, g) => Math.max(m, Math.abs(g - BASELINE)), 0);
const HALF_PX = 38; // px height of each half of the chart (above / below baseline)

const AMPLIFIED = GAINS.filter((g) => g > BASELINE).length;
const SUPPRESSED = GAINS.filter((g) => g < BASELINE).length;
const GAIN_MIN_OBSERVED = Math.min(...GAINS);
const GAIN_MAX_OBSERVED = Math.max(...GAINS);

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        'rounded px-2.5 py-1 text-xs font-medium transition-colors',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
      ].join(' ')}
    >
      {children}
    </button>
  );
}

/** One feature: a bar that grows up (amplified) or down (suppressed) from the baseline. */
function GainBar({ gain, on }: { gain: number; on: boolean }) {
  const dev = on ? gain - BASELINE : 0;
  const amplified = dev > 0;
  const heightPx = MAX_DEVIATION > 0 ? (Math.abs(dev) / MAX_DEVIATION) * HALF_PX : 0;
  return (
    <div className="flex h-full min-w-0 flex-1 flex-col items-stretch" aria-hidden="true">
      {/* Upper half — amplified bars grow downward to meet the baseline. */}
      <div className="flex flex-1 flex-col justify-end">
        <div
          className="rounded-t-[1px] bg-primary"
          style={{ height: `${amplified ? heightPx : 0}px`, transition: `height 450ms ${EASE}` }}
        />
      </div>
      {/* Lower half — suppressed bars grow downward from the baseline. */}
      <div className="flex flex-1 flex-col justify-start">
        <div
          className="rounded-b-[1px] bg-amber-500/70"
          style={{ height: `${amplified ? 0 : heightPx}px`, transition: `height 450ms ${EASE}` }}
        />
      </div>
    </div>
  );
}

export function LearnedGainInspector() {
  const [on, setOn] = React.useState(true);

  const chartLabel = on
    ? `Per-feature gain for ${FEATURE_COUNT} sampled features: ${AMPLIFIED} amplified above the baseline of 1.0 (shown growing up), ${SUPPRESSED} suppressed below it (shown growing down). Gain ranges from ${GAIN_MIN_OBSERVED.toFixed(2)} to ${GAIN_MAX_OBSERVED.toFixed(2)}.`
    : `Gain turned off: all ${FEATURE_COUNT} features sit flat at the baseline gain of 1.0, scaled identically.`;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Learned gain γ — the only trained parameter
        </div>
        <div
          className="inline-flex rounded-md border border-border p-0.5"
          role="group"
          aria-label="Toggle learned gain"
        >
          <ToggleButton active={on} onClick={() => setOn(true)}>
            Gain on (learned γ)
          </ToggleButton>
          <ToggleButton active={!on} onClick={() => setOn(false)}>
            Gain off (γ = 1)
          </ToggleButton>
        </div>
      </div>

      <MathDisplay latex={String.raw`y_i=\dfrac{x_i}{\mathrm{RMS}(x)}\cdot g_i`} />

      <p className="text-[12px] text-foreground/85">
        RMSNorm&apos;s only learned parameter is this per-feature gain — one scalar per feature (length{' '}
        <span className="font-mono">{HIDDEN_DIM}</span>, initialised to 1.0, no bias). Training learns which features to
        amplify and which to damp.
      </p>

      {/* Diverging bar chart: amplified (γ>1) up, suppressed (γ<1) down, around the γ=1 baseline. */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between text-[11px] text-muted-foreground">
          <span>Per-feature gain (sample of {FEATURE_COUNT})</span>
          <span className="font-mono">{on ? 'learned γ' : 'γ = 1 (init)'}</span>
        </div>
        <div className="relative rounded-sm bg-muted/30 px-1" role="img" aria-label={chartLabel}>
          <div className="flex items-stretch gap-px" style={{ height: `${HALF_PX * 2}px` }}>
            {GAINS.map((gain, i) => (
              <GainBar key={i} gain={gain} on={on} />
            ))}
          </div>
          {/* Baseline at γ = 1.0 — the init value, drawn across the chart midline. */}
          <div className="pointer-events-none absolute inset-x-1 top-1/2 -translate-y-1/2 border-t border-dashed border-foreground/50" />
          <div className="pointer-events-none absolute left-1 top-1/2 -translate-y-1/2 bg-muted/30 pr-1 font-mono text-[9px] text-muted-foreground">
            γ = 1.0
          </div>
        </div>
        <div className="flex justify-between font-mono text-[9px] text-muted-foreground">
          <span className="text-primary">↑ amplified (γ &gt; 1)</span>
          <span className="text-amber-600 dark:text-amber-500">↓ suppressed (γ &lt; 1)</span>
        </div>
      </div>

      {/* Readout — counts and observed range. */}
      <div className="grid grid-cols-2 gap-2 text-[12px] sm:grid-cols-4">
        <div className="rounded-md border border-border bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Amplified</div>
          <div className="font-mono text-foreground/90">
            <span className="text-primary">{AMPLIFIED}</span> features
          </div>
        </div>
        <div className="rounded-md border border-border bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Suppressed</div>
          <div className="font-mono text-foreground/90">
            <span className="text-amber-600 dark:text-amber-500">{SUPPRESSED}</span> features
          </div>
        </div>
        <div className="rounded-md border border-border bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Min γ</div>
          <div className="font-mono text-foreground/90">{GAIN_MIN_OBSERVED.toFixed(2)}</div>
        </div>
        <div className="rounded-md border border-border bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Max γ</div>
          <div className="font-mono text-foreground/90">{GAIN_MAX_OBSERVED.toFixed(2)}</div>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        {on ? (
          <>
            With the learned gain, each feature is re-weighted independently — <strong>{AMPLIFIED} amplified</strong> (γ
            &gt; 1) and <strong>{SUPPRESSED} suppressed</strong> (γ &lt; 1) in this sample.
          </>
        ) : (
          <>
            With the gain off (<span className="font-mono">γ = 1</span> everywhere), RMSNorm scales every feature{' '}
            <strong>identically</strong> — it only standardises magnitude and can no longer favour any feature.
          </>
        )}
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — γ is a synthetic sample of 48 of the 1024 features (a fixed seeded draw centered on 1.0), not
        Qwen3.5-0.8B&apos;s actual learned gain. The shape — most features near 1, a few amplified or suppressed —
        mirrors real norms.
      </p>
    </div>
  );
}
