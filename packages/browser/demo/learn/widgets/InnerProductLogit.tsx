import * as React from 'react';

import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 10 (LM head) supplement — the *geometry* of one logit.
 *
 * `LmHeadWalkthrough` walks the matmul SHAPE; `WeightTyingVisual` shows the
 * embedding/unembedding TIE. This widget fills the remaining gap: WHY a logit
 * is what it is. A token's logit is the inner (dot) product of the final hidden
 * state `h` with that token's row `w` in the output matrix — and that dot
 * product equals |h|·|w|·cos θ, so the logit measures how *aligned* the token's
 * direction is with `h`.
 *
 * To isolate angle from magnitude we use a TOY 8-dim space (real h is 1024-dim;
 * the caption says so) and three candidate vectors of the SAME norm C = 1.5,
 * differing only in their angle to a fixed hidden vector H. All three geometric
 * cases are constructed FROM H so they come out exact (verified in code, not
 * hardcoded):
 *   - Aligned (0°):    w = (H/|H|)·C     → every product h_d·w_d ≥ 0; logit = |H|·C = +2.230; cos = +1
 *   - Orthogonal (90°): w = (U/|U|)·C     where U rotates each (even,odd) pair 90°;
 *                       H·U = 0 exactly, |U| = |H| → logit = 0.000; cos = 0
 *   - Opposed (180°):  w = -(H/|H|)·C    → every product ≤ 0; logit = -2.230; cos = -1
 *
 * The per-dimension chart renders each signed product h_d·w_d as a DIVERGING
 * bar around a center baseline; the bars visibly sum to the logit, and the
 * angle/cosine readout connects that sum to |h|·|w|·cos θ. No color-only
 * encoding: every row also shows its numeric product.
 *
 * Self-contained: NO model / worker / WASM. The vectors are scripted; all
 * products, the logit, and cos θ are computed live below.
 */

// Fixed toy hidden state. |H| = sqrt(2.21) ≈ 1.4866.
const H = [0.9, -0.4, 0.7, 0.2, -0.6, 0.5, 0.1, -0.3];
// Shared norm for all three candidate vectors, so only the ANGLE differs.
const C = 1.5;

// Diverging-bar colors: positive contributions warm, negative cool. (Each row
// also prints its signed value, so the chart never relies on color alone.)
const POS_BG = 'oklch(0.7 0.13 150 / 0.55)'; // green-ish: pushes the logit up
const NEG_BG = 'oklch(0.66 0.16 25 / 0.55)'; //   red-ish: pulls the logit down

type Angle = 'aligned' | 'orthogonal' | 'opposed';

const norm = (v: number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

const NORM_H = norm(H);
// Unit hidden direction, scaled to length C → the "aligned" candidate.
const W_ALIGNED = H.map((x) => (x / NORM_H) * C);
// Rotate each adjacent (even, odd) pair by 90°: (a, b) → (-b, a). This is
// orthogonal to H (H·U = 0 exactly) and preserves length (|U| = |H|).
const U = [-H[1]!, H[0]!, -H[3]!, H[2]!, -H[5]!, H[4]!, -H[7]!, H[6]!];
const W_ORTHO = U.map((x) => (x / norm(U)) * C);
// Antiparallel hidden direction, length C → the "opposed" candidate.
const W_OPPOSED = H.map((x) => -(x / NORM_H) * C);

const CANDIDATES: Record<Angle, { label: string; w: number[] }> = {
  aligned: { label: 'Aligned (0°)', w: W_ALIGNED },
  orthogonal: { label: 'Orthogonal (90°)', w: W_ORTHO },
  opposed: { label: 'Opposed (180°)', w: W_OPPOSED },
};

const ORDER: Angle[] = ['aligned', 'orthogonal', 'opposed'];

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

export function InnerProductLogit() {
  const [angle, setAngle] = React.useState<Angle>('aligned');
  const w = CANDIDATES[angle].w;

  // Per-dimension signed products h_d·w_d, the logit (their sum), and the angle.
  const { products, logit, cos, theta, maxAbs } = React.useMemo(() => {
    const prods = H.map((h, i) => h * w[i]!);
    const sum = prods.reduce((s, x) => s + x, 0);
    const c = sum / (NORM_H * norm(w));
    const cClamped = Math.max(-1, Math.min(1, c));
    const deg = (Math.acos(cClamped) * 180) / Math.PI;
    const maxMagnitude = Math.max(...prods.map((p) => Math.abs(p)), 1e-6);
    return { products: prods, logit: sum, cos: c, theta: deg, maxAbs: maxMagnitude };
  }, [w]);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">A logit is an inner product</div>
        <div className="inline-flex rounded-md border border-border p-0.5">
          {ORDER.map((key) => (
            <ToggleButton key={key} active={angle === key} onClick={() => setAngle(key)}>
              {CANDIDATES[key].label}
            </ToggleButton>
          ))}
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Every vocabulary token owns a row <span className="font-mono">w_t</span> in the output matrix (the tied
        embedding); its logit is the dot product of that row with the final hidden state{' '}
        <span className="font-mono">h</span>, and the softmax (shown elsewhere in the course) turns the whole vector of
        logits into probabilities. All three candidates here have the <strong>same length</strong>{' '}
        <span className="font-mono">|w| = {C.toFixed(1)}</span>, so the only thing that moves the logit is the{' '}
        <strong>angle</strong> to <span className="font-mono">h</span>.
      </p>

      <MathDisplay
        latex={String.raw`\text{logit}_t = h \cdot w_t = \sum_d h_d\, w_{t,d} = \lVert h\rVert\,\lVert w_t\rVert\cos\theta`}
      />

      {/* Per-dimension diverging bars: positive products extend right, negative
          left, around a center baseline. The 1fr/1fr split puts the baseline at
          the visual center; each row prints its signed value so the chart is
          legible without color. The bars sum to the logit shown below. */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Per-dimension contribution h_d · w_d</span>
          <span className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: POS_BG }} aria-hidden />
              positive
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: NEG_BG }} aria-hidden />
              negative
            </span>
          </span>
        </div>
        <div
          role="img"
          aria-label={`Per-dimension contributions for the ${CANDIDATES[angle].label} token: ${products
            .map((p, i) => `dimension ${i} contributes ${p.toFixed(3)}`)
            .join('; ')}. They sum to the logit ${logit.toFixed(3)}.`}
          className="space-y-1 rounded-md border border-border/60 bg-muted/20 p-2"
        >
          {products.map((p, i) => {
            const pct = (Math.abs(p) / maxAbs) * 100;
            const positive = p >= 0;
            return (
              <div
                key={i}
                className="grid grid-cols-[2.5rem_minmax(0,1fr)_3.75rem] items-center gap-2 text-[11px]"
                title={`d=${i} · h=${H[i]!.toFixed(2)} · w=${w[i]!.toFixed(3)} · product=${p.toFixed(3)}`}
              >
                <span className="font-mono text-muted-foreground">d{i}</span>
                {/* Two equal halves meeting at a center baseline; the active
                    half fills outward from the middle. */}
                <div className="relative flex h-4 w-full items-stretch overflow-hidden rounded bg-muted">
                  <div className="flex h-full w-1/2 justify-end">
                    {!positive ? (
                      <div
                        className="h-full origin-right rounded-l"
                        style={{ width: `${pct}%`, backgroundColor: NEG_BG }}
                      />
                    ) : null}
                  </div>
                  <div className="absolute inset-y-0 left-1/2 w-px bg-border" aria-hidden />
                  <div className="flex h-full w-1/2 justify-start">
                    {positive ? (
                      <div
                        className="h-full origin-left rounded-r"
                        style={{ width: `${pct}%`, backgroundColor: POS_BG }}
                      />
                    ) : null}
                  </div>
                </div>
                <span className="text-right font-mono text-foreground/80">
                  {p >= 0 ? '+' : ''}
                  {p.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Sum readout: the bars above add up to exactly this. */}
      <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-[12px] text-foreground/85">
        <span className="font-mono">Σ_d h_d·w_d = logit = </span>
        <span className="font-mono font-semibold text-primary">
          {logit >= 0 ? '+' : ''}
          {logit.toFixed(3)}
        </span>
        <span className="text-muted-foreground"> — the eight bars above sum to this single score.</span>
      </div>

      {/* Angle / cosine readout: connects the sum to |h|·|w|·cos θ. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-foreground/85">
        <span className="font-mono">
          cos θ = (h·w)/(|h||w|) ={' '}
          <span className="font-semibold text-foreground">
            {cos >= 0 ? '+' : ''}
            {cos.toFixed(3)}
          </span>
        </span>
        <span className="font-mono">
          θ ≈ <span className="font-semibold text-foreground">{theta.toFixed(0)}°</span>
        </span>
        <span className="text-muted-foreground">
          (|h| ≈ {NORM_H.toFixed(3)}, |w| = {C.toFixed(1)})
        </span>
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative 8-dimensional toy vectors (the real hidden state is 1024-dim) — not live output from the model.
      </p>
    </div>
  );
}
