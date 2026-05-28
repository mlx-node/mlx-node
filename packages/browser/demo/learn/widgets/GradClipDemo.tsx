import * as React from 'react';

/**
 * Chapter 13 supplement — visualize gradient clipping by norm.
 *
 * The standard rule: if ||g|| > c, rescale g := g * (c / ||g||); otherwise
 * leave g alone. The plot shows a per-parameter gradient vector (just 8
 * components for legibility) with its L2 norm next to the clip threshold.
 * Slider controls a synthetic "raw gradient scale" so the user can dial up a
 * gradient explosion and watch the clip kick in.
 */

const N_COMPS = 8;

// Hand-picked directional pattern so the raw gradient looks "interesting"
// rather than uniform. Magnitudes are intentionally asymmetric.
const BASE_DIRECTION = [0.8, -0.4, 0.2, -0.7, 0.5, 0.3, -0.6, 0.4];

function l2(v: number[]): number {
  return Math.sqrt(v.reduce((a, x) => a + x * x, 0));
}

export function GradClipDemo() {
  const [scale, setScale] = React.useState(0.6);
  const [clip, setClip] = React.useState(1.0);

  const raw = BASE_DIRECTION.map((d) => d * scale * 10);
  const norm = l2(raw);
  const clipped = norm > clip ? raw.map((g) => g * (clip / norm)) : raw;
  const wasClipped = norm > clip;

  // Bar geometry — symmetric around a zero axis at the center, so positive
  // gradients go right, negative go left.
  const maxGradForScale = 10;
  function barWidthPct(g: number): number {
    return Math.min(50, (Math.abs(g) / maxGradForScale) * 50);
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        Gradient clipping — rescale when the norm exceeds the threshold
      </div>

      <p className="text-[12px] text-foreground/85">
        After backprop computes per-parameter gradients, take their global L2 norm <code>||g||</code>. If it exceeds the
        clip threshold <code>c</code>, rescale every component by <code>c / ||g||</code>. The <em>direction</em> is
        preserved; only the magnitude is capped.
      </p>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">raw gradient (per param)</div>
          <div className="space-y-0.5 rounded-md border border-border/60 bg-muted/20 p-2">
            {raw.map((g, i) => (
              <div key={`r-${i}`} className="flex items-center gap-1">
                <span className="w-5 font-mono text-[9px] text-muted-foreground">g{i}</span>
                <div className="relative flex h-2.5 flex-1 items-stretch overflow-hidden rounded-sm bg-muted/30">
                  <div className="flex flex-1 justify-end">
                    {g < 0 ? (
                      <div
                        className="h-full bg-destructive/60 transition-all"
                        style={{ width: `${barWidthPct(g) * 2}%` }}
                      />
                    ) : null}
                  </div>
                  <div className="w-px bg-foreground/40" />
                  <div className="flex flex-1">
                    {g > 0 ? (
                      <div
                        className="h-full bg-primary/60 transition-all"
                        style={{ width: `${barWidthPct(g) * 2}%` }}
                      />
                    ) : null}
                  </div>
                </div>
                <span className="w-9 text-right font-mono text-[9px] text-foreground/75">{g.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="text-right text-[10px] text-muted-foreground">
            ||g|| ={' '}
            <span
              className={['font-mono', wasClipped ? 'text-amber-600 dark:text-amber-400' : 'text-foreground/80'].join(
                ' ',
              )}
            >
              {norm.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">after clip-norm</div>
          <div
            className={[
              'space-y-0.5 rounded-md border p-2 transition-colors',
              wasClipped ? 'border-amber-500/40 bg-amber-500/5' : 'border-border/60 bg-muted/20',
            ].join(' ')}
          >
            {clipped.map((g, i) => (
              <div key={`c-${i}`} className="flex items-center gap-1">
                <span className="w-5 font-mono text-[9px] text-muted-foreground">g{i}</span>
                <div className="relative flex h-2.5 flex-1 items-stretch overflow-hidden rounded-sm bg-muted/30">
                  <div className="flex flex-1 justify-end">
                    {g < 0 ? (
                      <div
                        className="h-full bg-destructive/60 transition-all"
                        style={{ width: `${barWidthPct(g) * 2}%` }}
                      />
                    ) : null}
                  </div>
                  <div className="w-px bg-foreground/40" />
                  <div className="flex flex-1">
                    {g > 0 ? (
                      <div
                        className="h-full bg-primary/60 transition-all"
                        style={{ width: `${barWidthPct(g) * 2}%` }}
                      />
                    ) : null}
                  </div>
                </div>
                <span className="w-9 text-right font-mono text-[9px] text-foreground/75">{g.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="text-right text-[10px] text-muted-foreground">
            ||g_clipped|| = <span className="font-mono text-foreground/80">{l2(clipped).toFixed(2)}</span>
            {wasClipped ? <span className="ml-1 text-amber-600 dark:text-amber-400">clipped</span> : null}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>raw gradient scale</span>
            <span>×{scale.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.05}
            max={2.5}
            step={0.05}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
            className="w-full"
          />
        </label>
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>clip threshold c</span>
            <span>{clip.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.2}
            max={5.0}
            step={0.1}
            value={clip}
            onChange={(e) => setClip(Number(e.target.value))}
            className="w-full"
          />
        </label>
      </div>

      <p className="text-[11px] text-muted-foreground">
        Slide the gradient scale up — past <code>~0.5</code> you'll see <code>||g||</code> exceed the threshold and the
        clipped panel turn amber. Without this guardrail, a single bad batch (mid-sequence loss spike) can drive a
        24-layer stack's weights into a regime training can't recover from. <code>c = 1.0</code> is the LLM-pretraining
        default.
      </p>
    </div>
  );
}
