import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 13 supplement — visualize gradient clipping by norm.
 *
 * The standard rule: if ||g|| > c, rescale g := g * (c / ||g||); otherwise
 * leave g alone. The plot shows a per-parameter gradient vector (just 8
 * components for legibility) with its L2 norm next to the clip threshold.
 * Slider controls a synthetic "raw gradient scale" so the user can dial up a
 * gradient explosion and watch the clip kick in.
 */

// Hand-picked directional pattern so the raw gradient looks "interesting"
// rather than uniform. Magnitudes are intentionally asymmetric.
const BASE_DIRECTION = [0.8, -0.4, 0.2, -0.7, 0.5, 0.3, -0.6, 0.4];

function l2(v: number[]): number {
  return Math.sqrt(v.reduce((a, x) => a + x * x, 0));
}

// Per-locale copy — every user-visible English string moved here verbatim.
// Math (g = [3, 4], ||g|| = 5, ×0.2 …) and identifiers stay as-is in both locales.
const COPY = {
  en: {
    title: 'Gradient clipping — rescale when the norm exceeds the threshold',
    intro: (
      <>
        After backprop computes per-parameter gradients, take their global L2 norm <code>||g||</code> — square every
        component, sum, square-root. If it exceeds the clip threshold <code>c</code>, rescale every component by{' '}
        <code>c / ||g||</code>. The <em>direction</em> is preserved; only the magnitude is capped. Worked tiny:{' '}
        <code>g = [3, 4]</code> → <code>||g|| = √(9 + 16) = 5</code>; clip to <code>c = 1.0</code> → multiply by{' '}
        <code>1/5 = 0.2</code> → <code>[0.6, 0.8]</code> — same direction, one-fifth the length.
      </>
    ),
    rawTitle: 'raw gradient (per param)',
    clippedTitle: 'after clip-norm',
    clippedBadge: 'clipped',
    scaleLabel: 'raw gradient scale',
    clipLabel: 'clip threshold c',
    footer: (
      <>
        Slide the gradient scale up — past <code>~0.07</code> you'll see <code>||g||</code> exceed the threshold and the
        clipped panel turn amber. Without this guardrail, a single bad batch (mid-sequence loss spike) can drive a
        24-layer stack's weights into a regime training can't recover from. <code>c = 1.0</code> is the LLM-pretraining
        default.
      </>
    ),
  },
  zh: {
    title: '梯度裁剪——范数超过阈值时整体缩放',
    intro: (
      <>
        反向传播算出逐参数梯度后，取它们的全局 L2 范数 <code>||g||</code>
        ——每个分量平方、求和、再开方。若超过裁剪阈值 <code>c</code>，就把每个分量按 <code>c / ||g||</code> 缩放。
        <em>方向</em>保持不变；只有幅度被封顶。一个小算例：<code>g = [3, 4]</code> →{' '}
        <code>||g|| = √(9 + 16) = 5</code>；裁剪到 <code>c = 1.0</code> → 乘以 <code>1/5 = 0.2</code> →{' '}
        <code>[0.6, 0.8]</code>——方向相同，长度只剩五分之一。
      </>
    ),
    rawTitle: '原始梯度（逐参数）',
    clippedTitle: '按范数裁剪之后',
    clippedBadge: '已裁剪',
    scaleLabel: '原始梯度缩放',
    clipLabel: '裁剪阈值 c',
    footer: (
      <>
        把梯度缩放往上拉——超过 <code>~0.07</code> 后你会看到 <code>||g||</code>{' '}
        超出阈值、裁剪面板变成琥珀色。没有这道护栏，单个坏 batch（序列中段的损失尖峰）就能把 24
        层堆叠的权重推进训练无法恢复的区域。<code>c = 1.0</code> 是 LLM 预训练的默认值。
      </>
    ),
  },
} as const;

export function GradClipDemo() {
  const copy = COPY[useLocale()];
  const [scale, setScale] = React.useState(0.05);
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
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>

      <p className="text-[12px] text-foreground/85">{copy.intro}</p>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.rawTitle}</div>
          <div className="space-y-0.5 rounded-md border border-border/60 bg-muted/20 p-2">
            {raw.map((g, i) => (
              <div key={`r-${i}`} className="flex items-center gap-1">
                <span className="w-5 font-mono text-[11px] text-muted-foreground">g{i}</span>
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
                <span className="w-10 text-right font-mono text-[11px] text-foreground/75">{g.toFixed(2)}</span>
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
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.clippedTitle}</div>
          <div
            className={[
              'space-y-0.5 rounded-md border p-2 transition-colors',
              wasClipped ? 'border-amber-500/40 bg-amber-500/5' : 'border-border/60 bg-muted/20',
            ].join(' ')}
          >
            {clipped.map((g, i) => (
              <div key={`c-${i}`} className="flex items-center gap-1">
                <span className="w-5 font-mono text-[11px] text-muted-foreground">g{i}</span>
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
                <span className="w-10 text-right font-mono text-[11px] text-foreground/75">{g.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="text-right text-[10px] text-muted-foreground">
            ||g_clipped|| = <span className="font-mono text-foreground/80">{l2(clipped).toFixed(2)}</span>
            {wasClipped ? <span className="ml-1 text-amber-600 dark:text-amber-400">{copy.clippedBadge}</span> : null}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>{copy.scaleLabel}</span>
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
            <span>{copy.clipLabel}</span>
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

      <p className="text-[11px] text-muted-foreground">{copy.footer}</p>
    </div>
  );
}
