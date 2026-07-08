import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * DynamicQuantMix — the idea behind imatrix / Unsloth "Dynamic" GGUFs: at a fixed
 * size budget, don't quantize every tensor to the same width. Spend bits where a
 * calibration pass says they matter — keep the sensitive tensors high, push the
 * rest lower — and land at a LOWER error than a uniform quant of the same size.
 *
 * Toggle "uniform" vs "dynamic": the same nine per-layer tensors, two ways to
 * spend the budget. A small always-on gauge shows error-vs-original is lower for
 * dynamic. Self-contained and STATIC — NO worker / model / WASM / network /
 * animation / Date.now / Math.random; only state is the selected mode, so it
 * server-renders deterministically (server == client) with no reduced-motion path.
 *
 * THE ONE HARD FACT (everything else here is a labelled schematic): attention.wv
 * (attn_v) and feed_forward.w2 (ffn_down) are the tensors that EVERY one of
 * llama.cpp's Q*_K_M mixes already bumps to a higher type (k-quants PR #1684).
 * The mixes differ on the rest: Q3_K_M also bumps attention.wo (attn_o), whereas
 * Q4_K_M / Q5_K_M bump only HALF of wv and w2. token_embd and
 * the output/lm_head are conventionally kept high too. "Dynamic 2.0" just makes
 * that choice per-layer and per-model, driven by a >1.5M-token calibration set,
 * and judges the result by KL-divergence rather than perplexity ("Accuracy is Not
 * All You Need," arXiv 2407.09141). The exact 2/3/6-bit numbers below are a
 * schematic budget for the picture, NOT a published per-tensor recipe.
 */

type Mode = 'uniform' | 'dynamic';

type Tensor = {
  id: string;
  label: string; // English glossary — never translated
  sensitive: boolean; // bumped high by the _M mixes / conventionally kept high
};

// One transformer layer's tensors, plus the shared embedding and output.
const TENSORS: Tensor[] = [
  { id: 'embed', label: 'embed', sensitive: true },
  { id: 'q', label: 'q', sensitive: false },
  { id: 'k', label: 'k', sensitive: false },
  { id: 'v', label: 'v', sensitive: true },
  { id: 'o', label: 'o', sensitive: false }, // attn_o is bumped by Q3_K_M but NOT Q4_K_M/Q5_K_M; shown low here to keep the budget readable
  { id: 'gate', label: 'gate', sensitive: false },
  { id: 'up', label: 'up', sensitive: false },
  { id: 'down', label: 'down', sensitive: true },
  { id: 'out', label: 'out', sensitive: true },
];

const UNIFORM_BITS = 3; // every tensor, same width
const DYN_HI = 6; // sensitive tensors kept high
const DYN_LO = 2; // insensitive tensors robbed to pay for it
const MAX_BITS = 8; // height scale headroom (Q8_0-ish)

function dynBits(t: Tensor) {
  return t.sensitive ? DYN_HI : DYN_LO;
}

const EMERALD = '#10b981'; // bits kept HIGH where they matter
const RED = '#ef4444'; // the taller error bar (uniform)

const COPY = {
  en: {
    heading: 'Spend the bits where they matter — dynamic / imatrix quant',
    modeUniform: 'uniform',
    modeDynamic: 'dynamic (imatrix)',
    modeLabel: 'budget',
    pickAria: (m: Mode) => (m === 'uniform' ? 'Uniform: every tensor the same width' : 'Dynamic: bits moved to the sensitive tensors'),
    tallerBits: 'taller = more bits',
    keptHigh: 'kept high (sensitive)',
    errHeading: 'error vs original',
    errUniform: 'uniform',
    errDynamic: 'dynamic',
    relative: 'relative · schematic',
    captionUniform:
      'Every tensor at the same 3-bit width. The bits spent on q / k / gate / up barely change the output — and the value & down-projection tensors are starved.',
    captionDynamic:
      'Same rough size budget, spent differently. The sensitive tensors — attn_v, ffn_down, embed, output — are kept at 6-bit; the rest drop to 2-bit. attn_v & ffn_down are the tensors every Q*_K_M mix already bumps — Dynamic 2.0 just makes it per-layer and calibration-driven.',
    svgAria: (m: string) =>
      `Seven per-layer tensors plus the shared embed and output as bit-width bars, in ${m} mode, beside a gauge showing error-vs-original is higher for uniform and lower for dynamic.`,
  },
  zh: {
    heading: '把 bits 花在该花的地方 —— dynamic / imatrix 量化',
    modeUniform: 'uniform',
    modeDynamic: 'dynamic（imatrix）',
    modeLabel: '预算',
    pickAria: (m: Mode) => (m === 'uniform' ? 'Uniform：每个张量都同样宽度' : 'Dynamic：把 bits 挪到敏感张量上'),
    tallerBits: '越高 = bits 越多',
    keptHigh: '保持高位（敏感）',
    errHeading: '相对原模型的误差',
    errUniform: 'uniform',
    errDynamic: 'dynamic',
    relative: '相对值 · 示意',
    captionUniform:
      '每个张量都用同样的 3-bit。花在 q / k / gate / up 上的 bits 几乎不改变输出 —— 而 value 与 down-projection 张量却被饿着。',
    captionDynamic:
      '大小预算大致不变，只是花法不同。敏感的那几个 —— attn_v、ffn_down、embed、output —— 保持 6-bit；其余降到 2-bit。attn_v 与 ffn_down 正是每一种 Q*_K_M 混合都会抬高的张量 —— Dynamic 2.0 只是把这件事做到每层、且由 calibration 驱动。',
    svgAria: (m: string) =>
      `七个每层张量加上共享的 embed 与 output，画成 bit-width 的柱子，处于 ${m} 模式，旁边一个 gauge 显示 uniform 的误差更高、dynamic 的更低。`,
  },
} as const;

// ── SVG geometry ──
const VB_W = 560;
const VB_H = 168;
const STRIP_X = 8;
const STRIP_W = 410; // tensor strip on the left
const GAUGE_X = 446; // error gauge on the right
const BASE_Y = 118; // baseline the bars stand on
const MAX_BAR_H = 84;
const LABEL_Y = 134;
const COL_W = STRIP_W / TENSORS.length;

function barH(bits: number) {
  return (bits / MAX_BITS) * MAX_BAR_H;
}

export function DynamicQuantMix() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('dynamic');
  const isDyn = mode === 'dynamic';
  const caption = isDyn ? copy.captionDynamic : copy.captionUniform;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header + mode toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.modeLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {(['uniform', 'dynamic'] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
                aria-label={copy.pickAria(m)}
                className={[
                  'px-2 py-0.5 font-medium transition-colors',
                  mode === m ? 'bg-primary/15 text-foreground' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {m === 'uniform' ? copy.modeUniform : copy.modeDynamic}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span>{copy.tallerBits}</span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: EMERALD }} />
          {copy.keptHigh}
        </span>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria(isDyn ? copy.modeDynamic : copy.modeUniform)}>
        {/* baseline */}
        <line x1={STRIP_X} y1={BASE_Y} x2={STRIP_X + STRIP_W} y2={BASE_Y} style={{ stroke: 'var(--border)' }} strokeWidth={1} />

        {TENSORS.map((t, i) => {
          const bits = isDyn ? dynBits(t) : UNIFORM_BITS;
          const h = barH(bits);
          const cx = STRIP_X + i * COL_W + COL_W / 2;
          const bw = COL_W - 7;
          const hi = isDyn && t.sensitive;
          return (
            <g key={t.id}>
              <rect
                x={cx - bw / 2}
                y={BASE_Y - h}
                width={bw}
                height={h}
                rx={2}
                style={{ fill: hi ? EMERALD : 'var(--muted-foreground)', opacity: hi ? 1 : 0.55 }}
              />
              {/* bit-width number atop the bar */}
              <text x={cx} y={BASE_Y - h - 4} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
                {bits}
              </text>
              {/* tensor name */}
              <text
                x={cx}
                y={LABEL_Y}
                textAnchor="middle"
                style={{ fill: hi ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className="font-mono text-[9px]"
              >
                {t.label}
              </text>
            </g>
          );
        })}
        {/* strip caption: bit unit */}
        <text x={STRIP_X} y={VB_H - 8} style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
          {copy.relative}
        </text>

        {/* ── error gauge (right) — always shows both, uniform tall / dynamic short ── */}
        <text x={GAUGE_X} y={14} style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
          {copy.errHeading}
        </text>
        {(() => {
          const gw = 30;
          const gGap = 18;
          const uH = MAX_BAR_H; // uniform: tall
          const dH = MAX_BAR_H * 0.34; // dynamic: short (schematic, relative)
          const uX = GAUGE_X + 4;
          const dX = uX + gw + gGap;
          return (
            <g>
              <rect x={uX} y={BASE_Y - uH} width={gw} height={uH} rx={2} style={{ fill: RED, opacity: isDyn ? 0.35 : 0.9 }} />
              <text x={uX + gw / 2} y={LABEL_Y} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
                {copy.errUniform}
              </text>
              <rect x={dX} y={BASE_Y - dH} width={gw} height={dH} rx={2} style={{ fill: EMERALD, opacity: isDyn ? 0.95 : 0.4 }} />
              <text x={dX + gw / 2} y={LABEL_Y} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
                {copy.errDynamic}
              </text>
            </g>
          );
        })()}
      </svg>

      <p className="text-[12px] text-muted-foreground">{caption}</p>
    </div>
  );
}
