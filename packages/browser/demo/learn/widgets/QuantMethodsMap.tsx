import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * QuantMethodsMap — the quantization-method landscape on two axes for the
 * quantization sub-chapter. It answers two questions at a glance:
 *   x = WHAT do you quantize?  weight-only (left) → weights + activations / W8A8 (right)
 *   y = HOW MUCH retraining?   PTQ, calibrate-only (bottom) → QAT, retrain through
 *       fake-quant (top)
 * Click a labelled dot → a detail panel with the one-line idea, what it
 * quantizes, the bit-width, and the arXiv id. The map makes the big lesson
 * legible: almost all open-weights / local-inference quant lives in the
 * bottom-left (weight-only PTQ); W8A8 sits bottom-right; QAT is nearly empty.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation, NO timers — pure React + Tailwind/SVG, so it server-renders for
 * crawlers (createRoot replaces it on boot) and is identical server vs client.
 * The only state is which dot is selected, so it needs no reduced-motion path.
 *
 * Every fact is source-verified — none invented:
 *   - RTN: round-to-nearest grid level, no error compensation (the naive
 *     weight-only baseline), used at ~8/4-bit.
 *   - GPTQ (arXiv 2210.17323): layer-wise, approximate second-order (Hessian)
 *     info compensates rounding error column-by-column; weight-only, 3–4 bit.
 *   - AWQ (arXiv 2306.00978): the NAMING TRAP — "Activation-aware" but
 *     WEIGHT-ONLY; activation magnitude only PICKS ~1% salient weight channels
 *     to protect via per-channel scaling; 4-bit.
 *   - SmoothQuant (arXiv 2211.10438): migrates activation-outlier difficulty
 *     into the weights via a per-channel smoothing factor so both go 8-bit;
 *     W8A8.
 *   - LLM.int8() (arXiv 2208.07339, bitsandbytes): keeps >99.9% of values in
 *     int8, isolates rare outlier feature dimensions into an fp16 matmul; W8A8.
 *   - GGUF k-quants (llama.cpp): a FILE FORMAT, not an algorithm; mixes
 *     per-block scales at different bit-widths — a "Q4_K" is ~4.5 bits/weight,
 *     NOT 4; effective ~2.6–6.6 bpw; weight-only.
 *   - QLoRA / NF4 (arXiv 2305.14314): freeze a 4-bit NF4 base, train small LoRA
 *     adapters — a fine-tuning method (hybrid), weight-only base, 4-bit.
 *
 * HONESTY: this map is a qualitative landscape, not a benchmark. Axis positions
 * are categorical (which side of weight-only / W8A8, how much retraining), not a
 * measured score; the only precise numbers are the bit-widths and arXiv ids,
 * which are exact.
 */

const EMERALD = '#10b981'; // weight-only — the easy memory win

type MethodId = 'rtn' | 'gptq' | 'awq' | 'smoothquant' | 'llmint8' | 'gguf' | 'qlora';

type Quant = 'weight' | 'w8a8';

type Method = {
  id: MethodId;
  /** display label (kept English in both locales — these are glossary terms) */
  label: string;
  /** what it quantizes — drives the dot tint */
  quant: Quant;
  /** x in [0,1]: 0 = weight-only, 1 = weights + activations (categorical, not measured) */
  x: number;
  /** y in [0,1]: 0 = PTQ (calibrate only), 1 = QAT (retrain). Hybrid (NF4) sits mid */
  y: number;
  /** label offset relative to the dot, to stagger and avoid TEXTOVERLAP */
  lx: number;
  ly: number;
  /** text anchor for the label */
  anchor: 'start' | 'middle' | 'end';
};

// Categorical coordinates. x/y are landscape positions, NOT scores. The dots
// are nudged apart and labels staggered so no two <text> elements overlap.
const METHODS: Method[] = [
  // weight-only PTQ cluster (bottom-left)
  { id: 'rtn', label: 'RTN', quant: 'weight', x: 0.06, y: 0.12, lx: 0, ly: -12, anchor: 'middle' },
  { id: 'gptq', label: 'GPTQ', quant: 'weight', x: 0.24, y: 0.06, lx: 12, ly: 4, anchor: 'start' },
  { id: 'awq', label: 'AWQ', quant: 'weight', x: 0.36, y: 0.22, lx: 0, ly: -12, anchor: 'middle' },
  { id: 'gguf', label: 'GGUF', quant: 'weight', x: 0.14, y: 0.3, lx: -12, ly: 4, anchor: 'end' },
  // hybrid: NF4 base is weight-only but the method trains adapters → nudged up the y-axis
  { id: 'qlora', label: 'QLoRA / NF4', quant: 'weight', x: 0.2, y: 0.56, lx: 12, ly: 4, anchor: 'start' },
  // W8A8 PTQ (bottom-right)
  { id: 'smoothquant', label: 'SmoothQuant', quant: 'w8a8', x: 0.82, y: 0.1, lx: -12, ly: 4, anchor: 'end' },
  { id: 'llmint8', label: 'LLM.int8()', quant: 'w8a8', x: 0.9, y: 0.26, lx: 0, ly: -12, anchor: 'middle' },
];

const DEFAULT_ID: MethodId = 'awq'; // open on the naming-trap method

const COPY = {
  en: {
    heading: 'The quantization-method map · what & how-much-retrain',
    xLeft: 'weight-only',
    xRight: 'weights + activations (W8A8)',
    xAxis: 'WHAT you quantize →',
    yBottom: 'PTQ · calibrate only',
    yTop: 'QAT · retrain',
    yAxis: 'HOW MUCH retraining ↑',
    legendWeight: 'weight-only',
    legendW8a8: 'W8A8 (weights + activations)',
    pickAria: (m: string) => `Show details for ${m}`,
    panelQuant: 'quantizes',
    panelBits: 'bits',
    panelRef: 'paper',
    quantWeight: 'weight-only',
    quantWeightBase: 'weight-only base',
    quantW8a8: 'W8A8',
    axisNote: 'positions are categorical (which side / how much retrain), not a measured score',
    detail: {
      rtn: {
        idea: 'Round each weight to the nearest grid level — no error compensation. The naive baseline everything else improves on.',
        quant: 'weight-only',
        bits: '~8 / 4-bit',
        ref: '(baseline)',
      },
      gptq: {
        idea: 'Layer-wise: uses approximate second-order (Hessian) info to compensate the rounding error column-by-column.',
        quant: 'weight-only',
        bits: '3–4 bit',
        ref: 'arXiv 2210.17323',
      },
      awq: {
        idea: 'Naming trap: "Activation-aware" but WEIGHT-ONLY. Activation magnitude only PICKS the ~1% salient weight channels to protect via per-channel scaling.',
        quant: 'weight-only',
        bits: '4-bit',
        ref: 'arXiv 2306.00978',
      },
      smoothquant: {
        idea: 'Migrates the activation-outlier difficulty into the weights via a per-channel smoothing factor, so weights AND activations both go 8-bit.',
        quant: 'W8A8',
        bits: '8-bit',
        ref: 'arXiv 2211.10438',
      },
      llmint8: {
        idea: 'Keeps >99.9% of values in int8 and isolates the rare outlier feature dimensions into a separate fp16 matmul (bitsandbytes).',
        quant: 'W8A8',
        bits: '8-bit',
        ref: 'arXiv 2208.07339',
      },
      gguf: {
        idea: 'A llama.cpp FILE FORMAT, not an algorithm: mixes per-block scales at different bit-widths. A "Q4_K" is ~4.5 bits/weight, not 4.',
        quant: 'weight-only',
        bits: '~2.6–6.6 effective bpw',
        ref: '(llama.cpp)',
      },
      qlora: {
        idea: 'Freeze a 4-bit NF4 base, train small LoRA adapters on top — a fine-tuning method, so it edges toward retraining (hybrid).',
        quant: 'weight-only base',
        bits: '4-bit',
        ref: 'arXiv 2305.14314',
      },
    } satisfies Record<MethodId, { idea: string; quant: string; bits: string; ref: string }>,
    caption: (
      <>
        Most local-inference quant (<strong>GPTQ</strong> / <strong>AWQ</strong> / <strong>GGUF</strong> /{' '}
        <strong>NF4</strong>) is <em>weight-only PTQ</em>: it shrinks memory and speeds the memory-bound{' '}
        <span className="font-mono">decode</span> without touching activations.{' '}
        <strong>W8A8</strong> (<strong>SmoothQuant</strong>, <strong>LLM.int8()</strong>) also quantizes
        activations for a faster integer matmul — but must tame the outliers. QAT (retrain through fake-quant) is
        rare for open weights because it needs a full training loop. Two traps to remember:{' '}
        <strong>AWQ is weight-only despite the "Activation-aware" name</strong>, and a GGUF{' '}
        <span className="font-mono">Q4_K</span> ≈ <strong>4.5 bpw</strong>, not 4 (other Q4* variants differ).
      </>
    ),
    mapAria:
      'A two-axis map of quantization methods. The horizontal axis runs from weight-only on the left to weights-plus-activations W8A8 on the right; the vertical axis runs from PTQ calibrate-only at the bottom to QAT retraining at the top. Most methods cluster in the bottom-left weight-only PTQ region: RTN, GPTQ, AWQ, GGUF, with QLoRA-NF4 nudged up as a hybrid. SmoothQuant and LLM.int8() sit in the bottom-right W8A8 region. Select a dot for its details.',
  },
  zh: {
    heading: 'quantization 方法地图 · 量化什么 & 需要多少再训练',
    xLeft: 'weight-only',
    xRight: 'weights + activations（W8A8）',
    xAxis: '量化什么 →',
    yBottom: 'PTQ · 仅 calibration',
    yTop: 'QAT · 再训练',
    yAxis: '需要多少再训练 ↑',
    legendWeight: 'weight-only',
    legendW8a8: 'W8A8（weights + activations）',
    pickAria: (m: string) => `查看 ${m} 的详情`,
    panelQuant: '量化对象',
    panelBits: 'bits',
    panelRef: '论文',
    quantWeight: 'weight-only',
    quantWeightBase: 'weight-only 基座',
    quantW8a8: 'W8A8',
    axisNote: '位置是分类的（在哪一侧 / 需要多少再训练），不是测量出的分数',
    detail: {
      rtn: {
        idea: '把每个 weight 四舍五入到最近的格点——没有任何误差补偿。这是其他方法都在改进的朴素基线。',
        quant: 'weight-only',
        bits: '~8 / 4-bit',
        ref: '（基线）',
      },
      gptq: {
        idea: '逐层进行：用近似的二阶（Hessian）信息逐列补偿四舍五入带来的误差。',
        quant: 'weight-only',
        bits: '3–4 bit',
        ref: 'arXiv 2210.17323',
      },
      awq: {
        idea: '命名陷阱：名字叫「Activation-aware」，但其实是 WEIGHT-ONLY。activation 幅度只是用来挑出约 1% 的显著 weight 通道，再用 per-channel scaling 去保护它们。',
        quant: 'weight-only',
        bits: '4-bit',
        ref: 'arXiv 2306.00978',
      },
      smoothquant: {
        idea: '用一个 per-channel 的平滑因子，把 activation outlier 的难度迁移到 weights 里，于是 weights 和 activations 都能走 8-bit。',
        quant: 'W8A8',
        bits: '8-bit',
        ref: 'arXiv 2211.10438',
      },
      llmint8: {
        idea: '把 >99.9% 的值保留在 int8，并把罕见的 outlier 特征维度单独拎出来做一个 fp16 matmul（bitsandbytes）。',
        quant: 'W8A8',
        bits: '8-bit',
        ref: 'arXiv 2208.07339',
      },
      gguf: {
        idea: '这是 llama.cpp 的一种文件格式，不是算法：在 per-block 上混用不同 bit-width 的 scale。一个「Q4_K」≈ 4.5 bits/weight，不是 4。',
        quant: 'weight-only',
        bits: '~2.6–6.6 有效 bpw',
        ref: '（llama.cpp）',
      },
      qlora: {
        idea: '冻结一个 4-bit 的 NF4 基座，在其上训练小的 LoRA adapter——这是一种 fine-tuning 方法，所以它偏向再训练一侧（混合）。',
        quant: 'weight-only 基座',
        bits: '4-bit',
        ref: 'arXiv 2305.14314',
      },
    } satisfies Record<MethodId, { idea: string; quant: string; bits: string; ref: string }>,
    caption: (
      <>
        大多数本地推理的 quant（<strong>GPTQ</strong> / <strong>AWQ</strong> / <strong>GGUF</strong> /{' '}
        <strong>NF4</strong>）都是 <em>weight-only PTQ</em>：它缩小显存、加速 memory-bound 的{' '}
        <span className="font-mono">decode</span>，却不碰 activations。<strong>W8A8</strong>（
        <strong>SmoothQuant</strong>、<strong>LLM.int8()</strong>）会连 activations 一起量化，换来更快的整数
        matmul——但必须驯服 outlier。QAT（让梯度穿过 fake-quant 再训练）对开放权重很少见，因为它需要完整的训练循环。两个要
        记住的陷阱：<strong>AWQ 名字里有 activation，但其实是 weight-only</strong>；以及 GGUF 的{' '}
        <span className="font-mono">Q4_K</span> ≈ <strong>4.5 bpw</strong>，不是 4（其它 Q4* 变体各不相同）。
      </>
    ),
    mapAria:
      'quantization 方法的二维地图。横轴从左侧的 weight-only 到右侧的 weights + activations（W8A8）；纵轴从底部的 PTQ（仅 calibration）到顶部的 QAT（再训练）。大多数方法聚在左下角的 weight-only PTQ 区：RTN、GPTQ、AWQ、GGUF，QLoRA / NF4 作为混合方法略微上移。SmoothQuant 和 LLM.int8() 落在右下角的 W8A8 区。选择一个点查看详情。',
  },
} as const;

// ── SVG geometry. Sized for the wide EN axis labels (EN > zh). The plot is a
// square-ish region with margins; axis captions sit OUTSIDE the plot so they
// never collide with dots/labels. ────────────────────────────────────────────
const VB_W = 560;
const VB_H = 340;
const PLOT_L = 70; // room for the y-axis caption + left dot labels
const PLOT_R = 24;
const PLOT_T = 30; // room for the QAT-top caption
const PLOT_B = 56; // room for the x-axis caption + bottom dot labels
const PLOT_W = VB_W - PLOT_L - PLOT_R; // 466
const PLOT_H = VB_H - PLOT_T - PLOT_B; // 254

// Map a method's categorical (x,y) in [0,1] to plot pixels (y inverted: 1=top).
function px(x: number) {
  return PLOT_L + x * PLOT_W;
}
function py(y: number) {
  return PLOT_T + (1 - y) * PLOT_H;
}

export function QuantMethodsMap() {
  const copy = COPY[useLocale()];
  const [selected, setSelected] = React.useState<MethodId>(DEFAULT_ID);
  // Below 40rem the map downscales and the SVG dots become too small to tap, so
  // a touch-friendly <button> chip row replaces them. Expose exactly ONE
  // interactive selector set per mode — chips when narrow, SVG dots otherwise —
  // so a method is never focusable/clickable twice.
  const [narrow, setNarrow] = React.useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(max-width: 639px)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(max-width: 639px)');
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    setNarrow(mql.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  const sel = METHODS.find((m) => m.id === selected)!;
  const det = copy.detail[selected];

  // mid-x divider between weight-only and W8A8 (categorical hint, not a measure).
  const dividerX = px(0.6);

  const quantLabel = (q: Quant, base: boolean) =>
    q === 'w8a8' ? copy.quantW8a8 : base ? copy.quantWeightBase : copy.quantWeight;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + legend */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: EMERALD }} />
            {copy.legendWeight}
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'var(--primary)' }} />
            {copy.legendW8a8}
          </span>
        </div>
      </div>

      {/* The 2-D map */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.mapAria}>
        {/* plot frame */}
        <rect
          x={PLOT_L}
          y={PLOT_T}
          width={PLOT_W}
          height={PLOT_H}
          rx={6}
          fill="var(--card)"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        {/* weight-only | W8A8 categorical divider */}
        <line
          x1={dividerX}
          y1={PLOT_T}
          x2={dividerX}
          y2={PLOT_T + PLOT_H}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
          strokeDasharray="4 4"
        />

        {/* ── axis captions (OUTSIDE the plot, never over dots) ── */}
        {/* y-axis: QAT top, PTQ bottom, rotated axis title on the far left */}
        <text
          x={PLOT_L}
          y={PLOT_T - 12}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.yTop}
        </text>
        <text
          x={PLOT_L}
          y={PLOT_T + PLOT_H + 18}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.yBottom}
        </text>
        <text
          x={16}
          y={PLOT_T + PLOT_H / 2}
          textAnchor="middle"
          transform={`rotate(-90 16 ${PLOT_T + PLOT_H / 2})`}
          style={{ fill: 'var(--foreground)' }}
          className="text-[11px] font-semibold"
        >
          {copy.yAxis}
        </text>

        {/* x-axis: weight-only left, W8A8 right, axis title bottom-right */}
        <text
          x={PLOT_L}
          y={PLOT_T + PLOT_H + 38}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.xLeft}
        </text>
        <text
          x={PLOT_L + PLOT_W}
          y={PLOT_T + PLOT_H + 38}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.xRight}
        </text>

        {/* ── method dots + staggered labels ── */}
        {METHODS.map((m) => {
          const cx = px(m.x);
          const cy = py(m.y);
          const isSel = m.id === selected;
          const color = m.quant === 'weight' ? EMERALD : 'var(--primary)';
          return (
            <g key={m.id}>
              {/* invisible larger hit target keeps small dots clickable */}
              <circle
                cx={cx}
                cy={cy}
                r={14}
                fill="transparent"
                // When narrow, the chip row below is the interactive selector;
                // deactivate the dots so a method isn't focusable/clickable twice.
                role={narrow ? undefined : 'button'}
                tabIndex={narrow ? -1 : 0}
                aria-hidden={narrow ? true : undefined}
                aria-label={narrow ? undefined : copy.pickAria(m.label)}
                aria-pressed={narrow ? undefined : isSel}
                onClick={narrow ? undefined : () => setSelected(m.id)}
                onKeyDown={
                  narrow
                    ? undefined
                    : (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setSelected(m.id);
                        }
                      }
                }
                className="cursor-pointer focus-visible:outline-none"
                style={{ outline: 'none', pointerEvents: narrow ? 'none' : undefined }}
              />
              {/* selection ring */}
              {isSel ? (
                <circle cx={cx} cy={cy} r={9} fill="none" style={{ stroke: color }} strokeWidth={2} />
              ) : null}
              <circle
                cx={cx}
                cy={cy}
                r={isSel ? 5.5 : 5}
                style={{ fill: color, opacity: isSel ? 1 : 0.7, pointerEvents: 'none' }}
              />
              <text
                x={cx + m.lx}
                y={cy + m.ly + 4}
                textAnchor={m.anchor}
                style={{
                  fill: isSel ? 'var(--foreground)' : 'var(--muted-foreground)',
                  pointerEvents: 'none',
                }}
                className={isSel ? 'font-mono text-[11px] font-semibold' : 'font-mono text-[11px]'}
              >
                {m.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Touch-friendly method selector: real ≥44px <button> chips shown ONLY on
          narrow screens (the SVG dots above are deactivated there), so each
          method is selectable exactly once per input mode. */}
      {narrow && (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label={copy.heading}>
        {METHODS.map((m) => {
          const isSel = m.id === selected;
          const color = m.quant === 'weight' ? EMERALD : 'var(--primary)';
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setSelected(m.id)}
              aria-pressed={isSel}
              aria-label={copy.pickAria(m.label)}
              className={[
                'min-h-[44px] inline-flex items-center px-3 rounded border font-mono text-xs transition-colors',
                isSel
                  ? 'border-primary text-foreground ring-1 ring-primary'
                  : 'border-border/60 text-muted-foreground hover:bg-foreground/5',
              ].join(' ')}
            >
              <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: color }} />
              {m.label}
            </button>
          );
        })}
        </div>
      )}

      {/* axis-honesty note */}
      <p className="text-[11px] text-muted-foreground">{copy.axisNote}</p>

      {/* detail panel for the selected method */}
      <div className="rounded-md border border-border bg-card p-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <span className="font-mono text-[13px] font-semibold text-foreground">{sel.label}</span>
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
            <span>
              {copy.panelQuant}:{' '}
              <span
                style={{
                  color: sel.quant === 'weight' ? EMERALD : 'var(--primary)',
                }}
              >
                {quantLabel(sel.quant, det.quant.includes('base'))}
              </span>
            </span>
            <span>
              {copy.panelBits}: <span className="text-foreground">{det.bits}</span>
            </span>
            <span>
              {copy.panelRef}: <span className="text-foreground">{det.ref}</span>
            </span>
          </span>
        </div>
        <p className="mt-1.5 min-h-[2.5rem] text-[13px] text-foreground/90">{det.idea}</p>
      </div>

      {/* big-picture caption with the two traps */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}
