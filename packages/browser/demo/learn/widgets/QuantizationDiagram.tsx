import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * QuantizationDiagram — the "what if we quantized it" explorer for the
 * quantization sub-chapter. Fewer bits per weight (bf16 → fp8 / int8 / int4)
 * shrinks the model and speeds the memory-bound decode the course already
 * taught — at a quality cost.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers
 * (createRoot replaces it on boot) and is robust while the model downloads. The
 * first render is deterministic and identical server vs client; the only state
 * is the selected format, so it needs no prefers-reduced-motion handling.
 *
 * THE LESSON (every number verified against Inference Engineering, Kiely /
 * Baseten 2026, plus the arXiv primaries — none invented):
 *   - bytes/param = bits/8. The course model is Qwen3.5-0.8B (~0.8B params), so
 *     its weight footprint in GB (10^9 bytes, the book's "≈1 GB per billion
 *     params in fp8" convention) is just 0.8 × bytes/param:
 *       fp32 4 B/param → 3.2 GB · bf16 2 B → 1.6 GB (THE DEMO SHIPS THIS) ·
 *       fp8 1 B → 0.8 GB · int8 1 B → 0.8 GB · int4 0.5 B → 0.4 GB.
 *   - SPEED: a batch-1 decode step is memory-bandwidth-bound (see the roofline
 *     sub-chapter), so cutting bytes/weight directly relaxes the bottleneck. But
 *     the book is explicit it is NOT a clean 2× per level — overhead means
 *     "30 to 50 percent better performance" per single precision level down
 *     (p.120). We model ≈ ×1.4 per level down, compounding: fp32 0.7× · bf16 1.0×
 *     (baseline) · fp8/int8 1.4× · int4 1.9× (≈1.4²). Labelled approximate.
 *   - QUALITY (qualitative — the book gives NO perplexity numbers, so neither do
 *     we): book sensitivity ranking least→most fragile is Weights < Activations
 *     < KV cache < Attention. FP8/MXFP8 is "the sweet spot for performance
 *     without sacrificing quality"; "integer formats are not suitable for
 *     quality-sensitive workloads due to their lack of dynamic range" → for
 *     quality-sensitive work "stick with floating point".
 *   - MECHANISM: quantization maps continuous weights onto a discrete grid via a
 *     scale factor (+ optional zero-point): real ≈ scale × (q − zero_point).
 *     Symmetric (weights) vs asymmetric (activations); granularity per-tensor /
 *     per-channel / per-block.
 *   - METHODS (cited accurately): GPTQ (weight-only, Hessian-based, 2210.17323),
 *     AWQ (weight-only despite the name — activations only PICK salient weight
 *     channels, 2306.00978), SmoothQuant (weight+activation W8A8, migrates
 *     difficulty act→weight, 2211.10438), LLM.int8()/bitsandbytes (W8A8, keeps
 *     outlier dims in fp16, 2208.07339).
 *
 * HONESTY: the browser demo ships bf16 (no quantization); production serving
 * commonly runs fp8/int8/int4. The book notes integer / sub-4-bit quant is
 * mostly a LOCAL-inference practice (GGUF), distinct from datacenter serving —
 * so bf16 here is a defensible, honestly-labelled choice. A quantized copy of
 * this same model would be ≈2× (int8) to ≈4× (int4) smaller and a meaningful
 * chunk faster to decode.
 */

type Fmt = 'fp32' | 'bf16' | 'fp8' | 'int8' | 'int4';
const FORMATS: Fmt[] = ['fp32', 'bf16', 'fp8', 'int8', 'int4'];
const DEMO_FMT: Fmt = 'bf16'; // the in-browser demo ships this — "you are here"

// Tailwind palette hexes for SVG fills/strokes (SVG can't take TW classes).
const EMERALD = '#10b981'; // good / keep
const RED = '#ef4444'; // cost / loss

// Bit-layout segment kinds, drawn as a colored strip for floats.
type Seg = { kind: 'sign' | 'exp' | 'mant'; bits: number };

// ── Per-format ground truth (verified — do not edit numbers without a source). ─
type FormatInfo = {
  /** total bits per param */
  bits: number;
  /** bytes per param = bits / 8 */
  bytesPerParam: number;
  /** weight footprint in GB for ~0.8B params (10^9 bytes) */
  gb: number;
  /** float bit layout (sign/exp/mantissa); undefined for integer formats */
  segs?: Seg[];
  /** integer level count, for int formats */
  intLevels?: number;
  /** relative decode speed vs bf16 (=1.0) */
  speed: number;
  /** coarse quality rating 1..5 (ordinal pips only — the book gives NO perplexity numbers) */
  qualityLevel: number;
  /** is this a floating-point format? */
  isFloat: boolean;
};

const INFO: Record<Fmt, FormatInfo> = {
  fp32: {
    bits: 32,
    bytesPerParam: 4,
    gb: 3.2,
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 8 },
      { kind: 'mant', bits: 23 },
    ],
    speed: 0.7,
    qualityLevel: 5,
    isFloat: true,
  },
  bf16: {
    bits: 16,
    bytesPerParam: 2,
    gb: 1.6,
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 8 },
      { kind: 'mant', bits: 7 },
    ],
    speed: 1.0,
    qualityLevel: 5,
    isFloat: true,
  },
  fp8: {
    bits: 8,
    bytesPerParam: 1,
    gb: 0.8,
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 4 },
      { kind: 'mant', bits: 3 },
    ],
    speed: 1.4,
    qualityLevel: 4,
    isFloat: true,
  },
  int8: {
    bits: 8,
    bytesPerParam: 1,
    gb: 0.8,
    intLevels: 256,
    speed: 1.4,
    qualityLevel: 4,
    isFloat: false,
  },
  int4: {
    bits: 4,
    bytesPerParam: 0.5,
    gb: 0.4,
    intLevels: 16,
    speed: 1.9,
    qualityLevel: 3,
    isFloat: false,
  },
};

// Segment fills: sign = foreground, exponent = primary, mantissa = muted.
const SEG_FILL: Record<Seg['kind'], string> = {
  sign: 'var(--foreground)',
  exp: 'var(--primary)',
  mant: 'var(--muted-foreground)',
};

const MAX_GB = INFO.fp32.gb; // 3.2 — the longest footprint bar
const MAX_SPEED = INFO.int4.speed; // 1.9 — the fastest decode readout
const QUALITY_MAX = 5; // quality is a coarse 5-point ORDINAL rating (book gives no perplexity numbers), not a measurement

// ── Per-locale copy. COPY.en strings are the original English. zh keeps the
// glossary terms English (quantization, bf16, fp8, int8, int4, fp32, scale
// factor, GPTQ, AWQ, SmoothQuant, LLM.int8(), KV cache, decode, memory-bound,
// bytes/param, GB, GPU, weight, token, perplexity) and uses full-width
// punctuation; only the surrounding prose translates. ────────────────────────
const COPY = {
  en: {
    heading: 'Quantization — fewer bits per weight, smaller & faster',
    formatLabel: 'precision',
    youAreHere: 'you are here · the demo ships bf16',
    pickAria: (f: string) => `Select the ${f} precision format`,
    // section labels
    layoutHeading: 'bit layout · bytes/param',
    footprintHeading: 'weight footprint (~0.8B params)',
    speedHeading: 'decode speed vs bf16',
    qualityHeading: 'quality',
    qualityScale: 'ordinal · not measured',
    // bit-layout readouts
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    bytesPerParam: (b: number) => `${b} bytes/param`,
    fp8Max: 'fp8 E4M3 · max ±448',
    intLevels: (n: number) => `${n} integer levels`,
    int8Layout: 'int8 · 256 integer levels',
    int4Layout: 'int4 · 16 integer levels',
    // footprint
    gbUnit: 'GB',
    shrinkNote: (x: string) => `≈ ${x} smaller than bf16`,
    sameAsBf: 'same size, but a tighter grid',
    baselineSize: 'baseline — the demo ships this',
    // speed
    speedX: (s: number) => `${s.toFixed(1)}×`,
    speedCaveat: '≈ +30–50% per level down — overhead means it is NOT a clean 2×',
    speedBaseline: 'baseline (1.0×) — memory-bound decode',
    speedSlower: 'slower — 2× the bytes to stream per weight',
    // quality, per format
    qualityLine: {
      fp32: 'reference — full precision',
      bf16: '~lossless — it is the training precision',
      fp8: 'sweet spot — near-zero perceptible loss',
      int8: 'good — with outlier handling (SmoothQuant / LLM.int8())',
      int4: 'can be near-lossless with GPTQ / AWQ, but integers lack dynamic range → risky for quality-sensitive work; sub-4-bit is the cliff (local-use territory)',
    } satisfies Record<Fmt, string>,
    sensitivity: 'book fragility order (least → most): weights < activations < KV cache < attention',
    // captions
    mechanism: (
      <>
        Mechanism — quantization snaps continuous weights onto a discrete grid through a{' '}
        <strong>scale factor</strong> (plus an optional zero-point):{' '}
        <span className="font-mono">real ≈ scale × (q − zero_point)</span>. Weights use a{' '}
        <em>symmetric</em> grid, activations an <em>asymmetric</em> one; granularity can be per-tensor, per-channel, or
        per-block. Weight-only methods <strong>GPTQ</strong> (Hessian-based) and <strong>AWQ</strong> (named for
        activations, but it only <em>picks</em> the salient weight channels to keep precise) quantize the weights alone;{' '}
        <strong>SmoothQuant</strong> and <strong>LLM.int8()</strong> are W8A8 (weights <em>and</em> activations) —
        SmoothQuant migrates the hard dynamic range from activations into weights, LLM.int8() keeps the rare outlier
        dimensions in fp16.
      </>
    ),
    honesty: (
      <>
        Honesty for this demo: the in-browser model ships in <strong>bf16</strong> — no quantization at all. Datacenter
        serving commonly runs <span className="font-mono">fp8</span> / <span className="font-mono">int8</span>, while more
        aggressive integer and sub-4-bit quant (<span className="font-mono">int4</span> and below) is mostly a{' '}
        <strong>local-inference</strong> practice (GGUF / Ollama / llama.cpp). So bf16 here is a defensible,
        honestly-labelled choice — but a quantized copy of <em>this same model</em> would be ≈2× (int8) to ≈4× (int4)
        smaller and a meaningful chunk faster to decode.
      </>
    ),
    layoutAria: (f: string) =>
      `Bit layout strip for ${f}: colored segments for sign, exponent, and mantissa on floating-point formats, or a count of integer levels for integer formats, with the bytes-per-parameter.`,
    footprintAria: (f: string, gb: number) =>
      `Weight-footprint bar for ${f}: ${gb} gigabytes for the roughly 0.8-billion-parameter model, shown against the 3.2-gigabyte fp32 maximum.`,
    speedAria: (f: string, s: number) =>
      `Relative decode-speed bar for ${f}: about ${s.toFixed(1)} times the bf16 baseline, since a batch-1 decode step is memory-bandwidth-bound.`,
    qualityAria: (f: string) =>
      `Qualitative quality bar for ${f} — a hedged indicator, not a perplexity number, since the book gives none.`,
  },
  zh: {
    heading: 'quantization——每个 weight 用更少的 bit，更小更快',
    formatLabel: '精度',
    youAreHere: '你在这里 · demo 用的是 bf16',
    pickAria: (f: string) => `选择 ${f} 精度格式`,
    layoutHeading: 'bit 布局 · bytes/param',
    footprintHeading: 'weight 占用（约 0.8B 参数）',
    speedHeading: 'decode 速度（相对 bf16）',
    qualityHeading: '质量',
    qualityScale: '序数 · 非测量',
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    bytesPerParam: (b: number) => `${b} bytes/param`,
    fp8Max: 'fp8 E4M3 · max ±448',
    intLevels: (n: number) => `${n} 个整数档位`,
    int8Layout: 'int8 · 256 个整数档位',
    int4Layout: 'int4 · 16 个整数档位',
    gbUnit: 'GB',
    shrinkNote: (x: string) => `≈ 比 bf16 小 ${x}`,
    sameAsBf: '大小相同，但格点更密',
    baselineSize: '基准——demo 用的就是这个',
    speedX: (s: number) => `${s.toFixed(1)}×`,
    speedCaveat: '≈ 每降一级 +30–50%——因为有开销，并不是干净的 2×',
    speedBaseline: '基准（1.0×）——memory-bound decode',
    speedSlower: '更慢——每个 weight 要搬运 2 倍的字节',
    qualityLine: {
      fp32: '参考——全精度',
      bf16: '≈ 无损——它就是训练精度',
      fp8: 'sweet spot——几乎察觉不到的损失',
      int8: '不错——配合 outlier 处理（SmoothQuant / LLM.int8()）',
      int4: '配合 GPTQ / AWQ 有时能近乎无损，但整数缺少动态范围 → 对质量敏感的任务有风险；低于 4-bit 就是悬崖（本地使用的地盘）',
    } satisfies Record<Fmt, string>,
    sensitivity: '书中的脆弱度排序（由弱到强）：weights < activations < KV cache < attention',
    mechanism: (
      <>
        机制——quantization 通过一个 <strong>scale factor</strong>（外加可选的 zero-point）把连续的 weight 吸附到离散的格点上：
        <span className="font-mono">real ≈ scale × (q − zero_point)</span>。weight 用 <em>对称</em> 格点，activation 用{' '}
        <em>非对称</em> 格点；粒度可以是 per-tensor、per-channel 或 per-block。weight-only 方法 <strong>GPTQ</strong>（基于
        Hessian）和 <strong>AWQ</strong>（名字带 activation，但它只是<em>挑出</em>需要保精度的显著 weight 通道）只量化
        weight；<strong>SmoothQuant</strong> 和 <strong>LLM.int8()</strong> 是 W8A8（weight <em>和</em> activation 一起）——
        SmoothQuant 把困难的动态范围从 activation 迁移到 weight，LLM.int8() 把罕见的 outlier 维度保留在 fp16。
      </>
    ),
    honesty: (
      <>
        对本 demo 要诚实：浏览器内的模型以 <strong>bf16</strong> 发布——完全没有 quantization。数据中心的服务常跑{' '}
        <span className="font-mono">fp8</span> / <span className="font-mono">int8</span>，而更激进的整数与低于 4-bit 的
        quant（<span className="font-mono">int4</span> 及更低）多半是 <strong>本地推理</strong> 的做法（GGUF / Ollama /
        llama.cpp）。所以这里选 bf16 是一个站得住、且诚实标注的选择——但 <em>同一个模型</em> 的量化副本会小 ≈2×（int8）到
        ≈4×（int4），decode 也会快上一截。
      </>
    ),
    layoutAria: (f: string) =>
      `${f} 的 bit 布局条：浮点格式用彩色段表示 sign、exponent、mantissa，整数格式则显示整数档位数量，并标注 bytes/param。`,
    footprintAria: (f: string, gb: number) =>
      `${f} 的 weight 占用条：约 0.8B 参数的模型为 ${gb} GB，与 3.2 GB 的 fp32 上限对比。`,
    speedAria: (f: string, s: number) =>
      `${f} 的相对 decode 速度条：约为 bf16 基准的 ${s.toFixed(1)} 倍，因为 batch-1 的 decode step 受 memory bandwidth 限制。`,
    qualityAria: (f: string) => `${f} 的定性质量条——这是一个有保留的指示，而非 perplexity 数字，因为书里没有给。`,
  },
} as const;

// ── SVG geometry. EN labels are wider than zh, so every box is sized for EN. ──
// A single column of horizontal bars: the bit-layout strip, then a footprint
// bar, then a decode-speed bar, then a quality bar. Wide section labels sit
// ABOVE their bars (never wedged beside them), so nothing collides across
// locales. The track starts well right of the left labels.
const VB_W = 560;
const VB_H = 300;
const PAD_L = 14;
const PAD_R = 14;
const TRACK_X = PAD_L; // bars are full-width; labels sit above
const TRACK_W = VB_W - PAD_L - PAD_R; // 532

// Row anchors (y of each section's heading text; bars sit below).
const STRIP_LABEL_Y = 16;
const STRIP_Y = 26;
const STRIP_H = 30;
const FOOT_LABEL_Y = 92;
const FOOT_Y = 100;
const BAR_H = 26;
const SPEED_LABEL_Y = 158;
const SPEED_Y = 166;
const QUAL_LABEL_Y = 224;
const QUAL_Y = 232;

function sectionLabel(text: string, value: string, y: number) {
  return (
    <>
      <text x={TRACK_X} y={y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
        {text}
      </text>
      <text
        x={TRACK_X + TRACK_W}
        y={y}
        textAnchor="end"
        style={{ fill: 'var(--foreground)' }}
        className="font-mono text-[11px] font-semibold"
      >
        {value}
      </text>
    </>
  );
}

export function QuantizationDiagram() {
  const copy = COPY[useLocale()];
  const [fmt, setFmt] = React.useState<Fmt>(DEMO_FMT);
  const info = INFO[fmt];

  const isDemo = fmt === DEMO_FMT;
  const slower = info.speed < 1; // fp32 only
  const sameSizeAsBf = info.gb === INFO.bf16.gb; // fp8 / int8

  // Footprint + speed bar widths (fractions of the track).
  const footW = (info.gb / MAX_GB) * TRACK_W;
  const speedW = (info.speed / MAX_SPEED) * TRACK_W;
  // bf16 reference tick on the speed bar (1.0× position).
  const speedRefX = TRACK_X + (1 / MAX_SPEED) * TRACK_W;

  // shrink-vs-bf16 readout text for the footprint row.
  const shrinkText = (() => {
    if (isDemo) return copy.baselineSize;
    if (sameSizeAsBf) return copy.sameAsBf;
    if (info.gb < INFO.bf16.gb) {
      const factor = INFO.bf16.gb / info.gb; // 1.6/0.4 = 4
      return copy.shrinkNote(`${factor.toFixed(0)}×`);
    }
    return ''; // fp32 is larger; nothing to claim here
  })();

  // bit-layout strip: float segments OR integer-levels block.
  const totalBits = info.bits;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + format ladder */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.formatLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {FORMATS.map((f, i) => (
              <button
                key={f}
                type="button"
                onClick={() => setFmt(f)}
                aria-pressed={fmt === f}
                aria-label={copy.pickAria(f)}
                className={[
                  'px-2 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: fmt === f ? 'var(--primary)' : 'transparent',
                  color: fmt === f ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Locked "you are here · bf16" badge (always shown — bf16 is the demo). */}
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={[
            'inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-medium',
            isDemo
              ? 'border-foreground/40 bg-foreground/5 text-foreground/80'
              : 'border-border bg-muted text-muted-foreground',
          ].join(' ')}
        >
          🔒 {copy.youAreHere}
        </span>
      </div>

      {/* The four-bar diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={`${copy.layoutAria(fmt)} ${copy.footprintAria(fmt, info.gb)} ${copy.speedAria(
          fmt,
          info.speed,
        )} ${copy.qualityAria(fmt)}`}
      >
        {/* ── bit layout strip + bytes/param ── */}
        {sectionLabel(copy.layoutHeading, copy.bytesPerParam(info.bytesPerParam), STRIP_LABEL_Y)}
        {info.isFloat && info.segs ? (
          <>
            {(() => {
              let cursor = TRACK_X;
              return info.segs.map((seg) => {
                const w = (seg.bits / totalBits) * TRACK_W;
                const x = cursor;
                cursor += w;
                const showLabel = w > 40;
                const labelText =
                  seg.kind === 'sign' ? copy.sign : seg.kind === 'exp' ? copy.exp : copy.mant;
                return (
                  <g key={seg.kind}>
                    <rect
                      x={x}
                      y={STRIP_Y}
                      width={Math.max(1, w - 1)}
                      height={STRIP_H}
                      rx={3}
                      style={{ fill: SEG_FILL[seg.kind], opacity: seg.kind === 'mant' ? 0.35 : 0.85 }}
                    />
                    {showLabel ? (
                      <text
                        x={x + w / 2}
                        y={STRIP_Y + STRIP_H / 2 + 4}
                        textAnchor="middle"
                        style={{ fill: 'var(--background)' }}
                        className="text-[10px] font-medium"
                      >
                        {labelText} {seg.bits}
                      </text>
                    ) : (
                      <text
                        x={x + w / 2}
                        y={STRIP_Y + STRIP_H / 2 + 4}
                        textAnchor="middle"
                        style={{ fill: 'var(--background)' }}
                        className="text-[10px] font-semibold"
                      >
                        {seg.bits}
                      </text>
                    )}
                  </g>
                );
              });
            })()}
            {/* fp8 max-value note sits ABOVE-right of the strip, never inside it */}
            {fmt === 'fp8' ? (
              <text
                x={TRACK_X + TRACK_W}
                y={STRIP_Y + STRIP_H + 14}
                textAnchor="end"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {copy.fp8Max}
              </text>
            ) : null}
          </>
        ) : (
          <>
            {/* integer-levels block (no sign/exp/mantissa for ints) */}
            <rect
              x={TRACK_X}
              y={STRIP_Y}
              width={TRACK_W}
              height={STRIP_H}
              rx={3}
              style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
              strokeWidth={1}
            />
            <text
              x={TRACK_X + TRACK_W / 2}
              y={STRIP_Y + STRIP_H / 2 + 4}
              textAnchor="middle"
              style={{ fill: 'var(--foreground)' }}
              className="font-mono text-[11px] font-semibold"
            >
              {fmt === 'int8' ? copy.int8Layout : copy.int4Layout}
            </text>
          </>
        )}

        {/* ── footprint bar ── */}
        {sectionLabel(copy.footprintHeading, `${info.gb.toFixed(1)} ${copy.gbUnit}`, FOOT_LABEL_Y)}
        {/* full-width track ghost (the fp32 maximum) */}
        <rect
          x={TRACK_X}
          y={FOOT_Y}
          width={TRACK_W}
          height={BAR_H}
          rx={4}
          fill="none"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <rect
          x={TRACK_X}
          y={FOOT_Y}
          width={Math.max(2, footW)}
          height={BAR_H}
          rx={4}
          style={{ fill: 'var(--primary)', opacity: isDemo ? 0.85 : 0.55 }}
        />
        {shrinkText ? (
          <text
            x={TRACK_X + TRACK_W}
            y={FOOT_Y + BAR_H + 13}
            textAnchor="end"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[10px]"
          >
            {shrinkText}
          </text>
        ) : null}

        {/* ── decode-speed bar (relative to bf16 = 1.0×) ── */}
        {sectionLabel(copy.speedHeading, copy.speedX(info.speed), SPEED_LABEL_Y)}
        <rect
          x={TRACK_X}
          y={SPEED_Y}
          width={TRACK_W}
          height={BAR_H}
          rx={4}
          fill="none"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <rect
          x={TRACK_X}
          y={SPEED_Y}
          width={Math.max(2, speedW)}
          height={BAR_H}
          rx={4}
          style={{ fill: slower ? RED : EMERALD, opacity: 0.55 }}
        />
        {/* bf16 1.0× reference tick */}
        <line
          x1={speedRefX}
          y1={SPEED_Y - 4}
          x2={speedRefX}
          y2={SPEED_Y + BAR_H + 4}
          style={{ stroke: 'var(--foreground)', opacity: 0.6 }}
          strokeWidth={1}
          strokeDasharray="3 2"
        />
        <text
          x={TRACK_X + TRACK_W}
          y={SPEED_Y + BAR_H + 13}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {isDemo ? copy.speedBaseline : slower ? copy.speedSlower : copy.speedCaveat}
        </text>

        {/* ── quality rating (coarse 5-point ORDINAL pips — the book gives NO perplexity
            numbers, so we never render a precise/continuous quality bar) ── */}
        {sectionLabel(copy.qualityHeading, copy.qualityScale, QUAL_LABEL_Y)}
        {Array.from({ length: QUALITY_MAX }).map((_, i) => {
          const gap = 6;
          const pipW = (TRACK_W - gap * (QUALITY_MAX - 1)) / QUALITY_MAX;
          const filled = i < info.qualityLevel;
          return (
            <rect
              key={i}
              x={TRACK_X + i * (pipW + gap)}
              y={QUAL_Y}
              width={pipW}
              height={BAR_H}
              rx={4}
              style={{
                fill: filled ? EMERALD : 'var(--muted)',
                opacity: filled ? 0.55 : 1,
                stroke: 'var(--border)',
              }}
              strokeWidth={1}
            />
          );
        })}
      </svg>

      {/* per-format quality line + book sensitivity order */}
      <p className="min-h-[2.5rem] text-[13px] text-foreground/90">
        <span className="font-mono text-foreground">{fmt}</span> — {copy.qualityLine[fmt]}
      </p>
      <p className="text-[11px] text-muted-foreground">{copy.sensitivity}</p>

      {/* Mechanism + methods caption */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.mechanism}</p>

      {/* Honesty framing: this demo (bf16) vs production serving */}
      <p className="text-[12px] text-muted-foreground">{copy.honesty}</p>
    </div>
  );
}
