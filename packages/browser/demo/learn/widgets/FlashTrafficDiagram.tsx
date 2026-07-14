import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * FlashTrafficDiagram — chapter 4 deep-dive "Flash Attention", the
 * "fraction of the traffic" claim made quantitative.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It puts numbers on the score-matrix traffic. For the course's Qwen3.5-0.8B
 * full-attention layer (H = 8 query heads, scores in bf16 = 2 bytes), the naïve
 * recipe drags the H × N × N score matrix across the slow HBM bus FOUR times —
 * write the raw scores, read them back for softmax, write the probabilities,
 * read them again to weight V. That is 4 × 8 × N² × 2 bytes of pure score
 * traffic: ≈1.07 GB at N = 4,096 (the matrix itself is ~268 MB — same figure
 * as the memory-wall page — times 4 trips), exploding to ≈17 GB at N = 16,384.
 * FlashAttention's score traffic is 0 bytes at every N: each small score tile
 * is computed, folded into the streaming softmax, and discarded without ever
 * leaving SRAM. Its largest live score tensor is one fixed 128×128 fp32 tile
 * (≈64 KB), flat no matter how long the sequence gets.
 *
 * A slider sweeps N over {512 … 16,384}; two bar pairs (bus traffic, largest
 * live score tensor) update live, naïve blowing up quadratically while flash
 * stays at zero / flat. Bars use a square-root scale so small values stay
 * visible (labelled). The SVG shows the four bus trips cycling on the naïve
 * side. Honors prefers-reduced-motion (all four trips shown at once, no
 * cycling, no pulse).
 */

// ── The model the chapter uses: Qwen3.5-0.8B full-attention layer. ──────────
const HEADS = 8; // query heads
const BYTES_PER_SCORE = 2; // bf16
const BUS_TRIPS = 4; // write S · read for softmax · write P · read for P·V
const FLASH_TILE_BYTES = 128 * 128 * 4; // one 128×128 fp32 score tile ≈ 64 KB

const SEQ_STEPS = [512, 1024, 2048, 4096, 8192, 16384];
const DEFAULT_SEQ_IDX = 3; // N = 4,096 — matches the chapter prose
const MAX_N = SEQ_STEPS[SEQ_STEPS.length - 1];

/** All H heads' N×N scores in bf16 — the biggest live tensor the naïve recipe holds. */
function scoreMatrixBytes(n: number): number {
  return HEADS * n * n * BYTES_PER_SCORE;
}

/** Score bytes the naïve recipe drags across the HBM bus: 4 full trips. */
function naiveTrafficBytes(n: number): number {
  return BUS_TRIPS * scoreMatrixBytes(n);
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b} B`;
}

function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

function fmtSeqShort(n: number): string {
  return n >= 1024 ? `${n / 1024}K` : `${n}`;
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

// ── The four bus trips of the naïve recipe. Directions are structural; the
// labels/captions live in the per-locale COPY record below. ──────────────────
const TRIP_DIRS: Array<'toHbm' | 'toSram'> = ['toHbm', 'toSram', 'toHbm', 'toSram'];
const TRIP_MS = 1400;

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
const COPY = {
  en: {
    trips: [
      { label: '① write scores', caption: 'write the raw N×N scores out to HBM' },
      { label: '② read for softmax', caption: 'read them all back in to run softmax' },
      { label: '③ write probs', caption: 'write the probabilities back out' },
      { label: '④ read for P·V', caption: 'read them one more time to weight V' },
    ],
    title: 'Score traffic over the bus — naïve vs flash, in bytes',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    ariaLabel:
      'Two panels comparing memory-bus traffic for the attention score matrix. Naïve attention drags the full N by N score matrix across the slow bus between HBM and SRAM four times: write scores, read for softmax, write probabilities, read to weight V. FlashAttention computes each small score tile inside SRAM and discards it, so zero score bytes cross the bus.',
    naivePanelTitle: 'Naïve attention',
    naivePanelSubtitle: 'the full N×N score matrix rides the bus — four times',
    flashPanelTitle: 'FlashAttention',
    flashPanelSubtitle: 'the score tile never leaves SRAM',
    slow: 'slow',
    fast: 'fast',
    nxnScores: 'N×N scores',
    noScoresHere: 'no scores here',
    tileLivesDies: 'tile lives & dies',
    tripLine: (n: number, caption: string) => `trip ${n} of 4 · ${caption}`,
    allTrips: '4 trips, every one carries the whole score matrix',
    perTrip: (per: string, total: string) => `each trip moves ${per} → 4 trips = ${total}`,
    scoreBytesOverBus: 'score bytes over the bus',
    flashCaption: 'each tile is computed, folded into the running softmax, discarded',
    flashTrafficLine: 'score traffic = 0 B — at every N',
    seqLen: 'Sequence length',
    tokensSuffix: 'tokens',
    dragHint: 'drag to stretch the context',
    sliderAria: 'Sequence length in tokens',
    sliderValueText: (n: string) => `${n} tokens`,
    metricTraffic: 'Score bytes crossing the HBM bus',
    naiveBar: 'naïve',
    flashBar: 'flash',
    metricTrafficNote: (heads: number) =>
      `naïve: 4 trips × ${heads} heads × N² scores in bf16 (2 bytes) · flash: the tile dies in SRAM — nothing to move`,
    metricLive: 'Largest live score tensor',
    metricLiveNote: (heads: number) =>
      `naïve: all ${heads} heads' N×N scores at once — grows with N² · flash: one 128×128 fp32 tile — flat at every N`,
    sqrtNote: 'Bars are drawn on a square-root scale so the small values stay visible; the numbers are exact.',
    reducedNote: 'Reduced motion: showing all four bus trips at once instead of cycling through them.',
    honestFootnote:
      'Honest footnote: Q, K, V and the output still cross the bus in both schemes — tens of MB at these sizes (≈42 MB at 4K tokens for this model), similar for both, growing only linearly with N — and flash re-reads the K/V tiles a few times. The point is that the unbounded N² score term disappears entirely.',
  },
  zh: {
    trips: [
      { label: '① 写出分数', caption: '把原始的 N×N 分数写出到 HBM' },
      { label: '② 读回做 softmax', caption: '把它们全部读回来跑 softmax' },
      { label: '③ 写出概率', caption: '把概率再写回去' },
      { label: '④ 读回做 P·V', caption: '再读一次来给 V 加权' },
    ],
    title: '跨总线的分数流量——朴素 vs flash，以字节计',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    ariaLabel:
      '两块面板对比注意力分数矩阵的内存总线流量。朴素注意力让完整的 N×N 分数矩阵在 HBM 与 SRAM 之间的慢速总线上往返四趟：写出分数、读回做 softmax、写出概率、读回给 V 加权。FlashAttention 在 SRAM 内部算出每个小分数块并将其丢弃，因此零分数字节跨过总线。',
    naivePanelTitle: '朴素注意力',
    naivePanelSubtitle: '完整的 N×N 分数矩阵在总线上往返——整整四趟',
    flashPanelTitle: 'FlashAttention',
    flashPanelSubtitle: '分数块从不离开 SRAM',
    slow: '慢',
    fast: '快',
    nxnScores: 'N×N 分数',
    noScoresHere: '这里没有分数',
    tileLivesDies: '分数块在此生灭',
    tripLine: (n: number, caption: string) => `第 ${n}/4 趟 · ${caption}`,
    allTrips: '4 趟，每一趟都搬运整个分数矩阵',
    perTrip: (per: string, total: string) => `每趟搬 ${per} → 4 趟 = ${total}`,
    scoreBytesOverBus: '跨总线的分数字节',
    flashCaption: '每个分数块算完即并入滚动 softmax，然后丢弃',
    flashTrafficLine: '分数流量 = 0 B——任何 N 都如此',
    seqLen: '序列长度',
    tokensSuffix: '个 token',
    dragHint: '拖动以拉长上下文',
    sliderAria: '序列长度（token 数）',
    sliderValueText: (n: string) => `${n} 个 token`,
    metricTraffic: '跨过 HBM 总线的分数字节',
    naiveBar: '朴素',
    flashBar: 'flash',
    metricTrafficNote: (heads: number) =>
      `朴素：4 趟 × ${heads} 个头 × N² 个 bf16 分数（2 字节）· flash：分数块死在 SRAM 里——无需搬运`,
    metricLive: '最大的存活分数张量',
    metricLiveNote: (heads: number) =>
      `朴素：全部 ${heads} 个头的 N×N 分数同时存活——随 N² 增长 · flash：一个 128×128 fp32 块——任何 N 都不变`,
    sqrtNote: '条形按平方根刻度绘制，让小值保持可见；数字本身是精确的。',
    reducedNote: '减少动态效果：同时展示全部四趟总线往返，不再循环播放。',
    honestFootnote:
      '诚实的脚注：两种方案里 Q、K、V 和输出仍要过总线——在这些尺寸下是几十 MB（此模型在 4K token 时 ≈42 MB），两边相近，且只随 N 线性增长——flash 还会把 K/V 块重读几次。重点在于：无界的 N² 分数项彻底消失了。',
  },
} as const;

// ── SVG geometry: two side-by-side panels, naïve left, flash right. ─────────
const VB_W = 760;
const VB_H = 248;
const PANEL = { y: 8, w: 362, h: 232 };
const NAIVE_X = 10;
const FLASH_X = 388;
// Inside a panel: HBM box on the left, SRAM box on the right, bus between.
const BOX = { y: 60, w: 84, h: 124 };
const LANE_YS = [84, 112, 140, 168]; // the four trip arrows, stacked

export function FlashTrafficDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [seqIdx, setSeqIdx] = React.useState(DEFAULT_SEQ_IDX);
  const [trip, setTrip] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (reducedMotion) setPlaying(false);
  }, [reducedMotion]);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setTrip((x) => (x + 1) % TRIP_DIRS.length), TRIP_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const n = SEQ_STEPS[seqIdx];
  const matrixBytes = scoreMatrixBytes(n); // also = largest live tensor, naïve
  const trafficBytes = naiveTrafficBytes(n);

  // Bar fractions on a fixed √ scale (max = naïve at N = 16,384), so the bars
  // stay visible at small N and the quadratic blow-up still reads clearly.
  const trafficFrac = Math.sqrt(trafficBytes / naiveTrafficBytes(MAX_N));
  const liveFrac = Math.sqrt(matrixBytes / scoreMatrixBytes(MAX_N));
  const flashLiveFrac = Math.sqrt(FLASH_TILE_BYTES / scoreMatrixBytes(MAX_N)); // ≈ 0.004 — a sliver

  // Under reduced motion no trip is singled out — all four are shown at once.
  const activeTrip = reducedMotion ? -1 : trip;

  const step = () => {
    setPlaying(false);
    setTrip((x) => (x + 1) % TRIP_DIRS.length);
  };

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1">
          {!reducedMotion ? (
            <>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 max-sm:py-2 max-sm:min-h-[44px] text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 max-sm:py-2 max-sm:min-h-[44px] text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {/* The diagram: two panels */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.ariaLabel}
      >
        {/* ── Naïve panel ── */}
        <Panel x={NAIVE_X} title={copy.naivePanelTitle} subtitle={copy.naivePanelSubtitle} />
        {/* HBM box with the parked score matrix */}
        <MemoryBox x={NAIVE_X + 14} name="HBM" hint={copy.slow}>
          <rect
            x={NAIVE_X + 14 + 22}
            y={BOX.y + 46}
            width={40}
            height={40}
            rx={5}
            style={{ fill: 'var(--primary)', fillOpacity: 0.16, stroke: 'var(--primary)' }}
            strokeWidth={1.5}
          />
          <text
            x={NAIVE_X + 14 + 42}
            y={BOX.y + 70}
            textAnchor="middle"
            style={{ fill: 'var(--primary)' }}
            className="font-mono text-[10px] font-medium"
          >
            S
          </text>
          <text
            x={NAIVE_X + 14 + 42}
            y={BOX.y + 102}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[9px]"
          >
            {copy.nxnScores}
          </text>
        </MemoryBox>
        {/* SRAM box: just the math */}
        <MemoryBox x={NAIVE_X + 264} name="SRAM" hint={copy.fast}>
          <text
            x={NAIVE_X + 264 + 42}
            y={BOX.y + 72}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[9px]"
          >
            softmax · P·V
          </text>
        </MemoryBox>
        {/* The four trips */}
        {TRIP_DIRS.map((dir, i) => (
          <TripArrow
            key={copy.trips[i].label}
            x1={NAIVE_X + 106}
            x2={NAIVE_X + 256}
            y={LANE_YS[i]}
            dir={dir}
            label={copy.trips[i].label}
            active={i === activeTrip}
            allEqual={reducedMotion}
          />
        ))}
        {/* Naïve caption: which trip + the per-trip and total bytes */}
        <text x={NAIVE_X + 14} y={206} style={{ fill: 'var(--foreground)' }} className="text-[11px]">
          {activeTrip >= 0 ? copy.tripLine(activeTrip + 1, copy.trips[activeTrip].caption) : copy.allTrips}
        </text>
        <text x={NAIVE_X + 14} y={224} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[10px]">
          {copy.perTrip(fmtBytes(matrixBytes), fmtBytes(trafficBytes))}
        </text>

        {/* ── Flash panel ── */}
        <Panel x={FLASH_X} title={copy.flashPanelTitle} subtitle={copy.flashPanelSubtitle} />
        {/* HBM box: pointedly empty of scores */}
        <MemoryBox x={FLASH_X + 14} name="HBM" hint={copy.slow}>
          <rect
            x={FLASH_X + 14 + 22}
            y={BOX.y + 46}
            width={40}
            height={40}
            rx={5}
            fill="none"
            style={{ stroke: 'var(--border)', opacity: 0.9 }}
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <text
            x={FLASH_X + 14 + 42}
            y={BOX.y + 102}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[9px]"
          >
            {copy.noScoresHere}
          </text>
        </MemoryBox>
        {/* SRAM box: the small score tile lives and dies here */}
        <MemoryBox x={FLASH_X + 264} name="SRAM" hint={copy.fast}>
          <rect
            x={FLASH_X + 264 + 29}
            y={BOX.y + 48}
            width={26}
            height={26}
            rx={4}
            style={{ fill: 'var(--primary)', fillOpacity: 0.2, stroke: 'var(--primary)' }}
            strokeWidth={1.5}
          />
          <text
            x={FLASH_X + 264 + 42}
            y={BOX.y + 65}
            textAnchor="middle"
            style={{ fill: 'var(--primary)' }}
            className="font-mono text-[9px] font-medium"
          >
            S
          </text>
          <text
            x={FLASH_X + 264 + 42}
            y={BOX.y + 92}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[9px]"
          >
            {copy.tileLivesDies}
          </text>
        </MemoryBox>
        {/* The bus, empty of score traffic */}
        <line
          x1={FLASH_X + 106}
          y1={LANE_YS[1] + 14}
          x2={FLASH_X + 256}
          y2={LANE_YS[1] + 14}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1.5}
          strokeDasharray="5 4"
        />
        <text
          x={FLASH_X + 181}
          y={LANE_YS[1] - 4}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.scoreBytesOverBus}
        </text>
        <text
          x={FLASH_X + 181}
          y={LANE_YS[2] + 8}
          textAnchor="middle"
          style={{ fill: 'var(--primary)' }}
          className="font-mono text-[16px] font-semibold"
        >
          0 B
        </text>
        <text x={FLASH_X + 14} y={206} style={{ fill: 'var(--foreground)' }} className="text-[11px]">
          {copy.flashCaption}
        </text>
        <text x={FLASH_X + 14} y={224} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[10px]">
          {copy.flashTrafficLine}
        </text>
      </svg>

      {/* Slider + the two metrics, side by side as bar pairs */}
      <div className="space-y-3 border-t border-border/60 pt-3">
        {/* N slider */}
        <div className="space-y-1">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[11px] text-muted-foreground">
              {copy.seqLen}{' '}
              <span className="font-mono text-foreground/90">N = {fmtInt(n)}</span> {copy.tokensSuffix}
            </div>
            <div className="text-[10px] text-muted-foreground/70">{copy.dragHint}</div>
          </div>
          <input
            type="range"
            min={0}
            max={SEQ_STEPS.length - 1}
            step={1}
            value={seqIdx}
            onChange={(e) => setSeqIdx(Number(e.target.value))}
            aria-label={copy.sliderAria}
            aria-valuetext={copy.sliderValueText(fmtInt(n))}
            className="w-full cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            style={{ accentColor: 'var(--primary)' }}
          />
          <div className="flex justify-between font-mono text-[9px] text-muted-foreground/70" aria-hidden>
            {SEQ_STEPS.map((s) => (
              <span key={s}>{fmtSeqShort(s)}</span>
            ))}
          </div>
        </div>

        {/* Metric 1: bus traffic */}
        <div className="space-y-1">
          <div className="text-[11px] font-medium text-foreground/90">{copy.metricTraffic}</div>
          <BarRow name={copy.naiveBar} frac={trafficFrac} valueText={fmtBytes(trafficBytes)} reducedMotion={reducedMotion} />
          <BarRow name={copy.flashBar} frac={0} valueText="0 B" reducedMotion={reducedMotion} />
          <div className="text-[10px] text-muted-foreground/70">{copy.metricTrafficNote(HEADS)}</div>
        </div>

        {/* Metric 2: largest live score tensor */}
        <div className="space-y-1">
          <div className="text-[11px] font-medium text-foreground/90">{copy.metricLive}</div>
          <BarRow name={copy.naiveBar} frac={liveFrac} valueText={fmtBytes(matrixBytes)} reducedMotion={reducedMotion} />
          <BarRow
            name={copy.flashBar}
            frac={flashLiveFrac}
            valueText={fmtBytes(FLASH_TILE_BYTES)}
            reducedMotion={reducedMotion}
          />
          <div className="text-[10px] text-muted-foreground/70">{copy.metricLiveNote(HEADS)}</div>
        </div>

        <p className="text-[10px] text-muted-foreground/70">{copy.sqrtNote}</p>
      </div>

      {reducedMotion ? <p className="text-[10px] text-muted-foreground/70">{copy.reducedNote}</p> : null}

      {/* Honesty footnote */}
      <p className="text-[10px] text-muted-foreground/70">{copy.honestFootnote}</p>
    </div>
  );
}

/** A panel card with its title and subtitle. */
function Panel({ x, title, subtitle }: { x: number; title: string; subtitle: string }) {
  return (
    <g>
      <rect
        x={x}
        y={PANEL.y}
        width={PANEL.w}
        height={PANEL.h}
        rx={10}
        style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
        strokeWidth={1.5}
      />
      <text x={x + 14} y={PANEL.y + 24} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
        {title}
      </text>
      <text x={x + 14} y={PANEL.y + 41} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {subtitle}
      </text>
    </g>
  );
}

/** A small memory-tier box (HBM or SRAM) with its name, speed hint, and children. */
function MemoryBox({
  x,
  name,
  hint,
  children,
}: {
  x: number;
  name: string;
  hint: string;
  children?: React.ReactNode;
}) {
  return (
    <g>
      <rect
        x={x}
        y={BOX.y}
        width={BOX.w}
        height={BOX.h}
        rx={8}
        style={{ fill: 'var(--bg)', stroke: 'var(--border)' }}
        strokeWidth={1}
      />
      <text
        x={x + BOX.w / 2}
        y={BOX.y + 20}
        textAnchor="middle"
        style={{ fill: 'var(--foreground)' }}
        className="text-[11px] font-semibold"
      >
        {name}
      </text>
      <text
        x={x + BOX.w / 2}
        y={BOX.y + 34}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[9px]"
      >
        {hint}
      </text>
      {children}
    </g>
  );
}

/** One bus trip: an arrow between the two memory boxes. The active trip is
 * highlighted; under reduced motion all four are drawn equally (allEqual). */
function TripArrow({
  x1,
  x2,
  y,
  dir,
  label,
  active,
  allEqual,
}: {
  x1: number;
  x2: number;
  y: number;
  dir: 'toHbm' | 'toSram';
  label: string;
  active: boolean;
  allEqual: boolean;
}) {
  const lit = active || allEqual;
  const tipX = dir === 'toHbm' ? x1 : x2;
  const tipDir = dir === 'toHbm' ? -1 : 1;
  return (
    <g style={{ opacity: lit ? 1 : 0.45 }}>
      <line
        x1={x1 + 4}
        y1={y}
        x2={x2 - 4}
        y2={y}
        style={{ stroke: active ? 'var(--primary)' : lit ? 'var(--foreground)' : 'var(--border)' }}
        strokeWidth={active ? 2 : 1.2}
      />
      <polygon
        points={`${tipX},${y} ${tipX - tipDir * 8},${y - 4.5} ${tipX - tipDir * 8},${y + 4.5}`}
        style={{ fill: active ? 'var(--primary)' : lit ? 'var(--foreground)' : 'var(--border)' }}
      />
      <text
        x={(x1 + x2) / 2}
        y={y - 5}
        textAnchor="middle"
        style={{ fill: active ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="text-[9px]"
      >
        {label}
      </text>
    </g>
  );
}

/** One labelled bar: name on the left, √-scaled fill, exact bytes on the right. */
function BarRow({
  name,
  frac,
  valueText,
  reducedMotion,
}: {
  name: string;
  frac: number;
  valueText: string;
  reducedMotion: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-12 shrink-0 text-[10px] text-muted-foreground">{name}</span>
      <div
        className="h-3.5 flex-1 overflow-hidden rounded-sm border border-border"
        style={{ background: 'var(--muted)' }}
        aria-hidden
      >
        <div
          className="h-full"
          style={{
            width: `${frac * 100}%`,
            minWidth: frac > 0 ? '2px' : undefined,
            background: 'var(--primary)',
            transition: reducedMotion ? undefined : 'width 500ms ease',
          }}
        />
      </div>
      <span className="w-20 shrink-0 text-right font-mono text-[11px] text-foreground/90">{valueText}</span>
    </div>
  );
}
