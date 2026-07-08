import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Flash3AsyncDiagram — chapter 4 deep-dive "FlashAttention-3 on Hopper".
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It teaches the Hopper-specific trick of FlashAttention-3 (arXiv 2407.08608,
 * Shah/Bikshandi/Dao et al., 2024): ASYNCHRONY. Attention alternates two kinds
 * of work — big matmuls that run on the "Tensor Cores" (the GPU's very fast
 * matrix-multiply units) and the softmax (exp/scale), which runs on slower
 * general-purpose units. FlashAttention-2 runs them one after another, so the
 * Tensor Cores sit IDLE while softmax catches up. FlashAttention-3 overlaps
 * them ("pingpong" / warp-specialized scheduling — "asynchronous" just means two
 * kinds of work running at the same time on different warps, a warp being one
 * lockstep group of GPU threads): while the Tensor Cores chew on tile j+1, the
 * exp units do the softmax for tile j. The Tensor Cores stay busy and the whole
 * schedule finishes sooner. A second lever — FP8, an 8-bit number format with
 * ~2× the throughput of 16-bit but less precision — shrinks the matmul blocks
 * further (with block-quantization to stay accurate).
 *
 * Animation model: a Gantt-style pipeline. Toggle FA2 (serial, idle gaps) vs
 * FA3 (overlapped, short), and FP16 vs FP8 (relabel + shrink matmul blocks). An
 * optional Play sweeps a "now" cursor across the schedule. Honors
 * prefers-reduced-motion (renders FA3's overlapped schedule statically, no
 * cursor sweep).
 */

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

type Schedule = 'fa2' | 'fa3';
type Precision = 'fp16' | 'fp8';

// ── Pipeline model (in time-units, mapped to SVG x later) ────────────────────
// Four tiles of work. Each tile needs a matmul (Tensor Cores) then a softmax
// (exp units). FP8 halves the matmul cost (~2× throughput). FA2 runs strictly
// serial; FA3 overlaps softmax of tile j with matmul of tile j+1.
const TILES = 4;
const MM_FP16 = 4; // matmul duration in time-units at FP16
const MM_FP8 = 2; // matmul duration at FP8 (~2× throughput)
const SOFT = 3; // softmax (exp/scale) duration — same on both precisions

type Block = {
  lane: 'mm' | 'soft';
  tile: number;
  start: number;
  dur: number;
};

/** Build the schedule for a given mode + precision; returns blocks + makespan. */
function buildSchedule(schedule: Schedule, precision: Precision): { blocks: Block[]; total: number } {
  const mm = precision === 'fp8' ? MM_FP8 : MM_FP16;
  const blocks: Block[] = [];

  if (schedule === 'fa2') {
    // Serial: matmul → softmax → matmul → softmax … one tile fully finishes
    // before the next starts. The Tensor-Core lane is idle during each softmax.
    let t = 0;
    for (let i = 0; i < TILES; i++) {
      blocks.push({ lane: 'mm', tile: i, start: t, dur: mm });
      t += mm;
      blocks.push({ lane: 'soft', tile: i, start: t, dur: SOFT });
      t += SOFT;
    }
    return { blocks, total: t };
  }

  // FA3 pingpong: the Tensor-Core lane runs matmuls back-to-back with no gaps;
  // each tile's softmax starts as soon as that tile's matmul finishes and runs
  // concurrently on the exp units (overlapping the NEXT tile's matmul).
  let mmCursor = 0;
  let softEnd = 0;
  for (let i = 0; i < TILES; i++) {
    const mmStart = mmCursor;
    blocks.push({ lane: 'mm', tile: i, start: mmStart, dur: mm });
    mmCursor += mm;
    // Softmax can't start before its matmul ends, nor before the exp unit frees.
    const softStart = Math.max(mmCursor, softEnd);
    blocks.push({ lane: 'soft', tile: i, start: softStart, dur: SOFT });
    softEnd = softStart + SOFT;
  }
  return { blocks, total: Math.max(mmCursor, softEnd) };
}

// ── SVG geometry ─────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 268;
const PLOT_X = 150; // left edge of the lanes (room for lane labels)
const PLOT_W = 588; // drawing width for the time axis
const UNIT = 26; // px per time-unit (max FA2 FP16 makespan = 28 units → fits)
const MM_Y = 54;
const SOFT_Y = 110;
const LANE_H = 36;
const TOTALBAR_Y = 186;
const TOTALBAR_H = 22;

const FA2_FP16_TOTAL = buildSchedule('fa2', 'fp16').total; // longest schedule = full-width reference

function xOf(t: number): number {
  return PLOT_X + t * UNIT * (PLOT_W / (FA2_FP16_TOTAL * UNIT));
}
function wOf(dur: number): number {
  return dur * UNIT * (PLOT_W / (FA2_FP16_TOTAL * UNIT));
}

// Per-locale copy. English strings are moved verbatim from the original JSX.
// Code-ish labels (mm1/soft1/·FP8, Tensor Cores, Exp / Softmax, FP16/FP8, the
// width-constrained "× vs FA2" readout) stay English in both locales.
const COPY = {
  en: {
    heading: 'FlashAttention-3 — overlap the math so the Tensor Cores barely idle',
    fa2Toggle: 'FA2 · serial',
    fa3Toggle: 'FA3 · overlap',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    svgAria:
      'Gantt-style timeline of FlashAttention. The Tensor-Core matmul lane and the exp/softmax lane run one after another in FlashAttention-2, leaving the Tensor Cores idle, but overlap concurrently in FlashAttention-3 so the schedule finishes sooner.',
    mmLaneSub: 'matmul · very fast',
    softLaneSub: 'scale · slower units',
    idle: 'idle',
    makespanTitle: (total: number) => `makespan ${total} time-units`,
    totalTime: 'total time',
    timeAxis: 'time →',
    captionFa3: (idlePct: number) => (
      <>
        <strong>FlashAttention-3</strong> overlaps the two kinds of work: while the Tensor Cores grind tile{' '}
        <span className="font-mono">j+1</span>&apos;s matmul, the exp units already run tile{' '}
        <span className="font-mono">j</span>&apos;s softmax.{' '}
        {idlePct <= 25 ? (
          <>
            The Tensor-Core lane stays nearly solid — only{' '}
            <span className="font-mono text-primary">{idlePct}% idle</span> — so the schedule finishes sooner.
          </>
        ) : (
          <>
            The Tensor-Core lane is <span className="font-mono text-primary">{idlePct}% idle</span> — FP8 halves
            the matmuls, so the slower softmax becomes the bottleneck — but the schedule still finishes sooner.
          </>
        )}{' '}
        (Each &ldquo;matmul&rdquo; bar is simplified: a real tile runs two GEMMs,{' '}
        <span className="font-mono">QKᵀ</span> then <span className="font-mono">P·V</span>, with the softmax
        between.)
      </>
    ),
    captionFa2: (idlePct: number) => (
      <>
        <strong>FlashAttention-2</strong> runs strictly serial: each tile&apos;s matmul, then its softmax, then the next.
        The Tensor Cores sit <span className="font-mono text-primary">{idlePct}% idle</span> (the dashed{' '}
        <span className="font-mono">idle</span> gaps) waiting on the slower softmax — wasted time on the fastest unit.
      </>
    ),
    precisionLabel: 'matmul precision',
    scheduleLength: 'schedule length:',
    timeUnits: 'time-units',
    fp8Note: 'FP8 (8-bit) matmuls run ~2× faster — but need block-quantization to stay accurate.',
    fp16Note: 'FP16 (16-bit) matmuls — switch to FP8 to shrink the blue blocks ~2×.',
    footer: (
      <>
        The speedup comes from two levers: keeping the Tensor Cores busy by overlapping matmul with softmax (warp-specialized
        &ldquo;pingpong&rdquo; scheduling), and cheaper FP8 math. Block sizes here are illustrative, not measured.
      </>
    ),
  },
  zh: {
    heading: 'FlashAttention-3——重叠两种计算，让 Tensor Core 几乎不闲置',
    fa2Toggle: 'FA2 · 串行',
    fa3Toggle: 'FA3 · 重叠',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    svgAria:
      '甘特图式的 FlashAttention 时间线。在 FlashAttention-2 中，Tensor Core 矩阵乘法通道与 exp/softmax 通道一前一后轮流执行，让 Tensor Core 闲置；在 FlashAttention-3 中两者并发重叠，整个调度更早完工。',
    mmLaneSub: 'matmul · 极快',
    softLaneSub: '缩放 · 更慢的单元',
    idle: '闲置',
    makespanTitle: (total: number) => `总工期 ${total} 个时间单位`,
    totalTime: '总耗时',
    timeAxis: '时间 →',
    captionFa3: (idlePct: number) => (
      <>
        <strong>FlashAttention-3</strong> 把两种工作重叠起来：当 Tensor Core 还在啃第{' '}
        <span className="font-mono">j+1</span> 块的矩阵乘法时，指数单元已经在跑第{' '}
        <span className="font-mono">j</span> 块的 softmax。{' '}
        {idlePct <= 25 ? (
          <>
            Tensor Core 通道几乎排满——只有{' '}
            <span className="font-mono text-primary">{idlePct}% 闲置</span>——整个调度更早完工。
          </>
        ) : (
          <>
            Tensor Core 通道有 <span className="font-mono text-primary">{idlePct}% 闲置</span>——FP8
            把矩阵乘法减半，较慢的 softmax 成了瓶颈——但调度仍然更早完工。
          </>
        )}
        （每条“matmul”柱是简化的：真实的一块要跑两次 GEMM，先 <span className="font-mono">QKᵀ</span> 再{' '}
        <span className="font-mono">P·V</span>，softmax 夹在中间。）
      </>
    ),
    captionFa2: (idlePct: number) => (
      <>
        <strong>FlashAttention-2</strong> 严格串行：每块先矩阵乘法，再 softmax，然后下一块。Tensor Core 干坐着{' '}
        <span className="font-mono text-primary">{idlePct}% 闲置</span>（那些虚线的{' '}
        <span className="font-mono">闲置</span>空档），干等较慢的 softmax——在最快的单元上浪费时间。
      </>
    ),
    precisionLabel: 'matmul 精度',
    scheduleLength: '调度长度：',
    timeUnits: '时间单位',
    fp8Note: 'FP8（8 位）矩阵乘法约快 2×——但需要块量化才能保住精度。',
    fp16Note: 'FP16（16 位）矩阵乘法——切到 FP8 可把蓝色块缩小约 2×。',
    footer: (
      <>
        加速来自两根杠杆：用 matmul 与 softmax 的重叠让 Tensor Core
        保持忙碌（warp 分工的“乒乓（pingpong）”调度），以及更便宜的 FP8 数学。这里的块尺寸只是示意，并非实测。
      </>
    ),
  },
} as const;

export function Flash3AsyncDiagram() {
  const reducedMotion = usePrefersReducedMotion();
  const copy = COPY[useLocale()];
  const [schedule, setSchedule] = React.useState<Schedule>('fa3');
  const [precision, setPrecision] = React.useState<Precision>('fp16');
  const [playing, setPlaying] = React.useState<boolean>(!reducedMotion);
  const [now, setNow] = React.useState<number>(0); // cursor position in time-units

  const { blocks, total } = React.useMemo(() => buildSchedule(schedule, precision), [schedule, precision]);

  // Sweep the "now" cursor across the current schedule.
  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const id = window.setInterval(() => {
      setNow((x) => {
        const next = x + 0.4;
        return next > total + 1 ? 0 : next;
      });
    }, 60);
    return () => window.clearInterval(id);
  }, [playing, reducedMotion, total]);

  // Reset the cursor whenever the schedule changes so it reads cleanly.
  React.useEffect(() => {
    setNow(0);
  }, [schedule, precision]);

  const step = () => {
    setPlaying(false);
    setNow((x) => (x + 2 > total + 1 ? 0 : x + 2));
  };

  const cursorVisible = !reducedMotion && playing;
  const isFa3 = schedule === 'fa3';

  // Idle fraction on the Tensor-Core lane = (makespan − busy matmul time) / makespan.
  const mmBusy = blocks.filter((b) => b.lane === 'mm').reduce((s, b) => s + b.dur, 0);
  const idlePct = Math.round(((total - mmBusy) / total) * 100);
  // Speedup vs the FA2 schedule at the SAME precision.
  const fa2Total = buildSchedule('fa2', precision).total;
  const speedup = (fa2Total / total).toFixed(2);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + schedule toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.heading}
        </div>
        <div className="flex items-center gap-2">
          <div className="inline-flex overflow-hidden rounded border border-border">
            {(['fa2', 'fa3'] as Schedule[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSchedule(s)}
                aria-pressed={s === schedule}
                className={[
                  'px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:ring-2 focus-visible:ring-primary',
                  s === schedule ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {s === 'fa2' ? copy.fa2Toggle : copy.fa3Toggle}
              </button>
            ))}
          </div>
          {!reducedMotion ? (
            <>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {/* The Gantt-style pipeline */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria}
      >
        {/* ── Lane labels ── */}
        <text x={16} y={MM_Y + 16} style={{ fill: 'var(--foreground)' }} className="text-[12px] font-semibold">
          Tensor Cores
        </text>
        <text x={16} y={MM_Y + 31} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.mmLaneSub}
        </text>
        <text x={16} y={SOFT_Y + 16} style={{ fill: 'var(--foreground)' }} className="text-[12px] font-semibold">
          Exp / Softmax
        </text>
        <text x={16} y={SOFT_Y + 31} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.softLaneSub}
        </text>

        {/* ── Lane backgrounds (the full time window = FA2 FP16 makespan) ── */}
        <rect
          x={PLOT_X}
          y={MM_Y}
          width={PLOT_W}
          height={LANE_H}
          rx={6}
          style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <rect
          x={PLOT_X}
          y={SOFT_Y}
          width={PLOT_W}
          height={LANE_H}
          rx={6}
          style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
          strokeWidth={1}
        />

        {/* ── In FA2, mark the Tensor-Core idle gaps explicitly ── */}
        {!isFa3
          ? blocks
              .filter((b) => b.lane === 'soft')
              .map((b) => (
                <g key={`idle-${b.tile}`}>
                  <rect
                    x={xOf(b.start)}
                    y={MM_Y + 4}
                    width={wOf(b.dur)}
                    height={LANE_H - 8}
                    rx={4}
                    style={{ fill: 'var(--border)', fillOpacity: 0.25, stroke: 'var(--border)' }}
                    strokeWidth={1}
                    strokeDasharray="4 3"
                  />
                  <text
                    x={xOf(b.start) + wOf(b.dur) / 2}
                    y={MM_Y + LANE_H / 2 + 4}
                    textAnchor="middle"
                    style={{ fill: 'var(--muted-foreground)' }}
                    className="text-[9px] font-medium"
                  >
                    {copy.idle}
                  </text>
                </g>
              ))
          : null}

        {/* ── Work blocks ── */}
        {blocks.map((b) => {
          const y = b.lane === 'mm' ? MM_Y : SOFT_Y;
          const isMm = b.lane === 'mm';
          // Only room for the "·FP8" suffix when the bar is wide enough (FP8 bars
          // are half-width); the precision is also shown by the toggle + footer.
          const fp8Suffix = precision === 'fp8' && wOf(b.dur) >= 70 ? ' ·FP8' : '';
          const label = isMm ? `mm${b.tile + 1}${fp8Suffix}` : `soft${b.tile + 1}`;
          return (
            <g key={`${b.lane}-${b.tile}`}>
              <rect
                x={xOf(b.start) + 2}
                y={y + 4}
                width={Math.max(wOf(b.dur) - 4, 2)}
                height={LANE_H - 8}
                rx={4}
                style={{
                  fill: isMm ? 'var(--primary)' : 'var(--card)',
                  fillOpacity: isMm ? 0.85 : 1,
                  stroke: isMm ? 'var(--primary)' : 'var(--border)',
                }}
                strokeWidth={1.5}
              />
              <text
                x={xOf(b.start) + Math.max(wOf(b.dur) - 4, 2) / 2 + 2}
                y={y + LANE_H / 2 + 4}
                textAnchor="middle"
                style={{ fill: isMm ? 'var(--bg)' : 'var(--foreground)' }}
                className="text-[9px] font-semibold"
              >
                {label}
              </text>
            </g>
          );
        })}

        {/* ── "now" cursor sweeping across the schedule ── */}
        {cursorVisible ? (
          <line
            x1={xOf(Math.min(now, total))}
            y1={MM_Y - 8}
            x2={xOf(Math.min(now, total))}
            y2={SOFT_Y + LANE_H + 8}
            style={{ stroke: 'var(--primary)' }}
            strokeWidth={2}
            strokeDasharray="3 3"
          />
        ) : null}

        {/* ── Total-time comparison bar (this schedule vs the FA2 FP16 reference) ── */}
        <text x={16} y={TOTALBAR_Y + 15} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.totalTime}
        </text>
        {/* faint full-width reference = FA2 FP16 makespan */}
        <rect
          x={PLOT_X}
          y={TOTALBAR_Y}
          width={PLOT_W}
          height={TOTALBAR_H}
          rx={5}
          fill="none"
          style={{ stroke: 'var(--border)', opacity: 0.7 }}
          strokeWidth={1.5}
          strokeDasharray="5 4"
        />
        <rect
          x={PLOT_X}
          y={TOTALBAR_Y}
          width={wOf(total)}
          height={TOTALBAR_H}
          rx={5}
          style={{ fill: 'var(--primary)', fillOpacity: 0.16, stroke: 'var(--primary)' }}
          strokeWidth={1.5}
        >
          <title>{copy.makespanTitle(total)}</title>
        </rect>
        {/* Place the readout after the bar when there's room; otherwise tuck it
            inside the (near-full-width) bar so it never clips past the viewBox. */}
        {(() => {
          const barEnd = PLOT_X + wOf(total);
          const labelOutside = barEnd + 78 <= VB_W;
          return (
            <text
              x={labelOutside ? barEnd + 10 : barEnd - 8}
              y={TOTALBAR_Y + TOTALBAR_H / 2 + 4}
              textAnchor={labelOutside ? 'start' : 'end'}
              style={{ fill: 'var(--primary)' }}
              className="text-[11px] font-semibold"
            >
              {speedup}× vs FA2
            </text>
          );
        })()}

        {/* time axis hint */}
        <text x={PLOT_X} y={TOTALBAR_Y + TOTALBAR_H + 16} style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
          {copy.timeAxis}
        </text>
      </svg>

      {/* Caption — explains the active schedule */}
      <p className="min-h-[3rem] text-[13px] text-foreground/90">
        {isFa3 ? copy.captionFa3(idlePct) : copy.captionFa2(idlePct)}
      </p>

      {/* Precision sub-toggle + grounding */}
      <div className="flex flex-wrap items-end justify-between gap-3 border-t border-border/60 pt-3">
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{copy.precisionLabel}</div>
          <div className="inline-flex overflow-hidden rounded border border-border">
            {(['fp16', 'fp8'] as Precision[]).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setPrecision(p)}
                aria-pressed={p === precision}
                className={[
                  'px-2.5 py-1 font-mono text-[11px] transition-colors focus-visible:ring-2 focus-visible:ring-primary',
                  p === precision ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {p === 'fp16' ? 'FP16' : 'FP8'}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1 text-right">
          <div className="text-[11px] text-muted-foreground">
            {copy.scheduleLength}{' '}
            <span className="font-mono text-primary">{total}</span>{' '}
            <span className="text-muted-foreground/80">{copy.timeUnits}</span>
          </div>
          <div className="text-[10px] text-muted-foreground/80">
            {precision === 'fp8' ? copy.fp8Note : copy.fp16Note}
          </div>
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground/70">
        {copy.footer}
      </p>
    </div>
  );
}
