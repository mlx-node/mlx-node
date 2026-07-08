import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * RooflineDiagram — chapter-4 deep-dive sibling of MemoryWallDiagram. Puts a
 * NUMBER on "decode is memory-bound" with the classic roofline chart.
 *
 * Self-contained: NO worker, NO model, NO WASM, NO network — pure SVG + React
 * state, so it server-renders for crawlers (createRoot replaces it on boot) and
 * is robust while the model downloads. The only interactive state is a batch-size
 * slider; there is no autoplay, so prefers-reduced-motion only disables the dot's
 * slide transition (the matchMedia-in-useState idiom is copied verbatim from the
 * sibling widgets — it is the deliberate convention here).
 *
 * THE PHYSICS (every number verified, not paraphrased):
 *   A GPU has two ceilings — compute (FLOPs/sec) and memory bandwidth (bytes/sec).
 *   Their ratio is the "ops:byte" balance point (the ridge). An H100 in FP16 does
 *   989 TFLOPS against 3.35 TB/s → ridge = 989e12 / 3.35e12 ≈ 295 ops:byte.
 *   (Both numbers match the FlashAttention-3 prose in the Attention chapter.)
 *
 *   An algorithm's "arithmetic intensity" = work (FLOPs) ÷ memory traffic (bytes).
 *   The roofline plots intensity (x, log) vs achievable performance (y, log): a
 *   DIAGONAL bandwidth ceiling perf = BW·intensity rises until it meets the flat
 *   COMPUTE ceiling at the ridge. LEFT of the ridge = memory-bound (you can't fill
 *   the FLOPs because you're starved on bytes); RIGHT = compute-bound.
 *
 *   - prefill processes the whole prompt in parallel — lots of matmul per weight
 *     read — so it sits RIGHT of the ridge, on the compute ceiling.
 *   - decode at batch 1 streams every weight to make ONE token → far-left memory
 *     cliff. As batch N rises, N tokens reuse each weight read, so intensity ≈ ×N
 *     and the dot MARCHES RIGHT up the diagonal; near batch ≈246–256 it reaches
 *     the ridge (295) and flips to compute-bound.
 *
 * Worked example in the caption (a real batch-1 decode step, NOT the N×N
 * attention op — see the memory-wall sub-chapter's caveat that single-token
 * decode never builds an N×N matrix): producing one token does ~2 FLOPs per
 * parameter (one multiply-add) but must read every parameter from HBM (2 bytes
 * in bf16), so intensity ≈ 2 FLOP / 2 bytes ≈ 1 op:byte ≪ 295 → memory-bound.
 * That ratio is ~model-size-independent, which is why every batch-1 LLM decode
 * is bandwidth-pinned. (This matches the far-left decode dot at I≈1.2 ops:byte;
 * batching N reuses each weight read, so intensity ≈ ×N marches it rightward.)
 *
 * HONESTY: the in-browser demo is pinned at batch 1 on the far-left cliff (one
 * user, batch size 1, plain autoregressive decode) — a faster GPU's extra FLOPs
 * can't help because you're bandwidth-pinned. Production batches hundreds of
 * users to climb the diagonal to the ridge. That "you are here · batch 1" marker
 * is locked on the far left.
 */

// ── Hardware constants (H100, FP16) — verified against the Attention chapter. ─
const PEAK_TFLOPS = 989; // H100 Tensor-Core FP16 peak
const BW_TBPS = 3.35; // H100 HBM3 bandwidth, TB/s
// perf(TFLOPS) = BW(B/s)·I(FLOP/B) / 1e12 = 3.35·I, and the ridge is PEAK/BW.
const RIDGE = PEAK_TFLOPS / BW_TBPS; // ≈ 295 ops:byte

// Decode arithmetic intensity at batch 1 (far-left memory cliff). Batch N
// reuses each weight read across N tokens, so intensity ≈ base·N.
const DECODE_BASE_I = 1.2; // ops:byte at batch 1
// Batch presets the slider snaps to (clean powers of two + 256).
const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256] as const;
// prefill: the whole prompt at once, lots of matmul per weight → compute-bound.
const PREFILL_I = 800; // ops:byte, comfortably right of the ridge

function decodeIntensity(batch: number): number {
  return DECODE_BASE_I * batch;
}
/** Achievable perf under the roofline: min(compute ceiling, bandwidth·I). */
function rooflinePerf(intensity: number): number {
  return Math.min(PEAK_TFLOPS, BW_TBPS * intensity);
}
function isMemoryBound(intensity: number): boolean {
  return intensity < RIDGE;
}

function fmtI(i: number): string {
  if (i >= 100) return Math.round(i).toString();
  if (i >= 10) return i.toFixed(0);
  return i.toFixed(1);
}

// ── Per-locale copy. COPY.en strings are the original English, verbatim. zh keeps
// the glossary terms English (roofline, ops:byte, FLOPs, H100, GPU, TFLOPS,
// prefill, decode, batch, KV cache, token, bandwidth) and uses full-width
// punctuation; only the surrounding prose translates. ────────────────────────
const COPY = {
  en: {
    heading: 'The roofline — putting a number on "memory-bound"',
    batchLabel: 'decode batch size',
    batchHint: 'tokens computed per one weight-read',
    xAxis: 'arithmetic intensity — ops:byte (FLOPs ÷ bytes moved), log scale',
    yAxis: 'TFLOPS',
    computeCeiling: 'compute ceiling · 989 TFLOPS',
    bwCeiling: 'bandwidth ceiling · 3.35 TB/s',
    ridgeLabel: 'ridge ≈ 295 ops:byte',
    memorySide: 'memory-bound',
    computeSide: 'compute-bound',
    prefillDot: 'prefill',
    decodeDot: (b: number) => `decode · batch ${b}`,
    youAreHere: 'you are here · batch 1',
    verdictMem: 'memory-bound',
    verdictCompute: 'compute-bound',
    intensityNow: 'intensity now',
    verdict: 'verdict',
    opsbyte: 'ops:byte',
    marchHint: 'Drag the slider: each token in the batch reuses the same weight read, so intensity climbs ≈ ×batch and the decode dot marches right toward the ridge.',
    flipHint: (b: number) =>
      `At batch ${b} the decode dot crosses the ridge — now there is finally enough math per byte to keep the GPU's FLOPs busy.`,
    ariaLabel:
      'Roofline chart: arithmetic intensity (log x-axis) versus achievable performance (log y-axis). A diagonal bandwidth ceiling rises until it meets a flat compute ceiling at the ridge near 295 ops:byte. The prefill point sits right of the ridge (compute-bound) and the decode point sits left of it (memory-bound), marching right toward the ridge as batch size grows.',
    caption: (
      <>
        Worked example — a <strong>batch-1 decode step</strong> does about{' '}
        <span className="font-mono">2 FLOPs</span> of math <em>per parameter</em> (one multiply-add) but must read{' '}
        <em>every</em> parameter from memory (<span className="font-mono">2 bytes</span> in bf16) just to emit one token:{' '}
        <span className="font-mono">~2 FLOP ÷ 2 bytes</span> ≈{' '}
        <span className="font-mono text-primary">1.2 op:byte</span> ≪ 295, pinned to the memory cliff. That ratio barely
        depends on model size — it&apos;s why <em>every</em> batch-1 LLM decode is memory-bound.
      </>
    ),
    honesty: (
      <>
        This in-browser demo is pinned at <strong>batch 1</strong> on the far-left memory cliff — one user, plain
        autoregressive decode. A faster GPU&apos;s extra FLOPs can&apos;t help you here, because you&apos;re
        bandwidth-pinned, not compute-pinned. Production serving batches <strong>hundreds of users</strong> together to
        climb the diagonal up to the ridge — that&apos;s how a GPU&apos;s compute actually gets used.
      </>
    ),
  },
  zh: {
    heading: 'roofline——给「memory-bound」一个数字',
    batchLabel: 'decode batch 大小',
    batchHint: '每读一次权重所计算的 token 数',
    xAxis: 'arithmetic intensity——ops:byte（FLOPs ÷ 搬运的字节数），对数刻度',
    yAxis: 'TFLOPS',
    computeCeiling: 'compute 上限 · 989 TFLOPS',
    bwCeiling: 'bandwidth 上限 · 3.35 TB/s',
    ridgeLabel: 'ridge ≈ 295 ops:byte',
    memorySide: 'memory-bound',
    computeSide: 'compute-bound',
    prefillDot: 'prefill',
    decodeDot: (b: number) => `decode · batch ${b}`,
    youAreHere: '你在这里 · batch 1',
    verdictMem: 'memory-bound',
    verdictCompute: 'compute-bound',
    intensityNow: '当前 intensity',
    verdict: '判定',
    opsbyte: 'ops:byte',
    marchHint: '拖动滑块：batch 里的每个 token 都复用同一次权重读取，所以 intensity 大约按 ×batch 攀升，decode 点向右朝 ridge 前进。',
    flipHint: (b: number) =>
      `在 batch ${b} 时 decode 点越过了 ridge——此时每字节终于有足够的算术量，能把 GPU 的 FLOPs 喂饱。`,
    ariaLabel:
      'roofline 图：arithmetic intensity（对数 x 轴）对比可达性能（对数 y 轴）。一条对角 bandwidth 上限不断上升，直到在约 295 ops:byte 的 ridge 处与水平的 compute 上限相交。prefill 点位于 ridge 右侧（compute-bound），decode 点位于其左侧（memory-bound），并随 batch 增大向 ridge 右移。',
    caption: (
      <>
        计算示例——一个 <strong>batch-1 decode step</strong> <em>每个参数</em>只做约{' '}
        <span className="font-mono">2 FLOPs</span> 的数学（一次乘加），却必须把<em>每一个</em>参数都从内存里读出来（bf16
        下 <span className="font-mono">2 字节</span>），只为吐出一个 token：<span className="font-mono">~2 FLOP ÷ 2 字节</span> ≈{' '}
        <span className="font-mono text-primary">1.2 op:byte</span> ≪ 295，被钉在 memory 悬崖上。这个比值几乎与模型大小无关——这正是为什么<em>每个</em>
        batch-1 的 LLM decode 都是 memory-bound。
      </>
    ),
    honesty: (
      <>
        这个浏览器内的 demo 被钉在 <strong>batch 1</strong>，停在最左侧的 memory 悬崖上——单个用户、普通自回归
        decode。在这里，更快 GPU 的额外 FLOPs 帮不上忙，因为你受 bandwidth 限制，而非 compute 限制。生产环境的服务会把{' '}
        <strong>数百个用户</strong> 一起批处理，沿对角线爬升到 ridge——GPU 的算力正是这样才被真正用起来。
      </>
    ),
  },
} as const;

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

// ── SVG geometry ─────────────────────────────────────────────────────────────
const VB_W = 720;
const VB_H = 360;
const ML = 52; // left margin (y-axis labels)
const MR = 118; // right margin (room for the "compute-bound" side label + dot labels)
const MT = 18; // top margin
const MB = 52; // bottom margin (x-axis labels)
const PX0 = ML;
const PX1 = VB_W - MR;
const PY0 = MT;
const PY1 = VB_H - MB;

// Log domains. x: intensity 0.5 … 4000 ops:byte. y: perf 4 … 1400 TFLOPS.
const X_MIN = 0.5;
const X_MAX = 4000;
const Y_MIN = 4;
const Y_MAX = 1400;
const LX_MIN = Math.log10(X_MIN);
const LX_MAX = Math.log10(X_MAX);
const LY_MIN = Math.log10(Y_MIN);
const LY_MAX = Math.log10(Y_MAX);

function xPix(intensity: number): number {
  const t = (Math.log10(intensity) - LX_MIN) / (LX_MAX - LX_MIN);
  return PX0 + t * (PX1 - PX0);
}
function yPix(perf: number): number {
  const t = (Math.log10(perf) - LY_MIN) / (LY_MAX - LY_MIN);
  return PY1 - t * (PY1 - PY0);
}

// Log gridlines (decade ticks) for each axis.
const X_TICKS = [1, 10, 100, 1000] as const;
const Y_TICKS = [10, 100, 1000] as const;

export function RooflineDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  // Slider indexes into BATCHES so it snaps to clean powers of two.
  const [batchIdx, setBatchIdx] = React.useState(0); // start at batch 1
  const batch = BATCHES[batchIdx];

  const decI = decodeIntensity(batch);
  const decP = rooflinePerf(decI);
  const memBound = isMemoryBound(decI);

  // Roofline polyline: rises along the bandwidth diagonal, then flat at the peak.
  const ridgeX = xPix(RIDGE);
  const ridgeY = yPix(PEAK_TFLOPS);
  // Start the bandwidth diagonal where it EXITS the bottom of the plot, not at
  // X_MIN: at very low intensity perf = BW·I drops under the y-axis floor (Y_MIN),
  // so anchoring at X_MIN drew the line below the frame. The floor crossing is at
  // I = Y_MIN / BW, which lands right under the batch-1 decode dot.
  const diagStartX = xPix(Y_MIN / BW_TBPS);
  const diagStartY = PY1; // bottom edge (== yPix(Y_MIN))
  const flatEndX = xPix(X_MAX);

  // Dot pixel positions.
  const decX = xPix(decI);
  const decY = yPix(decP);
  const preX = xPix(PREFILL_I);
  const preY = yPix(rooflinePerf(PREFILL_I));
  const lockedX = xPix(decodeIntensity(1)); // far-left batch-1 marker
  const lockedY = yPix(rooflinePerf(decodeIntensity(1)));

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      {/* The chart */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* plot frame */}
        <rect
          x={PX0}
          y={PY0}
          width={PX1 - PX0}
          height={PY1 - PY0}
          fill="none"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />

        {/* x gridlines + decade labels */}
        {X_TICKS.map((t) => {
          const x = xPix(t);
          return (
            <g key={`x${t}`}>
              <line x1={x} y1={PY0} x2={x} y2={PY1} style={{ stroke: 'var(--border)', opacity: 0.4 }} strokeWidth={1} />
              <text
                x={x}
                y={PY1 + 16}
                textAnchor="middle"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {t}
              </text>
            </g>
          );
        })}
        {/* y gridlines + decade labels */}
        {Y_TICKS.map((t) => {
          const y = yPix(t);
          return (
            <g key={`y${t}`}>
              <line x1={PX0} y1={y} x2={PX1} y2={y} style={{ stroke: 'var(--border)', opacity: 0.4 }} strokeWidth={1} />
              <text
                x={PX0 - 8}
                y={y + 3}
                textAnchor="end"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {t}
              </text>
            </g>
          );
        })}
        {/* y-axis unit — sits at the very top-left, raised clear of the topmost
            "1000" decade tick so their labels never touch. */}
        <text
          x={PX0 - 8}
          y={PY0 + 4}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.yAxis}
        </text>

        {/* memory-bound shading (left of ridge) */}
        <rect
          x={PX0}
          y={PY0}
          width={ridgeX - PX0}
          height={PY1 - PY0}
          style={{ fill: 'var(--primary)', opacity: 0.05 }}
        />

        {/* ridge vertical line */}
        <line
          x1={ridgeX}
          y1={ridgeY}
          x2={ridgeX}
          y2={PY1}
          style={{ stroke: 'var(--muted-foreground)', opacity: 0.5 }}
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        {/* ridge label — top, right-aligned to JUST LEFT of the ridge so it never
            runs into the compute-ceiling label that lives at the top-right. */}
        <text
          x={ridgeX - 8}
          y={PY0 + 12}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.ridgeLabel}
        </text>

        {/* The roofline itself: bandwidth diagonal → compute ceiling. */}
        <polyline
          points={`${diagStartX},${diagStartY} ${ridgeX},${ridgeY} ${flatEndX},${ridgeY}`}
          fill="none"
          style={{ stroke: 'var(--foreground)' }}
          strokeWidth={2}
        />
        {/* compute-ceiling label — top-RIGHT, right-aligned to the frame and sitting
            just above the flat ceiling, so it clears both the ridge label (top-left
            of the ridge) and the prefill dot's label (now below its dot). */}
        <text
          x={PX1}
          y={PY0 + 12}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.computeCeiling}
        </text>
        {/* bandwidth-ceiling label, angled along the diagonal's lower stretch */}
        <text
          x={xPix(3)}
          y={yPix(rooflinePerf(3)) + 16}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
          transform={`rotate(-31 ${xPix(3)} ${yPix(rooflinePerf(3)) + 16})`}
        >
          {copy.bwCeiling}
        </text>

        {/* memory / compute side labels just under the top frame */}
        <text
          x={(PX0 + ridgeX) / 2}
          y={PY1 - 8}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)', opacity: 0.85 }}
          className="text-[11px] font-medium"
        >
          ← {copy.memorySide}
        </text>
        <text
          x={(ridgeX + PX1) / 2}
          y={PY1 - 8}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)', opacity: 0.85 }}
          className="text-[11px] font-medium"
        >
          {copy.computeSide} →
        </text>

        {/* locked "you are here · batch 1" marker (far left, hollow). */}
        {batch !== 1 ? (
          <g>
            <circle
              cx={lockedX}
              cy={lockedY}
              r={5}
              fill="var(--background)"
              style={{ stroke: 'var(--muted-foreground)' }}
              strokeWidth={1.5}
            />
            <text
              x={lockedX + 9}
              y={lockedY + 3}
              style={{ fill: 'var(--muted-foreground)' }}
              className="text-[9px]"
            >
              {copy.youAreHere}
            </text>
          </g>
        ) : null}

        {/* prefill dot — compute-bound, on the flat ceiling right of the ridge. Its
            label sits BELOW the dot (into the empty compute-bound interior) so it
            stays clear of the compute-ceiling label on the top row. */}
        <g>
          <circle cx={preX} cy={preY} r={6} fill="var(--foreground)" />
          <text
            x={preX}
            y={preY + 18}
            textAnchor="middle"
            style={{ fill: 'var(--foreground)' }}
            className="text-[11px] font-semibold"
          >
            {copy.prefillDot}
          </text>
        </g>

        {/* decode dot — marches right with batch size. */}
        <g
          style={{
            transition: reducedMotion ? undefined : 'transform 350ms cubic-bezier(0.4, 0, 0.2, 1)',
            transform: `translate(${decX - lockedX}px, ${decY - lockedY}px)`,
          }}
        >
          {/* drawn at the locked origin; the <g> translates it to the live point */}
          <circle
            cx={lockedX}
            cy={lockedY}
            r={7}
            style={{ fill: memBound ? 'var(--primary)' : 'var(--foreground)' }}
          />
          {/* Label above the dot while it climbs the diagonal (lots of empty room
              there). Once the dot reaches the ridge/ceiling at max batch, flip the
              label BELOW-left so it clears the top row and the nearby prefill dot. */}
          <text
            x={memBound ? lockedX : lockedX - 2}
            y={memBound ? lockedY - 13 : lockedY + 20}
            textAnchor={memBound ? 'middle' : 'end'}
            style={{ fill: memBound ? 'var(--primary)' : 'var(--foreground)' }}
            className="text-[11px] font-semibold"
          >
            {copy.decodeDot(batch)}
          </text>
        </g>

        {/* x-axis caption centered under the plot */}
        <text
          x={(PX0 + PX1) / 2}
          y={PY1 + 38}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.xAxis}
        </text>
      </svg>

      {/* Batch-size slider + live readout */}
      <div className="flex flex-col gap-2 border-t border-border/60 pt-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex-1 space-y-1">
          <label htmlFor="roofline-batch" className="block text-[11px] text-muted-foreground">
            {copy.batchLabel}:{' '}
            <span className="font-mono text-foreground/90">{batch}</span>{' '}
            <span className="text-muted-foreground/70">({copy.batchHint})</span>
          </label>
          <input
            id="roofline-batch"
            type="range"
            min={0}
            max={BATCHES.length - 1}
            step={1}
            value={batchIdx}
            onChange={(e) => setBatchIdx(Number(e.target.value))}
            aria-valuetext={`batch ${batch}`}
            className="w-full accent-[var(--primary)]"
          />
        </div>

        {/* live intensity + verdict chips */}
        <div className="flex items-center gap-3 sm:flex-col sm:items-end sm:gap-1">
          <div className="text-[11px] text-muted-foreground">
            {copy.intensityNow}:{' '}
            <span className="font-mono text-primary">{fmtI(decI)}</span>{' '}
            <span className="text-muted-foreground/70">{copy.opsbyte}</span>
          </div>
          <div className="text-[11px] text-muted-foreground">
            {copy.verdict}:{' '}
            <span
              className={[
                'rounded px-1.5 py-0.5 text-[11px] font-medium',
                memBound
                  ? 'bg-primary/15 text-primary'
                  : 'bg-muted text-foreground/80',
              ].join(' ')}
            >
              {memBound ? copy.verdictMem : copy.verdictCompute}
            </span>
          </div>
        </div>
      </div>

      {/* Marching hint flips to the crossover message once compute-bound. */}
      <p className="min-h-[2.5rem] text-[13px] text-foreground/90">
        {memBound ? copy.marchHint : copy.flipHint(batch)}
      </p>

      {/* Worked-example caption */}
      <p className="text-[12px] text-muted-foreground">{copy.caption}</p>

      {/* Honesty framing */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.honesty}</p>
    </div>
  );
}
