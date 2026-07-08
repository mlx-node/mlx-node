import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * InferenceMetricsDiagram — the two clocks of streamed inference speed, and why
 * the average latency lies.
 *
 * Self-contained: NO worker, NO model, NO WASM, NO network — pure SVG + React
 * state, so it server-renders for crawlers (createRoot replaces it on boot) and
 * is robust while the model downloads. All numbers are ILLUSTRATIVE round figures
 * chosen to teach the shapes, not measurements of this model.
 *
 * Two panels, side by side (stacking on narrow screens):
 *
 *  (1) STREAMED-OUTPUT TIMELINE. When output is streamed, two metrics matter:
 *      - TTFT (time to first token): the pause before the first word, set by the
 *        COMPUTE-BOUND prefill (one big parallel matmul over the whole prompt).
 *      - TPOT / ITL (inter-token latency): the streaming speed after that, set by
 *        the MEMORY-BOUND decode. An ITL of 10 ms = 100 tokens/sec for one user.
 *      The panel draws a TTFT bar that fills during prefill, then evenly spaced
 *      TPOT token ticks. An optional Play animates the ticks; reduced motion
 *      renders the static final frame and hides Play.
 *
 *  (2) RIGHT-SKEWED LATENCY HISTOGRAM. Inference latency is right-skewed: most
 *      requests cluster near the mode, with a long slow tail. So the MEAN sits to
 *      the RIGHT of the MEDIAN — the average is dragged up by the tail and hides
 *      it. Engineers report percentiles instead: P50 (median, 1 in 2 slower),
 *      P90 (1 in 10), P95 (1 in 20), P99 (1 in 100). Reliability work targets
 *      P90/P99. Vertical markers sit on the distribution.
 *
 * HONESTY (caption): the in-browser demo measures the RAWEST TTFT/TPOT — no
 * network, no queue, batch 1, a single user. Production reports p95/p99
 * END-TO-END (+network +queue) because under load the slow tail is what users
 * remember. This demo is the floor; production lives in the tail. "When inference
 * is fast but end-to-end is slow, fix infrastructure, not the model."
 *
 * Honors prefers-reduced-motion: no auto-play, renders the final (all-ticks)
 * frame of the timeline, and the Play button is hidden.
 */

// ── Illustrative numbers (round, teaching shapes — not measured) ──────────────
const TTFT_MS = 220; // pause before the first token (compute-bound prefill)
const ITL_MS = 10; // inter-token latency (memory-bound decode); 10 ms = 100 tok/s
const TPS = Math.round(1000 / ITL_MS); // tokens per second for one user
const TICKS = 10; // how many streamed tokens we draw after the first

const EMERALD = '#10b981'; // emerald-500 — the "fast" accent (decode stream)

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

// ── Per-locale copy. zh keeps glossary terms in English; full-width punctuation. ─
const COPY = {
  en: {
    heading: 'The two clocks of inference speed',
    // Panel 1 — streamed timeline
    timelineTitle: 'Streamed output — two metrics',
    play: '▶ Play',
    replay: '↻ Replay',
    pause: '❚❚ Pause',
    ttftLabel: 'TTFT',
    ttftSub: 'time to first token · compute-bound prefill',
    streamLabel: 'then stream',
    itlLabel: 'ITL / TPOT',
    itlSub: 'inter-token latency · memory-bound decode',
    firstTokenTag: 'first token',
    tpsChip: (itl: number, tps: number) => (
      <>
        ITL <span className="font-mono text-foreground">{itl} ms</span> ={' '}
        <span className="font-mono text-foreground">{tps} tokens/sec</span> per user
      </>
    ),
    timelineCaption: (
      <>
        Two clocks: <strong>TTFT</strong> is the pause before the first word (the compute-bound prefill); after that the
        words <strong>stream</strong> at a steady <strong>ITL / TPOT</strong> (the memory-bound decode). TPS is
        ambiguous — &ldquo;perceived TPS&rdquo; is a per-user feel, &ldquo;total TPS&rdquo; is the whole service&apos;s
        throughput. For a non-streamed call (an agent tool call) you measure total response time instead.
      </>
    ),
    timelineAria:
      'Streamed-output timeline: a TTFT bar fills during the compute-bound prefill, then evenly spaced token ticks stream at a steady inter-token latency set by the memory-bound decode.',
    // Panel 2 — latency histogram
    histTitle: 'Latency is right-skewed — the average hides the tail',
    medianLabel: 'P50',
    medianTag: 'median',
    meanLabel: 'mean',
    p90: 'P90',
    p95: 'P95',
    p99: 'P99',
    tailLabel: 'long slow tail →',
    histCaption: (
      <>
        Latency is <strong>right-skewed</strong>: most requests cluster near the mode, with a long slow tail. So the{' '}
        <strong>mean</strong> sits to the <em>right</em> of the <strong>median</strong> — the average is dragged up by
        the tail and hides it. Engineers report <strong>percentiles</strong>: P50 (1 in 2 slower), P90 (1 in 10), P95 (1
        in 20), P99 (1 in 100). Reliability work targets <strong>P90 / P99</strong>.
      </>
    ),
    histAria:
      'A right-skewed latency histogram: a tall cluster near the mode with a long slow tail to the right. Vertical markers show P50 (median), P90, P95 and P99; the mean marker sits to the right of the median, dragged up by the tail.',
    percentileLegend: 'P50 = 1 in 2 slower · P90 = 1 in 10 · P95 = 1 in 20 · P99 = 1 in 100',
    // Honesty
    honestyTitle: 'inference-time vs end-to-end',
    honesty: (
      <>
        This in-browser demo measures the <strong>rawest</strong> TTFT and TPOT — no network, no queue, batch 1, one
        user. Production reports <strong>p95 / p99 end-to-end</strong> (on-GPU time <em>plus</em> network <em>plus</em>{' '}
        queue), because under load the slow tail is what users remember. When inference is fast but end-to-end is slow,
        fix the infrastructure, not the model. <strong>Your demo is the floor; production lives in the tail.</strong>
      </>
    ),
    illustrative: 'Illustrative round numbers — teaching the shapes, not this model’s measured latency.',
  },
  zh: {
    heading: '推理速度的两个时钟',
    timelineTitle: '流式输出——两个指标',
    play: '▶ 播放',
    replay: '↻ 重放',
    pause: '❚❚ 暂停',
    ttftLabel: 'TTFT',
    ttftSub: 'time to first token · 受算力限制的 prefill',
    streamLabel: '随后流式吐出',
    itlLabel: 'ITL / TPOT',
    itlSub: 'inter-token latency · 受显存带宽限制的 decode',
    firstTokenTag: '首个 token',
    tpsChip: (itl: number, tps: number) => (
      <>
        ITL <span className="font-mono text-foreground">{itl} ms</span> ={' '}
        <span className="font-mono text-foreground">{tps} tokens/sec</span> / 用户
      </>
    ),
    timelineCaption: (
      <>
        两个时钟：<strong>TTFT</strong> 是第一个词之前的停顿（受算力限制的 prefill）；此后这些词以稳定的{' '}
        <strong>ITL / TPOT</strong> <strong>流式</strong>吐出（受显存带宽限制的 decode）。TPS 是有歧义的——
        &ldquo;perceived TPS&rdquo; 是单个用户的体感，&ldquo;total TPS&rdquo; 是整个服务的 throughput。
        对于非流式调用（比如一次 agent 工具调用），你衡量的则是总响应时间。
      </>
    ),
    timelineAria:
      '流式输出时间线：TTFT 柱在受算力限制的 prefill 阶段填充，随后等间距的 token 刻度以受显存带宽限制的 decode 决定的稳定 inter-token latency 流式吐出。',
    histTitle: 'latency 右偏——平均值掩盖了长尾',
    medianLabel: 'P50',
    medianTag: '中位数',
    meanLabel: 'mean',
    p90: 'P90',
    p95: 'P95',
    p99: 'P99',
    tailLabel: '长长的慢尾 →',
    histCaption: (
      <>
        latency 是<strong>右偏</strong>的：大多数请求聚集在众数附近，拖着一条长长的慢尾。于是{' '}
        <strong>mean</strong> 落在 <strong>median</strong> 的<em>右侧</em>——平均值被长尾拉高，反而把它掩盖了。
        工程师改用<strong>百分位</strong>来报告：P50（1/2 更慢）、P90（1/10）、P95（1/20）、P99（1/100）。
        可靠性工作盯的是 <strong>P90 / P99</strong>。
      </>
    ),
    histAria:
      '右偏的 latency 直方图：众数附近有一个高耸的聚集，向右拖出一条长长的慢尾。竖直标记线标出 P50（中位数）、P90、P95 和 P99；mean 标记落在 median 右侧，被长尾拉高。',
    percentileLegend: 'P50 = 1/2 更慢 · P90 = 1/10 · P95 = 1/20 · P99 = 1/100',
    honestyTitle: 'inference-time vs end-to-end',
    honesty: (
      <>
        这个浏览器内的 demo 衡量的是<strong>最原始</strong>的 TTFT 与 TPOT——没有网络、没有排队、batch 为 1、单个用户。
        生产环境报告的是 <strong>p95 / p99 end-to-end</strong>（GPU 上的时间<em>加上</em>网络<em>加上</em>排队），
        因为在高负载下，用户记住的正是那条慢尾。当 inference 很快但 end-to-end 很慢时，要修的是基础设施，而不是模型。{' '}
        <strong>你的 demo 是下限；生产环境活在长尾里。</strong>
      </>
    ),
    illustrative: '示意性的整数——讲的是形状，并非本模型实测的 latency。',
  },
} as const;

// ── Panel 1 geometry: streamed-output timeline ────────────────────────────────
const T_VB_W = 360;
const T_VB_H = 150;
const T_AXIS_Y = 86; // baseline of the timeline
const T_AXIS_X0 = 18; // left edge of the timeline track
const T_AXIS_X1 = 344; // right edge
const T_TTFT_W = 96; // pixel width of the TTFT bar (illustrative, not to scale with ITL)
const T_TICK_X0 = T_AXIS_X0 + T_TTFT_W; // first token lands at the end of the TTFT bar
const T_TICK_GAP = (T_AXIS_X1 - T_TICK_X0) / TICKS;

/** Panel 1 — the streamed-output timeline. `visible` ticks are drawn (animation). */
function TimelinePanel({ visible }: { visible: number }) {
  const copy = COPY[useLocale()];
  return (
    <svg viewBox={`0 0 ${T_VB_W} ${T_VB_H}`} className="w-full" role="img" aria-label={copy.timelineAria}>
      {/* TTFT bar — fills during the compute-bound prefill */}
      <rect
        x={T_AXIS_X0}
        y={T_AXIS_Y - 12}
        width={T_TTFT_W}
        height={24}
        rx={5}
        style={{ fill: 'var(--primary)', fillOpacity: 0.16, stroke: 'var(--primary)' }}
        strokeWidth={1.5}
      />
      <text
        x={T_AXIS_X0 + T_TTFT_W / 2}
        y={T_AXIS_Y + 4}
        textAnchor="middle"
        style={{ fill: 'var(--foreground)' }}
        className="text-[12px] font-semibold"
      >
        {copy.ttftLabel}
      </text>
      {/* TTFT label group sits ABOVE the bar so the wide EN sublabel never overlaps ticks */}
      <text
        x={T_AXIS_X0}
        y={T_AXIS_Y - 22}
        style={{ fill: 'var(--primary)' }}
        className="text-[11px] font-medium"
      >
        {`${copy.ttftLabel} ≈ ${TTFT_MS} ms`}
      </text>
      <text x={T_AXIS_X0} y={T_AXIS_Y - 22 - 14} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {copy.ttftSub}
      </text>

      {/* baseline track for the decode stream */}
      <line
        x1={T_TICK_X0}
        y1={T_AXIS_Y}
        x2={T_AXIS_X1}
        y2={T_AXIS_Y}
        style={{ stroke: 'var(--border)' }}
        strokeWidth={1.5}
      />

      {/* token ticks — evenly spaced TPOT, streamed after the first token */}
      {Array.from({ length: TICKS }).map((_, i) => {
        const x = T_TICK_X0 + (i + 1) * T_TICK_GAP;
        const on = i < visible;
        return (
          <g key={`tick-${i}`} style={{ opacity: on ? 1 : 0.15, transition: 'opacity 220ms' }}>
            <line x1={x} y1={T_AXIS_Y - 9} x2={x} y2={T_AXIS_Y + 9} stroke={EMERALD} strokeWidth={2} />
            <circle cx={x} cy={T_AXIS_Y - 9} r={2.5} fill={EMERALD} />
          </g>
        );
      })}

      {/* "first token" marker at the TTFT/stream boundary */}
      <circle cx={T_TICK_X0} cy={T_AXIS_Y} r={3.5} style={{ fill: 'var(--primary)' }} />
      <text
        x={T_TICK_X0}
        y={T_AXIS_Y + 26}
        textAnchor="middle"
        style={{ fill: 'var(--primary)' }}
        className="text-[10px] font-medium"
      >
        ↑ {copy.firstTokenTag}
      </text>

      {/* "then stream" + ITL bracket UNDER the ticks (kept below to avoid the tick row) */}
      <text
        x={(T_TICK_X0 + T_AXIS_X1) / 2}
        y={T_AXIS_Y + 44}
        textAnchor="middle"
        fill={EMERALD}
        className="text-[11px] font-medium"
      >
        {copy.streamLabel} · {copy.itlLabel} ≈ {ITL_MS} ms
      </text>
    </svg>
  );
}

// ── Panel 2 geometry: right-skewed latency histogram ──────────────────────────
const H_VB_W = 360;
const H_VB_H = 180;
const H_X0 = 24; // plot left
const H_X1 = 340; // plot right
const H_BASE_Y = 132; // baseline of the bars
const H_TOP_Y = 24; // tallest a bar may reach
const N_BARS = 26;

/**
 * A smooth-ish right-skewed (log-normal-ish) distribution as bar heights in
 * [0,1]. Hand-tuned so the mode sits at the left third and a long thin tail
 * runs to the right — deterministic (no RNG) so SSR === client.
 */
const SKEW_HEIGHTS: number[] = (() => {
  const xs: number[] = [];
  // Peak near bar 6; decay slower to the right than to the left → right skew.
  const mode = 6;
  for (let i = 0; i < N_BARS; i++) {
    const d = i - mode;
    // asymmetric bell: narrow rise on the left, fat tail on the right
    const sigma = d < 0 ? 3.0 : 7.5;
    const h = Math.exp(-(d * d) / (2 * sigma * sigma));
    xs.push(h);
  }
  const max = Math.max(...xs);
  return xs.map((h) => h / max);
})();

const BAR_GAP = 2;
const BAR_W = (H_X1 - H_X0 - (N_BARS - 1) * BAR_GAP) / N_BARS;

/** x of the CENTER of bar i. */
function barCx(i: number): number {
  return H_X0 + i * (BAR_W + BAR_GAP) + BAR_W / 2;
}

// Marker positions, expressed as bar indices (fractional). Median sits at the
// mode-ish center of mass; mean is dragged RIGHT by the tail; percentiles march
// further right toward P99 deep in the tail.
const MARK_P50 = 7.5; // median
const MARK_MEAN = 10.5; // mean — to the RIGHT of the median
const MARK_P90 = 16;
const MARK_P95 = 19.5;
const MARK_P99 = 24;

function markX(barIdx: number): number {
  return H_X0 + barIdx * (BAR_W + BAR_GAP) + BAR_W / 2;
}

/** Panel 2 — right-skewed histogram with P50/mean/P90/P95/P99 markers. */
function HistogramPanel() {
  const copy = COPY[useLocale()];
  return (
    <svg viewBox={`0 0 ${H_VB_W} ${H_VB_H}`} className="w-full" role="img" aria-label={copy.histAria}>
      {/* bars */}
      {SKEW_HEIGHTS.map((h, i) => {
        const barH = (H_BASE_Y - H_TOP_Y) * h;
        const cx = barCx(i);
        // bars left of the median are solid muted; tail bars (right of mean) tint warn-ish
        const inTail = i >= MARK_MEAN;
        return (
          <rect
            key={`bar-${i}`}
            x={cx - BAR_W / 2}
            y={H_BASE_Y - barH}
            width={BAR_W}
            height={barH}
            rx={1.5}
            style={{
              fill: inTail ? 'var(--muted-foreground)' : 'var(--primary)',
              fillOpacity: inTail ? 0.28 : 0.5,
            }}
          />
        );
      })}

      {/* baseline */}
      <line x1={H_X0} y1={H_BASE_Y} x2={H_X1} y2={H_BASE_Y} style={{ stroke: 'var(--border)' }} strokeWidth={1.5} />

      {/* P50 (median) marker — solid, tallest label row */}
      <line
        x1={markX(MARK_P50)}
        y1={H_TOP_Y - 6}
        x2={markX(MARK_P50)}
        y2={H_BASE_Y}
        style={{ stroke: 'var(--foreground)' }}
        strokeWidth={2}
      />
      <text
        x={markX(MARK_P50)}
        y={H_TOP_Y - 10}
        textAnchor="middle"
        style={{ fill: 'var(--foreground)' }}
        className="text-[11px] font-semibold"
      >
        {copy.medianLabel}
      </text>

      {/* mean marker — dashed, to the RIGHT of the median */}
      <line
        x1={markX(MARK_MEAN)}
        y1={H_TOP_Y + 4}
        x2={markX(MARK_MEAN)}
        y2={H_BASE_Y}
        stroke={EMERALD}
        strokeWidth={2}
        strokeDasharray="5 4"
      />
      <text x={markX(MARK_MEAN) + 4} y={H_TOP_Y + 12} textAnchor="start" fill={EMERALD} className="text-[11px] font-semibold">
        {copy.meanLabel}
      </text>

      {/* "mean is right of median" connector — arrow under the two markers */}
      <line
        x1={markX(MARK_P50)}
        y1={H_BASE_Y + 12}
        x2={markX(MARK_MEAN)}
        y2={H_BASE_Y + 12}
        style={{ stroke: 'var(--muted-foreground)' }}
        strokeWidth={1.25}
      />
      <polygon
        points={`${markX(MARK_MEAN)},${H_BASE_Y + 12} ${markX(MARK_MEAN) - 6},${H_BASE_Y + 9} ${markX(MARK_MEAN) - 6},${H_BASE_Y + 15}`}
        style={{ fill: 'var(--muted-foreground)' }}
      />

      {/* P90 / P95 / P99 markers in the tail — light verticals + labels above */}
      {[
        { x: MARK_P90, label: copy.p90 },
        { x: MARK_P95, label: copy.p95 },
        { x: MARK_P99, label: copy.p99 },
      ].map((m) => (
        <g key={m.label}>
          <line
            x1={markX(m.x)}
            y1={H_TOP_Y + 18}
            x2={markX(m.x)}
            y2={H_BASE_Y}
            style={{ stroke: 'var(--muted-foreground)' }}
            strokeWidth={1.5}
            strokeDasharray="3 3"
          />
          <text
            x={markX(m.x)}
            y={H_TOP_Y + 14}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[10px] font-medium"
          >
            {m.label}
          </text>
        </g>
      ))}

      {/* "long slow tail →" under the tail region */}
      <text
        x={markX(MARK_P95)}
        y={H_BASE_Y + 30}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px] italic"
      >
        {copy.tailLabel}
      </text>
    </svg>
  );
}

export function InferenceMetricsDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  // Under reduced motion: show all ticks at once, never play.
  const [visible, setVisible] = React.useState(reducedMotion ? TICKS : 0);
  const [playing, setPlaying] = React.useState(false);

  // Drive the streamed ticks while playing, at a (slowed-for-legibility) cadence.
  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => {
      setVisible((v) => {
        if (v >= TICKS) {
          window.clearInterval(t);
          setPlaying(false);
          return v;
        }
        return v + 1;
      });
    }, 240);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const onPlay = () => {
    if (reducedMotion) return;
    setVisible(0);
    setPlaying(true);
  };

  const done = visible >= TICKS;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        {!reducedMotion ? (
          <button
            type="button"
            onClick={() => (playing ? setPlaying(false) : onPlay())}
            aria-pressed={playing}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            {playing ? copy.pause : done ? copy.replay : copy.play}
          </button>
        ) : null}
      </div>

      {/* Two panels side by side; stack on narrow screens. */}
      <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
        {/* Panel 1 — streamed-output timeline */}
        <div className="flex flex-1 flex-col gap-2 rounded-md border border-border bg-card/40 p-3">
          <div className="text-[11px] font-semibold text-foreground">{copy.timelineTitle}</div>
          <TimelinePanel visible={reducedMotion ? TICKS : visible} />
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded border border-emerald-500/50 bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300">
              {copy.tpsChip(ITL_MS, TPS)}
            </span>
          </div>
          <p className="text-[12px] leading-snug text-foreground/85">{copy.timelineCaption}</p>
        </div>

        {/* Panel 2 — right-skewed latency histogram */}
        <div className="flex flex-1 flex-col gap-2 rounded-md border border-border bg-card/40 p-3">
          <div className="text-[11px] font-semibold text-foreground">{copy.histTitle}</div>
          <HistogramPanel />
          <div className="text-[10px] text-muted-foreground/80">{copy.percentileLegend}</div>
          <p className="text-[12px] leading-snug text-foreground/85">{copy.histCaption}</p>
        </div>
      </div>

      {/* Honesty — inference-time vs end-to-end */}
      <div className="rounded-md border border-border/60 bg-muted/20 p-3">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.honestyTitle}</div>
        <p className="mt-1.5 text-[12px] leading-snug text-foreground/85">{copy.honesty}</p>
      </div>

      <p className="text-[10px] italic text-muted-foreground/70">{copy.illustrative}</p>
    </div>
  );
}
