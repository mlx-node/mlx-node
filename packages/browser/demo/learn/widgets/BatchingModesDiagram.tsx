import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * BatchingModesDiagram — the three server batching strategies, and the
 * latency ↔ throughput trade-off they all dance around.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads. Every number here is ILLUSTRATIVE timing on a
 * synthetic millisecond axis — it teaches the SHAPE of each schedule, it does
 * not benchmark a real server.
 *
 * THE LESSON (mirrors the book's batching figures 7.6–7.8): a single decode
 * step streams ALL of the model's weights from memory exactly once. That one
 * weight-read is the bottleneck the course already named "memory-bound decode".
 * If N requests ride that SAME weight-read, the server produces N tokens for
 * the price of one memory sweep — so total throughput climbs with the batch
 * while the memory traffic stays put. Batching is the cure for the memory wall.
 *
 * Three schedules, switched by a 3-way toggle (each re-lays-out the request
 * lanes — a queue-wait segment then an inference segment — on the ms axis):
 *   STATIC      — the server waits until the batch is FULL before it starts, so
 *                 early arrivals sit in a long queue and everyone launches late,
 *                 together.
 *   DYNAMIC     — waits until full OR a timeout fires; whoever made the cutoff
 *                 runs now, the stragglers are batched separately afterwards.
 *   CONTINUOUS  — a.k.a. in-flight batching (vLLM, SGLang; TensorRT-LLM calls it
 *                 "in-flight"): the server swaps new requests into free slots at
 *                 the TOKEN level, so each request starts almost immediately and
 *                 the queue stays tiny.
 *
 * A batch-size slider drives two readouts that move in OPPOSITE directions:
 * total throughput rises then saturates (you stop being memory-bound and start
 * being compute-bound), while per-user latency rises (more lane-mates to share
 * each step with). That tension is the whole reason batching is a knob, not a
 * free win.
 *
 * HONESTY (the "you are here" pin): this in-browser demo is a SINGLE user at
 * BATCH = 1 — zero batching, the pathological worst case for the very
 * memory-bound decode the course taught. A shared production API weaves dozens
 * of strangers into one continuous batch, which is why a hosted model feels
 * fast under load while your private in-tab copy stays bandwidth-bottlenecked.
 * Same model, opposite economics.
 *
 * Honors prefers-reduced-motion: no animation is used (the schedules are static
 * snapshots), so there is no Play control and nothing to suppress — the toggle
 * and slider are the only interactions.
 */

type Mode = 'static' | 'dynamic' | 'continuous';
const MODES: Mode[] = ['static', 'dynamic', 'continuous'];

// Emerald accent for the "inference / running" segment (matches the project's
// other widgets). Queue-wait segments use the muted token.
const EMERALD = '#10b981';

// ── A synthetic four-request workload on a millisecond axis. Each request has
// an arrival time; each schedule decides when it actually STARTS running. The
// inference segment length is fixed per request (illustrative). These are the
// only ground-truth inputs — the per-schedule start times are derived below. ──
type Req = { id: number; arrive: number; infer: number };
const REQS: Req[] = [
  { id: 1, arrive: 0, infer: 30 },
  { id: 2, arrive: 8, infer: 30 },
  { id: 3, arrive: 16, infer: 30 },
  { id: 4, arrive: 40, infer: 30 }, // arrives after the others — the "straggler"
];

const BATCH_FULL = 3; // a "full" batch in this toy = 3 requests
const DYN_TIMEOUT = 24; // dynamic-batch cutoff: launch what's queued at t=24ms
const T_MAX = 80; // axis max (ms)

type Lane = { id: number; arrive: number; start: number; infer: number; group: number };

/**
 * Derive the per-request (arrive, start) lanes for one schedule.
 *  - static:     nobody starts until the batch is full (3rd arrival). The
 *                straggler (#4) forms a second full-wait batch of its own.
 *  - dynamic:    launch the queue at the timeout (or when full, whichever is
 *                first). #1–3 are full at t=16 → launch at 16; #4 arrives later
 *                and goes in a separate batch at its own arrival.
 *  - continuous: each request starts the moment it arrives (slots are free at
 *                the token level), so start ≈ arrive.
 */
function lanesFor(mode: Mode): Lane[] {
  if (mode === 'continuous') {
    return REQS.map((r) => ({ id: r.id, arrive: r.arrive, start: r.arrive, infer: r.infer, group: 0 }));
  }
  if (mode === 'dynamic') {
    // First three are full by t=16; the cutoff is t=DYN_TIMEOUT (24) but "full"
    // comes first, so group 0 launches at the 3rd arrival (16). #4 is a late
    // straggler that arrives WHILE group 0 is still running. Dynamic batching
    // forms whole batches, so it cannot slip into the in-flight one — that
    // token-level injection is exactly the continuous-batching trick. So it
    // waits for the GPU to free at group 0's end, then runs as its own batch:
    // a short but visible queue wait (vs continuous, which starts it at arrival).
    const fullAt = REQS[BATCH_FULL - 1]!.arrive; // 16
    const g0Start = Math.min(fullAt, DYN_TIMEOUT);
    const g0End = g0Start + REQS[0]!.infer; // GPU busy until here (16 + 30 = 46)
    return REQS.map((r, i) =>
      i < BATCH_FULL
        ? { id: r.id, arrive: r.arrive, start: g0Start, infer: r.infer, group: 0 }
        : { id: r.id, arrive: r.arrive, start: Math.max(r.arrive, g0End), infer: r.infer, group: 1 },
    );
  }
  // static: the whole first batch waits for the batch to be FULL (3rd arrival),
  // and everyone launches together then. #4 waits for a second full batch — but
  // with no more arrivals it would stall; we show it launching when it arrives
  // PLUS the same full-wait penalty, i.e. it too pays the "wait for full".
  const fullAt = REQS[BATCH_FULL - 1]!.arrive; // 16
  return REQS.map((r, i) =>
    i < BATCH_FULL
      ? { id: r.id, arrive: r.arrive, start: fullAt, infer: r.infer, group: 0 }
      : { id: r.id, arrive: r.arrive, start: r.arrive + (fullAt - REQS[0]!.arrive), infer: r.infer, group: 1 },
  );
}

/** Mean per-request latency (queue wait + inference) for a schedule, in ms. */
function meanLatency(lanes: Lane[]): number {
  const total = lanes.reduce((s, l) => s + (l.start - l.arrive) + l.infer, 0);
  return total / lanes.length;
}

// ── Batch-size → (throughput, latency) toy model. Throughput rises with batch
// then saturates (a smooth approach to a ceiling: you leave the memory-bound
// regime and hit a compute/SRAM ceiling). Per-user latency rises roughly
// linearly (more lane-mates per step). Both normalized to readable units. ────
// Chosen so the two readouts stay PHYSICALLY CONSISTENT: total throughput must
// equal batch × per-user token rate. With latencyFor(b) = 16 + 2b ms/token, the
// per-user rate is 1000/(16+2b) tok/s, so total = 1000·b/(16+2b) = 500·b/(b+8).
// That is exactly throughputFor(b) when ceiling = 500 and half = 8 — at batch 1
// both readouts describe the same single stream (≈56 tok/s ⇔ 18 ms/token).
const TPUT_CEILING = 500; // tokens/s ceiling (illustrative; see note above)
const TPUT_HALF = 8; // batch at which throughput hits half the ceiling (250 tok/s)

/** Saturating throughput: ceiling · b / (b + half). Monotone up, saturates. */
function throughputFor(batch: number): number {
  return (TPUT_CEILING * batch) / (batch + TPUT_HALF);
}
/** Per-user latency in ms: a floor plus a per-lane-mate cost (monotone up). */
function latencyFor(batch: number): number {
  return 18 + 2.0 * (batch - 1);
}

const COPY = {
  en: {
    title: 'Batching — three schedules, one trade-off',
    modeLabel: 'schedule',
    modes: {
      static: 'static',
      dynamic: 'dynamic',
      continuous: 'continuous',
    },
    modeAria: {
      static: 'Static batching — wait until the batch is full before starting',
      dynamic: 'Dynamic batching — start when full or when a timeout fires',
      continuous: 'Continuous (in-flight) batching — swap requests in at the token level',
    },
    modeTag: {
      static: 'wait for full',
      dynamic: 'full or timeout',
      continuous: 'in-flight · vLLM / SGLang',
    },
    blurbs: {
      static: (
        <>
          The server waits until the batch is <strong>full</strong> before it runs. Early requests sit in a long{' '}
          <span className="text-muted-foreground">queue</span> and everyone launches late, together — simple, but the
          first arrival pays for the last.
        </>
      ),
      dynamic: (
        <>
          Waits until <strong>full OR a timeout</strong> fires. Whoever made the cutoff runs now; the late straggler
          gets <em>batched separately</em> in the next group — a compromise between filling the GPU and not making
          early requests wait forever.
        </>
      ),
      continuous: (
        <>
          Also called <strong>in-flight</strong> batching (TensorRT-LLM&apos;s term). New requests slot into free seats
          at the <strong>token level</strong>, so each one starts almost immediately and the queue stays tiny. This is
          what <span className="font-mono">vLLM</span> and <span className="font-mono">SGLang</span> do.
        </>
      ),
    },
    axisLabel: 'time (ms) →',
    arriveLabel: 'arrives',
    queueLabel: 'queue',
    runLabel: 'running',
    reqLabel: (n: number) => `req ${n}`,
    laneAria: (mode: string) => `Request timeline for ${mode} batching: each lane shows queue wait then inference.`,
    meanLatency: 'mean wait + run',
    whyHeading: 'Why batching cures the memory wall',
    why: (
      <>
        One decode step streams <strong>all</strong> the model&apos;s weights from memory <strong>once</strong>. Put{' '}
        <strong>N</strong> requests on that same weight-read and the GPU emits <strong>N tokens</strong> for one memory
        sweep — total throughput climbs while the memory traffic stays flat. That is the cure for the{' '}
        <em>memory-bound</em> decode this course already taught.
      </>
    ),
    tradeoffHeading: 'The knob: latency ↔ throughput',
    batchLabel: 'batch size',
    throughputReadout: 'total throughput',
    latencyReadout: 'per-user latency',
    tputUnit: 'tok/s',
    latUnit: 'ms / token',
    risesSaturates: 'rises, then saturates',
    risesLinear: 'rises with the batch',
    youAreHere: 'you are here',
    youAreHereDetail: 'batch = 1 · single user · zero batching',
    honestyHeading: 'Your in-tab copy vs a shared API',
    honesty: (
      <>
        This in-browser demo runs as a <strong>single user at batch = 1</strong> — the <em>worst</em> case for the
        memory-bound decode, because that one expensive weight-read carries exactly one token. A shared production API
        weaves dozens of strangers into one <strong>continuous</strong> batch, amortizing the same weight-read across
        all of them. That is why a hosted model feels fast under load while your private copy stays
        bandwidth-bottlenecked. <strong>Same model, opposite economics.</strong>
      </>
    ),
    footnote:
      'Illustrative timings on a synthetic millisecond axis and a toy throughput/latency model — they teach the shape of each schedule, not a real benchmark.',
    ariaLabel:
      'Diagram of three server batching schedules — static, dynamic, and continuous (in-flight) — as request lanes on a millisecond axis, plus a batch-size slider whose throughput readout rises and saturates while per-user latency rises, and a locked "you are here: batch = 1" marker for the single-user in-browser demo.',
  },
  zh: {
    title: 'batching——三种 schedule，一个 trade-off',
    modeLabel: 'schedule',
    modes: {
      static: 'static',
      dynamic: 'dynamic',
      continuous: 'continuous',
    },
    modeAria: {
      static: 'static batching——等到 batch 装满才开始',
      dynamic: 'dynamic batching——装满或超时触发时开始',
      continuous: 'continuous（in-flight）batching——在 token 级别把请求换入空位',
    },
    modeTag: {
      static: '等装满',
      dynamic: '装满或超时',
      continuous: 'in-flight · vLLM / SGLang',
    },
    blurbs: {
      static: (
        <>
          server 等到 batch <strong>装满</strong> 才运行。先到的请求要在长长的{' '}
          <span className="text-muted-foreground">queue</span> 里干等，最后大家一起晚一拍启动——简单，但第一个到的为最后一个埋单。
        </>
      ),
      dynamic: (
        <>
          等到 <strong>装满或超时</strong> 触发。赶上 cutoff 的现在就跑；迟到的 straggler 被{' '}
          <em>单独 batch</em> 进下一组——在喂饱 GPU 与不让先到者无限等待之间折中。
        </>
      ),
      continuous: (
        <>
          也叫 <strong>in-flight</strong> batching（TensorRT-LLM 的叫法）。新请求在{' '}
          <strong>token 级别</strong> 插进空座位，于是每个几乎立刻开始、queue 始终很短。这正是{' '}
          <span className="font-mono">vLLM</span> 和 <span className="font-mono">SGLang</span> 的做法。
        </>
      ),
    },
    axisLabel: 'time（ms）→',
    arriveLabel: '到达',
    queueLabel: 'queue',
    runLabel: '运行中',
    reqLabel: (n: number) => `req ${n}`,
    laneAria: (mode: string) => `${mode} batching 的请求时间线：每条 lane 先是 queue 等待，再是推理。`,
    meanLatency: '平均等待 + 运行',
    whyHeading: '为什么 batching 能治好 memory wall',
    why: (
      <>
        一个 decode step 把模型的 <strong>全部</strong> 权重从内存里读 <strong>一遍</strong>。让 <strong>N</strong>{' '}
        个请求共用这一次权重读取，GPU 就用一次内存扫描吐出 <strong>N 个 token</strong>——total throughput
        上升，而内存流量纹丝不动。这正是本课程讲过的 <em>memory-bound</em> decode 的解药。
      </>
    ),
    tradeoffHeading: '那个旋钮：latency ↔ throughput',
    batchLabel: 'batch size',
    throughputReadout: 'total throughput',
    latencyReadout: 'per-user latency',
    tputUnit: 'tok/s',
    latUnit: 'ms / token',
    risesSaturates: '先升，后饱和',
    risesLinear: '随 batch 上升',
    youAreHere: '你在这里',
    youAreHereDetail: 'batch = 1 · 单用户 · 零 batching',
    honestyHeading: '你这一标签页的副本 vs 共享 API',
    honesty: (
      <>
        这个浏览器内的 demo 以 <strong>单用户、batch = 1</strong> 运行——这是 memory-bound decode 的{' '}
        <em>最坏</em> 情况，因为那一次昂贵的权重读取只驮着一个 token。共享的生产 API 把几十个陌生人编织进一个{' '}
        <strong>continuous</strong> batch，让同一次权重读取被所有人分摊。这就是为什么托管模型在高负载下反而感觉快，
        而你的私有副本始终卡在带宽上。<strong>同一个模型，相反的经济学。</strong>
      </>
    ),
    footnote:
      '合成毫秒轴上的示意时序，加上一个玩具版的 throughput/latency 模型——它讲的是每种 schedule 的形状，不是真实 benchmark。',
    ariaLabel:
      '三种 server batching schedule（static、dynamic、continuous / in-flight）的示意图：把请求画成毫秒轴上的 lane，外加一个 batch size 滑块——throughput 读数先升后饱和，per-user latency 随之上升——以及一个锁定的「你在这里：batch = 1」标记，对应单用户的浏览器内 demo。',
  },
} as const;

// ── Lane SVG geometry. EN labels are wider than zh; sized for EN. ────────────
const VB_W = 720;
const LANE_H = 26;
const LANE_GAP = 8;
const LANE_LABEL_W = 64; // left gutter for "req N"
const TRACK_X = LANE_LABEL_W + 8;
const TRACK_W = VB_W - TRACK_X - 16;
const TOP_PAD = 20; // room for the time axis ticks
const BOT_PAD = 18; // room for the axis label
const VB_H = TOP_PAD + REQS.length * (LANE_H + LANE_GAP) + BOT_PAD;

/** Map a time in ms to an x coordinate on the track. */
function tx(ms: number): number {
  return TRACK_X + (ms / T_MAX) * TRACK_W;
}

function LaneChart({ mode }: { mode: Mode }) {
  const copy = COPY[useLocale()];
  const lanes = React.useMemo(() => lanesFor(mode), [mode]);
  const ticks = [0, 20, 40, 60, 80];

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="w-full"
      role="img"
      aria-label={copy.laneAria(copy.modes[mode])}
    >
      {/* time axis ticks + gridlines */}
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={tx(t)}
            y1={TOP_PAD - 4}
            x2={tx(t)}
            y2={VB_H - BOT_PAD}
            style={{ stroke: 'var(--border)' }}
            strokeWidth={1}
            strokeDasharray="2 3"
          />
          <text
            x={tx(t)}
            y={TOP_PAD - 8}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="font-mono text-[10px]"
          >
            {t}
          </text>
        </g>
      ))}

      {/* one lane per request */}
      {lanes.map((lane, i) => {
        const y = TOP_PAD + i * (LANE_H + LANE_GAP);
        const qx0 = tx(lane.arrive);
        const qx1 = tx(lane.start);
        const rx1 = tx(lane.start + lane.infer);
        const queueW = Math.max(0, qx1 - qx0);
        const runW = Math.max(2, rx1 - qx1);
        return (
          <g key={lane.id}>
            {/* lane label */}
            <text
              x={4}
              y={y + LANE_H / 2 + 4}
              style={{ fill: 'var(--muted-foreground)' }}
              className="font-mono text-[11px]"
            >
              {copy.reqLabel(lane.id)}
            </text>
            {/* baseline track */}
            <line
              x1={TRACK_X}
              y1={y + LANE_H / 2}
              x2={TRACK_X + TRACK_W}
              y2={y + LANE_H / 2}
              style={{ stroke: 'var(--border)' }}
              strokeWidth={1}
            />
            {/* arrival tick */}
            <line
              x1={qx0}
              y1={y + 2}
              x2={qx0}
              y2={y + LANE_H - 2}
              style={{ stroke: 'var(--muted-foreground)' }}
              strokeWidth={1.5}
            />
            {/* queue-wait segment (muted, hatched feel via low opacity) */}
            {queueW > 1 ? (
              <rect
                x={qx0}
                y={y + 4}
                width={queueW}
                height={LANE_H - 8}
                rx={3}
                style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
                strokeWidth={1}
              />
            ) : null}
            {/* inference / running segment (emerald) */}
            <rect
              x={qx1}
              y={y + 2}
              width={runW}
              height={LANE_H - 4}
              rx={3}
              fill={EMERALD}
              fillOpacity={0.18}
              stroke={EMERALD}
              strokeWidth={1.5}
            />
            {/* running label inside the segment when wide enough */}
            {runW > 44 ? (
              <text
                x={qx1 + runW / 2}
                y={y + LANE_H / 2 + 4}
                textAnchor="middle"
                fill={EMERALD}
                className="text-[10px] font-medium"
              >
                {copy.runLabel}
              </text>
            ) : null}
          </g>
        );
      })}

      {/* axis caption */}
      <text
        x={TRACK_X}
        y={VB_H - 4}
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {copy.axisLabel}
      </text>
    </svg>
  );
}

/**
 * The throughput-vs-latency mini chart: two curves over batch size with a moving
 * dot at the current batch, plus a LOCKED "you are here: batch = 1" marker.
 */
const TV_W = 420;
const TV_H = 150;
const TV_PAD_L = 36;
const TV_PAD_R = 36;
const TV_PAD_T = 14;
const TV_PAD_B = 22;
const BATCH_MIN = 1;
const BATCH_MAX = 32;

function TradeoffChart({ batch }: { batch: number }) {
  const copy = COPY[useLocale()];
  const plotW = TV_W - TV_PAD_L - TV_PAD_R;
  const plotH = TV_H - TV_PAD_T - TV_PAD_B;

  const bx = (b: number) => TV_PAD_L + ((b - BATCH_MIN) / (BATCH_MAX - BATCH_MIN)) * plotW;
  // Throughput uses the left y-axis (0..ceiling), latency the right (0..max).
  const maxLat = latencyFor(BATCH_MAX);
  const tputY = (v: number) => TV_PAD_T + (1 - v / TPUT_CEILING) * plotH;
  const latY = (v: number) => TV_PAD_T + (1 - v / maxLat) * plotH;

  const samples = React.useMemo(() => {
    const out: number[] = [];
    for (let b = BATCH_MIN; b <= BATCH_MAX; b++) out.push(b);
    return out;
  }, []);
  const tputPath = samples.map((b, i) => `${i === 0 ? 'M' : 'L'} ${bx(b)} ${tputY(throughputFor(b))}`).join(' ');
  const latPath = samples.map((b, i) => `${i === 0 ? 'M' : 'L'} ${bx(b)} ${latY(latencyFor(b))}`).join(' ');

  const curTputY = tputY(throughputFor(batch));
  const curLatY = latY(latencyFor(batch));

  return (
    <svg viewBox={`0 0 ${TV_W} ${TV_H}`} className="w-full" aria-hidden="true" focusable="false">
      {/* plot frame */}
      <line
        x1={TV_PAD_L}
        y1={TV_PAD_T}
        x2={TV_PAD_L}
        y2={TV_H - TV_PAD_B}
        style={{ stroke: 'var(--border)' }}
        strokeWidth={1}
      />
      <line
        x1={TV_PAD_L}
        y1={TV_H - TV_PAD_B}
        x2={TV_W - TV_PAD_R}
        y2={TV_H - TV_PAD_B}
        style={{ stroke: 'var(--border)' }}
        strokeWidth={1}
      />

      {/* throughput curve (primary) */}
      <path d={tputPath} fill="none" style={{ stroke: 'var(--primary)' }} strokeWidth={2} />
      {/* latency curve (emerald) */}
      <path d={latPath} fill="none" stroke={EMERALD} strokeWidth={2} strokeDasharray="5 3" />

      {/* "you are here" locked marker at batch = 1 */}
      <line
        x1={bx(1)}
        y1={TV_PAD_T}
        x2={bx(1)}
        y2={TV_H - TV_PAD_B}
        style={{ stroke: 'var(--foreground)' }}
        strokeWidth={1.5}
        strokeDasharray="3 3"
      />
      <text
        x={bx(1) + 4}
        y={TV_PAD_T + 9}
        style={{ fill: 'var(--foreground)' }}
        className="text-[10px] font-semibold"
      >
        🔒 {copy.youAreHere}
      </text>

      {/* current-batch dots */}
      <circle cx={bx(batch)} cy={curTputY} r={4} style={{ fill: 'var(--primary)' }} />
      <circle cx={bx(batch)} cy={curLatY} r={4} fill={EMERALD} />
      <line
        x1={bx(batch)}
        y1={TV_PAD_T}
        x2={bx(batch)}
        y2={TV_H - TV_PAD_B}
        style={{ stroke: 'var(--primary)', opacity: 0.35 }}
        strokeWidth={1}
      />

      {/* axis labels */}
      <text
        x={TV_PAD_L}
        y={TV_H - 6}
        style={{ fill: 'var(--muted-foreground)' }}
        className="font-mono text-[10px]"
      >
        {BATCH_MIN}
      </text>
      <text
        x={TV_W - TV_PAD_R}
        y={TV_H - 6}
        textAnchor="end"
        style={{ fill: 'var(--muted-foreground)' }}
        className="font-mono text-[10px]"
      >
        {BATCH_MAX}
      </text>
      <text
        x={(TV_PAD_L + (TV_W - TV_PAD_R)) / 2}
        y={TV_H - 6}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {copy.batchLabel} →
      </text>
    </svg>
  );
}

export function BatchingModesDiagram() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('continuous');
  const [batch, setBatch] = React.useState(1);

  const lanes = React.useMemo(() => lanesFor(mode), [mode]);
  const mlat = meanLatency(lanes);

  const tput = throughputFor(batch);
  const lat = latencyFor(batch);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + 3-way schedule toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.modeLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {MODES.map((m, i) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
                aria-label={copy.modeAria[m]}
                className={[
                  'px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: mode === m ? 'var(--primary)' : 'transparent',
                  color: mode === m ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {copy.modes[m]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Active-mode tag + blurb */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded border border-primary/50 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
          {copy.modeTag[mode]}
        </span>
      </div>
      <p className="min-h-[3rem] text-[13px] text-foreground/90">{copy.blurbs[mode]}</p>

      {/* Request lanes on the ms axis */}
      <LaneChart mode={mode} />

      {/* Legend + mean latency for the active schedule */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-3 text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-3 rounded-[2px] border border-border"
              style={{ background: 'var(--muted)' }}
            />
            {copy.queueLabel}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-3 rounded-[2px]"
              style={{ background: 'rgba(16,185,129,0.18)', border: `1.5px solid ${EMERALD}` }}
            />
            {copy.runLabel}
          </span>
        </div>
        <div className="text-muted-foreground">
          {copy.meanLatency}: <span className="font-mono text-foreground/80">{mlat.toFixed(0)} ms</span>
        </div>
      </div>

      {/* Why batching cures the memory wall */}
      <div className="rounded-md border border-border/70 bg-card/40 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {copy.whyHeading}
        </div>
        <p className="text-[13px] text-foreground/90">{copy.why}</p>
      </div>

      {/* The latency ↔ throughput knob */}
      <div className="space-y-2 rounded-md border border-border/70 bg-card/40 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            {copy.tradeoffHeading}
          </div>
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <label htmlFor="batching-batch-slider" className="lowercase">
              {copy.batchLabel}
            </label>
            <input
              id="batching-batch-slider"
              type="range"
              min={BATCH_MIN}
              max={BATCH_MAX}
              step={1}
              value={batch}
              onChange={(e) => setBatch(Number.parseInt(e.target.value, 10))}
              className="h-2 w-40 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
            />
            <span className="w-6 font-mono text-foreground/80">{batch}</span>
          </div>
        </div>

        {/* Two opposite readouts */}
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded border border-primary/40 bg-primary/5 px-3 py-2">
            <div className="text-[11px] text-muted-foreground">{copy.throughputReadout}</div>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-[18px] font-semibold text-primary">{tput.toFixed(0)}</span>
              <span className="text-[11px] text-muted-foreground">{copy.tputUnit}</span>
              <span className="ml-auto text-[10px] text-primary/80">↑ {copy.risesSaturates}</span>
            </div>
          </div>
          <div
            className="rounded border px-3 py-2"
            style={{ borderColor: 'rgba(16,185,129,0.4)', background: 'rgba(16,185,129,0.05)' }}
          >
            <div className="text-[11px] text-muted-foreground">{copy.latencyReadout}</div>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-[18px] font-semibold" style={{ color: EMERALD }}>
                {lat.toFixed(0)}
              </span>
              <span className="text-[11px] text-muted-foreground">{copy.latUnit}</span>
              <span className="ml-auto text-[10px]" style={{ color: EMERALD }}>
                ↑ {copy.risesLinear}
              </span>
            </div>
          </div>
        </div>

        {/* Curves + locked you-are-here marker */}
        <TradeoffChart batch={batch} />
        <div className="flex items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1.5 rounded border border-foreground/30 bg-foreground/5 px-2 py-0.5 font-medium text-foreground/80">
            🔒 {copy.youAreHere}: {copy.youAreHereDetail}
          </span>
        </div>
      </div>

      {/* Honesty — single user vs shared API */}
      <div className="rounded-md border border-emerald-500/40 bg-emerald-500/5 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">
          {copy.honestyHeading}
        </div>
        <p className="text-[13px] text-foreground/90">{copy.honesty}</p>
      </div>

      {/* Hidden full-diagram aria description for the whole widget */}
      <span className="sr-only">{copy.ariaLabel}</span>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
