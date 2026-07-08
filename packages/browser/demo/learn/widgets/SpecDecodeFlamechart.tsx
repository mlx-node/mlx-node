import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SpecDecodeFlamechart — Multi-Token Prediction sub-chapter: an illustrative,
 * GPU-profiler-style timeline of WHERE THE TIME GOES during decode, comparing
 * plain autoregressive decoding against speculative decoding on one clock.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * What it teaches:
 *   - a decode pass is memory-bound: the zoom panel shows the memory row busy
 *     wall-to-wall while the compute row sits nearly idle, so a verify pass
 *     that scores D+1 positions rides the same weight stream and costs about
 *     the same as a plain one-token pass — the verify free ride;
 *   - if K of D drafts are accepted, the pass emits K+1 tokens (never 0);
 *   - the discovery arc under the toggles: D has diminishing returns,
 *     predictability (acceptance) gates everything, and rejected drafts NEVER
 *     widen a pass — the waste rides a memory sweep that was paid for anyway.
 *
 * Where every number comes from:
 *   - ≈20 ms per pass: this tab's measured ~50 tok/s plain WebGPU decode on an
 *     M-series Mac — the number the chat HUD shows. NOT an exact constant;
 *     labelled "≈" and "synthetic axis" in the UI.
 *   - ~1.6 GB: the course's Qwen3.5-0.8B bf16 weight footprint.
 *   - ≈1 ms PER DRAFT TOKEN: the MTP head is ≈ 1 transformer layer of 24 →
 *     ~0.8 ms of a 20 ms pass, and the head runs once per draft token (each
 *     guess feeds the next — see "The loop" in the chapter), so a depth-D
 *     cycle pays a D × 1 ms drafting tax. Illustrative — labelled as such.
 *   - D = 1 default: Qwen3.5's real MTP depth; higher D is engine-raised.
 *   - 1.06–1.46×: the ONLY measured speculative speedup in this project
 *     (native Metal engine, depth D = 1). The ACCEPTS tables below are
 *     scripted, NOT measured — the UI says so unconditionally.
 *
 * HONESTY: this browser tab never runs the speculative lane — its WebGPU
 * decode loop is plain autoregressive (the checkpoint ships 0 mtp.* tensors),
 * and the whole story is batch-1 only: at high batch the compute row is
 * already busy and servers switch speculation off. The illustrative caption,
 * footnote, and batch note below the chart are never gated on state.
 *
 * Reduced motion: the frame is pinned to the tally (including when the
 * preference flips on mid-animation), Play/Step are removed from the DOM, and
 * all CSS transitions are nulled. The D/scenario toggles — the core
 * interaction — still fully work.
 */

type ScenarioId = 'predictable' | 'typical' | 'surprising';

// Scripted K (accepted drafts) per verify pass, [scenario][D-1] → 6 passes each.
// Hand-authored near prefix-acceptance with per-position odds ~0.8 / ~0.55 / ~0.3.
// NOT measured — labelled illustrative in the UI (illustrativeCaption + footnote).
// INVARIANTS (checkable mechanically):
//   - every K ≤ its D (no accepting more than was drafted)
//   - every pass emits K + 1 ≥ 1 tokens (K+1 rule; no pass emits 0)
//   - totals: plain lane always 6 tokens / 120 ms; spec lane per cell =
//     sum(K)+6 tokens / (120 + 6·D) ms — the head drafts one token at a time,
//     so every cycle pays a D × 1 ms drafting tax
//   - VERIFIED speedup matrix, ×(20 · tokens / (120 + 6·D)) to 2 dp:
//                    D=1     D=2     D=3     D=4
//     predictable   ×1.75   ×2.27   ×2.61   ×2.78
//     typical       ×1.43   ×1.67   ×1.74   ×1.81
//     surprising    ×1.27   ×1.21   ×1.30   ×1.25
//   - DEFAULT cell (typical, D=1) must compute to ×1.43 — inside the measured 1.06–1.46× range
//   - surprising is NON-monotonic in D (×1.27 → ×1.21 at D=2): on hard text a
//     deeper draft grows the tax faster than acceptance — deliberate, teachable
const ACCEPTS: Record<ScenarioId, [number[], number[], number[], number[]]> = {
  predictable: [
    [1, 1, 0, 1, 1, 1],
    [2, 1, 2, 0, 2, 2],
    [3, 2, 3, 0, 2, 2],
    [4, 2, 3, 0, 3, 2],
  ],
  typical: [
    [1, 0, 1, 1, 0, 0],
    [2, 0, 1, 1, 0, 1],
    [2, 0, 1, 3, 0, 0],
    [2, 0, 1, 4, 0, 0],
  ],
  surprising: [
    [0, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 2, 0, 0],
    [1, 0, 0, 2, 0, 0],
  ],
};
const SCENARIO_ODDS: Record<ScenarioId, number> = { predictable: 80, typical: 55, surprising: 30 };
const SCENARIO_IDS = ['predictable', 'typical', 'surprising'] as const;

// ── Tuning consts ────────────────────────────────────────────────────────────
const PASS_MS = 20; // ≈20 ms per forward pass — from this tab's measured ~50 tok/s plain decode
//                     (M-series Mac, the number the chat HUD shows). NOT an exact constant —
//                     labelled "≈" in the UI. Cross-check: 1.6 GB / 20 ms = 80 GB/s effective
//                     bandwidth, physically sane for M-series WebGPU.
const DRAFT_MS = 1; // per DRAFT TOKEN — the MTP head runs once per guess (each feeds the next),
//                     so a depth-D cycle pays D × 1 ms. The head ≈ 1 transformer layer of 24 →
//                     ~0.8 ms of a 20 ms pass. Illustrative — labelled as such.
const D_MAX = 4; // deepest engine-raised setting the toggle offers
const N_PASSES = 6; // fixed pass count per lane; keeps geometry stable across D/scenario
const PLAIN_TOTAL_MS = N_PASSES * PASS_MS; // 120 → 6 tokens ⇒ 50 tok/s, matching the anchor
// The spec lane's clock depends on D: step = 20 + D·1 ms, total = 120 + 6·D ms
// (126/132/138/144). Computed IN the component so the trace geometry, the axis
// and the readout all derive from the one specTotalMs — never duplicated.
const MAX_SPEC_TOTAL_MS = N_PASSES * (PASS_MS + D_MAX * DRAFT_MS); // 144 — the fixed axis extent
const EMERALD = '#10b981'; // emerald-500 — accepted
const RED = '#ef4444'; // red-500 — rejected
const FRAME_MS = 1500;
const TOTAL_FRAMES = N_PASSES + 2; // 8: 0 = start; 1..6 = after pass i; 7 = tally (playhead fades)

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
const COPY = {
  en: {
    title: 'Where the time goes — an illustrative GPU timeline',
    play: '▶ Play',
    pause: '❚❚ Pause',
    step: 'Step ›',
    dLabel: 'drafts per pass D:',
    dAria: (n: number) => `Draft ${n} token${n === 1 ? '' : 's'} per pass (scripted trace)`,
    dNote: "D = 1 is Qwen3.5's own MTP depth; higher D is an engine-raised, illustrative setting.",
    scenarioLabel: 'how predictable is the text?',
    scenarioNames: { predictable: 'predictable', typical: 'typical', surprising: 'surprising' },
    scenarioAria: (name: string, pct: number) =>
      `${name} text — per-position draft survival odds around ${pct}% (scripted)`,
    scenarioHint: (acc: number, total: number, pct: number) =>
      `scripted outcome here: ${acc}/${total} draft positions survive verify — per-position odds ~${pct}%, but a position only counts if everything before it survived`,
    laneAName: 'plain autoregressive — one ≈20 ms pass per token',
    laneBName: 'speculative — ≈1 ms per draft token + one ≈20 ms verify pass',
    axisLabel: 'ms — synthetic axis, anchored at ≈20 ms per pass',
    passTok: (n: number) => `${n} tok`,
    zoomTitle: 'one pass, magnified — where its ≈20 ms actually go',
    zoomPlainLabel: 'plain pass — scores 1 position',
    zoomVerifyLabel: (n: number) => `verify pass — scores ${n} positions on the same weight stream`,
    memLabel: 'memory',
    computeLabel: 'compute',
    memBusyLabel: 'streaming ~1.6 GB of bf16 weights — busy the entire pass',
    computeIdleLabel: 'mostly idle — the math is a rounding error',
    equalNote:
      'Same width on purpose: checking D + 1 positions is still one pass over the weights, so it costs about the same as producing one token.',
    legendPass: 'verify pass (streams all the weights)',
    legendDraft: 'draft — MTP head, ≈1 ms per draft token',
    legendAccepted: 'accepted draft position',
    legendWasted: 'wasted position — computed, then discarded',
    legendFreeTok: 'free token (from an accepted draft)',
    legendAnywayTok: 'the token the pass would have produced anyway',
    readoutPlainTitle: 'plain lane',
    readoutPlainValue: '6 tokens / 120 ms ≈ 50 tok/s',
    readoutSpecTitle: 'speculative lane',
    readoutSpecValue: (t: number, ms: number, r: number) => `${t} tokens / ${ms} ms ≈ ${r} tok/s`,
    readoutSpeedupTitle: 'speedup (this scripted trace)',
    readoutSpeedupValue: (x: string) => `×${x}`,
    readoutMeasured: 'measured, for real: 1.06–1.46× at D = 1',
    readoutWaste: (n: number) => (
      <>
        {n} draft positions computed, then <span className="text-red-600 dark:text-red-400">thrown away</span> — inside
        passes that were paid for anyway
      </>
    ),
    readoutTax: (tax: number, total: number) => `drafting tax: ${tax} ms of ${total} ms`,
    captionStart: 'Two GPUs, same job. Press Play — or step — to sweep the playhead and watch tokens land.',
    captionSweep: (p: number, a: number, b: number) =>
      `After pass ${p}: the plain lane has ${a} token${a === 1 ? '' : 's'}, the speculative lane has ${b}. Every verify pass emits at least one.`,
    captionTally: (x: string) =>
      `Six passes each: with the drafting tax paid, the speculative lane comes out at ×${x} the throughput — each verify pass rides the weight stream a plain pass pays for anyway. Now raise D, or make the text harder to guess.`,
    batchChip: 'batch = 1 — you are here',
    batchNote:
      'The free ride exists because at batch 1 the compute row above is nearly empty. At high batch the chip is already busy — servers switch speculation off.',
    rules: [
      'One pass ≈ 20 ms whether it scores 1 position or D + 1 — the weights stream through either way.',
      'If K of D drafts are accepted, the pass emits K + 1 tokens; the worst case still emits 1.',
      'Rejected positions add zero wall-clock — the only real extra cost is the drafting, ≈1 ms per draft token.',
      'Accept verdicts here are scripted (greedy, temperature-0-style); higher temperature lowers acceptance.',
    ],
    illustrativeCaption:
      "Illustrative trace on a synthetic millisecond axis — scripted accept/reject verdicts, not a real profiler capture, and not this tab: the browser's WebGPU decode loop is plain autoregressive.",
    footnote:
      "Anchor: this tab's plain WebGPU decode measures roughly 50 tok/s on an M-series Mac — about 20 ms per pass, the number the chat HUD shows (it varies by machine and run). The only measured speculative speedup in this project is ~1.06–1.46× (native Metal engine, Qwen3.5's default depth D = 1); the larger ratios this widget can show come from scripted, favorable acceptance — and, past D = 1, from engine-raised depths on top.",
    ariaLabel:
      'Illustrative GPU timeline comparing plain autoregressive decoding with speculative decoding. Top lane: six 20-millisecond forward passes, one token each. Bottom lane: each pass starts with a draft sliver of roughly 1 millisecond per draft token — about D milliseconds per cycle — and the verify pass is the same 20 milliseconds wide but emits one token plus every accepted draft, shown in emerald; rejected draft positions are shown struck through in red inside the same pass and add no extra time. A magnified panel shows that in both kinds of pass the memory row streams roughly 1.6 gigabytes of weights the whole time while the compute row sits mostly idle.',
  },
  zh: {
    title: '时间都花在了哪里——一条示意 GPU 时间线',
    play: '▶ 播放',
    pause: '❚❚ 暂停',
    step: '单步 ›',
    dLabel: '每步 draft 数 D：',
    dAria: (n: number) => `每步 draft ${n} 个 token（脚本化 trace）`,
    dNote: 'D = 1 是 Qwen3.5 自己的 MTP 深度；更大的 D 是推理引擎调高后的示意设置。',
    scenarioLabel: '文本有多好猜？',
    scenarioNames: { predictable: '好猜', typical: '一般', surprising: '难猜' },
    scenarioAria: (name: string, pct: number) => `${name}——单个 draft 位置的存活几率约 ${pct}%（脚本设定）`,
    scenarioHint: (acc: number, total: number, pct: number) =>
      `这里的脚本结果：${total} 个 draft 位置里有 ${acc} 个通过 verify——单个位置的几率约 ${pct}%，但一个位置只有在它前面的全部存活时才算数`,
    laneAName: '朴素自回归——每个 token 一次 ≈20 ms 的前向',
    laneBName: 'speculative——每个 draft token ≈1 ms + 一次 ≈20 ms 的 verify 前向',
    axisLabel: 'ms——示意时间轴，锚定在每次前向 ≈20 ms',
    passTok: (n: number) => `${n} tok`,
    zoomTitle: '把一次前向放大——这 ≈20 ms 究竟花在哪',
    zoomPlainLabel: '朴素前向——只给 1 个位置打分',
    zoomVerifyLabel: (n: number) => `verify 前向——在同一条权重流上给 ${n} 个位置打分`,
    memLabel: '访存',
    computeLabel: '算力',
    memBusyLabel: '流式读入 ~1.6 GB 的 bf16 权重——整个前向都在忙',
    computeIdleLabel: '大多在闲着——那点数学只是零头',
    equalNote: '两条一样宽是故意的：核验 D + 1 个位置仍然只是把权重过一遍，代价和产出一个 token 差不太多。',
    legendPass: 'verify 前向（把全部权重流一遍）',
    legendDraft: 'draft——MTP 头，每个 draft token ≈1 ms',
    legendAccepted: '被 accept 的 draft 位置',
    legendWasted: '浪费的位置——算完即弃',
    legendFreeTok: '白赚的 token（来自被 accept 的 draft）',
    legendAnywayTok: '这次前向本来就会产出的那个 token',
    readoutPlainTitle: '朴素那条',
    readoutPlainValue: '6 个 token / 120 ms ≈ 50 tok/s',
    readoutSpecTitle: 'speculative 那条',
    readoutSpecValue: (t: number, ms: number, r: number) => `${t} 个 token / ${ms} ms ≈ ${r} tok/s`,
    readoutSpeedupTitle: '加速比（这条脚本化 trace）',
    readoutSpeedupValue: (x: string) => `×${x}`,
    readoutMeasured: '真实测量：D = 1 时约 1.06–1.46×',
    readoutWaste: (n: number) => (
      <>
        {n} 个 draft 位置算完就被<span className="text-red-600 dark:text-red-400">丢掉</span>
        ——但它们搭的是反正要付钱的那几次前向
      </>
    ),
    readoutTax: (tax: number, total: number) => `草稿税：${total} ms 里的 ${tax} ms`,
    captionStart: '两块 GPU，同一份活。按播放——或单步——扫过播放头，看 token 一个个落地。',
    captionSweep: (p: number, a: number, b: number) =>
      `第 ${p} 次前向之后：朴素那条攒了 ${a} 个 token，speculative 那条攒了 ${b} 个。每次 verify 前向至少吐出一个。`,
    captionTally: (x: string) =>
      `同样六次前向：付完草稿税，speculative 那条的吞吐是 ×${x}——每次 verify 前向搭的正是朴素前向反正要付钱的那条权重流。现在把 D 调高，或把文本换难猜些。`,
    batchChip: 'batch = 1——你就在这里',
    batchNote:
      '这趟白车之所以存在，是因为 batch = 1 时上面那条算力行几乎是空的。batch 一大芯片本来就忙——服务器就会把 speculation 关掉。',
    rules: [
      '一次前向 ≈ 20 ms——不管给 1 个还是 D + 1 个位置打分，权重反正都要整个流一遍。',
      'D 个 draft 里有 K 个被 accept，这一步就吐出 K + 1 个 token；最差情形也有 1 个。',
      '被拒的位置不增加一毫秒墙钟时间——真正多花的只有 draft 本身：每个 draft token ≈1 ms。',
      '这里的 accept 判决是脚本化的（greedy、相当于 temperature 0）；temperature 越高，接受率越低。',
    ],
    illustrativeCaption:
      '示意 trace，时间轴是合成的——判决是脚本，不是真实的 profiler 抓取，也不是这个标签页：浏览器里的 WebGPU 解码循环就是朴素自回归。',
    footnote:
      '锚点：这个标签页的朴素 WebGPU decode 在 M 系列 Mac 上实测约 50 tok/s——每次前向约 20 ms，就是聊天 HUD 显示的那个数（随机器与批次波动）。本项目唯一实测过的 speculative 加速是 ~1.06–1.46×（原生 Metal 引擎，Qwen3.5 默认深度 D = 1）；这个小部件能摆出的更大倍数，来自脚本化的、偏乐观的接受率——在 D 大于 1 时，还叠加了引擎调高的深度。',
    ariaLabel:
      '示意 GPU 时间线，对比朴素自回归解码与 speculative 解码。上面一条：六次各 20 毫秒的前向，每次产出一个 token。下面一条：每次前向前多一条 draft 细条，每个 draft token 约 1 毫秒——每轮约 D 毫秒；verify 前向同样是 20 毫秒宽，但除了自己的一个 token，还吐出所有被 accept 的 draft（绿色）；被拒的 draft 位置以红色划掉画在同一次前向内部，不增加任何时间。放大面板显示：无论哪种前向，访存一行都在整趟流式读入约 1.6 GB 权重，而算力一行大多闲着。',
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

// ── SVG geometry ────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 324;
const X0 = 14;
const PX_PER_MS = 5; // sized so the LONGEST spec lane fits: tx(144) = 734, inside 760
const tx = (ms: number) => X0 + ms * PX_PER_MS; // trace x-mapping, shared by lanes, axis and playhead
const ZX0 = 96;
const ZW = 634;
const zx = (ms: number) => ZX0 + ms * (ZW / 20); // zoom x-mapping; BOTH strips span exactly zx(0)→zx(20)
// CRITICAL: both lanes' pass widths and both zoom strips derive from these
// shared consts, never duplicated literals — equal width IS the lesson, and a
// 1-px difference visually breaks the "costs about the same" claim.
const PASS_W = PASS_MS * PX_PER_MS - 2; // 98 — one pass, minus a 2-px hairline inset
// The draft sliver's width is D-dependent (D × 1 ms) — computed in the component.
const LANE_H = 26;
const LANE_A_Y = 24;
const LANE_A_CHIP_Y = 58;
const LANE_B_Y = 100;
const LANE_B_CHIP_Y = 134;
const CHIP = 9; // token chip side
const CHIP_PITCH = 11; // chip spacing inside a cluster (max 5 chips = 55 px < PASS_W)
const SLOT_H = 7; // accept/reject slot strip along the verify block's bottom edge
const SLOT_INSET = 4;
const SLOT_SPAN = PASS_W - SLOT_INSET * 2; // 90
const AXIS_Y = 150;
const AXIS_TICKS = [0, 20, 40, 60, 80, 100, 120, 140];
const ZOOM_TITLE_Y = 208;
const ZOOM_PLAIN_LABEL_Y = 222; // → memory row y228, compute row y245
const ZOOM_VERIFY_LABEL_Y = 280; // → memory row y286, compute row y303
const ROW_H = 15;
const ROW_GAP = 2;

export function SpecDecodeFlamechart() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [dIdx, setDIdx] = React.useState(0); // D = dIdx + 1; default 1 = Qwen3.5's own depth
  const [scenario, setScenario] = React.useState<ScenarioId>('typical');
  // Under reduced motion: pin to the completed tally frame, never auto-play.
  const [frame, setFrame] = React.useState(reducedMotion ? TOTAL_FRAMES - 1 : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  // If the reduced-motion preference flips on WHILE the animation runs, honor
  // the same contract as the initializers: pin to the completed tally, stop.
  // (The useState initializers above only ran once, on mount.)
  React.useEffect(() => {
    if (!reducedMotion) return;
    setFrame(TOTAL_FRAMES - 1);
    setPlaying(false);
  }, [reducedMotion]);

  const d = dIdx + 1;
  const ks = ACCEPTS[scenario][dIdx];
  // The head drafts ONE TOKEN AT A TIME (each guess feeds the next), so every
  // cycle pays D × 1 ms — the sliver, the clock and the readout all share these.
  const draftMsPerCycle = d * DRAFT_MS;
  const specStepMs = PASS_MS + draftMsPerCycle; // 21 / 22 / 23 / 24
  const specTotalMs = N_PASSES * specStepMs; // 126 / 132 / 138 / 144
  const draftW = draftMsPerCycle * PX_PER_MS - 1; // the D × 1 ms draft sliver
  // The readout counts the drawn trace — the same ACCEPTS row drives the bars,
  // slots, chips, zoom slivers AND this math, so picture and numbers can never diverge.
  const specTokens = ks.reduce((s, k) => s + k + 1, 0);
  const accepted = specTokens - N_PASSES; // Σ K — draft positions that survived verify
  const wasted = ks.reduce((s, k) => s + (d - k), 0);
  const specRate = Math.round((specTokens * 1000) / specTotalMs);
  const speedup = (specTokens / specTotalMs / (N_PASSES / PLAIN_TOTAL_MS)).toFixed(2);

  const passesDone = Math.min(frame, N_PASSES);
  const isTally = frame === TOTAL_FRAMES - 1;
  const specCount = ks.slice(0, passesDone).reduce((s, k) => s + k + 1, 0);

  // ONE shared reset used by BOTH toggles: re-sweep under the new setting;
  // reduced-motion users stay on the completed tally view.
  const resetFrame = () => {
    setFrame(reducedMotion ? TOTAL_FRAMES - 1 : 0);
    setPlaying(!reducedMotion);
  };
  const step = () => {
    setPlaying(false);
    setFrame((x) => (x + 1) % TOTAL_FRAMES);
  };
  const pickD = (i: number) => {
    setDIdx(i);
    resetFrame();
  };
  const pickScenario = (id: ScenarioId) => {
    setScenario(id);
    resetFrame();
  };

  let caption: React.ReactNode;
  if (frame === 0) caption = copy.captionStart;
  else if (isTally) caption = copy.captionTally(speedup);
  else caption = copy.captionSweep(passesDone, passesDone, specCount);

  // Zoom-panel compute slivers, live-updated from the CURRENT trace's pass 0.
  const k0 = ks[0];
  const plainSlivers = [{ cx: zx(PASS_MS / 2), fill: 'var(--primary)' }];
  const verifySlivers = Array.from({ length: d + 1 }, (_, j) => ({
    cx: zx((PASS_MS * (j + 0.5)) / (d + 1)),
    fill: j <= k0 ? EMERALD : RED,
  }));

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* [1] Header: title + Play/Step (both removed entirely under reduced motion) */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1">
          {!reducedMotion ? (
            <>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {/* [2] Controls: D toggle · scenario toggle · batch chip, hints beneath */}
      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <span className="hidden sm:inline">{copy.dLabel}</span>
            <span aria-hidden className="sm:hidden">
              D:
            </span>
            <div className="flex overflow-hidden rounded border border-border">
              {[0, 1, 2, 3].map((i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => pickD(i)}
                  aria-pressed={dIdx === i}
                  aria-label={copy.dAria(i + 1)}
                  className={[
                    'px-2.5 py-1 font-mono text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    i > 0 ? 'border-l border-border' : '',
                    dIdx === i ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                  ].join(' ')}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <span className="hidden sm:inline">{copy.scenarioLabel}</span>
            <div className="flex overflow-hidden rounded border border-border">
              {SCENARIO_IDS.map((id, i) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => pickScenario(id)}
                  aria-pressed={scenario === id}
                  aria-label={copy.scenarioAria(copy.scenarioNames[id], SCENARIO_ODDS[id])}
                  className={[
                    'px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    i > 0 ? 'border-l border-border' : '',
                    scenario === id ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                  ].join(' ')}
                >
                  {copy.scenarioNames[id]}
                </button>
              ))}
            </div>
          </div>
          <span className="inline-flex items-center rounded border border-border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
            {copy.batchChip}
          </span>
        </div>
        <p className="text-[10px] text-muted-foreground">
          {copy.scenarioHint(accepted, N_PASSES * d, SCENARIO_ODDS[scenario])}
        </p>
        <p className="text-[10px] text-muted-foreground">{copy.dNote}</p>
      </div>

      {/* [3] The trace + zoom panel */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* Lane A — plain autoregressive, 0–120 ms */}
        <text x={X0} y={16} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.laneAName}
        </text>
        {Array.from({ length: N_PASSES }, (_, i) => (
          <LanePass
            key={`a${i}`}
            x={tx(PASS_MS * i) + 1}
            y={LANE_A_Y}
            variant="plain"
            label={copy.passTok(1)}
            emphasized={!isTally && i === passesDone - 1}
          />
        ))}
        {Array.from({ length: N_PASSES }, (_, i) => (
          <TokenChips
            key={`ac${i}`}
            rightX={tx(PASS_MS * (i + 1)) - 5}
            y={LANE_A_CHIP_Y}
            emeraldCount={0}
            visible={i < passesDone}
            reducedMotion={reducedMotion}
          />
        ))}

        {/* Lane B — speculative: D×1 ms draft sliver + verify pass, 0–(120+6·D) ms */}
        <text x={X0} y={92} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.laneBName}
        </text>
        {ks.map((k, i) => (
          <g key={`b${i}`}>
            <rect
              x={tx(specStepMs * i) + 0.5}
              y={LANE_B_Y}
              width={draftW}
              height={LANE_H}
              fill="var(--primary)"
              fillOpacity={0.5}
            />
            <LanePass
              x={tx(specStepMs * i + draftMsPerCycle) + 1}
              y={LANE_B_Y}
              variant="verify"
              label={copy.passTok(k + 1)}
              emphasized={!isTally && i === passesDone - 1}
              k={k}
              depth={d}
            />
            <TokenChips
              rightX={tx(specStepMs * (i + 1)) - 4}
              y={LANE_B_CHIP_Y}
              emeraldCount={k}
              visible={i < passesDone}
              reducedMotion={reducedMotion}
            />
          </g>
        ))}

        {/* Shared ms axis — a FIXED ruler out to the longest cell (144 ms at
            D = 4), so the spec lane visibly grows against it as D rises */}
        <line x1={tx(0)} y1={AXIS_Y} x2={tx(MAX_SPEC_TOTAL_MS)} y2={AXIS_Y} stroke="var(--border)" strokeWidth={1} />
        {AXIS_TICKS.map((t) => (
          <g key={t}>
            <line x1={tx(t)} y1={AXIS_Y} x2={tx(t)} y2={AXIS_Y + 5} stroke="var(--border)" strokeWidth={1} />
            <text
              x={tx(t)}
              y={AXIS_Y + 18}
              textAnchor="middle"
              style={{ fill: 'var(--muted-foreground)' }}
              className="font-mono text-[10px]"
            >
              {t}
            </text>
          </g>
        ))}
        <text
          x={tx(MAX_SPEC_TOTAL_MS)}
          y={AXIS_Y + 34}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.axisLabel}
        </text>

        {/* Zoom panel — one pass of each kind, magnified to ~31.7 px/ms */}
        <text x={X0} y={ZOOM_TITLE_Y} style={{ fill: 'var(--foreground)' }} className="text-[11px] font-medium">
          {copy.zoomTitle}
        </text>
        <ZoomStrip label={copy.zoomPlainLabel} y={ZOOM_PLAIN_LABEL_Y} slivers={plainSlivers} showIdleLabel />
        <ZoomStrip label={copy.zoomVerifyLabel(d + 1)} y={ZOOM_VERIFY_LABEL_Y} slivers={verifySlivers} />

        {/* Playhead — tracks the spec lane's clock ((20+D)·i ≥ 20·i, so lane A is always behind it) */}
        <g
          aria-hidden
          style={{
            transform: `translateX(${specStepMs * passesDone * PX_PER_MS}px)`,
            transition: reducedMotion ? undefined : 'transform 1100ms cubic-bezier(0.4, 0, 0.2, 1)',
            opacity: frame === TOTAL_FRAMES - 1 ? 0 : 1,
          }}
        >
          <polygon points={`${X0 - 4},13 ${X0 + 4},13 ${X0},20`} fill="var(--foreground)" />
          <line x1={X0} y1={20} x2={X0} y2={158} stroke="var(--foreground)" strokeWidth={1} />
        </g>
      </svg>

      {/* [4] The equal-width claim, in words — "about the same", never "exactly" */}
      <p className="text-[13px] text-foreground/90">{copy.equalNote}</p>

      {/* [5] Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-muted-foreground">
        <LegendItem
          swatch={<span aria-hidden className="h-2.5 w-4 rounded-sm border border-primary bg-primary/15" />}
          label={copy.legendPass}
        />
        <LegendItem
          swatch={
            <span aria-hidden className="h-2.5 w-4 rounded-sm" style={{ background: 'var(--primary)', opacity: 0.5 }} />
          }
          label={copy.legendDraft}
        />
        <LegendItem
          swatch={<span aria-hidden className="h-2.5 w-3 rounded-sm" style={{ background: EMERALD, opacity: 0.9 }} />}
          label={copy.legendAccepted}
        />
        <LegendItem
          swatch={
            <span
              aria-hidden
              className="h-2.5 w-3 rounded-sm"
              style={{
                border: `1.5px solid ${RED}`,
                background: `linear-gradient(to top right, transparent 44%, ${RED} 46%, ${RED} 54%, transparent 56%)`,
              }}
            />
          }
          label={copy.legendWasted}
        />
        <LegendItem
          swatch={
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-sm"
              style={{ border: `1.5px solid ${EMERALD}`, background: 'rgba(16,185,129,0.14)' }}
            />
          }
          label={copy.legendFreeTok}
        />
        <LegendItem
          swatch={<span aria-hidden className="h-2.5 w-2.5 rounded-sm border border-border bg-card" />}
          label={copy.legendAnywayTok}
        />
      </div>

      {/* [6] Per-frame caption (reserved height, no layout shift) */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">{caption}</p>

      {/* [7] Readout — counted from the same ACCEPTS row that drew the picture */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        <div className="rounded border border-border bg-card/40 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutPlainTitle}</div>
          <div className="font-mono text-[13px] text-foreground/90">{copy.readoutPlainValue}</div>
        </div>
        <div className="rounded border border-border bg-card/40 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutSpecTitle}</div>
          <div className="font-mono text-[13px] text-foreground/90">
            {copy.readoutSpecValue(specTokens, specTotalMs, specRate)}
          </div>
        </div>
        <div className="rounded border border-border bg-card/40 p-2" aria-live="polite">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutSpeedupTitle}</div>
          <div className="font-mono text-[13px] font-semibold text-primary">{copy.readoutSpeedupValue(speedup)}</div>
          <div className="text-[10px] text-muted-foreground">{copy.readoutMeasured}</div>
        </div>
      </div>

      {/* [8] Waste + drafting tax */}
      <p className="text-[12px] text-foreground/90">
        {copy.readoutWaste(wasted)}{' '}
        <span className="text-muted-foreground">· {copy.readoutTax(N_PASSES * draftMsPerCycle, specTotalMs)}</span>
      </p>

      {/* [9] Rules box */}
      <div className="space-y-1 rounded border border-border bg-muted/40 p-2.5">
        {copy.rules.map((r) => (
          <div key={r} className="flex gap-2 text-[12px] text-foreground/90">
            <span aria-hidden className="select-none text-primary">
              ·
            </span>
            <span>{r}</span>
          </div>
        ))}
      </div>

      {/* [10] Low-batch caveat — unconditional */}
      <p className="text-[11px] text-muted-foreground">{copy.batchNote}</p>

      {/* [11] + [12] Honesty strings — unconditional, never gated on state */}
      <p className="text-[10px] italic text-muted-foreground/70">{copy.illustrativeCaption}</p>
      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}

/** One pass block in a lane: a plain pass, or a verify pass carrying its accept/reject slot strip. */
function LanePass({
  x,
  y,
  variant,
  label,
  emphasized,
  k = 0,
  depth = 0,
}: {
  x: number;
  y: number;
  variant: 'plain' | 'verify';
  label: string;
  emphasized: boolean;
  k?: number;
  depth?: number;
}) {
  const isVerify = variant === 'verify';
  const slotY = y + LANE_H - SLOT_H - 3;
  const pitch = SLOT_SPAN / (depth + 1);
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={PASS_W}
        height={LANE_H}
        rx={3}
        style={{
          fill: isVerify ? 'var(--primary)' : 'var(--card)',
          fillOpacity: isVerify ? 0.16 : 1,
          stroke: isVerify ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={emphasized ? 2 : 1.5}
      />
      <text
        x={x + PASS_W / 2}
        y={isVerify ? y + 12 : y + LANE_H / 2 + 4}
        textAnchor="middle"
        style={{ fill: isVerify ? 'var(--foreground)' : 'var(--muted-foreground)' }}
        className="font-mono text-[11px]"
      >
        {label}
      </text>
      {/* D+1 scored POSITIONS (not time slices — the zoom panel disambiguates).
          Rejects are struck out IN PLACE: a reject never widens the pass. */}
      {isVerify
        ? Array.from({ length: depth + 1 }, (_, j) => {
            const sx = x + SLOT_INSET + j * pitch;
            const sw = pitch - 2;
            return j <= k ? (
              <rect key={j} x={sx} y={slotY} width={sw} height={SLOT_H} fill={EMERALD} fillOpacity={0.9} />
            ) : (
              <g key={j}>
                <rect
                  x={sx}
                  y={slotY}
                  width={sw}
                  height={SLOT_H}
                  fill={RED}
                  fillOpacity={0.14}
                  stroke={RED}
                  strokeWidth={1}
                />
                <line x1={sx} y1={slotY} x2={sx + sw} y2={slotY + SLOT_H} stroke={RED} strokeWidth={1} />
              </g>
            );
          })
        : null}
    </g>
  );
}

/** Right-aligned token-chip cluster for one pass: K emerald "free" chips + 1 neutral anyway-chip. */
function TokenChips({
  rightX,
  y,
  emeraldCount,
  visible,
  reducedMotion,
}: {
  rightX: number;
  y: number;
  emeraldCount: number;
  visible: boolean;
  reducedMotion: boolean;
}) {
  const total = emeraldCount + 1;
  return (
    <g style={{ opacity: visible ? 1 : 0, transition: reducedMotion ? undefined : 'opacity 300ms' }}>
      {Array.from({ length: total }, (_, c) => {
        const isNeutral = c === total - 1; // the rightmost chip: the token the pass would emit anyway
        return (
          <rect
            key={c}
            x={rightX - CHIP - (total - 1 - c) * CHIP_PITCH}
            y={y}
            width={CHIP}
            height={CHIP}
            rx={2}
            style={{
              fill: isNeutral ? 'var(--card)' : EMERALD,
              stroke: isNeutral ? 'var(--border)' : EMERALD,
            }}
            fillOpacity={isNeutral ? 1 : 0.14}
            strokeWidth={1}
          />
        );
      })}
    </g>
  );
}

/** One magnified pass in the zoom panel: a wall-to-wall memory row + a mostly-idle compute row. */
function ZoomStrip({
  label,
  y,
  slivers,
  showIdleLabel = false,
}: {
  label: string;
  y: number;
  slivers: { cx: number; fill: string }[];
  showIdleLabel?: boolean;
}) {
  const copy = COPY[useLocale()];
  const memY = y + 6;
  const compY = memY + ROW_H + ROW_GAP;
  return (
    <g>
      <text x={ZX0} y={y} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {label}
      </text>
      {/* memory row — busy the whole pass */}
      <text x={90} y={memY + 11} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {copy.memLabel}
      </text>
      <rect
        x={zx(0)}
        y={memY}
        width={ZW}
        height={ROW_H}
        fill="var(--primary)"
        fillOpacity={0.18}
        stroke="var(--primary)"
        strokeWidth={1}
      />
      <text
        x={ZX0 + ZW / 2}
        y={memY + 11}
        textAnchor="middle"
        style={{ fill: 'var(--foreground)' }}
        className="text-[11px]"
      >
        {copy.memBusyLabel}
      </text>
      {/* compute row — idle box + honest ~1%-of-the-pass slivers */}
      <text x={90} y={compY + 11} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {copy.computeLabel}
      </text>
      <rect
        x={zx(0)}
        y={compY}
        width={ZW}
        height={ROW_H}
        style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
        strokeWidth={1}
      />
      {slivers.map((s, i) => (
        <rect key={i} x={s.cx - 3} y={compY + 2} width={6} height={ROW_H - 4} fill={s.fill} />
      ))}
      {showIdleLabel ? (
        <text
          x={zx(20) - 6}
          y={compY + 11}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.computeIdleLabel}
        </text>
      ) : null}
    </g>
  );
}

/** One legend entry: a decorative swatch + its text label. */
function LegendItem({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {swatch}
      <span>{label}</span>
    </span>
  );
}
