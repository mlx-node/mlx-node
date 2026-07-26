import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import {
  DASH,
  DIM,
  DiagramFrame,
  EMERALD,
  FlowDots,
  FrameLabel,
  HatchDefs,
  hatchFill,
  PanBox,
  PanelFrame,
  RED,
  RX,
  SW,
  useStepPlayer,
} from '../motion';

/**
 * FlashTilingDiagram — chapter 4 deep-dive "Flash Attention", the centerpiece.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It makes the FlashAttention fused-tiling loop visible. The core idea — tile
 * the attention, keep the score block on-chip, and fold it into a streaming
 * ("online") softmax so the full N×N score matrix NEVER lands in slow HBM — is
 * FlashAttention-1's contribution (Dao et al., 2022, arXiv 2205.14135). The
 * naïve recipe parks the full N×N matrix in HBM and drags it across the bus
 * ~4 times (see the MemoryWallDiagram). Flash never builds that matrix at all.
 *
 * The loop ORDER drawn here — outer over query tiles, inner over key/value
 * tiles, each O_i fully accumulated on-chip and written back exactly once — is
 * the modern schedule (FA2 swapped FA1's original outer-over-K/V order to avoid
 * re-reading/re-writing O across passes; see the Flash2ParallelismDiagram). We
 * show that order because it is the one production kernels use, and the
 * never-materialize-N×N point this widget makes holds for both:
 *
 *   OUTER loop over query tiles i:
 *     load Q_i  →  fast on-chip SRAM (SRAM = a tiny scratchpad next to compute;
 *                  HBM = the big-but-slow main memory)
 *     INNER loop over key/value tiles j:
 *       load K_j, V_j → SRAM
 *       S_ij = Q_i · K_jᵀ        ← a SMALL score tile, computed IN SRAM
 *       update running m, ℓ, O_i  ← streaming ("online") softmax
 *       discard S_ij             ← it lives and dies in SRAM, never hits HBM
 *     write the finished O_i tile back to HBM
 *
 * The counter drives the point home: HBM score-matrix writes stay at 0 the
 * whole way through — the full N×N S never exists in slow memory.
 *
 * Animation model: `useStepPlayer` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval` + private `usePrefersReducedMotion` + private
 * play/step buttons. Same 28-frame script, same 1500ms pace, same
 * reduced-motion still (`restFrame` is pinned to STATIC_FRAME — the
 * representative mid-inner-loop frame this widget's copy describes — instead of
 * the kit's default last frame).
 *
 * ── THE PALETTE ───────────────────────────────────────────────────────────────
 *
 * `skin` allows exactly two hues and says a hue is a box's PERMANENT IDENTITY,
 * never a state. The two identities this diagram is *about* are the two memory
 * tiers, so they take the two hues and never change:
 *
 *   RED      HBM — the big, far-away, expensive-to-touch store. Red in EVERY
 *            frame; the lesson is that the scores never go there.
 *   EMERALD  SRAM — the on-chip scratchpad where every tile is kept and the
 *            softmax is finished. Emerald in EVERY frame, and inherited by the
 *            things that come to life inside it (resident slots, the score tile,
 *            the accumulator, and an O tile once it is stored).
 *
 * State rides the orthogonal channels instead:
 *   hatch (`hatchFill`)  this box is busy THIS step — the S tile being computed,
 *                        the accumulator folding it in, an O tile landing
 *   dash + DIM           this slot is empty / checked out / not real yet
 *   FLOW_BLUE dots       bytes crossing the bus right now (exempt from the
 *                        two-hue rule; it is never a box colour)
 *
 * Before this change every one of those states was painted `var(--primary)`
 * (book-pink), i.e. hue-as-state — the one thing this palette cannot do. No
 * neighbour widget depends on the old colours: nothing here was ever a shared
 * constant, and the sibling widgets on this page (FlashTrafficDiagram,
 * GpuHierarchyDiagram) carry no HBM/SRAM colour contract of their own.
 */

const N_Q_TILES = 2; // query row-blocks
const N_KV_TILES = 3; // key/value blocks

// ── Frame script. Each frame is one snapshot of the fused tiling loop. ────────
type Phase = 'loadQ' | 'loadKV' | 'score' | 'update' | 'discard' | 'writeO';

type Frame = {
  /** Current query tile (outer loop index). */
  i: number;
  /** Current key/value tile (inner loop index), or -1 when between inner loops. */
  j: number;
  phase: Phase;
  /** Q tile sits in SRAM this frame. */
  qInSram: boolean;
  /** K_j / V_j sit in SRAM this frame. */
  kvInSram: boolean;
  /** The small S_ij score tile is alive in SRAM this frame. */
  sAlive: boolean;
  /** Pulse the streaming-softmax accumulator (m, ℓ, O_i). */
  accumulate: boolean;
  /** O_i is being written back to HBM this frame (and which i). */
  writeBack: boolean;
  /** Q tiles whose O_i is finished and parked in HBM. */
  doneO: number;
};

function buildFrames(): Frame[] {
  const frames: Frame[] = [];
  for (let i = 0; i < N_Q_TILES; i++) {
    const doneBefore = i; // O tiles finished before this outer pass
    // 1 · load Q_i into SRAM
    frames.push({
      i,
      j: -1,
      phase: 'loadQ',
      qInSram: true,
      kvInSram: false,
      sAlive: false,
      accumulate: false,
      writeBack: false,
      doneO: doneBefore,
    });
    // 2 · inner loop over K/V tiles
    for (let j = 0; j < N_KV_TILES; j++) {
      frames.push({
        i,
        j,
        phase: 'loadKV',
        qInSram: true,
        kvInSram: true,
        sAlive: false,
        accumulate: false,
        writeBack: false,
        doneO: doneBefore,
      });
      frames.push({
        i,
        j,
        phase: 'score',
        qInSram: true,
        kvInSram: true,
        sAlive: true,
        accumulate: false,
        writeBack: false,
        doneO: doneBefore,
      });
      frames.push({
        i,
        j,
        phase: 'update',
        qInSram: true,
        kvInSram: true,
        sAlive: true,
        accumulate: true,
        writeBack: false,
        doneO: doneBefore,
      });
      frames.push({
        i,
        j,
        phase: 'discard',
        qInSram: true,
        kvInSram: false,
        sAlive: false,
        accumulate: false,
        writeBack: false,
        doneO: doneBefore,
      });
    }
    // 3 · write finished O_i back to HBM
    frames.push({
      i,
      j: -1,
      phase: 'writeO',
      qInSram: false,
      kvInSram: false,
      sAlive: false,
      accumulate: false,
      writeBack: true,
      doneO: doneBefore + 1,
    });
  }
  return frames;
}

// Subscript helper for compact tile labels.
function sub(n: number): string {
  const subs = '₀₁₂₃₄₅₆₇₈₉';
  return String(n)
    .split('')
    .map((d) => subs[Number(d)] ?? d)
    .join('');
}

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
//
// Play / Pause / Step are NOT here any more: those verbs belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word.
const COPY = {
  en: {
    captions: {
      loadQ: (i: number, _j: number) => (
        <>
          <strong>Outer loop, tile {i}.</strong> Load query tile{' '}
          <span className="font-mono">Q{sub(i)}</span> from HBM into fast on-chip SRAM. We will reuse it across every K/V
          tile.
        </>
      ),
      loadKV: (_i: number, j: number) => (
        <>
          <strong>Inner loop, tile {j}.</strong> Stream in key/value tile{' '}
          <span className="font-mono">
            K{sub(j)},V{sub(j)}
          </span>{' '}
          from HBM into SRAM. Just this slice — not the whole sequence.
        </>
      ),
      score: (i: number, j: number) => (
        <>
          Compute the <strong>small score tile</strong>{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)} = Q{sub(i)}·K{sub(j)}ᵀ
          </span>{' '}
          <strong>inside SRAM</strong>. It is tiny — one Q tile against one K tile, never the full N×N.
        </>
      ),
      update: (i: number, j: number) => (
        <>
          Fold{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)}
          </span>{' '}
          into the running{' '}
          <strong>streaming softmax</strong>: update the max <span className="font-mono">m</span>, the sum{' '}
          <span className="font-mono">ℓ</span>, and the output accumulator{' '}
          <span className="font-mono">O{sub(i)}</span> — all in SRAM.
        </>
      ),
      discard: (i: number, j: number) => (
        <>
          <strong>Discard</strong>{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)}
          </span>
          . It lived and died in SRAM — it never crossed the bus to HBM.{' '}
          <span style={{ color: EMERALD }}>HBM score writes still 0.</span>
        </>
      ),
      writeO: (i: number, _j: number) => (
        <>
          Inner loop done. The streaming softmax has the exact answer for this tile — write{' '}
          <strong>
            only <span className="font-mono">O{sub(i)}</span>
          </strong>{' '}
          back to HBM. The scores never went there.
        </>
      ),
    },
    title: 'Flash tiling — the score tile lives and dies in SRAM',
    ariaLabel:
      'Diagram of the FlashAttention tiling loop: query tiles loaded from slow HBM into fast on-chip SRAM, key and value tiles streamed in, a small score tile computed and discarded inside SRAM via streaming softmax, and only the finished output tile written back to HBM — the full N by N score matrix never reaches HBM.',
    hbmTitle: 'HBM · main memory',
    hbmHint: 'huge — but far away and slow',
    qTiles: 'Q tiles',
    kTiles: 'K tiles',
    vTiles: 'V tiles',
    oOutput: 'O output',
    busWrite: '← write Oᵢ',
    busLoad: 'load tiles →',
    sramTitle: 'SRAM · on-chip scratchpad',
    sramHint: 'tiny — but ~10× faster · the whole op fuses here',
    streamingSoftmax: 'streaming softmax',
    running: (i: number) => `running m, ℓ, O${sub(i)}`,
    sramFootnote: 'full N×N score matrix never built here or in HBM',
    loopPosition: 'loop position',
    loopTiles: (q: number, kv: number) => `(${q} query tiles × ${kv} K/V tiles)`,
    fusedLine: 'One dispatch, fused: load → score → update → discard, then write Oᵢ.',
    hbmWritesLabel: 'HBM score-matrix writes:',
    naiveNote: 'naïve recipe crosses the slow bus ~4× to move N×N — flash crosses it 0× for scores.',
    reducedNote:
      'Reduced motion: the loop never plays on its own. It opens on a representative frame — the score tile S₀₁ alive inside SRAM, with HBM score writes at 0 — and Step walks the rest of the loop by hand; Reset returns to that frame.',
    scheduleNote:
      "Schedule shown: queries on the outer loop, keys/values streamed on the inner — the modern (FA2-style) order. v1's contribution is the invariant either order keeps: the N×N score matrix never reaches HBM.",
    discarded: 'discarded',
  },
  zh: {
    captions: {
      loadQ: (i: number, _j: number) => (
        <>
          <strong>外层循环，块 {i}。</strong>把 query 块 <span className="font-mono">Q{sub(i)}</span> 从 HBM
          加载进快速的片上 SRAM。它会在每个 K/V 块上被复用。
        </>
      ),
      loadKV: (_i: number, j: number) => (
        <>
          <strong>内层循环，块 {j}。</strong>把 key/value 块{' '}
          <span className="font-mono">
            K{sub(j)},V{sub(j)}
          </span>{' '}
          从 HBM 流式载入 SRAM。只取这一小片——不是整条序列。
        </>
      ),
      score: (i: number, j: number) => (
        <>
          在 <strong>SRAM 内部</strong>计算<strong>小分数块</strong>{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)} = Q{sub(i)}·K{sub(j)}ᵀ
          </span>
          。它很小——一个 Q 块对一个 K 块，永远不是完整的 N×N。
        </>
      ),
      update: (i: number, j: number) => (
        <>
          把{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)}
          </span>{' '}
          并入滚动的<strong>流式 softmax</strong>：更新最大值 <span className="font-mono">m</span>、和{' '}
          <span className="font-mono">ℓ</span>，以及输出累加器 <span className="font-mono">O{sub(i)}</span>
          ——全部在 SRAM 里。
        </>
      ),
      discard: (i: number, j: number) => (
        <>
          <strong>丢弃</strong>{' '}
          <span className="font-mono">
            S{sub(i)}
            {sub(j)}
          </span>
          。它在 SRAM 中生灭——从未跨过总线去 HBM。<span style={{ color: EMERALD }}>HBM 分数写入仍为 0。</span>
        </>
      ),
      writeO: (i: number, _j: number) => (
        <>
          内层循环结束。流式 softmax 已得到这一块的精确答案——只把{' '}
          <strong>
            <span className="font-mono">O{sub(i)}</span>
          </strong>{' '}
          写回 HBM。分数从未去过那里。
        </>
      ),
    },
    title: 'Flash 分块——分数块在 SRAM 中生灭',
    ariaLabel:
      'FlashAttention 分块循环示意图：query 块从慢速 HBM 加载进快速片上 SRAM，key 和 value 块流式载入，小分数块在 SRAM 内通过流式 softmax 计算并丢弃，只有算完的输出块写回 HBM——完整的 N×N 分数矩阵从不进入 HBM。',
    hbmTitle: 'HBM · 主内存',
    hbmHint: '巨大——但遥远而慢',
    qTiles: 'Q 块',
    kTiles: 'K 块',
    vTiles: 'V 块',
    oOutput: 'O 输出',
    busWrite: '← 写回 Oᵢ',
    busLoad: '加载块 →',
    sramTitle: 'SRAM · 片上暂存区',
    sramHint: '极小——但快约 10×，整个运算在此融合',
    streamingSoftmax: '流式 softmax',
    running: (i: number) => `滚动 m, ℓ, O${sub(i)}`,
    sramFootnote: '完整的 N×N 分数矩阵从未在此处或 HBM 中构造',
    loopPosition: '循环位置',
    loopTiles: (q: number, kv: number) => `（${q} 个 query 块 × ${kv} 个 K/V 块）`,
    fusedLine: '一次分发，全融合：加载 → 算分 → 更新 → 丢弃，最后写回 Oᵢ。',
    hbmWritesLabel: 'HBM 分数矩阵写入：',
    naiveNote: '朴素做法让 N×N 在慢速总线上往返约 4 趟——flash 的分数过桥次数是 0。',
    reducedNote:
      '减少动态效果：循环不会自动播放。它从一帧代表性画面开始——分数块 S₀₁ 存活在 SRAM 中，HBM 分数写入为 0——用「单步」可以手动走完整个循环，「重置」回到这一帧。',
    scheduleNote:
      '图中调度：query 在外层循环，key/value 在内层流式扫过——现代（FA2 风格）顺序。v1 的贡献是两种顺序共守的不变量：N×N 分数矩阵永不触及 HBM。',
    discarded: '已丢弃',
  },
} as const;

const FRAMES = buildFrames();
const TOTAL_FRAMES = FRAMES.length;
const FRAME_MS = 1500;

// The representative static frame under reduced motion: mid inner-loop, i=0,
// j=1, the S tile alive in SRAM (the "score" phase).
const STATIC_FRAME = FRAMES.findIndex((f) => f.phase === 'score' && f.i === 0 && f.j === 1);
// The deterministic initial frame (server === client): frame 0, load Q_0.
const INITIAL_FRAME = 0;

// ── SVG geometry ────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 330;

const HBM = { x: 18, y: 40, w: 318, h: 270 };
const SRAM = { x: 430, y: 40, w: 312, h: 270 };

// Inside HBM: Q tiles (left column), K tiles, V tiles, O output column.
const Q_TILE = { w: 40, h: 44, gx: 60, gy: 70 };
const KV_TILE = { w: 30, h: 30 };

export function FlashTilingDiagram() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, {
    frameMs: FRAME_MS,
    initialFrame: INITIAL_FRAME,
    // NOT the kit's default (last frame): the still this widget's copy
    // describes is mid inner-loop, with S alive in SRAM.
    restFrame: STATIC_FRAME >= 0 ? STATIC_FRAME : 0,
  });
  const f = FRAMES[player.frame];

  const caption = copy.captions[f.phase](f.i, f.j);

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the beat changes. Built by mapping the
  // frame script itself, which makes it exhaustive BY CONSTRUCTION: the visible
  // caption is exactly `copy.captions[FRAMES[frame].phase](i, j)` and `frame`
  // never leaves [0, FRAMES.length). Every arm of the six-way phase switch and
  // every (i, j) the builders interpolate is therefore covered.
  const captions: React.ReactNode[] = FRAMES.map((fr) => copy.captions[fr.phase](fr.i, fr.j));

  const ease = player.reducedMotion ? undefined : 'opacity 400ms ease';

  // Counters. HBM score-matrix writes are always zero — that is the headline.
  const hbmScoreWrites = 0;

  return (
    <DiagramFrame
      title={copy.title}
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      // Two footnotes, two ELEMENTS. `note`'s container is `space-y-2`, which
      // is a direct-CHILD rule — bare text nodes in a fragment are not children
      // it can space, so the two sentences used to render as one run-on line.
      note={
        player.reducedMotion ? (
          <>
            <p>{copy.reducedNote}</p>
            <p>{copy.scheduleNote}</p>
          </>
        ) : (
          <p>{copy.scheduleNote}</p>
        )
      }
      // 28 frames at 1500ms is a 42-second sweep — long enough that a reader
      // who wants the top again should not have to wait for the wrap. Under
      // reduced motion `StepControls` shows Reset unconditionally, and
      // `reducedNote` above points at it.
      showReset
    >
      {/* The `min-w` below is the reason this svg — and ONLY this svg — is
          wrapped in `PanBox`. A bare `w-full` svg does not reflow in a narrow
          column — it SCALES the whole canvas, text included, so at a 375px
          viewport (a 320-unit column) the HBM/SRAM titles land near 4.2 CSS px
          and this diagram stops being readable at all. The floor stops the
          shrink; `PanBox` pans to reach the rest.

          8px legibility floor at the smallest size a reader must read:
          8 * 760 / 9 = 675.6 → 676. That smallest size is the 9px `discarded`
          tag over the score tile. It is a real word, not a superscript — it is
          the beat where the whole lesson lands ("it lived and died in SRAM") —
          so it does not get to duck under the floor, and 676 is still well
          inside the 760 viewBox, so nothing needs its fontSize raised. At 676
          the 9px tag renders at 8.0px, the 10px column headers and frame chips
          at 8.9px, and the 11px tile labels at 9.8px.

          The floor belongs to the svg ALONE, which is why `PanBox` wraps the
          svg and nothing else: the counters and loop-position row below stay
          DiagramFrame's own children, so they size to the VISIBLE width, keep
          the frame's `space-y-4` gaps, and stay reachable without panning. */}
      <PanBox locale={locale}>
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full min-w-[676px]" role="img" aria-label={copy.ariaLabel}>
          <HatchDefs />

          {/* ── HBM stage (big, slow main memory). RED in every frame: that is
              its identity, not a state — the whole lesson is that the scores
              never come here. ── */}
          <PanelFrame x={HBM.x} y={HBM.y} w={HBM.w} h={HBM.h} tone="red" fill="none" />
          <FrameLabel x={HBM.x + 16} y={HBM.y} label={copy.hbmTitle} fill={RED} />
          <text x={HBM.x + 16} y={HBM.y + 24} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.hbmHint}
          </text>

          {/* Q tiles column */}
          <text x={HBM.x + 22} y={HBM.y + 74} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
            {copy.qTiles}
          </text>
          {Array.from({ length: N_Q_TILES }, (_, qi) => {
            const x = HBM.x + 22;
            const y = HBM.y + 82 + qi * (Q_TILE.h + 10);
            const inSram = f.qInSram && qi === f.i;
            return (
              <TileCell
                key={`q-${qi}`}
                x={x}
                y={y}
                w={Q_TILE.w}
                h={Q_TILE.h}
                label={`Q${sub(qi)}`}
                ghost={inSram}
                active={qi === f.i}
              />
            );
          })}

          {/* K tiles column */}
          <text x={HBM.x + 96} y={HBM.y + 74} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
            {copy.kTiles}
          </text>
          {Array.from({ length: N_KV_TILES }, (_, kj) => {
            const x = HBM.x + 96;
            const y = HBM.y + 82 + kj * (KV_TILE.h + 8);
            const inSram = f.kvInSram && kj === f.j;
            return (
              <TileCell
                key={`k-${kj}`}
                x={x}
                y={y}
                w={KV_TILE.w}
                h={KV_TILE.h}
                label={`K${sub(kj)}`}
                ghost={inSram}
                active={kj === f.j}
              />
            );
          })}

          {/* V tiles column */}
          <text x={HBM.x + 142} y={HBM.y + 74} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
            {copy.vTiles}
          </text>
          {Array.from({ length: N_KV_TILES }, (_, vj) => {
            const x = HBM.x + 142;
            const y = HBM.y + 82 + vj * (KV_TILE.h + 8);
            const inSram = f.kvInSram && vj === f.j;
            return (
              <TileCell
                key={`v-${vj}`}
                x={x}
                y={y}
                w={KV_TILE.w}
                h={KV_TILE.h}
                label={`V${sub(vj)}`}
                ghost={inSram}
                active={vj === f.j}
              />
            );
          })}

          {/* O output column — fills in as tiles finish. An O cell is the ONE
              place a solid fill is earned: it is the only thing this kernel ever
              writes back, so "stored" is the payoff the diagram builds to. */}
          <text x={HBM.x + 232} y={HBM.y + 74} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
            {copy.oOutput}
          </text>
          {Array.from({ length: N_Q_TILES }, (_, oi) => {
            const x = HBM.x + 232;
            const y = HBM.y + 82 + oi * (Q_TILE.h + 10);
            const filled = oi < f.doneO;
            const landing = f.writeBack && oi === f.i;
            return (
              <OutputCell
                key={`o-${oi}`}
                x={x}
                y={y}
                w={Q_TILE.w}
                h={Q_TILE.h}
                label={`O${sub(oi)}`}
                filled={filled}
                landing={landing}
              />
            );
          })}

          {/* ── The bus / transfer arrow between the two cards ── */}
          <BusArrow
            fromX={HBM.x + HBM.w}
            toX={SRAM.x}
            y={HBM.y + 150}
            direction={f.writeBack ? 'left' : 'right'}
            label={f.writeBack ? copy.busWrite : copy.busLoad}
            active={f.phase === 'loadQ' || f.phase === 'loadKV' || f.phase === 'writeO'}
            paused={!player.playing}
          />

          {/* ── SRAM stage (tiny, fast scratchpad — where the action happens).
              EMERALD in every frame, for the same reason HBM is always red: it is
              an identity. "Busy right now" is carried by the hatch on the boxes
              INSIDE it, which is more precise than lighting up the whole card. ── */}
          <PanelFrame x={SRAM.x} y={SRAM.y} w={SRAM.w} h={SRAM.h} tone="emerald" fill="none" />
          <FrameLabel x={SRAM.x + 14} y={SRAM.y} label={copy.sramTitle} fill={EMERALD} />
          <text x={SRAM.x + 14} y={SRAM.y + 24} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.sramHint}
          </text>

          {/* Resident Q tile in SRAM */}
          <SramSlot x={SRAM.x + 18} y={SRAM.y + 70} w={48} h={48} label={`Q${sub(f.i)}`} present={f.qInSram} />
          {/* Resident K_j / V_j tiles in SRAM */}
          <SramSlot
            x={SRAM.x + 76}
            y={SRAM.y + 70}
            w={40}
            h={48}
            label={f.j >= 0 ? `K${sub(f.j)}` : 'Kⱼ'}
            present={f.kvInSram}
          />
          <SramSlot
            x={SRAM.x + 122}
            y={SRAM.y + 70}
            w={40}
            h={48}
            label={f.j >= 0 ? `V${sub(f.j)}` : 'Vⱼ'}
            present={f.kvInSram}
          />

          {/* The small score tile S_ij — alive only on score/update frames */}
          <ScoreTile
            x={SRAM.x + 182}
            y={SRAM.y + 66}
            alive={f.sAlive}
            dying={f.phase === 'discard'}
            label={f.j >= 0 ? `S${sub(f.i)}${sub(f.j)}` : 'Sᵢⱼ'}
            ease={ease}
          />

          {/* Streaming-softmax accumulator (m, ℓ, O_i). It exists in every frame —
              it IS the running state — so it is never dashed and never changes
              hue. The hatch is the only thing that moves: it marks the frame where
              a score tile is being folded in. (This used to be `animate-pulse`,
              which kept pulsing while the player was paused.) */}
          <g>
            <rect
              x={SRAM.x + 18}
              y={SRAM.y + 158}
              width={SRAM.w - 36}
              height={56}
              rx={RX}
              fill={f.accumulate ? hatchFill('emerald') : 'none'}
              style={{ stroke: EMERALD, opacity: f.accumulate ? 1 : DIM.HUED_INNER, transition: ease }}
              strokeWidth={SW.INNER}
            />
            <text
              x={SRAM.x + 30}
              y={SRAM.y + 180}
              style={{ fill: 'var(--foreground)' }}
              className="text-[11px] font-semibold"
            >
              {copy.streamingSoftmax}
            </text>
            <text
              x={SRAM.x + 30}
              y={SRAM.y + 200}
              style={{ fill: 'var(--muted-foreground)' }}
              className="font-mono text-[11px]"
            >
              {copy.running(f.i)}
            </text>
          </g>

          {/* Caption strip pinned at the bottom of the SRAM card */}
          <text
            x={SRAM.x + 18}
            y={SRAM.y + 244}
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[10px]"
          >
            {copy.sramFootnote}
          </text>
        </svg>
      </PanBox>

      {/* Live counters + loop position */}
      <div className="flex flex-wrap items-end justify-between gap-3 border-t border-border/60 pt-3">
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">
            {copy.loopPosition}{' '}
            <span className="font-mono text-foreground/90">
              i = {f.i}
              {f.j >= 0 ? `, j = ${f.j}` : ''}
            </span>{' '}
            <span className="text-muted-foreground/80">{copy.loopTiles(N_Q_TILES, N_KV_TILES)}</span>
          </div>
          <div className="text-[10px] text-muted-foreground/70">{copy.fusedLine}</div>
        </div>

        <div className="space-y-1 text-right">
          <div className="text-[11px] text-muted-foreground">
            {copy.hbmWritesLabel}{' '}
            <span className="font-mono" style={{ color: EMERALD }}>
              {hbmScoreWrites}
            </span>
          </div>
          <div className="text-[10px] text-muted-foreground/70">{copy.naiveNote}</div>
        </div>
      </div>
    </DiagramFrame>
  );
}

/**
 * A tile cell inside HBM. Neutral in every frame — these bytes are permanently
 * here, so they carry no hue of their own; the red frame around them already
 * says which tier they sit in.
 *
 *   ghost    a copy is currently up in SRAM → dashed + dimmed, the slot is out
 *   active   this is the tile the loop is on → hatched, "busy this step"
 */
function TileCell({
  x,
  y,
  w,
  h,
  label,
  ghost,
  active,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  ghost: boolean;
  active: boolean;
}) {
  return (
    <g style={{ opacity: ghost ? DIM.EMPTY_SLOT : 1 }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={RX}
        fill={active && !ghost ? hatchFill('muted') : 'none'}
        style={{ stroke: 'var(--border)' }}
        strokeWidth={SW.INNER}
        strokeDasharray={ghost ? DASH.BORDER : undefined}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: active ? 'var(--foreground)' : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/**
 * An output cell inside HBM — an empty dashed slot until its O_i is written
 * back, then EMERALD and filled: stored, done, and the only thing the kernel
 * ever sends across the bus. `landing` hatches the frame it arrives on.
 */
function OutputCell({
  x,
  y,
  w,
  h,
  label,
  filled,
  landing,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  filled: boolean;
  landing: boolean;
}) {
  const lit = filled || landing;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={RX}
        fill={landing ? hatchFill('emerald') : filled ? EMERALD : 'none'}
        fillOpacity={filled && !landing ? DIM.FILLED : 1}
        style={{ stroke: lit ? EMERALD : 'var(--border)', opacity: lit ? 1 : DIM.EMPTY_SLOT }}
        strokeWidth={SW.INNER}
        strokeDasharray={lit ? undefined : DASH.SLOT}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: filled && !landing ? 'var(--bg)' : lit ? EMERALD : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/**
 * The transfer arrow across the bus between HBM and SRAM. The edge always
 * exists (dashed + dimmed when idle); the FLOW_BLUE dots are what say bytes are
 * crossing RIGHT NOW, and they ride the byte-identical `d` of the static line —
 * dots never invent an edge. `reverse` turns them around on the write-back.
 */
function BusArrow({
  fromX,
  toX,
  y,
  direction,
  label,
  active,
  paused,
}: {
  fromX: number;
  toX: number;
  y: number;
  direction: 'left' | 'right';
  label: string;
  active: boolean;
  /** Pass `!player.playing` — a Pause that leaves the dots travelling is a lie. */
  paused: boolean;
}) {
  const tipX = direction === 'right' ? toX : fromX;
  const tipDir = direction === 'right' ? 1 : -1;
  const d = `M ${fromX + 2} ${y} L ${toX - 2} ${y}`;
  return (
    <g>
      {/* Dimming the static parts as a GROUP, with the dots outside it: element
          opacity composites children, so folding the dots in here would fade
          them too. */}
      <g style={{ opacity: active ? 1 : DIM.IDLE_EDGE }}>
        <path
          d={d}
          fill="none"
          style={{ stroke: 'var(--muted-foreground)' }}
          strokeWidth={SW.INNER}
          strokeDasharray={active ? undefined : DASH.BORDER}
        />
        <polygon
          points={`${tipX},${y} ${tipX - tipDir * 9},${y - 5} ${tipX - tipDir * 9},${y + 5}`}
          style={{ fill: 'var(--muted-foreground)' }}
        />
        <text
          x={(fromX + toX) / 2}
          y={y - 10}
          textAnchor="middle"
          style={{ fill: active ? 'var(--foreground)' : 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          {label}
        </text>
      </g>
      <FlowDots d={d} hidden={!active} reverse={direction === 'left'} paused={paused} spacing={12} />
    </g>
  );
}

/**
 * A resident slot inside SRAM. Empty = an empty repeated slot (dashed, dim);
 * loaded = real, so it takes SRAM's own EMERALD — the standard "becomes real"
 * transform, not a hue used as a state.
 */
function SramSlot({
  x,
  y,
  w,
  h,
  label,
  present,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  present: boolean;
}) {
  return (
    <g style={{ opacity: present ? 1 : DIM.EMPTY_SLOT }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={RX}
        fill="none"
        style={{ stroke: present ? EMERALD : 'var(--border)' }}
        strokeWidth={SW.INNER}
        strokeDasharray={present ? undefined : DASH.SLOT}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: present ? EMERALD : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/**
 * The small score tile S_ij — hatched EMERALD while it is being computed and
 * folded in (busy, on-chip), a dashed ghost before and after. It never becomes
 * anything else, because it never goes anywhere: it lives and dies right here.
 */
function ScoreTile({
  x,
  y,
  alive,
  dying,
  label,
  ease,
}: {
  x: number;
  y: number;
  alive: boolean;
  dying: boolean;
  label: string;
  /** `undefined` under reduced motion. */
  ease: string | undefined;
}) {
  const copy = COPY[useLocale()];
  const w = 84;
  const h = 60;
  return (
    <g
      style={{
        opacity: alive ? 1 : dying ? DIM.EMPTY_SLOT : DIM.IDLE_EDGE,
        transition: ease,
      }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={RX}
        fill={alive ? hatchFill('emerald') : 'none'}
        style={{ stroke: alive ? EMERALD : 'var(--border)' }}
        strokeWidth={SW.INNER}
        strokeDasharray={alive ? undefined : DASH.BORDER}
      />
      {/* mini score grid to read as a matrix tile */}
      {Array.from({ length: 3 }, (_, r) =>
        Array.from({ length: 4 }, (_, c) => (
          <rect
            key={`${r}-${c}`}
            x={x + 10 + c * 12}
            y={y + 8 + r * 10}
            width={9}
            height={7}
            rx={RX}
            fill={EMERALD}
            fillOpacity={alive ? DIM.FILLED : DIM.IDLE_EDGE}
          />
        )),
      )}
      <text
        x={x + w / 2}
        y={y + h - 8}
        textAnchor="middle"
        style={{ fill: EMERALD }}
        className="font-mono text-[11px] font-semibold"
      >
        {label}
      </text>
      {dying ? (
        <text
          x={x + w / 2}
          y={y - 6}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[9px]"
        >
          {copy.discarded}
        </text>
      ) : null}
    </g>
  );
}
