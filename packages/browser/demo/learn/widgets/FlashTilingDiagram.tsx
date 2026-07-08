import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

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
 * Animation model: a flat list of discrete frames stepping through i ∈ {0,1}
 * and j ∈ {0,1,2}. The first render is a fixed frame (deterministic, identical
 * server vs client). Auto-advance lives in a useEffect guarded by playing &&
 * !reducedMotion. Honors prefers-reduced-motion (a representative static frame
 * mid inner-loop, showing the S tile alive in SRAM, with the "0 HBM writes" note).
 */

const N_Q_TILES = 2; // query row-blocks
const N_KV_TILES = 3; // key/value blocks

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
          . It lived and died in SRAM — it never crossed the bus to HBM. <span className="text-primary">HBM score writes still 0.</span>
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
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    ariaLabel:
      'Diagram of the FlashAttention tiling loop: query tiles loaded from slow HBM into fast on-chip SRAM, key and value tiles streamed in, a small score tile computed and discarded inside SRAM via streaming softmax, and only the finished output tile written back to HBM — the full N by N score matrix never reaches HBM.',
    hbmTitle: 'HBM · main memory',
    hbmHint: 'huge — but far away and slow',
    qTiles: 'Q tiles',
    kTiles: 'K tiles',
    vTiles: 'V tiles',
    oOutput: 'O output',
    busWrite: 'write Oᵢ →',
    busLoad: '← load tiles',
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
      'Reduced motion: showing a representative frame — the score tile S₀₁ alive inside SRAM, with HBM score writes at 0.',
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
          。它在 SRAM 中生灭——从未跨过总线去 HBM。<span className="text-primary">HBM 分数写入仍为 0。</span>
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
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    ariaLabel:
      'FlashAttention 分块循环示意图：query 块从慢速 HBM 加载进快速片上 SRAM，key 和 value 块流式载入，小分数块在 SRAM 内通过流式 softmax 计算并丢弃，只有算完的输出块写回 HBM——完整的 N×N 分数矩阵从不进入 HBM。',
    hbmTitle: 'HBM · 主内存',
    hbmHint: '巨大——但遥远而慢',
    qTiles: 'Q 块',
    kTiles: 'K 块',
    vTiles: 'V 块',
    oOutput: 'O 输出',
    busWrite: '写回 Oᵢ →',
    busLoad: '← 加载块',
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
    reducedNote: '减少动态效果：展示一帧代表性画面——分数块 S₀₁ 存活在 SRAM 中，HBM 分数写入为 0。',
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
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  // First render is deterministic and identical server vs client: a fixed
  // frame. We only retarget to the static frame AFTER mount, inside an effect,
  // so SSR output never depends on matchMedia.
  const [frame, setFrame] = React.useState(INITIAL_FRAME);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (reducedMotion) {
      setPlaying(false);
      setFrame(STATIC_FRAME >= 0 ? STATIC_FRAME : 0);
    }
  }, [reducedMotion]);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((x) => (x + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const f = FRAMES[frame];

  const step = () => {
    setPlaying(false);
    setFrame((x) => (x + 1) % TOTAL_FRAMES);
  };

  // Counters. HBM score-matrix writes are always zero — that is the headline.
  const hbmScoreWrites = 0;

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

      {/* The diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.ariaLabel}
      >
        {/* ── HBM card (big, slow main memory) ── */}
        <rect
          x={HBM.x}
          y={HBM.y}
          width={HBM.w}
          height={HBM.h}
          rx={10}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text x={HBM.x + 16} y={HBM.y + 26} style={{ fill: 'var(--foreground)' }} className="text-[14px] font-semibold">
          {copy.hbmTitle}
        </text>
        <text x={HBM.x + 16} y={HBM.y + 44} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
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

        {/* O output column — fills in as tiles finish */}
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
        />

        {/* ── SRAM card (tiny, fast scratchpad — where the action happens) ── */}
        <rect
          x={SRAM.x}
          y={SRAM.y}
          width={SRAM.w}
          height={SRAM.h}
          rx={10}
          style={{
            fill: 'var(--card)',
            stroke: f.sAlive || f.accumulate ? 'var(--primary)' : 'var(--border)',
          }}
          strokeWidth={f.sAlive || f.accumulate ? 2 : 1.5}
        />
        <text x={SRAM.x + 14} y={SRAM.y + 26} style={{ fill: 'var(--foreground)' }} className="text-[14px] font-semibold">
          {copy.sramTitle}
        </text>
        <text x={SRAM.x + 14} y={SRAM.y + 44} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.sramHint}
        </text>

        {/* Resident Q tile in SRAM */}
        <SramSlot
          x={SRAM.x + 18}
          y={SRAM.y + 70}
          w={48}
          h={48}
          label={`Q${sub(f.i)}`}
          present={f.qInSram}
          kind="q"
        />
        {/* Resident K_j / V_j tiles in SRAM */}
        <SramSlot
          x={SRAM.x + 76}
          y={SRAM.y + 70}
          w={40}
          h={48}
          label={f.j >= 0 ? `K${sub(f.j)}` : 'Kⱼ'}
          present={f.kvInSram}
          kind="kv"
        />
        <SramSlot
          x={SRAM.x + 122}
          y={SRAM.y + 70}
          w={40}
          h={48}
          label={f.j >= 0 ? `V${sub(f.j)}` : 'Vⱼ'}
          present={f.kvInSram}
          kind="kv"
        />

        {/* The small score tile S_ij — alive only on score/update frames */}
        <ScoreTile
          x={SRAM.x + 182}
          y={SRAM.y + 66}
          alive={f.sAlive}
          dying={f.phase === 'discard'}
          label={f.j >= 0 ? `S${sub(f.i)}${sub(f.j)}` : 'Sᵢⱼ'}
        />

        {/* Streaming-softmax accumulator (m, ℓ, O_i) */}
        <g className={f.accumulate ? 'animate-pulse' : undefined}>
          <rect
            x={SRAM.x + 18}
            y={SRAM.y + 158}
            width={SRAM.w - 36}
            height={56}
            rx={8}
            style={{
              fill: f.accumulate ? 'var(--primary)' : 'var(--muted)',
              fillOpacity: f.accumulate ? 0.16 : 1,
              stroke: f.accumulate ? 'var(--primary)' : 'var(--border)',
            }}
            strokeWidth={f.accumulate ? 2 : 1}
          />
          <text
            x={SRAM.x + 30}
            y={SRAM.y + 180}
            style={{ fill: 'var(--foreground)' }}
            className="text-[11px] font-semibold"
          >
            {copy.streamingSoftmax}
          </text>
          <text x={SRAM.x + 30} y={SRAM.y + 200} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[11px]">
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

      {/* Caption — what the current frame does */}
      <p className="min-h-[3rem] text-[13px] text-foreground/90">{copy.captions[f.phase](f.i, f.j)}</p>

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
            <span className="font-mono text-primary">{hbmScoreWrites}</span>
          </div>
          <div className="text-[10px] text-muted-foreground/70">{copy.naiveNote}</div>
        </div>
      </div>

      {reducedMotion ? <p className="text-[10px] text-muted-foreground/70">{copy.reducedNote}</p> : null}

      <p className="text-[10px] text-muted-foreground/70">{copy.scheduleNote}</p>
    </div>
  );
}

/** A tile cell inside HBM. `ghost` dims it while a copy of it sits in SRAM. */
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
    <g style={{ opacity: ghost ? 0.3 : 1 }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        style={{
          fill: active ? 'var(--primary)' : 'var(--muted)',
          fillOpacity: active ? 0.18 : 1,
          stroke: active ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={active ? 1.5 : 1}
        strokeDasharray={ghost ? '4 3' : undefined}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: active ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/** An output cell inside HBM — empty until its O_i is written back. */
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
    <g className={landing ? 'animate-pulse' : undefined}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        fill="none"
        style={{
          fill: lit ? 'var(--primary)' : 'none',
          fillOpacity: lit ? 0.85 : 1,
          stroke: lit ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={lit ? 1.5 : 1}
        strokeDasharray={lit ? undefined : '4 3'}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: lit ? 'var(--bg)' : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/** The transfer arrow across the bus between HBM and SRAM. */
function BusArrow({
  fromX,
  toX,
  y,
  direction,
  label,
  active,
}: {
  fromX: number;
  toX: number;
  y: number;
  direction: 'left' | 'right';
  label: string;
  active: boolean;
}) {
  const tipX = direction === 'right' ? toX : fromX;
  const tipDir = direction === 'right' ? 1 : -1;
  return (
    <g style={{ opacity: active ? 1 : 0.4 }}>
      <line
        x1={fromX + 2}
        y1={y}
        x2={toX - 2}
        y2={y}
        style={{ stroke: active ? 'var(--primary)' : 'var(--border)' }}
        strokeWidth={active ? 2 : 1.5}
        strokeDasharray={active ? undefined : '5 4'}
      />
      <polygon
        points={`${tipX},${y} ${tipX - tipDir * 9},${y - 5} ${tipX - tipDir * 9},${y + 5}`}
        style={{ fill: active ? 'var(--primary)' : 'var(--border)' }}
      />
      <text
        x={(fromX + toX) / 2}
        y={y - 10}
        textAnchor="middle"
        style={{ fill: active ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="font-mono text-[10px]"
      >
        {label}
      </text>
    </g>
  );
}

/** A resident slot inside SRAM holding a loaded tile (or empty when absent). */
function SramSlot({
  x,
  y,
  w,
  h,
  label,
  present,
  kind,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  present: boolean;
  kind: 'q' | 'kv';
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill="none"
        style={{
          fill: present ? 'var(--primary)' : 'none',
          fillOpacity: present ? (kind === 'q' ? 0.22 : 0.12) : 1,
          stroke: present ? 'var(--primary)' : 'var(--border)',
          opacity: present ? 1 : 0.5,
        }}
        strokeWidth={present ? 1.5 : 1}
        strokeDasharray={present ? undefined : '4 3'}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        style={{ fill: present ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="font-mono text-[11px] font-medium"
        opacity={present ? 1 : 0.5}
      >
        {label}
      </text>
    </g>
  );
}

/** The small score tile S_ij — appears in SRAM, then fades/discards. */
function ScoreTile({
  x,
  y,
  alive,
  dying,
  label,
}: {
  x: number;
  y: number;
  alive: boolean;
  dying: boolean;
  label: string;
}) {
  const copy = COPY[useLocale()];
  const w = 84;
  const h = 60;
  return (
    <g
      style={{
        opacity: alive ? 1 : dying ? 0.25 : 0.18,
        transition: 'opacity 400ms ease',
      }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        style={{
          fill: 'var(--primary)',
          fillOpacity: alive ? 0.16 : 0.05,
          stroke: alive ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={alive ? 2 : 1}
        strokeDasharray={dying ? '5 4' : undefined}
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
            rx={1.5}
            style={{ fill: 'var(--primary)', fillOpacity: alive ? 0.8 : 0.3 }}
          />
        )),
      )}
      <text
        x={x + w / 2}
        y={y + h - 8}
        textAnchor="middle"
        style={{ fill: 'var(--primary)' }}
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
