import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 12 (Training) supplement — backprop as a picture, no calculus.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers and works while the model downloads.
 *
 * The loss is one number at the top of the stack. Backprop walks DOWN the same
 * blocks the forward pass walked up, one block at a time. Each block answers
 * one question — "how much did my output move the loss?" — and records that
 * answer as its gradient. That hand-off, block to block, is the chain rule; no
 * notation needed to see it. Backprop only COMPUTES gradients — the optimizer
 * applies the weight updates afterward (the captions keep that separation).
 *
 * Animation model: discrete frames advanced by a setInterval inside a guarded
 * useEffect. Honors prefers-reduced-motion (jumps to the final frame, no
 * autoplay).
 */

// Stack drawn top to bottom. A "⋮ 22 more layers" ellipsis row sits between
// "layer 24" and "layer 1". Labels live in COPY (per locale).
const BLOCKS = [{ id: 'lm-head' }, { id: 'layer-24' }, { id: 'layer-1' }, { id: 'embedding' }] as const;

// Per-locale copy. Every user-visible English string moved here verbatim; the
// captions are JSX fragments (the active-block caption is a builder taking the
// localized block label).
const COPY = {
  en: {
    blocks: ['LM head', 'layer 24', 'layer 1', 'embedding'],
    title: 'Backprop — the loss flows back down the stack',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    svgAria:
      'Diagram of backpropagation: a loss node at the top, then LM head, layer 24, layer 1, and embedding stacked below. Faint forward-pass arrows go up; gradient arrows animate down block by block. Each block asks how much its output moved the loss and records that answer as its gradient; the optimizer applies the weight updates afterward.',
    lossNode: 'loss',
    oneNumber: 'one number',
    forwardCol: 'forward ↑ (done)',
    gradientCol: 'gradient ↓',
    moreLayers: '⋮ 22 more layers',
    gradReady: '✓ grad ready',
    captionStart: (
      <>
        Forward pass done (faint arrows going <strong>up</strong>): the whole model boiled the batch down to{' '}
        <strong>one number</strong>, the loss. Now that number flows back <strong>down</strong> the same blocks.
      </>
    ),
    captionDone: (
      <>
        Done. Every block got its answer — its <strong>gradient</strong> — and that block-by-block hand-off down the
        stack is the whole of backprop. Note what has <em>not</em> happened yet: no weight changed. The optimizer
        (AdamW, below) now uses those ~800M gradients to take one small step — then the next batch.
      </>
    ),
    captionActive: (label: string) => (
      <>
        <span className="font-medium text-primary">{label}</span> asks:{' '}
        <em>&ldquo;how much did my output move the loss?&rdquo;</em> — writes that answer down (its{' '}
        <strong>gradient</strong>), and passes the question on down. No weights change yet.
      </>
    ),
    footer: (
      <>
        That question-and-answer hand-off is the chain rule in action — each block only needs to know how its own
        output moved the loss, never the whole model at once. Backprop only <em>computes</em> the gradients; the
        optimizer (AdamW) is the separate step that turns them into actual weight nudges.
      </>
    ),
  },
  zh: {
    blocks: ['LM head', '第 24 层', '第 1 层', '嵌入'],
    title: '反向传播——损失沿堆叠往下回流',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    svgAria:
      '反向传播示意图：顶部是一个损失节点，下方依次堆叠 LM head、第 24 层、第 1 层和嵌入。浅色的前向箭头朝上；梯度箭头逐块向下推进。每个块询问自己的输出让损失动了多少，并把答案记为自己的梯度；权重更新由优化器在之后施加。',
    lossNode: '损失',
    oneNumber: '一个数',
    forwardCol: '前向 ↑（已完成）',
    gradientCol: '梯度 ↓',
    moreLayers: '⋮ 还有 22 层',
    gradReady: '✓ 梯度就绪',
    captionStart: (
      <>
        前向传播已完成（朝<strong>上</strong>的浅色箭头）：整个模型把这个 batch 浓缩成了<strong>一个数</strong>
        ——损失。现在这个数沿同样的块往<strong>下</strong>回流。
      </>
    ),
    captionDone: (
      <>
        完成。每个块都拿到了自己的答案——它的<strong>梯度</strong>
        ——这种沿堆叠逐块向下的接力，就是反向传播的全部。注意有什么<em>还没</em>
        发生：没有任何权重被改动。接下来由优化器（AdamW，见下文）用这约 800M 个梯度迈出一小步——然后处理下一个
        batch。
      </>
    ),
    captionActive: (label: string) => (
      <>
        <span className="font-medium text-primary">{label}</span> 问：<em>「我的输出让损失动了多少？」</em>
        ——把这个答案记下来（它的<strong>梯度</strong>），再把问题继续往下传。此时还没有任何权重被改动。
      </>
    ),
    footer: (
      <>
        这种一问一答的接力，就是链式法则的实际运作——每个块只需要知道自己的输出让损失动了多少，从不需要一次看到整个模型。反向传播只负责
        <em>计算</em>梯度；把梯度变成对权重的实际微调，是优化器（AdamW）的另一个独立步骤。
      </>
    ),
  },
} as const;

const FRAME_MS = 2100;
// Frame 0: forward pass done, loss computed. Frames 1..4: the gradient reaches
// block 0..3 (top to bottom). Frame 5: done — every block has its gradient.
const TOTAL_FRAMES = BLOCKS.length + 2;
const DONE_FRAME = TOTAL_FRAMES - 1;

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
const VB_W = 700;
const VB_H = 396;
const BLOCK_W = 240;
const BLOCK_H = 44;
const CX = VB_W / 2;
const BLOCK_X = CX - BLOCK_W / 2;
const LOSS_Y = 14;
const LOSS_H = 40;
const STACK_TOP = 86;
const ROW_GAP = 26;
const DOTS_GAP = 24; // extra space for the ellipsis row

// Up-arrows (forward, faint) on the left of center; down-arrows (gradient) on
// the right of center, both inside the block width so the page stays narrow.
const FWD_X = CX - 88;
const GRAD_X = CX + 88;

function blockY(i: number): number {
  const extra = i >= 2 ? DOTS_GAP : 0;
  return STACK_TOP + i * (BLOCK_H + ROW_GAP) + extra;
}

/** Gap above block i: [topY, bottomY] of the empty space the arrows live in. */
function gapAbove(i: number): [number, number] {
  if (i === 0) return [LOSS_Y + LOSS_H, blockY(0)];
  return [blockY(i - 1) + BLOCK_H, blockY(i)];
}

export function BackpropFlowDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [frame, setFrame] = React.useState(reducedMotion ? DONE_FRAME : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const step = () => {
    setPlaying(false);
    setFrame((x) => (x + 1) % TOTAL_FRAMES);
  };

  // Index of the block the gradient is currently visiting (-1 before it
  // starts). On the done frame every block has been visited.
  const activeIdx = frame === 0 ? -1 : Math.min(frame - 1, BLOCKS.length - 1);
  const isDone = frame === DONE_FRAME;

  let caption: React.ReactNode;
  if (frame === 0) {
    caption = copy.captionStart;
  } else if (isDone) {
    caption = copy.captionDone;
  } else {
    caption = copy.captionActive(copy.blocks[activeIdx]!);
  }

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
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

      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria}
      >
        <defs>
          <marker id="bp-fwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 Z" style={{ fill: 'var(--muted-foreground)' }} opacity={0.45} />
          </marker>
          <marker id="bp-grad" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 Z" style={{ fill: 'var(--primary)' }} />
          </marker>
        </defs>

        {/* ── Loss node at the top ── */}
        <rect
          x={CX - 80}
          y={LOSS_Y}
          width={160}
          height={LOSS_H}
          rx={20}
          style={{ fill: 'var(--primary)', fillOpacity: 0.14, stroke: 'var(--primary)' }}
          strokeWidth={2}
        />
        <text
          x={CX}
          y={LOSS_Y + LOSS_H / 2 + 4}
          textAnchor="middle"
          style={{ fill: 'var(--primary)' }}
          className="font-mono text-[13px] font-semibold"
        >
          {copy.lossNode}
        </text>
        <text
          x={CX + 96}
          y={LOSS_Y + LOSS_H / 2 + 4}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.oneNumber}
        </text>

        {/* ── Column labels ── */}
        <text
          x={FWD_X}
          y={LOSS_Y + 12}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.forwardCol}
        </text>
        <text
          x={GRAD_X}
          y={LOSS_Y + 12}
          textAnchor="start"
          style={{ fill: frame > 0 ? 'var(--primary)' : 'var(--muted-foreground)' }}
          className="text-[10px] font-medium"
        >
          {copy.gradientCol}
        </text>

        {/* ── Ellipsis between layer 24 and layer 1 ── */}
        <text
          x={CX}
          y={(blockY(1) + BLOCK_H + blockY(2)) / 2 + 4}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[11px]"
        >
          {copy.moreLayers}
        </text>

        {/* ── Arrows in each gap: faint forward up (left), gradient down (right) ── */}
        {BLOCKS.map((b, i) => {
          const [top, bottom] = gapAbove(i);
          const pad = 5;
          const gradReached = isDone || activeIdx >= i;
          const gradHot = !isDone && activeIdx === i;
          return (
            <g key={`arrows-${b.id}`}>
              {/* forward: bottom → top (marker at the top end, auto-oriented up) */}
              <line
                x1={FWD_X}
                y1={bottom - pad}
                x2={FWD_X}
                y2={top + pad}
                style={{ stroke: 'var(--muted-foreground)' }}
                strokeOpacity={0.35}
                strokeWidth={1.5}
                markerEnd="url(#bp-fwd)"
              />
              {/* gradient: top → bottom, only once the gradient has come this far */}
              {gradReached ? (
                <line
                  x1={GRAD_X}
                  y1={top + pad}
                  x2={GRAD_X}
                  y2={bottom - pad}
                  style={{ stroke: 'var(--primary)' }}
                  strokeOpacity={gradHot || isDone ? 1 : 0.45}
                  strokeWidth={2}
                  markerEnd="url(#bp-grad)"
                />
              ) : null}
            </g>
          );
        })}

        {/* ── The stacked blocks ── */}
        {BLOCKS.map((b, i) => {
          const y = blockY(i);
          const isActive = !isDone && activeIdx === i;
          const nudged = isDone || (activeIdx >= 0 && activeIdx > i) || isActive;
          return (
            <g key={b.id}>
              <rect
                x={BLOCK_X}
                y={y}
                width={BLOCK_W}
                height={BLOCK_H}
                rx={8}
                style={{
                  fill: isActive ? 'var(--primary)' : 'var(--card)',
                  fillOpacity: isActive ? 0.16 : 1,
                  stroke: isActive ? 'var(--primary)' : 'var(--border)',
                }}
                strokeWidth={isActive ? 2 : 1.5}
              />
              <text
                x={CX}
                y={y + BLOCK_H / 2 + 4}
                textAnchor="middle"
                style={{ fill: isActive ? 'var(--primary)' : 'var(--foreground)' }}
                className="font-mono text-[13px] font-semibold"
              >
                {copy.blocks[i]}
              </text>
              {nudged && !isActive ? (
                <text
                  x={BLOCK_X + BLOCK_W - 10}
                  y={y + BLOCK_H / 2 + 4}
                  textAnchor="end"
                  style={{ fill: 'var(--primary)' }}
                  className="text-[10px]"
                >
                  {copy.gradReady}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>

      {/* Caption — what this frame means */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">{caption}</p>

      <p className="text-[10px] text-muted-foreground/70">{copy.footer}</p>
    </div>
  );
}
