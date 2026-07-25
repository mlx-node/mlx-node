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
  PanelFrame,
  RED,
  RX,
  SW,
  useStepPlayer,
} from '../motion';

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
 * Animation model: `useStepPlayer` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval` + private `usePrefersReducedMotion`. The
 * swap is behaviour-identical — mounts playing, wraps modulo TOTAL_FRAMES,
 * rests reduced-motion readers on the last frame — and additionally fixes a
 * real bug the local version had: flipping the OS preference mid-session left
 * the old effect early-returning with `playing` still true and the controls
 * hidden, i.e. permanently frozen with no way to step.
 *
 * ── THE PALETTE, AND THE ONE PLACE IT DEPARTS FROM `skin` ──────────────────
 *
 * `skin` says hue is a box's PERMANENT IDENTITY and state is carried by hatch
 * (busy) and dash (not real yet). That rule assumes boxes with different jobs.
 * Here all four boxes have the SAME job — they are the model's blocks — and the
 * only thing the reader has to track is WHERE THE SWEEP IS. So hue tracks the
 * sweep, and there is no competing identity for it to lose:
 *
 *   dashed, dim, neutral   the gradient has not reached this block yet
 *   RED + diagonal hatch   this block is being differentiated ON THIS FRAME
 *   EMERALD                this block's gradient is recorded — done
 *   FLOW_BLUE dots         the gradient in flight, on the edge it is crossing
 *
 * RED is used exactly as `skin` defines it (expensive / compute happening now):
 * the loss node is RED in every frame because it is the head of the backward
 * pass, and the block under differentiation is RED because that is where the
 * compute is this instant. EMERALD is used exactly as `skin` defines it
 * (stored / done): recorded gradients, and the stretch of route already walked.
 *
 * The widget has no colour contract with any neighbour — it hardcoded no hex at
 * all before this change, and its marker ids are file-local — so adopting the
 * two hues drags nothing else along.
 */

// Stack drawn top to bottom. A "⋮ 22 more layers" ellipsis row sits between
// "layer 24" and "layer 1". Labels live in COPY (per locale).
const BLOCKS = [{ id: 'lm-head' }, { id: 'layer-24' }, { id: 'layer-1' }, { id: 'embedding' }] as const;

// Per-locale copy. Every user-visible English string moved here verbatim; the
// captions are JSX fragments (the active-block caption is a builder taking the
// localized block label).
//
// Play / Pause / Step are NOT here any more: those verbs now belong to the
// shared `StepControls`, which carries its own en/zh pair so every animated
// widget in the course says the same word.
const COPY = {
  en: {
    blocks: ['LM head', 'layer 24', 'layer 1', 'embedding'],
    title: 'Backprop — the loss flows back down the stack',
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
        <span className="font-medium" style={{ color: RED }}>
          {label}
        </span>{' '}
        asks: <em>&ldquo;how much did my output move the loss?&rdquo;</em> — writes that answer down (its{' '}
        <strong>gradient</strong>), and passes the question on down. No weights change yet.
      </>
    ),
    footer: (
      <>
        That question-and-answer hand-off is the chain rule in action — each block only needs to know how its own output
        moved the loss, never the whole model at once. Backprop only <em>computes</em> the gradients; the optimizer
        (AdamW) is the separate step that turns them into actual weight nudges.
      </>
    ),
  },
  zh: {
    blocks: ['LM head', '第 24 层', '第 1 层', '嵌入'],
    title: '反向传播——损失沿堆叠往下回流',
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
        发生：没有任何权重被改动。接下来由优化器（AdamW，见下文）用这约 800M 个梯度迈出一小步——然后处理下一个 batch。
      </>
    ),
    captionActive: (label: string) => (
      <>
        <span className="font-medium" style={{ color: RED }}>
          {label}
        </span>{' '}
        问：<em>「我的输出让损失动了多少？」</em>
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

// ── SVG geometry ────────────────────────────────────────────────────────────
//
// 460 wide, NOT 700, and this is the whole ballgame for legibility. The svg is
// `w-full` inside chapter 13's reading column, which measures 451px at a 1280px
// viewport, so a viewBox unit is 451/VB_W CSS pixels. At VB_W = 700 the skin's
// 10px chip type landed at 6.4 CSS px — measured, not estimated — which is
// smaller than any other text on the page by half. At 460 the scale is 0.98 and
// the type sizes mean what they say. See the CANVAS WIDTH note in skin.tsx.
//
// The height is whatever the stack needs (376 + a 16 margin). A four-block
// vertical stack in a narrow column comes out TALLER than wide; forcing it to
// 16:9 is what pushed the width to 700 in the first place.
const VB_W = 460;
const VB_H = 392;
const CX = VB_W / 2;

// Narrower than the blocks, and narrower than the lane spacing: the loss box
// spans CX +/- 80 while the lanes run at CX +/- 100, which is what leaves the
// gap-0 elbows a real 20-unit horizontal run out of the box's own edges.
const LOSS_W = 160;
const LOSS_X = CX - LOSS_W / 2;
const LOSS_Y = 16;
const LOSS_H = 40;
const LOSS_CY = LOSS_Y + LOSS_H / 2;

const BLOCK_W = 300;
const BLOCK_X = CX - BLOCK_W / 2;
const BLOCK_H = 40;
const STACK_TOP = 94;
const ROW_GAP = 32;
const DOTS_GAP = 26; // extra space for the ellipsis row

// Up-arrows (forward, faint) on the left of center; down-arrows (gradient) on
// the right of center, both inside the block width so the page stays narrow.
// They sit well clear of the name chips, which are centred on CX: the widest
// chip ("embedding") spans roughly CX +/- 39, these lanes are at CX +/- 100.
const FWD_X = CX - 100;
const GRAD_X = CX + 100;
const ARROW_PAD = 5;

/** Baseline for the two lane captions — below the loss box, above block 0. */
const COL_LABEL_Y = 78;

function blockY(i: number): number {
  const extra = i >= 2 ? DOTS_GAP : 0;
  return STACK_TOP + i * (BLOCK_H + ROW_GAP) + extra;
}

/** Gap above block i: [topY, bottomY] of the empty space the arrows live in. */
function gapAbove(i: number): [number, number] {
  if (i === 0) return [LOSS_Y + LOSS_H, blockY(0)];
  return [blockY(i - 1) + BLOCK_H, blockY(i)];
}

// Route paths. Right angles only — there are no curves in this course.
//
// The two paths for gap 0 elbow out of the loss box's own edges instead of
// starting in mid-air beside it, so "the loss is where the gradient comes from"
// is drawn rather than implied. Gaps 1..3 are plain vertical runs.
//
// Each `gradD` string is used TWICE and is the same string both times: once for
// the static dashed connector that is always painted, once for the FlowDots
// that ride it on the frame the gradient crosses it. Dots animate an edge that
// exists; they never invent one.
function gradD(i: number): string {
  const [top, bottom] = gapAbove(i);
  if (i === 0) return `M ${LOSS_X + LOSS_W} ${LOSS_CY} L ${GRAD_X} ${LOSS_CY} L ${GRAD_X} ${bottom - ARROW_PAD}`;
  return `M ${GRAD_X} ${top + ARROW_PAD} L ${GRAD_X} ${bottom - ARROW_PAD}`;
}

function fwdD(i: number): string {
  const [top, bottom] = gapAbove(i);
  if (i === 0) return `M ${FWD_X} ${bottom - ARROW_PAD} L ${FWD_X} ${LOSS_CY} L ${LOSS_X} ${LOSS_CY}`;
  return `M ${FWD_X} ${bottom - ARROW_PAD} L ${FWD_X} ${top + ARROW_PAD}`;
}

export function BackpropFlowDiagram() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS });
  const { frame } = player;

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

  const ease = player.reducedMotion ? undefined : 'stroke 300ms ease, opacity 300ms ease';

  return (
    <DiagramFrame title={copy.title} player={player} locale={locale} caption={caption} note={copy.footer}>
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        <HatchDefs />
        <defs>
          <marker id="bp-fwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 Z" style={{ fill: 'var(--muted-foreground)' }} opacity={DIM.IDLE_EDGE} />
          </marker>
          <marker id="bp-grad" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 Z" style={{ fill: EMERALD }} />
          </marker>
        </defs>

        {/* ── Loss node at the top: RED in every frame — it is the head of the
            backward pass, not a state that comes and goes. ── */}
        <PanelFrame x={LOSS_X} y={LOSS_Y} w={LOSS_W} h={LOSS_H} tone="red" fill="none" />
        <text
          x={CX}
          y={LOSS_CY + 4}
          textAnchor="middle"
          style={{ fill: RED }}
          className="text-[10px] font-semibold uppercase tracking-wider"
        >
          {copy.oneNumber}
        </text>

        {/* ── Lane captions, centred over the lane each one names. They sit in
            the band between the loss box (ends y=56) and block 0 (starts y=94),
            so they cannot collide with either. Centring rather than pushing
            them outward is what keeps the widest string ("FORWARD ↑ (DONE)",
            ~105 units) inside a 460-wide canvas in both locales. ── */}
        <text
          x={FWD_X}
          y={COL_LABEL_Y}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[9px] uppercase tracking-wider"
        >
          {copy.forwardCol}
        </text>
        <text
          x={GRAD_X}
          y={COL_LABEL_Y}
          textAnchor="middle"
          style={{ fill: frame > 0 ? EMERALD : 'var(--muted-foreground)' }}
          className="text-[9px] font-medium uppercase tracking-wider"
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
          const d = gradD(i);
          // Already walked: solid, EMERALD, arrowhead landing on the block.
          const delivered = isDone || activeIdx > i;
          // Being walked right now: the dots, and only the dots.
          const inFlight = !isDone && activeIdx === i;
          return (
            <g key={`arrows-${b.id}`}>
              {/* Forward: bottom → top. SOLID and dim, never dashed. Dash means
                  "has not happened yet" everywhere else in this course, and the
                  forward pass is the one thing on this diagram that HAS already
                  happened — the caption says so on every frame. Dashing it made
                  this lane byte-identical to the un-walked gradient lane.

                  `strokeOpacity`, NOT `opacity`: element opacity composites the
                  marker too, so 0.4 on the path times 0.4 inside the marker put
                  the arrowhead at 0.16 — and the arrowhead is the only thing
                  carrying "this lane points UP". strokeOpacity leaves it alone. */}
              <path
                d={fwdD(i)}
                fill="none"
                style={{ stroke: 'var(--muted-foreground)' }}
                strokeOpacity={DIM.IDLE_EDGE}
                strokeWidth={SW.INNER}
                markerEnd="url(#bp-fwd)"
              />
              {/* gradient: the edge always exists — dashed until it carries something */}
              <path
                d={d}
                fill="none"
                style={{
                  stroke: delivered ? EMERALD : 'var(--muted-foreground)',
                  opacity: delivered ? 1 : DIM.IDLE_EDGE,
                  transition: ease,
                }}
                strokeWidth={delivered ? SW.OUTER : SW.INNER}
                strokeDasharray={delivered ? undefined : DASH.BORDER}
                markerEnd={delivered ? 'url(#bp-grad)' : undefined}
              />
              <FlowDots d={d} hidden={!inFlight} paused={!player.playing} spacing={12} />
            </g>
          );
        })}

        {/* ── The stacked blocks ── */}
        {BLOCKS.map((b, i) => {
          const y = blockY(i);
          const isActive = !isDone && activeIdx === i;
          const gradReady = isDone || activeIdx > i;
          return (
            <g key={b.id}>
              {isActive || gradReady ? (
                <PanelFrame
                  x={BLOCK_X}
                  y={y}
                  w={BLOCK_W}
                  h={BLOCK_H}
                  tone={isActive ? 'red' : 'emerald'}
                  hatched={isActive}
                  fill="none"
                />
              ) : (
                // Not reached yet — but SOLID, and a real `var(--border)`, not
                // the dashed dim ghost. These blocks exist and have already run:
                // the forward pass went through all four of them before frame 0,
                // which is exactly what the caption says. Dash is this course's
                // spelling for "does not exist yet"; using it here contradicted
                // the lesson and drew the boxes at 2.2:1 besides. The state that
                // IS changing is carried by hue — neutral → RED → EMERALD.
                <rect
                  x={BLOCK_X}
                  y={y}
                  width={BLOCK_W}
                  height={BLOCK_H}
                  rx={RX}
                  fill="none"
                  style={{ stroke: 'var(--border)', transition: ease }}
                  strokeWidth={SW.OUTER}
                />
              )}
              {gradReady ? (
                <text
                  x={CX}
                  y={y + BLOCK_H / 2 + 4}
                  textAnchor="middle"
                  style={{ fill: EMERALD }}
                  className="text-[10px] font-medium uppercase tracking-wider"
                >
                  {copy.gradReady}
                </text>
              ) : null}
              {/* Chip last: its job is to erase the border under the label. */}
              <FrameLabel x={CX} y={y} label={copy.blocks[i]!} align="middle" />
            </g>
          );
        })}

        {/* Loss chip drawn after the lanes so the elbows tuck under it cleanly. */}
        <FrameLabel x={CX} y={LOSS_Y} label={copy.lossNode} align="middle" />
      </svg>
    </DiagramFrame>
  );
}
