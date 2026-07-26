import * as React from 'react';

import { Button } from '../../components/ui/button';
import { useLocale } from '../../lib/i18n-react';
import { DASH, DiagramFrame, FlowDots, PanBox, RX, SW, useStepPlayer } from '../motion';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 8 supplement — make the "residual stream" name concrete.
 *
 * The 3D widget already shows that each layer has the shape (norm → attn →
 * residual add, norm → mlp → residual add), but a reader can still finish the
 * chapter without internalizing that the *same vector* is what's being
 * accumulated into. This widget animates that single accumulation.
 *
 * We render a vertical "stream" of bar-blocks growing upward. A token's
 * hidden state enters at the bottom; each layer's attention and MLP outputs
 * are drawn as side-arrows that *write into* the stream (think `h += attn(h)`).
 * Two layers, four writes, then the LM head reads the top of the stream. The
 * stream's height growing is the residual norm growing — a visual referent
 * for the magnitude climb the 3D tower color-codes.
 *
 * Pure JS animation, no model. Loops, with a play/pause toggle.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval`, its `playing` state
 * seeded from `matchMedia` inside a `useState` initializer, and its Play / ▶
 * buttons. Same 7 beats, same 1400ms pace. The ◀ button is the one control the
 * kit has no slot for, so it stays widget-owned — see the `prev` COPY note.
 *
 * ── THE PALETTE, AND WHY THE SKIN'S TWO HUES ARE NOT APPLIED HERE ──────────
 *
 * `skin` allows exactly two hues (RED = busy, EMERALD = done) because in most
 * diagrams hue has one job. Here it has THREE, and they are all identity, not
 * state:
 *
 *   COLORS[0..4]   the 5-stop oklch ramp 250° → 120° = DEPTH. Which hₙ the
 *                  stream currently holds; the same encoding the chapter's 3D
 *                  tower uses for the magnitude climb.
 *   ATTN_COLOR     warm orange = this write came from attention.
 *   MLP_COLOR      purple      = this write came from the MLP.
 *
 * Three channels do not fit in two hues, so the hues stay exactly as they were
 * and only the STRUCTURAL tokens are adopted: RX (square corners), the SW
 * stroke ladder, DASH.BORDER for a write that has not happened yet, and
 * FlowDots for the write that is happening right now. FlowDots runs in the
 * writing sub-block's OWN hue rather than the kit's FLOW_BLUE: `#3b82f6` is
 * ≈ oklch(0.62 0.19 259), which sits ~9° from COLORS[0] (oklch(0.65 0.13 250))
 * and would read as a fourth thing that is somehow also the stream. The dots
 * carry no new colour this way — they are the arrow they ride on.
 *
 * Opacity is deliberately left alone: the fills under `attn(·)`, `mlp(·)` and
 * `lm_head` are tints behind text (0.08–0.25), not dims, and the DIM ladder's
 * 0.4 floor would eat their contrast.
 */

const STEP_LABELS = [
  { kind: 'enter', label: 'h₀  (embedding + RoPE)' },
  { kind: 'attn', label: 'h₁ = h₀ + attn(norm(h₀))' },
  { kind: 'mlp', label: 'h₂ = h₁ + mlp(norm(h₁))' },
  { kind: 'attn', label: 'h₃ = h₂ + attn(norm(h₂))' },
  { kind: 'mlp', label: 'h₄ = h₃ + mlp(norm(h₃))' },
  { kind: 'exit', label: 'lm_head(h₄) → logits' },
] as const;

// In overwrite mode the same steps REPLACE instead of add, so the readout
// formulas drop the "hₙ₋₁ +" term (h := Δ). Parallel to STEP_LABELS by index.
const OVERWRITE_STEP_LABELS = [
  'h₀  (embedding + RoPE)',
  'h₁ = attn(norm(h₀))',
  'h₂ = mlp(norm(h₁))',
  'h₃ = attn(norm(h₂))',
  'h₄ = mlp(norm(h₃))',
  'lm_head(h₄) → logits',
] as const;

// Heights chosen to monotonically grow — each layer adds a contribution, so
// L2 norm climbs. Real Qwen3.5 magnitudes climb roughly like this through
// the 24-layer stack; we compress to 4 writes for visualization.
const HEIGHTS = [22, 30, 38, 48, 60];
// Overwrite mode: each sub-block REPLACES the stream with just its own output,
// so the column jumps to roughly one block's magnitude and never accumulates.
// Same length as HEIGHTS so every geometry helper indexes identically.
const OVERWRITE_HEIGHTS = [24, 21, 26, 20, 25];
const COLORS = [
  'oklch(0.65 0.13 250)', // h0 — cool blue, "input"
  'oklch(0.68 0.14 230)',
  'oklch(0.7 0.14 200)',
  'oklch(0.72 0.14 160)',
  'oklch(0.75 0.15 120)', // h4 — warmer, "almost output"
];

const ATTN_COLOR = 'oklch(0.7 0.16 60)'; // warm orange
const MLP_COLOR = 'oklch(0.62 0.18 300)'; // purple

// Play / Pause and the old ▶ arrow are NOT here any more: those verbs belong to
// the shared `StepControls`, which carries its own en/zh pair so every animated
// widget in the course says the same word (▶ is its "Step").
//
// `prev` STAYS. `StepControls` has no backward control at all — Step is
// forward-only and Reset rewinds to the first beat, so neither of them can move
// the reader back ONE beat, which is the thing the dropped ◀ did. Same word as
// BpeMergeWalkthrough's own Prev, for the same reason.
const COPY = {
  en: {
    header: 'The residual stream — one vector, written into 2× per layer',
    addOption: 'Add (h += Δ)',
    overwriteOption: 'Overwrite (h = new)',
    prev: 'Prev',
    addIntro: (
      <>
        The hidden state for one token, drawn as a column that grows as each sub-block writes a correction into it.
        Attention writes from the left, MLP writes from the right. Nothing ever overwrites — every operation is{' '}
        <code>h := h + Δ</code>. Two layers shown; Qwen3.5-0.8B does this 24 times.
      </>
    ),
    overwriteIntro: (
      <>
        In this mode each sub-block <strong>replaces</strong> the stream — <code>h := Δ</code> — so the running state
        is thrown away every write. Watch the column fail to climb.
      </>
    ),
    svgAria: 'Residual stream animation',
    entryLabel: 'entry — h₀',
    stepPrefix: (step: number) => `Step ${step}: `,
    nextTokenLabel: 'next token sampled →',
    stepLabels: STEP_LABELS.map((s) => s.label),
    overwriteStepLabels: OVERWRITE_STEP_LABELS,
    addOutro: (
      <>
        Three things to notice. First, the stream never gets <em>narrower</em> — there is no operation in the layer
        that subtracts or replaces. Second, the same column is read and written by every sub-block; it's a single
        running address space the whole network shares. Third, the LM head at the top reads the <em>top</em> of the
        stream — every layer's contribution is visible to the final prediction, not just the last one.
      </>
    ),
    overwriteOutro: (
      <>
        This is a deliberately destructive toy — a pure <code>h := Δ</code> that throws the old state away on every
        write, so the column never accumulates. Real non-residual networks aren&apos;t this extreme (they transform
        the state, <code>h := F(h)</code>), but they still give up what the residual stream buys for free: a clean
        identity path, every layer&apos;s contribution preserved by addition, and stable gradients straight back to
        the input. That is why real transformers <em>add</em> (<code>h += Δ</code>) instead of replacing.
      </>
    ),
    footnote: (
      <>
        Illustrative — heights are a synthetic stand-in for the residual-stream magnitude; Qwen3.5-0.8B does this 24
        times. Not live output from the model.
      </>
    ),
  },
  zh: {
    header: '残差流——同一个向量，每层被写入 2 次',
    addOption: '相加 (h += Δ)',
    overwriteOption: '覆盖 (h = new)',
    prev: '上一步',
    addIntro: (
      <>
        这是单个 token
        的隐藏状态，画成一根柱子：每个子块向它写入一次修正，柱子随之长高。注意力从左侧写入，MLP
        从右侧写入。从来没有任何操作会覆盖它——每次操作都是 <code>h := h + Δ</code>。图中只画了两层；Qwen3.5-0.8B
        会做 24 次。
      </>
    ),
    overwriteIntro: (
      <>
        在这个模式里，每个子块都<strong>替换</strong>整条流——<code>h := Δ</code>
        ——运行中的状态在每次写入时都被丢弃。看看柱子是怎么爬不上去的。
      </>
    ),
    svgAria: '残差流动画',
    entryLabel: '入口——h₀',
    stepPrefix: (step: number) => `第 ${step} 步：`,
    nextTokenLabel: '采样下一个 token →',
    stepLabels: [
      'h₀  (嵌入 + RoPE)',
      'h₁ = h₀ + attn(norm(h₀))',
      'h₂ = h₁ + mlp(norm(h₁))',
      'h₃ = h₂ + attn(norm(h₂))',
      'h₄ = h₃ + mlp(norm(h₃))',
      'lm_head(h₄) → logits',
    ],
    overwriteStepLabels: [
      'h₀  (嵌入 + RoPE)',
      'h₁ = attn(norm(h₀))',
      'h₂ = mlp(norm(h₁))',
      'h₃ = attn(norm(h₂))',
      'h₄ = mlp(norm(h₃))',
      'lm_head(h₄) → logits',
    ],
    addOutro: (
      <>
        有三点值得注意。第一，这条流从不会变<em>窄</em>
        ——层里没有任何做减法或替换的操作。第二，每个子块读写的都是同一根柱子；它是整张网络共享的一段运行地址空间。第三，顶部的
        LM head 读取的是流的<em>顶端</em>——每一层的贡献对最终预测都可见，而不只是最后一层。
      </>
    ),
    overwriteOutro: (
      <>
        这是一个故意做坏的玩具——纯粹的 <code>h := Δ</code>
        ，每次写入都把旧状态扔掉，所以柱子永远积累不起来。真实的非残差网络没有这么极端（它们会变换状态，
        <code>h := F(h)</code>），但它们同样失去了残差流免费带来的东西：一条干净的恒等通路、每一层的贡献都靠加法保留、以及一路直达输入的稳定梯度。这就是真实的
        transformer 选择<em>相加</em>（<code>h += Δ</code>）而不是替换的原因。
      </>
    ),
    footnote: (
      <>
        示意图——柱高只是残差流幅度的合成替身；Qwen3.5-0.8B 会做 24 次。并非模型的实时输出。
      </>
    ),
  },
} as const;

const FRAME_MS = 1400;
/** The six labelled beats, plus a trailing "next token sampled" beat. */
const TOTAL_STEPS = STEP_LABELS.length + 1;

type Mode = 'add' | 'overwrite';

export function ResidualStream() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_STEPS, { frameMs: FRAME_MS });
  const step = player.frame;
  const [mode, setMode] = React.useState<Mode>('add');
  const heights = mode === 'add' ? HEIGHTS : OVERWRITE_HEIGHTS;

  // The two inline transitions are the only motion the reduced-motion media
  // query in `styles.css` cannot reach — it can stop CSS animations, not an
  // inline `transition` — so they are gated here instead.
  const growEase = player.reducedMotion ? undefined : 'all 700ms cubic-bezier(0.4, 0, 0.2, 1)';
  const fadeEase = player.reducedMotion ? undefined : 'opacity 600ms';

  const W = 520;
  // 400 / 30 px stream-top headroom — earlier values of H=360, topY=30 put
  // the lm_head box (y=topY-14 to topY+16 ⇒ 16..46) at exactly the same y
  // as the h4 tick at the top of the fully-filled stream (h4 height=60,
  // tick y = baseY-300 = 30), so the box visually swallowed the tick label
  // and the stream column poked through the box's interior. Bumping H to
  // 400 + topY to 20 gives a clean ~34px gap between the lm_head box
  // bottom (y=36) and the h4 tick (y=70), and the arrow from stream top
  // into lm_head now has visible length (~30px).
  const H = 400;
  const streamX = 200;
  const baseY = H - 30;
  const topY = 20;
  const streamW = 28;

  // Cumulative height at each step (after k writes have happened). step=0
  // shows only the input; step=5 shows the final read by lm_head.
  const filledHeight = (s: number): number => {
    if (s <= 0) return heights[0]!;
    if (s >= heights.length) return heights[heights.length - 1]!;
    return heights[s]!;
  };
  const filled = filledHeight(step);

  // Y position of the topmost edge of the stream after `step` writes. The
  // stream grows upward, so smaller Y = taller stream.
  const topOfStreamY = baseY - filled * 5;

  // Side-block geometry: attention writes from the left, MLP from the right.
  const SIDE_OFFSET = 110;

  /**
   * The per-beat readout, which used to be its own bordered strip under the
   * svg. It is the CAPTION now — it is the one line that changes every frame,
   * so it is what the frame's live region has to carry. The "Step N: " prefix
   * stays inside it because `StepControls` renders no counter slot.
   */
  const captionFor = (s: number, m: Mode): React.ReactNode => (
    <>
      <span>{copy.stepPrefix(s)}</span>
      <span className="font-mono text-foreground/95">
        {s < STEP_LABELS.length
          ? m === 'add'
            ? copy.stepLabels[s]!
            : copy.overwriteStepLabels[s]!
          : copy.nextTokenLabel}
      </span>
    </>
  );

  // Every caption the sweep can reach, so the frame reserves the tallest and
  // the chapter body cannot hop when the beat changes. The visible caption is
  // `captionFor(step, mode)` and nothing else, `step` is `player.frame` which
  // useStepPlayer keeps in [0, TOTAL_STEPS), and `mode` has two values — so
  // the full product below is exhaustive by construction, including the
  // `s === STEP_LABELS.length` "next token" arm. BOTH modes are in here on
  // purpose: the toggle swaps the whole formula set, and which set is tallest
  // is not the same question in `en` and `zh`.
  const captions = Array.from({ length: TOTAL_STEPS }, (_, s) => s).flatMap((s) => [
    captionFor(s, 'add'),
    captionFor(s, 'overwrite'),
  ]);

  // `showReset`: a 7-beat sweep is the "long sweep" case the flag exists for —
  // the reader is meant to re-read hₙ against the column, and wants a way back
  // to h₀ that is not a full lap. Reset is NOT the backward control though; it
  // rewinds to the first beat. Going back ONE beat is the Prev button below.
  return (
    <DiagramFrame
      title={copy.header}
      player={player}
      locale={locale}
      caption={captionFor(step, mode)}
      captions={captions}
      note={
        <>
          <p>{mode === 'add' ? copy.addOutro : copy.overwriteOutro}</p>
          <p>{copy.footnote}</p>
        </>
      }
      showReset
    >
      {/* The widget's own controls — DiagramFrame owns play/step/reset only —
          so they sit in their own row above the diagram body, OUTSIDE the pan
          container below so they can never scroll out of reach on a phone. */}
      <div className="flex flex-wrap items-center gap-2">
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          options={[
            { value: 'add', label: copy.addOption },
            { value: 'overwrite', label: copy.overwriteOption },
          ]}
        />
        {/* The exact inverse of `StepControls`' Step, restored: this sweep
            LOOPS, Step wraps forward, so Prev wraps backward. `goTo` pauses
            first — same as Step, and same as the ◀ this replaces. It is a real
            `<button>`, so DiagramFrame's click-capture counts it as the reader
            asking, and the live region announces the beat it lands on. */}
        <Button
          size="sm"
          variant="outline"
          className="max-sm:min-h-[44px]"
          onClick={() => player.goTo((step - 1 + TOTAL_STEPS) % TOTAL_STEPS)}
        >
          {copy.prev}
        </Button>
      </div>

      <p className="text-[12px] text-foreground/85">{mode === 'add' ? copy.addIntro : copy.overwriteIntro}</p>

      {/* PanBox wraps the svg and NOTHING ELSE. A `w-full` svg does not reflow
          in a narrow column — it scales the whole canvas, text and all — so at
          a 375px viewport (column = 320px) this 520-wide viewBox renders at
          0.62×. The smallest type on it is the 9px h0..h4 tick ladder down the
          left edge, and that ladder is not decoration: it is the same hₙ the
          caption's formulas name, so the reader has to read it to connect
          "h₂ = h₁ + mlp(…)" to a level on the column. Unfloored it lands at
          5.5px. 8px is the legibility floor, so ceil(8 * 520 / 9) = 463px, and
          PanBox pans the remaining ~143px. The mode toggle and Prev button
          above are siblings, not children, so they stay at the visible width. */}
      <PanBox locale={locale}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block h-auto w-full min-w-[463px]"
          role="img"
          aria-label={copy.svgAria}
        >
          {/* baseline ticks for h0..h4 */}
          {heights.map((h, i) => {
            const y = baseY - h * 5;
            return (
              <g key={`tick-${i}`}>
                <line
                  x1={streamX - streamW / 2 - 6}
                  y1={y}
                  x2={streamX - streamW / 2 - 2}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity={0.25}
                  strokeWidth={SW.INNER}
                />
                <text
                  x={streamX - streamW / 2 - 10}
                  y={y + 3}
                  fontSize={9}
                  textAnchor="end"
                  fill="currentColor"
                  fillOpacity={0.5}
                  fontFamily="monospace"
                >
                  h{i}
                </text>
              </g>
            );
          })}

          {/* The stream column itself, growing as `filled` grows. The one real
              owned container on this canvas, so SW.OUTER. */}
          <rect
            x={streamX - streamW / 2}
            y={topOfStreamY}
            width={streamW}
            height={baseY - topOfStreamY}
            rx={RX}
            fill={COLORS[Math.min(step, COLORS.length - 1)]}
            fillOpacity={0.4}
            stroke={COLORS[Math.min(step, COLORS.length - 1)]}
            strokeOpacity={0.85}
            strokeWidth={SW.OUTER}
            style={{ transition: growEase }}
          />

          {/* Floor line — "the stream starts here" */}
          <line
            x1={streamX - streamW / 2 - 18}
            y1={baseY}
            x2={streamX + streamW / 2 + 18}
            y2={baseY}
            stroke="currentColor"
            strokeOpacity={0.35}
            strokeWidth={SW.INNER}
          />
          <text x={streamX} y={baseY + 18} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.5}>
            {copy.entryLabel}
          </text>

          {/* Sub-block side write arrows. We mark them visible/active based on
              which step we're showing. Attention writes happen at step 1 and 3,
              MLP at 2 and 4. */}
          {STEP_LABELS.map((s, idx) => {
            if (s.kind === 'enter' || s.kind === 'exit') return null;
            // Step idx → which level of the stream this write lands at.
            // Write lands at heights[idx] (idx === 1..4 maps to the 1st..4th write).
            const landingY = baseY - heights[idx]! * 5;
            const isActive = step === idx;
            const wasDone = step > idx;
            const isAttn = s.kind === 'attn';
            const sideX = isAttn ? streamX - SIDE_OFFSET : streamX + SIDE_OFFSET;
            const arrowEndX = isAttn ? streamX - streamW / 2 - 2 : streamX + streamW / 2 + 2;
            const blockColor = isAttn ? ATTN_COLOR : MLP_COLOR;
            const opacity = isActive ? 1 : wasDone ? 0.35 : 0.12;
            // ONE string, used twice and identical both times: the static
            // connector that is always painted, and the dots that ride it on the
            // frame this write actually happens. Dots animate an edge that
            // exists; they never invent one.
            const writeD = `M ${isAttn ? sideX + 36 : sideX - 36} ${landingY} L ${arrowEndX} ${landingY}`;
            return (
              <g key={`write-${idx}`} style={{ transition: fadeEase, opacity }}>
                {/* Sub-block label box. NOT dashed even before its write lands:
                    the sub-block exists in every frame — it is a module of the
                    model. What has not happened yet is the WRITE, so the dash
                    goes on the connector below, not on this box. */}
                <rect
                  x={sideX - 36}
                  y={landingY - 12}
                  width={72}
                  height={24}
                  rx={RX}
                  fill={blockColor}
                  fillOpacity={isActive ? 0.25 : 0.12}
                  stroke={blockColor}
                  strokeOpacity={isActive ? 0.9 : 0.4}
                  strokeWidth={SW.OUTER}
                />
                <text
                  x={sideX}
                  y={landingY + 1}
                  fontSize={10}
                  textAnchor="middle"
                  fill={blockColor}
                  fontFamily="monospace"
                >
                  {isAttn ? 'attn(·)' : 'mlp(·)'}
                </text>
                {/* Arrow into the stream. Dashed until this write has happened:
                    dash is this course's spelling for "not real yet", and the
                    edge itself is always drawn so the reader can see where the
                    next write will land. */}
                <path
                  d={writeD}
                  fill="none"
                  stroke={blockColor}
                  strokeOpacity={isActive ? 0.95 : 0.45}
                  strokeWidth={isActive ? SW.OUTER : SW.INNER}
                  strokeDasharray={isActive || wasDone ? undefined : DASH.BORDER}
                  markerEnd={`url(#arrow-${isAttn ? 'attn' : 'mlp'})`}
                />
                {/* The correction in flight, on the frame it is being written.
                    In the writing sub-block's OWN hue, not the kit's FLOW_BLUE —
                    see the palette note at the top of this file. */}
                <FlowDots d={writeD} color={blockColor} hidden={!isActive} paused={!player.playing} spacing={12} />
                {/* "+= " / ":= " label */}
                {isActive ? (
                  <text
                    x={(sideX + arrowEndX) / 2}
                    y={landingY - 4}
                    fontSize={10}
                    textAnchor="middle"
                    fill={blockColor}
                    fontFamily="monospace"
                  >
                    {mode === 'add' ? '+=' : ':='}
                  </text>
                ) : null}
              </g>
            );
          })}

          {/* LM head reading at the top, lit when step === final */}
          {step >= STEP_LABELS.length - 1 ? (
            <g style={{ transition: fadeEase, opacity: step === STEP_LABELS.length - 1 ? 1 : 0.6 }}>
              <line
                x1={streamX}
                y1={topOfStreamY - 4}
                x2={streamX}
                y2={topY + 16}
                stroke="currentColor"
                strokeOpacity={0.7}
                strokeWidth={SW.OUTER}
                markerEnd="url(#arrow-out)"
              />
              {/* A bare rect, not PanelFrame: the title has to stay INSIDE it.
                  The box's top edge is at y=6, and a FrameLabel chip straddling
                  that line would reach y=-1 — outside the viewBox — and the
                  geometry that would fix it is not this conversion's to move.
                  The 0.08 fill is a tint behind the label, not a DIM step. */}
              <rect
                x={streamX - 50}
                y={topY - 14}
                width={100}
                height={30}
                rx={RX}
                fill="currentColor"
                fillOpacity={0.08}
                stroke="currentColor"
                strokeOpacity={0.5}
                strokeWidth={SW.OUTER}
              />
              <text
                x={streamX}
                y={topY + 5}
                fontSize={10}
                textAnchor="middle"
                fill="currentColor"
                fontFamily="monospace"
              >
                lm_head
              </text>
            </g>
          ) : null}

          {/* arrowhead markers */}
          <defs>
            <marker
              id="arrow-attn"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 Z" fill={ATTN_COLOR} fillOpacity={0.95} />
            </marker>
            <marker id="arrow-mlp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={MLP_COLOR} fillOpacity={0.95} />
            </marker>
            <marker id="arrow-out" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill="currentColor" fillOpacity={0.7} />
            </marker>
          </defs>
        </svg>
      </PanBox>
    </DiagramFrame>
  );
}
