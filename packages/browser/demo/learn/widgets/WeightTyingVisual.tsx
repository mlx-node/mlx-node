import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { DASH, DiagramFrame, FLOW_BLUE, PanBox, RX, SW, useStepPlayer } from '../motion';

/**
 * Chapter 10 (LM head) supplement — show that for Qwen3.5 (and most modern
 * decoder-only LLMs) the embedding matrix and the LM head are the *same*
 * tensor, just transposed.
 *
 *   embed_tokens.weight : [V, d]  — bottom of the stack, "id → vector"
 *   lm_head.weight      : [V, d]  — top of the stack, "vector → vocab scores"
 *                                    (we use its transpose: [d, V])
 *
 * Verified for Qwen3.5: in `crates/mlx-core/src/models/qwen3_5/model.rs`,
 * when `tie_word_embeddings=true`, the model never allocates a separate
 * `lm_head` — it uses `embed_tokens.weight.T` directly for the final matmul.
 *
 * The widget animates a token particle traveling through the model: it gets
 * looked up via `embed_tokens.weight` at the top, flows through the 24-layer
 * stack, then gets projected back to a vocab score via the *same matrix*
 * (now `lm_head`) at the bottom. The dashed "tied — same tensor" arc lights
 * up whenever either matrix is "active" to drive home that both are the same
 * floats.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval`, its private
 * `matchMedia` read inside a `useState` initializer, and its own play button /
 * step counter / caption box / live region.
 *
 * ── THE PALETTE, AND WHY IT IS NOT THE TWO-HUE ONE ────────────────────────
 *
 * `skin` says a diagram gets RED and EMERALD and nothing else. This widget is a
 * documented carve-out and keeps its own hue. It used to keep TWO — blue for
 * `embed_tokens.weight` at the input, orange for `lm_head` at the output — but
 * that painted the misconception the widget exists to kill. There is no untied
 * state here: the label already reads `lm_head.weight (= embed_tokens.weight)`
 * and both cell grids are generated from the same `sin(i*1.3 + j*0.7)` texture,
 * so hue was the last thing still claiming two tensors, and the tie arc had to
 * argue the picture back down. One tensor, one hue.
 *
 * It is ORANGE, not blue, because the travelling particle is FLOW_BLUE
 * (`#3b82f6` ≈ `oklch(0.62 0.19 258)`) and the old blue was
 * `oklch(0.65 0.13 250)` — eight degrees of hue apart. Unifying to blue would
 * have parked a solid blue dot on a blue box at BOTH ends of its route. Active
 * vs idle rides opacity and stroke width, never hue, so nothing was lost.
 *
 * The one colour that DID move is the travelling particle, now `FLOW_BLUE` —
 * the kit's "data moving now" accent, which is explicitly exempt from the
 * two-hue rule because it is never a box colour. The dashed tie arc keeps its
 * own hue: it is a permanent identity link between the two matrices, not data
 * in flight, and FLOW_BLUE would assert the opposite.
 */

const STEPS = [
  'embedding lookup — read row of embed_tokens.weight',
  'flow through 24 transformer layers (residual stream)',
  'final RMSNorm + LM head — project against the SAME matrix',
  'top-K vocab scores — model predicts " floor"',
] as const;

// Play / Pause are NOT here any more: those verbs belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word. The old `step 1/4` counter is gone with them —
// `StepControls` has no counter slot, and `stepPrefix` already opens every
// caption with "Step 1: " / "第 1 步：".
const COPY = {
  en: {
    header: 'Weight tying — one matrix, used twice',
    intro: (
      <>
        Qwen3.5-0.8B (and most modern decoder LLMs) sets <span className="font-mono">tie_word_embeddings = true</span>.
        That means the embedding matrix at the input and the LM head at the output are{' '}
        <em>literally the same tensor</em> in memory — the same <span className="font-mono">[248,320 × 1024]</span> grid
        of floats, used once for <span className="font-mono">id → vector</span> and once (transposed) for{' '}
        <span className="font-mono">vector → vocab scores</span>.
      </>
    ),
    svgAria: 'Weight tying animation showing one tensor used at both ends of the model',
    layersLabel: '24 layers',
    embedSubLabel: 'token id → vector (lookup)',
    lmHeadSubLabel: 'vector → vocab scores',
    tiedLabel: 'tied — same tensor',
    stepPrefix: (step: number) => `Step ${step}: `,
    steps: STEPS,
    savings: (
      <>
        <strong>Parameter savings:</strong> the matrix is{' '}
        <span className="font-mono">248,320 × 1024 ≈ 254.3M floats</span>. Tying skips a second copy at the LM head — a
        ~254M-parameter reduction on a 0.8B-parameter model. That's close to a third of the model, gone, just by reusing
        the dictionary.
      </>
    ),
    outro: (
      <>
        Conceptually tying says:{' '}
        <em>
          the same dictionary that maps a token id to its incoming representation also maps an outgoing representation
          back to a vocab score
        </em>
        . Reading and writing share one alphabet. Not every model ties — large GPT-style models sometimes keep them
        separate for a small quality win — but for sub-billion-parameter models, tying is the standard.
      </>
    ),
  },
  zh: {
    header: '权重共享——同一个矩阵，用两次',
    intro: (
      <>
        Qwen3.5-0.8B（以及大多数现代解码器 LLM）设置了{' '}
        <span className="font-mono">tie_word_embeddings = true</span>。这意味着输入端的嵌入矩阵和输出端的 LM head
        在内存中<em>就是同一个张量</em>——同一个 <span className="font-mono">[248,320 × 1024]</span>{' '}
        的浮点数网格，一次用于 <span className="font-mono">id → vector</span>，一次（转置后）用于{' '}
        <span className="font-mono">vector → vocab scores</span>。
      </>
    ),
    svgAria: '权重共享动画：同一个张量在模型两端各用一次',
    layersLabel: '24 层',
    embedSubLabel: 'token id → 向量（查表）',
    lmHeadSubLabel: '向量 → 词表分数',
    tiedLabel: '共享——同一个张量',
    stepPrefix: (step: number) => `第 ${step} 步：`,
    steps: [
      '嵌入查表——读取 embed_tokens.weight 的一行',
      '流经 24 个 transformer 层（残差流）',
      '最终 RMSNorm + LM head——对着同一个矩阵做投影',
      'top-K 词表分数——模型预测出 " floor"',
    ],
    savings: (
      <>
        <strong>参数节省：</strong>这个矩阵有 <span className="font-mono">248,320 × 1024 ≈ 254.3M</span>{' '}
        个浮点数。权重共享省掉了 LM head 处的第二份拷贝——在一个 0.8B 参数的模型上少了约 254M
        个参数。仅仅是复用这本字典，就省掉了接近模型三分之一的参数。
      </>
    ),
    outro: (
      <>
        从概念上讲，权重共享是在说：
        <em>把 token id 映射成输入表示的那本字典，同样把输出表示映射回词表分数</em>
        。读和写共用同一套字母表。并非所有模型都做共享——大型 GPT
        风格的模型有时为了一点质量提升而把两者分开——但对参数量低于十亿的模型来说，共享是标准做法。
      </>
    ),
  },
} as const;

/** Unchanged from the hand-rolled `setInterval` this replaces. */
const FRAME_MS = 1800;

/** The particle's travel curve, shared by the two circles and the label. */
const MOVE_EASING = 'cubic-bezier(0.4, 0, 0.2, 1)';

/**
 * The still a reduced-motion reader is parked on. NOT the kit's default (the
 * LAST frame): step 3 is the one frame where nothing this widget is about is
 * emphasised — `topActive`, `botActive` and `arcLit` are all false, so both
 * matrix glyphs sit at their idle 0.18 fill and the tie arc drops to SW.INNER
 * at 0.55, and the particle has already left the diagram carrying `" floor"`.
 * That is the epilogue, not the point.
 *
 * Step 2 is the point, and it is the only frame that renders all three parts of
 * it at once: `botActive` lights the bottom glyph — whose own label reads
 * `lm_head.weight (= embed_tokens.weight)` — `arcLit` puts the dashed "tied —
 * same tensor" arc at full weight back up to the embedding matrix, and the
 * caption is "final RMSNorm + LM head — project against the SAME matrix". Step
 * 0 also lights the arc, but only the INPUT end of it: the reuse the widget
 * exists to show has not happened yet there.
 */
const REST_FRAME = 2;

export function WeightTyingVisual() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(STEPS.length, { frameMs: FRAME_MS, restFrame: REST_FRAME });
  const step = player.frame;

  // Every transition is gated on the preference. Reduced-motion readers are
  // retargeted to the resting frame in an effect AFTER mount, so an ungated
  // transition would animate the particle right across the diagram once, on
  // load, for exactly the readers who asked for less of that.
  const easeBox = player.reducedMotion ? undefined : 'all 400ms ease-out';
  const easeCell = player.reducedMotion ? undefined : 'fill-opacity 400ms';
  const easeArc = player.reducedMotion ? undefined : 'stroke-opacity 400ms, stroke-width 400ms';
  const easeMove = player.reducedMotion ? undefined : `cx 900ms ${MOVE_EASING}, cy 900ms ${MOVE_EASING}`;
  const easeMoveText = player.reducedMotion ? undefined : `x 900ms ${MOVE_EASING}, y 900ms ${MOVE_EASING}`;

  const W = 540;
  const H = 320;

  const matW = 120;
  const matH = 38;

  const topMatX = 50;
  const topMatY = 30;

  const botMatX = W - 50 - matW;
  const botMatY = H - 30 - matH;

  // Vertical "stack" between them suggests the decoder layers run vertically
  // between embedding (top of input) and LM head (bottom of stack to logits).
  const stackX = W / 2 - 22;
  const stackY = 90;
  const stackW = 44;
  const stackH = 140;

  // Particle position: lerp through 4 anchor points keyed off `step`.
  // step 0 → entering top matrix.   pos ≈ topMatX + matW/2, topMatY + matH/2
  // step 1 → halfway through stack. pos ≈ stackX + stackW/2, stackY + stackH/2
  // step 2 → at bottom matrix.       pos ≈ botMatX + matW/2, botMatY + matH/2
  // step 3 → past bottom matrix (output).
  const ANCHORS: Array<{ x: number; y: number; label: string }> = [
    { x: topMatX + matW / 2, y: topMatY + matH / 2, label: '"the"' },
    { x: stackX + stackW / 2, y: stackY + stackH / 2, label: 'h₀ … h₂₃' },
    { x: botMatX + matW / 2, y: botMatY + matH / 2, label: 'h_last' },
    { x: botMatX + matW + 30, y: botMatY + matH / 2, label: '" floor"' },
  ];
  const particle = ANCHORS[step]!;
  const topActive = step === 0;
  const botActive = step === 2;
  const arcLit = topActive || botActive;

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the beat changes.
  //
  // Coverage is BY CONSTRUCTION rather than by enumeration: the visible caption
  // is `captions[step]`, `step` is `player.frame` which `useStepPlayer` keeps in
  // [0, STEPS.length), and this array is exactly `copy.steps` — the same list,
  // same length, both locales. There is no branch to miss, and nothing to keep
  // in sync when a step is reworded.
  const steps: readonly string[] = copy.steps;
  const captions: React.ReactNode[] = steps.map((s, i) => (
    <>
      <span className="font-medium">{copy.stepPrefix(i + 1)}</span>
      {s}
    </>
  ));

  return (
    <>
      {/* Body-sized teaching prose, not a footnote — so it stays a sibling
          OUTSIDE the frame rather than getting demoted into `note`. It also has
          to stay ABOVE the diagram, which `note` (which renders last) cannot
          do. Same for the savings callout and the closing paragraph below. */}
      <p className="text-[12px] text-foreground/85">{copy.intro}</p>

      <DiagramFrame title={copy.header} player={player} locale={locale} caption={captions[step]} captions={captions}>
        {/* `min-w` is why this svg — and ONLY this svg — is wrapped in
            `PanBox`. A bare `w-full` svg does not reflow in a narrow column —
            it SCALES, text and all, so at a 375px viewport (a 320-unit column)
            the 9px labels land at 5.3 CSS px and stop being readable. The floor
            stops the shrink and `PanBox` pans to reach the rest.

            8px legibility floor / smallest size a reader must read:
            8 * 540 / 9 = 480. The 9 is real reading text three times over —
            "24 layers" inside the stack, and the two sub-labels under the
            matrices ("token id → vector", "vector → vocab scores") — so none of
            it gets the superscript exemption. It stays 9 rather than moving up:
            480 already fits inside the 540 viewBox, and the divider-opacity
            note below shows how little room "24 layers" has to grow into.

            The floor belongs to the svg ALONE, which is why `PanBox` wraps the
            svg and nothing else. This widget has no sibling controls today, but
            the rule is the point — anything added next to the drawing stays a
            child of DiagramFrame, sizes to the visible width, keeps the frame's
            `space-y-4` gaps, and stays reachable without panning. */}
        <PanBox locale={locale}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="block h-auto w-full min-w-[480px]"
            role="img"
            aria-label={copy.svgAria}
          >
            {/* Stack of decoder layers in the middle. A bare rect, not a
                PanelFrame: this is CONTEXT the particle passes through, not a
                stage, and PanelFrame's double border would not fit inside 44
                units next to the six divider lines anyway. */}
            <rect
              x={stackX}
              y={stackY}
              width={stackW}
              height={stackH}
              rx={RX}
              fill="currentColor"
              fillOpacity={0.05}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeWidth={SW.INNER}
            />
            {Array.from({ length: 6 }, (_, i) => (
              <line
                key={`l-${i}`}
                x1={stackX}
                y1={stackY + (stackH / 6) * (i + 1)}
                x2={stackX + stackW}
                y2={stackY + (stackH / 6) * (i + 1)}
                stroke="currentColor"
                // NOT DIM.IDLE_EDGE (0.4), deliberately: divider 3 of 6 lands on
                // y = 160 and the "24 layers" baseline is y = 164, so the line
                // runs through the digits. At 0.18 that is invisible; at 0.4 it
                // is a strike-through. FrameLabel is the usual fix for a label on
                // a stroke, but its plate for "24 layers" estimates ~76 units
                // wide against a 44-unit box, so it would erase the stack's own
                // side walls. The dim value stays until the geometry changes.
                strokeOpacity={0.18}
                strokeWidth={SW.INNER}
                strokeDasharray={DASH.SLOT}
              />
            ))}
            <text
              x={stackX + stackW / 2}
              y={stackY + stackH / 2 + 4}
              fontSize={9}
              textAnchor="middle"
              fill="currentColor"
              fillOpacity={0.55}
              fontFamily="monospace"
            >
              {copy.layersLabel}
            </text>

            {/* Top matrix glyph — embedding lookup. Same hue as the bottom glyph
                because it is the same tensor; `topActive` carries "in use now" on
                opacity and stroke width, never on hue. `2 : 1` maps exactly onto
                SW.OUTER : SW.INNER. */}
            <rect
              x={topMatX}
              y={topMatY}
              width={matW}
              height={matH}
              fill="oklch(0.7 0.15 60)"
              fillOpacity={topActive ? 0.45 : 0.18}
              stroke="oklch(0.7 0.15 60)"
              strokeOpacity={topActive ? 1 : 0.6}
              strokeWidth={topActive ? SW.OUTER : SW.INNER}
              rx={RX}
              style={{ transition: easeBox }}
            />
            {Array.from({ length: 12 }, (_, i) =>
              Array.from({ length: 4 }, (_, j) => (
                <rect
                  key={`tg-${i}-${j}`}
                  x={topMatX + 4 + i * 9.5}
                  y={topMatY + 4 + j * 8}
                  width={7}
                  height={6}
                  fill="oklch(0.7 0.15 60)"
                  // A per-cell texture, not a step on the DIM ladder — and NOT
                  // given rx={RX}: at 7x6 a non-zero radius turns every cell into
                  // a blob, so these stay square by construction.
                  fillOpacity={(topActive ? 0.55 : 0.25) + 0.3 * Math.abs(Math.sin(i * 1.3 + j * 0.7))}
                  style={{ transition: easeCell }}
                />
              )),
            )}
            {/* Bare <text>, not FrameLabel: these are literal checkpoint tensor
                names, and FrameLabel uppercases ("EMBED_TOKENS.WEIGHT") and would
                plate ~177 units of chip across a 120-unit box. */}
            <text x={topMatX + matW / 2} y={topMatY - 4} fontSize={10} textAnchor="middle" fill="oklch(0.7 0.15 60)">
              embed_tokens.weight
            </text>
            <text
              x={topMatX + matW / 2}
              y={topMatY + matH + 12}
              fontSize={9}
              textAnchor="middle"
              fill="currentColor"
              fillOpacity={0.55}
            >
              {copy.embedSubLabel}
            </text>

            {/* Bottom matrix glyph — LM head (transpose of same matrix) */}
            <rect
              x={botMatX}
              y={botMatY}
              width={matW}
              height={matH}
              fill="oklch(0.7 0.15 60)"
              fillOpacity={botActive ? 0.45 : 0.18}
              stroke="oklch(0.7 0.15 60)"
              strokeOpacity={botActive ? 1 : 0.6}
              strokeWidth={botActive ? SW.OUTER : SW.INNER}
              rx={RX}
              style={{ transition: easeBox }}
            />
            {Array.from({ length: 12 }, (_, i) =>
              Array.from({ length: 4 }, (_, j) => (
                <rect
                  key={`bg-${i}-${j}`}
                  x={botMatX + 4 + i * 9.5}
                  y={botMatY + 4 + j * 8}
                  width={7}
                  height={6}
                  fill="oklch(0.7 0.15 60)"
                  fillOpacity={(botActive ? 0.55 : 0.25) + 0.3 * Math.abs(Math.sin(i * 1.3 + j * 0.7))}
                  style={{ transition: easeCell }}
                />
              )),
            )}
            <text x={botMatX + matW / 2} y={botMatY - 4} fontSize={10} textAnchor="middle" fill="oklch(0.7 0.15 60)">
              lm_head.weight (= embed_tokens.weight)
            </text>
            <text
              x={botMatX + matW / 2}
              y={botMatY + matH + 12}
              fontSize={9}
              textAnchor="middle"
              fill="currentColor"
              fillOpacity={0.55}
            >
              {copy.lmHeadSubLabel}
            </text>

            {/* Tied-weights arc connecting the two matrices — lights up when
                either matrix is "active". Still a bezier: `elbow()` emits
                down-across-down and this link is out-left-down-right, which it
                cannot express, and squaring it by hand is a geometry redesign
                rather than a token swap. Flagged in the report. */}
            <path
              d={`M ${topMatX + matW / 2} ${topMatY + matH / 2} C ${15} ${H / 2}, ${15} ${H / 2}, ${botMatX + matW / 2} ${botMatY + matH / 2}`}
              fill="none"
              stroke="oklch(0.7 0.18 25)"
              strokeWidth={arcLit ? SW.OUTER : SW.INNER}
              strokeDasharray={DASH.BORDER}
              strokeOpacity={arcLit ? 0.95 : 0.55}
              style={{ transition: easeArc }}
            />
            <text
              x={28}
              y={H / 2 - 6}
              fontSize={10}
              fill="oklch(0.7 0.18 25)"
              fillOpacity={arcLit ? 1 : 0.7}
              transform={`rotate(-90, 28, ${H / 2 - 6})`}
            >
              {copy.tiedLabel}
            </text>

            {/* The traveling particle — FLOW_BLUE, the kit's "data moving now"
                accent. NOT FlowDots: the dots are anonymous and this particle
                carries a per-step label ("the" -> h₀…h₂₃ -> h_last -> " floor")
                that is the lesson, and there is no static connector drawn along
                its route for the dots to ride (the kit forbids inventing one). */}
            <circle
              cx={particle.x}
              cy={particle.y}
              r={9}
              fill={FLOW_BLUE}
              fillOpacity={0.18}
              style={{ transition: easeMove }}
            />
            <circle
              cx={particle.x}
              cy={particle.y}
              r={5}
              fill={FLOW_BLUE}
              // A background ring, not a lighter tint of the dot. On step 0 the
              // particle sits ON the blue embed matrix, so a same-hue ring would
              // vanish; `--background` separates it from whatever it lands on in
              // either theme.
              stroke="var(--background)"
              strokeWidth={SW.INNER}
              style={{ transition: easeMove }}
            />
            <text
              x={particle.x}
              y={particle.y - 16}
              fontSize={10}
              textAnchor="middle"
              fill={FLOW_BLUE}
              fontFamily="monospace"
              style={{ transition: easeMoveText }}
            >
              {particle.label}
            </text>
          </svg>
        </PanBox>
      </DiagramFrame>

      <div className="my-4 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[12px]">
        {copy.savings}
      </div>

      <p className="text-[11px] text-muted-foreground">{copy.outro}</p>
    </>
  );
}
