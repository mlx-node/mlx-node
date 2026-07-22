import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

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
 * floats. Loop, play/pause.
 */

const STEPS = [
  'embedding lookup — read row of embed_tokens.weight',
  'flow through 24 transformer layers (residual stream)',
  'final RMSNorm + LM head — project against the SAME matrix',
  'top-K vocab scores — model predicts " floor"',
] as const;

const COPY = {
  en: {
    header: 'Weight tying — one matrix, used twice',
    pause: 'Pause',
    play: 'Play',
    pauseAria: 'Pause weight-tying animation',
    playAria: 'Play weight-tying animation',
    stepCounter: (step: number, total: number) => `step ${step}/${total}`,
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
    pause: '暂停',
    play: '播放',
    pauseAria: '暂停权重共享动画',
    playAria: '播放权重共享动画',
    stepCounter: (step: number, total: number) => `第 ${step}/${total} 步`,
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

export function WeightTyingVisual() {
  const copy = COPY[useLocale()];
  const [step, setStep] = React.useState(0);
  const [playing, setPlaying] = React.useState(() =>
    typeof window !== 'undefined' ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches : true,
  );

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setStep((s) => (s + 1) % STEPS.length);
    }, 1800);
    return () => window.clearInterval(t);
  }, [playing]);

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

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="inline-flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] max-sm:px-2.5 max-sm:py-1.5 max-sm:text-xs max-sm:min-h-[36px] hover:bg-muted/70"
            aria-pressed={playing}
            aria-label={playing ? copy.pauseAria : copy.playAria}
          >
            {playing ? copy.pause : copy.play}
          </button>
          <span className="font-mono text-[11px] text-muted-foreground">
            {copy.stepCounter(step + 1, STEPS.length)}
          </span>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.intro}</p>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={copy.svgAria}>
        {/* Stack of decoder layers in the middle */}
        <rect
          x={stackX}
          y={stackY}
          width={stackW}
          height={stackH}
          fill="currentColor"
          fillOpacity={0.05}
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        {Array.from({ length: 6 }, (_, i) => (
          <line
            key={`l-${i}`}
            x1={stackX}
            y1={stackY + (stackH / 6) * (i + 1)}
            x2={stackX + stackW}
            y2={stackY + (stackH / 6) * (i + 1)}
            stroke="currentColor"
            strokeOpacity={0.18}
            strokeDasharray="2 3"
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

        {/* Top matrix glyph — embedding lookup */}
        <rect
          x={topMatX}
          y={topMatY}
          width={matW}
          height={matH}
          fill="oklch(0.65 0.13 250)"
          fillOpacity={topActive ? 0.45 : 0.18}
          stroke="oklch(0.65 0.13 250)"
          strokeOpacity={topActive ? 1 : 0.6}
          strokeWidth={topActive ? 2 : 1}
          rx={3}
          style={{ transition: 'all 400ms ease-out' }}
        />
        {Array.from({ length: 12 }, (_, i) =>
          Array.from({ length: 4 }, (_, j) => (
            <rect
              key={`tg-${i}-${j}`}
              x={topMatX + 4 + i * 9.5}
              y={topMatY + 4 + j * 8}
              width={7}
              height={6}
              fill="oklch(0.65 0.13 250)"
              fillOpacity={(topActive ? 0.55 : 0.25) + 0.3 * Math.abs(Math.sin(i * 1.3 + j * 0.7))}
              style={{ transition: 'fill-opacity 400ms' }}
            />
          )),
        )}
        <text x={topMatX + matW / 2} y={topMatY - 4} fontSize={10} textAnchor="middle" fill="oklch(0.65 0.13 250)">
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
          strokeWidth={botActive ? 2 : 1}
          rx={3}
          style={{ transition: 'all 400ms ease-out' }}
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
              style={{ transition: 'fill-opacity 400ms' }}
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

        {/* Tied-weights arc connecting the two matrices — lights up when either matrix is "active" */}
        <path
          d={`M ${topMatX + matW / 2} ${topMatY + matH / 2} C ${15} ${H / 2}, ${15} ${H / 2}, ${botMatX + matW / 2} ${botMatY + matH / 2}`}
          fill="none"
          stroke="oklch(0.7 0.18 25)"
          strokeWidth={arcLit ? 2 : 1.2}
          strokeDasharray="5 4"
          strokeOpacity={arcLit ? 0.95 : 0.55}
          style={{ transition: 'stroke-opacity 400ms, stroke-width 400ms' }}
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

        {/* The traveling particle — small dot that pulses at the active step */}
        <circle
          cx={particle.x}
          cy={particle.y}
          r={9}
          fill="oklch(0.8 0.15 25)"
          fillOpacity={0.18}
          style={{ transition: 'cx 900ms cubic-bezier(0.4, 0, 0.2, 1), cy 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        <circle
          cx={particle.x}
          cy={particle.y}
          r={5}
          fill="oklch(0.7 0.18 25)"
          stroke="oklch(0.85 0.18 25)"
          strokeWidth={1.2}
          style={{ transition: 'cx 900ms cubic-bezier(0.4, 0, 0.2, 1), cy 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        <text
          x={particle.x}
          y={particle.y - 16}
          fontSize={10}
          textAnchor="middle"
          fill="oklch(0.85 0.18 25)"
          fontFamily="monospace"
          style={{ transition: 'x 900ms cubic-bezier(0.4, 0, 0.2, 1), y 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        >
          {particle.label}
        </text>
      </svg>

      <div
        className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-[12px] text-foreground/95"
        aria-live={playing ? 'off' : 'polite'}
        aria-atomic="true"
      >
        <span className="text-muted-foreground">{copy.stepPrefix(step + 1)}</span>
        {copy.steps[step]}
      </div>

      <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[12px]">{copy.savings}</div>

      <p className="text-[11px] text-muted-foreground">{copy.outro}</p>
    </div>
  );
}
