import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { FlowDots, HatchDefs, StepControls, hatchFill, useStepPlayer } from '../motion';

/**
 * ChunkedPrefillEngine — one engine step at a time: a long prompt is prefilled
 * in CHUNKS, each chunk shares its forward pass with other requests' decode
 * steps, its keys/values land in fixed-size KV blocks, and the first output
 * token is sampled only after the LAST chunk.
 *
 * This is the animation the rest of the sub-chapter is written around, and it
 * is the first widget built on `learn/motion/` (useStepPlayer + FlowDots +
 * HatchDefs) rather than hand-rolling its own timer and reduced-motion hook.
 *
 * WHAT IS BEING TAUGHT (and what is a simplification — see `copy.note`):
 *
 *  - Prefill does NOT have to be one giant forward pass. A scheduler with a
 *    per-step token budget splits it, so a 100k-token prompt cannot monopolise
 *    the GPU and stall everyone else's decode ("generation stall").
 *  - Every chunk still WRITES its keys/values to the cache — that is the whole
 *    point of the pass. Nothing is recomputed later.
 *  - Only the FINAL chunk produces a sampled token, because the logits that
 *    matter live at the last prompt position. Intermediate chunks are pure
 *    cache-population.
 *  - The KV cache is paged: fixed-size blocks of `BLOCK` tokens, filled left to
 *    right, with the last block usually partly empty. That partial block is the
 *    only internal waste — bounded by one block per sequence, instead of the
 *    reserve-for-the-worst-case waste of a contiguous cache.
 *
 * The token counts (18 / 8 / 4) are ILLUSTRATIVE — chosen so the arithmetic is
 * visible at a glance (8 + 8 + 2, and 18 tokens = 4 full blocks + a half-full
 * 5th). Real engines use budgets in the thousands and block sizes of 16 (see
 * the sub-chapter prose). The SHAPE of the animation is what generalises, not
 * the numbers.
 *
 * Self-contained and SSR-safe: NO worker, NO model, NO WASM, NO network. Pure
 * React + SVG, so it server-renders for crawlers (createRoot replaces it on
 * boot). The first render is a fixed frame, identical server and client;
 * reduced-motion readers are retargeted to the finished picture after mount.
 */

// ── Configuration. Every number the widget teaches lives here. ────────────────
const PROMPT_TOKENS = 18;
const CHUNK = 8; // per-engine-step token budget
const BLOCK = 4; // KV cache block size, in tokens

/** Chunk sizes: full budget until the remainder. → [8, 8, 2] */
const CHUNKS: number[] = (() => {
  const out: number[] = [];
  for (let left = PROMPT_TOKENS; left > 0; left -= CHUNK) out.push(Math.min(CHUNK, left));
  return out;
})();

const N_BLOCKS = Math.ceil(PROMPT_TOKENS / BLOCK); // 5

/** Cumulative tokens cached after chunk i completes. → [8, 16, 18] */
const CUMULATIVE: number[] = CHUNKS.reduce<number[]>((acc, n) => [...acc, (acc[acc.length - 1] ?? 0) + n], []);

// ── Frame model ──────────────────────────────────────────────────────────────
// Each chunk plays three beats, mirroring what the engine actually does:
//   load    — the scheduler picks this chunk's tokens (dots leave the prompt)
//   compute — ONE forward pass over this chunk + other requests' decode steps
//   write   — the resulting K/V land in cache blocks
// then one final beat samples the first token.
type Phase = 'idle' | 'load' | 'compute' | 'write' | 'sample';
type Frame = { phase: Phase; chunk: number };

const FRAMES: Frame[] = [
  { phase: 'idle', chunk: -1 },
  ...CHUNKS.flatMap((_, i): Frame[] => [
    { phase: 'load', chunk: i },
    { phase: 'compute', chunk: i },
    { phase: 'write', chunk: i },
  ]),
  { phase: 'sample', chunk: CHUNKS.length - 1 },
];
const TOTAL_FRAMES = FRAMES.length; // 11
const FRAME_MS = 1400;

// ── Geometry ─────────────────────────────────────────────────────────────────
const VB_W = 600;
const VB_H = 436;

// Prompt strip (top)
const P_X = 44;
const P_Y = 26;
const P_W = VB_W - P_X - 24;
const P_H = 54;
const CELL_GAP = 2;
const CELL_W = (P_W - 20 - CELL_GAP * (PROMPT_TOKENS - 1)) / PROMPT_TOKENS;
const CELL_H = 22;
const CELL_Y = P_Y + P_H - CELL_H - 9;
const cellX = (i: number) => P_X + 10 + i * (CELL_W + CELL_GAP);

// Forward pass (centre)
const F_X = 228;
const F_Y = 186;
const F_W = 150;
const F_H = 74;

// Other requests (left)
const O_X = 24;
const O_Y = 186;
const O_W = 132;
const O_H = 74;

// Output (right)
const T_X = 452;
const T_W = 108;
const T_H = 30;
const T_Y = F_Y + (F_H - T_H) / 2;

// KV cache (bottom)
const K_X = 132;
const K_Y = 306;
const K_W = 336;
// Tall enough that the "block size" caption clears the block row (ends at 396)
// AND sits inside the inner dashed border (ends at K_Y + K_H - 6).
const K_H = 120;
const BLK_GAP = 8;
const BLK_W = (K_W - 24 - BLK_GAP * (N_BLOCKS - 1)) / N_BLOCKS;
const BLK_H = 46;
const BLK_Y = K_Y + 44;
const blkX = (b: number) => K_X + 12 + b * (BLK_W + BLK_GAP);
const SLOT_GAP = 2;
const SLOT_W = (BLK_W - 10 - SLOT_GAP * (BLOCK - 1)) / BLOCK;
const SLOT_H = 14;

const RED = '#ef4444'; // red-500 — the expensive compute box
const EMERALD = '#10b981'; // emerald-500 — cached / done

// ── Copy. zh translates prose only; every glossary term stays English. ───────
const COPY = {
  en: {
    heading: 'Chunked prefill — one engine step at a time',
    budget: `≤ ${CHUNK} tokens per engine step`,
    prompt: `prompt · ${PROMPT_TOKENS} tokens`,
    others: 'other requests',
    forward: ['LLM', 'forward pass'],
    kv: 'paged KV cache',
    blockSize: `block size = ${BLOCK} tokens`,
    stepLabel: (i: number, n: number) => `engine step ${i} / ${n}`,
    firstToken: 'first token',
    sampledNote: ['sampled only after', 'the last chunk'],
    phase: {
      idle: 'A long prompt does not have to be one giant forward pass.',
      load: (i: number, n: number) =>
        `Step ${i}: the scheduler takes the next ${n} tokens — never more than the budget.`,
      compute: 'One forward pass. Other requests ride along in the same batch, so their decode never stalls.',
      write: 'The keys and values for this chunk land in cache blocks. Nothing is recomputed later.',
      sample: 'Only now — after the final chunk — are logits sampled into the first output token.',
    },
    legendActive: 'busy this step',
    legendCached: 'cached',
    note: `Sizes are illustrative: ${PROMPT_TOKENS} tokens, a ${CHUNK}-token budget and ${BLOCK}-token blocks keep the arithmetic visible (${CHUNKS.join(' + ')} = ${PROMPT_TOKENS}). Real engines use budgets in the thousands and 16-token blocks. The shape generalises; the numbers do not.`,
    svgAria: (s: string) =>
      `An animated engine diagram. An ${PROMPT_TOKENS}-token prompt is prefilled in ${CHUNKS.length} chunks of at most ${CHUNK} tokens. Each chunk flows into a single LLM forward pass shared with other requests, then its keys and values fill fixed-size blocks of a paged KV cache. The first output token is sampled only after the final chunk. Current step: ${s}`,
  },
  zh: {
    heading: 'Chunked prefill——一次一个 engine step',
    budget: `每个 engine step ≤ ${CHUNK} tokens`,
    prompt: `prompt · ${PROMPT_TOKENS} tokens`,
    others: '其他请求',
    forward: ['LLM', 'forward pass'],
    kv: 'paged KV cache',
    blockSize: `block size = ${BLOCK} tokens`,
    stepLabel: (i: number, n: number) => `engine step ${i} / ${n}`,
    firstToken: 'first token',
    sampledNote: ['只在最后一个 chunk', '之后才采样'],
    phase: {
      idle: '一个很长的 prompt，不一定要挤进一次巨大的 forward pass。',
      load: (i: number, n: number) => `第 ${i} 步：scheduler 取走接下来的 ${n} 个 tokens——绝不超过预算。`,
      compute: '一次 forward pass。其他请求搭同一趟车，它们的 decode 因此不会被卡住。',
      write: '这个 chunk 的 keys 和 values 写进 cache blocks。后面不会重算。',
      sample: '直到现在——最后一个 chunk 之后——logits 才被采样成第一个输出 token。',
    },
    legendActive: '本步正忙',
    legendCached: '已缓存',
    note: `这里的尺寸只是示意：${PROMPT_TOKENS} tokens、${CHUNK} tokens 的预算、${BLOCK} tokens 的 block，是为了让算术一眼看得清（${CHUNKS.join(' + ')} = ${PROMPT_TOKENS}）。真实引擎的预算是几千，block 通常是 16 tokens。可以推广的是这个形状，不是这些数字。`,
    svgAria: (s: string) =>
      `一张动画引擎示意图。${PROMPT_TOKENS} 个 tokens 的 prompt 被拆成 ${CHUNKS.length} 个 chunk，每个最多 ${CHUNK} 个 tokens。每个 chunk 进入一次与其他请求共享的 LLM forward pass，然后它的 keys 和 values 填进 paged KV cache 的固定大小 block。第一个输出 token 只在最后一个 chunk 之后才被采样。当前步骤：${s}`,
  },
} as const;

/** Elbow path: down from `from`, across, then into `to`. */
function elbow(x1: number, y1: number, x2: number, y2: number, midY: number): string {
  return `M ${x1} ${y1} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2}`;
}

export function ChunkedPrefillEngine() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS, initialFrame: 0 });
  const { phase, chunk } = FRAMES[player.frame];

  // Tokens whose K/V are already in the cache. `write` fills during its own
  // beat, so the count includes the current chunk from `write` onward.
  const cached =
    phase === 'idle' ? 0 : phase === 'write' || phase === 'sample' ? CUMULATIVE[chunk] : (CUMULATIVE[chunk - 1] ?? 0);

  // Tokens the current chunk covers, as a [start, end) range on the prompt.
  const chunkStart = chunk > 0 ? CUMULATIVE[chunk - 1] : 0;
  const chunkEnd = chunk >= 0 ? CUMULATIVE[chunk] : 0;
  const chunkActive = phase === 'load' || phase === 'compute' || phase === 'write';

  const fwdBusy = phase === 'compute';
  const kvBusy = phase === 'write';
  const done = phase === 'sample';

  const caption =
    phase === 'idle'
      ? copy.phase.idle
      : phase === 'load'
        ? copy.phase.load(chunk + 1, CHUNKS[chunk])
        : phase === 'compute'
          ? copy.phase.compute
          : phase === 'write'
            ? copy.phase.write
            : copy.phase.sample;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <StepControls player={player} locale={locale} showReset />
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria(caption)}>
        <HatchDefs />

        {/* budget label, top-left — appears once the sweep starts */}
        <text
          x={P_X}
          y={16}
          style={{ fill: RED, opacity: phase === 'idle' ? 0 : 1, transition: 'opacity 300ms ease' }}
          className="text-[10px] font-semibold uppercase tracking-wider"
        >
          {copy.budget}
        </text>

        {/* ── Prompt strip ────────────────────────────────────────────────── */}
        <rect
          x={P_X}
          y={P_Y}
          width={P_W}
          height={P_H}
          rx={6}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <rect
          x={P_X + 5}
          y={P_Y + 5}
          width={P_W - 10}
          height={P_H - 10}
          rx={4}
          fill="none"
          style={{ stroke: 'var(--muted-foreground)', opacity: 0.45 }}
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        <text
          x={P_X + P_W / 2}
          y={P_Y + 17}
          textAnchor="middle"
          style={{ fill: 'var(--foreground)' }}
          className="text-[10px] font-semibold uppercase tracking-wider"
        >
          {copy.prompt}
        </text>

        {Array.from({ length: PROMPT_TOKENS }, (_, i) => {
          const inChunk = chunkActive && i >= chunkStart && i < chunkEnd;
          const isCached = i < cached;
          return (
            <g key={i}>
              <rect
                x={cellX(i)}
                y={CELL_Y}
                width={CELL_W}
                height={CELL_H}
                rx={2}
                style={{
                  fill: inChunk ? 'var(--muted)' : 'var(--background)',
                  stroke: isCached && !inChunk ? 'var(--border)' : 'var(--muted-foreground)',
                  opacity: isCached && !inChunk ? 0.45 : 1,
                  transition: player.reducedMotion ? undefined : 'fill 250ms ease, opacity 250ms ease',
                }}
                strokeWidth={1}
              />
              <text
                x={cellX(i) + CELL_W / 2}
                y={CELL_Y + CELL_H / 2 + 3}
                textAnchor="middle"
                style={{ fill: 'var(--muted-foreground)', opacity: isCached && !inChunk ? 0.5 : 1 }}
                className="text-[8px]"
              >
                {i + 1}
              </text>
            </g>
          );
        })}

        {/* red group-outline around the chunk being scheduled */}
        {chunkActive ? (
          <rect
            x={cellX(chunkStart) - 3}
            y={CELL_Y - 3}
            width={(chunkEnd - chunkStart) * (CELL_W + CELL_GAP) - CELL_GAP + 6}
            height={CELL_H + 6}
            rx={3}
            fill="none"
            stroke={RED}
            strokeWidth={1.5}
            style={{ transition: player.reducedMotion ? undefined : 'x 300ms ease, width 300ms ease' }}
          />
        ) : null}

        {/* Engine step counter. Sits BELOW the output box — at the forward
            pass's vertical centre it would print straight over it. */}
        <text
          x={VB_W - 24}
          y={T_Y + T_H + 24}
          textAnchor="end"
          style={{ fill: RED, opacity: chunk >= 0 ? 1 : 0, transition: 'opacity 300ms ease' }}
          className="text-[11px] font-semibold uppercase tracking-wider"
        >
          {chunk >= 0 ? copy.stepLabel(chunk + 1, CHUNKS.length) : ''}
        </text>

        {/* ── Other requests ──────────────────────────────────────────────── */}
        <rect
          x={O_X}
          y={O_Y}
          width={O_W}
          height={O_H}
          rx={6}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text
          x={O_X + O_W / 2}
          y={O_Y + 20}
          textAnchor="middle"
          style={{ fill: 'var(--foreground)' }}
          className="text-[10px] font-semibold uppercase tracking-wider"
        >
          {copy.others}
        </text>
        {['R7', 'R8'].map((r, i) => (
          <g key={r}>
            <rect
              x={O_X + 18 + i * 52}
              y={O_Y + 36}
              width={44}
              height={22}
              rx={3}
              style={{
                fill: fwdBusy ? 'var(--muted)' : 'var(--background)',
                stroke: 'var(--muted-foreground)',
                transition: player.reducedMotion ? undefined : 'fill 250ms ease',
              }}
              strokeWidth={1}
            />
            <text
              x={O_X + 18 + i * 52 + 22}
              y={O_Y + 51}
              textAnchor="middle"
              style={{ fill: 'var(--muted-foreground)' }}
              className="text-[9px] font-mono"
            >
              {r}
            </text>
          </g>
        ))}

        {/* ── Forward pass ────────────────────────────────────────────────── */}
        <rect
          x={F_X}
          y={F_Y}
          width={F_W}
          height={F_H}
          rx={6}
          style={{ fill: 'var(--card)', stroke: RED }}
          strokeWidth={1.5}
        />
        {fwdBusy ? <rect x={F_X} y={F_Y} width={F_W} height={F_H} rx={6} fill={hatchFill('red')} /> : null}
        <rect
          x={F_X + 6}
          y={F_Y + 6}
          width={F_W - 12}
          height={F_H - 12}
          rx={4}
          fill="none"
          style={{ stroke: RED, opacity: 0.5 }}
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        {copy.forward.map((line, i) => (
          <text
            key={line}
            x={F_X + F_W / 2}
            y={F_Y + F_H / 2 - 4 + i * 14}
            textAnchor="middle"
            style={{ fill: 'var(--foreground)' }}
            className="text-[11px] font-semibold uppercase tracking-wider"
          >
            {line}
          </text>
        ))}

        {/* ── Output ──────────────────────────────────────────────────────── */}
        <rect
          x={T_X}
          y={T_Y}
          width={T_W}
          height={T_H}
          rx={4}
          fill="none"
          style={{
            stroke: done ? RED : 'var(--muted-foreground)',
            opacity: done ? 1 : 0.5,
            transition: player.reducedMotion ? undefined : 'stroke 300ms ease',
          }}
          strokeWidth={done ? 1.5 : 1}
          strokeDasharray={done ? undefined : '4 4'}
        />
        {done ? (
          <>
            <text
              x={T_X + T_W / 2}
              y={T_Y + T_H / 2 + 4}
              textAnchor="middle"
              style={{ fill: 'var(--foreground)' }}
              className="text-[10px] font-semibold uppercase tracking-wider"
            >
              {copy.firstToken}
            </text>
            {copy.sampledNote.map((line, i) => (
              <text
                key={line}
                x={T_X + T_W / 2}
                y={T_Y - 30 + i * 13}
                textAnchor="middle"
                style={{ fill: RED }}
                className="text-[9px] font-medium uppercase tracking-wider"
              >
                {line}
              </text>
            ))}
          </>
        ) : null}

        {/* ── Static connectors ───────────────────────────────────────────── */}
        {[
          `M ${O_X + O_W} ${O_Y + 30} L ${F_X} ${O_Y + 30}`,
          `M ${O_X + O_W} ${O_Y + 52} L ${F_X} ${O_Y + 52}`,
          `M ${F_X + F_W} ${T_Y + T_H / 2} L ${T_X} ${T_Y + T_H / 2}`,
          `M ${F_X + F_W / 2} ${F_Y + F_H} L ${F_X + F_W / 2} ${K_Y}`,
        ].map((d) => (
          <path
            key={d}
            d={d}
            fill="none"
            style={{ stroke: 'var(--muted-foreground)', opacity: 0.4 }}
            strokeWidth={1}
            strokeDasharray="4 4"
          />
        ))}

        {/* ── Travelling dots: the "happening right now" layer ────────────── */}
        {/* prompt chunk → forward pass */}
        <FlowDots
          hidden={phase !== 'load'}
          d={elbow(
            chunk >= 0 ? cellX(chunkStart) + ((chunkEnd - chunkStart) * (CELL_W + CELL_GAP)) / 2 : 0,
            CELL_Y + CELL_H + 4,
            F_X + F_W / 2,
            F_Y,
            F_Y - 46,
          )}
        />
        {/* other requests → forward pass (they share the batch) */}
        <FlowDots hidden={phase !== 'compute'} d={`M ${O_X + O_W} ${O_Y + 41} L ${F_X} ${O_Y + 41}`} spacing={12} />
        {/* forward pass → KV cache */}
        <FlowDots
          hidden={phase !== 'write'}
          d={`M ${F_X + F_W / 2} ${F_Y + F_H} L ${F_X + F_W / 2} ${K_Y}`}
          spacing={12}
        />
        {/* forward pass → first token */}
        <FlowDots
          hidden={phase !== 'sample'}
          d={`M ${F_X + F_W} ${T_Y + T_H / 2} L ${T_X} ${T_Y + T_H / 2}`}
          spacing={12}
        />

        {/* ── Paged KV cache ──────────────────────────────────────────────── */}
        <rect
          x={K_X}
          y={K_Y}
          width={K_W}
          height={K_H}
          rx={6}
          style={{ fill: 'var(--card)', stroke: EMERALD }}
          strokeWidth={1.5}
        />
        {kvBusy ? <rect x={K_X} y={K_Y} width={K_W} height={K_H} rx={6} fill={hatchFill('emerald')} /> : null}
        <rect
          x={K_X + 6}
          y={K_Y + 6}
          width={K_W - 12}
          height={K_H - 12}
          rx={4}
          fill="none"
          style={{ stroke: EMERALD, opacity: 0.5 }}
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        <text
          x={K_X + K_W / 2}
          y={K_Y + 26}
          textAnchor="middle"
          style={{ fill: 'var(--foreground)' }}
          className="text-[10px] font-semibold uppercase tracking-wider"
        >
          {copy.kv}
        </text>

        {Array.from({ length: N_BLOCKS }, (_, b) => (
          <g key={b}>
            <rect
              x={blkX(b)}
              y={BLK_Y}
              width={BLK_W}
              height={BLK_H}
              rx={3}
              fill="none"
              style={{ stroke: 'var(--muted-foreground)', opacity: 0.55 }}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <text
              x={blkX(b) + BLK_W / 2}
              y={BLK_Y + 14}
              textAnchor="middle"
              style={{ fill: 'var(--muted-foreground)' }}
              className="text-[9px] font-mono font-semibold"
            >
              B{b + 1}
            </text>
            {Array.from({ length: BLOCK }, (_, s) => {
              const filled = b * BLOCK + s < cached;
              return (
                <rect
                  key={s}
                  x={blkX(b) + 5 + s * (SLOT_W + SLOT_GAP)}
                  y={BLK_Y + BLK_H - SLOT_H - 7}
                  width={SLOT_W}
                  height={SLOT_H}
                  rx={1.5}
                  style={{
                    fill: filled ? EMERALD : 'transparent',
                    fillOpacity: filled ? 0.85 : 0,
                    stroke: filled ? EMERALD : 'var(--muted-foreground)',
                    strokeOpacity: filled ? 1 : 0.5,
                    transition: player.reducedMotion ? undefined : 'fill-opacity 300ms ease, stroke 300ms ease',
                  }}
                  strokeWidth={1}
                />
              );
            })}
          </g>
        ))}

        <text
          x={K_X + K_W / 2}
          y={K_Y + K_H - 14}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[9px] uppercase tracking-wider"
        >
          {copy.blockSize}
        </text>
      </svg>

      {/* Caption — the sentence for the beat currently on screen. Fixed height
          so the layout does not jump as the text changes length. */}
      <p className="min-h-[2.5rem] text-[12px] leading-relaxed text-muted-foreground" role="status" aria-live="polite">
        {caption}
      </p>

      <p className="border-t border-border/60 pt-2 text-[11px] leading-relaxed text-muted-foreground">{copy.note}</p>
    </div>
  );
}
