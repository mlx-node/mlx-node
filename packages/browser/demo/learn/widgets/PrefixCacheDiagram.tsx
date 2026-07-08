import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * PrefixCacheDiagram — prefix caching: reuse one prompt's KV cache ACROSS
 * requests, not just within a single generation (which the KV-cache chapter
 * already covers).
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO animation — pure
 * React + Tailwind/SVG, so it server-renders for crawlers (createRoot replaces
 * it on boot) and is robust while the model downloads. The first render is
 * deterministic and identical server vs client; there is no time-dependent
 * state, so it also needs no prefers-reduced-motion handling. The toggle is a
 * pure scenario switch (two canned prompt pairs), not playback.
 *
 * What it teaches (mirrors the book's prefix-cache figures 5.7–5.8):
 *   - Two requests that share a leading PREFIX reuse the KV computed for those
 *     shared tokens. The shared span is computed ONCE; the second request skips
 *     prefill on it, so its TTFT (time-to-first-token) drops.
 *   - The prefix ends at the FIRST differing token. The model is autoregressive,
 *     so one different token changes the KV for EVERYTHING after it — even text
 *     that looks identical downstream. Past the first miss the cache is useless.
 *
 * Per-token verdict (palette matches SpeculativeDecodeDiagram's accept/reject):
 *   MATCH     — emerald: in the shared prefix, KV reused, prefill skipped.
 *   MISS      — red: the first token that differs (the prefix boundary).
 *   DISCARDED — muted: everything after the miss; its KV differs too, recompute.
 *
 * Two scenarios via the toggle:
 *   A "shared prefix"  — Weather/in/{SF|NYC}/? → "Weather","in" reused; "SF"/"NYC"
 *                        is the miss; "?" discarded. Real, useful reuse.
 *   B "novel first token" — {SF|NYC}/weather/today/? → first token differs → ZERO
 *                        reuse even though "weather","today","?" are identical.
 *                        The counter-example: novel tokens early kill the cache.
 *
 * Honesty (caption): the single-user browser demo has nothing to share a prefix
 * WITH — every request re-prefills the whole prompt cold. Production reuses a
 * fixed system prompt across millions of requests, which is why pay-per-token
 * APIs bill "cache-hit" tokens cheaper and feel near-instant on the boilerplate.
 */

type Verdict = 'match' | 'miss' | 'discarded';

type Scenario = {
  id: 'A' | 'B';
  /** request 1 tokens (illustrative strings — stay English, they stand in for raw tokens) */
  rowA: string[];
  /** request 2 tokens */
  rowB: string[];
  /** index of the first differing token (the miss / prefix boundary) */
  missIndex: number;
};

// Two scenarios — illustrative tokens only. Token strings stay English in both
// locales (they are raw model tokens shown in cells, not UI copy).
const SCENARIOS: Scenario[] = [
  {
    id: 'A',
    rowA: ['Weather', 'in', 'SF', '?'],
    rowB: ['Weather', 'in', 'NYC', '?'],
    missIndex: 2, // "Weather","in" match; "SF" vs "NYC" is the first difference
  },
  {
    id: 'B',
    rowA: ['SF', 'weather', 'today', '?'],
    rowB: ['NYC', 'weather', 'today', '?'],
    missIndex: 0, // first token already differs → nothing before it to reuse
  },
];

const N = 4; // tokens per row (both scenarios)

// Tailwind palette hexes, used for SVG fills/strokes (SVG can't take TW classes).
const EMERALD = '#10b981'; // emerald-500 — match / cache-hit
const RED = '#ef4444'; // red-500 — miss (the prefix boundary)

/** Verdict for token i in a row given the first-difference index. */
function verdictAt(i: number, missIndex: number): Verdict {
  if (i < missIndex) return 'match';
  if (i === missIndex) return 'miss';
  return 'discarded';
}

/** How many leading tokens are reused (= missIndex; 0 means no reuse at all). */
function reusedCount(s: Scenario): number {
  return s.missIndex;
}

// ── Per-locale copy. zh keeps the English glossary terms (prefix, prefix cache,
// KV cache, cache hit, prefill, TTFT, token, system prompt, RAG) verbatim, with
// full-width punctuation in prose. ──────────────────────────────────────────
const COPY = {
  en: {
    title: 'Prefix cache — reuse one prompt’s KV across requests',
    scenarioLabel: 'scenario',
    scenarioA: 'shared prefix',
    scenarioB: 'novel first token',
    scenarioAAria:
      'Scenario A — shared prefix: two requests share "Weather in", so that KV is reused; the first difference is "SF" versus "NYC".',
    scenarioBAria:
      'Scenario B — novel first token: the very first token differs, so nothing is reused even though the rest is identical.',
    req1Label: 'request 1',
    req2Label: 'request 2',
    sharedBracket: 'shared prefix — KV reused',
    boundaryBadge: 'first difference',
    legendMatch: 'Match — KV reused (prefill skipped)',
    legendMiss: 'Miss — first differing token',
    legendDiscarded: 'Discarded — recomputed',
    reuseChip: (n: number) =>
      n > 0 ? (
        <>
          reuse: <span className="font-mono text-emerald-600 dark:text-emerald-400">{n} / {N}</span> tokens
        </>
      ) : (
        <>
          reuse: <span className="font-mono text-red-600 dark:text-red-400">0 / {N}</span> tokens
        </>
      ),
    captionShared: (
      <>
        The two requests share the leading <strong>prefix</strong> <span className="font-mono">Weather in</span>. Its{' '}
        <strong>KV cache</strong> is computed <strong>once</strong>, so the second request{' '}
        <span className="text-emerald-600 dark:text-emerald-400">skips prefill</span> on those tokens — a{' '}
        <strong>cache hit</strong> that cuts its <strong>TTFT</strong>. The prefix ends at the{' '}
        <span className="text-red-600 dark:text-red-400">first difference</span> (<span className="font-mono">SF</span> vs{' '}
        <span className="font-mono">NYC</span>); the model is autoregressive, so every token after the miss has different
        KV too and is <span className="text-muted-foreground">recomputed</span>.
      </>
    ),
    captionNovel: (
      <>
        Here the <strong>very first token</strong> differs (<span className="font-mono">SF</span> vs{' '}
        <span className="font-mono">NYC</span>), so there is <strong>nothing to reuse</strong> — even though{' '}
        <span className="font-mono">weather today ?</span> is identical in both. One different token changes the KV for
        everything after it. Lesson: put <strong>novel tokens as late as possible</strong> in the context.
      </>
    ),
    wins: (
      <>
        Big wins when a long <strong>prefix</strong> is shared across requests: long <strong>system prompts</strong>{' '}
        (agents, chatbots, tool calls), <strong>RAG</strong> / retrieved documents, and shared code context. Multi-turn
        chat reuses the whole conversation so far on every turn.
      </>
    ),
    honesty: (
      <>
        Honesty for this demo: as a single user at batch size 1, you have nothing to share a prefix <em>with</em> — every
        request re-prefills the whole prompt cold. A production API reuses one fixed <strong>system prompt</strong> across
        millions of requests, which is why pay-per-token APIs bill <strong>cache-hit</strong> tokens cheaper and feel
        near-instant on the boilerplate, only “thinking” on the new tokens.
      </>
    ),
    illustrativeCaption: 'Illustrative tokens — the concept, not this model’s real tokenization.',
    ariaLabel:
      'Diagram of prefix caching across two requests: shared leading tokens are marked as cache-hit matches (green) whose KV is reused, the first differing token is the miss (red) that ends the prefix, and every token after it is discarded and recomputed (muted).',
  },
  zh: {
    title: 'prefix cache——跨请求复用一个 prompt 的 KV',
    scenarioLabel: '场景',
    scenarioA: '共享 prefix',
    scenarioB: '首 token 不同',
    scenarioAAria:
      '场景 A——共享 prefix：两个请求共享「Weather in」，于是这段 KV 被复用；第一个差异在「SF」与「NYC」处。',
    scenarioBAria: '场景 B——首 token 不同：第一个 token 就不同，所以即使后面完全一致也没有任何可复用。',
    req1Label: '请求 1',
    req2Label: '请求 2',
    sharedBracket: 'shared prefix——KV 复用',
    boundaryBadge: '第一个差异',
    legendMatch: 'Match——KV 复用（跳过 prefill）',
    legendMiss: 'Miss——第一个不同的 token',
    legendDiscarded: 'Discarded——重新计算',
    reuseChip: (n: number) =>
      n > 0 ? (
        <>
          复用：<span className="font-mono text-emerald-600 dark:text-emerald-400">{n} / {N}</span> 个 token
        </>
      ) : (
        <>
          复用：<span className="font-mono text-red-600 dark:text-red-400">0 / {N}</span> 个 token
        </>
      ),
    captionShared: (
      <>
        两个请求共享前缀 <strong>prefix</strong> <span className="font-mono">Weather in</span>。这段{' '}
        <strong>KV cache</strong> 只计算 <strong>一次</strong>，于是第二个请求在这些 token 上{' '}
        <span className="text-emerald-600 dark:text-emerald-400">跳过 prefill</span>——这是一次{' '}
        <strong>cache hit</strong>，降低了它的 <strong>TTFT</strong>。prefix 在{' '}
        <span className="text-red-600 dark:text-red-400">第一个差异</span> 处结束（<span className="font-mono">SF</span> 对{' '}
        <span className="font-mono">NYC</span>）；模型是自回归的，所以 miss 之后的每个 token 的 KV 也不同，需要{' '}
        <span className="text-muted-foreground">重新计算</span>。
      </>
    ),
    captionNovel: (
      <>
        这里 <strong>第一个 token</strong> 就不同（<span className="font-mono">SF</span> 对{' '}
        <span className="font-mono">NYC</span>），所以 <strong>没有任何可复用</strong>——即便{' '}
        <span className="font-mono">weather today ?</span> 在两者中完全一致。一个不同的 token 会改变其后所有 token 的
        KV。结论：把 <strong>新内容尽量放在 context 的靠后位置</strong>。
      </>
    ),
    wins: (
      <>
        当一段较长的 <strong>prefix</strong> 在多个请求间共享时收益巨大：长 <strong>system prompt</strong>（agent、
        chatbot、工具调用）、<strong>RAG</strong> / 检索到的文档，以及共享的代码上下文。多轮对话每一轮都会复用到目前
        为止的整段对话。
      </>
    ),
    honesty: (
      <>
        对本 demo 要诚实：作为 batch size 1 的单一用户，你没有任何对象可以共享 prefix——每个请求都要把整段 prompt 冷启动
        重新 prefill 一遍。生产环境的 API 会在数百万请求间复用同一段固定的 <strong>system prompt</strong>，这正是按 token
        计费的 API 把 <strong>cache-hit</strong> 的 token 算得更便宜、在样板内容上几乎瞬时、只在新 token 上「思考」的原因。
      </>
    ),
    illustrativeCaption: '示意 token——讲概念，并非本模型的真实分词。',
    ariaLabel:
      'prefix 缓存跨两个请求的示意图：共享的前导 token 标为 cache-hit 匹配（绿色），其 KV 被复用；第一个不同的 token 是 miss（红色），它结束了 prefix；其后的每个 token 都被丢弃并重新计算（灰显）。',
  },
} as const;

// ── SVG geometry ─────────────────────────────────────────────────────────────
// Sized for EN (wider than zh): the widest cell token is "Weather" (7 mono
// chars). At 14px mono each cell comfortably holds it; row labels sit to the
// left of the grid, kept inside the viewBox.
const CELL_W = 96;
const CELL_H = 42;
const CELL_GAP = 10;
const GRID_X = 92; // left edge of the first cell; row labels live to its left
const ROW1_Y = 56;
const ROW_GAP = 22;
const ROW2_Y = ROW1_Y + CELL_H + ROW_GAP;
const VB_W = GRID_X + N * CELL_W + (N - 1) * CELL_GAP + 16; // 92 + 384 + 30 + 16 = 522
const VB_H = ROW2_Y + CELL_H + 40;

function cellX(i: number): number {
  return GRID_X + i * (CELL_W + CELL_GAP);
}

/** Visual style for a token cell given its verdict. */
function cellStyle(v: Verdict): {
  stroke: string;
  fill: string;
  fillOpacity: number;
  textColor: string;
  dash?: string;
} {
  if (v === 'match') {
    return { stroke: EMERALD, fill: EMERALD, fillOpacity: 0.14, textColor: EMERALD };
  }
  if (v === 'miss') {
    return { stroke: RED, fill: RED, fillOpacity: 0.14, textColor: RED };
  }
  // discarded — muted + dashed, reads as "thrown away / recomputed"
  return {
    stroke: 'var(--border)',
    fill: 'var(--muted)',
    fillOpacity: 0.5,
    textColor: 'var(--muted-foreground)',
    dash: '5 4',
  };
}

function TokenRow({
  tokens,
  missIndex,
  yTop,
  label,
}: {
  tokens: string[];
  missIndex: number;
  yTop: number;
  label: string;
}) {
  return (
    <g>
      {/* row label, left of the grid (right-aligned so it never crosses GRID_X) */}
      <text
        x={GRID_X - 14}
        y={yTop + CELL_H / 2 + 4}
        textAnchor="end"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[11px]"
      >
        {label}
      </text>
      {tokens.map((tok, i) => {
        const v = verdictAt(i, missIndex);
        const st = cellStyle(v);
        const x = cellX(i);
        return (
          <g key={i}>
            <rect
              x={x}
              y={yTop}
              width={CELL_W}
              height={CELL_H}
              rx={6}
              fill={st.fill}
              fillOpacity={st.fillOpacity}
              stroke={st.stroke}
              strokeWidth={v === 'discarded' ? 1.5 : 2}
              strokeDasharray={st.dash}
            />
            <text
              x={x + CELL_W / 2}
              y={yTop + CELL_H / 2 + 5}
              textAnchor="middle"
              fill={st.textColor}
              className="font-mono text-[14px] font-semibold"
            >
              {tok}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export function PrefixCacheDiagram() {
  const copy = COPY[useLocale()];
  const [scenarioIdx, setScenarioIdx] = React.useState(0);
  const scenario = SCENARIOS[scenarioIdx];
  const isShared = scenario.missIndex > 0;
  const reused = reusedCount(scenario);

  const caption = isShared ? copy.captionShared : copy.captionNovel;

  // The shared-prefix bracket spans cells [0 .. missIndex-1]; only drawn when
  // there is at least one reused token (missIndex > 0).
  const bracketX = cellX(0);
  const bracketW = scenario.missIndex * CELL_W + Math.max(0, scenario.missIndex - 1) * CELL_GAP;
  const missX = cellX(scenario.missIndex);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + scenario toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.scenarioLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            <button
              type="button"
              onClick={() => setScenarioIdx(0)}
              aria-pressed={scenarioIdx === 0}
              aria-label={copy.scenarioAAria}
              className="px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              style={{
                background: scenarioIdx === 0 ? 'var(--primary)' : 'transparent',
                color: scenarioIdx === 0 ? 'var(--background)' : 'var(--muted-foreground)',
              }}
            >
              A · {copy.scenarioA}
            </button>
            <button
              type="button"
              onClick={() => setScenarioIdx(1)}
              aria-pressed={scenarioIdx === 1}
              aria-label={copy.scenarioBAria}
              className="border-l border-border px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              style={{
                background: scenarioIdx === 1 ? 'var(--primary)' : 'transparent',
                color: scenarioIdx === 1 ? 'var(--background)' : 'var(--muted-foreground)',
              }}
            >
              B · {copy.scenarioB}
            </button>
          </div>
        </div>
      </div>

      {/* The diagram */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* shared-prefix bracket over the matched cells (scenario A only) */}
        {isShared ? (
          <g>
            <rect
              x={bracketX - 4}
              y={ROW1_Y - 26}
              width={bracketW + 8}
              height={ROW2_Y + CELL_H - ROW1_Y + 30}
              rx={9}
              fill={EMERALD}
              fillOpacity={0.06}
              stroke={EMERALD}
              strokeWidth={1.5}
              strokeDasharray="2 0"
            />
            <text
              x={bracketX + bracketW / 2}
              y={ROW1_Y - 14}
              textAnchor="middle"
              fill={EMERALD}
              className="text-[11px] font-semibold"
            >
              {copy.sharedBracket}
            </text>
          </g>
        ) : null}

        {/* first-difference (miss) boundary marker — a vertical tick + label */}
        <g>
          <line
            x1={missX - CELL_GAP / 2}
            y1={ROW1_Y - 26}
            x2={missX - CELL_GAP / 2}
            y2={ROW2_Y + CELL_H + 6}
            stroke={RED}
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <text
            x={missX + CELL_W / 2}
            y={ROW2_Y + CELL_H + 24}
            textAnchor="middle"
            fill={RED}
            className="text-[11px] font-semibold"
          >
            ↑ {copy.boundaryBadge}
          </text>
        </g>

        {/* the two request rows */}
        <TokenRow tokens={scenario.rowA} missIndex={scenario.missIndex} yTop={ROW1_Y} label={copy.req1Label} />
        <TokenRow tokens={scenario.rowB} missIndex={scenario.missIndex} yTop={ROW2_Y} label={copy.req2Label} />
      </svg>

      {/* Legend + reuse count */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: EMERALD, opacity: 0.7 }} />
          <span className="text-muted-foreground">{copy.legendMatch}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: RED, opacity: 0.7 }} />
          <span className="text-muted-foreground">{copy.legendMiss}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm border border-dashed"
            style={{ borderColor: 'var(--muted-foreground)', background: 'var(--muted)' }}
          />
          <span className="text-muted-foreground">{copy.legendDiscarded}</span>
        </span>
        <span className="ml-auto inline-flex items-center gap-1.5 rounded border border-border bg-muted px-2 py-0.5 text-muted-foreground">
          {copy.reuseChip(reused)}
        </span>
      </div>

      {/* Illustrative-tokens caption */}
      <p className="text-[10px] italic text-muted-foreground/70">{copy.illustrativeCaption}</p>

      {/* Plain-language caption per scenario */}
      <p className="min-h-[4.5rem] text-[13px] text-foreground/90">{caption}</p>

      {/* Where it pays off */}
      <p className="text-[13px] text-foreground/90">{copy.wins}</p>

      {/* Honesty framing: this demo vs production serving */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.honesty}</p>
    </div>
  );
}
