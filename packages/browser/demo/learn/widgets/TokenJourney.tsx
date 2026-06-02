import { cn } from '@/lib/utils';
import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { Button } from '../../components/ui/button';
import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';

/**
 * Course capstone — "Follow one token's journey".
 *
 * A scrubber that walks ONE concrete example ("The cat sat on the" → " floor")
 * through the whole transformer forward pass, one stage at a time, so a
 * beginner finally sees the end-to-end flow in a single place. Every earlier
 * chapter zooms into one box; this widget puts them back on one rail and keeps
 * a "right now the data is …" spine under every stage.
 *
 * Self-contained by design: no worker, no model, no async, no network. Every
 * number is a precomputed deterministic constant — visuals vary by stage/index,
 * never by `Math.random` or `Date.now` — so it renders instantly even while the
 * real model is still downloading. The numbers and heatmaps are schematic
 * stand-ins for the real 1024-dim vectors and 248,320 logits, not live output.
 */

// ---------------------------------------------------------------------------
// The running example — exact constants shared with the rest of the course.
// ---------------------------------------------------------------------------

const PROMPT = 'The cat sat on the';
const TOKEN_TEXTS: readonly string[] = ['The', ' cat', ' sat', ' on', ' the'];
const TOKEN_IDS: readonly number[] = [791, 9059, 7731, 389, 279];

const NUM_LAYERS = 24;

// Stage-6 distribution (precomputed; probs sum to 1.00). " floor" is the greedy
// continuation this course actually measured for the prompt (see ChapterIndex and
// the LM-head chapter) — deliberately NOT the cliché " mat", so the journey shows
// what the real model does, not what a human would guess.
const DIST_TEXTS: readonly string[] = [' floor', ' mat', ' ground', ' table', ' sofa', ' grass'];
const DIST_PROBS: readonly number[] = [0.41, 0.19, 0.13, 0.12, 0.09, 0.06];
const DIST_IDS: readonly number[] = [6558, 2450, 5015, 1295, 28304, 16763];
const SAMPLED_TOKEN_ID = 6558; // the " floor" row, highlighted by TopKBars

// Stage 3 — illustrative causal-attention weights from the last token (" the")
// back toward [The, cat, sat, on, the-self]. "cat"/"sat" are made prominent.
const ATTENTION_WEIGHTS: readonly number[] = [0.3, 0.34, 0.16, 0.12, 0.08];

const AUTOPLAY_MS = 2200;
const TOTAL_STAGES = 6;

// ---------------------------------------------------------------------------
// Deterministic illustrative embedding heatmap — 5 rows × 12 cells in [-1, 1].
// Varies by (row, col), never by randomness.
// ---------------------------------------------------------------------------

const HEATMAP_ROWS = TOKEN_TEXTS.length; // 5
const HEATMAP_COLS = 12;

function heatmapValue(row: number, col: number): number {
  return Math.sin(row * 1.7 + col * 0.6) * 0.8;
}

/** Map an illustrative value in [-1, 1] to a diverging fill (teal +, amber −). */
function heatmapFill(value: number): string {
  const mag = Math.min(1, Math.abs(value));
  const alpha = (0.12 + mag * 0.68).toFixed(3);
  return value >= 0
    ? `oklch(0.72 0.12 185 / ${alpha})` // teal for positive
    : `oklch(0.78 0.14 60 / ${alpha})`; // amber for negative
}

// ---------------------------------------------------------------------------
// Stage metadata. Copy with **bold** spans is rendered as JSX per stage (so the
// emphasis is real <strong>, not parsed markup); the rail/spine read this table.
// ---------------------------------------------------------------------------

type StageMeta = {
  /** 1-indexed position on the rail. */
  n: number;
  title: string;
  /** The through-line "data right now" string shown in the spine badge. */
  dataNow: string;
  /** Label + chapterId for the "Learn more" cross-link. */
  chapterLabel: string;
  chapterId: string;
};

const STAGES: readonly StageMeta[] = [
  {
    n: 1,
    title: 'Text → tokens',
    dataNow: '[791, 9059, 7731, 389, 279]  ·  5 token ids',
    chapterLabel: 'Tokenization',
    chapterId: 'tokenization',
  },
  {
    n: 2,
    title: 'Tokens → vectors',
    dataNow: '5 tokens × 1024 numbers',
    chapterLabel: 'Embeddings',
    chapterId: 'embeddings',
  },
  {
    n: 3,
    title: 'Tokens share context',
    dataNow: '5 × 1024 numbers  ·  now context-mixed',
    chapterLabel: 'Self-attention',
    chapterId: 'attention',
  },
  {
    n: 4,
    title: 'Refine, ×24 layers',
    dataNow: '5 × 1024 numbers  ·  after 24 layers',
    chapterLabel: 'Full transformer block',
    chapterId: 'full-block',
  },
  {
    n: 5,
    title: 'Top vector → 248,320 scores',
    dataNow: '248,320 scores (logits)',
    chapterLabel: 'LM head',
    chapterId: 'lm-head',
  },
  {
    n: 6,
    title: 'Scores → the next token',
    dataNow: '" floor"  ·  appended, then loop',
    chapterLabel: 'Sampling',
    chapterId: 'sampling',
  },
];

// ---------------------------------------------------------------------------
// Reduced-motion hook — read the media query once (SSR-guarded) into state.
// ---------------------------------------------------------------------------

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(
    () =>
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

// ---------------------------------------------------------------------------
// Stage 1 — token chips: token text above its id, leading space made visible.
// ---------------------------------------------------------------------------

function TokenChips() {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-stretch gap-2">
        {TOKEN_TEXTS.map((raw, i) => {
          const hasLeadingSpace = raw.startsWith(' ');
          const visible = raw.replace(/^ /, '');
          return (
            <div
              key={`${TOKEN_IDS[i]}-${i}`}
              className="flex min-w-[3.5rem] flex-col items-center rounded-md border border-border bg-muted/30 px-2 py-1.5"
            >
              <span className="font-mono text-[13px] text-foreground/90">
                {hasLeadingSpace ? (
                  <span
                    aria-hidden="true"
                    className="mr-0.5 inline-block w-2 rounded-sm border-b border-dashed border-foreground/40 align-middle"
                  />
                ) : null}
                {visible}
              </span>
              <span className="mt-0.5 font-mono text-[10px] tabular-nums text-muted-foreground">{TOKEN_IDS[i]}</span>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] text-muted-foreground">the leading space is part of the token.</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 2 — illustrative 5×12 embedding heatmap.
// ---------------------------------------------------------------------------

function EmbeddingHeatmap() {
  const cellW = 26;
  const cellH = 22;
  const labelW = 40;
  const gap = 3;
  const width = labelW + HEATMAP_COLS * (cellW + gap);
  const height = HEATMAP_ROWS * (cellH + gap);
  return (
    <div className="space-y-1.5">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Illustrative embedding heatmap: 5 token rows by 12 columns of coloured cells, teal for positive values and amber for negative, standing in for each token's 1024-number vector."
        className="block h-auto w-full max-w-[460px]"
      >
        {TOKEN_TEXTS.map((raw, r) => {
          const y = r * (cellH + gap);
          return (
            <g key={`row-${r}`}>
              <text
                x={labelW - 6}
                y={y + cellH / 2 + 3.5}
                fontSize={10}
                textAnchor="end"
                fill="currentColor"
                fillOpacity={0.7}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {cleanupTokenText(raw)}
              </text>
              {Array.from({ length: HEATMAP_COLS }).map((_, c) => {
                const v = heatmapValue(r, c);
                return (
                  <rect
                    key={`cell-${r}-${c}`}
                    x={labelW + c * (cellW + gap)}
                    y={y}
                    width={cellW}
                    height={cellH}
                    rx={3}
                    fill={heatmapFill(v)}
                    stroke="currentColor"
                    strokeOpacity={0.08}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="font-mono text-[10px] text-muted-foreground">5 tokens × 1024 numbers (12 shown)</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 3 — causal attention arcs from the last token back to earlier ones.
// Mirrors the quadratic-path arc idiom from the KV-cache DecodeAnimation.
// ---------------------------------------------------------------------------

function AttentionArcs() {
  const slotW = 78;
  const padX = 12;
  const rowY = 78;
  const slotH = 30;
  const queryY = 14;
  const width = padX * 2 + TOKEN_TEXTS.length * slotW;
  const height = 130;
  const queryIdx = TOKEN_TEXTS.length - 1;

  function slotCx(i: number) {
    return padX + i * slotW + slotW / 2;
  }

  const qCx = slotCx(queryIdx);
  const qCy = queryY + 22;
  const maxW = Math.max(...ATTENTION_WEIGHTS);

  function arcTo(targetIdx: number): string {
    const tx = slotCx(targetIdx);
    const ty = rowY;
    const dx = tx - qCx;
    const midX = qCx + dx / 2;
    const midY = (qCy + ty) / 2 - Math.max(14, Math.abs(dx) * 0.22);
    return `M ${qCx} ${qCy} Q ${midX} ${midY} ${tx} ${ty}`;
  }

  return (
    <div className="space-y-1.5">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Causal attention arcs: the last token 'the' (the query) reaches back to each earlier token, with thicker arcs toward 'cat' and 'sat' showing more attention; no arc points forward."
        className="block h-auto w-full max-w-[460px]"
      >
        {/* Query bubble above the last token. */}
        <rect
          x={qCx - slotW / 2 + 6}
          y={queryY}
          width={slotW - 12}
          height={slotH - 6}
          rx={5}
          fill="oklch(0.65 0.2 25)"
        />
        <text
          x={qCx}
          y={queryY + 16}
          fontSize={11}
          textAnchor="middle"
          fill="white"
          style={{ fontFamily: 'var(--font-mono, monospace)' }}
        >
          query
        </text>

        {/* Backward arcs, thickness/opacity ∝ attention weight. */}
        {ATTENTION_WEIGHTS.map((w, i) => {
          const t = w / maxW;
          return (
            <path
              key={`arc-${i}`}
              d={arcTo(i)}
              fill="none"
              stroke="oklch(0.65 0.2 25)"
              strokeOpacity={0.3 + t * 0.6}
              strokeWidth={1 + t * 3.5}
            />
          );
        })}

        {/* Token row. */}
        {TOKEN_TEXTS.map((raw, i) => {
          const x = padX + i * slotW;
          const isQuery = i === queryIdx;
          return (
            <g key={`tok-${i}`}>
              <rect
                x={x + 4}
                y={rowY}
                width={slotW - 8}
                height={slotH}
                rx={5}
                fill={isQuery ? 'oklch(0.65 0.2 25)' : 'currentColor'}
                fillOpacity={isQuery ? 0.18 : 0.06}
                stroke={isQuery ? 'oklch(0.65 0.2 25)' : 'currentColor'}
                strokeOpacity={isQuery ? 0.8 : 0.18}
                strokeWidth={isQuery ? 1.4 : 1}
              />
              <text
                x={slotCx(i)}
                y={rowY + slotH / 2 + 4}
                fontSize={11}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.85}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {cleanupTokenText(raw)}
              </text>
              <text
                x={slotCx(i)}
                y={rowY + slotH + 14}
                fontSize={9}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.5}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {ATTENTION_WEIGHTS[i]?.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="text-[11px] text-muted-foreground">
        Arc thickness = how much the last token pulls from each earlier one (it can only look backward).
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 4 — 24 stacked layer bars, the last token's vector intensifying down.
// ---------------------------------------------------------------------------

function LayerStack() {
  const width = 320;
  const barH = 5;
  const gap = 2.5;
  const padTop = 4;
  const padX = 44;
  const height = padTop * 2 + NUM_LAYERS * (barH + gap);
  return (
    <div className="flex items-start gap-3">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Twenty-four stacked layer bars from input at the top to output at the bottom; each lower bar is slightly more saturated, showing the running vector growing richer with depth."
        className="block h-auto w-full max-w-[340px]"
      >
        <text x={padX - 6} y={padTop + 8} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          in
        </text>
        <text x={padX - 6} y={height - padTop - 2} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          out
        </text>
        {Array.from({ length: NUM_LAYERS }).map((_, i) => {
          const y = padTop + i * (barH + gap);
          // Intensify with depth: deeper layers carry a richer running vector.
          const t = i / (NUM_LAYERS - 1);
          const alpha = (0.22 + t * 0.66).toFixed(3);
          return (
            <g key={`layer-${i}`}>
              <rect
                x={padX}
                y={y}
                width={width - padX - 8}
                height={barH}
                rx={2}
                fill={`oklch(0.62 0.16 265 / ${alpha})`}
              />
              {(i + 1) % 4 === 0 ? (
                <text
                  x={width - 4}
                  y={y + barH}
                  fontSize={7.5}
                  textAnchor="end"
                  fill="currentColor"
                  fillOpacity={0.4}
                  style={{ fontFamily: 'var(--font-mono, monospace)' }}
                >
                  {i + 1}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="flex flex-col justify-center self-stretch">
        <div className="rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-center">
          <div className="font-mono text-lg font-semibold text-primary">× 24</div>
          <div className="text-[10px] text-muted-foreground">attention + MLP layers</div>
        </div>
        <p className="mt-2 max-w-[10rem] text-[11px] text-muted-foreground">
          Each layer reads the running total and adds a small refinement.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 5 — last token's vector → arrow → a wide thin logits strip.
// ---------------------------------------------------------------------------

// Deterministic illustrative logit bar heights across the long axis.
const LOGIT_BAR_COUNT = 40;
function logitBarHeight(i: number): number {
  // Vary by index, never randomly. A couple of taller spikes among low scores.
  return 0.18 + 0.5 * Math.abs(Math.sin(i * 0.9 + 0.4)) * (0.4 + 0.6 * Math.abs(Math.cos(i * 0.27)));
}

function LogitsStrip() {
  const width = 480;
  const height = 96;
  const vecCells = 6;
  const vecCellW = 14;
  const vecCellH = 14;
  const vecX = 6;
  const vecY = height / 2 - vecCellH / 2;
  const arrowX0 = vecX + vecCells * (vecCellW + 2) + 6;
  const arrowX1 = arrowX0 + 30;
  const stripX = arrowX1 + 8;
  const stripW = width - stripX - 8;
  const stripTop = 16;
  const stripH = 56;
  const barGap = stripW / LOGIT_BAR_COUNT;
  return (
    <div className="space-y-1.5">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="The last token's small vector on the left, an arrow, then a wide thin strip of 248,320 logits drawn as many short bars of varying height — one raw score per vocabulary token."
        className="block h-auto w-full"
      >
        {/* Last token's vector — a small row of cells. */}
        {Array.from({ length: vecCells }).map((_, i) => (
          <rect
            key={`vec-${i}`}
            x={vecX + i * (vecCellW + 2)}
            y={vecY}
            width={vecCellW}
            height={vecCellH}
            rx={2}
            fill={heatmapFill(heatmapValue(HEATMAP_ROWS - 1, i))}
            stroke="currentColor"
            strokeOpacity={0.1}
          />
        ))}
        <text x={vecX} y={vecY - 6} fontSize={9} fill="currentColor" fillOpacity={0.55}>
          last vector
        </text>

        {/* Arrow. */}
        <line
          x1={arrowX0}
          y1={height / 2}
          x2={arrowX1 - 4}
          y2={height / 2}
          stroke="currentColor"
          strokeOpacity={0.5}
          strokeWidth={1.4}
        />
        <path
          d={`M ${arrowX1 - 4} ${height / 2 - 4} L ${arrowX1 + 3} ${height / 2} L ${arrowX1 - 4} ${height / 2 + 4} Z`}
          fill="currentColor"
          fillOpacity={0.5}
        />

        {/* Wide logits strip baseline + bars. */}
        <line
          x1={stripX}
          y1={stripTop + stripH}
          x2={stripX + stripW}
          y2={stripTop + stripH}
          stroke="currentColor"
          strokeOpacity={0.2}
        />
        {Array.from({ length: LOGIT_BAR_COUNT }).map((_, i) => {
          const h = logitBarHeight(i) * stripH;
          const isWinner = i === 4; // one prominent spike standing in for the winning token
          const bh = isWinner ? Math.max(h, stripH * 0.92) : h;
          return (
            <rect
              key={`logit-${i}`}
              x={stripX + i * barGap + 0.6}
              y={stripTop + stripH - bh}
              width={Math.max(1.5, barGap - 1.2)}
              height={bh}
              rx={0.8}
              fill={isWinner ? 'oklch(0.68 0.17 150)' : 'currentColor'}
              fillOpacity={isWinner ? 1 : 0.4}
            />
          );
        })}
        <text x={stripX} y={stripTop - 5} fontSize={9} fill="currentColor" fillOpacity={0.55}>
          248,320 logits
        </text>
      </svg>
      <p className="text-[11px] text-muted-foreground">
        One raw score per vocabulary token — only a handful of the 248,320 bars are drawn here.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-stage explanation copy (with real <strong> emphasis) and visual.
// ---------------------------------------------------------------------------

function StageBody({ stage }: { stage: number }) {
  switch (stage) {
    case 1:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            The model can&apos;t read raw text. The tokenizer first chops your text into known chunks called{' '}
            <strong>tokens</strong> — here, 5 of them — and looks up each one&apos;s <strong>id</strong> (a plain
            integer). From here on, the model only ever sees these numbers.
          </p>
          <TokenChips />
        </>
      );
    case 2:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            Each id is looked up in a big table and becomes a <strong>vector</strong> — a list of 1024 numbers placing
            that token in &quot;meaning space&quot;. The same id always maps to the same row; <em>where</em> a token
            sits in the sentence is folded in later, inside <strong>attention</strong> — not here.
          </p>
          <EmbeddingHeatmap />
        </>
      );
    case 3:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            <strong>Attention</strong> lets each token look back at the earlier ones and pull in what it needs. To guess
            what follows &quot;the&quot;, the last token gathers meaning from &quot;cat&quot; and &quot;sat&quot;.
            Crucially, a token can only look <strong>backward</strong>, never forward.
          </p>
          <AttentionArcs />
        </>
      );
    case 4:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            Attention plus a small per-token network (the <strong>MLP</strong>) make up one <strong>layer</strong>.
            Qwen3.5 stacks <strong>24</strong> of them. Each layer reads the running total, adds a small refinement, and
            passes it up — so the vectors grow richer with depth.
          </p>
          <LayerStack />
        </>
      );
    case 5:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            Only the <strong>last</strong> token&apos;s vector decides the next word. The <strong>LM head</strong>{' '}
            scores that vector against the whole vocabulary, producing one raw score — a <strong>logit</strong> — for
            every one of the 248,320 tokens the model knows.
          </p>
          <LogitsStrip />
        </>
      );
    case 6:
    default:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">
            <strong>Softmax</strong> turns those raw scores into probabilities that sum to 1, and the model picks one.
            This small model&apos;s top pick here is <strong>&quot; floor&quot;</strong> — a real model is often less
            predictable than the obvious &quot; mat&quot;. That token is appended to the text and the whole journey runs
            again — that&apos;s how a sentence is written, one token at a time.
          </p>
          <TopKBars
            ids={[...DIST_IDS]}
            probs={[...DIST_PROBS]}
            texts={[...DIST_TEXTS]}
            sampledTokenId={SAMPLED_TOKEN_ID}
            runKey={stage}
          />
          <p className="text-[11px] text-muted-foreground">
            → append &apos; floor&apos;, then run the whole journey again (the generation loop).
          </p>
        </>
      );
  }
}

// ---------------------------------------------------------------------------
// Main widget.
// ---------------------------------------------------------------------------

export function TokenJourney() {
  const [stage, setStage] = React.useState(1);
  const [playing, setPlaying] = React.useState(false);
  const reducedMotion = usePrefersReducedMotion();

  const atStart = stage <= 1;
  const atEnd = stage >= TOTAL_STAGES;
  const current = STAGES[stage - 1]!;

  // Autoplay: advance one stage every AUTOPLAY_MS, stop at the last stage.
  // Disabled entirely under reduced motion (Play is hidden below).
  React.useEffect(() => {
    if (!playing || reducedMotion) return undefined;
    const id = window.setInterval(() => {
      setStage((s) => {
        if (s + 1 >= TOTAL_STAGES) {
          setPlaying(false);
          return TOTAL_STAGES;
        }
        return s + 1;
      });
    }, AUTOPLAY_MS);
    return () => window.clearInterval(id);
  }, [playing, reducedMotion]);

  function go(next: number) {
    setPlaying(false);
    setStage(Math.max(1, Math.min(TOTAL_STAGES, next)));
  }

  function togglePlay() {
    if (atEnd) {
      // "Replay" — jump back to the start and play forward again.
      setStage(1);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }

  const playLabel = playing ? 'Pause' : atEnd ? 'Replay' : 'Play';

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header + the running example. */}
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Follow one token&apos;s journey</div>
        <div className="font-mono text-[11px] text-muted-foreground">
          &quot;{PROMPT}&quot; <span className="text-foreground/70">→ ?</span>
        </div>
      </div>

      {/* Stage rail: 6 numbered, clickable chips. */}
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Forward-pass stages">
        {STAGES.map((s) => {
          const isCurrent = s.n === stage;
          return (
            <button
              key={s.n}
              type="button"
              onClick={() => go(s.n)}
              aria-current={isCurrent ? 'step' : undefined}
              aria-label={`Go to stage ${s.n}: ${s.title}`}
              className={cn(
                'flex items-center gap-1.5 rounded-md border px-2 py-1 text-left text-[11px] transition-colors',
                reducedMotion && 'transition-none',
                isCurrent
                  ? 'border-primary/50 bg-primary/15 text-primary'
                  : 'border-border/60 bg-muted/20 text-muted-foreground hover:bg-foreground/5',
              )}
            >
              <span
                className={cn(
                  'inline-flex h-4 w-4 items-center justify-center rounded-full font-mono text-[10px] tabular-nums',
                  isCurrent ? 'bg-primary text-primary-foreground' : 'bg-foreground/10 text-foreground/70',
                )}
              >
                {s.n}
              </span>
              <span className="hidden font-medium sm:inline">{s.title}</span>
            </button>
          );
        })}
      </div>

      {/* Through-line spine: always-visible "right now the data is" badge. */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">Right now the data is:</span>
        <span className="font-mono text-[12px] font-medium text-foreground/90">{current.dataNow}</span>
      </div>

      {/* Stage panel. A single visually-hidden status region announces the stage
          number, title, AND the current data shape on every change — the
          data-shape through-line is the whole point, so screen-reader users hear
          it, not just the title. The verbose SVG aria-labels sit outside it, so
          they are not re-read on every scrub or autoplay tick. */}
      <div className="space-y-3 rounded-md border border-border/60 bg-muted/10 p-3">
        <div className="sr-only" role="status" aria-live="polite">
          {`Stage ${stage} of ${TOTAL_STAGES}: ${current.title}. Data now: ${current.dataNow}.`}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[11px] text-muted-foreground">
            Stage {stage} / {TOTAL_STAGES}
          </span>
          <span className="text-sm font-semibold text-foreground">{current.title}</span>
        </div>
        <StageBody stage={stage} />
        <div className="text-[12px]">
          <Link
            to="/chapters/$chapterId"
            params={{ chapterId: current.chapterId }}
            search={(prev) => prev}
            className="text-primary underline-offset-4 hover:underline"
          >
            Learn more: {current.chapterLabel} →
          </Link>
        </div>
      </div>

      {/* Controls. */}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => go(stage - 1)} disabled={atStart}>
          ‹ Prev
        </Button>
        <Button size="sm" variant="outline" onClick={() => go(stage + 1)} disabled={atEnd}>
          Next ›
        </Button>
        {reducedMotion ? null : (
          <Button size="sm" onClick={togglePlay} aria-pressed={playing}>
            {playLabel}
          </Button>
        )}
        <span className="ml-auto text-[11px] text-muted-foreground">
          {reducedMotion
            ? 'Use Prev / Next to step through the forward pass.'
            : 'Scrub the rail, or press Play to watch the forward pass run.'}
        </span>
      </div>

      {/* Honesty footer. */}
      <p className="text-[10px] text-muted-foreground">
        Illustrative — a schematic of the real forward pass; the numbers and heatmaps are stand-ins for the real
        1024-dim vectors and 248,320 logits, not live model output.
      </p>
    </div>
  );
}
