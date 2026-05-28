import * as React from 'react';

import type { AttentionLayer, AttentionRun } from '../../../src/inspector-types';
import { TokenStrip } from './TokenStrip';
import { renderTokenDisplay } from './TopKBars';

export type AttentionHeatmapProps = {
  run: AttentionRun;
  /** Pixel size of each score cell. Capped to fit on screen for long prompts. */
  preferredCellSize?: number;
  /**
   * Bumped by the parent on every successful Run to trigger a top-to-bottom
   * causal-reveal animation. The component animates an internal progress
   * counter from 0 to seqLen rows; the draw step honors that via the
   * `revealRows` option in {@link drawAttentionHeatmap}. Default `0` means
   * "no animation, fully revealed" — used when the heatmap is embedded
   * outside a Run-driven flow (e.g. chapter 4's small per-head panels).
   */
  runKey?: number;
};

/**
 * Canvas-backed token×token attention heatmap.
 *
 * Why canvas: with sequences of even a few hundred tokens the seq_len² grid
 * gets dense enough that a per-cell DOM element would be slow. The current
 * draw implementation uses per-cell fillRect; this is acceptable for the
 * sequences this demo renders (≤200 tokens). If we ever need to draw larger
 * sequences, switch to a single ImageData blit.
 *
 * The component:
 *  - lets the user pick a layer and a head via shadcn <Select>
 *  - draws softmaxed scores with a single-hue (blue → cyan) ramp
 *  - on hover, shows a tooltip with the query token, key token, and value
 *  - keeps token strips on top (key) and to the left (query) aligned cell-to-cell
 */
export function AttentionHeatmap({ run, preferredCellSize = 32, runKey = 0 }: AttentionHeatmapProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  // Initial default: prefer the first layer whose kind === 'full' so the
  // heatmap doesn't open on a linear layer ("This layer does not expose
  // softmax scores."). Computed lazily so it only runs on mount; subsequent
  // run swaps are handled by the useEffect below.
  const [selectedLayerIdx, setSelectedLayerIdx] = React.useState(() => {
    const idx = run.attention.findIndex((layer) => layer.kind === 'full');
    return idx >= 0 ? idx : 0;
  });
  const [selectedHead, setSelectedHead] = React.useState(0);
  // Out-of-bounds guard only: if the new run has fewer layers than the
  // current selection (e.g. someone swapped models), snap back into range.
  // We deliberately do NOT force the selection onto a full-attention layer
  // — with the slider UI the user can scrub through linear layers, and
  // when they land on one the heatmap canvas already renders a "This layer
  // does not expose softmax scores." notice. Auto-reverting would jump the
  // slider thumb away mid-drag and break scrubbing entirely.
  React.useEffect(() => {
    if (selectedLayerIdx >= run.attention.length) {
      const idx = run.attention.findIndex((layer) => layer.kind === 'full');
      setSelectedLayerIdx(idx >= 0 ? idx : 0);
    }
  }, [run, selectedLayerIdx]);
  const [hover, setHover] = React.useState<{
    queryIndex: number;
    keyIndex: number;
    x: number;
    y: number;
    score: number;
  } | null>(null);
  // Bumped whenever the `run` prop reference changes so we can restart the
  // bottom-row "prediction step" flash animation without remounting the
  // canvas. Same class-toggle pattern as the chapter-3 heatmap wrapper.
  const lastRowFlashRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const el = lastRowFlashRef.current;
    if (!el) return;
    el.classList.remove('last-row-flash');
    void el.offsetWidth;
    el.classList.add('last-row-flash');
  }, [run]);
  // Bumped on window resize (and DPR-relevant changes like moving between
  // monitors or zooming). Used to retrigger the draw effect so the canvas
  // re-syncs its backing-store size with the current devicePixelRatio.
  const [resizeTick, setResizeTick] = React.useState(0);
  React.useEffect(() => {
    const onResize = () => setResizeTick((n) => n + 1);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
    };
  }, []);

  // Clamp head if the layer changes and the new layer has fewer heads.
  React.useEffect(() => {
    const layer = run.attention[selectedLayerIdx];
    if (!layer) return;
    if (selectedHead >= layer.numHeads) {
      setSelectedHead(0);
    }
  }, [run, selectedLayerIdx, selectedHead]);

  const seqLen = run.tokens.length;
  // Causal-reveal animation: each Run (parent bumps `runKey`) animates this
  // float from 0 → seqLen over `revealDurationMs`, fading in rows top-to-
  // bottom. Default state is `seqLen` (no reveal in progress, fully drawn)
  // so the canvas paints normally whenever runKey === 0 or layer/head
  // change during static viewing.
  const [revealRows, setRevealRows] = React.useState<number>(seqLen);
  React.useEffect(() => {
    if (runKey === 0 || seqLen === 0) {
      setRevealRows(seqLen);
      return;
    }
    const reducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) {
      setRevealRows(seqLen);
      return;
    }
    // ~280 ms per row, capped so a long prompt doesn't make the animation
    // drag for many seconds. 1800 ms is enough for ~6 rows at full speed
    // and visibly faster (but still legible) for longer prompts.
    const perRow = 280;
    const revealDurationMs = Math.min(2200, perRow * seqLen);
    setRevealRows(0);
    let rafId: number | null = null;
    let startTime: number | null = null;
    const step = (t: number) => {
      if (startTime === null) startTime = t;
      const elapsed = t - startTime;
      const progress = Math.min(1, elapsed / revealDurationMs);
      // Ease-out cubic — fast initial rows, gentle landing on the last row.
      const eased = 1 - Math.pow(1 - progress, 3);
      setRevealRows(eased * seqLen);
      if (progress < 1) {
        rafId = requestAnimationFrame(step);
      }
    };
    rafId = requestAnimationFrame(step);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [runKey, seqLen]);

  // Heatmap area should fit comfortably in the right-side panel. Cap at the
  // preferred size, but shrink for very long sequences.
  const maxBoardPx = 360;
  const cellSize = Math.max(8, Math.min(preferredCellSize, Math.floor(maxBoardPx / Math.max(1, seqLen))));
  const board = cellSize * seqLen;

  // Render the heatmap onto the canvas whenever the layer/head/run changes.
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = board * dpr;
    canvas.height = board * dpr;
    canvas.style.width = `${board}px`;
    canvas.style.height = `${board}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background ("masked / zero") color.
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    ctx.fillRect(0, 0, board, board);

    const layer = run.attention[selectedLayerIdx];
    if (!layer || layer.scores.length === 0) {
      // Linear layers ship empty scores; show a notice.
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '12px var(--font-mono, monospace)';
      ctx.fillText('This layer does not expose softmax scores.', 12, 24);
      return;
    }
    const head = Math.min(selectedHead, layer.numHeads - 1);
    drawAttentionHeatmap(ctx, layer, head, cellSize, { revealRows });

    // Grid (subtle).
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= seqLen; i++) {
      const p = i * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(board, p);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, board);
      ctx.stroke();
    }
  }, [run, selectedLayerIdx, selectedHead, seqLen, cellSize, board, resizeTick, revealRows]);

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const j = Math.floor(x / cellSize);
    const i = Math.floor(y / cellSize);
    if (i < 0 || i >= seqLen || j < 0 || j >= seqLen) {
      setHover(null);
      return;
    }
    const layer = run.attention[selectedLayerIdx];
    if (!layer || layer.scores.length === 0) {
      setHover(null);
      return;
    }
    const head = Math.min(selectedHead, layer.numHeads - 1);
    const headStride = seqLen * seqLen;
    const score = layer.scores[head * headStride + i * seqLen + j] ?? 0;
    setHover({ queryIndex: i, keyIndex: j, x, y, score });
  }

  function onMouseLeave() {
    setHover(null);
  }

  const numLayers = run.attention.length;
  const fullAttnIndices = React.useMemo(
    () => run.attention.map((l, i) => (l.kind === 'full' ? i : -1)).filter((i) => i >= 0),
    [run.attention],
  );
  const currentLayer = run.attention[selectedLayerIdx];
  const headCount = currentLayer?.numHeads ?? 0;
  const layerKindLabel = currentLayer ? (currentLayer.kind === 'full' ? 'full attn' : 'linear · GDN') : '';
  const currentHead = Math.min(selectedHead, Math.max(0, headCount - 1));
  const ariaLabel = currentLayer
    ? `Attention heatmap ${seqLen}×${seqLen}, layer ${currentLayer.layerIndex} head ${currentHead}`
    : `Attention heatmap ${seqLen}×${seqLen}`;
  const liveAnnouncement = hover
    ? `Layer ${currentLayer?.layerIndex ?? selectedLayerIdx} head ${currentHead}, ` +
      `query token ${hover.queryIndex} (${run.tokens[hover.queryIndex]?.text ?? ''}) ` +
      `attends to key token ${hover.keyIndex} (${run.tokens[hover.keyIndex]?.text ?? ''}) ` +
      `with score ${hover.score.toFixed(4)}.`
    : '';

  return (
    <div className="space-y-3">
      {/* Layer + head sliders.
       *
       * Why sliders instead of <Select>: with 24 layers the dropdown menu
       * fills the whole viewport and forces a click-scroll-click cycle per
       * layer. A native range input lets you scrub through the stack with
       * the mouse or arrow keys and watch the attention pattern morph in
       * real time — much closer to the lesson's "see how attention changes
       * across the stack" goal. Heads (8 in Qwen3.5) are simpler but kept
       * as a slider for consistency.
       *
       * The layer slider has *tick marks* at the 6 full-attention layer
       * indices ([3, 7, 11, 15, 19, 23] for Qwen3.5's `full_attention_
       * interval = 4` schedule). They're rendered as cyan vertical bars
       * absolutely positioned above the slider track so you can see at a
       * glance where the full-attn (vs linear / GDN) layers live without
       * having to scrub through every position.
       */}
      <div className="space-y-3">
        <SliderRow
          id="attn-heatmap-layer"
          label="Layer"
          value={selectedLayerIdx}
          min={0}
          max={Math.max(0, numLayers - 1)}
          step={1}
          onChange={setSelectedLayerIdx}
          valueLabel={
            currentLayer ? (
              <span className="font-mono text-foreground/80">
                Layer {currentLayer.layerIndex}
                <span className={currentLayer.kind === 'full' ? 'ml-2 text-cyan-400' : 'ml-2 text-amber-400/80'}>
                  · {layerKindLabel}
                </span>
              </span>
            ) : (
              <span className="font-mono text-foreground/80">Layer —</span>
            )
          }
          tickIndices={fullAttnIndices}
        />
        {headCount > 0 ? (
          <SliderRow
            id="attn-heatmap-head"
            label="Head"
            value={currentHead}
            min={0}
            max={Math.max(0, headCount - 1)}
            step={1}
            onChange={setSelectedHead}
            valueLabel={<span className="font-mono text-foreground/80">Head {currentHead}</span>}
          />
        ) : null}
      </div>

      {/* Heatmap with aligned token strips */}
      <div className="inline-block">
        {/* Top strip: key tokens, indented by the left strip's width. */}
        <div className="flex" style={{ paddingLeft: 80 }}>
          <TokenStrip
            tokens={run.tokens}
            cellSize={cellSize}
            orientation="top"
            highlightIndex={hover?.keyIndex ?? null}
          />
        </div>
        <div className="flex">
          {/* Left strip: query tokens */}
          <TokenStrip
            tokens={run.tokens}
            cellSize={cellSize}
            orientation="left"
            highlightIndex={hover?.queryIndex ?? null}
          />
          {/* Canvas itself */}
          <div className="relative">
            <canvas
              ref={canvasRef}
              onMouseMove={onMouseMove}
              onMouseLeave={onMouseLeave}
              className="block rounded-sm border border-border"
              style={{ width: board, height: board }}
              role="img"
              aria-label={ariaLabel}
            />
            {/*
              Screen-reader-only live region. The visual tooltip is purely
              decorative for sighted users — assistive tech reads this string
              when the user moves their cursor across cells.
            */}
            <div
              aria-live="polite"
              aria-atomic="true"
              style={{
                position: 'absolute',
                width: 1,
                height: 1,
                overflow: 'hidden',
                clip: 'rect(0,0,0,0)',
                whiteSpace: 'nowrap',
              }}
            >
              {liveAnnouncement}
            </div>
            {/*
              Persistent dashed outline on the bottom row of the heatmap —
              that row is what produced the next token. The .last-row-flash
              class is re-added on every successful Run (via the ref + reflow
              trick) so the outline briefly glows accent-color, then settles
              back to the steady dashed border.
            */}
            <div
              ref={lastRowFlashRef}
              className="last-row-highlight pointer-events-none"
              aria-hidden="true"
              style={{
                left: 0,
                top: board - cellSize,
                width: board,
                height: cellSize,
              }}
            />
          </div>
        </div>
      </div>

      {/* Cell inspector — lives outside the heatmap so it never occludes
          cells on small boards. Updates live on hover; falls back to a
          short instruction when nothing is hovered. */}
      <CellInspector
        hover={hover}
        queryToken={hover ? run.tokens[hover.queryIndex] : undefined}
        keyToken={hover ? run.tokens[hover.keyIndex] : undefined}
      />

      {/* Color legend */}
      <Legend />

      {/* How-to-read panel — teaches a beginner what the matrix and its
          bottom row actually represent. Always visible. */}
      <div className="space-y-1.5 rounded-md border border-dashed border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <div>
          <span className="font-mono text-foreground">↓</span> row = token doing the looking ·{' '}
          <span className="font-mono text-foreground">→</span> column = token being looked at · brighter = stronger
          attention.
        </div>
        <div>
          The outlined <strong className="text-foreground">bottom row</strong> is the prediction step — those bright
          cells are the tokens the model focused on when choosing the next word below.
        </div>
      </div>

      {/* Next-token card — the concrete artifact of one Run. This is the
          single thing a learner can point at and say "the model just
          predicted that". After a Run, fades in just after the causal row
          reveal in the canvas finishes — delay is reveal duration + small
          buffer. Key bumps via runKey so each Run replays the animation. */}
      <div
        key={`next-token-${runKey}`}
        className={
          runKey > 0
            ? 'attn-next-token-pop rounded-md border border-primary/40 bg-primary/5 px-4 py-3'
            : 'rounded-md border border-primary/40 bg-primary/5 px-4 py-3'
        }
        style={runKey > 0 ? { animationDelay: `${Math.min(2200, 280 * seqLen) + 100}ms` } : undefined}
      >
        <div className="text-[0.7rem] uppercase tracking-wider text-muted-foreground">Next token</div>
        <div className="mt-1.5 flex flex-wrap items-baseline gap-2">
          <span className="rounded-md border border-primary/40 bg-background/60 px-2.5 py-1 font-mono text-base text-foreground">
            {renderTokenDisplay(run.generatedToken.text)}
          </span>
          <span className="font-mono text-[0.7rem] text-muted-foreground">id {run.generatedToken.id}</span>
        </div>
        <div className="mt-2 text-xs text-muted-foreground">
          Greedy argmax of the model's vocabulary at the last position. Edit the prompt above and Run again to see this
          change.
        </div>
      </div>

      {/* Model meta footer */}
      <div className="text-[0.7rem] text-muted-foreground">
        <span className="font-mono">{run.modelMeta.name}</span> · {seqLen} tokens
      </div>
    </div>
  );
}

/**
 * Persistent cell-inspector bar that lives below the heatmap rather than
 * floating over its cells. The earlier in-board tooltip occluded data on
 * small boards (a 7-token prompt yields a 224×224 px board, and the tooltip
 * was 200×80). Beginners couldn't read the cell they were inspecting because
 * the tooltip was sitting on top of it.
 *
 * The cross-hair token-strip highlights (row label + column label) already
 * tell the user *which* cell they're hovering visually — this bar adds the
 * numeric score and the literal token strings without competing for canvas
 * real estate.
 */
function CellInspector({
  hover,
  queryToken,
  keyToken,
}: {
  hover: {
    queryIndex: number;
    keyIndex: number;
    score: number;
  } | null;
  queryToken: { text: string } | undefined;
  keyToken: { text: string } | undefined;
}) {
  if (!hover || !queryToken || !keyToken) {
    return (
      <div
        className="rounded-md border border-dashed border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground"
        aria-live="polite"
      >
        Hover any cell to see its query token, key token, and attention score.
      </div>
    );
  }
  return (
    <div className="rounded-md border border-border bg-card/40 px-3 py-2 font-mono text-xs" aria-live="polite">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-muted-foreground">row {hover.queryIndex}</span>
        <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5 text-foreground">
          {JSON.stringify(queryToken.text)}
        </span>
        <span className="text-muted-foreground">→</span>
        <span className="text-muted-foreground">column {hover.keyIndex}</span>
        <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5 text-foreground">
          {JSON.stringify(keyToken.text)}
        </span>
        <span className="text-muted-foreground">=</span>
        <span className="text-foreground">{hover.score.toFixed(4)}</span>
      </div>
    </div>
  );
}

function Legend() {
  // Eight-stop legend matching colorForScore.
  const stops = Array.from({ length: 8 }, (_, i) => (i + 1) / 8);
  return (
    <div className="flex items-center gap-2 text-[0.7rem] text-muted-foreground">
      <span>0</span>
      <div className="flex h-3 overflow-hidden rounded-sm border border-border">
        {stops.map((s) => (
          <div key={s} style={{ background: colorForScore(s), width: 16, height: '100%' }} />
        ))}
      </div>
      <span>1</span>
      <span className="ml-2 italic">attention weight</span>
    </div>
  );
}

/**
 * Single-hue blue ramp for score in [0, 1]. Higher score = brighter, more
 * saturated. Black-ish for ~0 so the causal upper triangle visually drops out.
 */
function colorForScore(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  // Lightness 14 → 70, saturation 70%.
  const l = 14 + t * 56;
  const s = 70;
  // 210 hue is a deep blue → 195 leans cyan as it brightens.
  const h = 210 - t * 15;
  return `hsl(${h.toFixed(0)} ${s}% ${l.toFixed(0)}%)`;
}

/**
 * Pure draw helper for the per-cell attention grid. Iterates the seqLen×seqLen
 * grid for `headIdx` in `layer.scores`, mapping each score → CSS color via
 * `options.colorFn` (defaults to the canonical {@link colorForScore} ramp used
 * by {@link AttentionHeatmap}), and writes one `fillRect` per non-zero cell.
 *
 * The caller owns:
 *  - canvas backing-store sizing (`canvas.width` / `.height`)
 *  - DPR scaling via `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)`
 *  - background fill (so masked / zero cells show the panel background)
 *  - any post-draw decoration (grid lines, axes)
 *
 * This is a pure CanvasRenderingContext2D writer — no React, no state, no
 * DPR math. Shared between {@link AttentionHeatmap} and the per-head
 * `SmallHeatmap` panels in chapter 4 so the iteration + mapping stay in sync.
 */
export function drawAttentionHeatmap(
  ctx: CanvasRenderingContext2D,
  layer: AttentionLayer,
  headIdx: number,
  pixelSize: number,
  options?: {
    colorFn?: (v: number) => string;
    /**
     * Number of rows to reveal, as a float in [0, seqLen]. Used by the
     * chapter-3 Run animation to do a top-to-bottom causal sweep. Rows below
     * `floor(revealRows)` are not drawn; the row at `floor(revealRows)` is
     * drawn at fractional alpha = `revealRows - floor(revealRows)`. Default
     * is `seqLen` (no animation — fully reveal everything).
     */
    revealRows?: number;
  },
): void {
  const colorFn = options?.colorFn ?? colorForScore;
  const seqLen = Math.round(Math.sqrt(layer.scores.length / Math.max(1, layer.numHeads)));
  if (seqLen === 0) return;
  const reveal = options?.revealRows === undefined ? seqLen : Math.max(0, Math.min(seqLen, options.revealRows));
  const fullRows = Math.floor(reveal);
  const partialAlpha = reveal - fullRows;
  const safeHead = Math.min(headIdx, layer.numHeads - 1);
  const headStride = seqLen * seqLen;
  const headOffset = safeHead * headStride;
  for (let i = 0; i < seqLen; i++) {
    if (i > fullRows) break;
    const alpha = i < fullRows ? 1 : partialAlpha;
    if (alpha <= 0) continue;
    ctx.globalAlpha = alpha;
    for (let j = 0; j < seqLen; j++) {
      const v = layer.scores[headOffset + i * seqLen + j] ?? 0;
      if (v <= 0) continue;
      ctx.fillStyle = colorFn(v);
      ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
    }
  }
  ctx.globalAlpha = 1;
}

/**
 * Compact slider row with an inline label, a native range input, and an
 * optional set of tick marks rendered as absolutely positioned cyan bars
 * above the slider track. We use a native `<input type="range">` to get
 * free keyboard arrow-key navigation, accessible value announcements, and
 * touch support — shadcn ships a `<Slider>` but the project doesn't depend
 * on it, and rolling our own avoids pulling another component for this
 * single use site. The tick marks are pure CSS overlay; they don't
 * intercept clicks so the slider still drags normally over them.
 */
function SliderRow({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  valueLabel,
  tickIndices,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  valueLabel: React.ReactNode;
  tickIndices?: readonly number[];
}) {
  const range = Math.max(1, max - min);
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <label htmlFor={id} className="uppercase tracking-wider text-muted-foreground">
          {label}
        </label>
        {valueLabel}
      </div>
      <div className="relative h-6">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.currentTarget.value))}
          className="relative z-10 block h-6 w-full cursor-pointer appearance-none bg-transparent accent-primary"
        />
        {tickIndices && tickIndices.length > 0 ? (
          <div className="pointer-events-none absolute inset-0 flex items-center">
            {tickIndices.map((i) => (
              <span
                key={i}
                className="absolute h-2 w-px -translate-x-1/2 rounded-sm bg-cyan-400/70"
                style={{ left: `${((i - min) / range) * 100}%` }}
                aria-hidden="true"
              />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
