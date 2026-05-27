import * as React from "react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";

import type { AttentionLayer, AttentionRun } from "../../../src/inspector-types";
import { TokenStrip } from "./TokenStrip";
import { renderTokenDisplay } from "./TopKBars";

export type AttentionHeatmapProps = {
  run: AttentionRun;
  /** Pixel size of each score cell. Capped to fit on screen for long prompts. */
  preferredCellSize?: number;
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
export function AttentionHeatmap({
  run,
  preferredCellSize = 32,
}: AttentionHeatmapProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  // Initial default: prefer the first layer whose kind === 'full' so the
  // heatmap doesn't open on a linear layer ("This layer does not expose
  // softmax scores."). Computed lazily so it only runs on mount; subsequent
  // run swaps are handled by the useEffect below.
  const [selectedLayerIdx, setSelectedLayerIdx] = React.useState(() => {
    const idx = run.attention.findIndex((layer) => layer.kind === "full");
    return idx >= 0 ? idx : 0;
  });
  const [selectedHead, setSelectedHead] = React.useState(0);
  // Reset only when the user's current selection is no longer a full-attention
  // layer in the incoming run (e.g. the backend swapped models entirely). If
  // the selection is still valid we keep it — otherwise typing into the prompt
  // and clicking Run would silently revert "Layer 7" to "Layer 3" every time.
  React.useEffect(() => {
    const current = run.attention[selectedLayerIdx];
    if (current && current.kind === "full") return;
    const idx = run.attention.findIndex((layer) => layer.kind === "full");
    setSelectedLayerIdx(idx >= 0 ? idx : 0);
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
    el.classList.remove("last-row-flash");
    void el.offsetWidth;
    el.classList.add("last-row-flash");
  }, [run]);
  // Bumped on window resize (and DPR-relevant changes like moving between
  // monitors or zooming). Used to retrigger the draw effect so the canvas
  // re-syncs its backing-store size with the current devicePixelRatio.
  const [resizeTick, setResizeTick] = React.useState(0);
  React.useEffect(() => {
    const onResize = () => setResizeTick((n) => n + 1);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
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
  // Heatmap area should fit comfortably in the right-side panel. Cap at the
  // preferred size, but shrink for very long sequences.
  const maxBoardPx = 360;
  const cellSize = Math.max(
    8,
    Math.min(preferredCellSize, Math.floor(maxBoardPx / Math.max(1, seqLen))),
  );
  const board = cellSize * seqLen;

  // Render the heatmap onto the canvas whenever the layer/head/run changes.
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = board * dpr;
    canvas.height = board * dpr;
    canvas.style.width = `${board}px`;
    canvas.style.height = `${board}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background ("masked / zero") color.
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, board, board);

    const layer = run.attention[selectedLayerIdx];
    if (!layer || layer.scores.length === 0) {
      // Linear layers ship empty scores; show a notice.
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "12px var(--font-mono, monospace)";
      ctx.fillText("This layer does not expose softmax scores.", 12, 24);
      return;
    }
    const head = Math.min(selectedHead, layer.numHeads - 1);
    drawAttentionHeatmap(ctx, layer, head, cellSize);

    // Grid (subtle).
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
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
  }, [run, selectedLayerIdx, selectedHead, seqLen, cellSize, board, resizeTick]);

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

  const layerOptions = run.attention.map((layer, idx) => ({
    idx,
    label: `Layer ${layer.layerIndex} (${layer.kind})`,
  }));
  const currentLayer = run.attention[selectedLayerIdx];
  const headCount = currentLayer?.numHeads ?? 0;
  const currentHead = Math.min(selectedHead, Math.max(0, headCount - 1));
  const ariaLabel = currentLayer
    ? `Attention heatmap ${seqLen}×${seqLen}, layer ${currentLayer.layerIndex} head ${currentHead}`
    : `Attention heatmap ${seqLen}×${seqLen}`;
  const liveAnnouncement = hover
    ? `Layer ${currentLayer?.layerIndex ?? selectedLayerIdx} head ${currentHead}, ` +
      `query token ${hover.queryIndex} (${run.tokens[hover.queryIndex]?.text ?? ""}) ` +
      `attends to key token ${hover.keyIndex} (${run.tokens[hover.keyIndex]?.text ?? ""}) ` +
      `with score ${hover.score.toFixed(4)}.`
    : "";

  return (
    <div className="space-y-3">
      {/* Layer + head selectors */}
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-xs uppercase tracking-wider text-muted-foreground">
          Layer
        </label>
        <Select
          value={`${selectedLayerIdx}`}
          onValueChange={(v) => setSelectedLayerIdx(Number(v))}
        >
          <SelectTrigger size="sm" className="min-w-[10rem]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {layerOptions.map((opt) => (
              <SelectItem key={opt.idx} value={`${opt.idx}`}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <label className="ml-2 text-xs uppercase tracking-wider text-muted-foreground">
          Head
        </label>
        <Select
          value={`${selectedHead}`}
          onValueChange={(v) => setSelectedHead(Number(v))}
        >
          <SelectTrigger size="sm" className="min-w-[6rem]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Array.from({ length: headCount }).map((_, h) => (
              <SelectItem key={h} value={`${h}`}>
                {`Head ${h}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
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
                position: "absolute",
                width: 1,
                height: 1,
                overflow: "hidden",
                clip: "rect(0,0,0,0)",
                whiteSpace: "nowrap",
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
            {hover && (
              <HeatmapTooltip
                hover={hover}
                queryToken={run.tokens[hover.queryIndex]!}
                keyToken={run.tokens[hover.keyIndex]!}
                board={board}
              />
            )}
          </div>
        </div>
      </div>

      {/* Color legend */}
      <Legend />

      {/* How-to-read panel — teaches a beginner what the matrix and its
          bottom row actually represent. Always visible. */}
      <div className="space-y-1.5 rounded-md border border-dashed border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <div>
          <span className="font-mono text-foreground">↓</span> row = token doing
          the looking ·{" "}
          <span className="font-mono text-foreground">→</span> column = token
          being looked at · brighter = stronger attention.
        </div>
        <div>
          The outlined <strong className="text-foreground">bottom row</strong>{" "}
          is the prediction step — those bright cells are the tokens the model
          focused on when choosing the next word below.
        </div>
      </div>

      {/* Next-token card — the concrete artifact of one Run. This is the
          single thing a learner can point at and say "the model just
          predicted that". */}
      <div className="rounded-md border border-primary/40 bg-primary/5 px-4 py-3">
        <div className="text-[0.7rem] uppercase tracking-wider text-muted-foreground">
          Next token
        </div>
        <div className="mt-1.5 flex flex-wrap items-baseline gap-2">
          <span className="rounded-md border border-primary/40 bg-background/60 px-2.5 py-1 font-mono text-base text-foreground">
            {renderTokenDisplay(run.generatedToken.text)}
          </span>
          <span className="font-mono text-[0.7rem] text-muted-foreground">
            id {run.generatedToken.id}
          </span>
        </div>
        <div className="mt-2 text-xs text-muted-foreground">
          Greedy argmax of the model's vocabulary at the last position. Edit
          the prompt above and Run again to see this change.
        </div>
      </div>

      {/* Model meta footer */}
      <div className="text-[0.7rem] text-muted-foreground">
        <span className="font-mono">{run.modelMeta.name}</span> · {seqLen}{" "}
        tokens
      </div>
    </div>
  );
}

function HeatmapTooltip({
  hover,
  queryToken,
  keyToken,
  board,
}: {
  hover: {
    queryIndex: number;
    keyIndex: number;
    x: number;
    y: number;
    score: number;
  };
  queryToken: { text: string };
  keyToken: { text: string };
  board: number;
}) {
  // Position the tooltip so it never goes off the right/bottom edge. On very
  // small boards (board < 220), the clamp would underflow to a negative value
  // and pin the tooltip off-screen to the left — guard with Math.max(0, ...).
  const left = Math.max(0, Math.min(hover.x + 12, board - 220));
  const top = Math.max(0, Math.min(hover.y + 12, board - 80));
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-md border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md"
      style={{ left, top, minWidth: 200 }}
    >
      <div className="font-mono text-[0.7rem] text-muted-foreground">
        query {hover.queryIndex} → key {hover.keyIndex}
      </div>
      <div className="mt-1">
        <span className="text-muted-foreground">query:</span>{" "}
        <span className="font-mono">{JSON.stringify(queryToken.text)}</span>
      </div>
      <div>
        <span className="text-muted-foreground">key:</span>{" "}
        <span className="font-mono">{JSON.stringify(keyToken.text)}</span>
      </div>
      <div className="mt-1">
        <span className="text-muted-foreground">score:</span>{" "}
        <span className="font-mono">{hover.score.toFixed(4)}</span>
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
          <div
            key={s}
            style={{ background: colorForScore(s), width: 16, height: "100%" }}
          />
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
  options?: { colorFn?: (v: number) => string },
): void {
  const colorFn = options?.colorFn ?? colorForScore;
  const seqLen = Math.round(
    Math.sqrt(layer.scores.length / Math.max(1, layer.numHeads)),
  );
  if (seqLen === 0) return;
  const safeHead = Math.min(headIdx, layer.numHeads - 1);
  const headStride = seqLen * seqLen;
  const headOffset = safeHead * headStride;
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      const v = layer.scores[headOffset + i * seqLen + j] ?? 0;
      if (v <= 0) continue;
      ctx.fillStyle = colorFn(v);
      ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
    }
  }
}
