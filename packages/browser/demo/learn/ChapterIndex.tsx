import { ArrowLeftIcon, MessageSquareIcon, PlayIcon } from "lucide-react";
import * as React from "react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { CHAPTERS, type ChapterMeta } from "./chapters";

export type ChapterIndexProps = {
  onOpenChapter: (chapterId: string) => void;
  onBackToLanding: () => void;
  onOpenFreeChat: () => void;
};

// (Previous "curriculum phase" colour bands lived here. They drove the
// horizontal pill rail that has been replaced by the animated forward-pass
// flow below — see {@link ForwardPassFlow}. The flow's lit-stage gradient
// renders the same "conceptual grouping" idea more directly, by showing
// where each chapter's content actually fires inside one inference step.)

export function ChapterIndex({
  onOpenChapter,
  onBackToLanding,
  onOpenFreeChat,
}: ChapterIndexProps) {
  return (
    <div className="absolute inset-0 z-10 overflow-y-auto bg-background">
      <div className="mx-auto flex w-full max-w-5xl flex-col px-6 py-10">
        {/* Top bar */}
        <div className="mb-8 flex items-center justify-between">
          <button
            type="button"
            onClick={onBackToLanding}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeftIcon className="size-4" />
            Back
          </button>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenFreeChat}
            className="gap-2"
          >
            <MessageSquareIcon className="size-4" />
            Open free chat
          </Button>
        </div>

        {/* Title */}
        <div className="mb-8">
          <div className="mb-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Learn LLMs · powered by Qwen3.5 in your browser
          </div>
          <h1 className="text-4xl font-semibold tracking-tight text-foreground">
            Chapters
          </h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            Ten guided lessons that explain how a modern transformer LLM works,
            using the real model running in your browser via WebGPU. Read the
            prose on the left, then poke at the live model on the right.
          </p>
        </div>

        {/* Animated forward-pass flow — shows the order each chapter's
            content runs inside the model. Replaces the older horizontal
            "Suggested order" pill rail. */}
        <ForwardPassFlow onOpenChapter={onOpenChapter} />

        {/* Grid of chapter cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CHAPTERS.map((chapter) => (
            <ChapterCard
              key={chapter.id}
              chapter={chapter}
              onOpen={() =>
                chapter.available ? onOpenChapter(chapter.id) : undefined
              }
            />
          ))}
        </div>

        <p className="mt-10 text-xs text-muted-foreground">
          Each chapter is self-contained, but the suggested order builds the
          model from the bottom up — text in, attention through the middle,
          sampling out.
        </p>
      </div>
    </div>
  );
}

/**
 * Animated forward-pass flow diagram.
 *
 * Visualizes what one inference step actually does inside the model, with
 * a glowing "hidden-state" dot that travels through every stage. Each
 * stage cross-references a chapter — click any stage to open the chapter
 * that explains it. The middle of the flow is wrapped in a "× 24 layers"
 * boundary that, during animation, ticks a layer counter from 1 to 24 to
 * convey that the loop body repeats 24 times for every forward pass.
 *
 * Replaces the older horizontal "Suggested order" pill rail. The lesson is
 * the same — order matters — but now the order is the real execution
 * order, not the teaching order.
 *
 * Honors prefers-reduced-motion: skips the traveling dot and counter
 * animations and just shows the static diagram with the final dot at the
 * sampling stage.
 */
type StageId =
  | "tokenize"
  | "embedding"
  | "rmsnorm_pre"
  | "attn"
  | "attn_rope"
  | "attn_mha"
  | "attn_kvcache"
  | "residual_attn"
  | "rmsnorm_post"
  | "mlp"
  | "residual_mlp"
  | "final_norm"
  | "lm_head"
  | "sampling";

type StageVisual = {
  id: StageId;
  label: string;
  /** Chapter id to open on click — undefined means non-clickable (e.g. residual). */
  chapterId?: string;
  /** Chapter number shown as a pill on the right of the box. */
  chapterNum?: number;
  /** Top-left SVG y. */
  y: number;
  /** Height in SVG units. */
  height: number;
  /** Box variant — controls width + styling. */
  variant: "outer" | "inner" | "sub" | "residual";
};

// Layout constants. The SVG is rendered with a fixed viewBox and scales
// responsively, so all positions here are in SVG units.
const SVG_WIDTH = 600;
const OUTER_X = 150;
const OUTER_W = 300;
const LOOP_X = 50;
const LOOP_W = 500;
const INNER_X = 110;
const INNER_W = 380;
const SUB_X = 150;
const SUB_W = 300;
const RES_X = 200;
const RES_W = 200;

const STAGES: readonly StageVisual[] = [
  { id: "tokenize", label: "Tokenize", chapterId: "tokenization", chapterNum: 1, y: 18, height: 36, variant: "outer" },
  { id: "embedding", label: "Embedding lookup", chapterId: "embeddings", chapterNum: 2, y: 82, height: 36, variant: "outer" },
  // Loop boundary spans y=146 to y=506
  { id: "rmsnorm_pre", label: "RMSNorm", chapterId: "rmsnorm", chapterNum: 6, y: 178, height: 32, variant: "inner" },
  { id: "attn", label: "Self-attention", chapterId: "attention", chapterNum: 3, y: 234, height: 28, variant: "inner" },
  { id: "attn_rope", label: "apply RoPE to Q, K", chapterId: "rope", chapterNum: 5, y: 270, height: 22, variant: "sub" },
  { id: "attn_mha", label: "8 query heads · 2 K/V heads", chapterId: "multi-head-gqa", chapterNum: 4, y: 296, height: 22, variant: "sub" },
  { id: "attn_kvcache", label: "read past K/V · append new", chapterId: "kv-cache", chapterNum: 10, y: 322, height: 22, variant: "sub" },
  { id: "residual_attn", label: "+ residual", y: 354, height: 22, variant: "residual" },
  { id: "rmsnorm_post", label: "RMSNorm", chapterId: "rmsnorm", chapterNum: 6, y: 388, height: 32, variant: "inner" },
  { id: "mlp", label: "MLP block (SwiGLU)", chapterId: "mlp", chapterNum: 7, y: 444, height: 32, variant: "inner" },
  { id: "residual_mlp", label: "+ residual", y: 500, height: 22, variant: "residual" },
  // (loop boundary ends ~y=536)
  { id: "final_norm", label: "Final RMSNorm", chapterId: "rmsnorm", chapterNum: 6, y: 564, height: 32, variant: "outer" },
  { id: "lm_head", label: "LM head → 152K logits", y: 628, height: 32, variant: "outer" },
  { id: "sampling", label: "Sampling → next token", chapterId: "sampling", chapterNum: 9, y: 692, height: 36, variant: "outer" },
];

const LOOP_TOP_Y = 146;
const LOOP_BOTTOM_Y = 536;
// 750 was tight; the "next token → ·floor" reveal that appears under
// Sampling needs ~50 more units of canvas so it isn't clipped.
const SVG_HEIGHT = 800;

function stageX(variant: StageVisual["variant"]): { x: number; w: number } {
  switch (variant) {
    case "outer":
      return { x: OUTER_X, w: OUTER_W };
    case "inner":
      return { x: INNER_X, w: INNER_W };
    case "sub":
      return { x: SUB_X, w: SUB_W };
    case "residual":
      return { x: RES_X, w: RES_W };
  }
}

function stageById(id: StageId): StageVisual {
  const found = STAGES.find((s) => s.id === id);
  if (!found) throw new Error(`Unknown stage id: ${id}`);
  return found;
}

// Dot lives in the fixed left margin (x=DOT_GUTTER_X) and slides
// vertically as scenes advance. Aligned with the centre-line of each
// active stage, it reads as a "you are here" pointer next to the boxes
// — never overlapping text, never ambiguous about which box is current.
// Earlier iterations placed the dot on the central spine, which either
// covered the box label or sat in the arrow gutter above the box (making
// it look mid-transit between the previous stage and the active one).
const DOT_GUTTER_X = 80;
function dotPosition(id: StageId): { cx: number; cy: number } {
  const stage = stageById(id);
  return { cx: DOT_GUTTER_X, cy: stage.y + stage.height / 2 };
}

// Scene script — what the dot does on Play. Each scene names the stage it
// arrives at and how long it stays there before moving to the next. The
// layer-counter ticks independently while the dot is inside the loop.
type Scene = { active: StageId; durationMs: number };
const SCENES: readonly Scene[] = [
  { active: "tokenize", durationMs: 800 },
  { active: "embedding", durationMs: 800 },
  // Loop body — one visible iteration.
  { active: "rmsnorm_pre", durationMs: 600 },
  { active: "attn", durationMs: 500 },
  { active: "attn_rope", durationMs: 450 },
  { active: "attn_mha", durationMs: 450 },
  { active: "attn_kvcache", durationMs: 500 },
  { active: "residual_attn", durationMs: 400 },
  { active: "rmsnorm_post", durationMs: 600 },
  { active: "mlp", durationMs: 700 },
  { active: "residual_mlp", durationMs: 1200 }, // hold here while counter races to 24
  // Exit loop.
  { active: "final_norm", durationMs: 700 },
  { active: "lm_head", durationMs: 800 },
  { active: "sampling", durationMs: 1200 },
];

const LOOP_STAGES = new Set<StageId>([
  "rmsnorm_pre",
  "attn",
  "attn_rope",
  "attn_mha",
  "attn_kvcache",
  "residual_attn",
  "rmsnorm_post",
  "mlp",
  "residual_mlp",
]);

/** Hard-coded example used to ground the animation in a concrete prompt.
 * We don't actually run the model here — the index page mounts before the
 * worker is even started — but the prompt + predicted next token give the
 * user a real, gradable artifact ("for THIS sentence, the model would
 * produce THIS token") instead of an abstract pipeline diagram. */
const EXAMPLE_PROMPT = "The cat sat on the";
const EXAMPLE_PREDICTED_TOKEN = "·floor";

function ForwardPassFlow({
  onOpenChapter,
}: {
  onOpenChapter: (chapterId: string) => void;
}) {
  // Current scene index. Before the user clicks Run we sit at scene 0 with
  // the dot perched above Tokenize — the diagram is fully drawn but motion
  // hasn't started yet. status === "idle" gates that.
  const [activeIdx, setActiveIdx] = React.useState<number>(0);
  // Layer counter ticks from 1 to 24 while the dot is in the loop.
  const [layerCounter, setLayerCounter] = React.useState<number>(1);
  // Lifecycle: "idle" before first Run, "playing" while the timeline runs,
  // "done" when the dot has reached Sampling.
  const [status, setStatus] = React.useState<"idle" | "playing" | "done">("idle");
  // Bumped on each Run / Replay click to (re)schedule the animation.
  // -1 means "never clicked" — we stay idle on mount.
  const [runTick, setRunTick] = React.useState<number>(-1);

  React.useEffect(() => {
    if (runTick < 0) return; // idle on initial mount

    const reducedMotion =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reducedMotion) {
      // Snap to the final state — dot at sampling, counter at 24.
      setActiveIdx(SCENES.length - 1);
      setLayerCounter(24);
      setStatus("done");
      return;
    }

    setActiveIdx(0);
    setLayerCounter(1);
    setStatus("playing");

    const sceneTimers: ReturnType<typeof setTimeout>[] = [];
    let acc = 0;
    SCENES.forEach((scene, i) => {
      if (i === 0) return; // scene 0 starts immediately
      acc += SCENES[i - 1]!.durationMs;
      const t = setTimeout(() => setActiveIdx(i), acc);
      sceneTimers.push(t);
    });
    const totalMs = acc + SCENES[SCENES.length - 1]!.durationMs;
    const doneTimer = setTimeout(() => setStatus("done"), totalMs);
    sceneTimers.push(doneTimer);

    // Layer counter: start at the first loop scene and tick 1 → 24
    // across the loop section. We compute the loop start/end offsets so
    // the counter pace adapts to the actual scene durations.
    const firstLoopIdx = SCENES.findIndex((s) => LOOP_STAGES.has(s.active));
    const lastLoopIdx = SCENES.map((s, i) => (LOOP_STAGES.has(s.active) ? i : -1))
      .filter((i) => i >= 0)
      .reduce((a, b) => Math.max(a, b), -1);
    let loopStartMs = 0;
    for (let i = 0; i < firstLoopIdx; i++) loopStartMs += SCENES[i]!.durationMs;
    let loopEndMs = loopStartMs;
    for (let i = firstLoopIdx; i <= lastLoopIdx; i++) {
      loopEndMs += SCENES[i]!.durationMs;
    }
    const counterTimers: ReturnType<typeof setInterval | typeof setTimeout>[] = [];
    const counterStart = setTimeout(() => {
      const duration = Math.max(100, loopEndMs - loopStartMs - 200);
      const ticksTotal = 23; // we already start at 1
      const tickMs = duration / ticksTotal;
      let tick = 0;
      const counterInterval = setInterval(() => {
        tick += 1;
        setLayerCounter((prev) => Math.min(24, prev + 1));
        if (tick >= ticksTotal) {
          clearInterval(counterInterval);
        }
      }, tickMs);
      counterTimers.push(counterInterval);
    }, loopStartMs + 50);
    counterTimers.push(counterStart);

    return () => {
      for (const t of sceneTimers) clearTimeout(t);
      for (const t of counterTimers) {
        clearTimeout(t as ReturnType<typeof setTimeout>);
        clearInterval(t as ReturnType<typeof setInterval>);
      }
    };
  }, [runTick]);

  const activeStageId = SCENES[activeIdx]?.active ?? "tokenize";
  const activeStage = stageById(activeStageId);
  const { cx, cy } = dotPosition(activeStageId);

  const runButtonLabel =
    status === "idle" ? "Run" : status === "playing" ? "Restart" : "Replay";

  return (
    <div className="mb-8 space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
          One forward pass, end to end
        </div>
        <div className="text-[11px] text-muted-foreground">
          The dot is the hidden state. Each box is a stage — click to open
          its chapter.
        </div>
      </div>

      {/* Prompt + Run row — grounds the abstract pipeline in a concrete
          example. The prompt is fixed (we don't run the real model on the
          index page) but the predicted next token revealed at the end is
          the actual greedy completion the chapters' demos produce. */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-dashed border-border bg-muted/20 px-3 py-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          Prompt
        </span>
        <code className="rounded bg-background px-2 py-0.5 font-mono text-xs text-foreground">
          {EXAMPLE_PROMPT}
        </code>
        <button
          type="button"
          onClick={() => setRunTick((n) => n + 1)}
          disabled={status === "playing"}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-foreground hover:bg-primary/20 disabled:opacity-50"
        >
          <PlayIcon className="size-3.5" />
          {runButtonLabel}
        </button>
        <span className="basis-full text-[11px] text-muted-foreground">
          Click <strong>Run</strong> to watch the model compute the next
          token for this prompt — one full forward pass through every stage
          below.
        </span>
      </div>

      <div className="overflow-x-auto">
        <svg
          role="img"
          aria-label="Animated diagram of one transformer forward pass — tokenize, embedding lookup, 24 layers, final norm, LM head, sampling"
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="block h-auto w-full max-w-[680px]"
        >
          {/* Loop boundary box. */}
          <rect
            x={LOOP_X}
            y={LOOP_TOP_Y}
            width={LOOP_W}
            height={LOOP_BOTTOM_Y - LOOP_TOP_Y}
            rx={12}
            className="fill-emerald-500/5 stroke-emerald-500/40"
            strokeDasharray="6 4"
            strokeWidth={1.4}
          />
          {/* Loop title pill (× 24 layers · ch 8). Wider than the earlier
              132u so the text doesn't bleed past the rounded background. */}
          <rect
            x={LOOP_X + 14}
            y={LOOP_TOP_Y - 12}
            width={180}
            height={24}
            rx={12}
            className="fill-background stroke-emerald-500/60"
            strokeWidth={1}
          />
          <text
            x={LOOP_X + 14 + 90}
            y={LOOP_TOP_Y + 5}
            textAnchor="middle"
            className="fill-foreground"
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: 11,
              letterSpacing: "0.04em",
            }}
          >
            × 24 layers · chapter 8
          </text>
          {/* Loop iteration counter pill (right side). Widened from 78u
              so "layer 24 / 24" fits without clipping the trailing "4". */}
          <rect
            x={LOOP_X + LOOP_W - 124}
            y={LOOP_TOP_Y - 12}
            width={110}
            height={24}
            rx={12}
            className="fill-background stroke-emerald-500/60"
            strokeWidth={1}
          />
          <text
            x={LOOP_X + LOOP_W - 124 + 55}
            y={LOOP_TOP_Y + 5}
            textAnchor="middle"
            className="fill-foreground"
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: 11,
            }}
          >
            layer {layerCounter} / 24
          </text>

          {/* Connecting arrows between stages. */}
          {STAGES.map((stage, i) => {
            if (i === STAGES.length - 1) return null;
            const next = STAGES[i + 1]!;
            // Suppress arrows between sub-items of attention and between
            // residual rows where the spacing is too tight for an arrow head.
            const skip = stage.variant === "sub" && next.variant === "sub";
            if (skip) return null;
            const fromX = SVG_WIDTH / 2;
            const fromY = stage.y + stage.height;
            const toX = SVG_WIDTH / 2;
            const toY = next.y;
            return (
              <line
                key={`arrow-${stage.id}-${next.id}`}
                x1={fromX}
                y1={fromY + 2}
                x2={toX}
                y2={toY - 4}
                className="stroke-muted-foreground/50"
                strokeWidth={1.2}
                markerEnd="url(#fp-arrowhead)"
              />
            );
          })}

          <defs>
            <marker
              id="fp-arrowhead"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" className="fill-muted-foreground/60" />
            </marker>
          </defs>

          {/* Stage boxes. */}
          {STAGES.map((stage) => {
            const { x, w } = stageX(stage.variant);
            const isActive = stage.id === activeStageId;
            // For the parent attention header: also highlight when one of
            // its sub-items is the active scene.
            const isAttnHeader =
              stage.id === "attn" &&
              (activeStageId === "attn_rope" ||
                activeStageId === "attn_mha" ||
                activeStageId === "attn_kvcache");
            const lit = isActive || isAttnHeader;
            const clickable = stage.chapterId !== undefined;
            const handleClick = () => {
              if (stage.chapterId) onOpenChapter(stage.chapterId);
            };
            // Background + border classes per variant + lit state.
            const fill =
              stage.variant === "residual"
                ? "transparent"
                : lit
                  ? "url(#fp-lit-grad)"
                  : "var(--background)";
            const strokeClass =
              stage.variant === "residual"
                ? "stroke-muted-foreground/30"
                : lit
                  ? "stroke-primary"
                  : stage.variant === "sub"
                    ? "stroke-emerald-500/30"
                    : "stroke-border";
            const textColor =
              stage.variant === "residual" ? "fill-muted-foreground" : "fill-foreground";
            return (
              <g
                key={stage.id}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : undefined}
                aria-label={
                  clickable
                    ? `Open chapter ${stage.chapterNum}: ${stage.label}`
                    : undefined
                }
                onClick={clickable ? handleClick : undefined}
                onKeyDown={
                  clickable
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleClick();
                        }
                      }
                    : undefined
                }
                style={
                  clickable
                    ? { cursor: "pointer", outline: "none" }
                    : { pointerEvents: "none" }
                }
              >
                <rect
                  x={x}
                  y={stage.y}
                  width={w}
                  height={stage.height}
                  rx={stage.variant === "residual" ? 10 : 8}
                  fill={fill}
                  className={[
                    strokeClass,
                    stage.variant === "residual" ? "" : "transition-colors",
                  ].join(" ")}
                  strokeWidth={lit ? 1.6 : 1}
                  strokeDasharray={stage.variant === "residual" ? "4 3" : undefined}
                />
                <text
                  x={x + 14}
                  y={stage.y + stage.height / 2 + 4}
                  className={textColor}
                  style={{
                    fontFamily:
                      stage.variant === "sub"
                        ? "var(--font-mono, monospace)"
                        : "inherit",
                    fontSize: stage.variant === "sub" ? 11 : 13,
                    fontWeight:
                      stage.variant === "inner" || stage.variant === "outer"
                        ? 500
                        : 400,
                  }}
                >
                  {stage.variant === "sub" ? `└─ ${stage.label}` : stage.label}
                </text>
                {/* Chapter number pill on the right. */}
                {stage.chapterNum ? (
                  <g>
                    <rect
                      x={x + w - 56}
                      y={stage.y + stage.height / 2 - 10}
                      width={44}
                      height={20}
                      rx={10}
                      className={
                        lit
                          ? "fill-primary/15 stroke-primary/50"
                          : "fill-muted/40 stroke-border"
                      }
                      strokeWidth={1}
                    />
                    <text
                      x={x + w - 34}
                      y={stage.y + stage.height / 2 + 4}
                      textAnchor="middle"
                      className="fill-muted-foreground"
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: 10,
                      }}
                    >
                      ch {stage.chapterNum}
                    </text>
                  </g>
                ) : null}
              </g>
            );
          })}

          <defs>
            <linearGradient id="fp-lit-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.18" />
              <stop offset="100%" stopColor="var(--primary)" stopOpacity="0.06" />
            </linearGradient>
            <radialGradient id="fp-dot-grad">
              <stop offset="0%" stopColor="var(--primary)" stopOpacity="1" />
              <stop offset="60%" stopColor="var(--primary)" stopOpacity="0.7" />
              <stop offset="100%" stopColor="var(--primary)" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Predicted next token reveal — appears when the dot reaches the
              Sampling stage and stays until the next Run. Tied to the actual
              token the chapter-3/4 demos produce for the example prompt. */}
          {(() => {
            const samplingStage = stageById("sampling");
            const showPrediction =
              activeStageId === "sampling" || status === "done";
            const cyPred = samplingStage.y + samplingStage.height + 18;
            return (
              <g
                style={{
                  opacity: showPrediction ? 1 : 0,
                  transition: "opacity 320ms ease-out",
                }}
                aria-hidden={!showPrediction}
              >
                <text
                  x={SVG_WIDTH / 2}
                  y={cyPred}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: 12,
                  }}
                >
                  next token →
                </text>
                <rect
                  x={SVG_WIDTH / 2 - 48}
                  y={cyPred + 8}
                  width={96}
                  height={26}
                  rx={6}
                  className="fill-primary/15 stroke-primary/60"
                  strokeWidth={1.2}
                />
                <text
                  x={SVG_WIDTH / 2}
                  y={cyPred + 26}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: 14,
                    fontWeight: 600,
                  }}
                >
                  {EXAMPLE_PREDICTED_TOKEN}
                </text>
              </g>
            );
          })()}

          {/* The traveling "hidden state" dot. CSS transition smoothly tweens
              its (cx, cy) between scenes. Floats just above the active box
              in the arrow gutter so it never overlaps box text. The dot is
              hidden in the "idle" state — it appears once Run is clicked. */}
          {status !== "idle" ? (
            <>
              <circle
                cx={cx}
                cy={cy}
                r={12}
                fill="url(#fp-dot-grad)"
                style={{
                  transition:
                    "cx 320ms cubic-bezier(0.4, 0, 0.2, 1), cy 320ms cubic-bezier(0.4, 0, 0.2, 1)",
                }}
                aria-hidden="true"
              />
              <circle
                cx={cx}
                cy={cy}
                r={4}
                className="fill-primary"
                style={{
                  transition:
                    "cx 320ms cubic-bezier(0.4, 0, 0.2, 1), cy 320ms cubic-bezier(0.4, 0, 0.2, 1)",
                }}
                aria-hidden="true"
              />
            </>
          ) : null}
        </svg>
      </div>

      <div
        className="text-xs text-muted-foreground"
        aria-live="polite"
      >
        {status === "idle" ? (
          <>Click <strong>Run</strong> above to start the animation.</>
        ) : (
          <>
            {status === "playing" ? "Now:" : "Done — "}{" "}
            <span className="font-mono text-foreground">{activeStage.label}</span>
            {activeStage.chapterNum ? (
              <span className="ml-1.5 text-muted-foreground">
                (chapter {activeStage.chapterNum})
              </span>
            ) : null}
          </>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        All chapters are independent — click any box to open that chapter.
        This isn't a recommended <em>reading</em> order; it's the order each
        chapter's content runs inside the model.
      </p>
    </div>
  );
}

function ChapterCard({
  chapter,
  onOpen,
}: {
  chapter: ChapterMeta;
  onOpen: () => void;
}) {
  const interactive = chapter.available;
  return (
    <Card
      onClick={interactive ? onOpen : undefined}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : -1}
      onKeyDown={(e) => {
        if (!interactive) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      className={
        interactive
          ? "cursor-pointer transition-colors hover:bg-card/80 hover:border-primary/40"
          : "opacity-60"
      }
      aria-disabled={!interactive}
    >
      <CardHeader>
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">
            Ch. {chapter.number.toString().padStart(2, "0")}
          </span>
          {chapter.available ? (
            <Badge variant="default">Ready</Badge>
          ) : (
            <Badge variant="secondary">Coming soon</Badge>
          )}
        </div>
        <CardTitle className="text-lg">{chapter.title}</CardTitle>
        <CardDescription>{chapter.blurb}</CardDescription>
      </CardHeader>
      <CardContent>
        <span
          className={
            interactive
              ? "text-sm text-primary"
              : "text-sm text-muted-foreground"
          }
        >
          {interactive ? "Open chapter →" : "Not yet authored"}
        </span>
      </CardContent>
    </Card>
  );
}
