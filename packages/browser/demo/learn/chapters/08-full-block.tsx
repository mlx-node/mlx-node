import * as React from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import type {
  AttentionRun,
  HiddenStatePointStats,
  HiddenStateStep,
} from "../../../src/inspector-types";
import { runForInspector } from "../../lib/inspector-client";
import { DemoCallout } from "../inspector/DemoCallout";
import { Prose } from "../Prose";
import { ChapterFrame } from "../scaffolding/ChapterFrame";
import type { ChapterLearningData } from "../scaffolding/learning-data";

const DEFAULT_PROMPT = "A transformer stacks the same block many times.";
const DEFAULT_NUM_LAYERS = 24;
// Qwen3.5-0.8B; matches the model's hidden_size.
const HIDDEN_DIM = 1024;
// Qwen3.5-0.8B uses full attention every 4th layer.
const FULL_ATTN_INTERVAL = 4;

const CAPTURE_POINTS = [
  "pre_attn_input",
  "post_attn_norm",
  "attn_output",
  "post_attn_residual",
  "post_mlp_norm",
  "mlp_output",
  "post_mlp_residual",
] as const;

type CapturePoint = (typeof CAPTURE_POINTS)[number];

const POINT_LABELS: Record<CapturePoint, string> = {
  pre_attn_input: "Layer input",
  post_attn_norm: "After pre-attn RMSNorm",
  attn_output: "Attention output",
  post_attn_residual: "After attn residual",
  post_mlp_norm: "After pre-MLP RMSNorm",
  mlp_output: "MLP output",
  post_mlp_residual: "Layer output",
};

/**
 * Scaffolding metadata for chapter 8 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[7].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: "full-block",
  objective:
    "Read a transformer layer end-to-end: pre-norm, attention, residual, pre-norm, MLP, residual, repeat.",
  problem:
    "Each sub-block was introduced in isolation; you still need to see how they wire together and how the stack composes.",
  minutes: 7,
  glossary: [
    {
      term: "decoder layer",
      definition:
        "One pre-norm block: x_attn = x + Attention(RMSNorm(x)); x_out = x_attn + MLP(RMSNorm(x_attn)). Qwen3.5-0.8B stacks 24 of these.",
    },
    {
      term: "residual stream",
      definition:
        "The un-normalized hidden state running the full depth of the tower. Every block reads from it and writes a delta back.",
    },
    {
      term: "sub-block",
      definition:
        "Either attention or the MLP. Each layer contains exactly two sub-blocks, each wrapped in its own norm + residual.",
    },
    {
      term: "hybrid attention",
      definition:
        "Qwen3.5's recipe: every fourth layer uses softmax attention, the rest use a linear variant (GatedDeltaNet).",
    },
    {
      term: "linear attention",
      definition:
        "An approximate attention that runs in fixed memory per token. Trades some exact-recall for very cheap long-context decode.",
    },
    {
      term: "capture point",
      definition:
        "A named place inside a layer (pre_attn_input, mlp_output, ...) where the inspector reads activation stats out.",
    },
  ],
  takeaways: [
    "Every layer is the same shape — two pre-norm + sub-block + residual pairs — and that shape stacks 24 times.",
    "Residual stream magnitude grows with depth even though each individual sub-block's output stays small.",
    "Qwen3.5 isn't pure transformer — only 6 of its 24 layers use softmax attention; the rest run linear attention.",
  ],
  exercise: {
    prompt:
      "After the auto-run, drag the 3D tower to face you, then click layer 3 and layer 4. How does the 'Layer 3 - Full attention' label change vs 'Layer 4 - Linear attention', and how do the warm/cool ring colors compare across the tower?",
    answer:
      "Layer 3 (and 7, 11, 15, 19, 23) shows the 'Full attention' label and uses a warm orange-to-red ramp; the others show 'Linear attention' and use a cool indigo-to-violet ramp. The colour gradient also shows the residual-stream magnitude climbing as you scan up the tower — a direct picture of the highway accumulating contributions.",
  },
  quiz: [
    {
      id: "q1-block-order",
      prompt: "In a Qwen3.5 layer (pre-norm), what is the actual order of operations?",
      options: [
        {
          id: "a",
          label:
            "RMSNorm -> Attention -> add to residual; RMSNorm -> MLP -> add to residual.",
        },
        {
          id: "b",
          label: "Attention -> RMSNorm -> MLP -> RMSNorm -> residual at the end.",
        },
        {
          id: "c",
          label: "Attention -> MLP -> single RMSNorm at the end.",
        },
      ],
      correctId: "a",
      explanation:
        "Pre-norm wraps each sub-block individually. Each sub-block writes its result back into the un-normalized residual stream.",
    },
    {
      id: "q2-hybrid-full",
      prompt: "How many of Qwen3.5-0.8B's 24 layers use the softmax attention you read about in chapter 3?",
      options: [
        { id: "a", label: "All 24." },
        { id: "b", label: "6 — one in every four." },
        { id: "c", label: "12 — every other layer." },
      ],
      correctId: "b",
      explanation:
        "Layers 3, 7, 11, 15, 19, 23 are full softmax attention. The 18 others run GatedDeltaNet linear attention.",
    },
    {
      id: "q3-residual-grows",
      prompt: "Why does the per-token L2 of the residual stream grow as you scan up the tower?",
      options: [
        {
          id: "a",
          label:
            "Each layer adds its sub-block outputs into the residual stream; magnitudes accumulate even though each individual write is small.",
        },
        {
          id: "b",
          label: "Later layers use larger learned gains in RMSNorm.",
        },
        {
          id: "c",
          label: "The model upcasts to higher precision at deeper layers.",
        },
      ],
      correctId: "a",
      explanation:
        "The residual stream is a sum of additions, never overwrites. Even small contributions compound across 24 layers.",
    },
  ],
};

export function FullBlockChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
      <h1>The full transformer block</h1>
      <p>
        The last five chapters introduced individual ingredients — attention,
        multi-head &amp; GQA, RoPE, RMSNorm, the gated MLP — as if each lived
        in isolation. They do not. Every one of Qwen3.5-0.8B's 24 decoder
        layers stitches them together in the <em>same</em> fixed pattern, then
        the model stacks that pattern 24 times. This chapter zooms out: one
        layer as a whole, and a stack of them as a tower.
      </p>

      <h2>Pre-norm: norm, sub-block, residual</h2>
      <p>
        Qwen3.5 follows the modern <strong>pre-norm</strong> recipe. Inside
        each layer there are two sub-blocks (attention and MLP), and each is
        wrapped the same way: normalize the input, run the sub-block on the
        normalized copy, add the result back into the residual stream.
      </p>
      <pre>
        <code>{`x_attn = x + Attention(RMSNorm(x))
x_out  = x_attn + MLP(RMSNorm(x_attn))`}</code>
      </pre>
      <p>
        Two RMSNorms, two residual adds, one attention, one MLP — that is the
        full layer. The 3D widget below shows it as a tall, narrow tower:
        the residual stream runs from bottom to top, and each "ring" you can
        see in the tower is one of those 24 layers.
      </p>

      <h2>What each piece contributes</h2>
      <ul>
        <li>
          <strong>Attention</strong> (chapter 3) — the only place in the layer
          where information moves <em>across tokens</em>. Without it, every
          position would evolve independently.
        </li>
        <li>
          <strong>Multi-head &amp; GQA</strong> (chapter 4) — attention is run
          in parallel by many heads sharing a smaller pool of K/V projections,
          so the model can look at several patterns at once without inflating
          the KV cache.
        </li>
        <li>
          <strong>RoPE</strong> (chapter 5) — the rotation applied to Q and K
          inside attention. It's how the model knows token 5 is later than
          token 3 without a separate position embedding.
        </li>
        <li>
          <strong>RMSNorm</strong> (chapter 6) — applied <em>before</em> each
          sub-block. It collapses the magnitude of the input to roughly{" "}
          <code>sqrt(hidden_dim)</code> ≈ {Math.sqrt(HIDDEN_DIM).toFixed(0)} so
          the attention scores and MLP activations don't explode.
        </li>
        <li>
          <strong>MLP</strong> (chapter 7) — a per-token feed-forward with the
          SwiGLU gated nonlinearity. It is where most of the parameters live,
          and where per-token feature computation happens.
        </li>
      </ul>

      <h2>The residual highway</h2>
      <p>
        Notice that nothing inside the layer <em>replaces</em> the hidden
        state — both the attention and the MLP results are <em>added</em> to
        whatever was already there. The residual stream is a highway that
        runs the full depth of the model. Each block reads from it, computes
        a small correction, and writes the correction back. The model's final
        prediction is the accumulation of every block's contribution, not the
        output of the last block alone.
      </p>
      <p>
        Two consequences fall out. First, gradients: backprop can flow
        straight through the identity path even in a 24-layer stack, which is
        why deep transformers train at all. Second, magnitudes: the residual
        stream grows in L2 with depth (each layer adds a non-zero
        contribution), while every <em>individual</em> sub-block's output
        stays small. The 3D widget colors each ring by the per-token L2 of
        the layer output — you can see the magnitude climb as you scan up
        the tower.
      </p>

      <h2>Hybrid attention: not every layer is "full"</h2>
      <p>
        Qwen3.5 is a <strong>hybrid</strong> stack. Most layers use a fast
        linear-attention variant called <em>GatedDeltaNet</em>; only every
        fourth layer (indices 3, 7, 11, 15, 19, 23 on the 0.8B model) uses
        the classic <code>softmax(QKᵀ/√d)</code> formulation from chapter 3.
        Linear-attention layers do almost all the across-token mixing under
        the tight memory budget of long contexts; full-attention layers
        appear at fixed intervals to do the heavy modelling that linear
        attention can't. The 3D widget highlights full-attention layers in a
        warmer color so you can see the interleave at a glance. Chapter 10
        will dig into how this hybrid recipe works and why it's the new
        default for high-throughput open LLMs.
      </p>

      <h2>What the widget shows</h2>
      <p>
        Press <em>Run</em> to capture all seven hidden-state stats per layer
        for a single forward pass. The 3D tower renders 24 stacked tiles —
        one per layer — colored by per-token L2 of the layer output. Drag to
        rotate, scroll to zoom, click a layer to inspect its stats in the
        side panel. The mini bar chart inside the panel walks the seven
        capture points within the selected layer so you can trace the signal
        from the layer's input through the two sub-blocks and out the top.
      </p>
      </Prose>
    </ChapterFrame>
  );
}

export type FullBlockDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus =
  | { kind: "ok" }
  | { kind: "error"; error: string }
  | { kind: "aborted" }
  | { kind: "empty-prompt" };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function fullAttentionLayerIndices(numLayers: number): number[] {
  const out: number[] = [];
  for (
    let idx = FULL_ATTN_INTERVAL - 1;
    idx < numLayers;
    idx += FULL_ATTN_INTERVAL
  ) {
    out.push(idx);
  }
  return out;
}

function getLayerStats(
  steps: HiddenStateStep[] | undefined,
  layerIdx: number,
): Partial<Record<CapturePoint, HiddenStatePointStats>> {
  const out: Partial<Record<CapturePoint, HiddenStatePointStats>> = {};
  if (!steps || steps.length === 0) return out;
  const step = steps[0];
  if (!step) return out;
  const layer = step.layers.find((l) => l.layerIdx === layerIdx);
  if (!layer) return out;
  for (const stat of layer.stats) {
    if ((CAPTURE_POINTS as readonly string[]).includes(stat.point)) {
      out[stat.point as CapturePoint] = stat;
    }
  }
  return out;
}

function perTokenL2(stats: HiddenStatePointStats | undefined): number | null {
  if (!stats) return null;
  const seqLen = stats.count / HIDDEN_DIM;
  if (seqLen <= 0) return null;
  return stats.l2Norm / Math.sqrt(seqLen);
}

function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(digits);
}

/**
 * Milliseconds between layer reveals when animating the forward pass up
 * the tower. 24 layers × 70ms ≈ 1.7s — long enough to read as motion,
 * short enough not to feel sluggish.
 */
const FORWARD_PASS_STEP_MS = 70;

export function FullBlockDemo({ workerRef, abortRef }: FullBlockDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  const [selectedLayer, setSelectedLayer] = React.useState(0);
  /**
   * Highest layer index currently revealed during the forward-pass
   * animation. `null` means no animation in progress — show the full
   * final state of all 24 tiles. During animation, layers with index
   * above this cursor render as muted gray, and `selectedLayer`
   * follows the cursor so the right-side stats also march up the stack.
   */
  const [revealedUpTo, setRevealedUpTo] = React.useState<number | null>(null);

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);
  const revealTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(
    null,
  );

  const clearRevealTimer = React.useCallback(() => {
    if (revealTimerRef.current !== null) {
      clearInterval(revealTimerRef.current);
      revealTimerRef.current = null;
    }
  }, []);

  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
      clearRevealTimer();
    },
    [clearRevealTimer],
  );

  async function handleRun() {
    if (prompt.trim().length === 0) {
      setStatus({ kind: "empty-prompt" });
      return;
    }

    const myGen = ++runGenRef.current;
    // Cancel any in-flight reveal animation from a previous run so a
    // fast double-click doesn't leave two timers racing.
    clearRevealTimer();
    setRevealedUpTo(null);
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      console.error(
        "[full-block-demo] worker is unavailable — chapter view should have been gated on modelReady",
      );
      setStatus({ kind: "error", error: "Worker is unavailable. Reload the page." });
      setRunning(false);
      return;
    }

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;

    const appSignal = abortRef.current?.signal;
    if (appSignal?.aborted) {
      ctrl.abort();
    }
    const onAppAbort = () => ctrl.abort();
    if (appSignal && !appSignal.aborted) {
      appSignal.addEventListener("abort", onAppAbort, { once: true });
    }

    try {
      const result = await runForInspector(
        worker,
        {
          prompt,
          hiddenStates: {
            points: [...CAPTURE_POINTS],
          },
          maxNewTokens: 1,
        },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;
      console.log("[full-block-demo] runForInspector ok", {
        promptLength: prompt.length,
        layers: result.hiddenStates?.[0]?.layers.length ?? 0,
      });
      setRun(result);
      setStatus({ kind: "ok" });
      const numLayersInResult =
        result.modelMeta?.numLayers ?? DEFAULT_NUM_LAYERS;
      const maxLayer = numLayersInResult - 1;
      if (selectedLayer > maxLayer) {
        setSelectedLayer(0);
      }
      // Animate the forward pass up the tower: reveal one layer at a
      // time so the learner sees the data climbing the stack. Honor
      // prefers-reduced-motion by skipping the animation entirely.
      clearRevealTimer();
      const reducedMotion =
        typeof window !== "undefined" &&
        typeof window.matchMedia === "function" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reducedMotion) {
        setRevealedUpTo(null);
        setSelectedLayer(maxLayer);
      } else {
        setRevealedUpTo(0);
        setSelectedLayer(0);
        revealTimerRef.current = setInterval(() => {
          setRevealedUpTo((prev) => {
            if (prev === null) return null;
            const next = prev + 1;
            if (next >= maxLayer) {
              // Final layer revealed — drop back to "no animation" so
              // the static color encoding takes over.
              clearRevealTimer();
              setSelectedLayer(maxLayer);
              return null;
            }
            setSelectedLayer(next);
            return next;
          });
        }, FORWARD_PASS_STEP_MS);
      }
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info(
          "[full-block-demo] runForInspector aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) {
          setStatus({ kind: "aborted" });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error("[full-block-demo] runForInspector failed", err);
        setStatus({ kind: "error", error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener("abort", onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      if (runGenRef.current === myGen) setRunning(false);
    }
  }

  // Auto-fire one Run on mount — model is guaranteed ready by the gate
  // around chapter rendering, so the 3D stack opens on real Qwen3.5 data.
  const didAutoRunRef = React.useRef(false);
  React.useEffect(() => {
    if (didAutoRunRef.current) return;
    didAutoRunRef.current = true;
    void handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hiddenSteps = run?.hiddenStates;
  const numLayers = run?.modelMeta?.numLayers ?? DEFAULT_NUM_LAYERS;
  const fullAttnIndices = React.useMemo(
    () =>
      new Set(
        run?.modelMeta?.fullAttentionLayerIndices ??
          fullAttentionLayerIndices(numLayers),
      ),
    [run?.modelMeta?.fullAttentionLayerIndices, numLayers],
  );

  const layerOutL2 = React.useMemo(() => {
    const out: Array<{ layerIdx: number; value: number | null }> = [];
    if (!hiddenSteps || hiddenSteps.length === 0) return out;
    const step = hiddenSteps[0];
    if (!step) return out;
    for (const layer of step.layers) {
      const stat = layer.stats.find((s) => s.point === "post_mlp_residual");
      out.push({ layerIdx: layer.layerIdx, value: perTokenL2(stat) });
    }
    out.sort((a, b) => a.layerIdx - b.layerIdx);
    return out;
  }, [hiddenSteps]);

  const layerStats = React.useMemo(
    () => getLayerStats(hiddenSteps, selectedLayer),
    [hiddenSteps, selectedLayer],
  );

  const hasData = Boolean(hiddenSteps && hiddenSteps.length > 0);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="full-block-demo-input"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Prompt
        </label>
        <Textarea
          id="full-block-demo-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="font-mono text-sm"
          placeholder={DEFAULT_PROMPT}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Captures all seven hidden-state stats per layer for a single
            forward pass.
          </span>
        </div>
      </div>

      {status?.kind === "error" ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Inspector run failed.</strong> {status.error}
        </div>
      ) : null}
      {status?.kind === "aborted" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Request cancelled — model was reloaded. Click <strong>Run</strong>{" "}
          again to retry.
        </div>
      ) : null}
      {status?.kind === "empty-prompt" ? (
        <div
          role="alert"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Enter a prompt to see the full transformer block in action.
        </div>
      ) : null}

      {!hasData && running ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Running first inspector pass…
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <StackViewer3D
          layerOutL2={layerOutL2}
          fullAttnIndices={fullAttnIndices}
          selectedLayer={selectedLayer}
          onSelect={(idx) => setSelectedLayer(idx)}
          revealedUpTo={revealedUpTo}
        />
        <LayerSidePanel
          layerIdx={selectedLayer}
          isFullAttention={fullAttnIndices.has(selectedLayer)}
          layerStats={layerStats}
          hasData={hasData}
          modelName={run?.modelMeta?.name ?? "—"}
          numLayers={numLayers}
        />
      </div>

      <DemoCallout
        items={[
          "Rotate the tower — each layer is Norm → Attention → Norm → MLP, with a residual skipping each sub-block.",
          "The colour ramp shows attention type: cooler colours are linear (GatedDeltaNet), warmer are full attention.",
          "Six of the 24 layers are full attention; the rest are linear. That's the hybrid pattern from Chapter 10.",
        ]}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
// 3D stack viewer.
// -----------------------------------------------------------------------------

type StackViewer3DProps = {
  layerOutL2: Array<{ layerIdx: number; value: number | null }>;
  fullAttnIndices: Set<number>;
  selectedLayer: number;
  onSelect: (layerIdx: number) => void;
  /**
   * Index of the highest layer that has been "revealed" by the
   * forward-pass animation. Layers above this index render as a muted
   * gray to imply they haven't been computed yet; the layer at this
   * index gets a brighter emissive boost as the animation cursor. When
   * `null`, no animation is in progress and every tile shows its
   * final L2-encoded color.
   */
  revealedUpTo: number | null;
};

const STACK_TOTAL_HEIGHT = 18;
const STACK_RADIUS = 1.6;
const STACK_LAYER_GAP = 0.12;

function colorForL2(
  value: number | null,
  vmin: number,
  vmax: number,
  isFullAttn: boolean,
): THREE.Color {
  // Two-ramp scheme:
  //   linear-attn layers ramp cool→warm purple/blue.
  //   full-attn  layers ramp warm orange→red to mark the interleave.
  if (value === null || !Number.isFinite(value)) {
    return new THREE.Color(0x444444);
  }
  const t = vmax > vmin ? (value - vmin) / (vmax - vmin) : 0.5;
  const clamped = Math.max(0, Math.min(1, t));
  if (isFullAttn) {
    const cold = new THREE.Color(0xf9c97a);
    const hot = new THREE.Color(0xef4444);
    return cold.lerp(hot, clamped);
  }
  const cold = new THREE.Color(0x4f46e5);
  const hot = new THREE.Color(0x9333ea);
  return cold.lerp(hot, clamped);
}

/** Muted tone used for layers that haven't been "computed" yet in the
 *  forward-pass animation. Reads as inert / pre-activation tile. */
const UNREVEALED_LAYER_COLOR = new THREE.Color(0x2a2f3a);

function StackViewer3D({
  layerOutL2,
  fullAttnIndices,
  selectedLayer,
  onSelect,
  revealedUpTo,
}: StackViewer3DProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const onSelectRef = React.useRef(onSelect);
  const selectedLayerRef = React.useRef(selectedLayer);

  React.useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);
  React.useEffect(() => {
    selectedLayerRef.current = selectedLayer;
  }, [selectedLayer]);

  type SceneState = {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    layerMeshes: THREE.Mesh[];
    layerMaterials: THREE.MeshStandardMaterial[];
    selectionEdges: THREE.LineSegments;
    raycaster: THREE.Raycaster;
    requestRender: () => void;
    fit: () => void;
    cleanup: () => void;
  };

  const sceneRef = React.useRef<SceneState | null>(null);

  // Set up scene once on mount.
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.touchAction = "none";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(7, 8, 10);
    camera.lookAt(0, STACK_TOTAL_HEIGHT / 2, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(4, 8, 6);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.35);
    fill.position.set(-5, 4, -3);
    scene.add(fill);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, STACK_TOTAL_HEIGHT / 2, 0);
    controls.minDistance = 6;
    controls.maxDistance = 40;
    controls.update();

    const layerMeshes: THREE.Mesh[] = [];
    const layerMaterials: THREE.MeshStandardMaterial[] = [];

    // Selection outline (placeholder geometry; updated when selection changes).
    const selectionEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }),
    );
    selectionEdges.visible = false;
    scene.add(selectionEdges);

    const raycaster = new THREE.Raycaster();

    // Render-on-demand: we don't run a rAF loop continuously. Anything that
    // changes scene state calls `requestRender()`, which either renders once
    // (with damping disabled) or arms a short rAF loop until controls settle.
    let damperFrameId: number | null = null;
    function renderOnce() {
      controls.update();
      renderer.render(scene, camera);
    }
    function runDamper() {
      damperFrameId = requestAnimationFrame(() => {
        const moved = controls.update();
        renderer.render(scene, camera);
        if (moved) {
          runDamper();
        } else {
          damperFrameId = null;
        }
      });
    }
    function requestRender() {
      if (damperFrameId === null) {
        runDamper();
      }
    }

    controls.addEventListener("change", () => requestRender());
    controls.addEventListener("start", () => requestRender());

    function fit() {
      const w = container!.clientWidth;
      const h = container!.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderOnce();
    }
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(container);

    function pickLayerAt(clientX: number, clientY: number) {
      const rect = renderer.domElement.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      const hits = raycaster.intersectObjects(layerMeshes, false);
      if (hits.length === 0) return null;
      const hit = hits[0]!.object as THREE.Mesh;
      return typeof hit.userData.layerIdx === "number"
        ? (hit.userData.layerIdx as number)
        : null;
    }

    let pointerDown: { x: number; y: number } | null = null;
    const handlePointerDown = (ev: PointerEvent) => {
      pointerDown = { x: ev.clientX, y: ev.clientY };
    };
    const handlePointerUp = (ev: PointerEvent) => {
      if (!pointerDown) return;
      const dx = ev.clientX - pointerDown.x;
      const dy = ev.clientY - pointerDown.y;
      pointerDown = null;
      if (Math.hypot(dx, dy) > 4) return; // treat as drag, not click
      const idx = pickLayerAt(ev.clientX, ev.clientY);
      if (idx !== null) onSelectRef.current(idx);
    };
    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);

    sceneRef.current = {
      renderer,
      scene,
      camera,
      controls,
      layerMeshes,
      layerMaterials,
      selectionEdges,
      raycaster,
      requestRender,
      fit,
      cleanup: () => {
        if (damperFrameId !== null) cancelAnimationFrame(damperFrameId);
        ro.disconnect();
        controls.dispose();
        renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
        renderer.domElement.removeEventListener("pointerup", handlePointerUp);
        for (const mesh of layerMeshes) {
          mesh.geometry.dispose();
          const mat = mesh.material as THREE.Material | THREE.Material[];
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else mat.dispose();
          scene.remove(mesh);
        }
        selectionEdges.geometry.dispose();
        (selectionEdges.material as THREE.Material).dispose();
        scene.remove(selectionEdges);
        renderer.forceContextLoss();
        renderer.dispose();
        if (renderer.domElement.parentNode === container) {
          container.removeChild(renderer.domElement);
        }
      },
    };

    return () => {
      sceneRef.current?.cleanup();
      sceneRef.current = null;
    };
  }, []);

  // Rebuild the layer meshes whenever the data shape changes. We dispose old
  // geometries and materials before adding new ones to keep GPU resources
  // bounded.
  React.useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;

    for (const mesh of state.layerMeshes) {
      mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[];
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat.dispose();
      state.scene.remove(mesh);
    }
    state.layerMeshes.length = 0;
    state.layerMaterials.length = 0;

    const numLayers = layerOutL2.length || DEFAULT_NUM_LAYERS;
    const tileHeight = Math.max(
      0.15,
      STACK_TOTAL_HEIGHT / numLayers - STACK_LAYER_GAP,
    );
    const finite = layerOutL2
      .map((d) => d.value)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    const vmin = finite.length > 0 ? Math.min(...finite) : 0;
    const vmax = finite.length > 0 ? Math.max(...finite) : 1;

    for (let i = 0; i < numLayers; i++) {
      const entry = layerOutL2[i];
      const layerIdx = entry?.layerIdx ?? i;
      const value = entry?.value ?? null;
      const isFull = false; // updated by selection/styling effect below
      const color = colorForL2(value, vmin, vmax, isFull);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.55,
        metalness: 0.1,
      });
      // Use a slightly chamfered cylinder to read as "ring of a tower". Open
      // top + bottom would look like a tube; closed cylinders read as tiles.
      const geom = new THREE.CylinderGeometry(
        STACK_RADIUS,
        STACK_RADIUS,
        tileHeight,
        24,
      );
      const mesh = new THREE.Mesh(geom, material);
      const yBase = i * (tileHeight + STACK_LAYER_GAP);
      mesh.position.set(0, yBase + tileHeight / 2, 0);
      mesh.userData.layerIdx = layerIdx;
      state.scene.add(mesh);
      state.layerMeshes.push(mesh);
      state.layerMaterials.push(material);
    }

    // Recenter camera target on the new stack height.
    const stackHeight = numLayers * tileHeight + (numLayers - 1) * STACK_LAYER_GAP;
    state.controls.target.set(0, stackHeight / 2, 0);
    state.controls.update();
    state.requestRender();
  }, [layerOutL2]);

  // Update tile colors when L2 data, full-attn set, or animation cursor
  // changes. During the forward-pass animation, layers above the cursor
  // render in a muted "pre-activation" gray; the layer at the cursor
  // gets an extra emissive bump so it reads as the "currently flowing"
  // tile climbing the tower.
  React.useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    const finite = layerOutL2
      .map((d) => d.value)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    const vmin = finite.length > 0 ? Math.min(...finite) : 0;
    const vmax = finite.length > 0 ? Math.max(...finite) : 1;
    for (let i = 0; i < state.layerMeshes.length; i++) {
      const mesh = state.layerMeshes[i]!;
      const material = state.layerMaterials[i]!;
      const layerIdx = mesh.userData.layerIdx as number;
      const value = layerOutL2[i]?.value ?? null;
      const isFull = fullAttnIndices.has(layerIdx);
      const isUnrevealed =
        revealedUpTo !== null && layerIdx > revealedUpTo;
      const isCursor = revealedUpTo !== null && layerIdx === revealedUpTo;
      if (isUnrevealed) {
        material.color.copy(UNREVEALED_LAYER_COLOR);
        material.emissive.copy(UNREVEALED_LAYER_COLOR).multiplyScalar(0.02);
      } else {
        material.color.copy(colorForL2(value, vmin, vmax, isFull));
        const emissiveScale = isCursor ? 0.45 : isFull ? 0.18 : 0.04;
        material.emissive.copy(material.color).multiplyScalar(emissiveScale);
      }
      material.needsUpdate = true;
    }
    state.requestRender();
  }, [layerOutL2, fullAttnIndices, revealedUpTo]);

  // Update selection outline when selectedLayer changes.
  React.useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    const targetMesh = state.layerMeshes.find(
      (m) => m.userData.layerIdx === selectedLayer,
    );
    if (!targetMesh) {
      state.selectionEdges.visible = false;
      state.requestRender();
      return;
    }
    // Rebuild edges for the selected mesh's geometry — cylinder geometries
    // can vary in tile height when num-layers changes.
    state.selectionEdges.geometry.dispose();
    state.selectionEdges.geometry = new THREE.EdgesGeometry(targetMesh.geometry);
    state.selectionEdges.position.copy(targetMesh.position);
    state.selectionEdges.scale.set(1.04, 1.02, 1.04);
    state.selectionEdges.visible = true;
    state.requestRender();
  }, [selectedLayer, layerOutL2]);

  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Transformer stack — 24 layers
        </div>
        <div className="text-[11px] text-muted-foreground">
          drag to rotate · scroll to zoom · click a layer
        </div>
      </div>
      <div
        ref={containerRef}
        className="relative h-[420px] w-full overflow-hidden rounded bg-gradient-to-b from-slate-900/60 to-slate-950/80"
        aria-label="Rotatable 3D view of the transformer stack"
      />
      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1">
        <span
          aria-hidden
          className="inline-block h-3 w-3 rounded-sm"
          style={{
            background:
              "linear-gradient(to right, rgb(79,70,229), rgb(147,51,234))",
          }}
        />
        Linear attention (GatedDeltaNet)
      </span>
      <span className="inline-flex items-center gap-1">
        <span
          aria-hidden
          className="inline-block h-3 w-3 rounded-sm"
          style={{
            background:
              "linear-gradient(to right, rgb(249,201,122), rgb(239,68,68))",
          }}
        />
        Full attention (softmax QKᵀ)
      </span>
      <span className="ml-auto">
        color = per-token L2 of layer output (post-residual)
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Side panel: layer stats grid + mini bar chart across the seven capture points.
// -----------------------------------------------------------------------------

function LayerSidePanel({
  layerIdx,
  isFullAttention,
  layerStats,
  hasData,
  modelName,
  numLayers,
}: {
  layerIdx: number;
  isFullAttention: boolean;
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
  hasData: boolean;
  modelName: string;
  numLayers: number;
}) {
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex items-baseline justify-between">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Layer {layerIdx} · {isFullAttention ? "Full attention" : "Linear attention"}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {modelName}
          {hasData ? ` · ${numLayers} layers` : ""}
        </div>
      </div>

      <PerPointBarChart layerStats={layerStats} />

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {CAPTURE_POINTS.map((point) => (
          <StatBox
            key={point}
            title={POINT_LABELS[point]}
            stats={layerStats[point]}
          />
        ))}
      </div>
    </div>
  );
}

function StatBox({
  title,
  stats,
}: {
  title: string;
  stats: HiddenStatePointStats | undefined;
}) {
  const l2 = perTokenL2(stats);
  return (
    <div className="rounded-md border border-border bg-background p-2 text-xs">
      <div className="mb-1 uppercase tracking-wider text-[10px] text-muted-foreground">
        {title}
      </div>
      <dl className="grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono">
        <dt className="text-muted-foreground">L2/tok</dt>
        <dd className="text-right text-foreground/90">
          {l2 !== null ? formatNumber(l2, 2) : "—"}
        </dd>
        <dt className="text-muted-foreground">|max|</dt>
        <dd className="text-right text-foreground/90">
          {stats ? formatNumber(stats.absMax, 2) : "—"}
        </dd>
        <dt className="text-muted-foreground">std</dt>
        <dd className="text-right text-foreground/90">
          {stats ? formatNumber(stats.std) : "—"}
        </dd>
        <dt className="text-muted-foreground">mean</dt>
        <dd className="text-right text-foreground/90">
          {stats ? formatNumber(stats.mean) : "—"}
        </dd>
      </dl>
    </div>
  );
}

function PerPointBarChart({
  layerStats,
}: {
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
}) {
  const values = CAPTURE_POINTS.map((point) => ({
    point,
    value: perTokenL2(layerStats[point]),
  }));
  const finite = values
    .map((v) => v.value)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const maxValue = finite.length > 0 ? Math.max(...finite) : 1;
  return (
    <div
      role="list"
      aria-label="Per-token L2 across the seven capture points in this layer"
      className="rounded-md border border-border bg-muted/30 p-3"
    >
      <div className="mb-2 text-[11px] uppercase tracking-wider text-muted-foreground">
        Signal flow through the layer (per-token L2)
      </div>
      <div className="space-y-1">
        {values.map(({ point, value }) => {
          const safeMax = maxValue > 0 ? maxValue : 1;
          const pct =
            value === null || !Number.isFinite(value)
              ? 0
              : Math.max(0, Math.min(100, (value / safeMax) * 100));
          return (
            <div
              key={point}
              role="listitem"
              className="grid grid-cols-[10rem_minmax(0,1fr)_3.5rem] items-center gap-2 text-[11px]"
            >
              <span className="truncate font-mono uppercase tracking-wider text-muted-foreground">
                {POINT_LABELS[point]}
              </span>
              <div className="relative h-3 w-full overflow-hidden rounded bg-muted">
                <div
                  className="h-full bg-primary/80"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-right font-mono text-foreground/80">
                {value === null ? "—" : formatNumber(value, 2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
