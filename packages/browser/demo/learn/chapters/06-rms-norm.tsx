import * as React from "react";

import { Button } from "../../components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
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

const DEFAULT_PROMPT = "The river flows softly through the valley.";
const DEFAULT_NUM_LAYERS = 24;
// Qwen3.5-0.8B; matches the model's hidden_size.
const HIDDEN_DIM = 1024;
const CAPTURE_POINTS = [
  "pre_attn_input",
  "post_attn_norm",
  "post_attn_residual",
  "post_mlp_norm",
] as const;

type CapturePoint = (typeof CAPTURE_POINTS)[number];

const NORM_PAIRS: Array<{
  label: string;
  inputPoint: CapturePoint;
  outputPoint: CapturePoint;
  inputLabel: string;
  outputLabel: string;
}> = [
  {
    label: "Attention RMSNorm",
    inputPoint: "pre_attn_input",
    outputPoint: "post_attn_norm",
    inputLabel: "Pre-attention norm input",
    outputLabel: "Pre-attention norm output",
  },
  {
    label: "MLP RMSNorm",
    inputPoint: "post_attn_residual",
    outputPoint: "post_mlp_norm",
    inputLabel: "Pre-MLP norm input",
    outputLabel: "Pre-MLP norm output",
  },
];

/**
 * Scaffolding metadata for chapter 6 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[5].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: "rmsnorm",
  objective:
    "Explain what RMSNorm does to a hidden vector and why pre-norm makes deep transformers trainable.",
  problem:
    "Stacked residual additions cause activation magnitudes to drift, which saturates softmaxes and breaks gradient flow.",
  minutes: 6,
  glossary: [
    {
      term: "RMSNorm",
      definition:
        "Divide x by sqrt(mean(x^2) + eps), then rescale element-wise by a learned gain g. No mean-centering, no bias.",
    },
    {
      term: "LayerNorm",
      definition:
        "The older sibling: subtracts mean(x) first, then divides by std, with both gain and bias. RMSNorm drops the centering and bias.",
    },
    {
      term: "pre-norm",
      definition:
        "Architecture choice: normalize the residual stream before each sub-block. Residual stream itself stays un-normalized.",
    },
    {
      term: "post-norm",
      definition:
        "The original 2017 transformer pattern: residual first, norm last. Easier to interpret, much harder to train deeply.",
    },
    {
      term: "L2 norm",
      definition:
        "sqrt(sum of squares) of a vector. A standard magnitude proxy; RMSNorm makes the per-token L2 land near sqrt(hidden_dim).",
    },
    {
      term: "learned gain (g)",
      definition:
        "A per-feature scale RMSNorm applies after dividing by RMS. The only learned parameter the normalizer carries.",
    },
  ],
  takeaways: [
    "RMSNorm collapses input magnitudes to roughly sqrt(hidden_dim) before the next sub-block reads them — about 32 for Qwen3.5-0.8B.",
    "Pre-norm keeps the residual stream un-normalized so gradients flow through an identity path; that's why 24-layer stacks train at all.",
    "Dropping LayerNorm's mean-centering and bias is essentially free quality-wise but slightly cheaper to compute — every modern open LLM does it.",
  ],
  exercise: {
    prompt:
      "After the auto-run, use the Layer selector to flip between layer 0 and layer 22. Compare the L2/tok numbers for 'Pre-attention norm input' vs 'Pre-attention norm output'. How does each value change with depth, and which one stays roughly constant?",
    answer:
      "The input L2/tok grows substantially with depth (residual additions accumulate), while the output L2/tok stays in the same order of magnitude (close to sqrt(1024) ≈ 32, modulated by the learned gain). That stability across depth is exactly the job of the norm.",
  },
  quiz: [
    {
      id: "q1-formula-diff",
      prompt: "What does RMSNorm drop compared to the older LayerNorm?",
      options: [
        {
          id: "a",
          label: "The learned gain g.",
        },
        {
          id: "b",
          label:
            "The mean-centering step and the additive bias — RMSNorm only divides by RMS and applies a learned gain.",
        },
        {
          id: "c",
          label: "The division step entirely; it only rescales.",
        },
      ],
      correctId: "b",
      explanation:
        "RMSNorm = LayerNorm minus subtracting the mean and minus the bias. Empirically a wash on quality, a small win on speed.",
    },
    {
      id: "q2-pre-vs-post-norm",
      prompt: "Why is pre-norm the standard choice for 20+ layer transformers?",
      options: [
        {
          id: "a",
          label:
            "Gradients flow through the un-normalized residual path as a clean identity, so deep stacks stay trainable.",
        },
        {
          id: "b",
          label: "Pre-norm uses fewer parameters than post-norm.",
        },
        {
          id: "c",
          label: "Pre-norm avoids needing a residual connection.",
        },
      ],
      correctId: "a",
      explanation:
        "Pre-norm keeps the residual highway un-normalized — the norm only touches the input to each sub-block. That preserves an unobstructed gradient path through depth.",
    },
    {
      id: "q3-output-magnitude",
      prompt: "After RMSNorm and the learned gain, roughly what magnitude does a token's hidden vector L2 land near?",
      options: [
        {
          id: "a",
          label: "Roughly 1 — RMSNorm normalises every vector to unit length.",
        },
        {
          id: "b",
          label:
            "Roughly sqrt(hidden_dim), scaled by the learned gain — about 32 for a 1024-dim hidden state.",
        },
        {
          id: "c",
          label: "Whatever the input magnitude was, unchanged.",
        },
      ],
      correctId: "b",
      explanation:
        "RMSNorm makes mean(x^2) ≈ 1, so the L2 of x is ≈ sqrt(hidden_dim). The learned gain modulates that per feature but never moves it by orders of magnitude.",
    },
  ],
};

export function RmsNormChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
      <h1>RMSNorm: keeping activations in check</h1>
      <p>
        A transformer is a tall stack — Qwen3.5-0.8B alone is 24 layers deep.
        Each layer reads a hidden state, runs attention and an MLP, then writes
        the result back into the same residual stream. Without intervention, the
        magnitude of those activations would drift: residual sums grow, dot
        products grow with them, softmaxes saturate, and gradients explode or
        vanish. <strong>Normalization</strong> is the tool we use to keep the
        activations bounded as they flow through depth.
      </p>

      <h2>The RMSNorm formula</h2>
      <p>
        RMSNorm divides each token's hidden vector by its root-mean-square,
        then rescales it with a learned per-feature gain <code>g</code>:
      </p>
      <pre>
        <code>{`RMS(x) = sqrt( mean(x²) + ε )
y_i   = (x_i / RMS(x)) * g_i`}</code>
      </pre>
      <p>
        Compared to the older <strong>LayerNorm</strong>, RMSNorm drops the
        mean-centering step (no subtracting <code>mean(x)</code> first) and
        drops the additive bias <code>b</code>. The result is a slightly faster
        op with fewer parameters that, empirically, works just as well. Every
        modern open LLM you'll meet — Llama, Mistral, Qwen, DeepSeek — uses
        RMSNorm.
      </p>

      <h2>Pre-norm vs. post-norm</h2>
      <p>
        Qwen3.5 (like nearly every contemporary transformer) is{" "}
        <strong>pre-norm</strong>: the norm is applied <em>before</em> each
        sub-block, and its output is what's fed into attention or the MLP. The
        residual stream itself is never normalized; it's a clean "highway" of
        un-normalized values that each block reads from and writes into.
      </p>
      <ul>
        <li>
          <strong>Post-norm</strong> (the original 2017 transformer): residual
          first, norm last. The thing flowing down the highway has bounded
          magnitude. Easier to interpret, but harder to train deeply — gradients
          have to fight through the norm at every layer.
        </li>
        <li>
          <strong>Pre-norm</strong> (Qwen3.5's choice): norm first, residual
          last. The residual stream's magnitude grows with depth, but
          gradients flow through an unobstructed identity path. Much more
          stable for 20+ layer stacks.
        </li>
      </ul>

      <h2>What the widget shows</h2>
      <p>
        Pressing <em>Run</em> captures hidden-state statistics at four points
        inside a single layer: the inputs and outputs of the two RMSNorms
        (one before attention, one before the MLP). The most readable summary
        is the <strong>L2 norm</strong> of the hidden vector. Watch it collapse
        from "whatever the residual stream happens to be" down to roughly{" "}
        <code>sqrt(hidden_dim)</code> ≈ {Math.sqrt(HIDDEN_DIM).toFixed(0)}
        {" "}after the norm — scaled by the learned gain, so layer-to-layer the
        post-norm L2 isn't identical, but it's always in the same order of
        magnitude.
      </p>
      <p>
        We show <strong>per-token L2 norm</strong> — the backend captures stats
        over the full prompt × hidden dimension, but we divide by{" "}
        <code>sqrt(seq_len)</code> so the values are comparable across prompts
        of different lengths.
      </p>
      <p>
        Try a few different prompts and layers. The norm <em>input</em>
        magnitudes vary wildly with depth and content; the norm{" "}
        <em>outputs</em> are tame and bounded. That is the entire job of
        RMSNorm — and the reason a 24-layer model can train at all.
      </p>
      </Prose>
    </ChapterFrame>
  );
}

export type RmsNormDemoProps = {
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

/**
 * Convert a raw `HiddenStatePointStats` into a per-token L2 norm.
 *
 * The backend reports `l2Norm = sqrt(sumsq)` over the flattened
 * `[seq, hidden_dim]` tensor (see `crates/mlx-core/src/inspector.rs`
 * `capture_hidden_state`). For a 9-token prompt with hidden_dim=1024 that
 * gives sqrt(9 * 1024) ≈ 96, not sqrt(1024) ≈ 32. Dividing by sqrt(seq_len)
 * yields the per-token L2 the prose and chart axes assume.
 */
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

export function RmsNormDemo({ workerRef, abortRef }: RmsNormDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  const [selectedLayer, setSelectedLayer] = React.useState(0);

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);

  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
    },
    [],
  );

  async function handleRun() {
    if (prompt.trim().length === 0) {
      setStatus({ kind: "empty-prompt" });
      return;
    }

    const myGen = ++runGenRef.current;
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      console.error(
        "[rms-norm-demo] worker is unavailable — chapter view should have been gated on modelReady",
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
      console.log("[rms-norm-demo] runForInspector ok", {
        promptLength: prompt.length,
        layers: result.hiddenStates?.[0]?.layers.length ?? 0,
      });
      setRun(result);
      setStatus({ kind: "ok" });
      const maxLayer = (result.modelMeta?.numLayers ?? DEFAULT_NUM_LAYERS) - 1;
      if (selectedLayer > maxLayer) {
        setSelectedLayer(0);
      }
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info(
          "[rms-norm-demo] runForInspector aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) {
          setStatus({ kind: "aborted" });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error("[rms-norm-demo] runForInspector failed", err);
        setStatus({ kind: "error", error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener("abort", onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      if (runGenRef.current === myGen) setRunning(false);
    }
  }

  // Auto-fire one Run on mount — model is guaranteed ready by the gate
  // around chapter rendering.
  const didAutoRunRef = React.useRef(false);
  React.useEffect(() => {
    if (didAutoRunRef.current) return;
    didAutoRunRef.current = true;
    void handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hiddenSteps = run?.hiddenStates;
  const numLayers = run?.modelMeta?.numLayers ?? DEFAULT_NUM_LAYERS;
  const availableLayers = React.useMemo(() => {
    if (!hiddenSteps || hiddenSteps.length === 0) {
      return Array.from({ length: numLayers }, (_, i) => i);
    }
    const step = hiddenSteps[0];
    if (!step) return Array.from({ length: numLayers }, (_, i) => i);
    return step.layers.map((l) => l.layerIdx).sort((a, b) => a - b);
  }, [hiddenSteps, numLayers]);

  // Clamp `selectedLayer` when the available-layers set changes (e.g. a new
  // run captured a different subset, or the backend swapped models entirely).
  // Without this, the <Select> can display a value that no longer exists in
  // `availableLayers`.
  React.useEffect(() => {
    if (availableLayers.length === 0) return;
    if (!availableLayers.includes(selectedLayer)) {
      setSelectedLayer(availableLayers[0]!);
    }
  }, [availableLayers, selectedLayer]);

  const layerStats = React.useMemo(
    () => getLayerStats(hiddenSteps, selectedLayer),
    [hiddenSteps, selectedLayer],
  );

  const aggregate = React.useMemo(() => {
    if (!hiddenSteps || hiddenSteps.length === 0) return null;
    const step = hiddenSteps[0];
    if (!step) return null;
    let preIn = 0;
    let preOut = 0;
    let mlpIn = 0;
    let mlpOut = 0;
    let count = 0;
    for (const layer of step.layers) {
      const map: Partial<Record<string, HiddenStatePointStats>> = {};
      for (const s of layer.stats) map[s.point] = s;
      const a = perTokenL2(map["pre_attn_input"]);
      const b = perTokenL2(map["post_attn_norm"]);
      const c = perTokenL2(map["post_attn_residual"]);
      const d = perTokenL2(map["post_mlp_norm"]);
      if (a !== null && b !== null && c !== null && d !== null) {
        preIn += a;
        preOut += b;
        mlpIn += c;
        mlpOut += d;
        count += 1;
      }
    }
    if (count === 0) return null;
    return {
      count,
      preIn: preIn / count,
      preOut: preOut / count,
      mlpIn: mlpIn / count,
      mlpOut: mlpOut / count,
    };
  }, [hiddenSteps]);

  const hasData = Boolean(hiddenSteps && hiddenSteps.length > 0);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="rms-norm-demo-input"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Prompt
        </label>
        <Textarea
          id="rms-norm-demo-input"
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
            Captures pre/post-norm activation stats for a single forward pass.
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label
          htmlFor="rms-norm-demo-layer"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Layer
        </label>
        <Select
          value={`${selectedLayer}`}
          onValueChange={(v) => setSelectedLayer(Number(v))}
        >
          <SelectTrigger
            id="rms-norm-demo-layer"
            size="sm"
            className="min-w-[8rem]"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {availableLayers.map((idx) => (
              <SelectItem key={idx} value={`${idx}`}>
                {`Layer ${idx}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          {run?.modelMeta?.name ?? "—"}
          {hasData ? ` · ${numLayers} layers` : ""}
        </span>
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
          Enter a prompt to see normalization in action.
        </div>
      ) : null}

      {hasData ? (
        <div className="space-y-3">
          <NormStatsGrid layerStats={layerStats} />
          <L2BeforeAfterChart layerStats={layerStats} />
          {aggregate ? (
            <AggregateFooter
              numLayers={aggregate.count}
              preIn={aggregate.preIn}
              preOut={aggregate.preOut}
              mlpIn={aggregate.mlpIn}
              mlpOut={aggregate.mlpOut}
            />
          ) : null}
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to capture pre- and post-RMSNorm
          activation stats for a single forward pass.
        </div>
      )}

      <DemoCallout
        items={[
          "Look at how the input distribution gets squeezed: the output L2 norm lands near sqrt(hidden_dim) regardless of input scale.",
          "Compare layer 0 vs a deep layer — late layers have larger residual streams, so the divisor RMS is bigger.",
          "RMSNorm has only a learned gain (no shift) — that's the simplification over LayerNorm.",
        ]}
      />
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
    <div className="rounded-md border border-border bg-background p-3 text-xs">
      <div className="mb-2 uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono">
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

function NormStatsGrid({
  layerStats,
}: {
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
}) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {NORM_PAIRS.flatMap((pair) => [
        <StatBox
          key={`${pair.inputPoint}-box`}
          title={pair.inputLabel}
          stats={layerStats[pair.inputPoint]}
        />,
        <StatBox
          key={`${pair.outputPoint}-box`}
          title={pair.outputLabel}
          stats={layerStats[pair.outputPoint]}
        />,
      ])}
    </div>
  );
}

function L2BeforeAfterChart({
  layerStats,
}: {
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
}) {
  const bars = NORM_PAIRS.flatMap((pair) => {
    const before = perTokenL2(layerStats[pair.inputPoint]);
    const after = perTokenL2(layerStats[pair.outputPoint]);
    return [
      {
        group: pair.label,
        label: "before",
        value: before,
      },
      {
        group: pair.label,
        label: "after",
        value: after,
      },
    ];
  });
  const finiteValues = bars
    .map((b) => b.value)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const maxValue = finiteValues.length > 0 ? Math.max(...finiteValues) : 1;
  return (
    <div
      role="list"
      aria-label="Per-token L2 norm before and after each RMSNorm"
      className="space-y-3 rounded-md border border-border bg-background p-3"
    >
      {NORM_PAIRS.map((pair) => {
        const before = perTokenL2(layerStats[pair.inputPoint]);
        const after = perTokenL2(layerStats[pair.outputPoint]);
        return (
          <div key={pair.label} role="listitem" className="space-y-1">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              {pair.label}
            </div>
            <BarRow label="before" value={before} max={maxValue} accent={false} />
            <BarRow label="after" value={after} max={maxValue} accent />
          </div>
        );
      })}
      <div className="pt-1 text-[11px] text-muted-foreground">
        Values are per-token L2 (raw L2 ÷ sqrt(seq_len)). Reference:
        sqrt(hidden_dim) ≈ {Math.sqrt(HIDDEN_DIM).toFixed(1)} for Qwen3.5-0.8B's
        1024-dim hidden state.
      </div>
    </div>
  );
}

function BarRow({
  label,
  value,
  max,
  accent,
}: {
  label: string;
  value: number | null;
  max: number;
  accent: boolean;
}) {
  const safeMax = max > 0 ? max : 1;
  const pct =
    value === null || !Number.isFinite(value)
      ? 0
      : Math.max(0, Math.min(100, (value / safeMax) * 100));
  return (
    <div className="grid grid-cols-[4rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-[12px]">
      <span className="font-mono uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <div className="relative h-4 w-full overflow-hidden rounded bg-muted">
        <div
          className={["h-full", accent ? "bg-primary" : "bg-foreground/40"].join(
            " ",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-right font-mono text-foreground/80">
        {value === null ? "—" : formatNumber(value, 2)}
      </span>
    </div>
  );
}

function AggregateFooter({
  numLayers,
  preIn,
  preOut,
  mlpIn,
  mlpOut,
}: {
  numLayers: number;
  preIn: number;
  preOut: number;
  mlpIn: number;
  mlpOut: number;
}) {
  return (
    <div className="rounded-md bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
      <div>
        Across {numLayers} layers · attn-norm L2/tok: {formatNumber(preIn, 1)} →{" "}
        {formatNumber(preOut, 2)} · mlp-norm L2/tok: {formatNumber(mlpIn, 1)} →{" "}
        {formatNumber(mlpOut, 2)}.
      </div>
    </div>
  );
}
