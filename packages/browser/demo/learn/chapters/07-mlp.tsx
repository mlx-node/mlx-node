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
import { Prose } from "../Prose";

const DEFAULT_PROMPT = "Once upon a time, in a forest far away,";
const DEFAULT_NUM_LAYERS = 24;
// Qwen3.5-0.8B; matches the model's hidden_size.
const HIDDEN_DIM = 1024;
const CAPTURE_POINTS = [
  "post_attn_residual",
  "post_mlp_norm",
  "mlp_output",
  "post_mlp_residual",
] as const;

type CapturePoint = (typeof CAPTURE_POINTS)[number];

const POINT_META: Record<
  CapturePoint,
  { title: string; description: string }
> = {
  post_attn_residual: {
    title: "Input to MLP path",
    description: "Residual stream entering this layer's MLP sub-block.",
  },
  post_mlp_norm: {
    title: "After pre-MLP RMSNorm",
    description: "Norm collapses the magnitude to ~sqrt(hidden_dim).",
  },
  mlp_output: {
    title: "MLP output (correction)",
    description: "Raw MLP output, before the residual add — typically small.",
  },
  post_mlp_residual: {
    title: "Layer output",
    description: "Input + correction. This is what the next layer reads.",
  },
};

export function MlpChapterBody() {
  return (
    <Prose>
      <h1>The MLP block: per-token feed-forward</h1>
      <p>
        Attention is the part of a transformer that mixes information{" "}
        <em>across tokens</em>: every position can read from every other
        position via softmax(QKᵀ)·V. The MLP block is the opposite — it touches
        every token in <em>isolation</em>, running the same little neural
        network on each position's hidden vector and writing the result back to
        that same position. Across-token mixing happens in attention; per-token
        computation happens in the MLP. A transformer layer alternates between
        the two.
      </p>

      <h2>Pre-MLP norm, then a gated MLP, then a residual add</h2>
      <p>
        Like its attention sibling, the MLP sub-block in Qwen3.5 is wrapped in
        a pre-norm + residual pattern:
      </p>
      <pre>
        <code>{`x_out = x_in + mlp(rmsnorm(x_in))`}</code>
      </pre>
      <p>
        The <strong>residual connection</strong> is the <code>x_in +</code>{" "}
        part. It exists for two reasons. First, gradients: backprop can flow
        straight through the identity path, so even a 24-layer stack stays
        trainable. Second, semantics: think of the residual stream as a
        <em> highway</em> that runs the full depth of the model. Each block
        reads from the highway, computes a small contribution, and adds it
        back. The model's prediction is the accumulation of every block's
        contribution — not the output of the last block alone.
      </p>

      <h2>What "gated MLP" actually means</h2>
      <p>
        Qwen3.5 (like Llama, Mistral, and most modern open LLMs) uses the
        <strong> SwiGLU-style gated MLP</strong>. It has three linear
        projections per layer:
      </p>
      <ul>
        <li>
          <code>gate_proj</code>: hidden_dim → intermediate_dim. Its output is
          run through a <strong>SiLU</strong> activation (also called Swish):
          <code> silu(z) = z · sigmoid(z)</code>. SiLU is smooth and nearly
          linear for large positive <code>z</code>, but suppresses negative
          values — a soft gate.
        </li>
        <li>
          <code>up_proj</code>: hidden_dim → intermediate_dim. This is the
          "value" path — the actual features being passed through.
        </li>
        <li>
          <code>down_proj</code>: intermediate_dim → hidden_dim. Projects the
          element-wise product back to the residual stream's width.
        </li>
      </ul>
      <pre>
        <code>{`MLP(x) = down_proj( silu(gate_proj(x))  *  up_proj(x) )`}</code>
      </pre>
      <p>
        Read this expression carefully. The <code>*</code> is{" "}
        <em>element-wise</em>, not matrix multiplication: each of the
        intermediate_dim features in <code>silu(gate_proj(x))</code> multiplies
        the corresponding feature in <code>up_proj(x)</code>. The
        gate-projection decides <em>how much</em> of each feature passes
        through; the up-projection supplies the value. The two are entangled
        per feature, then collapsed back to hidden_dim by <code>down_proj</code>.
      </p>
      <p>
        The intermediate dimension is where the model has its "scratch
        space" — usually around 4× the hidden dim (Qwen3.5-0.8B uses 3072 for a
        1024-dim hidden state). Most of the model's parameters live in these
        three matrices.
      </p>

      <h2>The MLP contributes a small correction</h2>
      <p>
        Here is the surprising part: the MLP's raw output is{" "}
        <em>much smaller</em> than the residual stream it writes into. The
        residual stream has been accumulating contributions for many layers,
        so its per-token L2 grows. Each individual MLP's output is a small
        delta on top — a correction, not a replacement. Watch the chart
        below: <strong>MLP output</strong> per-token L2 is a fraction of the
        layer's full output L2. The bulk of the magnitude in the residual
        stream came from the input, not from this layer's MLP.
      </p>
      <p>
        This is the "highway" intuition made quantitative. The all-layers
        strip shows that the size of the per-layer MLP contribution varies
        across depth: some layers contribute more than others, but no single
        layer dominates. Predictions emerge from the sum of many small writes
        — which is exactly the design choice that makes deep transformers
        learnable.
      </p>
    </Prose>
  );
}

export type MlpDemoProps = {
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
 * `capture_hidden_state`). For an 11-token prompt with hidden_dim=1024 that
 * gives sqrt(11 * 1024) ≈ 106, not sqrt(1024) ≈ 32. Dividing by sqrt(seq_len)
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

export function MlpDemo({ workerRef, abortRef }: MlpDemoProps) {
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
        "[mlp-demo] worker is unavailable — chapter view should have been gated on modelReady",
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
      console.log("[mlp-demo] runForInspector ok", {
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
          "[mlp-demo] runForInspector aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) {
          setStatus({ kind: "aborted" });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error("[mlp-demo] runForInspector failed", err);
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

  const allLayersMlpL2 = React.useMemo(() => {
    if (!hiddenSteps || hiddenSteps.length === 0) return [];
    const step = hiddenSteps[0];
    if (!step) return [];
    return step.layers
      .slice()
      .sort((a, b) => a.layerIdx - b.layerIdx)
      .map((layer) => {
        const stat = layer.stats.find((s) => s.point === "mlp_output");
        return { layerIdx: layer.layerIdx, value: perTokenL2(stat) };
      });
  }, [hiddenSteps]);

  const hasData = Boolean(hiddenSteps && hiddenSteps.length > 0);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="mlp-demo-input"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Prompt
        </label>
        <Textarea
          id="mlp-demo-input"
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
            Captures MLP path activation stats for a single forward pass.
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label
          htmlFor="mlp-demo-layer"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Layer
        </label>
        <Select
          value={`${selectedLayer}`}
          onValueChange={(v) => setSelectedLayer(Number(v))}
        >
          <SelectTrigger
            id="mlp-demo-layer"
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
          Enter a prompt to see the MLP block in action.
        </div>
      ) : null}

      {hasData ? (
        <div className="space-y-3">
          <MlpStatsGrid layerStats={layerStats} />
          <ContributionVsOutputChart layerStats={layerStats} />
          <AllLayersStrip
            data={allLayersMlpL2}
            selectedLayer={selectedLayer}
            onSelect={(idx) => setSelectedLayer(idx)}
          />
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to capture MLP path activation stats for
          a single forward pass.
        </div>
      )}
    </div>
  );
}

function StatBox({
  title,
  description,
  stats,
}: {
  title: string;
  description: string;
  stats: HiddenStatePointStats | undefined;
}) {
  const l2 = perTokenL2(stats);
  return (
    <div className="rounded-md border border-border bg-background p-3 text-xs">
      <div className="mb-1 uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      <div className="mb-2 text-[11px] text-muted-foreground/80">
        {description}
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

function MlpStatsGrid({
  layerStats,
}: {
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
}) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {CAPTURE_POINTS.map((point) => (
        <StatBox
          key={point}
          title={POINT_META[point].title}
          description={POINT_META[point].description}
          stats={layerStats[point]}
        />
      ))}
    </div>
  );
}

function ContributionVsOutputChart({
  layerStats,
}: {
  layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>>;
}) {
  const mlp = perTokenL2(layerStats.mlp_output);
  const layerOut = perTokenL2(layerStats.post_mlp_residual);
  const finite = [mlp, layerOut].filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  const maxValue = finite.length > 0 ? Math.max(...finite) : 1;
  const ratio =
    mlp !== null && layerOut !== null && layerOut > 0 ? mlp / layerOut : null;
  return (
    <div
      role="list"
      aria-label="MLP contribution vs full layer output, per-token L2"
      className="space-y-3 rounded-md border border-border bg-background p-3"
    >
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        Magnitude: MLP contribution vs layer output
      </div>
      <div role="listitem" className="space-y-1">
        <BarRow
          label="MLP only"
          value={mlp}
          max={maxValue}
          accent={false}
        />
        <BarRow label="Layer out" value={layerOut} max={maxValue} accent />
      </div>
      <div className="pt-1 text-[11px] text-muted-foreground">
        The MLP adds a small correction; the residual stream carries the bulk
        of the signal.
        {ratio !== null
          ? ` MLP / layer-output ratio: ${formatNumber(ratio * 100, 1)}%.`
          : ""}
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
    <div className="grid grid-cols-[5rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-[12px]">
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

function AllLayersStrip({
  data,
  selectedLayer,
  onSelect,
}: {
  data: Array<{ layerIdx: number; value: number | null }>;
  selectedLayer: number;
  onSelect: (idx: number) => void;
}) {
  const finite = data
    .map((d) => d.value)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const maxValue = finite.length > 0 ? Math.max(...finite) : 1;
  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        MLP contribution across all layers (per-token L2 of mlp_output)
      </div>
      <div
        role="list"
        aria-label="Per-layer MLP output magnitude"
        className="flex h-20 items-end gap-[2px]"
      >
        {data.map(({ layerIdx, value }) => {
          const safeMax = maxValue > 0 ? maxValue : 1;
          const pct =
            value === null || !Number.isFinite(value)
              ? 0
              : Math.max(2, Math.min(100, (value / safeMax) * 100));
          const isSelected = layerIdx === selectedLayer;
          return (
            <button
              key={layerIdx}
              type="button"
              role="listitem"
              onClick={() => onSelect(layerIdx)}
              title={`Layer ${layerIdx}: ${
                value === null ? "—" : formatNumber(value, 2)
              }`}
              aria-label={`Layer ${layerIdx} MLP L2 ${
                value === null ? "unknown" : formatNumber(value, 2)
              }`}
              aria-pressed={isSelected}
              className={[
                "group relative flex h-full min-w-[10px] flex-1 items-end rounded-sm transition-colors",
                "hover:bg-muted",
              ].join(" ")}
            >
              <span
                className={[
                  "block w-full rounded-sm",
                  isSelected ? "bg-primary" : "bg-foreground/40",
                  "group-hover:bg-primary/80",
                ].join(" ")}
                style={{ height: `${pct}%` }}
              />
            </button>
          );
        })}
      </div>
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Layer 0</span>
        <span>
          Selected: layer {selectedLayer} — click any bar to focus.
        </span>
        <span>Layer {data.length > 0 ? data[data.length - 1]!.layerIdx : 0}</span>
      </div>
    </div>
  );
}
