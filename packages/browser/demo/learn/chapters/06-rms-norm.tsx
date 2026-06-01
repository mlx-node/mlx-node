import * as React from 'react';

import type { AttentionRun, HiddenStatePointStats, HiddenStateStep } from '../../../src/inspector-types';
import { Textarea } from '../../components/ui/textarea';
import { runForInspector } from '../../lib/inspector-client';
import { DemoCallout } from '../inspector/DemoCallout';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { RunButton } from '../scaffolding/RunButton';
import { useRunFlash } from '../scaffolding/useRunFlash';
import { LearnedGainInspector } from '../widgets/LearnedGainInspector';

const DEFAULT_PROMPT = 'The river flows softly through the valley.';
const DEFAULT_NUM_LAYERS = 24;
// Qwen3.5-0.8B; matches the model's hidden_size.
const HIDDEN_DIM = 1024;
const CAPTURE_POINTS = ['pre_attn_input', 'post_attn_norm', 'post_attn_residual', 'post_mlp_norm'] as const;

// --------------------------------------------------------------------------
// Pure-JS helpers for the scale-invariance playground. These live at module
// scope (not inside a component) so they can be referenced from JSDoc and
// stay stable across re-renders.
// --------------------------------------------------------------------------

/** Seeded Gaussian via a small LCG + Box-Muller. Reproducible across renders. */
function seededGaussianVector(dim: number, seed: number): Float32Array {
  const v = new Float32Array(dim);
  let s = seed >>> 0 || 1;
  function nextU(): number {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s & 0x7fffffff) + 1) / 0x80000001;
  }
  for (let i = 0; i < dim; i++) {
    const u1 = Math.max(nextU(), 1e-9);
    const u2 = nextU();
    v[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return v;
}

/** y = x / sqrt(mean(x^2) + eps). No learned gain — we use g = 1 in the playground. */
function applyRmsNorm(x: Float32Array, eps = 1e-6): Float32Array {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i]! * x[i]!;
  const rms = Math.sqrt(ss / x.length + eps);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i]! / rms;
  return y;
}

function vecL2(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i]! * x[i]!;
  return Math.sqrt(s);
}

/** Bucket values into `bins` equal-width buckets over `[min, max]`. */
function histogram(x: Float32Array, bins: number, min: number, max: number): number[] {
  const out = Array.from<number>({ length: bins }).fill(0);
  const span = max - min;
  if (span <= 0) return out;
  for (let i = 0; i < x.length; i++) {
    const t = (x[i]! - min) / span;
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(t * bins)));
    out[idx]! += 1;
  }
  return out;
}

type CapturePoint = (typeof CAPTURE_POINTS)[number];

const NORM_PAIRS: Array<{
  label: string;
  inputPoint: CapturePoint;
  outputPoint: CapturePoint;
  inputLabel: string;
  outputLabel: string;
}> = [
  {
    label: 'Attention RMSNorm',
    inputPoint: 'pre_attn_input',
    outputPoint: 'post_attn_norm',
    inputLabel: 'Pre-attention norm input',
    outputLabel: 'Pre-attention norm output',
  },
  {
    label: 'MLP RMSNorm',
    inputPoint: 'post_attn_residual',
    outputPoint: 'post_mlp_norm',
    inputLabel: 'Pre-MLP norm input',
    outputLabel: 'Pre-MLP norm output',
  },
];

/**
 * Scaffolding metadata for chapter 6 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[5].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'rmsnorm',
  objective: 'Explain what RMSNorm does to a hidden vector and why pre-norm makes deep transformers trainable.',
  problem:
    "Each layer adds attention and MLP outputs back into the same residual stream — without normalization the magnitudes drift across 24 layers, softmaxes saturate, and training breaks. RMSNorm's whole job is to put the brakes on.",
  minutes: 7,
  glossary: [
    {
      term: 'RMSNorm',
      definition:
        'Divide x by sqrt(mean(x^2) + eps), then rescale element-wise by a learned gain g. No mean-centering, no bias.',
    },
    {
      term: 'LayerNorm',
      definition:
        'The older sibling: subtracts mean(x) first, then divides by std, with both gain and bias. RMSNorm drops the centering and bias.',
    },
    {
      term: 'pre-norm',
      definition:
        'Architecture choice: normalize the residual stream before each sub-block. Residual stream itself stays un-normalized.',
    },
    {
      term: 'post-norm',
      definition:
        'The original 2017 transformer pattern: residual first, norm last. Easier to interpret, much harder to train deeply.',
    },
    {
      term: 'L2 norm',
      definition:
        'sqrt(sum of squares) of a vector. A standard magnitude proxy; RMSNorm makes the per-token L2 land near sqrt(hidden_dim).',
    },
    {
      term: 'learned gain (g)',
      definition:
        'A per-feature scale RMSNorm applies after dividing by RMS. The only learned parameter the normalizer carries.',
    },
    {
      term: 'gradient path / identity path',
      definition:
        "The route the training signal takes backward through the network. The residual highway is an identity path because it passes that signal through unchanged (multiplies by 1), so it doesn't shrink toward zero (vanish) across many layers.",
    },
  ],
  takeaways: [
    'RMSNorm collapses input magnitudes to roughly sqrt(hidden_dim) before the next sub-block reads them — about 32 for Qwen3.5-0.8B.',
    "Pre-norm keeps the residual stream un-normalized so gradients flow through an identity path; that's why 24-layer stacks train at all.",
    "Dropping LayerNorm's mean-centering and bias is essentially free quality-wise but slightly cheaper to compute — every modern open LLM does it.",
  ],
  exercise: {
    prompt:
      "After the auto-run, use the Layer selector to flip between layer 0 and layer 22. Compare the L2/tok numbers for 'Pre-attention norm input' vs 'Pre-attention norm output'. How does each value change with depth, and which one stays roughly constant?",
    answer:
      'The input L2/tok grows substantially with depth (residual additions accumulate), while the output L2/tok stays in the same order of magnitude (close to sqrt(1024) ≈ 32, modulated by the learned gain). That stability across depth is exactly the job of the norm.',
  },
  quiz: [
    {
      id: 'q1-formula-diff',
      prompt: 'What does RMSNorm drop compared to the older LayerNorm?',
      options: [
        {
          id: 'a',
          label: 'The learned gain g.',
        },
        {
          id: 'b',
          label:
            'The mean-centering step and the additive bias — RMSNorm only divides by RMS and applies a learned gain.',
        },
        {
          id: 'c',
          label: 'The division step entirely; it only rescales.',
        },
      ],
      correctId: 'b',
      explanation:
        'RMSNorm = LayerNorm minus subtracting the mean and minus the bias. Empirically a wash on quality, a small win on speed.',
    },
    {
      id: 'q2-pre-vs-post-norm',
      prompt: 'Why is pre-norm the standard choice for 20+ layer transformers?',
      options: [
        {
          id: 'a',
          label:
            'Gradients flow through the un-normalized residual path as a clean identity, so deep stacks stay trainable.',
        },
        {
          id: 'b',
          label: 'Pre-norm uses fewer parameters than post-norm.',
        },
        {
          id: 'c',
          label: 'Pre-norm avoids needing a residual connection.',
        },
      ],
      correctId: 'a',
      explanation:
        'Pre-norm keeps the residual highway un-normalized — the norm only touches the input to each sub-block. That preserves an unobstructed gradient path through depth.',
    },
    {
      id: 'q3-output-magnitude',
      prompt: "After RMSNorm and the learned gain, roughly what magnitude does a token's hidden vector L2 land near?",
      options: [
        {
          id: 'a',
          label: 'Roughly 1 — RMSNorm normalises every vector to unit length.',
        },
        {
          id: 'b',
          label: 'Roughly sqrt(hidden_dim), scaled by the learned gain — about 32 for a 1024-dim hidden state.',
        },
        {
          id: 'c',
          label: 'Whatever the input magnitude was, unchanged.',
        },
      ],
      correctId: 'b',
      explanation:
        'RMSNorm makes mean(x^2) ≈ 1, so the L2 of x is ≈ sqrt(hidden_dim). The learned gain modulates that per feature but never moves it by orders of magnitude.',
    },
  ],
};

export function RmsNormChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>RMSNorm: keeping activations in check</h1>
        <p>
          <strong>One sentence:</strong> each of Qwen3.5's 24 layers adds attention and MLP outputs back into the same
          residual stream — without something holding magnitudes in check, the hidden vector grows until softmaxes
          saturate and gradient updates stop being meaningful. RMSNorm is the cheap, almost-stateless trick that puts
          the brakes on. The chart on the right shows a real prompt: the residual still climbs about 15× over depth
          (visible bottom line), but the input to each sub-block is held flat near √hidden_dim by RMSNorm (pink line).
        </p>

        <h2>The formula, in two lines</h2>
        <p>
          RMSNorm divides each token's hidden vector by its root-mean-square, then rescales it with a learned
          per-feature gain <code>g</code>:
        </p>
        <MathDisplay
          latex={String.raw`\begin{aligned} \text{RMS}(x) &= \sqrt{\text{mean}(x^2) + \varepsilon} \\ y_i &= \frac{x_i}{\text{RMS}(x)} \cdot g_i \end{aligned}`}
        />
        <p>
          The learned gain <code>g</code> is the only parameter the normalizer carries — one scalar per feature, length
          equal to <code>hidden_dim</code>. Divide by RMS first to make the vector's average squared magnitude land at
          1; with mean(x²) ≈ 1, the squared entries sum to ≈ hidden_dim, so the vector's length is{' '}
          <code>L2 = √(Σx²) ≈ √hidden_dim = √1024 = 32</code> — that is where the &ldquo;about 32&rdquo; comes from.
          Then let <code>g</code> reweight each feature however the model wants. Initially <code>g = 1</code> for every
          feature, so before training RMSNorm just standardises magnitudes; during training the model learns which
          features matter more than others and bakes that into <code>g</code>.
        </p>

        <LearnedGainInspector />

        <p>Side-by-side with the older LayerNorm, the simplification is obvious:</p>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-[12px] leading-relaxed">
          <code>{`# RMSNorm — what every modern LLM uses
y = x / sqrt(mean(x²) + ε) * g
                              ↑
                       single learned vector

# LayerNorm — what the 2017 transformer used
y = (x - mean(x)) / sqrt(var(x) + ε) * g + b
     ↑—————————                  ↑
   subtract mean first        plus learned bias`}</code>
        </pre>
        <p>
          Dropping the mean-subtraction and bias is the entire difference. Llama, Mistral, Qwen, DeepSeek — every open
          LLM you'll meet uses RMSNorm; the simpler formula is empirically a wash on quality and modestly faster to run.
        </p>

        <h2>Pre-norm vs. post-norm</h2>
        <p>
          The diagram below shows where the norm lives inside a layer. Modern stacks all use <strong>pre-norm</strong>:
          the residual highway stays un-normalized, so gradients flow through it as a clean identity path. Post-norm
          (the 2017 original) places the norm <em>on</em> the residual highway — fine for 6 layers, painful at 24.
        </p>
        <NormFlowDiagram />

        <h2>What the demo shows you</h2>
        <p>
          The <strong>L2/tok across layers</strong> chart (top of the demo panel) is the punchline of this chapter — one
          curve drifting up, one curve held flat. The per-layer breakdown below it lets you spot-check a single layer's
          stats. And the <em>scale-invariance playground</em> at the bottom is a synthetic 256-dim vector you can
          stretch by 0.1×–10× to convince yourself the output magnitude really doesn't depend on the input scale.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

export type RmsNormDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus = { kind: 'ok' } | { kind: 'error'; error: string } | { kind: 'aborted' } | { kind: 'empty-prompt' };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
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
  if (!Number.isFinite(value)) return '—';
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
  // The prompt that produced the currently displayed stats. Lets us disable
  // Run when the textarea hasn't changed since the last forward pass —
  // otherwise users click Run, see identical numbers, and assume nothing
  // happened (decoding is deterministic at temp=0).
  const [lastRunPrompt, setLastRunPrompt] = React.useState<string | null>(null);
  const resultFlashRef = useRunFlash(running);

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
      setStatus({ kind: 'empty-prompt' });
      return;
    }

    const myGen = ++runGenRef.current;
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      console.error('[rms-norm-demo] worker is unavailable — chapter view should have been gated on modelReady');
      setStatus({ kind: 'error', error: 'Worker is unavailable. Reload the page.' });
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
      appSignal.addEventListener('abort', onAppAbort, { once: true });
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
      console.log('[rms-norm-demo] runForInspector ok', {
        promptLength: prompt.length,
        layers: result.hiddenStates?.[0]?.layers.length ?? 0,
      });
      setRun(result);
      setStatus({ kind: 'ok' });
      setLastRunPrompt(prompt);
      const maxLayer = (result.modelMeta?.numLayers ?? DEFAULT_NUM_LAYERS) - 1;
      if (selectedLayer > maxLayer) {
        setSelectedLayer(0);
      }
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info('[rms-norm-demo] runForInspector aborted (worker terminated or superseded)');
        if (appSignal?.aborted) {
          setStatus({ kind: 'aborted' });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error('[rms-norm-demo] runForInspector failed', err);
        setStatus({ kind: 'error', error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
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
  // Without this, the slider can land on a value that no longer exists in
  // `availableLayers`.
  React.useEffect(() => {
    if (availableLayers.length === 0) return;
    if (!availableLayers.includes(selectedLayer)) {
      setSelectedLayer(availableLayers[0]!);
    }
  }, [availableLayers, selectedLayer]);

  const layerStats = React.useMemo(() => getLayerStats(hiddenSteps, selectedLayer), [hiddenSteps, selectedLayer]);

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
      const a = perTokenL2(map['pre_attn_input']);
      const b = perTokenL2(map['post_attn_norm']);
      const c = perTokenL2(map['post_attn_residual']);
      const d = perTokenL2(map['post_mlp_norm']);
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
        <label htmlFor="rms-norm-demo-input" className="text-xs uppercase tracking-wider text-muted-foreground">
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
          <RunButton
            onClick={handleRun}
            running={running}
            disabled={lastRunPrompt !== null && lastRunPrompt === prompt}
          />
          <span className="text-xs text-muted-foreground">
            {lastRunPrompt !== null && lastRunPrompt === prompt && !running
              ? 'Stats are current. Edit the prompt above to capture a new forward pass.'
              : 'Captures pre/post-norm activation stats for a single forward pass.'}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="rms-norm-demo-layer" className="text-xs uppercase tracking-wider text-muted-foreground">
          Layer
        </label>
        <input
          id="rms-norm-demo-layer"
          type="range"
          min={0}
          max={Math.max(availableLayers.length - 1, 0)}
          step={1}
          value={Math.max(availableLayers.indexOf(selectedLayer), 0)}
          disabled={availableLayers.length === 0}
          onChange={(e) => {
            const idx = Number.parseInt(e.target.value, 10);
            const next = availableLayers[idx];
            if (next !== undefined) setSelectedLayer(next);
          }}
          aria-valuetext={`Layer ${selectedLayer}`}
          className="h-2 w-48 cursor-pointer appearance-none rounded-full bg-muted accent-primary disabled:cursor-not-allowed disabled:opacity-50"
        />
        <span className="font-mono text-xs tabular-nums text-foreground/80">{`Layer ${selectedLayer}`}</span>
        <span className="text-xs text-muted-foreground">
          {run?.modelMeta?.name ?? '—'}
          {hasData ? ` · ${numLayers} layers` : ''}
        </span>
      </div>

      {status?.kind === 'error' ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Inspector run failed.</strong> {status.error}
        </div>
      ) : null}
      {status?.kind === 'aborted' ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Request cancelled — model was reloaded. Click <strong>Run</strong> again to retry.
        </div>
      ) : null}
      {status?.kind === 'empty-prompt' ? (
        <div
          role="alert"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Enter a prompt to see normalization in action.
        </div>
      ) : null}

      {hasData ? (
        <div ref={resultFlashRef} className="space-y-3 p-1">
          <LayerDriftChart steps={hiddenSteps} />
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Single-layer breakdown</div>
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
          Click <strong>Run</strong> to capture pre- and post-RMSNorm activation stats for a single forward pass.
        </div>
      )}

      <ScaleInvariancePlayground />

      <DemoCallout
        items={[
          'The drift chart is the headline: residual L2 climbs, norm L2 stays anchored near √hidden_dim.',
          'The playground proves it synthetically — pull the slider to 0.1× or 10×, the output L2 still parks at √256 ≈ 16.',
          'RMSNorm preserves direction (cosine = 1.0000). It only rescales magnitude, never rotates the vector.',
        ]}
      />
    </div>
  );
}

function StatBox({ title, stats }: { title: string; stats: HiddenStatePointStats | undefined }) {
  const l2 = perTokenL2(stats);
  return (
    <div className="rounded-md border border-border bg-background p-3 text-xs">
      <div className="mb-2 uppercase tracking-wider text-muted-foreground">{title}</div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono">
        <dt className="text-muted-foreground">L2/tok</dt>
        <dd className="text-right text-foreground/90">{l2 !== null ? formatNumber(l2, 2) : '—'}</dd>
        <dt className="text-muted-foreground">|max|</dt>
        <dd className="text-right text-foreground/90">{stats ? formatNumber(stats.absMax, 2) : '—'}</dd>
        <dt className="text-muted-foreground">std</dt>
        <dd className="text-right text-foreground/90">{stats ? formatNumber(stats.std) : '—'}</dd>
        <dt className="text-muted-foreground">mean</dt>
        <dd className="text-right text-foreground/90">{stats ? formatNumber(stats.mean) : '—'}</dd>
      </dl>
    </div>
  );
}

function NormStatsGrid({ layerStats }: { layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>> }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {NORM_PAIRS.flatMap((pair) => [
        <StatBox key={`${pair.inputPoint}-box`} title={pair.inputLabel} stats={layerStats[pair.inputPoint]} />,
        <StatBox key={`${pair.outputPoint}-box`} title={pair.outputLabel} stats={layerStats[pair.outputPoint]} />,
      ])}
    </div>
  );
}

function L2BeforeAfterChart({ layerStats }: { layerStats: Partial<Record<CapturePoint, HiddenStatePointStats>> }) {
  const bars = NORM_PAIRS.flatMap((pair) => {
    const before = perTokenL2(layerStats[pair.inputPoint]);
    const after = perTokenL2(layerStats[pair.outputPoint]);
    return [
      {
        group: pair.label,
        label: 'before',
        value: before,
      },
      {
        group: pair.label,
        label: 'after',
        value: after,
      },
    ];
  });
  const finiteValues = bars.map((b) => b.value).filter((v): v is number => v !== null && Number.isFinite(v));
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
            <div className="text-xs uppercase tracking-wider text-muted-foreground">{pair.label}</div>
            <BarRow label="before" value={before} max={maxValue} accent={false} />
            <BarRow label="after" value={after} max={maxValue} accent />
          </div>
        );
      })}
      <div className="pt-1 text-[11px] text-muted-foreground">
        Values are per-token L2 (raw L2 ÷ sqrt(seq_len)). Reference: sqrt(hidden_dim) ≈{' '}
        {Math.sqrt(HIDDEN_DIM).toFixed(1)} for Qwen3.5-0.8B's 1024-dim hidden state.
      </div>
    </div>
  );
}

function BarRow({ label, value, max, accent }: { label: string; value: number | null; max: number; accent: boolean }) {
  const safeMax = max > 0 ? max : 1;
  const pct = value === null || !Number.isFinite(value) ? 0 : Math.max(0, Math.min(100, (value / safeMax) * 100));
  return (
    <div className="grid grid-cols-[4rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-[12px]">
      <span className="font-mono uppercase tracking-wider text-muted-foreground">{label}</span>
      <div className="relative h-4 w-full overflow-hidden rounded bg-muted">
        <div
          className={['h-full', accent ? 'bg-primary' : 'bg-foreground/40'].join(' ')}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-right font-mono text-foreground/80">{value === null ? '—' : formatNumber(value, 2)}</span>
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
        Across {numLayers} layers · attn-norm L2/tok: {formatNumber(preIn, 1)} → {formatNumber(preOut, 2)} · mlp-norm
        L2/tok: {formatNumber(mlpIn, 1)} → {formatNumber(mlpOut, 2)}.
      </div>
    </div>
  );
}

/**
 * Two line chart over all captured layers:
 *   - residual stream entering the attn-norm (drifts upward with depth)
 *   - same residual after the attn-norm (stays near sqrt(hidden_dim))
 *
 * The two curves live on very different scales (residual is single-digit at
 * shallow layers; post-norm sits in the 30s-50s), so the gray line looks
 * flat if you put both on a shared linear y-axis. We instead label the
 * first/last value of each curve directly on the chart, and call out the
 * growth ratio in the caption — the lesson is "residual grows ~15×, RMSNorm
 * doesn't budge", and the labels make that concrete without a log axis.
 */
function LayerDriftChart({ steps }: { steps: HiddenStateStep[] | undefined }) {
  const series = React.useMemo(() => {
    if (!steps || !steps[0]) return null;
    const layers = steps[0].layers.slice().sort((a, b) => a.layerIdx - b.layerIdx);
    const xs: number[] = [];
    const inputs: Array<number | null> = [];
    const outputs: Array<number | null> = [];
    for (const layer of layers) {
      xs.push(layer.layerIdx);
      const m: Partial<Record<string, HiddenStatePointStats>> = {};
      for (const s of layer.stats) m[s.point] = s;
      inputs.push(perTokenL2(m['pre_attn_input']));
      outputs.push(perTokenL2(m['post_attn_norm']));
    }
    return { xs, inputs, outputs };
  }, [steps]);

  if (!series || series.xs.length < 2) return null;
  // Capture series in a local so closures defined below get the non-null type.
  const data = series;

  const W = 520;
  const H = 200;
  const padL = 36;
  const padR = 44;
  const padT = 32;
  const padB = 28;
  const iW = W - padL - padR;
  const iH = H - padT - padB;
  const finite = [...data.inputs, ...data.outputs].filter((v): v is number => v !== null && Number.isFinite(v));
  const reference = Math.sqrt(HIDDEN_DIM);
  const yMaxRaw = Math.max(...finite, reference * 1.4);
  const yMax = yMaxRaw * 1.1;
  const minL = data.xs[0]!;
  const maxL = data.xs[data.xs.length - 1]!;

  function xFor(layer: number): number {
    const t = maxL === minL ? 0.5 : (layer - minL) / (maxL - minL);
    return padL + t * iW;
  }
  function yFor(v: number): number {
    return padT + iH - (v / yMax) * iH;
  }
  function pathFor(vals: Array<number | null>): string {
    let d = '';
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !Number.isFinite(v)) continue;
      const x = xFor(data.xs[i]!).toFixed(1);
      const y = yFor(v).toFixed(1);
      d += d ? ` L ${x} ${y}` : `M ${x} ${y}`;
    }
    return d;
  }
  const referenceY = yFor(reference);

  // First and last *finite* values, with their layer indices, for endpoint labels.
  function endpoints(vals: Array<number | null>): {
    first?: { layer: number; v: number };
    last?: { layer: number; v: number };
  } {
    let first: { layer: number; v: number } | undefined;
    let last: { layer: number; v: number } | undefined;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !Number.isFinite(v)) continue;
      if (!first) first = { layer: data.xs[i]!, v };
      last = { layer: data.xs[i]!, v };
    }
    return { first, last };
  }
  const inputEnds = endpoints(data.inputs);
  const outputEnds = endpoints(data.outputs);
  const growthFactor =
    inputEnds.first && inputEnds.last && inputEnds.first.v > 1e-6 ? inputEnds.last.v / inputEnds.first.v : null;

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        What's drifting? · L2/tok across all {data.xs.length} layers
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mt-1 block h-auto w-full"
        role="img"
        aria-label="Per-token L2 norm of the residual stream entering each layer vs. after the attention RMSNorm"
      >
        {/* sqrt(d) reference line */}
        <line
          x1={padL}
          x2={W - padR}
          y1={referenceY}
          y2={referenceY}
          stroke="currentColor"
          strokeOpacity={0.25}
          strokeDasharray="4 3"
        />
        <text x={W - padR + 2} y={referenceY + 3} fontSize={9} fill="currentColor" fillOpacity={0.55}>
          √{HIDDEN_DIM}≈{reference.toFixed(0)}
        </text>
        {/* x-axis */}
        <line x1={padL} x2={W - padR} y1={padT + iH} y2={padT + iH} stroke="currentColor" strokeOpacity={0.2} />
        <text x={padL} y={H - 8} fontSize={9} fill="currentColor" fillOpacity={0.5}>
          layer {minL}
        </text>
        <text x={W - padR} y={H - 8} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          layer {maxL}
        </text>
        {/* y-axis tick labels */}
        <text x={padL - 4} y={yFor(yMax) + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          {yMax.toFixed(0)}
        </text>
        <text x={padL - 4} y={yFor(0) + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          0
        </text>
        {/* residual stream (drifts) — thicker so the small numbers don't disappear */}
        <path d={pathFor(data.inputs)} fill="none" stroke="currentColor" strokeOpacity={0.75} strokeWidth={2} />
        {/* normed output (anchored) */}
        <path d={pathFor(data.outputs)} fill="none" className="stroke-primary" strokeWidth={2} />
        {/* endpoint markers + value labels for the residual line */}
        {inputEnds.first ? (
          <g>
            <circle
              cx={xFor(inputEnds.first.layer)}
              cy={yFor(inputEnds.first.v)}
              r={3}
              fill="currentColor"
              fillOpacity={0.8}
            />
            <text
              x={xFor(inputEnds.first.layer) + 6}
              y={yFor(inputEnds.first.v) - 5}
              fontSize={10}
              fill="currentColor"
              fillOpacity={0.85}
            >
              {inputEnds.first.v.toFixed(1)}
            </text>
          </g>
        ) : null}
        {inputEnds.last ? (
          <g>
            <circle
              cx={xFor(inputEnds.last.layer)}
              cy={yFor(inputEnds.last.v)}
              r={3}
              fill="currentColor"
              fillOpacity={0.8}
            />
            <text
              x={xFor(inputEnds.last.layer) - 4}
              y={yFor(inputEnds.last.v) - 5}
              fontSize={10}
              textAnchor="end"
              fill="currentColor"
              fillOpacity={0.85}
            >
              {inputEnds.last.v.toFixed(1)}
            </text>
          </g>
        ) : null}
        {/* endpoint markers + value labels for the post-norm line */}
        {outputEnds.first ? (
          <g>
            <circle cx={xFor(outputEnds.first.layer)} cy={yFor(outputEnds.first.v)} r={3} className="fill-primary" />
            <text
              x={xFor(outputEnds.first.layer) + 6}
              y={yFor(outputEnds.first.v) - 5}
              fontSize={10}
              className="fill-primary"
            >
              {outputEnds.first.v.toFixed(1)}
            </text>
          </g>
        ) : null}
        {outputEnds.last ? (
          <g>
            <circle cx={xFor(outputEnds.last.layer)} cy={yFor(outputEnds.last.v)} r={3} className="fill-primary" />
            <text
              x={xFor(outputEnds.last.layer) - 4}
              y={yFor(outputEnds.last.v) - 5}
              fontSize={10}
              textAnchor="end"
              className="fill-primary"
            >
              {outputEnds.last.v.toFixed(1)}
            </text>
          </g>
        ) : null}
        {/* legend */}
        <g transform={`translate(${padL + 4}, 4)`}>
          <line x1={0} y1={8} x2={14} y2={8} stroke="currentColor" strokeOpacity={0.75} strokeWidth={2} />
          <text x={18} y={11} fontSize={10} fill="currentColor" fillOpacity={0.85}>
            residual entering layer (norm input)
          </text>
          <line x1={250} y1={8} x2={264} y2={8} className="stroke-primary" strokeWidth={2} />
          <text x={268} y={11} fontSize={10} fill="currentColor" fillOpacity={0.85}>
            after RMSNorm
          </text>
        </g>
      </svg>
      <div className="pt-1 text-[11px] text-muted-foreground">
        {growthFactor !== null && inputEnds.first && inputEnds.last ? (
          (() => {
            const outsAll = data.outputs.filter((v): v is number => v !== null && Number.isFinite(v));
            const outLo = Math.min(...outsAll);
            const outHi = Math.max(...outsAll);
            return (
              <>
                Residual L2/tok climbs {inputEnds.first.v.toFixed(1)} → {inputEnds.last.v.toFixed(1)} (
                {growthFactor.toFixed(0)}× over {data.xs.length} layers) — that's the drift. After RMSNorm + learned
                gain, L2/tok lives in the {outLo.toFixed(0)}–{outHi.toFixed(0)} band anchored at √{HIDDEN_DIM} ≈{' '}
                {reference.toFixed(0)}. Without the gain it would be dead flat at {reference.toFixed(0)}; the per-layer
                variation you see is g doing its job.
              </>
            );
          })()
        ) : (
          <>The residual grows with depth; RMSNorm pins each sub-block input back to ≈ √{HIDDEN_DIM}.</>
        )}
      </div>
    </div>
  );
}

/**
 * Pure-JS playground. Multiplies a fixed seeded 256-dim Gaussian by a slider
 * scale, applies RMSNorm with g=1, and shows the input vs. output L2 +
 * before/after histograms + the cosine similarity (always exactly 1.0 for
 * scalar multiples — RMSNorm doesn't rotate). No worker call.
 */
function ScaleInvariancePlayground() {
  const DIM = 256;
  const SEED = 0xc0ffee;
  const [scale, setScale] = React.useState(1);

  const baseVec = React.useMemo(() => seededGaussianVector(DIM, SEED), []);

  const computed = React.useMemo(() => {
    const inp = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) inp[i] = baseVec[i]! * scale;
    const out = applyRmsNorm(inp);
    let dot = 0;
    let a2 = 0;
    let b2 = 0;
    for (let i = 0; i < DIM; i++) {
      dot += inp[i]! * out[i]!;
      a2 += inp[i]! * inp[i]!;
      b2 += out[i]! * out[i]!;
    }
    const cos = dot / Math.sqrt(a2 * b2 || 1);
    return { input: inp, output: out, inputL2: vecL2(inp), outputL2: vecL2(out), cos };
  }, [baseVec, scale]);

  // Adaptive input range so the histogram doesn't squish or clip when the
  // user pulls the scale slider. Output range is fixed — that's the point.
  const inputRange = 4 * Math.max(1, scale);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">Try it: RMSNorm is scale-invariant</div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        A fixed seeded 256-dim Gaussian, multiplied by the slider's scale, then run through{' '}
        <code className="text-foreground/80">y = x / sqrt(mean(x²) + ε)</code> with g = 1. Move the slider and watch the
        output magnitude stay glued to √256 = {Math.sqrt(DIM).toFixed(1)}.
      </p>
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <label htmlFor="rmsnorm-scale" className="uppercase tracking-wider">
            input scale
          </label>
          <span className="font-mono text-foreground/80">{scale.toFixed(1)}×</span>
        </div>
        <input
          id="rmsnorm-scale"
          type="range"
          min={0.1}
          max={10}
          step={0.1}
          value={scale}
          onChange={(e) => setScale(Number.parseFloat(e.target.value))}
          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
        />
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="rounded border border-border/60 bg-muted/20 px-2 py-1.5">
          <div className="uppercase tracking-wider text-muted-foreground">input L2</div>
          <div className="mt-0.5 font-mono text-base text-foreground/90">{computed.inputL2.toFixed(1)}</div>
          <div className="mt-1 text-[10px] text-muted-foreground">scales linearly with the slider</div>
        </div>
        <div className="rounded border border-primary/40 bg-primary/5 px-2 py-1.5">
          <div className="uppercase tracking-wider text-muted-foreground">output L2</div>
          <div className="mt-0.5 font-mono text-base text-primary">{computed.outputL2.toFixed(1)}</div>
          <div className="mt-1 text-[10px] text-muted-foreground">
            parks at √{DIM} ≈ {Math.sqrt(DIM).toFixed(1)}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <MiniHistogram
          title="input distribution"
          data={histogram(computed.input, 24, -inputRange, inputRange)}
          min={-inputRange}
          max={inputRange}
          accent={false}
        />
        <MiniHistogram
          title="output distribution"
          data={histogram(computed.output, 24, -4, 4)}
          min={-4}
          max={4}
          accent
        />
      </div>
      <div className="text-[11px] text-muted-foreground">
        cosine(input, output) = <span className="font-mono text-foreground/80">{computed.cos.toFixed(4)}</span> —{' '}
        RMSNorm only changes magnitude; direction is preserved bit-for-bit.
      </div>
      <div className="text-[10px] text-muted-foreground/80">
        Note: we use g = 1 here for clarity. Real models multiply each output element by a learned scalar g<sub>i</sub>{' '}
        — that reweights features but never changes the cosine.
      </div>
    </div>
  );
}

function MiniHistogram({
  title,
  data,
  min,
  max,
  accent,
}: {
  title: string;
  data: number[];
  min: number;
  max: number;
  accent: boolean;
}) {
  const peak = Math.max(...data, 1);
  return (
    <div className="rounded-md border border-border/60 bg-muted/20 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">{title}</div>
      <div className="flex h-12 items-end gap-px" aria-hidden="true">
        {data.map((v, i) => (
          <div
            key={i}
            className={['flex-1 rounded-sm', accent ? 'bg-primary' : 'bg-foreground/40'].join(' ')}
            style={{ height: `${(v / peak) * 100}%`, minHeight: v > 0 ? 1 : 0 }}
          />
        ))}
      </div>
      <div className="flex justify-between pt-1 font-mono text-[9px] text-muted-foreground">
        <span>{min.toFixed(1)}</span>
        <span>0</span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  );
}

/**
 * Side-by-side flow diagram showing where the norm sits inside a transformer
 * sub-block. The accent on the RMSNorm box highlights the placement
 * difference; the dashed line traces the residual highway so the "gradient
 * path" claim in the prose has a visual referent.
 */
function NormFlowDiagram() {
  return (
    <div className="my-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
      <NormFlowBox variant="post" />
      <NormFlowBox variant="pre" />
    </div>
  );
}

function NormFlowBox({ variant }: { variant: 'pre' | 'post' }) {
  const W = 220;
  const H = 200;
  const isPre = variant === 'pre';

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">
        {isPre ? 'Pre-norm (Qwen3.5, modern LLMs)' : 'Post-norm (2017 original)'}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block h-auto w-full max-w-[260px]"
        role="img"
        aria-label={isPre ? 'Pre-norm sub-block diagram' : 'Post-norm sub-block diagram'}
      >
        <defs>
          <marker
            id={`arrow-${variant}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor" opacity={0.55} />
          </marker>
        </defs>
        {/* input label */}
        <text x={W / 2} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          x
        </text>
        {isPre ? (
          <>
            {/* split: norm path on the left, residual highway on the right (dashed) */}
            <line x1={W / 2} y1={18} x2={W / 2} y2={28} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={W / 2} y1={28} x2={70} y2={48} stroke="currentColor" strokeOpacity={0.4} />
            <line
              x1={W / 2}
              y1={28}
              x2={W / 2}
              y2={148}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
            <text x={W / 2 + 6} y={90} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              residual
            </text>
            <text x={W / 2 + 6} y={102} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              (un-normed)
            </text>
            {/* RMSNorm box (accented) */}
            <rect
              x={35}
              y={48}
              width={70}
              height={28}
              rx={5}
              className="fill-primary/15"
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={66} fontSize={11} textAnchor="middle" className="fill-primary">
              RMSNorm
            </text>
            <line x1={70} y1={76} x2={70} y2={94} stroke="currentColor" strokeOpacity={0.4} />
            {/* attn / mlp */}
            <rect
              x={35}
              y={94}
              width={70}
              height={28}
              rx={5}
              fill="currentColor"
              fillOpacity={0.06}
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={112} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              attn / mlp
            </text>
            {/* connect to + node */}
            <line x1={70} y1={122} x2={70} y2={148} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={70} y1={148} x2={W / 2 - 10} y2={148} stroke="currentColor" strokeOpacity={0.4} />
            <circle cx={W / 2} cy={148} r={10} fill="none" stroke="currentColor" strokeOpacity={0.5} />
            <text x={W / 2} y={152} fontSize={12} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              +
            </text>
            <line
              x1={W / 2}
              y1={158}
              x2={W / 2}
              y2={178}
              stroke="currentColor"
              strokeOpacity={0.55}
              markerEnd={`url(#arrow-${variant})`}
            />
            <text x={W / 2} y={194} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
              x_out
            </text>
          </>
        ) : (
          <>
            {/* residual stream on right (still un-normed inside the block) */}
            <line x1={W / 2} y1={18} x2={W / 2} y2={28} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={W / 2} y1={28} x2={70} y2={48} stroke="currentColor" strokeOpacity={0.4} />
            <line
              x1={W / 2}
              y1={28}
              x2={W / 2}
              y2={114}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
            <text x={W / 2 + 6} y={80} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              residual
            </text>
            {/* attn / mlp first */}
            <rect
              x={35}
              y={48}
              width={70}
              height={28}
              rx={5}
              fill="currentColor"
              fillOpacity={0.06}
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={66} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              attn / mlp
            </text>
            <line x1={70} y1={76} x2={70} y2={114} stroke="currentColor" strokeOpacity={0.4} />
            {/* + node */}
            <line x1={70} y1={114} x2={W / 2 - 10} y2={114} stroke="currentColor" strokeOpacity={0.4} />
            <circle cx={W / 2} cy={114} r={10} fill="none" stroke="currentColor" strokeOpacity={0.5} />
            <text x={W / 2} y={118} fontSize={12} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              +
            </text>
            {/* RMSNorm sits on the residual highway */}
            <line x1={W / 2} y1={124} x2={W / 2} y2={140} stroke="currentColor" strokeOpacity={0.4} />
            <rect
              x={W / 2 - 35}
              y={140}
              width={70}
              height={28}
              rx={5}
              className="fill-primary/15"
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={W / 2} y={158} fontSize={11} textAnchor="middle" className="fill-primary">
              RMSNorm
            </text>
            <line
              x1={W / 2}
              y1={168}
              x2={W / 2}
              y2={178}
              stroke="currentColor"
              strokeOpacity={0.55}
              markerEnd={`url(#arrow-${variant})`}
            />
            <text x={W / 2} y={194} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
              x_out
            </text>
          </>
        )}
      </svg>
      <div className="mt-1 text-[11px] text-muted-foreground">
        {isPre
          ? 'Norm only touches the sub-block input. The residual highway stays un-normalized — gradients flow through a clean identity path. Stable to 24+ layers.'
          : 'Norm sits on the residual highway. Gradients have to fight through one extra norm at every layer. Trains fine for 6 layers; painful past 20.'}
      </div>
    </div>
  );
}
