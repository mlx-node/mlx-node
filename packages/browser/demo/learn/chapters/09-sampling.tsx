import * as React from "react";

import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import type { AttentionRun } from "../../../src/inspector-types";
import { runForInspector } from "../../lib/inspector-client";
import { TopKBars } from "../inspector/TopKBars";
import { Prose } from "../Prose";
import { ChapterFrame } from "../scaffolding/ChapterFrame";
import type { ChapterLearningData } from "../scaffolding/learning-data";

/**
 * Chapter 9 — Sampling.
 *
 * Prose explains how logits → softmax → token works, including temperature
 * and top-p (nucleus) sampling. The interactive widget runs the real backend
 * inspector hook to capture per-step top-K logits, then lets the user re-apply
 * temperature/top-p sliders client-side over the cached raw logits — sliding
 * the controls does NOT re-run the model.
 */

const DEFAULT_PROMPT = "Once upon a time, in a forest far away";
const MAX_NEW_TOKENS = 6;
const TOP_K = 16;

/**
 * Scaffolding metadata for chapter 9 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[8].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: "sampling",
  objective:
    "Take a vector of logits, apply temperature and top-p, and reason about why each knob shifts the resulting distribution.",
  problem:
    "Every forward pass ends with logits; the model still needs a rule for turning that vector into one concrete next token.",
  minutes: 6,
  glossary: [
    {
      term: "logit",
      definition:
        "One unbounded real-valued score per vocab entry. The raw output of the model before any normalization.",
    },
    {
      term: "softmax",
      definition:
        "exp(l_i / T) / sum_j exp(l_j / T) — turns logits into a probability distribution that sums to 1.",
    },
    {
      term: "temperature (T)",
      definition:
        "Sharpness knob applied before softmax. T<1 sharpens (closer to greedy); T>1 flattens (more diverse).",
    },
    {
      term: "greedy decoding",
      definition:
        "Pick argmax(logits) every step. Deterministic and reproducible; tends to fall into repetition loops.",
    },
    {
      term: "top-p (nucleus)",
      definition:
        "Keep the smallest set of tokens whose probabilities sum to >= p, renormalize, sample from those. Trims the long tail.",
    },
    {
      term: "top-K",
      definition:
        "Keep only the K highest-probability tokens. The widget shows top-K=16 logits per step.",
    },
  ],
  takeaways: [
    "Logits become probabilities only after softmax — comparing raw logits across runs or models is meaningless.",
    "Temperature reshapes the distribution; top-p truncates the tail. A common default is T=0.7 with top_p=0.9.",
    "The model's actually-sampled token may not be the bar with the highest probability after you change the sliders — that's how the sliders steer generation.",
  ],
  exercise: {
    prompt:
      "After the auto-run, leave temperature at 1.0 and slowly lower top-p from 1.0 to 0.3 while watching the bar chart for step 1. At what point does the chart visibly collapse to one or two bars, and what does the renormalised height tell you?",
    answer:
      "Around top_p ≈ 0.5-0.7 the long tail disappears and the top two or three bars shoot up because the surviving mass is re-normalized to sum to 1. By top_p = 0.3 you usually have a single near-100% bar — top-p has effectively turned this step into greedy decoding. That collapse-point depends on how confident the step was: a confident step survives even very small p, an uncertain step needs more.",
  },
  quiz: [
    {
      id: "q1-greedy-issue",
      prompt: "Why doesn't production LLM serving just always use greedy decoding?",
      options: [
        { id: "a", label: "Greedy is too slow on GPUs." },
        {
          id: "b",
          label:
            "Greedy is deterministic and reproducible but prone to repetition — once the model finds a high-confidence loop it keeps re-entering it.",
        },
        {
          id: "c",
          label: "Greedy doesn't work with byte-pair encoding.",
        },
      ],
      correctId: "b",
      explanation:
        "Greedy is a perfectly valid decoder; it's just brittle. A small randomization (T or top-p) is what breaks the model out of locally-optimal loops.",
    },
    {
      id: "q2-temperature-effect",
      prompt: "What does setting temperature T = 0.2 do to the probability distribution?",
      options: [
        {
          id: "a",
          label:
            "Sharpens it — high-logit tokens dominate even more strongly than at T=1, so output behaves closer to greedy.",
        },
        {
          id: "b",
          label: "Flattens it — small differences in logits get washed out.",
        },
        {
          id: "c",
          label: "Removes the long tail entirely (equivalent to top-p).",
        },
      ],
      correctId: "a",
      explanation:
        "T < 1 divides logits by something smaller than 1 before exp, amplifying differences. The distribution gets spikier.",
    },
    {
      id: "q3-top-p-mechanism",
      prompt: "Top-p sampling with p = 0.9 means:",
      options: [
        {
          id: "a",
          label: "Multiply every probability by 0.9.",
        },
        {
          id: "b",
          label:
            "Keep the smallest set of tokens whose probabilities sum to >= 0.9, then renormalise that set to sum to 1 and sample from it.",
        },
        {
          id: "c",
          label: "Drop the 90th percentile of tokens.",
        },
      ],
      correctId: "b",
      explanation:
        "Top-p is dynamic: at a confident step the nucleus may be 2-3 tokens, at an uncertain step it may be dozens. That's why it adapts better than a fixed top-K.",
    },
  ],
};

export function SamplingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
      <h1>Sampling: turning logits into the next token</h1>
      <p>
        Every forward pass through Qwen3.5 ends the same way: a vector of{" "}
        <strong>logits</strong> — one real number per token in the model's
        vocabulary (~152k for Qwen3). A logit is an unbounded score:{" "}
        <em>how strongly</em> the model recommends that token as the next one.
        Sampling is the step that turns that vector into a single concrete
        choice.
      </p>

      <h2>Why not just pick the maximum?</h2>
      <p>
        The simplest rule is <strong>greedy decoding</strong>: take{" "}
        <code>argmax(logits)</code> at every step. It's deterministic and
        reproducible, but it has a famous failure mode — repetition. The
        instant the model finds a phrase whose continuation it's confident
        about, it keeps re-entering the same loop, because that loop is always
        the locally-best choice. Greedy decoding also throws away a lot of
        information: if two tokens have nearly equal logits, picking one and
        ignoring the other is a fragile tie-break.
      </p>

      <h2>Softmax: logits → probabilities</h2>
      <p>
        To sample, we first convert logits into a proper probability
        distribution with the <strong>softmax</strong> function:
      </p>
      <pre>
        <code>{`p_i = exp(l_i / T) / Σ_j exp(l_j / T)`}</code>
      </pre>
      <p>
        Two things are happening here. The exponential makes every value
        positive; the division by the sum makes them sum to 1. The numerator
        also includes a <strong>temperature</strong> <code>T</code> that
        scales the raw logits before the exponential. Temperature acts as a
        sharpness knob:
      </p>
      <ul>
        <li>
          <code>T &lt; 1</code> sharpens the distribution — high-logit tokens
          dominate even more, so output looks more like greedy.
        </li>
        <li>
          <code>T &gt; 1</code> flattens the distribution — small differences
          in logits are washed out, so output is more diverse but less
          coherent.
        </li>
        <li>
          <code>T = 0</code> collapses to greedy (argmax). The widget clamps
          to a tiny positive value to avoid dividing by zero, which is
          numerically equivalent.
        </li>
      </ul>

      <h2>Top-p (nucleus) sampling</h2>
      <p>
        Even with a sensible temperature, the long tail of the vocabulary
        still has tiny non-zero probability mass — and once in a while the
        sampler will land there. Most of those tail tokens are nonsense in
        context. <strong>Top-p sampling</strong> (also called{" "}
        <strong>nucleus sampling</strong>) trims the tail:
      </p>
      <ul>
        <li>Sort tokens by probability, descending.</li>
        <li>
          Walk the sorted list and keep accumulating probability until the
          cumulative sum reaches <code>p</code>.
        </li>
        <li>Throw everything after that away.</li>
        <li>Renormalize the survivors so they sum to 1 again.</li>
        <li>Sample from this nucleus.</li>
      </ul>
      <p>
        A common default is <code>T = 0.7</code> with <code>top_p = 0.9</code>
        : the temperature gives the model some room to be creative, and top-p
        guarantees we never sample from the absurd tail. The widget on the
        right lets you sweep both knobs over a captured run and see what would
        have happened.
      </p>

      <h2>How the widget works</h2>
      <p>
        Pressing <em>Run</em> generates <code>{MAX_NEW_TOKENS}</code> tokens
        with the inspector capturing the top-<code>{TOP_K}</code> raw logits
        at every step. The temperature and top-p sliders then re-apply
        softmax + truncation to the cached logits — no re-running the model.
        The bar that's highlighted is the token Qwen <em>actually</em>{" "}
        sampled (using its own internal sampler); compare it to the highest
        bar under your slider settings to see how a different temperature
        might have steered the generation.
      </p>

      <p className="mt-6 text-muted-foreground">
        Tip: try a low temperature (0.2) on a confident step versus a high
        temperature (1.5) on the same step. The visible "shape" of the bar
        chart is the entire reason sampling parameters matter for an LLM.
      </p>
      </Prose>
    </ChapterFrame>
  );
}

export type SamplingDemoProps = {
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

// Apply temperature + top-p to raw logits, returning candidate ids in their
// original (descending-logit) order paired with post-truncation probabilities.
// Returned probs always sum to 1 (unless every input was -Infinity, which the
// inspector never produces).
function applySampling(
  rawLogits: Float32Array,
  ids: number[],
  temperature: number,
  topP: number,
): { ids: number[]; probs: number[] } {
  const n = rawLogits.length;
  if (n === 0) return { ids: [], probs: [] };

  const T = Math.max(temperature, 1e-3);
  const scaled = new Array<number>(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = rawLogits[i]! / T;
    scaled[i] = v;
    if (v > max) max = v;
  }

  const exps = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(scaled[i]! - max);
    exps[i] = e;
    sum += e;
  }
  const probs = new Array<number>(n);
  if (sum > 0) {
    for (let i = 0; i < n; i++) probs[i] = exps[i]! / sum;
  } else {
    for (let i = 0; i < n; i++) probs[i] = 1 / n;
  }

  // Sort indices by probability descending. We need to remember the original
  // position so we can return ids in the requested (descending-logit) order
  // after truncation.
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => probs[b]! - probs[a]!,
  );

  // Determine cutoff in sorted order.
  let cutoff = order.length;
  if (topP < 1.0) {
    let acc = 0;
    cutoff = 0;
    for (let k = 0; k < order.length; k++) {
      acc += probs[order[k]!]!;
      cutoff = k + 1;
      if (acc >= topP) break;
    }
  }

  const kept = new Set<number>();
  let keptSum = 0;
  for (let k = 0; k < cutoff; k++) {
    const idx = order[k]!;
    kept.add(idx);
    keptSum += probs[idx]!;
  }

  const outIds: number[] = [];
  const outProbs: number[] = [];
  for (let i = 0; i < n; i++) {
    outIds.push(ids[i]!);
    if (kept.has(i) && keptSum > 0) {
      outProbs.push(probs[i]! / keptSum);
    } else {
      outProbs.push(0);
    }
  }
  return { ids: outIds, probs: outProbs };
}

export function SamplingDemo({ workerRef, abortRef }: SamplingDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  const [temperature, setTemperature] = React.useState(1.0);
  const [topP, setTopP] = React.useState(1.0);
  const [stepIdx, setStepIdx] = React.useState(0);

  // Per-call AbortController so a second click of "Run" tears down the
  // previous in-flight request before starting a new one.
  const runAbortRef = React.useRef<AbortController | null>(null);
  // Each new Run bumps a generation counter so a slow earlier reply doesn't
  // stomp UI state set by a newer click. Belt-and-suspenders alongside
  // `runAbortRef`: even if the abort dispatch races the listener firing, the
  // generation guard drops stale writes.
  const runGenRef = React.useRef(0);

  // Abort any in-flight run on unmount so the worker call doesn't keep
  // running after the demo is gone.
  React.useEffect(() => () => { runAbortRef.current?.abort(); }, []);

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
        "[sampling-demo] worker is unavailable — chapter view should have been gated on modelReady",
      );
      setStatus({ kind: "error", error: "Worker is unavailable. Reload the page." });
      setRunning(false);
      return;
    }

    // Cancel any prior in-flight call.
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
          logits: { topK: TOP_K },
          maxNewTokens: MAX_NEW_TOKENS,
        },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;
      console.log("[sampling-demo] runForInspector ok", {
        promptLength: prompt.length,
        steps: result.logits?.length ?? 0,
      });
      setRun(result);
      setStatus({ kind: "ok" });
      setStepIdx(0);
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info(
          "[sampling-demo] runForInspector aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) {
          setStatus({ kind: "aborted" });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error("[sampling-demo] runForInspector failed", err);
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

  const steps = run?.logits ?? [];
  const hasSteps = steps.length > 0;
  const safeStepIdx = hasSteps
    ? Math.min(Math.max(stepIdx, 0), steps.length - 1)
    : 0;
  const currentStep = hasSteps ? steps[safeStepIdx] ?? null : null;

  const distribution = React.useMemo(() => {
    if (!currentStep) return null;
    return applySampling(
      currentStep.topKLogits,
      currentStep.topKIds,
      temperature,
      topP,
    );
  }, [currentStep, temperature, topP]);

  // Walk topKTexts using each step's tokenId to find the actually-sampled
  // text. For exact-tie logits this may differ from `topKIds[0]` (see
  // inspector-types.ts).
  const actualGeneration = React.useMemo(() => {
    if (!hasSteps) return "";
    let out = "";
    for (const s of steps) {
      const idx = s.topKIds.indexOf(s.tokenId);
      out += idx >= 0 ? s.topKTexts[idx] ?? "" : "";
    }
    return out;
  }, [hasSteps, steps]);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="sampling-demo-input"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Prompt
        </label>
        <Textarea
          id="sampling-demo-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="font-mono text-sm"
          placeholder={DEFAULT_PROMPT}
        />
        <div className="flex items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Generates {MAX_NEW_TOKENS} tokens, captures top-{TOP_K} logits per
            step.
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <SliderRow
          id="sampling-demo-temperature"
          label="Temperature"
          value={temperature}
          min={0}
          max={2}
          step={0.05}
          onChange={setTemperature}
          format={(v) => v.toFixed(2)}
        />
        <SliderRow
          id="sampling-demo-topp"
          label="Top-p"
          value={topP}
          min={0}
          max={1}
          step={0.05}
          onChange={setTopP}
          format={(v) => v.toFixed(2)}
        />
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
          Enter a prompt to run sampling.
        </div>
      ) : null}

      {currentStep && distribution ? (
        <div className="space-y-3">
          <StepNavigator
            stepIdx={safeStepIdx}
            stepCount={steps.length}
            onStep={setStepIdx}
          />
          <TopKBars
            ids={distribution.ids}
            probs={distribution.probs}
            texts={currentStep.topKTexts}
            sampledTokenId={currentStep.tokenId}
          />
          <GenerationFooter
            prompt={run?.prompt ?? prompt}
            generated={actualGeneration}
          />
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to capture logits and visualize a few
          generation steps.
        </div>
      )}
    </div>
  );
}

function SliderRow({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <label htmlFor={id} className="uppercase tracking-wider">
          {label}
        </label>
        <span className="font-mono text-foreground/80">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
      />
    </div>
  );
}

function StepNavigator({
  stepIdx,
  stepCount,
  onStep,
}: {
  stepIdx: number;
  stepCount: number;
  onStep: (i: number) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={() => onStep(Math.max(0, stepIdx - 1))}
        disabled={stepIdx <= 0}
      >
        ‹ Prev
      </Button>
      <div className="font-mono text-xs text-muted-foreground">
        Step {stepIdx + 1} of {stepCount}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={() => onStep(Math.min(stepCount - 1, stepIdx + 1))}
        disabled={stepIdx >= stepCount - 1}
      >
        Next ›
      </Button>
    </div>
  );
}

function GenerationFooter({
  prompt,
  generated,
}: {
  prompt: string;
  generated: string;
}) {
  return (
    <div className="rounded-md bg-muted/40 px-3 py-2 font-mono text-xs">
      <div className="text-muted-foreground">Generated text (greedy / actual):</div>
      <div className="mt-1 break-words text-foreground/85">
        <span className="text-muted-foreground">{prompt}</span>
        <span className="text-primary">{generated}</span>
      </div>
    </div>
  );
}
