import * as React from 'react';

import type { AttentionRun } from '../../../src/inspector-types';
import { AttentionHeatmap } from '../../learn/inspector/AttentionHeatmap';
import { TopKBars } from '../../learn/inspector/TopKBars';
import { runForInspector } from '../../lib/inspector-client';
import { Button } from '../ui/button';

const MAX_NEW_TOKENS = 12;
const TOP_K = 16;

export type InspectorDrawerProps = {
  /** User prompt that produced the assistant message under inspection. The
   *  drawer re-runs the model on this exact text — the response may differ
   *  from the streamed one because temperature / RNG are independent. */
  prompt: string;
  /** Shared MLX worker. Null when the model is not yet loaded — the drawer
   *  surfaces the amber "model not loaded" banner in that case. */
  workerRef: React.RefObject<Worker | null>;
  /** App-shell abort signal tied to the worker's lifetime. Aborts on worker
   *  teardown so the in-flight inspector call rejects within a tick. */
  abortRef: React.RefObject<AbortController | null>;
  onClose: () => void;
};

type RunStatus =
  | { kind: 'loading' }
  | { kind: 'ok' }
  | { kind: 'no-worker' }
  | { kind: 'error'; error: string }
  | { kind: 'aborted' };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

export function InspectorDrawer({ prompt, workerRef, abortRef, onClose }: InspectorDrawerProps) {
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus>({ kind: 'loading' });
  const [stepIdx, setStepIdx] = React.useState(0);

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);

  React.useEffect(() => {
    const myGen = ++runGenRef.current;
    const worker = workerRef.current;
    if (!worker) {
      setStatus({ kind: 'no-worker' });
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

    setStatus({ kind: 'loading' });
    setRun(null);
    setStepIdx(0);

    runForInspector(
      worker,
      {
        prompt,
        attention: true,
        logits: { topK: TOP_K },
        maxNewTokens: MAX_NEW_TOKENS,
      },
      { signal: ctrl.signal },
    )
      .then((result) => {
        if (runGenRef.current !== myGen) return;
        setRun(result);
        setStatus({ kind: 'ok' });
      })
      .catch((err) => {
        if (runGenRef.current !== myGen) return;
        if (isAbortError(err)) {
          if (appSignal?.aborted) {
            setStatus({ kind: 'aborted' });
          }
        } else {
          const message = err instanceof Error ? err.message : String(err);
          console.error('[inspector-drawer] runForInspector failed', err);
          setStatus({ kind: 'error', error: message });
        }
      })
      .finally(() => {
        if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
        if (runAbortRef.current === ctrl) runAbortRef.current = null;
      });

    return () => {
      runGenRef.current++;
      ctrl.abort();
      if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
    };
  }, [prompt, workerRef, abortRef]);

  const steps = run?.logits ?? [];
  const hasSteps = steps.length > 0;
  const safeStepIdx = hasSteps ? Math.min(Math.max(stepIdx, 0), steps.length - 1) : 0;
  const currentStep = hasSteps ? (steps[safeStepIdx] ?? null) : null;

  return (
    <>
      <div className="inspector-drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="inspector-drawer" role="dialog" aria-label="Message inspector">
        <div className="inspector-drawer-header">
          <div>
            <div className="inspector-drawer-title">Inspect response</div>
            <div className="inspector-drawer-subtitle">
              Re-runs the model with the inspector enabled. The response may differ slightly from the original.
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close inspector">
            Close
          </Button>
        </div>

        <div className="inspector-drawer-body">
          <section className="space-y-2">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Prompt</div>
            <div className="rounded-md border border-border bg-muted/30 px-3 py-2 font-mono text-xs whitespace-pre-wrap break-words">
              {prompt}
            </div>
          </section>

          {status.kind === 'loading' ? (
            <div
              role="status"
              className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground"
            >
              Running inspector pass… (generating {MAX_NEW_TOKENS} tokens with attention + top-{TOP_K} logits captured)
            </div>
          ) : null}

          {status.kind === 'no-worker' ? (
            <div
              role="status"
              className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
            >
              Model isn't loaded yet — the inspector can't run. Send a chat message first to make sure the model is
              ready.
            </div>
          ) : null}

          {status.kind === 'aborted' ? (
            <div
              role="status"
              className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
            >
              Inspector run cancelled — the model was reloaded. Close and re-open the inspector to retry.
            </div>
          ) : null}

          {status.kind === 'error' ? (
            <div
              role="alert"
              className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
            >
              <strong>Inspector run failed.</strong> {status.error}
            </div>
          ) : null}

          {run && status.kind === 'ok' ? (
            <>
              <section className="space-y-2">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">Attention heatmap</div>
                <AttentionHeatmap run={run} />
              </section>

              {currentStep ? (
                <section className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">
                      Top-{TOP_K} logits per step
                    </div>
                    <div className="font-mono text-xs text-muted-foreground">
                      Step {safeStepIdx + 1} of {steps.length}
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setStepIdx(Math.max(0, safeStepIdx - 1))}
                      disabled={safeStepIdx <= 0}
                    >
                      ‹ Prev
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setStepIdx(Math.min(steps.length - 1, safeStepIdx + 1))}
                      disabled={safeStepIdx >= steps.length - 1}
                    >
                      Next ›
                    </Button>
                  </div>
                  <TopKBars
                    ids={currentStep.topKIds}
                    probs={normalizeLogitsToProbs(currentStep.topKLogits)}
                    texts={currentStep.topKTexts}
                    sampledTokenId={currentStep.tokenId}
                    // Re-fire the grow-in animation whenever the user
                    // navigates to a different step. A new run resets stepIdx
                    // to 0 so this also covers the first-step view of a fresh
                    // run; the AttentionHeatmap above shows the run boundary
                    // independently.
                    runKey={safeStepIdx}
                  />
                </section>
              ) : null}
            </>
          ) : null}
        </div>
      </aside>
    </>
  );
}

// Inspector returns raw top-K logits. Convert to probabilities so the bar
// chart shows distribution mass (matches what chapter 10 shows at T=1, p=1).
function normalizeLogitsToProbs(logits: Float32Array): number[] {
  const n = logits.length;
  if (n === 0) return [];
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i]!;
    if (v > max) max = v;
  }
  const exps = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i]! - max);
    exps[i] = e;
    sum += e;
  }
  if (sum <= 0) {
    return new Array<number>(n).fill(1 / n);
  }
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = exps[i]! / sum;
  return out;
}
