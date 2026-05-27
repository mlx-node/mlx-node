import * as React from "react";

import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import type { AttentionRun } from "../../../src/inspector-types";
import { runForInspector } from "../../lib/inspector-client";
import { AttentionHeatmap } from "../inspector/AttentionHeatmap";
import { Prose } from "../Prose";
import { makeMockAttentionRun } from "../mock-data";

/**
 * Chapter 3 — Self-attention.
 *
 * This is the only fully-authored chapter for the first cut. The prose
 * teaches the mechanism end-to-end; the right-hand panel runs the real
 * backend `runForInspector` hook when the model is loaded, and falls back to
 * a mocked AttentionRun (with a visible banner) when it isn't.
 */

const DEFAULT_PROMPT = "The cat sat on the mat.";

export type AttentionDemoProps = {
  /**
   * Ref to the shared MLX worker owned by the app shell. `null` while the
   * model is not loaded — in that case the demo falls back to mock data and
   * shows a banner so the user knows what they're looking at.
   */
  workerRef: React.RefObject<Worker | null>;
  /**
   * Ref to an AbortController bound to the current worker's lifetime. The
   * app shell aborts this before calling `worker.terminate()` (model swap /
   * reload kickoff). The Run handler threads `signal` into `runForInspector`
   * so an in-flight call rejects immediately on teardown instead of hanging
   * for the full 60s timeout. May be `null` between worker lifecycles.
   */
  abortRef: React.RefObject<AbortController | null>;
};

export function AttentionChapterBody() {
  return (
    <Prose>
      <h1>Self-attention: how every token looks at every other token</h1>
      <p>
        Up to this chapter we've turned text into tokens (chapter 1) and tokens
        into vectors (chapter 2). The interesting question is now: how does a
        token <em>know about the rest of the sentence?</em> The answer modern
        LLMs use is <strong>self-attention</strong>.
      </p>

      <h2>The intuition</h2>
      <p>
        Imagine reading the sentence <code>"The cat sat on the mat."</code> When
        you read <code>mat</code> you're implicitly looking back at{" "}
        <code>cat</code> (because cats sit on mats more often than, say,
        algorithms do) and at <code>on</code> (because that's the relationship).
        Self-attention is a learned, differentiable version of that "look
        back": for every token in the sequence, the model decides{" "}
        <em>how much</em> it should pay attention to every other token.
      </p>

      <h2>Q, K, V — three projections of the same vector</h2>
      <p>
        Each token comes in as a vector <code>x</code> (its embedding, after
        earlier layers have refined it). The model linearly projects{" "}
        <code>x</code> three different ways:
      </p>
      <ul>
        <li>
          <code>Q = x · Wq</code> — the <strong>query</strong>: what is this
          token looking for?
        </li>
        <li>
          <code>K = x · Wk</code> — the <strong>key</strong>: what does this
          token offer if you look at it?
        </li>
        <li>
          <code>V = x · Wv</code> — the <strong>value</strong>: what should
          this token contribute when it gets looked at?
        </li>
      </ul>
      <p>
        Each is a vector of length <code>d_head</code>. Crucially these aren't
        three different tokens — they're three different views of the same
        token, learned separately so attention can do something more
        interesting than just "compare embeddings."
      </p>

      <h2>The formula</h2>
      <p>The core operation, for a single head, is one line of math:</p>
      <pre>
        <code>{`attention(Q, K, V) = softmax( Q · Kᵀ / √d_head ) · V`}</code>
      </pre>
      <p>
        Let's read it left to right. <code>Q · Kᵀ</code> is a{" "}
        <code>[seq_len, seq_len]</code> matrix: cell <code>(i, j)</code> is the
        dot product of token <em>i</em>'s query with token <em>j</em>'s key —
        a raw score for "how well does <em>i</em>'s question match{" "}
        <em>j</em>'s offer?"
      </p>
      <p>
        We divide by <code>√d_head</code> for a numerical reason: as{" "}
        <code>d_head</code> grows, dot products grow on average too, and
        without scaling the softmax would saturate (one cell at 1.0, the rest
        near 0.0) and gradients would vanish. Dividing by the standard
        deviation of a random dot product (≈ <code>√d_head</code>) keeps the
        scores in a regime where softmax is informative.
      </p>
      <p>
        <code>softmax</code> turns each row of the score matrix into a
        probability distribution: every row sums to 1. That's the per-token
        <em> attention pattern</em>. Multiplying that <code>[seq_len, seq_len]</code>{" "}
        distribution by <code>V</code> mixes the values together, weighted by
        the pattern — and that mixture is what flows out of the attention layer
        as the new representation of token <em>i</em>.
      </p>

      <h2>Causal masking</h2>
      <p>
        During training (and inference for generative LLMs) we don't want a
        token to peek at tokens that come after it — otherwise the model would
        cheat at next-token prediction. Before the softmax we zero out (well,
        set to <code>-∞</code>) the upper triangle of the score matrix. After
        softmax those cells become 0. That's why the heatmap on the right is{" "}
        <strong>lower-triangular</strong>: row <em>i</em> only attends to{" "}
        keys <em>0 … i</em>.
      </p>

      <h2>What each cell in the heatmap means</h2>
      <p>
        The heatmap on the right is exactly the <code>softmax(QKᵀ/√d)</code>{" "}
        matrix for one layer and one head:
      </p>
      <ul>
        <li>
          rows are <strong>queries</strong> (the token doing the looking)
        </li>
        <li>
          columns are <strong>keys</strong> (the token being looked at)
        </li>
        <li>
          a bright cell at <code>(i, j)</code> means "when computing the next
          representation of token <em>i</em>, the model pulls a lot from token{" "}
          <em>j</em>'s value"
        </li>
        <li>
          every row sums to 1; the upper triangle is 0 because of causal
          masking
        </li>
      </ul>

      <h2>Why multiple heads, multiple layers</h2>
      <p>
        Different heads end up specializing — one might track "the previous
        token," another "the start of the sentence," another "syntactically
        related noun." Stacking attention layers lets later layers compose
        these patterns: chapter 4 looks at heads, chapter 8 at the full stack.
      </p>

      <p className="mt-6 text-muted-foreground">
        Type a sentence on the right and hit <em>Run</em>. If the model is
        loaded, the heatmap shows the real post-softmax attention scores from
        a forward pass; otherwise a synthetic example is shown with a banner
        above the heatmap. Switch the layer and head selectors to see how the
        pattern changes.
      </p>
    </Prose>
  );
}

type RunStatus =
  | { kind: "ok" }
  | { kind: "mock-no-worker" }
  | { kind: "mock-error"; error: string }
  | { kind: "aborted" };

function isAbortError(err: unknown): boolean {
  return (
    err instanceof DOMException &&
    err.name === "AbortError"
  );
}

export function AttentionDemo({ workerRef, abortRef }: AttentionDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  // Timestamp (epoch ms) of the most recent successful run. Greedy sampling
  // is deterministic, so re-running the same prompt produces an identical
  // heatmap — without this timestamp the user has no way to tell their click
  // actually did anything. Bumped on every successful run; rendered as a
  // live-updating "Last run: just now / 5s ago / 1m ago" pill.
  const [lastRunAt, setLastRunAt] = React.useState<number | null>(null);
  // Bumps on every successful run so we can trigger a brief border flash on
  // the heatmap container — purely visual feedback that "yes, the canvas
  // just re-rendered with new data, even if the pattern looks identical."
  const [runFlash, setRunFlash] = React.useState(0);

  function useMockFallback(reason: RunStatus, logMessage: string) {
    const mock = makeMockAttentionRun(prompt);
    console.log(logMessage, {
      promptLength: prompt.length,
      seqLen: mock.tokens.length,
    });
    setRun(mock);
    setStatus(reason);
  }

  async function handleRun() {
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      useMockFallback(
        { kind: "mock-no-worker" },
        "[attention-demo] using mock AttentionRun (model not loaded)",
      );
      setRunning(false);
      return;
    }

    try {
      const signal = abortRef.current?.signal;
      const result = await runForInspector(
        worker,
        {
          prompt,
          attention: true,
          maxNewTokens: 1,
        },
        signal ? { signal } : undefined,
      );
      setRun(result);
      setStatus({ kind: "ok" });
      setLastRunAt(Date.now());
      setRunFlash((n) => n + 1);
    } catch (err) {
      if (isAbortError(err)) {
        // Worker was terminated (model reload / swap) mid-call. Treat this as
        // a soft cancellation, not an error: leave any previously-rendered
        // run in place and show a neutral banner.
        console.info(
          "[attention-demo] runForInspector aborted (worker terminated)",
        );
        setStatus({ kind: "aborted" });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error(
          "[attention-demo] runForInspector failed; falling back to mock",
          err,
        );
        useMockFallback(
          { kind: "mock-error", error: message },
          "[attention-demo] using mock AttentionRun (backend error)",
        );
      }
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-xs uppercase tracking-wider text-muted-foreground">
          Prompt
        </label>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="font-mono text-sm"
          placeholder="The cat sat on the mat."
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Generates one token, captures attention scores.
          </span>
          {lastRunAt !== null ? <LastRunPill timestamp={lastRunAt} /> : null}
        </div>
        {lastRunAt !== null && !running ? (
          <p className="text-xs text-muted-foreground">
            Greedy sampling is deterministic — clicking <strong>Run</strong>{" "}
            again on the same prompt produces an identical heatmap. Edit the
            prompt above (or try a different layer/head) to see the pattern
            change.
          </p>
        ) : null}
      </div>

      {status?.kind === "aborted" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Inspector run cancelled — the model was reloaded. Click{" "}
          <strong>Run</strong> again to retry.
        </div>
      ) : null}
      {run ? (
        <div className="space-y-3 pt-2">
          {status?.kind === "mock-no-worker" ? (
            <div
              role="status"
              className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
            >
              Showing example data — load the model first to see real
              attention scores.
            </div>
          ) : null}
          {status?.kind === "mock-error" ? (
            <div
              role="alert"
              className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
            >
              <strong>Inspector run failed.</strong> Showing example data
              instead. {status.error}
            </div>
          ) : null}
          <div
            key={runFlash}
            className="run-flash-on-mount rounded-md"
          >
            <AttentionHeatmap run={run} />
          </div>
        </div>
      ) : status?.kind === "aborted" ? null : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to see the attention heatmap.
        </div>
      )}
    </div>
  );
}

/**
 * A tiny "Last run: just now / 5s ago / 1m ago" pill that ticks every second
 * so the user gets active feedback that their Run click did something. The
 * timer cleans up on unmount; once the elapsed time crosses a minute we ease
 * off to a 15s refresh so we don't churn React renders forever. */
function LastRunPill({ timestamp }: { timestamp: number }) {
  const [, force] = React.useReducer((n: number) => n + 1, 0);
  React.useEffect(() => {
    const elapsedMs = Date.now() - timestamp;
    const interval = elapsedMs < 60_000 ? 1_000 : 15_000;
    const handle = window.setInterval(() => force(), interval);
    return () => window.clearInterval(handle);
  }, [timestamp]);
  const elapsedSec = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  const label =
    elapsedSec < 2
      ? "just now"
      : elapsedSec < 60
        ? `${elapsedSec}s ago`
        : `${Math.round(elapsedSec / 60)}m ago`;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-primary"
      aria-live="polite"
    >
      <span className="size-1.5 rounded-full bg-primary" aria-hidden="true" />
      Last run: {label}
    </span>
  );
}
