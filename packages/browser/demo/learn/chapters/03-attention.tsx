import * as React from "react";

import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import type { AttentionRun } from "../../../src/inspector-types";
import { runForInspector } from "../../lib/inspector-client";
import { AttentionHeatmap } from "../inspector/AttentionHeatmap";
import { DemoCallout } from "../inspector/DemoCallout";
import { renderTokenDisplay } from "../inspector/TopKBars";
import { Prose } from "../Prose";
import { ChapterFrame } from "../scaffolding/ChapterFrame";
import type { ChapterLearningData } from "../scaffolding/learning-data";
import { CausalMaskVisual } from "../widgets/CausalMaskVisual";
import { ToyAttentionExample } from "../widgets/ToyAttentionExample";

/**
 * Chapter 3 — Self-attention.
 *
 * This is the only fully-authored chapter for the first cut. The prose
 * teaches the mechanism end-to-end; the right-hand panel runs the real
 * backend `runForInspector` hook against the loaded model. The app gates
 * chapter rendering on `modelReady`, so the worker is always available
 * by the time this component mounts.
 */

// Default prompt deliberately stops mid-sentence so the next-token
// prediction is a meaningful continuation a beginner can read (e.g.
// " floor" / " mat" / " bed"), not the model picking after a period.
const DEFAULT_PROMPT = "The cat sat on the";

export type AttentionDemoProps = {
  /**
   * Ref to the shared MLX worker owned by the app shell. The app gates this
   * component behind `modelReady`, so by mount-time this ref points at a live
   * worker.
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

/**
 * Scaffolding metadata for chapter 3 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[2].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: "attention",
  objective:
    "Read a softmax(QK^T/sqrt(d))V attention map and explain what every cell, row, and column means.",
  problem:
    "Without attention, a transformer has no way to let one token's representation depend on what came before it.",
  minutes: 8,
  glossary: [
    {
      term: "query (Q)",
      definition:
        "A learned projection of a token saying 'what am I looking for?' — one query vector per token, per head.",
    },
    {
      term: "key (K)",
      definition:
        "A learned projection saying 'what do I offer if you look at me?' — compared with Q via dot product.",
    },
    {
      term: "value (V)",
      definition:
        "A learned projection that carries the actual content a token contributes when it is attended to.",
    },
    {
      term: "attention pattern",
      definition:
        "One row of softmax(QK^T/sqrt(d)): a probability distribution over earlier positions for a single query token.",
    },
    {
      term: "scaled dot product",
      definition:
        "QK^T divided by sqrt(d_head). The scale keeps softmax in a regime where gradients don't vanish as d_head grows.",
    },
    {
      term: "causal mask",
      definition:
        "Setting the upper triangle of the score matrix to -inf so position i can only attend to positions 0..i.",
    },
  ],
  takeaways: [
    "Each row of the heatmap is one query token's attention distribution; rows always sum to 1 after softmax.",
    "The lower-triangular shape isn't decorative — it's the causal mask preventing the model from cheating during training.",
    "The scaling by sqrt(d_head) is what lets attention scale to wide heads without softmax collapsing to one cell.",
  ],
  exercise: {
    prompt:
      "Hit Run on the default 'The cat sat on the' prompt, then in the heatmap pick the bottom row (the last token before the prediction). Which earlier token gets the most attention? What does that tell you about the head you are looking at?",
    answer:
      "Most heads put their brightest cell on the previous token (' the') or on ' cat' — that head is doing some mix of 'attend to last token' and 'attend to the noun whose verb I'm continuing'. Different layer/head pairs will look strikingly different — head 0 of an early layer often looks almost diagonal, while deeper heads spread attention farther back.",
  },
  quiz: [
    {
      id: "q1-row-sum",
      prompt: "What do the values in a single row of the attention heatmap sum to?",
      options: [
        { id: "a", label: "Roughly d_head, because of the sqrt(d_head) scaling." },
        { id: "b", label: "Exactly 1 — softmax is applied per row." },
        { id: "c", label: "Whatever the unnormalised dot products happen to be." },
      ],
      correctId: "b",
      explanation:
        "Softmax over each row produces a probability distribution: every row of the post-softmax matrix sums to 1.",
    },
    {
      id: "q2-scale-by-sqrt-d",
      prompt: "Why is the score matrix divided by sqrt(d_head) before softmax?",
      options: [
        {
          id: "a",
          label:
            "Dot products grow with d_head, so without the scale softmax saturates and gradients vanish.",
        },
        {
          id: "b",
          label: "It's a normalization step so weights sum to 1.",
        },
        {
          id: "c",
          label: "To make the result invariant to the choice of token vocabulary.",
        },
      ],
      correctId: "a",
      explanation:
        "Dividing by sqrt(d_head) keeps the variance of the scores roughly constant as the head dimension grows, so softmax stays in an informative regime.",
    },
    {
      id: "q3-why-triangular",
      prompt: "Why is the attention heatmap lower-triangular instead of a full square?",
      options: [
        {
          id: "a",
          label: "Empty cells are tokens the tokenizer never emitted.",
        },
        {
          id: "b",
          label:
            "A causal mask sets the upper triangle to -infinity before softmax so a token at position i can only look at positions 0..i.",
        },
        {
          id: "c",
          label: "GPU memory is conserved by skipping the upper half.",
        },
      ],
      correctId: "b",
      explanation:
        "The mask exists so the model can't peek at future tokens during training — otherwise next-token prediction would be trivial.",
    },
  ],
};

export function AttentionChapterBody() {
  return (
    <ChapterFrame learning={learning}>
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
        Imagine you're the model and someone hands you the prompt{" "}
        <code>"The cat sat on the ___"</code> and asks for the next word. To
        guess well you have to look back at the earlier tokens:{" "}
        <code>cat</code> tells you the subject is an animal,{" "}
        <code>sat</code> tells you it's resting on something,{" "}
        <code>on the</code> tells you a noun is coming next. Self-attention
        is a learned, differentiable version of that "look back": for every
        token in the sequence the model decides <em>how much</em> it should
        pay attention to every other token. The panel on the right runs
        exactly this prompt and shows you the attention pattern.
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
        keys <em>0 … i</em>. The bottom row — the last token of the prompt —
        is the only row that gets to see the whole sentence, which is why
        that row is the one that produces the next-word prediction.
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

      <h2>Working a tiny example by hand</h2>
      <p>
        The matrix algebra is easier to feel with concrete numbers. The
        widget below walks through two tokens — "The" and "cat" — through
        every step of the attention formula. All numbers are computed from
        fixed Q, K, V values; step through and watch the score matrix,
        scale, mask, softmax, and final output appear.
      </p>

      <ToyAttentionExample />

      <h2>The causal mask, cell by cell</h2>
      <p>
        A 5×5 picture of the mask makes "row <em>i</em> can only see
        columns <em>0..i</em>" concrete. Click any cell to see what it
        means.
      </p>

      <CausalMaskVisual />

      <p className="mt-6 text-muted-foreground">
        Hit <em>Run</em> on the right. The model's predicted next token
        appears at the end of the prompt with a rainbow shimmer, and the
        heatmap shows the real post-softmax attention scores from the
        forward pass that produced it. Edit the prompt or change layer /
        head to watch the pattern (and the prediction) change.
      </p>
      </Prose>
    </ChapterFrame>
  );
}

type RunStatus =
  | { kind: "ok" }
  | { kind: "error"; error: string }
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
  // The exact prompt that produced `run`. We compare against the current
  // textarea content to decide whether to ghost the predicted next token
  // onto the textarea — if the user edits, the ghost disappears because the
  // prediction is now stale for the new text.
  const [lastRunPrompt, setLastRunPrompt] = React.useState<string | null>(null);
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
  // Auto-fire one Run on mount. The app gates chapter rendering on
  // modelReady, so by the time this component mounts the worker is alive
  // and we can show real data immediately.
  const didAutoRunRef = React.useRef(false);
  // Ref to the heatmap container so we can restart its CSS flash animation
  // without remounting the AttentionHeatmap subtree — remounting would
  // reset the Layer/Head <Select>s back to their defaults on every run.
  const heatmapWrapRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (runFlash === 0) return;
    const el = heatmapWrapRef.current;
    if (!el) return;
    el.classList.remove("run-flash-on-mount");
    // Force a reflow so removing + re-adding the class actually restarts
    // the keyframe animation. Reading offsetWidth is the standard trick.
    void el.offsetWidth;
    el.classList.add("run-flash-on-mount");
  }, [runFlash]);

  async function handleRun() {
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      console.error(
        "[attention-demo] worker is unavailable — chapter view should have been gated on modelReady",
      );
      setStatus({ kind: "error", error: "Worker is unavailable. Reload the page." });
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
      setLastRunPrompt(prompt);
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
        console.error("[attention-demo] runForInspector failed", err);
        setStatus({ kind: "error", error: message });
      }
    } finally {
      setRunning(false);
    }
  }

  React.useEffect(() => {
    if (didAutoRunRef.current) return;
    didAutoRunRef.current = true;
    void handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-xs uppercase tracking-wider text-muted-foreground">
          Prompt
        </label>
        {/*
          Ghost-prediction overlay: a div behind the textarea mirrors the
          current prompt (invisibly, just for layout) and then renders the
          predicted next token at the end with an animated chromatic rainbow.
          The textarea sits on top with a transparent background so the user
          can still type / select / cursor as normal. The overlay's font /
          padding / line-height MUST exactly match the textarea so the ghost
          token lands at the caret position.

          We only show the ghost when the prompt is still exactly what the
          model just ran on (`prompt === lastRunPrompt`) — otherwise the
          prediction is stale and we hide it until the next Run.
        */}
        <div className="relative">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="relative z-[1] bg-transparent font-mono text-sm"
            placeholder="The cat sat on the"
          />
          {run && prompt === lastRunPrompt && !running ? (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 z-0 whitespace-pre-wrap break-words rounded-md border border-transparent px-3 py-2 font-mono text-sm text-transparent"
            >
              {prompt}
              <span className="rainbow-text-animated font-mono text-sm">
                {renderTokenDisplay(run.generatedToken.text)}
              </span>
            </div>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Generates one token, captures attention scores.
          </span>
          {lastRunAt !== null ? <LastRunPill timestamp={lastRunAt} /> : null}
        </div>
        {lastRunAt !== null ? (
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
      {status?.kind === "error" ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Inspector run failed.</strong> {status.error}
        </div>
      ) : null}
      {run ? (
        <div className="space-y-3 pt-2">
          <div
            ref={heatmapWrapRef}
            className="rounded-md"
          >
            <AttentionHeatmap run={run} />
          </div>
        </div>
      ) : status?.kind === "aborted" || status?.kind === "error" ? null : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to see the attention heatmap.
        </div>
      )}

      <DemoCallout
        items={[
          "The brightness of each cell is the attention weight — bright means \"this query is looking hard at that key\".",
          "The outlined bottom row is the prediction step: those bright cells are what the model focused on when picking the next token.",
          "Switch layer/head dropdowns to see different patterns — some heads track recency, some track syntax.",
        ]}
      />
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
