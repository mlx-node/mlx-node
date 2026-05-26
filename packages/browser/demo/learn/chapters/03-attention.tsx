import * as React from "react";

import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import type { AttentionRun } from "../../../src/inspector-types";
import { AttentionHeatmap } from "../inspector/AttentionHeatmap";
import { Prose } from "../Prose";
import { makeMockAttentionRun } from "../mock-data";

/**
 * Chapter 3 — Self-attention.
 *
 * This is the only fully-authored chapter for the first cut. The prose
 * teaches the mechanism end-to-end; the right-hand panel runs a (currently
 * mocked) inspector and renders the resulting attention heatmap.
 *
 * When the backend `runForInspector` hook lands, replace `makeMockAttentionRun`
 * with a real worker call in <AttentionDemo/>.
 */

const DEFAULT_PROMPT = "The cat sat on the mat.";

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
        Type a sentence on the right and hit <em>Run</em>. The heatmap shows a
        synthetic but realistic-looking attention pattern from a mocked
        inspector — the live model hook is coming in a separate task. Switch
        the layer and head selectors to see how the pattern changes.
      </p>
    </Prose>
  );
}

export function AttentionDemo() {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [running, setRunning] = React.useState(false);

  function handleRun() {
    setRunning(true);
    // Pretend this is async to mirror what the real backend call will look
    // like, but resolve immediately for the mock.
    Promise.resolve()
      .then(() => makeMockAttentionRun(prompt))
      .then((result) => {
        // Backend wire-up is being built in parallel — log a reminder for the
        // dev so it's obvious we're on mock data.
        console.log(
          "[attention-demo] using mock AttentionRun; real backend wire-up pending",
          { promptLength: prompt.length, seqLen: result.tokens.length },
        );
        setRun(result);
      })
      .finally(() => setRunning(false));
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
        <div className="flex items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Generates one token, captures attention scores.
          </span>
        </div>
      </div>

      {run ? (
        <div className="pt-2">
          <AttentionHeatmap run={run} />
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to see the attention heatmap.
        </div>
      )}
    </div>
  );
}
