import * as React from 'react';

import type { AttentionRun } from '../../../src/inspector-types';
import { Textarea } from '../../components/ui/textarea';
import { runForInspector } from '../../lib/inspector-client';
import { DemoCallout } from '../inspector/DemoCallout';
import { TopKBars, renderTokenDisplay } from '../inspector/TopKBars';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { RunButton } from '../scaffolding/RunButton';
import { useRunFlash } from '../scaffolding/useRunFlash';
import { InnerProductLogit } from '../widgets/InnerProductLogit';
import { LmHeadWalkthrough } from '../widgets/LmHeadWalkthrough';
import { WeightTyingVisual } from '../widgets/WeightTyingVisual';

/**
 * Chapter 9 — LM head & weight tying.
 *
 * Bridges the gap between "last hidden state" (the top of the residual
 * stream, chapter 9) and "vector of logits" (the input to sampling, chapter
 * 11). The matmul itself is one line — but it's the line readers most often
 * skim past, and the *tying* trick that backs it is genuinely surprising.
 *
 * The right-hand panel runs the live `runForInspector` hook with
 * `logits: { topK: 12 }` for a fixed prompt and renders the actual top-K
 * logits via the shared `TopKBars`. The chapter body uses the two new
 * widgets (matmul animation + weight-tying diagram).
 */

const DEFAULT_PROMPT = 'The cat sat on the';
const TOP_K = 12;

export const learning: ChapterLearningData = {
  chapterId: 'lm-head',
  objective:
    'Explain how the last hidden state becomes a vector of logits, and why for Qwen3.5 the LM head is literally the embedding matrix re-used.',
  problem:
    'Chapter 9 ends with a hidden state; chapter 11 starts with logits. Something has to turn one into the other.',
  minutes: 6,
  glossary: [
    {
      term: 'LM head',
      definition:
        'The final linear projection of the model. Multiplies the last hidden state by a [d, V] matrix to get one logit per vocab token.',
    },
    {
      term: 'logit',
      definition:
        "One unbounded real-valued score per vocab entry — higher means the model favors that token more. They aren't probabilities yet; the sampling chapter turns them into probabilities with softmax.",
    },
    {
      term: 'weight tying',
      definition:
        'Reusing embed_tokens.weight as the LM head, so the input lookup and output projection share the same tensor (saves a large fraction of params on small models).',
    },
    {
      term: 'vocabulary size (V)',
      definition: 'Number of distinct tokens the tokenizer can emit. Qwen3.5 uses V = 248,320.',
    },
    {
      term: 'hidden dim (d)',
      definition:
        'Width of the residual stream. Qwen3.5-0.8B uses d = 1024 — every token carries 1024 floats through every layer.',
    },
    {
      term: 'inner product',
      definition:
        "Sum of element-wise products. logit_j = <last_hidden, W[j, :]>: the model's score for token j is how well the hidden state aligns with that token's row of the LM head.",
    },
  ],
  takeaways: [
    'The LM head is just one matmul: last_hidden @ embed_tokens.T → logits. Every vocab entry gets one score.',
    'Each row of the weight matrix is a learned "fingerprint" for one vocab token; the highest logit is the one most aligned with the hidden state.',
    'Qwen3.5 ties embed_tokens.weight with lm_head.weight — the same ~254M-float tensor is used at the input lookup AND the output projection.',
  ],
  exercise: {
    prompt:
      "Hit Run on 'The cat sat on the' and look at the top-K logits panel. The highest bar is the model's pick. Now find a token in the top-K whose text doesn't look 'like a real continuation' to you (e.g. ' floor' on a tiny model). What does its presence in the top-K tell you about the LM head's geometry?",
    answer:
      "Even tokens you'd consider 'unlikely' end up in the top-K because their row vectors in the LM head point in roughly the same direction as the hidden state. The model's representation of 'what should come after `the`' is broad — anything sittable, nameable, or position-of-an-object-like is geometrically close. Sampling temperature and top-p are what trim this broad cloud into one choice.",
  },
  quiz: [
    {
      id: 'q1-shape',
      prompt: 'What is the shape of the LM head matrix for Qwen3.5-0.8B?',
      options: [
        { id: 'a', label: '[1024, 1024] — one hidden vector per token.' },
        { id: 'b', label: '[248320, 1024] — one row per vocab token, of width equal to the hidden dim.' },
        { id: 'c', label: '[248320, 248320] — a vocab × vocab transition matrix.' },
      ],
      correctId: 'b',
      explanation:
        'The matrix has one row per vocab entry and one column per hidden dimension; the matmul against a [1, 1024] hidden state produces a [1, 248320] logit vector.',
    },
    {
      id: 'q2-tying',
      prompt: "What does 'tie_word_embeddings = true' mean in Qwen3.5's config?",
      options: [
        {
          id: 'a',
          label: 'The embedding values are clamped to [-1, 1] for stability.',
        },
        {
          id: 'b',
          label:
            'The same tensor is used for both the input embedding lookup and the output LM head — no separate lm_head.weight is allocated.',
        },
        {
          id: 'c',
          label: 'The model can only emit tokens it was previously shown.',
        },
      ],
      correctId: 'b',
      explanation:
        'With tying, the forward pass reads `embed_tokens.weight` for token-id → vector at the bottom, then re-uses its transpose at the top for vector → logits. Same ~254M floats, used twice.',
    },
    {
      id: 'q3-direction',
      prompt: "Which statement about the LM head's rows is true?",
      options: [
        {
          id: 'a',
          label: 'Each row is the average hidden state of every prompt that ever produced that token.',
        },
        {
          id: 'b',
          label:
            "Each row is a learned vector — token j's logit is the inner product of that row with the last hidden state.",
        },
        {
          id: 'c',
          label: 'Each row is a one-hot vector for one vocab id.',
        },
      ],
      correctId: 'b',
      explanation:
        'Think of each row as a learned "direction in hidden space" associated with one vocab token. The logit is just how well the current hidden state lines up with that direction.',
    },
  ],
};

export function LmHeadChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>The LM head: hidden state → logits</h1>
        <p>
          Chapter 9 left us at the top of the residual stream: a single hidden vector of width <code>d = 1024</code> per
          token, after the final RMSNorm. Chapter 11 (sampling) will start from a vector of <strong>logits</strong> —
          one real-valued score per vocab token. The step that connects them is one matmul, and it has a name:{' '}
          <strong>the LM head</strong>.
        </p>

        <h2>One line of math</h2>
        <MathDisplay latex={String.raw`\text{logits} = h_{\text{last}} \cdot W_{\text{lm}}^\top`} />
        <p>
          <code>h_last</code> is the last token's hidden state — shape <code>[1, 1024]</code> at decode time.{' '}
          <code>W_lm</code> is the LM head weight matrix — shape <code>[V, d] = [248,320, 1024]</code> for Qwen3.5-0.8B.
          The transpose makes it <code>[1024, 248,320]</code>; multiplying gives an output of shape{' '}
          <code>[1, 248,320]</code> — one logit per token in the vocabulary.
        </p>
        <p>
          The animation walks the matmul cell by cell. The scan beam highlights one output column at a time, with the
          matrix column that produces it lit up beside it. Every output entry is one independent inner product — which
          is also why this op parallelizes so cleanly on a GPU. Note the animation draws the <em>transposed</em> matrix{' '}
          <code>embed_tokens.weight.T = [d, V]</code>, so token <code>j</code>'s fingerprint is a{' '}
          <strong>column</strong> there — the same vector as <strong>row j</strong> of <code>W_lm = [V, d]</code>{' '}
          discussed below (a row of a matrix is a column of its transpose).
        </p>

        <LmHeadWalkthrough />

        <h2>
          What each output cell <em>means</em>
        </h2>
        <p>
          Think of each <strong>row</strong> of <code>W_lm</code> as a learned "fingerprint" for one vocab token. The
          logit for token <code>j</code> is the inner product of <code>h_last</code> with that row:
        </p>
        <MathDisplay latex={String.raw`\text{logit}_j = \langle h_{\text{last}},\, W_{\text{lm}}[j, :] \rangle`} />
        <p>
          So <em>high logit ⇔ hidden state points in roughly the same direction as that token's row</em>. The geometry
          of the hidden state is what determines the next-token distribution, and the LM head's rows are the dictionary
          the model uses to read out that geometry.
        </p>
        <p>
          The inner product unpacks as <code>h · w = |h|·|w|·cos θ</code>, where <code>cos θ</code> is the{' '}
          <em>cosine</em> of the angle between the two vectors — it runs from <code>+1</code> when they point the same
          way, through <code>0</code> at a right angle, to <code>−1</code> when they point opposite ways. With every
          token's row at a comparable length, the logit is really just a readout of the <em>angle</em> between the
          hidden state and that row. Toggle the three cases below — aligned, orthogonal, opposed — and watch the
          per-dimension products (which sum to the logit) and the cosine move together.
        </p>

        <InnerProductLogit />

        <p>
          This is why you'll sometimes see plausible-looking tokens you didn't expect at the top of the top-K: their
          rows happen to be aligned with the hidden state, even if the model wouldn't end up sampling them. The top-K
          panel on the right will show this concretely once you hit Run.
        </p>

        <h2>Weight tying: the same matrix, used twice</h2>
        <p>
          Here's the surprising part. The matrix <code>W_lm</code> in the formula above isn't a separately-learned
          tensor. For Qwen3.5 (and most modern decoder LLMs sub-7B), it is <em>literally</em> the embedding matrix from
          chapter 3 — the same <code>[248,320, 1024]</code> grid of floats that mapped token ids to vectors at the
          bottom of the stack is reused (transposed) at the top.
        </p>

        <WeightTyingVisual />

        <p>
          This is called <strong>weight tying</strong>, controlled by <code>tie_word_embeddings = true</code> in the
          config. There's a clean intuition behind it: the same "alphabet" the model uses to read tokens in should also
          be the alphabet it uses to write them out. If row <code>j</code> of <code>embed_tokens.weight</code> is what{' '}
          <em>"the"</em> looks like as a hidden vector, then the LM head should give <em>"the"</em> a high score
          whenever the residual stream looks like that row — and <code>row_j · h</code> is exactly that test.
        </p>
        <p>
          The practical win: a sub-billion-parameter model saves ~254M parameters (close to a third of its total) by not
          duplicating the matrix. The trade-off: a small quality loss at very large scale, which is why GPT-style models
          beyond a few billion parameters sometimes <em>untie</em> the head. For Qwen3.5-0.8B, the savings dominate.
        </p>

        <h2>What you'll see on the right</h2>
        <p>
          The panel runs one inspector call: tokenize the prompt, run a forward pass, capture the top-K logits at the
          last position, render them as bars. Each row is one of the <code>{TOP_K}</code> highest-logit vocab tokens;
          the row highlighted with the primary accent is the token the model actually sampled (greedy at this stage —
          chapter 11 will introduce temperature and top-p).
        </p>
        <p className="text-muted-foreground">
          <em>What to notice:</em> the panel runs softmax over those logits so the bar <em>heights</em> read as
          probabilities (each panel's bars sum to 1). But the raw scores the LM head emits are <em>logits</em> —
          arbitrary real numbers that can be negative and whose magnitudes aren't comparable across prompts. Chapter 11
          is where temperature and top-p reshape that softmax into the model's actual choice.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

// ----------------------------------------------------------------------------
// Live demo: top-K logits for the last position.
// ----------------------------------------------------------------------------

export type LmHeadDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus = { kind: 'ok' } | { kind: 'error'; error: string } | { kind: 'aborted' } | { kind: 'empty-prompt' };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

function softmax(values: Float32Array): number[] {
  if (values.length === 0) return [];
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i]! > max) max = values[i]!;
  const exps = new Array<number>(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const e = Math.exp(values[i]! - max);
    exps[i] = e;
    sum += e;
  }
  if (sum === 0) return exps.map(() => 1 / exps.length);
  return exps.map((e) => e / sum);
}

export function LmHeadDemo({ workerRef, abortRef }: LmHeadDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  // Bumped after every successful run so `<TopKBars runKey>` remounts its bar
  // fills and re-fires the grow-in animation, even when the prompt — and
  // therefore the deterministic greedy output — is unchanged.
  const [runKey, setRunKey] = React.useState(0);
  const resultFlashRef = useRunFlash(running);

  const didAutoRunRef = React.useRef(false);
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
      setStatus({ kind: 'error', error: 'Worker is unavailable. Reload the page.' });
      setRunning(false);
      return;
    }

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;

    const appSignal = abortRef.current?.signal;
    if (appSignal?.aborted) ctrl.abort();
    const onAppAbort = () => ctrl.abort();
    if (appSignal && !appSignal.aborted) appSignal.addEventListener('abort', onAppAbort, { once: true });

    try {
      const result = await runForInspector(
        worker,
        { prompt, logits: { topK: TOP_K }, maxNewTokens: 1 },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;
      setRun(result);
      setRunKey((k) => k + 1);
      setStatus({ kind: 'ok' });
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        if (appSignal?.aborted) setStatus({ kind: 'aborted' });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setStatus({ kind: 'error', error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      setRunning(false);
    }
  }

  React.useEffect(() => {
    if (didAutoRunRef.current) return;
    didAutoRunRef.current = true;
    void handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const step = run?.logits?.[0];
  const probs = step ? softmax(step.topKLogits) : [];

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-xs uppercase tracking-wider text-muted-foreground">Prompt</label>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="font-mono text-sm"
          placeholder="The cat sat on the"
        />
        <div className="flex flex-wrap items-center gap-2">
          <RunButton onClick={handleRun} running={running} />
          <span className="text-xs text-muted-foreground">
            Captures top-{TOP_K} logits for the next-token position.
          </span>
        </div>
      </div>

      {status?.kind === 'aborted' ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Inspector run cancelled. Click <strong>Run</strong> to retry.
        </div>
      ) : null}
      {status?.kind === 'empty-prompt' ? (
        <div role="alert" className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs">
          Enter at least one character of prompt to run.
        </div>
      ) : null}
      {status?.kind === 'error' ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Inspector run failed.</strong> {status.error}
        </div>
      ) : null}

      {step ? (
        <div ref={resultFlashRef} className="space-y-2 rounded-md">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">
            Top-{TOP_K} next-token candidates · probabilities (softmax over logits)
          </div>
          <TopKBars
            ids={step.topKIds}
            probs={probs}
            texts={step.topKTexts}
            sampledTokenId={step.tokenId}
            runKey={runKey}
          />
          <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-[12px] text-foreground/85">
            The highlighted row is what greedy decoding picks —{' '}
            <span className="font-mono">{renderTokenDisplay(run!.generatedToken.text)}</span>. Each bar is the softmax
            of one logit; the logits themselves are arbitrary real numbers (some negative), and it's only after softmax
            that they become probabilities.
          </div>
        </div>
      ) : status?.kind === 'aborted' || status?.kind === 'error' ? null : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to see the top-K logits.
        </div>
      )}

      <DemoCallout
        items={[
          'Each row is one of the highest-scoring vocab tokens at the last position.',
          'The bar length is its probability after softmax — every set of bars in one panel sums to 1.',
          "The accented row is the model's greedy pick. Chapter 11 will let you slide temperature and top-p to change that.",
        ]}
      />
    </div>
  );
}
