// Sub-chapter 10.1 — The logit lens: point the LM head at a middle layer.
//
// A "go deeper" deep-dive hung off chapter 10 (LM head & weight tying). It is
// the one-line generalization of that chapter's own mechanism: the tied
// unembedding W_U reads the vocabulary out of the FINAL hidden state — aim the
// same read at an INTERMEDIATE layer and you get the logit lens (nostalgebraist,
// 2020). Unlike the other sub-chapters this one embeds a LIVE widget
// (<LogitLensLive/>) that runs the plain logit lens on the in-browser
// Qwen3.5-0.8B. Honesty (binding): this page runs the PLAIN logit lens only —
// no Jacobian/tuned lens. The tuned lens is Belrose et al./EleutherAI (2023); the
// Jacobian lens is Anthropic's, on their own models; this page neither runs nor
// reproduces either fitted lens. The widget is SSG-safe
// (renders a static frame under prerender when no providers are mounted).

import { Prose } from '../Prose';
import { ChapterLink } from '../scaffolding/ChapterLink';
import { LogitLensLive } from '../widgets/jlens/LogitLensLive';

export function LmHeadLogitLensSection() {
  return (
    <Prose>
      <h1>The logit lens</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 10, LM head — take the one matmul this chapter just built and slide it down the stack: read
        the vocabulary out of every depth, not only the last.
      </p>
      <p>
        The <ChapterLink chapterId="lm-head">LM head</ChapterLink> reads the next-token distribution out of the{' '}
        <em>final</em> hidden state by multiplying it through the tied unembedding <code>W_U</code>. Nothing in that
        matmul actually requires the hidden state to be the last one. So here is the one-line question that opened a
        whole strand of interpretability: what if you aim the <em>same</em> <code>W_U</code> read at a <em>middle</em>{' '}
        layer? That is the <strong>logit lens</strong> — coined by <strong>nostalgebraist</strong> in 2020
        (&ldquo;interpreting GPT: the logit lens&rdquo;, LessWrong).
      </p>

      <h2>The idea: one read, every depth</h2>
      <p>
        A decoder LLM carries a single residual vector <code>h_ℓ</code> per token, refined boundary by boundary up the
        stack. The LM head only ever unembeds the <em>top</em> of that stack. The logit lens does the exact same thing at
        every boundary in between: take <code>h_ℓ</code>, push it straight through <code>W_U</code>, softmax, and read
        the top token — a crude &ldquo;what would the model say if it had to answer <em>right now</em>?&rdquo; at each
        depth. Line those reads up by layer and you can watch a prediction assemble itself.
      </p>

      <h2>See it live</h2>
      <p>
        The widget below runs precisely that on the Qwen3.5-0.8B loaded in your browser: one forward pass, then the tied{' '}
        <code>W_U</code> read at a dozen residual boundaries. Pick a short prompt and run it. Watch the per-layer top
        token climb out of gibberish in the middle of the stack into the real answer near the top — and watch a pinned
        answer token (say <code>Paris</code> for &ldquo;The capital of France is&rdquo;) climb the full-vocabulary ranks
        only in the late layers.
      </p>

      <LogitLensLive />

      <p>
        Two things are worth noticing. The middle rows are mostly noise — the model has not &ldquo;decided&rdquo; yet, so
        the top token there is often unrelated to the answer. And the answer does not fade in smoothly; it tends to snap
        into place in the last handful of layers. That is a recurring logit-lens observation: on this model the
        vocabulary-facing decision is made late, and the earlier layers are still doing something the unembedding cannot
        read cleanly.
      </p>

      <h2>Why it is only approximate — the blind spot</h2>
      <p>
        The logit lens is a <em>rough probe</em>, not ground truth, and it is worth being honest about why.{' '}
        <code>W_U</code> was trained to read exactly one distribution: the <em>final</em> layer&rsquo;s. A middle{' '}
        <code>h_ℓ</code> lives in a different region of the representation space — it is{' '}
        <strong>off-distribution</strong> for the unembedding — so reading it through <code>W_U</code> is a shortcut that
        can be badly wrong, and it is worst in the early layers. On a shallow 0.8B model the interpretable band is thin
        and lands late: those noisy mid-stack reads are the method&rsquo;s honest signature, not a defect in this demo.
      </p>

      <h2>The fix: a fitted lens (and what this page does not claim)</h2>
      <p>
        The blind spot has a fix: instead of the raw <code>W_U</code>, transport each <code>h_ℓ</code> through a small
        learned map first, so the read matches what that particular layer actually encodes. A <em>tuned lens</em> learns
        a per-layer affine probe (Belrose et al., EleutherAI, 2023); a <strong>Jacobian lens</strong> reads each{' '}
        <code>h_ℓ</code> through the network&rsquo;s own averaged Jacobian. That <em>Jacobian</em> lens is{' '}
        <strong>Anthropic&rsquo;s</strong>, developed and evaluated on <em>their</em> models, and it sharpens exactly the
        blurry middle layers you just watched refuse to resolve.
      </p>
      <p className="text-muted-foreground">
        To be clear about scope: this page runs the <strong>plain logit lens only</strong>. It does <em>not</em> run a
        Jacobian or tuned lens, and it makes no claim to reproduce Anthropic&rsquo;s results — it simply shows you, live,
        the blind spot that a fitted lens is built to close.
      </p>
    </Prose>
  );
}
