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
import { MathDisplay } from '../scaffolding/MathDisplay';
import { JacobianLensLive } from '../widgets/jlens/JacobianLensLive';
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

// Sub-chapter 10.2 — The Jacobian lens: transport each middle h_ℓ into the final
// layer's frame before the tied W_U reads it, so the blurry mid-stack the logit
// lens could not resolve becomes legible. Unlike the logit-lens page, this one
// runs a FITTED lens live. Honesty (binding, reviewed): this page runs OUR fit —
// J_1..J_23 fit on 100 WikiText prompts on THIS exact Qwen3.5-0.8B (~11 h). The
// METHOD is Anthropic's; the fit + every number on this page are OURS on a small
// model. Anthropic's frontier results are framed "what Anthropic saw on Claude"
// WITH citation, never claimed as ours. Eval numbers below are read verbatim from
// eval-results-v1.json (jAucHead/logitAucHead per suite, jWins6=6); band numbers
// from band-report.json.
export function LmHeadJacobianLensSection() {
  return (
    <Prose>
      <h1>The Jacobian lens</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 10, LM head — the fix for the blurry middle layers the logit lens could not read. And unlike
        that page, this one runs a <em>fitted</em> lens live on your in-browser Qwen3.5-0.8B.
      </p>
      <p>
        The <ChapterLink chapterId="lm-head">logit lens</ChapterLink> ended on an honest blind spot: a middle{' '}
        <code>h_ℓ</code> is <strong>off-distribution</strong> for the tied unembedding <code>W_U</code>, so pushing it
        straight through <code>W_U</code> gives noise in the mid-stack — exactly where you most want to read. The fix is
        to stop reading <code>h_ℓ</code> raw and first <em>transport</em> it into the frame the unembedding was trained
        for. That is the <strong>Jacobian lens</strong>, and unlike the logit-lens page, this one runs the fitted lens
        live: toggle it on and watch the same layers that refused to resolve become legible.
      </p>

      <h2>What a Jacobian lens is</h2>
      <p>
        Keep the LM head&rsquo;s one matmul, but slip a per-layer map <code>J_ℓ</code> in front of it:
      </p>
      <MathDisplay latex={String.raw`\operatorname{lens}(h_\ell) = \operatorname{softmax}\!\left(W_U \cdot \operatorname{norm}(J_\ell \, h_\ell)\right)`} />
      <p>
        <code>J_ℓ</code> is a per-layer, corpus-<em>averaged</em> first-order (Jacobian) map of how a small nudge at
        layer <code>ℓ</code> moves the <em>final</em> residual. Applying it transports <code>h_ℓ</code> into the final
        layer&rsquo;s frame <em>before</em> the tied unembedding reads it — so the read matches what the top of the stack
        expects, not what layer <code>ℓ</code> raw happens to look like. Two neighbours put it in context: the{' '}
        <strong>logit lens</strong> is this formula with <MathDisplay inline latex={String.raw`J_\ell = I`} /> (read the
        residual raw); a <strong>tuned lens</strong> replaces <code>J_ℓ</code> with a per-layer affine probe learned
        end-to-end (Belrose et al., EleutherAI, 2023).
      </p>

      <h2>Not a tuned lens — a different target</h2>
      <p>
        It is worth being precise about why this is not just a tuned lens with extra steps. A tuned lens is optimized so
        its readout <em>agrees with the next token</em> — it minimizes KL to the model&rsquo;s own output distribution.
        The Jacobian lens optimizes nothing of the sort: it reads what a layer <em>actually encodes</em> by transporting
        it through the network&rsquo;s own averaged Jacobian. Best next-token agreement and best intermediate readout are
        <em> different goals</em>, and when they disagree that is a feature, not a defect — the two lenses are answering
        two different questions.
      </p>

      <h2>What we fit — and what we did not</h2>
      <p>
        Be blunt about provenance. We fit the 23 maps <code>J_1..J_23</code> on <strong>100 WikiText prompts</strong>{' '}
        (wikitext-103-raw-v1, raw text, no chat template) on <em>this exact</em> Qwen3.5-0.8B — about{' '}
        <strong>11 hours</strong> on a laptop — with the fit target set to the final residual taken{' '}
        <em>before</em> the final norm. The output boundary <code>ℓ24</code> is left as{' '}
        <MathDisplay inline latex={String.raw`J = I`} /> by construction, so the fitted lens only ever acts mid-stack.
        The <strong>method is Anthropic&rsquo;s</strong>; the fit and every number on this page are{' '}
        <strong>ours</strong>, on a small open model. The fit script and the vendored evaluation suites live in the repo
        (<code>packages/browser/scripts/jlens/</code>), Apache-2.0, so the whole thing is reproducible end to end.
      </p>

      <h2>See it live</h2>
      <p>
        Toggle <strong>LOGIT | JACOBIAN</strong> on a curated prompt. The baked frame renders instantly with no
        download; &ldquo;compute live on your device&rdquo; recomputes it on the in-browser model (and, for JACOBIAN,
        loads the fitted pack once). Watch a concept the logit lens keeps at rank 999+ snap into the top handful of ranks
        in the middle of the stack under the fitted <code>J</code>.
      </p>

      <JacobianLensLive />

      <h2>Does the fitted lens actually help?</h2>
      <p>
        A live demo can cherry-pick. So here is the aggregate, on our six vendored evaluation suites (the paper&rsquo;s
        §methods-comparison prompts). The headline metric is a normalized log-<code>k</code> pass@<code>k</code>{' '}
        <strong>AUC</strong> — higher means the target/intermediate becomes legible <em>earlier and at more depths</em> —
        with the rank taken as the <em>min over the fitted domain</em> <code>ℓ1..23</code>. On this run the Jacobian lens
        beats the logit lens on <strong>6 of 6</strong> suites:
      </p>

      <div className="not-prose my-5 overflow-x-auto rounded-md border border-border bg-background">
        <table className="w-full border-collapse text-[13px]">
          <caption className="px-3 py-2 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
            J-lens vs logit-lens · headline AUC (min over ℓ1..23) · from eval-results-v1.json
          </caption>
          <thead>
            <tr className="border-y border-border text-[11px] uppercase tracking-wider text-muted-foreground">
              <th className="px-3 py-2 text-left font-medium">Suite</th>
              <th className="px-3 py-2 text-right font-medium">J-lens AUC</th>
              <th className="px-3 py-2 text-right font-medium">logit-lens AUC</th>
              <th className="px-3 py-2 text-right font-medium">J wins</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {[
              { suite: 'typo', j: '0.781', logit: '0.432' },
              { suite: 'order-ops', j: '0.638', logit: '0.242' },
              { suite: 'multihop', j: '0.530', logit: '0.262' },
              { suite: 'multilingual', j: '0.484', logit: '0.265' },
              { suite: 'association', j: '0.053', logit: '0.003' },
              { suite: 'poetry', j: '0.039', logit: '0.031' },
            ].map((r) => (
              <tr key={r.suite} className="border-b border-border/50 last:border-b-0">
                <td className="px-3 py-1.5 text-left text-foreground/90">{r.suite}</td>
                <td className="px-3 py-1.5 text-right font-semibold text-foreground">{r.j}</td>
                <td className="px-3 py-1.5 text-right text-muted-foreground">{r.logit}</td>
                <td className="px-3 py-1.5 text-right text-primary">✓</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        Read this honestly. A higher AUC does not mean &ldquo;the model got it right&rdquo; — it means the answer or an
        intermediate concept becomes <em>readable</em> earlier and across more layers under the fitted lens. And the two
        hardest suites, <strong>poetry</strong> and <strong>association</strong>, sit near the floor for{' '}
        <em>both</em> lenses (J-AUC 0.039 and 0.053; logit-AUC 0.031 and 0.003): the Jacobian lens still wins, but on
        tasks this hard for a 0.8B, neither lens reads much. Reproduce the whole table with{' '}
        <code>JLENS_PACK=lens-pack-v1.safetensors JLENS_OUT=eval-results-v1.json oxnode packages/browser/scripts/jlens/eval.mts</code>.
      </p>

      <h2>Where the gain lives</h2>
      <p>
        The advantage is not spread evenly up the stack — it is a <strong>mid-stack band</strong>. Four proxy detectors
        of band structure all fire (4/4), and the one that measures it directly — the fraction of evaluation
        intermediates where the J-lens rank beats the logit rank at a given boundary — rises out of the early band around{' '}
        <code>ℓ6-7</code> and <strong>peaks at ℓ17</strong> (fraction <strong>0.595</strong> vs an early-band fraction of{' '}
        <strong>0.158</strong>). Note what the headline AUC is and is not: it is a <em>min over the whole fitted domain</em>{' '}
        <code>ℓ1..23</code>, not a band-restricted score — the band story explains <em>where</em> the min tends to come
        from, it does not change how the number is computed.
      </p>
      <p>
        The French headline prompt in the widget is the qualitative face of this: for{' '}
        <code>La saison après l&rsquo;été est l&rsquo;</code>, the abstract concepts <code>season</code> and{' '}
        <code>summer</code> surface near ranks 1–2 around boundaries 16–17 under the fitted <code>J</code>, while the
        plain logit lens keeps them pinned at rank 999+ across the same depths.
      </p>

      <h2>What Anthropic saw at frontier scale</h2>
      <blockquote>
        On <em>their own</em> models (Claude), Anthropic report the Jacobian lens surfacing rich, verbalizable
        intermediate content — with a workspace band roughly <code>k~25</code> layers wide and readouts stable to within
        ≤10% variance. Those figures are <strong>Sonnet-4.5 numbers</strong>, on Anthropic&rsquo;s models, in their paper{' '}
        <em>Verbalizable Representations Form a Global Workspace in Language Models</em> (transformer-circuits.pub, 2026).
        We neither run nor reproduce their causal, swap, or tuned-lens experiments. Whether a 0.8B model is anywhere near
        as rich is <strong>explicitly unknown</strong> — our 6/6 above is a readout-only partial reproduction on a small
        open model, not a claim about theirs.
      </blockquote>

      <div className="not-prose my-5 space-y-2 rounded-md border border-border bg-muted/30 p-4 text-[13px] text-foreground/85">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Read the readouts honestly
        </div>
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            Every rank track follows a <strong>single-token surface form</strong> (the leading-space token); a concept
            split across tokens is only tracked by its first piece.
          </li>
          <li>
            A readout is a <strong>bag of concepts with no binding</strong> — it tells you a direction is present, not how
            it is combined or bound to a role. Expect readouts that resist a clean interpretation.
          </li>
          <li>
            The first ~third of the stack is <strong>noisy</strong> for both lenses; the default view hides{' '}
            <code>ℓ1..5</code> for that reason.
          </li>
          <li>
            The output boundary <code>ℓ24</code> is <MathDisplay inline latex={String.raw`J = I`} /> by construction, so
            its Jacobian and logit reads are identical — the fitted map only acts mid-stack.
          </li>
          <li>
            The <code>k~25</code> width and ≤10% variance are <strong>Sonnet-4.5&rsquo;s, not ours</strong>; whether a
            0.8B carries comparable structure is unknown.
          </li>
          <li>
            Our fit uses <strong>100 WikiText prompts and no chat template</strong>; it is a small, honest fit, not a
            production lens.
          </li>
        </ul>
        <p className="text-[11px] text-muted-foreground/80">
          The fit recipe and the six vendored evaluation suites are Apache-2.0 (upstream:{' '}
          <code>anthropics/jacobian-lens</code>); see <code>packages/browser/scripts/jlens/data/NOTICE</code>.
        </p>
      </div>
    </Prose>
  );
}
