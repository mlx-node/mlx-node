// Sub-chapter 14.1 — Scaling laws & compute-optimal training.
//
// A "go deeper" deep-dive hung off chapter 14 (Scaling & regularization). Pure,
// model-free reading prose — SSR-safe, so the prerenderer renders it straight
// to crawlable static HTML (see scripts/prerender.entry.tsx). Wrapped in
// <Prose> by the section route / prerenderer, so it returns just the article
// content.
//
// The chapter body (chapters/13-scaling.tsx) already has a ScaleLadder widget
// comparing raw PARAMETER COUNTS on a log axis, plus AdamW / LR schedule /
// gradient-clipping engineering content. This sub-chapter adds the missing
// axis — how many TOKENS a model of a given size should train on — and does
// NOT repeat any of the above.
//
// Every claim below is grounded in a primary source, verified this session:
//   - Kaplan et al. 2020, arXiv:2001.08361 ("Scaling Laws for Neural Language
//     Models"). Power laws L(N), L(D), L(C_min) with alpha_N ~ 0.076,
//     alpha_D ~ 0.095, alpha_Cmin ~ 0.050 (verified from the ar5iv full text,
//     not just the abstract). Their optimal-compute-allocation conclusion:
//     N ∝ C^0.73 — grow the model much faster than the data.
//   - Hoffmann et al. 2022, arXiv:2203.15556 ("Training Compute-Optimal Large
//     Language Models" — Chinchilla). Abstract, verbatim: "for compute-optimal
//     training, the model size and the number of training tokens should be
//     scaled equally: for every doubling of model size the number of training
//     tokens should also be doubled." Chinchilla itself: 70B params / 1.4T
//     tokens, compute-matched against Gopher (280B params / 300B tokens).
//     1.4e12 / 70e9 = 20 exactly — the "~20 tokens/param" ratio is a
//     self-computed derivation from the paper's own numbers, not a literally
//     quoted sentence in the paper.
//   - Meta, "The Llama 3 Herd of Models," arXiv:2407.21783 (2024-07). 405B
//     flagship pretrained on 15.6T tokens at 3.8e25 FLOPs; the paper's own
//     scaling-law fit predicted an optimal 402B params / 16.55T tokens for
//     that budget — the flagship landed close to its own optimum. Direct
//     quote on the smaller models: "we also train our smaller models for much
//     longer than is compute-optimal. The resulting models perform better
//     than compute-optimal models at the same inference budget." Llama 3 8B:
//     ~15T tokens / 8e9 params = 1,875 tokens/param (self-computed), ≈94× the
//     Chinchilla-optimal ~20:1 ratio.
//   - Honesty: no primary source discloses Qwen3.5-0.8B's own pretraining
//     token count. The only verified Qwen-family figure is the (different,
//     earlier) Qwen3 Technical Report (arXiv:2505.09388): ~36T tokens total
//     pretraining corpus — not extrapolated onto this model here.

import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterLink } from '../scaffolding/ChapterLink';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { TokensPerParamScatter } from '../widgets/TokensPerParamScatter';

/** Sub-chapter 14.1 — neural scaling laws and compute-optimal training. */
export function ScalingLawsSection() {
  return (
    <Prose>
      <h1>Scaling laws: how many tokens for how many parameters?</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 14, Scaling &amp; regularization — the chapter&apos;s ladder counted parameters; this is
        the other half of the compute budget: how much data those parameters should actually see.
      </p>

      <h2>The missing axis</h2>
      <p>
        The parameter ladder earlier in this chapter put this model, GPT-2 XL, and GPT-3 on one log axis of raw{' '}
        <strong>parameter count</strong>. That answers &ldquo;how big is the model,&rdquo; but a model in isolation is
        not a recipe — training it also means choosing <strong>how many tokens</strong> to run it over. Get that
        second number wrong and a bigger model can end up <em>worse</em> than a smaller one trained properly, for the
        same compute bill. Two research groups, two years apart, gave opposite answers to &ldquo;how many tokens,
        given a fixed compute budget?&rdquo; — and the correction between them is one of the most consequential
        results in LLM training history.
      </p>

      <h2>Kaplan 2020: grow the model, not the data</h2>
      <p>
        In 2020, OpenAI&apos;s <strong>Kaplan et al.</strong> (&ldquo;Scaling Laws for Neural Language Models,&rdquo;
        arXiv:2001.08361) ran a huge sweep of models and found that loss falls off as a clean power law in each of
        three things taken separately — parameters <code>N</code>, training tokens <code>D</code>, and compute{' '}
        <code>C</code>:
      </p>
      <MathDisplay
        latex={String.raw`L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\quad \alpha_N \approx 0.076 \qquad\qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D},\quad \alpha_D \approx 0.095`}
      />
      <p>
        Bigger exponent means the loss is more sensitive to that axis, so at first glance data (<code>0.095</code>)
        looks like the better lever than parameters (<code>0.076</code>). But that is not how Kaplan et al. read
        their own result. Fitting a third curve — loss versus <em>compute</em>, <code>C</code> — and asking how to
        split a fixed compute budget between a bigger model and more data, they found the optimal split is lopsided:
      </p>
      <MathDisplay latex={String.raw`N_{\text{optimal}} \propto C^{0.73}`} />
      <p>
        For every <MathDisplay inline latex={String.raw`10\times`} /> increase in compute, model size should grow by{' '}
        <MathDisplay inline latex={String.raw`10^{0.73} \approx 5.4\times`} />, while the data (and the number of
        training steps) grows far more modestly — the leftover exponent is only <code>0.27</code>. Their own words for
        the takeaway: <strong>train very large models on a relatively modest amount of data, and stop well before
        convergence</strong>. Given a fixed compute budget, undertrain a big model rather than fully train a small
        one. This was the field&apos;s working assumption for nearly two years — it is a large part of why models
        like GPT-3 (175B parameters) were trained on data corpora that, in hindsight, were too small for their size.
      </p>

      <h2>Chinchilla 2022: grow them equally</h2>
      <p>
        In 2022, DeepMind&apos;s <strong>Hoffmann et al.</strong> (&ldquo;Training Compute-Optimal Large Language
        Models,&rdquo; arXiv:2203.15556 — the paper that introduced <strong>Chinchilla</strong>) reran the experiment
        at far larger scale: 400+ models, from 70M to 16B parameters, on 5B to 500B tokens. Their correction to
        Kaplan&apos;s conclusion, quoted directly from the abstract:
      </p>
      <blockquote>
        &ldquo;for compute-optimal training, the model size and the number of training tokens should be scaled
        equally: for every doubling of model size the number of training tokens should also be doubled.&rdquo;
      </blockquote>
      <p>
        In symbols, both grow at the <em>same</em> rate with compute — <code>N ∝ C^0.5</code> and{' '}
        <code>D ∝ C^0.5</code> — not Kaplan&apos;s model-heavy <code>C^0.73</code> / <code>C^0.27</code> split. To
        prove the point, they built <strong>Chinchilla</strong>: 70B parameters trained on 1.4 trillion tokens, using
        the exact same training compute as DeepMind&apos;s own earlier <strong>Gopher</strong> model — 280B
        parameters, but only 300B tokens. Same compute bill, radically different split. Chinchilla won, decisively —
        for instance scoring 67.5% average on MMLU versus Gopher&apos;s lower score at four times the parameter
        count. Bigger was not better; better-fed was better.
      </p>
      <p>
        Do the arithmetic on Chinchilla&apos;s own headline numbers and a famous ratio falls out:
      </p>
      <MathDisplay latex={String.raw`\frac{1.4 \times 10^{12} \text{ tokens}}{70 \times 10^{9} \text{ params}} = 20 \text{ tokens per parameter}`} />
      <p>
        That is not a sentence quoted from the paper — it is the ratio implied by Chinchilla&apos;s own reported
        70B/1.4T figures, and it is how the result is near-universally cited: <strong>train on roughly 20 tokens for
        every parameter</strong>. Gopher, for comparison, sat at only <code>300e9 / 280e9 ≈ 1.07</code>{' '}
        tokens/parameter — badly under-fed relative to its size, exactly Kaplan&apos;s prescription and exactly what
        Chinchilla showed was a mistake.
      </p>

      <h2>Doing the arithmetic for this course&apos;s model</h2>
      <p>
        This model has <strong>{(852_985_920).toLocaleString('en-US')}</strong> parameters. If it were trained at the
        Chinchilla-optimal ~20:1 ratio, the token budget would be:
      </p>
      <MathDisplay
        latex={String.raw`852{,}985{,}920 \times 20 \approx 17.06 \times 10^{9} \text{ tokens} \approx 17.06\text{B tokens}`}
      />
      <p className="text-muted-foreground">
        Read that number carefully: it is <strong>this section&apos;s own multiplication</strong>, not a disclosed
        fact about this checkpoint. No primary source — not a Qwen blog post, not a technical report, not this
        repo&apos;s config — states how many tokens Qwen3.5-0.8B was actually pretrained on. The only concrete,
        verifiable token count anywhere in the Qwen family is for a <em>different, earlier</em> model generation: the
        Qwen3 Technical Report (arXiv:2505.09388) states the (much larger, non-hybrid) Qwen3 family was pretrained on
        roughly <strong>36 trillion tokens</strong>. That figure belongs to a different model and cannot be assumed to
        transfer to this 0.8B hybrid checkpoint — it is shown here only so you know what <em>is</em> and{' '}
        <em>is not</em> public.
      </p>

      <h2>Real models overtrain on purpose — Llama 3</h2>
      <p>
        Chinchilla answers &ldquo;what minimizes training compute for a target loss.&rdquo; It does not answer
        &ldquo;what minimizes cost once the model is deployed and answering millions of queries a day.&rdquo; Meta&apos;s
        Llama 3 (&ldquo;The Llama 3 Herd of Models,&rdquo; arXiv:2407.21783) makes that second objective explicit, in
        its own words:
      </p>
      <blockquote>
        &ldquo;While our scaling laws suggest our flagship model is an approximately compute-optimal size for our
        training budget, we also train our smaller models for much longer than is compute-optimal. The resulting
        models perform better than compute-optimal models at the same inference budget.&rdquo;
      </blockquote>
      <p>
        Do the same tokens-per-parameter arithmetic on Llama 3 8B, pretrained on &ldquo;over 15T tokens&rdquo; per
        Meta&apos;s own release announcement:
      </p>
      <MathDisplay
        latex={String.raw`\frac{15 \times 10^{12} \text{ tokens}}{8 \times 10^{9} \text{ params}} = 1{,}875 \text{ tokens per parameter}`}
      />
      <p>
        Chinchilla-optimal for an 8B model would be <code>8e9 × 20 ≈ 160B tokens</code>. Llama 3 8B used nearly a
        hundred times that: <strong>1,875 / 20 ≈ 94×</strong> the Chinchilla-optimal ratio — trained far past the
        point of minimizing training compute, on purpose, because a smaller model that answers a query slightly
        better per FLOP is enormously cheaper to <em>serve</em> at scale than a larger, more &ldquo;compute-optimal&rdquo;
        one would be. Contrast that with the flagship: the <strong>405B</strong> model was pretrained on 15.6T
        tokens at 3.8×10²⁵ FLOPs, and the paper&apos;s <em>own</em> scaling-law fit predicted an optimum of 402B
        params / 16.55T tokens for that exact compute budget (≈41 tokens/param at that scale) — the flagship landed
        almost exactly where its own math said to land, while the smaller siblings were deliberately pushed far
        beyond theirs.
      </p>

      <TokensPerParamScatter />

      <h2>Does your in-browser Qwen follow either recipe?</h2>
      <p>
        Unknown — and that is the honest answer, not a dodge. The theory above is real and independently verified
        from primary sources: Kaplan&apos;s power laws and model-heavy split, Chinchilla&apos;s equal-growth
        correction and its ~20:1 ratio, Llama 3&apos;s deliberate overtraining for cheaper inference. What is{' '}
        <em>not</em> verifiable is which of these recipes — or something else entirely — actually produced this
        specific 852,985,920-parameter checkpoint.{' '}
        <ChapterLink chapterId="training">Chapter 13</ChapterLink> already said this model saw &ldquo;trillions of
        tokens of generic web text, books, and code&rdquo; during pretraining — deliberately without a precise count,
        because none is public. This sub-chapter cannot responsibly put a real number where Qwen has not published
        one. The 17.06B-token figure above is a hypothetical, clearly marked as such in the chart — useful for
        understanding what &ldquo;Chinchilla-optimal at this size&rdquo; would even mean, not a fact about this
        model&apos;s training run.
      </p>
    </Prose>
  );
}
