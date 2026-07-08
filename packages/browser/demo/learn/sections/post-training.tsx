// Sub-chapter 15.1 — Knowledge distillation.
//
// A "go deeper" deep-dive hung off chapter 15 (From base model to assistant).
// Pure, model-free reading prose — SSR-safe, so the prerenderer renders it
// straight to crawlable static HTML (see scripts/prerender.entry.tsx). Wrapped
// in <Prose> by the section route / prerenderer, so it returns just the
// article content.
//
// Builds on chapter 15's own framing of SFT — "same next-token loss, a
// smaller curated dataset" — by asking where that dataset's LABELS come from.
// Distillation's answer: from a bigger TEACHER model instead of (or in
// addition to) a human annotator. Two eras, both grounded in primary sources:
//   - Classic: Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural
//     Network," arXiv:1503.02531 (2015) — soften the teacher's softmax with a
//     temperature T, train the student to match that soft distribution (plus
//     a lighter hard-label term); the BMW/garbage-truck/carrot quote and the
//     MNIST (146 -> 74 errors) / speech (>80% of ensemble gain transferred)
//     numbers are quoted/cited directly from the paper.
//   - Modern: DeepSeek-R1, arXiv:2501.12948 (2025), Appendix F — no soft
//     targets, no KL: plain SFT on (prompt, teacher-generated-response) pairs.
//     "we apply only SFT and do not include an RL stage" (quoted verbatim).
//     Appendix F.1's own number: distilling into Qwen2.5-32B reaches AIME 72.6
//     vs. 47.0 for RL-training that same base model directly, no distillation.
//   - Bridge: Qwen3 Technical Report, arXiv:2505.09388, section 4.5,
//     "Strong-to-Weak Distillation" — an official precedent for the classic
//     and modern recipes coexisting in one pipeline (off-policy SFT, then
//     on-policy KL-logit-matching) for QWEN3's small models. NOT confirmed
//     for Qwen3.5-0.8B: official Qwen3.5 materials (HF model card, GitHub
//     README, qwen.ai blog) attribute this family's training to pretraining
//     plus "Scaled Reinforcement Learning," never distillation — so the
//     closing section says so plainly instead of asserting it.

import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterLink } from '../scaffolding/ChapterLink';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { DistillationPipelines } from '../widgets/DistillationPipelines';

/** Sub-chapter 15.1 — knowledge distillation: SFT where the teacher, not a human, writes the label. */
export function PostTrainingDistillationSection() {
  return (
    <Prose>
      <h1>Knowledge distillation</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 15, From base model to assistant — training a small model to imitate a bigger one, instead
        of (or in addition to) imitating humans.
      </p>

      <h2>Same recipe, a different source for the label</h2>
      <p>
        The <ChapterLink chapterId="post-training">post-training chapter</ChapterLink> described instruction tuning
        (SFT) as the same next-token loss the model has always used, just pointed at a small, curated dataset of{' '}
        <strong>(instruction, response)</strong> pairs — usually written or vetted by humans. <strong>Knowledge
        distillation</strong> keeps that exact setup and changes one thing: instead of a human writing the response, a
        bigger, more capable <strong>teacher</strong> model writes it (or, in the classic version below, supplies the
        whole target distribution). The <strong>student</strong> — the model actually being trained, and usually much
        smaller than the teacher — learns to imitate the teacher instead of, or in addition to, a human annotator.
      </p>
      <p>
        It's still exactly the loss the <ChapterLink chapterId="training">training chapter</ChapterLink> introduced:{' '}
        <MathDisplay inline latex={String.raw`-\log p(\text{target token})`} />, averaged over every scored position —
        at initialization that's <MathDisplay inline latex={String.raw`\ln(248{,}320) \approx 12.4`} /> nats, uniform
        guessing over this model's vocabulary. Distillation never touches that formula. It only changes what counts as
        the <em>target</em>.
      </p>

      <h2>The classic recipe: soft targets and "dark knowledge"</h2>
      <p>
        The technique is older than the LLM era — Geoffrey Hinton, Oriol Vinyals, and Jeff Dean introduced it in 2015
        under the name <strong>distillation</strong> (building on an earlier idea from Rich Caruana and collaborators,
        who matched raw logits directly — a special case of Hinton's method in the high-temperature limit, as the paper
        proves). Their insight: a trained classifier's softmax assigns some probability to <em>every</em> class, even
        the wrong ones, and those small probabilities are not noise — they encode how the model tends to generalize.
        Their own example, quoted directly:
      </p>
      <p>
        <em>
          "An image of a BMW, for example, may only have a very small chance of being mistaken for a garbage truck,
          but that mistake is still many times more probable than mistaking it for a carrot."
        </em>{' '}
        — Hinton, Vinyals &amp; Dean, <em>Distilling the Knowledge in a Neural Network</em>, arXiv:1503.02531, §1
      </p>
      <p>
        A one-hot label ("this is a BMW") throws that pattern away entirely. To keep it, soften the teacher's softmax
        with a <strong>temperature</strong> <MathDisplay inline latex={String.raw`T`} />:
      </p>
      <MathDisplay latex={String.raw`q_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}`} />
      <p>
        At <MathDisplay inline latex={String.raw`T = 1`} /> this is ordinary softmax; raising <MathDisplay inline
        latex={String.raw`T`} /> flattens the distribution so the near-miss (garbage truck) and the far-miss (carrot)
        separate from each other instead of both rounding to zero. The student is then trained on a weighted mix of two
        cross-entropies: one against the teacher's <strong>soft targets</strong> at that same high <MathDisplay inline
        latex={String.raw`T`} />, and a lighter one against the ordinary <strong>hard label</strong> at{' '}
        <MathDisplay inline latex={String.raw`T = 1`} />. One detail worth keeping: because the soft-target term's
        gradient shrinks like <MathDisplay inline latex={String.raw`1/T^2`} />, Hinton's paper multiplies that term by{' '}
        <MathDisplay inline latex={String.raw`T^2`} /> so its contribution doesn't quietly vanish as{' '}
        <MathDisplay inline latex={String.raw`T`} /> is raised.
      </p>
      <p>
        The results were not subtle. On MNIST, a small 800-unit network trained the ordinary way made 146 test errors;
        the identical network trained only to match a larger, dropout-regularized network's soft targets (at{' '}
        <MathDisplay inline latex={String.raw`T = 20`} />) made 74 — roughly half. On a commercial speech model (~85M
        parameters, 2,000 hours of audio), a 10-model ensemble improved frame accuracy from 58.9% to 61.1%; a single
        model distilled from that ensemble reached 60.8% — the paper reports this captured "more than 80%" of the
        ensemble's gain, in one model instead of ten.
      </p>

      <h2>The modern recipe: just SFT, with a smarter source of text</h2>
      <p>
        Today's largest LLM distillation effort drops the soft-distribution machinery entirely. DeepSeek's R1 paper
        distills its own long chain-of-thought reasoning into six much smaller dense models (Qwen2.5-Math-1.5B/7B,
        Qwen2.5-14B/32B, Llama-3.1-8B, Llama-3.3-70B) using 800,000 (prompt, response) pairs where every response was
        generated by DeepSeek-R1 itself (about 600K reasoning traces plus 200K general-purpose examples). The method,
        stated verbatim in the paper's appendix: <em>"we apply only SFT and do not include an RL stage."</em> No
        temperature, no KL divergence, no teacher logits at all — the teacher's generated text is simply treated as an
        ordinary SFT target, exactly the recipe from the top of this page.
      </p>
      <p>
        The headline result argues distillation isn't just cheaper than training a small reasoner from scratch — it can
        be <em>better</em>. Directly reinforcement-learning a Qwen2.5-32B base model with no distillation ("Qwen2.5-32B-
        Zero") reaches AIME 2024 pass@1 of 47.0. Simply fine-tuning that same-size base model on DeepSeek-R1's
        generated responses — plain SFT, no RL at all — reaches 72.6. Even the smallest distilled model,
        DeepSeek-R1-Distill-Qwen-1.5B, beats GPT-4o and Claude-3.5-Sonnet on that benchmark. Compare both eras side by
        side:
      </p>

      <DistillationPipelines />

      <h2>Not mutually exclusive: Qwen's own strong-to-weak distillation</h2>
      <p>
        The two recipes above aren't rivals — Qwen's own (prior-generation) Qwen3 Technical Report documents a pipeline
        that uses both, one after the other, for Qwen3's small dense and MoE models. First an{' '}
        <strong>off-policy</strong> phase: plain SFT on responses generated by a larger teacher (Qwen3-32B or
        Qwen3-235B-A22B) — the modern recipe above. Then an <strong>on-policy</strong> phase: the small student
        generates its own rollouts, and is fine-tuned to minimize the <strong>KL divergence</strong> between its own
        logits and the teacher's — literally Hinton's 2015 method, just run on the student's own samples instead of a
        fixed dataset. Qwen reports this reached higher benchmark scores than their full four-stage training pipeline
        while using "only 1/10 of the GPU hours." It's a real, documented example of exactly the classic-vs-modern
        split this page just walked through, coexisting in one training run.
      </p>

      <h2>Does your in-browser Qwen do this?</h2>
      <p>
        Unconfirmed — and worth saying plainly rather than assuming it by analogy. The paragraph above is about Qwen3,
        the <em>previous</em> generation. For Qwen3.5 — the Qwen3-Next-based family this course actually teaches,
        including the 0.8B model running in this tab — no official source turned up any mention of distillation. The
        Qwen3.5-0.8B model card, the QwenLM/Qwen3.5 GitHub README, and the flagship qwen.ai announcement all describe
        this family's training as pretraining followed by <strong>"Scaled Reinforcement Learning"</strong> —
        reinforcement learning run across large numbers of agent environments — with no teacher model, no
        "strong-to-weak" pipeline, and no distillation mentioned anywhere for Qwen3.5 specifically. So the honest
        answer is: Qwen has publicly documented doing exactly what this page describes, for its <em>previous</em>{' '}
        generation's small models. Whether the same trick, or some variant of it, touched this specific 0.8B checkpoint
        is not something any public source confirms — so this course does not claim it.
      </p>
    </Prose>
  );
}
