// Sub-chapter bodies for chapter 16 (The whole model, end to end).
//
// "Local & edge inference": the deployment-economics companion to the chapter's
// model-size ladder. The chapter ends by situating this ~0.85B model near the
// bottom of the size ladder; this sub-chapter answers the natural next question
// — so why run a SMALL model on your OWN device at all? It is pure, model-free,
// SSR-safe reading prose (the prerenderer renders it to crawlable static HTML);
// the LocalEdgeTradeoffs widget carries the interactive comparison. It
// COMPLEMENTS the chapter's "this is a small model" note rather than repeating
// it — the angle here is the cloud-vs-edge deployment trade, not the size ladder.
//
// "Evaluation": perplexity (tied directly back to chapter 13's own ln(248,320)
// ≈ 12.4-nat init-loss number — exp of that IS this model's own uniform-random
// perplexity, ≈ 248,320), the three classic fixed-answer benchmarks (MMLU,
// HumanEval + pass@k, GSM8K), LLM-as-judge (Zheng et al. 2023, MT-Bench /
// Chatbot Arena, position/verbosity/self-enhancement bias), and contamination
// detection (GPT-3's 13-gram overlap -> GPT-4's 50-char substring check, which
// got BIG-bench dropped -> Min-K% Prob). Closes on the verified, directly-fetched
// gap in Qwen3.5-0.8B's own model card: it publishes MMLU-Pro/Redux, C-Eval,
// SuperGPQA, GPQA, IFEval — but NOT GSM8K, HumanEval, or classic MMLU for this
// exact checkpoint. Every paper/number here is primary-sourced; see the research
// dossier this sub-chapter was authored from for the full citation list.

import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterLink } from '../scaffolding/ChapterLink';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { LocalEdgeTradeoffs } from '../widgets/LocalEdgeTradeoffs';
import { PerplexityScale } from '../widgets/PerplexityScale';
import { QwenBenchmarkHonesty } from '../widgets/QwenBenchmarkHonesty';

/** Sub-chapter 16.1 — why run a small model on your own device: the cloud-vs-edge trade. */
export function ArchitectureLocalEdgeSection() {
  return (
    <Prose>
      <h1>Local &amp; edge inference</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 16, The whole model — why run a small model on your own device at all, and what you trade for
        it.
      </p>
      <p>
        The chapter just placed this model near the bottom of the size ladder:{' '}
        {(852_985_920).toLocaleString('en-US')} parameters, far below the frontier systems most people mean by
        &ldquo;a large language model.&rdquo; That raises a
        fair question — if a hosted API can serve a model 100× bigger, <em>why run a small one on your own device at
        all?</em> The answer is that &ldquo;small&rdquo; and &ldquo;on your device&rdquo; buy things a datacenter cannot,
        and the choice is a genuine trade with wins on both sides.
      </p>

      <h2>The trade, made concrete</h2>
      <p>
        Running locally — <strong>on the edge</strong>, on the device the user is already holding — wins on four things
        the book calls out: <strong>latency</strong> (no network round-trip, saving tens to hundreds of milliseconds),{' '}
        <strong>independence</strong> (no server, no outage, works offline), <strong>privacy</strong> (the data never
        leaves the device), and <strong>cost</strong> (it&apos;s free for the developer — the user&apos;s own hardware
        does the work). It pays for those in capability and speed: edge hardware is a fraction of a datacenter GPU, it
        throttles thermally, the hardware×software matrix is fragmented, and it drains the battery. Neither side wins
        every row — that is the trade:
      </p>

      <LocalEdgeTradeoffs />

      <h2>Where each size can run</h2>
      <p>
        Model size sets the floor. Phones struggle past a billion or two parameters — which is exactly where this model
        sits comfortably. A laptop runs an 8B in <code>bf16</code>; getting a 100B-plus model onto a high-end desktop
        takes the lever from an earlier <a href="/chapters/kv-cache/quantization">sub-chapter</a> —{' '}
        <strong>quantization</strong> is the bridge that fits a big model into small memory (via Ollama / llama.cpp), and
        a Mixture-of-Experts helps too by activating only a slice of its parameters per token. Past that, a frontier
        model in the hundreds of billions to trillions of parameters simply lives in the cloud; low-end laptops and
        Chromebooks run no meaningful local model at all.
      </p>
      <p>
        And the hardware itself encodes a trade. A gaming GPU and a workstation can cost in the same ballpark and
        optimize for opposite things — <strong>capacity versus speed</strong> — which is the second table in the widget
        above: a 32&nbsp;GB card with huge bandwidth against a 512&nbsp;GB unified-memory machine that holds far bigger
        models but streams them slower.
      </p>
      <p className="text-muted-foreground">
        You are here. This browser tab <em>is</em> edge inference: Qwen3.5-0.8B, <code>bf16</code>, batch 1, on WebGPU —
        no server, no API key, no upload. It buys you privacy, zero network latency, and $0 cost, and pays for it in
        capability (a 0.8B model, not a frontier one) and speed (your GPU, not an H100). The future of inference is not
        local <em>or</em> cloud but <strong>both</strong> — small models and quick queries on-device, heavier workloads
        in the datacenter — and this whole course has been the small-model, on-device half, running live in front of you.
      </p>
    </Prose>
  );
}

/** Sub-chapter 16.2 — perplexity, benchmarks, LLM-as-judge, and contamination: how a finished model gets scored, and how much to trust the score. */
export function ArchitectureEvaluationSection() {
  return (
    <Prose>
      <h1>Evaluation: perplexity, benchmarks, and how much to trust them</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 16, The whole model — the finished model exists now; how do you put a number on how good it
        is?
      </p>
      <p>
        <ChapterLink chapterId="training">Chapter 13</ChapterLink> built the training objective from{' '}
        <code>-log p(target)</code>: the model&apos;s loss is the negative log-probability it assigns the real next
        token, averaged over every position. That number never left training — it is also, unexponentiated, the
        <em>only</em> ingredient evaluation needs. <strong>Perplexity</strong> is that same loss, exponentiated:
      </p>
      <MathDisplay
        latex={String.raw`\text{PPL}(X) = \exp\!\left(-\frac{1}{t}\sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})\right) = \exp(\text{cross-entropy loss})`}
      />
      <p>
        which the Hugging Face Transformers docs state is exactly &ldquo;the exponentiation of the cross-entropy
        between the data and model predictions.&rdquo; It has a clean reading: perplexity is the size of the vocabulary
        the model is, on average, as confused as if it were guessing <em>uniformly</em> among that many equally-likely
        options. Guess among 2 options with total confidence and perplexity is 1; be utterly lost across the whole
        vocabulary and perplexity approaches the vocabulary size itself.
      </p>

      <h2>This exact model&apos;s own number, exponentiated</h2>
      <p>
        Chapter 13 already worked this out: at initialization every one of this model&apos;s 248,320 vocabulary tokens
        gets roughly the same probability, so the loss starts at <code>−ln(1/248,320) = ln(248,320) ≈ 12.4</code> nats.
        Exponentiate that and you get back exactly <strong>248,320</strong> — the vocabulary size itself. That is not a
        coincidence: the perplexity of a perfectly uniform guess over <code>V</code> outcomes is always exactly{' '}
        <code>V</code>, because cross-entropy over a uniform distribution is <code>ln V</code> and{' '}
        <code>exp(ln V) = V</code>. So before a single gradient step, this exact model&apos;s perplexity over its own
        vocabulary is <strong>≈ 248,320</strong> — and every nat the loss drops during training is this number
        shrinking. Two real, published anchors show where a well-trained model ends up: GPT-3 175B reached a
        perplexity of <strong>20.50</strong> on Penn Treebank (versus a prior GPT-2-era best of 35.8), and{' '}
        <strong>1.92</strong> on LAMBADA next-word prediction with few-shot prompting (versus a prior best of 8.63) —
        both from the original GPT-3 paper. GPT-2&apos;s second-largest model (762M parameters — 1.5B is the
        largest) reported a WikiText-2 perplexity of{' '}
        <strong>19.93</strong>. Same formula throughout the ladder below — from this model&apos;s own ≈248,320 at
        initialization down to GPT-3&apos;s 1.92, five orders of magnitude of improvement:
      </p>

      <PerplexityScale />

      <h2>Standard benchmarks: fixed questions, fixed answers</h2>
      <p>
        Perplexity is cheap to compute and needs no held-out task design, but it only measures how well the model
        predicts <em>generic</em> text — it says nothing about whether the model can answer a trivia question, write
        working code, or solve a math problem. For that, the field leans on <strong>benchmarks</strong>: fixed sets of
        questions with fixed, checkable answers.
      </p>
      <p>
        <strong>MMLU</strong> (Massive Multitask Language Understanding, Hendrycks et al. 2020) is 57 school- and
        professional-level subjects — history, law, medicine, physics, and more — packaged as 15,908 four-way
        multiple-choice questions. It is scored as plain multiple-choice accuracy against a 25% random-guess floor;
        GPT-3 175B reached 43.9% few-shot, while smaller GPT-3 sizes barely cleared chance (24.9–26.0%).
      </p>
      <p>
        <strong>HumanEval</strong> (Chen et al. 2021, the paper that introduced OpenAI Codex) is 164 hand-written
        Python problems: given a function signature and a docstring, complete the body, then run real unit tests
        against it (7.7 tests per problem on average). Because sampling is random, a model gets <code>k</code>{' '}
        attempts and the metric is <strong>pass@k</strong> — the fraction of problems solved by <em>at least one</em>{' '}
        of <code>k</code> samples, computed with an unbiased estimator so it doesn&apos;t just reward generating more
        samples:
      </p>
      <MathDisplay
        latex={String.raw`\text{pass@}k := \mathbb{E}_{\text{Problems}}\left[\,1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right]`}
      />
      <p>
        where <code>n</code> samples are drawn per problem and <code>c</code> of them pass every test. Codex reached
        pass@1 = 28.8% and pass@100 = 70.2% — while plain GPT-3, with no code fine-tuning, solved 0%.
      </p>
      <p>
        <strong>GSM8K</strong> (Cobbe et al. 2021) is 8.5K grade-school word problems (7.5K for training, 1K held out
        for testing) — &ldquo;Natalia sold clips to 48 friends in April, and half as many in May…&rdquo; — graded by{' '}
        <strong>exact match</strong> on the final numeric answer, not pass@k: there is no code to run, just one number
        to check. The paper&apos;s headline result is about a <em>verifier</em>, not raw scale: a 6B model fine-tuned
        to solve problems, paired with a second 6B model trained to verify solutions and re-rank 100 sampled
        candidates, slightly outperforms a fine-tuned 175B model sampling once with no verifier at all — a gain the
        paper describes as &ldquo;approximately equivalent to a 30x model size increase.&rdquo;
      </p>

      <h2>LLM-as-judge: when there is no single right answer</h2>
      <p>
        None of the above works for open-ended output — &ldquo;write me a birthday message&rdquo; has no fixed correct
        string to match. Zheng et al. (2023) proposed the fix the industry now leans on: have a strong LLM (they used
        GPT-4) read two candidate responses and <strong>judge</strong> which is better, the same way a human rater
        would. Their MT-Bench (a fixed set of multi-turn questions) and Chatbot Arena (crowdsourced head-to-head
        battles) are built on exactly this idea, and GPT-4-as-judge reached <strong>&gt;80%</strong> agreement with
        human preferences — comparable to how often two humans agree with each other.
      </p>
      <p>
        The paper is equally clear about where an LLM judge goes wrong, naming three specific biases: <strong>position
        bias</strong> (favoring whichever response is shown first), <strong>verbosity bias</strong> (favoring the
        longer answer, regardless of whether it&apos;s better), and <strong>self-enhancement bias</strong> (a judge
        favoring answers that read like its own family&apos;s style). LLM-as-judge is powerful precisely because it
        scales to open-ended text that fixed-answer benchmarks can&apos;t touch — but it inherits a model&apos;s own
        blind spots as the price of that flexibility.
      </p>

      <h2>Contamination: why none of this is trustworthy at face value</h2>
      <p>
        Every benchmark above is public text on the internet, and every modern model is pretrained on a huge crawl of
        the internet. If a benchmark&apos;s questions — or worse, its answers — leaked into pretraining data, a model
        can score well by <em>having memorized the test</em>, not by reasoning its way to the answer. This is{' '}
        <strong>contamination</strong>, and catching it has gotten stricter over time.
      </p>
      <p>
        GPT-3&apos;s own paper used a fairly loose check: <strong>13-gram overlap</strong> between an eval example and
        the training crawl. By that method, it flagged over 90% of QuAC, SQuADv2, and DROP examples as contaminated —
        a striking number that shows just how much public benchmark text ends up baked into a large-enough web crawl.
        GPT-4&apos;s technical report tightened the bar to a <strong>50-character substring check</strong>: strip
        spaces and symbols from both eval and training text, then see if any of three random 50-character chunks from
        an eval example shows up verbatim in training data. That stricter test found real leakage — OpenAI re-scored
        exams with the contaminated questions removed, and dropped an entire benchmark, BIG-bench, from its reported
        results once contamination in it was confirmed. A newer method skips the training corpus altogether:{' '}
        <strong>Min-K% Prob</strong> (Shi et al. 2023) looks only at the model itself, flagging a sequence as
        likely-memorized when its least-probable tokens are still suspiciously high-probability — a genuinely unseen
        example should contain a few real surprises, and a memorized one usually doesn&apos;t.
      </p>
      <p className="text-muted-foreground">
        So read every benchmark number in this sub-chapter, and every benchmark number you see anywhere, with that
        asterisk attached: it is a real, checkable score on a real test — and also a number that could be partly
        measuring memorization rather than capability, to a degree nobody outside the lab that trained the model can
        fully audit.
      </p>

      <h2>Does your in-browser Qwen have benchmark numbers?</h2>
      <p>
        Partly. Qwen3.5-0.8B&apos;s official Hugging Face model card publishes real, dated scores across a family of
        knowledge and reasoning benchmarks — MMLU-Pro, MMLU-Redux, C-Eval, SuperGPQA, GPQA, and IFEval among them —
        most reported twice (once for the model&apos;s default <strong>non-thinking</strong> mode and once for{' '}
        <strong>thinking</strong> mode, the{' '}
        <ChapterLink chapterId="post-training">&lt;think&gt; block</ChapterLink> from the post-training chapter), with
        one exception: GPQA, where the card reports only a thinking-mode score. What
        it does <em>not</em> publish, for this exact 0.8B checkpoint, is a GSM8K score, a HumanEval score, or a classic
        (non-Pro, non-Redux) MMLU score — and a direct search turns up no independent number for any of those three,
        either. Here is the card&apos;s own table, gaps included:
      </p>

      <QwenBenchmarkHonesty />

      <p className="text-muted-foreground">
        That gap is the honest answer, not a gap this course is going to paper over with an invented number. Not every
        checkpoint gets run through every benchmark — smaller, faster-shipping models in a family are frequently
        evaluated on a narrower slice than the flagship release — and reporting &ldquo;not measured&rdquo; is a more
        trustworthy state than reporting a number nobody checked.
      </p>
    </Prose>
  );
}
