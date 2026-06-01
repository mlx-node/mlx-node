import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { GradClipDemo } from '../widgets/GradClipDemo';
import { LrScheduleViz } from '../widgets/LrScheduleViz';
import { WarmupLossCurve } from '../widgets/WarmupLossCurve';

/**
 * Chapter 14 — Scaling & regularization.
 *
 * The training objective (chapter 13) was one line. Making it actually
 * converge on a 24-layer stack with hundreds of millions of parameters needs
 * a handful of engineering tricks that aren't part of the architecture but
 * without which nothing works: LR warmup + cosine, gradient clipping,
 * weight decay, and (historically) dropout.
 *
 * Pure JS, two interactive widgets. The goal is to give the reader the
 * vocabulary to read someone else's training config and know what each line
 * is doing.
 */

export const learning: ChapterLearningData = {
  chapterId: 'scaling',
  objective:
    'Read a real LLM training config (LR, warmup, clip, weight decay) and explain what each knob is preventing from going wrong.',
  problem:
    "Cross-entropy + AdamW on a deep transformer isn't enough — the loss diverges, the gradients explode, or the model overfits without specific engineering tricks.",
  minutes: 7,
  glossary: [
    {
      term: 'AdamW',
      definition:
        'The dominant optimizer for LLM pretraining. Adam (adaptive per-parameter step) plus decoupled weight decay (the W).',
    },
    {
      term: 'learning rate warmup',
      definition:
        "Linear ramp from 0 to lr_max over the first ~1-10% of training. Lets the optimizer's moving averages stabilize before taking large steps.",
    },
    {
      term: 'cosine decay',
      definition:
        'After warmup, the LR follows half a cosine wave from lr_max down to lr_min over the remaining steps. Standard for LLM pretraining.',
    },
    {
      term: 'gradient clipping',
      definition:
        'If ||g|| > c (typically 1.0), rescale g by c/||g||. Cap the step size when a bad batch produces a huge gradient.',
    },
    {
      term: 'weight decay',
      definition:
        'A penalty proportional to ||θ||^2 added to the loss (or, in AdamW, subtracted from the weights directly). Pulls weights toward zero, regularising.',
    },
    {
      term: 'dropout',
      definition:
        'Randomly zero a fraction of activations during training to prevent over-reliance on any one feature. Common in older models, often unused in modern LLM pretraining where dataset diversity does the regularisation.',
    },
  ],
  takeaways: [
    "A deep transformer doesn't train with a fixed learning rate — warmup-then-cosine is the standard schedule and the reason early loss curves stay sane.",
    'Gradient clipping is a one-line guardrail that turns "model diverged on batch 3,247" into a small bump in the loss curve.',
    'Dropout is mostly gone from modern LLM pretraining; weight decay + dataset scale + early-stopping discipline replaced it.',
  ],
  exercise: {
    prompt:
      'In the LR widget, set warmup to 0 and peak LR to ~1e-3, then look at the start of the curve. What would happen to a fresh model if you started training at that LR? Why does the cosine half of the schedule exist?',
    answer:
      "With no warmup at 1e-3, the very first step would be a huge update on essentially random weights. AdamW's running averages haven't built up, so each parameter takes a step of order lr * grad — likely into a region the optimizer can't recover from. Cosine decay solves the opposite problem at the end: as the model approaches a good minimum, small steps let it settle without bouncing out. The schedule is shaped like a hill (low, peak, low) for both reasons combined.",
  },
  quiz: [
    {
      id: 'q1-warmup',
      prompt: 'Why does LLM training start with a learning-rate warmup?',
      options: [
        { id: 'a', label: 'It gives the GPU time to warm up before the real training starts.' },
        {
          id: 'b',
          label:
            "The optimizer's running averages of gradient and squared gradient haven't filled in yet — large steps early on cause divergence.",
        },
        { id: 'c', label: 'Warmup is a regulatory requirement for foundation-model training.' },
      ],
      correctId: 'b',
      explanation:
        'AdamW tracks moving averages of g and g². With both at zero on step 1, the per-parameter step size is essentially the raw gradient. Warmup keeps that step size small until the averages have stabilised.',
    },
    {
      id: 'q2-clip',
      prompt: "What does 'clip gradient norm to 1.0' do?",
      options: [
        {
          id: 'a',
          label: 'If ||g|| > 1.0, rescale every component by 1.0 / ||g||. Direction preserved, magnitude capped.',
        },
        {
          id: 'b',
          label: 'Set any individual gradient component to ±1.0 if it exceeds that range.',
        },
        {
          id: 'c',
          label: 'Drop the largest-magnitude gradient component entirely.',
        },
      ],
      correctId: 'a',
      explanation:
        'Clip-norm preserves the gradient direction — only the global magnitude is capped. Per-component clipping (option b) is a different and rarer recipe.',
    },
    {
      id: 'q3-dropout-modern',
      prompt: "Why don't modern LLM pretraining configs use dropout the way 2017-era models did?",
      options: [
        {
          id: 'a',
          label: 'Dropout is mathematically equivalent to RMSNorm; modern models use one or the other.',
        },
        {
          id: 'b',
          label:
            'Pretraining on trillions of unique tokens is so data-rich that the model has no opportunity to overfit any single example; dropout becomes redundant and slows training.',
        },
        { id: 'c', label: 'Dropout breaks the residual stream.' },
      ],
      correctId: 'b',
      explanation:
        "Dropout was a regulariser for the data-limited regime. Pretraining LLMs see each token roughly once, so the over-fit failure mode dropout was designed to prevent simply doesn't show up.",
    },
  ],
};

export function ScalingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>Scaling &amp; regularization: making the training loop actually converge</h1>
        <p>
          Chapter 13 reduced training to one line of cross-entropy. That description is correct and it is{' '}
          <em>spectacularly insufficient</em> as a recipe. A 24-layer transformer trained on hundreds of billions of
          tokens with naive SGD will: diverge in the first 100 steps; have its loss spike to <code>NaN</code> from a
          single bad batch; settle into a local minimum that generalizes poorly. The fix is a small handful of
          engineering tricks — none of them are part of the model architecture, but every modern training run uses all
          of them.
        </p>

        <h2>The optimizer: AdamW, not SGD</h2>
        <p>
          Plain stochastic gradient descent (<code>θ := θ - η·g</code>) doesn't work well for transformers. Different
          parameters need very different step sizes — embedding rows for rare tokens need much larger updates than dense
          weights, for example. <strong>Adam</strong> tracks per-parameter running averages of <code>g</code> and{' '}
          <code>g²</code>, then takes a step normalized by <code>√g²</code>. <strong>AdamW</strong> adds{' '}
          <em>decoupled weight decay</em>: instead of penalising <code>||θ||²</code> through the loss, the optimizer
          subtracts a small fraction of <code>θ</code> from itself at every step. This is the standard recipe for every
          modern LLM pretrain.
        </p>

        <h2>The learning-rate schedule: warmup + cosine</h2>
        <p>
          Here's the most universally-used trick in LLM training. The learning rate is not a constant — it follows a
          hill shape:
        </p>
        <ul>
          <li>
            <strong>Linear warmup</strong> from 0 to <code>lr_max</code> over the first ~1-10% of training.
          </li>
          <li>
            <strong>Cosine decay</strong> from <code>lr_max</code> down to <code>lr_min</code> (≈ <code>1e-5</code>)
            over the rest.
          </li>
        </ul>
        <p>
          Warmup exists because AdamW's running averages need a few hundred steps of gradient history to be meaningful;
          taking a full-size step on step 1 is essentially shooting in a random direction. Cosine decay exists because
          late training benefits from progressively smaller steps — the model is closer to a minimum and large steps
          would bounce it out.
        </p>

        <LrScheduleViz />

        <WarmupLossCurve />

        <h2>Gradient clipping: the one-line guardrail</h2>
        <p>
          A single bad batch — say, a piece of text that's all whitespace, or a tokenizer edge case — can produce
          enormous gradients. Without protection, AdamW dutifully takes a giant step in that direction and the loss
          jumps from 2.5 to 8.0, and the model needs hundreds of steps to recover (if it recovers at all).
        </p>
        <p>
          Clip-by-norm is the universal answer. Compute the global L2 norm of every gradient in the model, and if it
          exceeds the clip threshold <code>c</code> (almost always <code>1.0</code>), rescale every component by{' '}
          <code>c / ||g||</code>. Direction preserved, magnitude capped, training continues smoothly. The widget below
          lets you tweak both knobs.
        </p>

        <GradClipDemo />

        <h2>Regularization: dropout's slow exit</h2>
        <p>
          The original Transformer paper used dropout aggressively — every attention layer, every MLP, every residual
          connection. Modern LLM pretraining configs typically set dropout to 0. Two reasons:
        </p>
        <ul>
          <li>
            <strong>Pretraining is data-rich.</strong> A model sees each token roughly once across a
            multi-trillion-token corpus. There is no "memorize the training set" failure mode for dropout to prevent.
          </li>
          <li>
            <strong>Weight decay covers most of the same ground.</strong> AdamW's decay term pulls weights toward zero,
            preventing any one feature from dominating.
          </li>
        </ul>
        <p>
          Fine-tuning is a different story — a small curated dataset <em>can</em> be overfit, and dropout often
          reappears at non-zero values (typically 0.05-0.1) in the fine-tuning recipe.
        </p>

        <h2>Reading a real config</h2>
        <p>
          The combination of tricks above turns a training run from "diverges immediately" into "converges at all." A
          typical LLM pretrain config looks roughly like:
        </p>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-[12px]">
          {`optimizer:   AdamW(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
lr:          3e-4 peak, 2000 warmup steps, cosine decay to 1e-5
grad_clip:   1.0  (clip by global norm)
dropout:     0.0  (pretraining)
batch_size:  4M tokens (gradient accumulation across many devices)
seq_len:     8192
total_steps: 500,000`}
        </pre>
        <p>
          Every one of those lines is a guardrail against a specific failure mode discovered the hard way during the
          last decade of LLM training. The architecture is the model; this is the recipe that makes the architecture
          finishable.
        </p>

        <h2>Where to learn more</h2>
        <p>
          This chapter is deliberately the lightest one in the course — you can train an LLM without internalising every
          detail here, but you can't read a research paper without recognising these knobs. For the trainable side of
          this codebase, see <code>@mlx-node/trl</code> (GRPO and SFT) and <code>crates/mlx-tui</code> (the{' '}
          <code>mlx-train</code> TUI binary) — they implement the same recipe on Apple Silicon.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

export type ScalingDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

export function ScalingDemo(_: ScalingDemoProps) {
  return (
    <div className="space-y-3 rounded-md border border-dashed border-border bg-background p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">No live run for this chapter</div>
      <p className="text-[13px] text-foreground/85">
        Training is out-of-process for this browser app — the interactive widgets live inside the chapter body. Try
        sliding the LR schedule's warmup down to 0 and watching the curve start at the peak: that's exactly the curve
        every "loss exploded on step 1" run starts with.
      </p>
    </div>
  );
}
