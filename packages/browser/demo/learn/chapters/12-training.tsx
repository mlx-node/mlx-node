import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { ExposureBias } from '../widgets/ExposureBias';
import { GradientDescent } from '../widgets/GradientDescent';
import { TeacherForcingAnimation } from '../widgets/TeacherForcingAnimation';
import { TrainingPlayground } from '../widgets/TrainingPlayground';

/**
 * Chapter 13 — Training objective & teacher forcing.
 *
 * The first 11 chapters are inference-first: how does a trained model do
 * one forward pass and pick one token. This chapter closes the loop with a
 * minimal but complete picture of what "trained" actually means:
 *
 *   - the objective: next-token cross-entropy
 *   - the trick that makes it parallelizable: teacher forcing + causal mask
 *   - the symmetry: training is just inference run on the whole sequence at
 *     once, scored against the ground-truth shift-by-one.
 *
 * Pure JS — no live model. The TeacherForcingAnimation widget walks the
 * student through one synthetic training step on 6 tokens.
 */

export const learning: ChapterLearningData = {
  chapterId: 'training',
  objective:
    'Explain what an LLM is actually trained to do: predict every next token in parallel under a causal mask, against ground-truth labels.',
  problem:
    "We've watched a trained model run. Where did all the weights come from? What was the model optimized to do?",
  minutes: 13,
  glossary: [
    {
      term: 'next-token prediction',
      definition:
        'The training task: given tokens [t1..ti], predict t_{i+1}. Repeated at every position of every training sequence.',
    },
    {
      term: 'gradient descent',
      definition:
        'The update rule behind training: the gradient points uphill — the direction that increases the loss fastest — so each weight is nudged the opposite way (downhill): w ← w − lr·gradient. The learning rate (lr) is the step size: too small crawls, too large overshoots and diverges. AdamW runs this per-parameter for all ~800M weights at once.',
    },
    {
      term: 'cross-entropy loss',
      definition:
        'Loss for a single prediction: −log p(target). Sums or averages across positions and batch elements to get the total loss.',
    },
    {
      term: 'teacher forcing',
      definition:
        "During training, the input at position i+1 is the true token from position i (not the model's predicted token). This supplies a valid ground-truth target at every position, so the model learns from correct context instead of its own (initially terrible) guesses.",
    },
    {
      term: 'one training step',
      definition:
        'Forward pass on a batch of sequences → compute loss → backpropagate → optimizer updates weights. Repeated trillions of times.',
    },
    {
      term: 'parallel positions',
      definition:
        'Thanks to the causal mask, every position in a sequence can be trained in a single forward pass — the model never sees the future, so the predictions stay valid.',
    },
    {
      term: 'pretraining vs fine-tuning',
      definition:
        'Pretraining: next-token prediction on trillions of tokens of generic text. Fine-tuning: same objective, smaller curated dataset, typically post-pretraining.',
    },
    {
      term: 'AdamW',
      definition:
        'The optimizer almost universally used for LLM training. It keeps a per-parameter running average of the gradient and its square (the "moments") so each weight gets an adaptive step size, with weight decay applied separately from the gradient.',
    },
    {
      term: 'backpropagation (autograd)',
      definition:
        'The algorithm that computes the gradient of the scalar loss with respect to every weight by applying the chain rule backward through the forward pass. The playground below runs real autograd on WebGPU — the same value_and_grad primitive the production trainer uses.',
    },
  ],
  takeaways: [
    'The entire training objective is one line: minimize -log p(next_token) at every position of every sequence.',
    'The causal mask is what makes training parallel — every position computes its loss in one forward pass without peeking at the future; teacher forcing supplies the ground-truth target at each position so those parallel predictions are learning from correct context.',
    'There is no separate "training mode" architecture — the same forward pass you run at inference is run during training, just scored against ground truth.',
  ],
  exercise: {
    prompt:
      'In the animation, position 5 (the last input " mat") shows "no target". Why is the last position never trained, and what would happen if you tried to train it anyway?',
    answer:
      'There is no t_{i+1} to predict for the last position — the sequence ended. If you tried, you\'d be guessing what comes after "mat" with no ground truth, which is exactly the inference setting. In practice, dataloaders shift the input/target arrays by one so this asymmetry is automatic: input is positions 0..N-1, target is positions 1..N, and the last input position has no target.',
  },
  quiz: [
    {
      id: 'q1-objective',
      prompt: 'What is the training objective of a decoder-only LLM like Qwen3.5?',
      options: [
        { id: 'a', label: 'Reconstruct masked tokens randomly hidden throughout the sequence (MLM).' },
        { id: 'b', label: 'Predict the next token at every position, scored by cross-entropy.' },
        { id: 'c', label: 'Translate the input from English to a fixed target language.' },
      ],
      correctId: 'b',
      explanation:
        'Decoder-only LLMs train on next-token prediction. Encoder-only models (BERT-style) do masked-LM; decoder LLMs do causal/autoregressive LM.',
    },
    {
      id: 'q2-teacher-forcing',
      prompt: 'What does "teacher forcing" mean?',
      options: [
        {
          id: 'a',
          label:
            "At each training position, the model's input is the true previous token — not what the model would have predicted.",
        },
        {
          id: 'b',
          label: 'The model is trained on a curated dataset of high-quality "teacher" outputs.',
        },
        {
          id: 'c',
          label: 'A scheduling trick where the learning rate is set by a teacher network.',
        },
      ],
      correctId: 'a',
      explanation:
        "Teacher forcing decouples training positions: each position's prediction is independent because each one starts from ground-truth context. Without it, training would be sequential and very slow.",
    },
    {
      id: 'q3-shift-by-one',
      prompt: "For a training sequence of length N, what's the standard input/target setup?",
      options: [
        { id: 'a', label: 'Both input and target are the full sequence; the loss ignores the diagonal.' },
        {
          id: 'b',
          label:
            'Input is positions [0..N-1], target is positions [1..N]. Position i of the model predicts target[i] which is input[i+1].',
        },
        { id: 'c', label: 'Input is the first half, target is the second half.' },
      ],
      correctId: 'b',
      explanation:
        "A simple shift-by-one: every position predicts its successor. The last input position has no corresponding target (off the end of the sequence) so it's typically masked out of the loss.",
    },
  ],
};

export function TrainingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>Training: what an LLM is actually optimized to do</h1>
        <p>
          So far this course has been about <em>inference</em>: a trained model takes a prompt and emits one token at a
          time. But where did the weights come from? What did the optimization actually solve for? The answer is
          surprisingly simple, and it's the same answer for every modern decoder LLM — GPT, Llama, Qwen, Gemma, Mistral.
          They all train on the same objective.
        </p>

        <h2>The objective in one line</h2>
        <MathDisplay
          latex={String.raw`\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(t_{i+1} \mid t_1, \dots, t_i)`}
        />
        <p>
          That's it.{' '}
          <strong>
            For every position in every sequence, predict the next token; minimize the negative log probability of the
            true next token.
          </strong>{' '}
          Summed over positions and averaged over the batch, this is what the optimizer pushes down. No reward model, no
          human in the loop (at pretraining time) — just enormous quantities of text and a shift-by-one target.
        </p>
        <p>
          The per-position loss <code>-log p(target)</code> is called <strong>cross-entropy</strong>. When{' '}
          <code>p(target)</code> is close to 1 the loss is close to 0; when <code>p(target)</code> is small the loss
          blows up. Why the <em>log</em> specifically? Two reasons: it turns &ldquo;multiply the probability of every
          token across the whole sequence&rdquo; into &ldquo;add up a per-token cost&rdquo; (a product of thousands of
          numbers below 1 would underflow to zero; a sum of logs stays well-behaved), and because{' '}
          <code>-log p → ∞</code> as <code>p → 0</code>, a confidently-wrong prediction is punished arbitrarily hard —
          the model is penalized most exactly when it is both wrong and certain. The optimizer&apos;s job is to push the
          model&apos;s probability mass onto the actual next token.
        </p>

        <h2>Teacher forcing — the trick that makes it parallel</h2>
        <p>
          A naive interpretation of "predict the next token" might run the model token by token: generate prediction 1,
          feed it back, generate prediction 2, and so on — exactly how inference works. That would be brutally slow at
          training time, and worse: the model would be learning from its own (initially terrible) predictions.
        </p>
        <p>
          The fix is <strong>teacher forcing</strong>. Instead of feeding the model's own prediction back as the next
          input, feed the <em>true</em> token. That's all teacher forcing does: it supplies a valid ground-truth target
          at every position, so the model learns from correct context rather than its own initially-terrible guesses.
          The <em>parallelism</em> comes from a separate mechanism — the causal mask (chapter 4). Because each position
          can only attend backward, every position's loss can be computed in a <em>single</em> forward pass without any
          position peeking at its own target. <em>Ground-truth targets (teacher forcing) + single-pass parallelism
          (causal mask) together let all N positions train at once.</em>
        </p>

        <TeacherForcingAnimation />

        <ExposureBias />

        <h2>Inference and training are the same forward pass</h2>
        <p>
          A consequence worth pausing on: there is no separate "training-time" architecture. The same forward pass —
          embedding lookup, 24 transformer layers with attention + MLP, RMSNorm at the top, LM head matmul — is what
          runs during both training and inference. Training just does it on a whole sequence at once and scores the
          output against ground truth; inference does it one token at a time and samples from the output.
        </p>
        <p>
          This symmetry is part of why the inference patterns from this course generalize to training. The KV cache
          (chapter 12) is an inference-only optimization, but the operations being cached are exactly the same
          operations that run during training, just without the cache.
        </p>

        <h2>Pretraining vs fine-tuning vs instruct/RLHF</h2>
        <p>
          The objective above describes <strong>pretraining</strong>: trillions of tokens of generic web text, books,
          and code, with the model learning to predict the next token in arbitrary contexts. This is where Qwen3.5's
          ~0.8B parameters get most of their knowledge — and where almost all the GPU-hours of LLM development go.
        </p>
        <p>
          <strong>Fine-tuning</strong> uses the same loss, the same forward pass — just a different, smaller, curated
          dataset (e.g. instruction-following dialogues). The model continues to predict the next token; the dataset is
          what shifts.
        </p>
        <p>
          <strong>RLHF / DPO / direct preference</strong> methods diverge from the simple cross-entropy story — they
          score model outputs against human preferences or a reward model, and the loss is no longer plain next-token
          cross-entropy. Those are out of scope for this chapter; the important point is that the <em>base behavior</em>{' '}
          of every one of these post-training stages still rests on the next-token cross-entropy foundation laid here.
        </p>

        <h2>What the optimizer actually does</h2>
        <p>
          The loss is one scalar. <strong>Backpropagation</strong> — working the chain rule backward through the forward
          pass — computes its gradient with respect to every weight in the model (~800M of them for Qwen3.5-0.8B). An
          optimizer (almost always <strong>AdamW</strong>, broken down in Chapter 14) takes a small step in the
          direction that lowers the loss. Strip it down to a single weight and the rule is easy to see: the gradient
          points uphill, so step the opposite way — and the size of that step (the learning rate) has a sweet spot.
        </p>

        <GradientDescent />

        <p>
          Repeat that per-weight step for all ~800M weights at once, for several million to several trillion tokens. The
          next chapter covers the engineering tricks — warmup, cosine decay, gradient clipping, dropout — that make this
          loop numerically stable across a stack this deep.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

// Live training playground. The learner trains a tiny char-level transformer
// from scratch on the WASM/WebGPU MLX autograd + AdamW stack and watches the
// real loss curve drop + generated samples improve. The route gates this
// chapter on the WebGPU *device* being up (device-only init), not the full
// 1.6 GB model — see DEVICE_ONLY_CHAPTERS in routes/chapters.$chapterId.tsx.
//
// The props signature matches the prior placeholder so the route mount site is
// unchanged; the TrainingPlayground widget owns the training loop.
export type TrainingDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

export function TrainingDemo({ workerRef, abortRef }: TrainingDemoProps) {
  return <TrainingPlayground workerRef={workerRef} abortRef={abortRef} />;
}
