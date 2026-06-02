import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { GenerationLoop } from '../widgets/GenerationLoop';
import { LogitsToSoftmax } from '../widgets/LogitsToSoftmax';
import { TokenJourney } from '../widgets/TokenJourney';

/**
 * Chapter 1 (overview) — "What is an LLM?". The top-down on-ramp the course was
 * missing: before any tokenizer or attention math, state the goal (a function
 * from tokens to a score for every next token) and the autoregressive
 * generation loop (predict → append → repeat). Prose + the self-contained
 * GenerationLoop animation; no model/worker needed, so it renders even while the
 * real model is still downloading. Calibrated for a programmer new to ML:
 * framed with array shapes and a pseudocode loop.
 */

const VOCAB = 248_320; // Qwen3.5-0.8B vocab_size — matches sibling chapters

export const learning: ChapterLearningData = {
  chapterId: 'overview',
  objective:
    'State, in one sentence, what an LLM computes — and trace how that single computation, run in a loop, produces text.',
  problem:
    'Every later chapter zooms into one component. Without the big picture first, the machinery has no purpose to hang on.',
  minutes: 6,
  glossary: [
    {
      term: 'language model',
      definition: 'A function that scores how likely each possible next token is, given the tokens so far.',
    },
    {
      term: 'token',
      definition: 'The unit an LLM reads and writes — a sub-word chunk, not a character or a whole word.',
    },
    {
      term: 'logits',
      definition:
        'The raw, unbounded scores the model outputs — one per vocabulary entry — before softmax turns them into probabilities.',
    },
    {
      term: 'softmax',
      definition:
        'Exponentiate every logit and divide by their sum: e^(z_i) / Σ_j e^(z_j). Turns a vector of raw scores into a probability distribution that sums to 1.',
    },
    {
      term: 'forward pass',
      definition: 'One evaluation of the model on a token sequence, producing the logits for the next position.',
    },
    {
      term: 'autoregressive',
      definition: 'Generating one token at a time, feeding each output back in as input for the next step.',
    },
    {
      term: 'generation loop',
      definition: 'forward pass → sample one token → append it → repeat, until a stop token or a length limit.',
    },
  ],
  takeaways: [
    'An LLM is one function: a list of tokens in, a score for every possible next token out.',
    'Text is generated one token at a time — the same forward pass runs in a loop, each pass seeing one more token.',
    '"Predict the next token" is the whole job at inference time; capability is what that objective forces the model to learn.',
  ],
  exercise: {
    prompt:
      'In the generation-loop animation, watch the pseudocode. Which line is highlighted at the exact moment a new token joins the strip on the left?',
    answer:
      'tokens.push(next) — the append step. The strip only grows after a token has been sampled from the forward pass, then pushed onto the list; the loop then runs again on the now-longer sequence.',
  },
  quiz: [
    {
      id: 'q1-one-pass',
      prompt: 'Mechanically, what does an LLM compute in a single forward pass?',
      options: [
        { id: 'a', label: 'The full sentence it will eventually produce.' },
        { id: 'b', label: 'A score (logit) for every token in its vocabulary, for the single next position.' },
        { id: 'c', label: 'A yes/no decision about whether the text so far is valid.' },
      ],
      correctId: 'b',
      explanation:
        'One forward pass yields one distribution over the whole vocabulary for the next token. Longer outputs come from running that pass in a loop.',
    },
    {
      id: 'q2-loop',
      prompt: 'How does an LLM produce a whole paragraph rather than just one token?',
      options: [
        { id: 'a', label: 'It plans the entire paragraph, then writes it out at once.' },
        { id: 'b', label: 'It runs the same forward pass in a loop, sampling one token and appending it each time.' },
        { id: 'c', label: 'It retrieves the closest matching paragraph from its training data.' },
      ],
      correctId: 'b',
      explanation: 'Generation is autoregressive: predict one token, append it, run again on the longer sequence.',
    },
    {
      id: 'q3-logits',
      prompt: 'What are "logits"?',
      options: [
        { id: 'a', label: 'Probabilities that already sum to 1.' },
        { id: 'b', label: 'Raw, unbounded scores — one per vocabulary token — before softmax normalizes them.' },
        { id: 'c', label: 'The token ids produced by the tokenizer.' },
      ],
      correctId: 'b',
      explanation:
        'Softmax turns logits into a probability distribution; the raw logits themselves are not yet normalized.',
    },
  ],
};

export function OverviewChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>What is an LLM?</h1>
        <p>
          Strip away the mystique and a large language model is a single function. You hand it a list of tokens — the
          text so far — and it returns a score for <em>every</em> token in its vocabulary: a guess at what comes next.
          That is the entire computation. Everything else in this course is either how that function is built, or how
          it's used.
        </p>

        <h2>One call: tokens in, a score for every next token</h2>
        <p>
          Concretely, the input is just a list of integers — one id per token, produced by the{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'tokenization' }} search={(prev) => prev}>
            tokenizer
          </Link>{' '}
          — and the output is a big array of decimals, one score per vocabulary word:
        </p>
        <pre>
          <code>{`tokens: number[]          // a list of integers, one id per token, e.g. [791, 9059, 7731, 389, 279]
   │
   ▼  one forward pass
logits: Float32Array(${VOCAB.toLocaleString()})   // an array of decimals — one raw score per vocab word`}</code>
        </pre>
        <p>
          Those raw scores are called <strong>logits</strong>. Run them through a softmax and you have a probability
          distribution over "what's next" — for Qwen3.5-0.8B, a distribution over all{' '}
          <code>{VOCAB.toLocaleString()}</code> tokens it knows. The model never outputs a word directly; it outputs a
          score for every word it <em>could</em> output, and a separate rule picks one.
        </p>
        <p>
          That "run them through a softmax" step is small enough to do by hand — here it is on a toy five-token
          vocabulary, so you can see exactly how raw scores become probabilities before we scale it up to all{' '}
          <code>{VOCAB.toLocaleString()}</code>:
        </p>
        <LogitsToSoftmax />
        <p>
          The <em>Sharpen ↔ flatten</em> slider scales the raw logits before the softmax: drag it up and the model
          commits harder to its single top guess (the distribution spikes), drag it down and probability spreads more
          evenly across the options. That is exactly the knob{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'sampling' }} search={(prev) => prev}>
            <strong>temperature</strong>
          </Link>{' '}
          tunes when you generate text — here it acts as an inverse temperature (higher = sharper).
        </p>

        <h2>The generation loop</h2>
        <p>
          A single forward pass gives you one token's worth of prediction. To write a sentence, you call the function in
          a loop — sample one token, append it, and run again on the slightly longer list:
        </p>
        <GenerationLoop />
        <p>
          This is what "generating text" means: it is <strong>autoregressive</strong>. Each new token is produced by
          re-running the whole model on a sequence that is one token longer than last time. (Re-running everything every
          step sounds wasteful — the{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'kv-cache' }} search={(prev) => prev}>
            KV-cache chapter
          </Link>{' '}
          is how it's made cheap.)
        </p>

        <h2>The whole pipeline, in one breath</h2>
        <p>
          Inside that one forward pass, a token's journey is: text → <strong>tokens</strong> → <strong>vectors</strong>{' '}
          (embeddings) → a deep stack of <strong>attention + MLP</strong> blocks → a final{' '}
          <strong>score for every token</strong> (the LM head) → <strong>sample</strong> one → append, and loop. Every
          remaining chapter opens up one of those boxes; the{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'architecture' }} search={(prev) => prev}>
            architecture chapter
          </Link>{' '}
          puts them back on one page.
        </p>
        <p>
          Better yet, here is that whole journey as something you can <em>step through</em> — one concrete token, from
          raw text to the next word, with the shape of the data shown at every step:
        </p>
        <TokenJourney />

        <h2>Why "predict the next token" is enough</h2>
        <p>
          It sounds almost too simple to be interesting. The catch is the standard it's held to: to predict the next
          token <em>well</em>, across all of human text — essays, code, dialogue, translation, arithmetic — the model
          has to internalize grammar, facts, and patterns of reasoning, because those are what make the next token
          predictable. Capability is a side effect of getting very good at one narrow objective.
        </p>
        <p>
          One honest caveat up front: what we've described is a <em>base model</em> — pure autocomplete. The jump to a
          ChatGPT-style assistant that follows your instructions is a separate <em>training</em> story, covered later in{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'post-training' }} search={(prev) => prev}>
            "From base model to assistant."
          </Link>
        </p>
      </Prose>
    </ChapterFrame>
  );
}
