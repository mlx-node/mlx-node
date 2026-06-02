import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { ChatTemplateExplorer } from '../widgets/ChatTemplateExplorer';
import { DataScaleBar } from '../widgets/DataScaleBar';
import { PreferencePair } from '../widgets/PreferencePair';
import { TrainingStages } from '../widgets/TrainingStages';

/**
 * Chapter 15 (post-training) — "From base model to assistant". The single
 * biggest gap for a beginner whose mental model is ChatGPT: a base model only
 * autocompletes; instruction tuning + preference tuning + chat templates are
 * what make it follow instructions. Prose + two static widgets (TrainingStages,
 * ChatTemplateExplorer). No model/worker — the chat template is shown literally.
 */

export const learning: ChapterLearningData = {
  chapterId: 'post-training',
  objective:
    'Explain how a base model becomes a helpful assistant — instruction tuning, preference tuning, and the chat template — without changing the architecture.',
  problem:
    'Everything so far describes a base model: pure autocomplete. But the LLM you actually use answers questions and follows instructions. Something bridges the two.',
  minutes: 8,
  glossary: [
    {
      term: 'pretraining',
      definition: 'Next-token prediction on trillions of tokens of generic text — where the model gets its knowledge.',
    },
    {
      term: 'instruction tuning (SFT)',
      definition:
        'Supervised fine-tuning on curated (instruction, response) pairs; teaches the model to answer in the assistant role.',
    },
    {
      term: 'RLHF',
      definition:
        'Reinforcement Learning from Human Feedback — train the model against a reward model built from human preference comparisons.',
    },
    {
      term: 'DPO',
      definition:
        'Direct Preference Optimization — optimize directly on preferred-vs-rejected response pairs, skipping a separate reward model.',
    },
    {
      term: 'chat template',
      definition:
        'The formatting convention that turns a multi-turn conversation into the single token string the model reads.',
    },
    {
      term: 'special token',
      definition:
        'A reserved token (e.g. <|im_start|>, <|im_end|>) that marks structure — turn boundaries, roles, end-of-turn — rather than literal text.',
    },
  ],
  takeaways: [
    'A base model only continues text; instruction tuning is what makes it answer instructions in an assistant role.',
    'Post-training (SFT, then preference tuning) reuses the same model and mostly the same loss — it shapes behavior, it does not add new architecture.',
    'A chat conversation is just a formatted string: special tokens mark whose turn it is; the model generates after a trailing assistant marker and stops at the end-of-turn token.',
  ],
  exercise: {
    prompt:
      "Toggle the chat-template widget to 'Raw text the model sees'. Which special token appears right before the model starts generating, and which one tells it to stop?",
    answer:
      'Generation begins right after <|im_start|>assistant (and a newline); the model stops when it emits <|im_end|>. Instruction tuning is what taught it to produce that end-of-turn token instead of rambling into a fake next turn.',
  },
  quiz: [
    {
      id: 'q1-base-behavior',
      prompt: 'What does a base (pretrained-only) model do if you type a question into it?',
      options: [
        { id: 'a', label: 'Answers it helpfully — answering is built into the architecture.' },
        {
          id: 'b',
          label: 'Continues the text in whatever way is plausible on the web, which may not be a helpful answer.',
        },
        { id: 'c', label: 'Returns an error, because it needs a chat template first.' },
      ],
      correctId: 'b',
      explanation:
        'A base model is autocomplete. Nothing has taught it that a question should be followed by a helpful answer — that behavior comes from instruction tuning.',
    },
    {
      id: 'q2-sft',
      prompt: 'How does instruction tuning (SFT) differ from pretraining?',
      options: [
        { id: 'a', label: 'It uses a completely different model architecture.' },
        {
          id: 'b',
          label: 'It uses the same next-token loss but a small, curated dataset of (instruction, response) pairs.',
        },
        { id: 'c', label: 'It replaces next-token prediction with reinforcement learning.' },
      ],
      correctId: 'b',
      explanation:
        'SFT is the same forward pass and the same cross-entropy loss; only the data changes — curated instruction-following examples.',
    },
    {
      id: 'q3-special-tokens',
      prompt: 'In the chat template, what is the role of special tokens like <|im_start|> and <|im_end|>?',
      options: [
        { id: 'a', label: 'They encrypt the conversation so the model cannot read it.' },
        {
          id: 'b',
          label: 'They mark turn boundaries and roles so a flat token string can encode a multi-turn conversation.',
        },
        { id: 'c', label: 'They are ordinary words the user typed.' },
      ],
      correctId: 'b',
      explanation:
        'The model only ever sees a flat string of tokens; the special tokens delimit turns and signal where the assistant turn begins and ends.',
    },
  ],
};

export function PostTrainingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>From base model to assistant</h1>
        <p>
          Here is a surprise that trips up almost everyone: the model the{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'training' }} search={(prev) => prev}>
            Training chapter
          </Link>{' '}
          described — a pure next-token predictor — would <em>not</em> behave like ChatGPT. Type "What is 2 + 2?" into a
          raw base model and it might continue with another question, or a list of homework problems, or anything that
          plausibly follows that string somewhere on the web. It doesn't <em>answer</em>, because nothing ever taught it
          that a question should be followed by a helpful reply. Three post-training stages fix that — and none of them
          touches the architecture.
        </p>

        <TrainingStages />

        <h2>Instruction tuning (SFT)</h2>
        <p>
          The first fix is the simplest. Take the pretrained model and keep training it — same forward pass, same
          next-token cross-entropy — but on a small, curated dataset of <strong>(instruction, response)</strong> pairs
          written or vetted by humans. After a few thousand to a few million examples, the model has learned the shape
          of the task: when the text so far is a user instruction, the most likely continuation is a helpful response in
          the assistant's voice. That's <strong>supervised fine-tuning</strong>, or SFT. It is tiny next to pretraining
          — it adds almost no new knowledge; it teaches the model how to <em>use</em> what it already knows.
        </p>

        <DataScaleBar />

        <h2>Preference tuning (RLHF / DPO)</h2>
        <p>
          SFT makes the model follow instructions; <strong>preference tuning</strong> makes it follow them <em>well</em>{' '}
          — more helpful, more honest, less likely to produce harmful or evasive answers. Humans compare candidate
          responses ("A is better than B"), and the model is trained to prefer the responses people preferred.{' '}
          <strong>RLHF</strong> does this with a separate <strong>reward model</strong> — a second model trained on the
          human comparisons to predict how much people would like a given response, which the main model is then
          optimized to score well on — and reinforcement learning; <strong>DPO</strong> optimizes the preference
          directly, skipping the reward model. Either way the objective is no longer plain next-token cross-entropy —
          this is the one stage that genuinely departs from the loss you've seen.
        </p>
        <p>
          Pick which response a human prefers and apply the update a few times to see what "training on a preference"
          does:
        </p>
        <PreferencePair />

        <h2>Chat templates: how turns are wired</h2>
        <p>
          Through all of this, the model still only ever sees a <em>flat string of tokens</em> — it has no native idea
          of "messages" or "roles." The <strong>chat template</strong> is the convention that flattens a multi-turn
          conversation into that string, using{' '}
          <Link to="/chapters/$chapterId" params={{ chapterId: 'tokenization' }} search={(prev) => prev}>
            special tokens
          </Link>{' '}
          to mark where each turn starts, who's speaking, and where a turn ends.
        </p>
        <ChatTemplateExplorer />
        <p>
          When you chat with the model in this app, it wraps your conversation in this same format for you, appends a
          trailing <code>{'<|im_start|>assistant'}</code>, and lets the model generate the answer — stopping the moment
          it emits <code>{'<|im_end|>'}</code>. SFT is what taught the model to <em>emit</em> that end-of-turn token
          instead of inventing a fake next user message.
        </p>

        <h2>It is still the same model</h2>
        <p>
          The thing to hold onto: none of this changed the network. Same 24 layers, same attention and MLP blocks, same
          forward pass producing logits for the next token. Post-training only nudged the <em>weights</em> so the
          function's outputs are shaped the way we want, and wrapped the input in a chat format. The "assistant" you
          talk to is the base model from every other chapter — wearing a chat template, with its weights gently steered
          toward being helpful.
        </p>
      </Prose>
    </ChapterFrame>
  );
}
