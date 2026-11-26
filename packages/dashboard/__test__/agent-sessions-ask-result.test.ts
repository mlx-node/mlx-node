/**
 * The scanner of a question call's stored result sentence (`src/agent-sessions/ask-result.ts`), against the exact
 * strings of the agent's `ask_user_question/format.rs`: answers are read with the call's own texts as anchors, an
 * answer that cannot be read one way only is kept as written, and a sentence it does not know stays text.
 */

import { describe, expect, it } from 'vite-plus/test';

import { askOf, readAskQuestions, readAskResult, type ScanQuestion } from '../src/agent-sessions/ask-result.js';

const HEAD = 'User has answered your questions: ';
const TAIL = ". You can now continue with the user's answers in mind.";
const CANCEL =
  'User declined to answer the questions. Continue with the task using your best judgment, or ask different questions.';
const NO_OPERATOR =
  'No user is available to answer questions in this non-interactive session. Continue with your best judgment; do not wait for clarification.';

function q(text: string, labels: string[], multi = false, previews: Record<string, string> = {}): ScanQuestion {
  return { text, multi, options: labels.map((label) => ({ label, preview: previews[label] ?? null })) };
}

/** As `format_accepted_tool_result` writes it. */
function accepted(entries: Array<{ q: string; labels: string[]; preview?: string; notes?: string }>): string {
  const parts = entries.map((e) => {
    const out = [`"${e.q}"="${e.labels.join(', ')}"`];
    if (e.preview !== undefined) out.push(`selected preview:\n${e.preview}`);
    if (e.notes !== undefined) out.push(`user notes: ${e.notes}`);
    return out.join(' ');
  });
  return `${HEAD}${parts.join(', ')}${TAIL}`;
}

const DB = q('Which database?', ['Redis', 'Postgres (Recommended)']);
const FRAMEWORK = q('Which framework?', ['React', 'Vue']);
const FEATURES = q('Which features?', ['Auth', 'Logging', 'Metrics'], true);

describe('readAskResult', () => {
  it('the declined sentence (cancel, Skip, Stop, timeout) and the one with no user are "none"', () => {
    expect(readAskResult(CANCEL, [DB])).toEqual({ kind: 'none' });
    expect(readAskResult(NO_OPERATOR, [DB])).toEqual({ kind: 'none' });
  });

  it('the plan-mode replies', () => {
    const chat =
      'The user wants to clarify these questions.\n    This means they may have additional information, context or questions for you.\n' +
      '    Take their response into account and then reformulate the questions if appropriate.\n' +
      '    Start by asking them what they would like to clarify.\n\n    Questions asked:\n- "Which database?"\n  Answer: Redis';
    const skip =
      'The user has indicated they have provided enough answers for the plan interview.\n' +
      'Stop asking clarifying questions and proceed to finish the plan with the information you have.\n\n' +
      'Questions asked and answers provided:\n- "Which database?"\n  (No answer provided)';
    expect(readAskResult(chat, [DB])).toEqual({ kind: 'discuss' });
    expect(readAskResult(skip, [DB])).toEqual({ kind: 'plan-now' });
  });

  it('accepted: one pick, a pick with a preview, notes, multi, own text alone, and an unanswered question', () => {
    expect(readAskResult(accepted([{ q: 'Which database?', labels: ['Postgres (Recommended)'] }]), [DB])).toEqual({
      kind: 'answered',
      answers: [{ picks: ['Postgres (Recommended)'], other: null }],
    });
    const db = q('Which database?', ['Redis', 'Postgres'], false, { Redis: '<div>redis preview</div>' });
    expect(
      readAskResult(
        accepted([
          { q: 'Which database?', labels: ['Redis'], preview: '<div>redis preview</div>' },
          { q: 'Which framework?', labels: ['React'], notes: 'I prefer React hooks' },
        ]),
        [db, FRAMEWORK],
      ),
    ).toEqual({
      kind: 'answered',
      answers: [
        { picks: ['Redis'], other: null },
        { picks: ['React'], other: 'I prefer React hooks' },
      ],
    });
    expect(
      readAskResult(
        accepted([
          { q: 'Which features?', labels: ['Auth', 'Logging'], notes: 'and audit' },
          { q: 'Which database?', labels: ['Other'], notes: 'I want to use DynamoDB' },
        ]),
        [FEATURES, FRAMEWORK, DB],
      ),
    ).toEqual({
      kind: 'answered',
      answers: [
        { picks: ['Auth', 'Logging'], other: 'and audit' },
        null,
        { picks: [], other: 'I want to use DynamoDB' },
      ],
    });
    const layout = q('Which layout?', ['Grid', 'Flex'], false, { Grid: '<div class="grid">...</div>' });
    expect(
      readAskResult(
        accepted([
          {
            q: 'Which layout?',
            labels: ['Grid'],
            preview: '<div class="grid">...</div>',
            notes: 'Use CSS Grid for the main layout',
          },
        ]),
        [layout],
      ),
    ).toEqual({ kind: 'answered', answers: [{ picks: ['Grid'], other: 'Use CSS Grid for the main layout' }] });
  });

  it('labels, questions and notes that hold ", ", quotes and = are read by their exact texts', () => {
    const tricky = q('Where does the "forked" build, = come from?', ['A, B', 'A', 'B "quoted"', 'C=D'], true);
    const next = q('Then?', ['Go', 'Stop']);
    expect(
      readAskResult(
        accepted([
          { q: tricky.text, labels: ['A, B', 'C=D'], notes: 'see "x", "y"=z, and more' },
          { q: 'Then?', labels: ['Go'] },
        ]),
        [tricky, next],
      ),
    ).toEqual({
      kind: 'answered',
      answers: [
        { picks: ['A, B', 'C=D'], other: 'see "x", "y"=z, and more' },
        { picks: ['Go'], other: null },
      ],
    });
    // A label that is the start of another: the longer one is tried first, and only a whole reading counts.
    const start = q('Pick?', ['Unit', 'Unit tests']);
    expect(readAskResult(accepted([{ q: 'Pick?', labels: ['Unit tests'] }]), [start])).toEqual({
      kind: 'answered',
      answers: [{ picks: ['Unit tests'], other: null }],
    });
    expect(readAskResult(accepted([{ q: 'Pick?', labels: ['Unit'] }]), [start])).toEqual({
      kind: 'answered',
      answers: [{ picks: ['Unit'], other: null }],
    });
  });

  it('an answer that reads two ways is kept as written; a pick is never guessed', () => {
    // "A, B" is the label "A, B" or the labels "A" and "B".
    const two = q('Which?', ['A', 'B', 'A, B'], true);
    const text = accepted([{ q: 'Which?', labels: ['A, B'] }]);
    expect(readAskResult(text, [two])).toEqual({ kind: 'answered', answers: [{ raw: 'A, B"' }] });
  });

  it('an answer that is not a list of the known labels is kept as written; the other questions still read', () => {
    const text = accepted([
      { q: 'Which database?', labels: ['Mongo'] },
      { q: 'Which framework?', labels: ['Vue'] },
    ]);
    expect(readAskResult(text, [DB, FRAMEWORK])).toEqual({
      kind: 'answered',
      answers: [{ raw: 'Mongo"' }, { picks: ['Vue'], other: null }],
    });
  });

  it('a sentence it does not know, or an accepted one with no question it can anchor, stays text', () => {
    const sent = 'Your questions have been presented to the user for answering:\n1. Which database? [options: Redis]';
    expect(readAskResult(sent, [DB])).toEqual({ kind: 'text', text: sent });
    const empty = `${HEAD}${TAIL}`;
    expect(readAskResult(empty, [DB])).toEqual({ kind: 'text', text: empty });
    const renamed = accepted([{ q: 'Which db?', labels: ['Redis'] }]);
    expect(readAskResult(renamed, [DB])).toEqual({ kind: 'text', text: renamed });
  });
});

describe('askOf', () => {
  it('reads the model input (multi_select, also as the agent leniently reads it) and the agent copy (multiSelect)', () => {
    expect(
      readAskQuestions({
        questions: [
          { question: 'a', options: [{ label: 'x', description: '' }], multi_select: 'yes' },
          { question: 'b', options: [], multi_select: true },
          { question: 'c', options: [{ label: 'y', description: '', preview: 'p' }] },
        ],
      }),
    ).toEqual([
      { text: 'a', multi: true, options: [{ label: 'x', preview: null }] },
      { text: 'b', multi: true, options: [] },
      { text: 'c', multi: false, options: [{ label: 'y', preview: 'p' }] },
    ]);
    expect(
      readAskQuestions({ variant: 'AskUserQuestion', questions: [{ question: 'a', options: [], multiSelect: null }] }),
    ).toEqual([{ text: 'a', multi: false, options: [] }]);
    expect(readAskQuestions({})).toBeNull();
    expect(readAskQuestions({ questions: [{ options: [] }] })).toBeNull();
  });

  it('no result before the call completes, nor for a failed call', () => {
    const input = { questions: [{ question: 'Which database?', options: [{ label: 'Redis', description: '' }] }] };
    expect(askOf(input, 'in_progress', null)).toEqual({
      questions: [{ text: 'Which database?', multi: false, options: ['Redis'] }],
      result: null,
    });
    expect(askOf(input, 'failed', 'Client returned an invalid response to user question: x')?.result).toBeNull();
    expect(askOf(input, 'completed', CANCEL)?.result).toEqual({ kind: 'none' });
  });
});
