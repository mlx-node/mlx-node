/**
 * The question request reader, the card it opens and the reply built from the page's indexes
 * (`src/chat/agent/questions.ts`), plus the answer route body (`interactionInput`).
 */

import { describe, expect, it } from 'vite-plus/test';

import {
  OPTIONS_MAX,
  QUESTIONS_MAX,
  questionInteraction,
  questionReply,
  readQuestionRequest,
  type QuestionRequest,
} from '../src/chat/agent/questions.js';
import { REDACTED } from '../src/chat/redact.js';
import { interactionInput } from '../src/chat/route-input.js';

const KEY = 'exa-KEY-0123456789abcdef';

/** The request as the agent sends it (`AskUserQuestionExtRequest`, camelCase; `multiSelect` null when unset). */
function wire(questions: unknown[], mode = 'default'): Record<string, unknown> {
  return { sessionId: 's1', toolCallId: 'call_1', questions, mode };
}

const DB = {
  question: 'Which database?',
  options: [
    { label: 'Postgres (Recommended)', description: 'Relational.' },
    { label: 'Redis', description: 'In memory.', preview: 'cache = redis()' },
  ],
  multiSelect: null,
};
const CHECKS = {
  question: 'Which checks?',
  options: [
    { label: 'Unit tests', description: '' },
    { label: 'Type check', description: '' },
    { label: 'Lint', description: '' },
  ],
  multiSelect: true,
};

function req(questions: unknown[], mode = 'default'): QuestionRequest {
  const read = readQuestionRequest(wire(questions, mode));
  if (read?.kind !== 'ok') throw new Error('not read');
  return read.req;
}

describe('readQuestionRequest', () => {
  it('reads the real shape: multiSelect null or missing, preview missing or null, an id, plan mode', () => {
    const read = readQuestionRequest(
      wire(
        [
          DB,
          { ...CHECKS, id: 'q2', options: CHECKS.options.map((o) => ({ ...o, preview: null })) },
          { question: 'Anything else?', options: [] },
        ],
        'plan',
      ),
    );
    expect(read).toEqual({
      kind: 'ok',
      req: {
        sessionId: 's1',
        toolCallId: 'call_1',
        plan: true,
        questions: [
          {
            text: 'Which database?',
            multi: false,
            options: [
              { label: 'Postgres (Recommended)', description: 'Relational.', preview: null },
              { label: 'Redis', description: 'In memory.', preview: 'cache = redis()' },
            ],
          },
          {
            text: 'Which checks?',
            multi: true,
            options: [
              { label: 'Unit tests', description: '', preview: null },
              { label: 'Type check', description: '', preview: null },
              { label: 'Lint', description: '', preview: null },
            ],
          },
          { text: 'Anything else?', multi: false, options: [] },
        ],
      },
    });
  });

  it('refuses shapes the agent never sends', () => {
    expect(readQuestionRequest(null)).toBeNull();
    expect(readQuestionRequest([])).toBeNull();
    expect(readQuestionRequest({ ...wire([DB]), sessionId: '' })).toBeNull();
    expect(readQuestionRequest({ ...wire([DB]), toolCallId: 7 })).toBeNull();
    expect(readQuestionRequest(wire([DB], 'auto'))).toBeNull();
    expect(readQuestionRequest({ sessionId: 's1', toolCallId: 'c', questions: [DB] })).toBeNull();
    expect(readQuestionRequest(wire([]))).toBeNull();
    expect(readQuestionRequest(wire([DB, DB]))).toBeNull();
    expect(readQuestionRequest(wire([{ ...DB, question: 3 }]))).toBeNull();
    expect(readQuestionRequest(wire([{ ...DB, options: 'a' }]))).toBeNull();
    expect(readQuestionRequest(wire([{ ...DB, multiSelect: 'yes' }]))).toBeNull();
    expect(readQuestionRequest(wire([{ ...DB, options: [{ label: 'a' }] }]))).toBeNull();
    expect(readQuestionRequest(wire([{ ...DB, options: [{ label: 'a', description: '', preview: 1 }] }]))).toBeNull();
  });

  it('a request over the limits is too long: questions, options, and each text', () => {
    const tooLong = { kind: 'too-long', sessionId: 's1', toolCallId: 'call_1' };
    const q = (i: number) => ({ question: `Q${i}?`, options: [] });
    expect(readQuestionRequest(wire(Array.from({ length: QUESTIONS_MAX }, (_, i) => q(i))))?.kind).toBe('ok');
    expect(readQuestionRequest(wire(Array.from({ length: QUESTIONS_MAX + 1 }, (_, i) => q(i))))).toEqual(tooLong);
    const options = (n: number) => Array.from({ length: n }, (_, i) => ({ label: `o${i}`, description: '' }));
    expect(readQuestionRequest(wire([{ question: 'q', options: options(OPTIONS_MAX) }]))?.kind).toBe('ok');
    expect(readQuestionRequest(wire([{ question: 'q', options: options(OPTIONS_MAX + 1) }]))).toEqual(tooLong);
    expect(readQuestionRequest(wire([{ question: 'q'.repeat(1_000), options: [] }]))?.kind).toBe('ok');
    expect(readQuestionRequest(wire([{ question: 'q'.repeat(1_001), options: [] }]))).toEqual(tooLong);
    const option = (o: Record<string, unknown>) =>
      wire([{ question: 'q', options: [{ label: 'a', description: '', ...o }] }]);
    expect(readQuestionRequest(option({ label: 'l'.repeat(200) }))?.kind).toBe('ok');
    expect(readQuestionRequest(option({ label: 'l'.repeat(201) }))).toEqual(tooLong);
    expect(readQuestionRequest(option({ description: 'd'.repeat(1_001) }))).toEqual(tooLong);
    expect(readQuestionRequest(option({ preview: 'p'.repeat(16_384) }))?.kind).toBe('ok');
    expect(readQuestionRequest(option({ preview: 'p'.repeat(16_385) }))).toEqual(tooLong);
  });
});

describe('questionInteraction', () => {
  it('redacts keys and shows hidden characters in every text; the tool call id stays', () => {
    const request = req([
      {
        question: `Use ${KEY}?\u202E`,
        options: [{ label: `a ${KEY}`, description: `b\u200B ${KEY}`, preview: `${KEY}\nx` }],
        multiSelect: true,
      },
    ]);
    const card = questionInteraction('i1', request, [KEY], 42);
    expect(card).toEqual({
      id: 'i1',
      kind: 'question',
      toolCallId: 'call_1',
      openedAt: 42,
      planMode: false,
      questions: [
        {
          text: `Use ${REDACTED}?\\u{202E}`,
          multi: true,
          options: [{ label: `a ${REDACTED}`, description: `b\\u{200B} ${REDACTED}`, preview: `${REDACTED}\nx` }],
        },
      ],
    });
    expect(JSON.stringify(card)).not.toContain(KEY);
  });
});

describe('questionReply', () => {
  const request = req([DB, CHECKS, { question: 'Anything else?', options: [] }]);

  it('builds the agent reply by index: labels, own text as notes, "Other" for own text alone, question order', () => {
    expect(
      questionReply(request, {
        answers: [{ picks: [1] }, { picks: [2, 0], other: ' also e2e ' }, { picks: [], other: 'no' }],
      }),
    ).toEqual({
      outcome: 'accepted',
      answers: { 'Which database?': ['Redis'], 'Which checks?': ['Unit tests', 'Lint'], 'Anything else?': ['Other'] },
      annotations: {
        'Which database?': { preview: 'cache = redis()' },
        'Which checks?': { notes: 'also e2e' },
        'Anything else?': { notes: 'no' },
      },
    });
    const reply = questionReply(request, { answers: [null, { picks: [1] }, { picks: [], other: '  ' }] });
    expect(reply).toEqual({ outcome: 'accepted', answers: { 'Which checks?': ['Type check'] } });
    expect(Object.keys((reply as { answers: object }).answers)).toEqual(['Which checks?']);
    expect(
      Object.keys(questionReply(request, { answers: [{ picks: [0] }, { picks: [0] }, null] })!.answers as object),
    ).toEqual(['Which database?', 'Which checks?']);
  });

  it('keys by the agent texts, never the redacted ones shown on the card', () => {
    const secret = req([{ question: `Push with ${KEY}?`, options: [{ label: `Yes, ${KEY}`, description: '' }] }]);
    const card = questionInteraction('i', secret, [KEY], 0);
    expect(card.questions[0]!.text).not.toContain(KEY);
    expect(questionReply(secret, { answers: [{ picks: [0] }] })).toEqual({
      outcome: 'accepted',
      answers: { [`Push with ${KEY}?`]: [`Yes, ${KEY}`] },
    });
  });

  it('a single-select pick sends its option preview back word for word; a multi pick and own text never do', () => {
    const preview = 'cache = redis()\n  .ttl(60) "quoted", <b>x</b>';
    const layouts = req([
      { ...DB, options: [DB.options[0], { ...DB.options[1], preview }] },
      {
        ...CHECKS,
        options: CHECKS.options.map((o) => ({ ...o, preview: `of ${o.label}` })),
      },
    ]);
    expect(questionReply(layouts, { answers: [{ picks: [1] }, { picks: [0, 1] }] })).toEqual({
      outcome: 'accepted',
      answers: { 'Which database?': ['Redis'], 'Which checks?': ['Unit tests', 'Type check'] },
      annotations: { 'Which database?': { preview } },
    });
    // An option with no preview, and own text alone, send none.
    expect(questionReply(layouts, { answers: [{ picks: [0] }, null] })).toEqual({
      outcome: 'accepted',
      answers: { 'Which database?': ['Postgres (Recommended)'] },
    });
    expect(
      questionReply(layouts, {
        answers: [
          { picks: [], other: 'Dynamo' },
          { picks: [2], other: 'and e2e' },
        ],
      }),
    ).toEqual({
      outcome: 'accepted',
      answers: { 'Which database?': ['Other'], 'Which checks?': ['Lint'] },
      annotations: { 'Which database?': { notes: 'Dynamo' }, 'Which checks?': { notes: 'and e2e' } },
    });
    // The agent's own preview goes back, never the redacted one shown on the card.
    const secret = req([{ question: 'Key?', options: [{ label: 'A', description: '', preview: `use ${KEY}` }] }]);
    expect(questionInteraction('i', secret, [KEY], 0).questions[0]!.options[0]!.preview).not.toContain(KEY);
    expect(questionReply(secret, { answers: [{ picks: [0] }] })).toEqual({
      outcome: 'accepted',
      answers: { 'Key?': ['A'] },
      annotations: { 'Key?': { preview: `use ${KEY}` } },
    });
  });

  it('plan mode: Talk it over and Plan now send the answers so far as labels only, no notes', () => {
    const planned = req([DB, CHECKS, { question: 'Anything else?', options: [] }], 'plan');
    const so = [{ picks: [1] }, { picks: [2, 0], other: 'and e2e' }, { picks: [], other: 'no' }];
    expect(questionReply(planned, { reply: 'discuss', answers: so })).toEqual({
      outcome: 'chat_about_this',
      partial_answers: { 'Which database?': 'Redis', 'Which checks?': 'Unit tests, Lint', 'Anything else?': 'Other' },
    });
    expect(questionReply(planned, { reply: 'plan-now', answers: [null, { picks: [1] }, null] })).toEqual({
      outcome: 'skip_interview',
      partial_answers: { 'Which checks?': 'Type check' },
    });
    // Nothing answered yet is still a plan reply.
    expect(questionReply(planned, { reply: 'discuss', answers: [null, null, null] })).toEqual({
      outcome: 'chat_about_this',
      partial_answers: {},
    });
    expect(questionReply(planned, { reply: 'plan-now' })).toEqual({ outcome: 'skip_interview', partial_answers: {} });
    expect(questionReply(planned, { reply: 'skip' })).toEqual({ outcome: 'cancelled' });
    // Answers that do not fit the card, and a plan reply on a card not in plan mode, are no answer.
    expect(questionReply(planned, { reply: 'discuss', answers: [{ picks: [0, 1] }, null, null] })).toBeNull();
    expect(questionReply(planned, { reply: 'discuss', answers: [{ picks: [0] }] })).toBeNull();
    expect(questionReply(planned, { reply: 'plan-now', answers: [{ picks: [5] }, null, null] })).toBeNull();
    expect(questionReply(request, { reply: 'discuss', answers: [{ picks: [0] }, null, null] })).toBeNull();
    expect(questionReply(request, { reply: 'plan-now' })).toBeNull();
  });

  it('Skip is "no answer"; an answer that does not fit the card is null', () => {
    expect(questionReply(request, { reply: 'skip' })).toEqual({ outcome: 'cancelled' });
    expect(questionReply(request, { answers: [null, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [0] }] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [2] }, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [-1] }, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [0.5] }, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [0, 1] }, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [{ picks: [0], other: 'x' }, null, null] })).toBeNull();
    expect(questionReply(request, { answers: [null, { picks: [1, 1] }, null] })).toBeNull();
    expect(questionReply(request, { answers: [null, null, { picks: [0] }] })).toBeNull();
  });
});

describe('the answer route body', () => {
  it('takes {answers}, {reply:"skip"} and {reply:"discuss"|"plan-now", answers?} exactly', () => {
    expect(interactionInput({ answers: [{ picks: [0] }, null, { picks: [], other: 'x' }] })).toEqual({
      answers: [{ picks: [0] }, null, { picks: [], other: 'x' }],
    });
    expect(interactionInput({ reply: 'skip' })).toEqual({ reply: 'skip' });
    expect(interactionInput({ reply: 'discuss' })).toEqual({ reply: 'discuss' });
    expect(interactionInput({ reply: 'plan-now', answers: [null, { picks: [1], other: 'x' }] })).toEqual({
      reply: 'plan-now',
      answers: [null, { picks: [1], other: 'x' }],
    });
    const bad = (body: unknown) => () => interactionInput(body);
    expect(bad({ reply: 'chat_about_this' })).toThrow();
    expect(bad({ reply: 'discuss', answers: [{ picks: [1, 1] }] })).toThrow();
    expect(bad({ reply: 'plan-now', answers: [] })).toThrow();
    expect(bad({ reply: 'discuss', extra: 1 })).toThrow();
    expect(bad({ reply: 'skip', answers: [null] })).toThrow();
    expect(bad({ reply: 'skip', answers: [] })).toThrow();
    expect(bad({ answers: [] })).toThrow();
    expect(bad({ answers: Array.from({ length: QUESTIONS_MAX + 1 }, () => null) })).toThrow();
    expect(bad({ answers: [{ picks: [0], extra: 1 }] })).toThrow();
    expect(bad({ answers: [{ other: 'x' }] })).toThrow();
    expect(bad({ answers: [{ picks: ['0'] }] })).toThrow();
    expect(bad({ answers: [{ picks: [1, 1] }] })).toThrow();
    expect(bad({ answers: [{ picks: [OPTIONS_MAX] }] })).toThrow();
    expect(bad({ answers: [{ picks: [], other: 'x'.repeat(4_001) }] })).toThrow();
    expect(bad({ answers: [{ picks: [], other: '\uD800' }] })).toThrow();
    expect(bad({ answers: ['x'] })).toThrow();
  });
});
