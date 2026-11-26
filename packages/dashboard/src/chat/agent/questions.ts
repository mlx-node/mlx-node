/**
 * Work: the agent's `_x.ai/ask_user_question` request (`ask_user_question/types.rs`), read and checked; the card it
 * opens; and the reply, built from the page's indexes in the agent's own strings. The agent keys every answer by the
 * exact question text and option label, while the card shows text that is redacted and cut, so shown text is never a
 * key.
 */

import type { QuestionInteraction } from '../types.js';
import { shownCut } from './permissions.js';

/** More questions, options or characters than these: the card is not opened and the agent gets "no answer". */
export const QUESTIONS_MAX = 10;
export const OPTIONS_MAX = 12;
const QUESTION_MAX_CHARS = 1_000;
const LABEL_MAX_CHARS = 200;
const DESCRIPTION_MAX_CHARS = 1_000;
const PREVIEW_MAX_CHARS = 16_384;

/** The label the agent reads for an answer that is only the user's own text (`answers[q] = ["Other"]`). */
export const OTHER_LABEL = 'Other';

export interface QuestionRequest {
  sessionId: string;
  toolCallId: string;
  /** The agent sent `mode: "plan"`. */
  plan: boolean;
  /** As the agent sent them: unique texts, the agent's order. */
  questions: Array<{
    text: string;
    multi: boolean;
    options: Array<{ label: string; description: string; preview: string | null }>;
  }>;
}

export type QuestionRead =
  | { kind: 'ok'; req: QuestionRequest }
  /** A good request over the card's limits. */
  | { kind: 'too-long'; sessionId: string; toolCallId: string };

/** Per question in order: the picked option indexes and the user's own text, or null for none. */
export type QuestionAnswers = Array<{ picks: number[]; other?: string } | null>;

/** The page's answer. */
export type QuestionAnswerInput =
  | { answers: QuestionAnswers }
  /** Skip: the agent reads "declined" and goes on with its best guess. */
  | { reply: 'skip' }
  /**
   * Plan mode only. Talk it over (`discuss`): the agent stops asking and waits for the user's message. Plan now
   * (`plan-now`): it writes the plan with what it has. The answers so far go as labels only.
   */
  | { reply: 'discuss' | 'plan-now'; answers?: QuestionAnswers };

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

type Question = QuestionRequest['questions'][number];

function readOption(raw: unknown): Question['options'][number] | null {
  if (!isPlainObject(raw)) return null;
  const { label, description, preview } = raw;
  if (typeof label !== 'string' || typeof description !== 'string') return null;
  if (preview !== undefined && preview !== null && typeof preview !== 'string') return null;
  return { label, description, preview: typeof preview === 'string' ? preview : null };
}

function readQuestion(raw: unknown): Question | null {
  if (!isPlainObject(raw)) return null;
  const { question, options, multiSelect } = raw;
  if (typeof question !== 'string' || !Array.isArray(options)) return null;
  if (multiSelect !== undefined && multiSelect !== null && typeof multiSelect !== 'boolean') return null;
  const read: Question['options'] = [];
  for (const option of options) {
    const one = readOption(option);
    if (one === null) return null;
    read.push(one);
  }
  return { text: question, multi: multiSelect === true, options: read };
}

function overLimits(questions: readonly Question[]): boolean {
  if (questions.length > QUESTIONS_MAX) return true;
  return questions.some(
    (q) =>
      q.text.length > QUESTION_MAX_CHARS ||
      q.options.length > OPTIONS_MAX ||
      q.options.some(
        (o) =>
          o.label.length > LABEL_MAX_CHARS ||
          o.description.length > DESCRIPTION_MAX_CHARS ||
          (o.preview?.length ?? 0) > PREVIEW_MAX_CHARS,
      ),
  );
}

/** The request, or null when its shape is not the agent's (no questions, two with one text, a wrong type). */
export function readQuestionRequest(params: unknown): QuestionRead | null {
  if (!isPlainObject(params)) return null;
  const { sessionId, toolCallId, questions, mode } = params;
  if (typeof sessionId !== 'string' || sessionId === '' || typeof toolCallId !== 'string' || toolCallId === '') {
    return null;
  }
  if (mode !== 'default' && mode !== 'plan') return null;
  if (!Array.isArray(questions) || questions.length === 0) return null;
  const read: Question[] = [];
  const texts = new Set<string>();
  for (const raw of questions) {
    const one = readQuestion(raw);
    if (one === null || texts.has(one.text)) return null;
    texts.add(one.text);
    read.push(one);
  }
  if (overLimits(read)) return { kind: 'too-long', sessionId, toolCallId };
  return { kind: 'ok', req: { sessionId, toolCallId, plan: mode === 'plan', questions: read } };
}

/** The card: every text redacted, cut to its limit and with hidden characters shown. */
export function questionInteraction(
  id: string,
  req: QuestionRequest,
  needles: readonly string[],
  openedAt: number,
): QuestionInteraction {
  return {
    id,
    kind: 'question',
    toolCallId: req.toolCallId,
    openedAt,
    planMode: req.plan,
    questions: req.questions.map((q) => ({
      text: shownCut(q.text, QUESTION_MAX_CHARS, needles),
      multi: q.multi,
      options: q.options.map((o) => ({
        label: shownCut(o.label, LABEL_MAX_CHARS, needles),
        description: shownCut(o.description, DESCRIPTION_MAX_CHARS, needles),
        preview: o.preview === null ? null : shownCut(o.preview, PREVIEW_MAX_CHARS, needles),
      })),
    })),
  };
}

/** One answered question: the picks in option order and the user's own text, not blank. */
type Read = { question: Question; picks: number[]; own: string };

/**
 * The answered questions in the agent's order, or null when the answers do not fit the card (a count other than the
 * questions, a pick out of range or twice, two picks or a pick and own text on a single-select question).
 */
function readAnswers(req: QuestionRequest, input: QuestionAnswers): Read[] | null {
  if (input.length !== req.questions.length) return null;
  const out: Read[] = [];
  for (const [i, question] of req.questions.entries()) {
    const answer = input[i];
    if (answer === null || answer === undefined) continue;
    const picks = [...answer.picks].sort((a, b) => a - b);
    for (const [n, pick] of picks.entries()) {
      if (!Number.isInteger(pick) || pick < 0 || pick >= question.options.length) return null;
      if (n > 0 && picks[n - 1] === pick) return null;
    }
    const own = answer.other?.trim() ?? '';
    if (!question.multi && (picks.length > 1 || (picks.length === 1 && own !== ''))) return null;
    if (picks.length === 0 && own === '') continue;
    out.push({ question, picks, own });
  }
  return out;
}

function labelsOf(read: Read): string[] {
  return read.picks.length === 0 ? [OTHER_LABEL] : read.picks.map((pick) => read.question.options[pick]!.label);
}

/**
 * The agent's reply for the page's answer, or null when the answer does not fit the card (see `readAnswers`; also
 * nothing answered, or a plan reply on a card not in plan mode). Answered questions only, in the agent's order, keyed
 * by its own texts; own text alone is `["Other"]`, own text rides in `annotations[q].notes`, and a single-select pick
 * of an option with a preview sends that preview back word for word in `annotations[q].preview`. A plan reply carries
 * `partial_answers`: labels only, picks of a multi-select question joined with `, `, own text as "Other", no notes
 * (`ask_user_question/types.rs`).
 */
export function questionReply(req: QuestionRequest, input: QuestionAnswerInput): Record<string, unknown> | null {
  if ('reply' in input && input.reply === 'skip') return { outcome: 'cancelled' };
  if ('reply' in input) {
    if (!req.plan) return null;
    const read = input.answers === undefined ? [] : readAnswers(req, input.answers);
    if (read === null) return null;
    const partial: Record<string, string> = {};
    for (const one of read) partial[one.question.text] = labelsOf(one).join(', ');
    return { outcome: input.reply === 'discuss' ? 'chat_about_this' : 'skip_interview', partial_answers: partial };
  }
  const read = readAnswers(req, input.answers);
  if (read === null || read.length === 0) return null;
  const answers: Record<string, string[]> = {};
  const annotations: Record<string, { preview?: string; notes?: string }> = {};
  for (const one of read) {
    answers[one.question.text] = labelsOf(one);
    const preview = one.question.multi || one.picks.length !== 1 ? null : one.question.options[one.picks[0]!]!.preview;
    if (preview === null && one.own === '') continue;
    annotations[one.question.text] = {
      ...(preview === null ? {} : { preview }),
      ...(one.own === '' ? {} : { notes: one.own }),
    };
  }
  return { outcome: 'accepted', answers, ...(Object.keys(annotations).length > 0 ? { annotations } : {}) };
}
