/**
 * `ask_user_question`: the questions of the model's call and what its result sentence says. The agent stores only the
 * sentence (`ask_user_question/format.rs`), written as `"Q"="A, B"` with no escaping, picks and entries joined with
 * `, `, so the sentence is read with the call's own question texts and option labels as exact anchors, never split on
 * quotes or commas. An answer that cannot be read in exactly one way is kept as written: a pick is never guessed.
 */

import type { TranscriptEntry } from './types.js';

type Ask = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['ask']>;
export type AskResult = NonNullable<Ask['result']>;
type Answer = Extract<AskResult, { kind: 'answered' }>['answers'][number];
type Picked = { picks: string[]; other: string | null };

/** A question as the call asked it: the exact text and labels, and each option's preview (echoed on a pick). */
export interface ScanQuestion {
  text: string;
  multi: boolean;
  options: Array<{ label: string; preview: string | null }>;
}

const CANCEL_TEXT =
  'User declined to answer the questions. Continue with the task using your best judgment, or ask different questions.';
const NO_OPERATOR_TEXT =
  'No user is available to answer questions in this non-interactive session. Continue with your best judgment; do not wait for clarification.';
const ACCEPTED_HEAD = 'User has answered your questions: ';
const ACCEPTED_TAIL = ". You can now continue with the user's answers in mind.";
const DISCUSS_HEAD = 'The user wants to clarify these questions.';
const PLAN_NOW_HEAD = 'The user has indicated they have provided enough answers for the plan interview.';
const PREVIEW = ' selected preview:\n';
const NOTES = ' user notes: ';
const SEP = ', ';
const OTHER = 'Other';
/** Steps of the search before a sentence counts as ambiguous. */
const BUDGET = 20_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** The agent's lenient bool (`serde_lenient.rs`): true, "true", "yes", "1", 1. */
function lenientTrue(value: unknown): boolean {
  if (value === true || value === 1) return true;
  if (typeof value !== 'string') return false;
  const word = value.trim().toLowerCase();
  return word === 'true' || word === 'yes' || word === '1';
}

/** The questions of the call's `rawInput` (the model's input, or the agent's own copy of it); null when unreadable. */
export function readAskQuestions(rawInput: unknown): ScanQuestion[] | null {
  if (!isRecord(rawInput) || !Array.isArray(rawInput.questions) || rawInput.questions.length === 0) return null;
  const out: ScanQuestion[] = [];
  for (const raw of rawInput.questions) {
    if (!isRecord(raw) || typeof raw.question !== 'string') return null;
    const options: ScanQuestion['options'] = [];
    for (const option of Array.isArray(raw.options) ? raw.options : []) {
      if (!isRecord(option) || typeof option.label !== 'string') return null;
      options.push({ label: option.label, preview: typeof option.preview === 'string' ? option.preview : null });
    }
    out.push({
      text: raw.question,
      multi: lenientTrue(raw.multiSelect) || lenientTrue(raw.multi_select),
      options,
    });
  }
  return out;
}

class Scan {
  private steps = 0;
  private readonly anchors: string[];
  private readonly body: string;
  private readonly questions: readonly ScanQuestion[];

  constructor(body: string, questions: readonly ScanQuestion[]) {
    this.body = body;
    this.questions = questions;
    this.anchors = questions.map((q) => `"${q.text}"="`);
  }

  get spent(): boolean {
    return this.steps > BUDGET;
  }

  private tick(): boolean {
    this.steps += 1;
    return this.steps <= BUDGET;
  }

  /** An entry of a question after `j` starts at `at`. */
  private laterEntry(at: number, j: number): boolean {
    for (let k = j + 1; k < this.anchors.length; k += 1) if (this.body.startsWith(this.anchors[k]!, at)) return true;
    return false;
  }

  /**
   * Where the user's own text of question `j` ends: at the first `, ` before an entry of a later question, else at the
   * body's end. A text read short would have to hold `, "<a later question>"="` word for word.
   */
  private notesEnd(from: number, j: number): number {
    for (let p = from; p < this.body.length; p += 1) {
      if (this.body.startsWith(SEP, p) && this.laterEntry(p + SEP.length, j)) return p;
    }
    return this.body.length;
  }

  /** Label lists of question `j` from `at`, each ending at its closing quote. Labels longest first. */
  private labelLists(at: number, j: number): Array<{ labels: string[]; quote: number }> {
    const question = this.questions[j]!;
    const labels = [...new Set(question.options.map((o) => o.label))]
      .filter((l) => l !== '')
      .sort((a, b) => b.length - a.length);
    const out: Array<{ labels: string[]; quote: number }> = [];
    if (!labels.includes(OTHER) && this.body.startsWith(`${OTHER}"`, at)) {
      out.push({ labels: [OTHER], quote: at + OTHER.length });
    }
    const walk = (p: number, seq: string[]): void => {
      if (!this.tick()) return;
      for (const label of labels) {
        if (!this.body.startsWith(label, p)) continue;
        const next = p + label.length;
        if (this.body[next] === '"') out.push({ labels: [...seq, label], quote: next });
        if (this.body.startsWith(SEP, next)) walk(next + SEP.length, [...seq, label]);
      }
    };
    walk(at, []);
    return out;
  }

  /** Each way the answer of question `j` can be read from `at`: the answer and where it ends. */
  values(at: number, j: number): Array<{ answer: Picked; end: number }> {
    const question = this.questions[j]!;
    const out: Array<{ answer: Picked; end: number }> = [];
    for (const list of this.labelLists(at, j)) {
      const after = list.quote + 1;
      const starts = [after];
      if (this.body.startsWith(PREVIEW, after) && list.labels.length === 1) {
        const option = question.options.find((o) => o.label === list.labels[0]);
        const preview = option?.preview ?? null;
        if (preview !== null && this.body.startsWith(preview, after + PREVIEW.length)) {
          starts.push(after + PREVIEW.length + preview.length);
        }
      }
      for (const from of starts) {
        out.push({ answer: picked(list.labels, null, question), end: from });
        if (!this.body.startsWith(NOTES, from)) continue;
        const notesAt = from + NOTES.length;
        const end = this.notesEnd(notesAt, j);
        out.push({ answer: picked(list.labels, this.body.slice(notesAt, end), question), end });
      }
    }
    return out;
  }

  /** Up to two whole readings of the body: entries in question order, each anchored on its exact text. */
  whole(): Array<Array<Picked | null>> {
    const out: Array<Array<Picked | null>> = [];
    const entries = (at: number, from: number, acc: Array<Picked | null>): void => {
      if (out.length >= 2 || !this.tick()) return;
      for (let j = from; j < this.questions.length; j += 1) {
        if (!this.body.startsWith(this.anchors[j]!, at)) continue;
        for (const value of this.values(at + this.anchors[j]!.length, j)) {
          const next = [...acc];
          next[j] = value.answer;
          if (value.end === this.body.length) {
            out.push(next);
            if (out.length >= 2) return;
          } else if (this.body.startsWith(SEP, value.end)) {
            entries(value.end + SEP.length, j + 1, next);
          }
        }
      }
    };
    entries(
      0,
      0,
      this.questions.map(() => null),
    );
    return out;
  }

  /** The first entry start of each question, in order: at the body's start or after `, `. */
  anchored(): Array<{ j: number; start: number }> {
    const out: Array<{ j: number; start: number }> = [];
    let cursor = 0;
    for (const [j, anchor] of this.anchors.entries()) {
      let p = this.body.indexOf(anchor, cursor);
      while (p !== -1 && p !== 0 && !this.body.startsWith(SEP, p - SEP.length)) p = this.body.indexOf(anchor, p + 1);
      if (p === -1) continue;
      out.push({ j, start: p });
      cursor = p + anchor.length;
    }
    return out;
  }

  anchorLength(j: number): number {
    return this.anchors[j]!.length;
  }
}

/** Own text alone is the label "Other" with notes, unless the question has an option of that name. */
function picked(labels: string[], notes: string | null, question: ScanQuestion): Picked {
  const own = labels.length === 1 && labels[0] === OTHER && !question.options.some((o) => o.label === OTHER);
  return own && notes !== null ? { picks: [], other: notes } : { picks: labels, other: notes };
}

/** The accepted body read per question: exact where only one reading fits, else the answer as written. */
function readAccepted(body: string, questions: readonly ScanQuestion[]): AskResult | null {
  const scan = new Scan(body, questions);
  const whole = scan.whole();
  if (whole.length === 1 && !scan.spent) return { kind: 'answered', answers: whole[0]! };
  const found = scan.anchored();
  if (found.length === 0) return null;
  const answers: Answer[] = questions.map(() => null);
  for (const [n, entry] of found.entries()) {
    const at = entry.start + scan.anchorLength(entry.j);
    const end = n + 1 < found.length ? found[n + 1]!.start - SEP.length : body.length;
    const raw = { raw: body.slice(at, Math.max(at, end)) };
    if (whole.length > 1 || scan.spent) {
      answers[entry.j] = raw;
      continue;
    }
    const fits = scan.values(at, entry.j).filter((value) => value.end === end);
    answers[entry.j] = fits.length === 1 ? fits[0]!.answer : raw;
  }
  return { kind: 'answered', answers };
}

/** What a stored result sentence of the call says. */
export function readAskResult(text: string, questions: readonly ScanQuestion[]): AskResult {
  if (text === CANCEL_TEXT || text === NO_OPERATOR_TEXT) return { kind: 'none' };
  if (text.startsWith(DISCUSS_HEAD)) return { kind: 'discuss' };
  if (text.startsWith(PLAN_NOW_HEAD)) return { kind: 'plan-now' };
  if (
    text.startsWith(ACCEPTED_HEAD) &&
    text.endsWith(ACCEPTED_TAIL) &&
    text.length > ACCEPTED_HEAD.length + ACCEPTED_TAIL.length
  ) {
    const read = readAccepted(text.slice(ACCEPTED_HEAD.length, text.length - ACCEPTED_TAIL.length), questions);
    if (read !== null) return read;
  }
  return { kind: 'text', text };
}

/** The call's `ask`: its questions, and the result once the call completed. */
export function askOf(rawInput: unknown, status: string, content: string | null): Ask | null {
  const questions = readAskQuestions(rawInput);
  if (questions === null) return null;
  const result =
    status === 'completed' && content !== null && content !== '' ? readAskResult(content, questions) : null;
  return {
    questions: questions.map((q) => ({ text: q.text, multi: q.multi, options: q.options.map((o) => o.label) })),
    result,
  };
}
