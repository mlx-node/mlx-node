/** Text decision requests compatible with TypeSafe System One. */
export type ClefValue = null | boolean | number | string | ClefValue[] | { [key: string]: ClefValue };
export type ClefQuestion =
  | { type: 'noul'; instructions?: ClefValue; criteria?: { true?: ClefValue; false?: ClefValue } }
  | { type: 'choice'; instructions?: ClefValue; criteria: Record<string, ClefValue> }
  | { type: 'score'; instructions?: ClefValue; criteria: ClefValue[] };
export interface ClefRequest {
  state: ClefValue;
  questions: Record<string, ClefQuestion>;
}
export type ClefAnswer =
  | { type: 'noul'; noul: number }
  | { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number }
  | {
      type: 'score';
      score: number;
      probabilities: Record<string, number>;
      legend: Record<string, string>;
      confidence: number;
    };
export interface ClefResult {
  answers: Record<string, ClefAnswer>;
  usage: { input_tokens: number; output_tokens: 0 };
}
export interface DecisionModel {
  readonly kind: 'decision';
  /** Raw JSON retains the caller's question ordering. */
  decideRaw(request: string, options?: { signal?: AbortSignal }): Promise<ClefResult>;
}
export interface ClefNativeResult {
  input_tokens: number;
  questions: { id: string; type: number; options: string[]; probabilities: number[] }[];
}

/** The official Jev formulas; Cloudflare's Python helper uses max(p) instead. */
export function formatClefResult(native: ClefNativeResult, request: ClefRequest): ClefResult {
  const answers: Record<string, ClefAnswer> = Object.create(null);
  for (const row of native.questions) {
    const p = row.probabilities;
    if (
      p.length < 1 ||
      p.length !== row.options.length ||
      p.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
      Math.abs(p.reduce((a, b) => a + b, 0) - 1) > 1e-5
    ) {
      throw new Error('Invalid CLEF probability distribution');
    }
    const mode = p.indexOf(Math.max(...p));
    const probabilities = Object.fromEntries(row.options.map((id, i) => [id, p[i]]));
    if (row.type === 0) answers[row.id] = { type: 'noul', noul: p[0] };
    else if (row.type === 1)
      answers[row.id] = {
        type: 'choice',
        choice: row.options[mode],
        probabilities,
        confidence: p.length === 1 ? 1 : Math.min(1, Math.max(0, (p.length * p[mode] - 1) / (p.length - 1))),
      };
    else {
      const question = request.questions[row.id];
      if (question.type !== 'score') throw new Error('CLEF returned a mismatched question type');
      const spread = p.reduce((sum, prob, i) => sum + prob * Math.abs(i - mode), 0);
      const evenSpread = p.reduce((sum, _, i) => sum + Math.abs(i - (p.length - 1) / 2), 0) / p.length;
      answers[row.id] = {
        type: 'score',
        score: p.reduce((sum, prob, i) => sum + i * prob, 0),
        probabilities,
        legend: Object.fromEntries(
          question.criteria.map((v, i) => [String(i), typeof v === 'string' ? v : JSON.stringify(v)]),
        ),
        confidence: Math.max(0, 1 - spread / evenSpread),
      };
    }
  }
  return { answers, usage: { input_tokens: native.input_tokens, output_tokens: 0 } };
}

export function validateClefRequest(body: unknown): asserts body is ClefRequest {
  const object = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);
  if (!object(body) || !Object.hasOwn(body, 'state') || !object(body.questions))
    throw new Error('state and a questions object are required');
  if (Object.keys(body).some((k) => !['state', 'questions', 'model'].includes(k)))
    throw new Error('Unsupported CLEF field; images and video are not supported');
  const questions = Object.values(body.questions);
  if (questions.length < 1 || questions.length > 256) throw new Error('Supply 1 to 256 questions');
  for (const q of questions) {
    if (!object(q) || Object.keys(q).some((k) => !['type', 'instructions', 'criteria'].includes(k)))
      throw new Error('Invalid question');
    if (q.type === 'choice') {
      if (!object(q.criteria) || Object.keys(q.criteria).length < 1 || Object.keys(q.criteria).length > 255)
        throw new Error('choice requires 1 to 255 options');
    } else if (q.type === 'score') {
      if (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10)
        throw new Error('score requires 2 to 10 levels');
    } else if (q.type === 'noul') {
      if (
        q.criteria != null &&
        (!object(q.criteria) || Object.keys(q.criteria).some((k) => k !== 'true' && k !== 'false'))
      )
        throw new Error('noul criteria may contain only true and false');
    } else throw new Error('question type must be noul, choice or score');
  }
}
