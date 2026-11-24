/**
 * Native-free types, validation, bidirectional schema/JSON mapping, presets,
 * span slicing, and synthetic sample generation for System One (Clairic & CLEF).
 *
 * Zero regex literals — uses character scanning helpers from `text-scan.ts`.
 */

import { isAsciiDigit } from './text-scan';
import type { SystemOneModelInfo } from './types';

export const CLAIRIC_MAX_IMAGES = 4;
export const CLAIRIC_MAX_MEDIA_BYTES = 4 * 1024 * 1024;
export const CLAIRIC_MAX_TOTAL_MEDIA_BYTES = 8 * 1024 * 1024;

export type SystemOneMode = 'extract' | 'decide';

export type ClairicDurationUnit = 'seconds' | 'minutes' | 'hours' | 'days' | 'weeks' | 'months' | 'years';
export type ClairicEstimateScale = 'linear' | 'log';
export type ClairicForecastInterval = 'step' | 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';
export type ClairicListItemKind = 'text' | 'number' | 'date' | 'object';
export type ClairicImageRotation = 0 | 90 | 180 | 270;

export type ClairicFieldType =
  | 'text'
  | 'number'
  | 'boolean'
  | 'enum'
  | 'object'
  | 'list'
  | 'date'
  | 'time'
  | 'datetime'
  | 'duration'
  | 'compute'
  | 'estimate'
  | 'forecast';

export type FieldGroup = 'AR' | 'Head';

export interface FieldTypeInfo {
  type: ClairicFieldType;
  label: string;
  group: FieldGroup;
  description: string;
}

export const CLAIRIC_FIELD_TYPES: readonly FieldTypeInfo[] = [
  // Extraction (AR)
  { type: 'text', label: 'text', group: 'AR', description: 'Free-form string extraction' },
  { type: 'number', label: 'number', group: 'AR', description: 'Extracted numeric value' },
  { type: 'boolean', label: 'boolean', group: 'AR', description: 'True or false verdict' },
  { type: 'enum', label: 'enum', group: 'AR', description: 'One choice from a closed set' },
  { type: 'object', label: 'object', group: 'AR', description: 'Nested structured record' },
  { type: 'list', label: 'list', group: 'AR', description: 'Array of scalars or objects' },
  // Typed Head (Head)
  { type: 'date', label: 'date', group: 'Head', description: 'ISO YYYY-MM-DD calendar date' },
  { type: 'time', label: 'time', group: 'Head', description: 'Clock time HH:MM[:SS]' },
  { type: 'datetime', label: 'datetime', group: 'Head', description: 'ISO timestamp YYYY-MM-DDTHH:MM' },
  { type: 'duration', label: 'duration', group: 'Head', description: 'Time span in a chosen unit' },
  { type: 'compute', label: 'compute', group: 'Head', description: 'Calculated numeric answer' },
  { type: 'estimate', label: 'estimate', group: 'Head', description: 'Bounded distributional estimate' },
  { type: 'forecast', label: 'forecast', group: 'Head', description: 'Multi-step horizon forecast' },
];

export const DURATION_UNITS: readonly ClairicDurationUnit[] = [
  'seconds',
  'minutes',
  'hours',
  'days',
  'weeks',
  'months',
  'years',
];

export const FORECAST_INTERVALS: readonly ClairicForecastInterval[] = [
  'step',
  'minute',
  'hour',
  'day',
  'week',
  'month',
  'quarter',
  'year',
];

export function fieldGroupOf(type: ClairicFieldType): FieldGroup {
  return CLAIRIC_FIELD_TYPES.find((item) => item.type === type)?.group ?? 'AR';
}

export function isSystemOneModelType(modelType: string | null | undefined): boolean {
  if (modelType == null) return false;
  const lower = modelType.toLowerCase();
  return lower === 'clairic' || lower === 'clef';
}

/**
 * The model `?model=` (and `&build=`) of a link names: the model whose id or name is `model`, else, with a `build`, the
 * model of that build in the folder `model` of the models directory. A catalog card links by its folder and its
 * build, and a GGUF model's id is its file name; one folder can hold several builds.
 */
export function requestedModel<T extends Pick<SystemOneModelInfo, 'id' | 'name' | 'path' | 'variant'>>(
  models: readonly T[],
  model: string,
  build: string | null,
): T | undefined {
  const byId = models.find((candidate) => candidate.id === model || candidate.name === model);
  if (byId !== undefined || build === null) return byId;
  return models.find((candidate) => candidate.variant === build && modelFolder(candidate.path) === model);
}

/** The folder of a model in the models directory: its path, or the folder of a GGUF file. */
function modelFolder(path: string): string {
  const parts = path.split('/');
  const last = parts[parts.length - 1] ?? '';
  return (last.toLowerCase().endsWith('.gguf') ? parts[parts.length - 2] : last) ?? '';
}

export function formatModelFamily(modelType: string): string {
  const lower = modelType.toLowerCase();
  if (lower === 'clairic') return 'Clairic';
  if (lower === 'clef') return 'CLEF';
  return modelType;
}

/** Visual builder representation of one Clairic field. */
export interface FieldDraft {
  id: string;
  name: string;
  type: ClairicFieldType;
  instructions: string;
  nullable: boolean;
  sourceOnly: boolean;
  knowledge: boolean;
  integer: boolean;
  decimals: string;
  unit: string;
  durationUnit: ClairicDurationUnit;
  enumValues: string[];
  listItemKind: ClairicListItemKind;
  listMaxItems: string;
  listObjectPropertiesJson: string;
  objectPropertiesJson: string;
  min: string;
  max: string;
  scale: ClairicEstimateScale;
  horizon: string;
  interval: ClairicForecastInterval;
}

/** Visual builder representation of one CLEF question. */
export type ClefQuestionType = 'noul' | 'choice' | 'score';

export interface ClefChoiceOption {
  id: string;
  key: string;
  description: string;
}

export interface ClefQuestionDraft {
  id: string;
  name: string;
  type: ClefQuestionType;
  instructions: string;
  noulTrue: string;
  noulFalse: string;
  choices: ClefChoiceOption[];
  scoreLevels: string[];
}

export interface MediaAttachment {
  id: string;
  name: string;
  contentType: string;
  sizeBytes: number;
  base64: string;
  dataUrl: string;
  rotation: ClairicImageRotation;
  kind: 'image' | 'pdf';
}

export interface ClairicUsage {
  state_tokens: number;
  image_tokens: number;
  schema_tokens: number;
  input_tokens: number;
  output_tokens: number;
  passes: number;
}

export type ClairicSpan = [number, number];

export interface ClairicScalarAnswer {
  value: string | number | boolean | Record<string, unknown> | null;
  guess?: string | number | null;
  valid?: boolean;
  complete?: boolean;
  confidence?: number;
  p_null?: number;
  span?: ClairicSpan | null;
  spans?: (ClairicSpan | null)[] | null;
  source?: string;
  unit?: string;
}

export interface ClairicListAnswer {
  value: (string | number | Record<string, unknown> | null)[];
  valid?: boolean;
  complete?: boolean;
  confidence?: number;
  confidences?: number[];
  p_null?: number;
  p_nulls?: number[];
  span?: ClairicSpan | null;
  spans?: (ClairicSpan | null)[] | null;
  source?: string;
}

export interface ClairicEstimateAnswer {
  value: number | string;
  mean: number | string;
  median: number | string;
  p10: number | string;
  p90: number | string;
  quantiles?: (number | string)[];
  p_below?: number;
  p_above?: number;
}

export interface ClairicForecastAnswer {
  value: (number | string)[];
  median: (number | string)[];
  p10: (number | string)[];
  p90: (number | string)[];
  steps: ClairicEstimateAnswer[];
}

export type ClairicAnswer = ClairicScalarAnswer | ClairicListAnswer | ClairicEstimateAnswer | ClairicForecastAnswer;

export interface ClairicResult {
  model?: string;
  answers: Record<string, ClairicAnswer>;
  truncated?: boolean;
  usage?: ClairicUsage;
  state_text?: string;
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
  usage?: { input_tokens: number; output_tokens: number };
}

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function isForecastAnswer(answer: unknown): answer is ClairicForecastAnswer {
  return isPlainObject(answer) && Array.isArray(answer.steps) && Array.isArray(answer.median);
}

export function isEstimateAnswer(answer: unknown): answer is ClairicEstimateAnswer {
  return (
    isPlainObject(answer) && !Array.isArray(answer.steps) && 'median' in answer && 'p10' in answer && 'p90' in answer
  );
}

export function isListAnswer(answer: unknown): answer is ClairicListAnswer {
  return isPlainObject(answer) && !isForecastAnswer(answer) && Array.isArray(answer.value);
}

export function isClefResult(result: unknown): result is ClefResult {
  if (!isPlainObject(result) || !isPlainObject(result.answers)) return false;
  const values = Object.values(result.answers);
  if (values.length === 0) return false;
  const first = values[0];
  return isPlainObject(first) && (first.type === 'noul' || first.type === 'choice' || first.type === 'score');
}

/** Collect all non-null `[start, end)` spans from an answer (supports both `span` and `spans`). */
export function extractAnswerSpans(answer: unknown): ClairicSpan[] {
  if (!isPlainObject(answer)) return [];
  const out: ClairicSpan[] = [];
  const single = answer.span;
  if (
    Array.isArray(single) &&
    single.length === 2 &&
    typeof single[0] === 'number' &&
    typeof single[1] === 'number' &&
    single[1] > single[0]
  ) {
    out.push([single[0], single[1]]);
  }
  const multi = answer.spans;
  if (Array.isArray(multi)) {
    for (const item of multi) {
      if (
        Array.isArray(item) &&
        item.length === 2 &&
        typeof item[0] === 'number' &&
        typeof item[1] === 'number' &&
        item[1] > item[0]
      ) {
        out.push([item[0], item[1]]);
      }
    }
  }
  return out;
}

let draftSeq = 1;
export function nextDraftId(prefix = 'fld'): string {
  draftSeq += 1;
  return `${prefix}-${String(draftSeq)}`;
}

/** Generate a reload-safe unique ID for persisted items such as saved specs. */
export function createSpecId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `spec-${crypto.randomUUID()}`;
  }
  return `spec-${String(Date.now())}-${Math.random().toString(36).slice(2, 10)}`;
}

export function createEmptyFieldDraft(type: ClairicFieldType = 'text', name = ''): FieldDraft {
  return {
    id: nextDraftId('fld'),
    name,
    type,
    instructions: '',
    nullable: false,
    sourceOnly: false,
    knowledge: false,
    integer: false,
    decimals: '',
    unit: '',
    durationUnit: 'hours',
    enumValues: type === 'enum' ? ['low', 'medium', 'high'] : [],
    listItemKind: 'text',
    listMaxItems: '',
    listObjectPropertiesJson: '{\n  "name": { "type": "text" },\n  "amount": { "type": "number" }\n}',
    objectPropertiesJson: '{\n  "name": { "type": "text" }\n}',
    min: type === 'estimate' || type === 'forecast' ? '0' : '',
    max: type === 'estimate' || type === 'forecast' ? '100' : '',
    scale: 'linear',
    horizon: type === 'forecast' ? '3' : '',
    interval: 'day',
  };
}

export function createEmptyClefQuestion(type: ClefQuestionType = 'noul', name = ''): ClefQuestionDraft {
  return {
    id: nextDraftId('q'),
    name,
    type,
    instructions: '',
    noulTrue: '',
    noulFalse: '',
    choices: [
      { id: nextDraftId('opt'), key: 'approve', description: 'Meets all policy requirements' },
      { id: nextDraftId('opt'), key: 'escalate', description: 'Needs manual human review' },
      { id: nextDraftId('opt'), key: 'reject', description: 'Violates policy or lacks required info' },
    ],
    scoreLevels: ['Very poor', 'Poor', 'Acceptable', 'Good', 'Excellent'],
  };
}

/** Check ISO date format `YYYY-MM-DD` without regex literals. */
export function isIsoDate(text: string): boolean {
  if (text.length !== 10) return false;
  if (text[4] !== '-' || text[7] !== '-') return false;
  for (let i = 0; i < 10; i += 1) {
    if (i === 4 || i === 7) continue;
    if (!isAsciiDigit(text[i]!)) return false;
  }
  const month = Number(text.slice(5, 7));
  const day = Number(text.slice(8, 10));
  return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

function parseOptionalNumber(text: string): number | undefined {
  const trimmed = text.trim();
  if (trimmed === '') return undefined;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : undefined;
}

function parseBoundValue(text: string): number | string | undefined {
  const trimmed = text.trim();
  if (trimmed === '') return undefined;
  if (isIsoDate(trimmed)) return trimmed;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : undefined;
}

export function supportsSourceOnly(type: ClairicFieldType): boolean {
  return type === 'number' || type === 'date' || type === 'time' || type === 'datetime' || type === 'list';
}

export function supportsKnowledge(type: ClairicFieldType): boolean {
  return type === 'text' || type === 'number' || type === 'list';
}

export function supportsNumericOptions(type: ClairicFieldType): boolean {
  return type === 'number' || type === 'compute' || type === 'duration';
}

/** Validate a list of Clairic field drafts; returns per-field warnings keyed by draft id. */
export function validateFieldDrafts(fields: readonly FieldDraft[]): Record<string, string[]> {
  const warnings: Record<string, string[]> = {};
  const seenNames = new Map<string, string[]>();

  for (const field of fields) {
    const trimmed = field.name.trim();
    const list: string[] = [];
    if (trimmed === '') {
      list.push('Field name is required.');
    } else if (trimmed === '__proto__' || trimmed === 'constructor' || trimmed === 'prototype') {
      list.push(`Field name "${trimmed}" is reserved.`);
    } else {
      const existing = seenNames.get(trimmed) ?? [];
      existing.push(field.id);
      seenNames.set(trimmed, existing);
    }

    if (field.type === 'enum' && field.enumValues.length === 0) {
      list.push('Add at least one enum value.');
    }

    if (field.type === 'estimate' || field.type === 'forecast') {
      const minVal = parseBoundValue(field.min);
      const maxVal = parseBoundValue(field.max);
      if (minVal === undefined) list.push('Set a numeric or YYYY-MM-DD min bound.');
      if (maxVal === undefined) list.push('Set a numeric or YYYY-MM-DD max bound.');
      if (minVal !== undefined && maxVal !== undefined) {
        if (typeof minVal !== typeof maxVal) {
          list.push('min and max must both be numbers or both be YYYY-MM-DD dates.');
        } else if (typeof minVal === 'number' && typeof maxVal === 'number' && minVal >= maxVal) {
          list.push('max must be strictly greater than min.');
        } else if (typeof minVal === 'string' && typeof maxVal === 'string' && minVal >= maxVal) {
          list.push('max date must be after min date.');
        }
        if (typeof minVal === 'number' && field.scale === 'log' && minVal <= 0) {
          list.push('Log scale requires min > 0.');
        }
      }
      if (field.type === 'forecast') {
        const horizon = parseOptionalNumber(field.horizon);
        if (horizon === undefined || !Number.isInteger(horizon) || horizon < 1) {
          list.push('Forecast horizon must be an integer >= 1.');
        }
      }
    }

    if (field.type === 'object') {
      try {
        const parsed: unknown = JSON.parse(field.objectPropertiesJson);
        if (!isPlainObject(parsed) || Object.keys(parsed).length === 0) {
          list.push('Object properties must be a non-empty JSON object.');
        }
      } catch {
        list.push('Object properties JSON is invalid.');
      }
    }

    if (field.type === 'list') {
      if (field.listMaxItems.trim() !== '') {
        const maxItems = parseOptionalNumber(field.listMaxItems);
        if (maxItems === undefined || !Number.isInteger(maxItems) || maxItems < 1) {
          list.push('max_items must be an integer >= 1.');
        }
      }
      if (field.listItemKind === 'object') {
        try {
          const parsed: unknown = JSON.parse(field.listObjectPropertiesJson);
          if (!isPlainObject(parsed) || Object.keys(parsed).length === 0) {
            list.push('List item properties must be a non-empty JSON object.');
          }
        } catch {
          list.push('List item properties JSON is invalid.');
        }
      }
    }

    if (field.decimals.trim() !== '') {
      const decimals = parseOptionalNumber(field.decimals);
      if (decimals === undefined || !Number.isInteger(decimals) || decimals < 0) {
        list.push('decimals must be a non-negative integer.');
      }
    }

    if (list.length > 0) warnings[field.id] = list;
  }

  for (const [name, ids] of seenNames) {
    if (ids.length > 1) {
      for (const id of ids) {
        warnings[id] = [...(warnings[id] ?? []), `Duplicate field name "${name}".`];
      }
    }
  }

  return warnings;
}

export function validateClefQuestions(questions: readonly ClefQuestionDraft[]): Record<string, string[]> {
  const warnings: Record<string, string[]> = {};
  const seen = new Map<string, string[]>();
  for (const q of questions) {
    const trimmed = q.name.trim();
    const list: string[] = [];
    if (trimmed === '') {
      list.push('Question ID is required.');
    } else if (trimmed === '__proto__' || trimmed === 'constructor' || trimmed === 'prototype') {
      list.push(`Question ID "${trimmed}" is reserved.`);
    } else {
      const ids = seen.get(trimmed) ?? [];
      ids.push(q.id);
      seen.set(trimmed, ids);
    }
    if (q.type === 'choice') {
      const validChoices = q.choices.filter((c) => c.key.trim() !== '');
      if (validChoices.length < 1) list.push('Choice requires at least 1 option.');
      const seenKeys = new Set<string>();
      for (const c of validChoices) {
        const key = c.key.trim();
        if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
          list.push(`Choice option "${key}" is reserved.`);
        } else if (seenKeys.has(key)) {
          list.push(`Duplicate choice option "${key}".`);
        } else {
          seenKeys.add(key);
        }
      }
    } else if (q.type === 'score') {
      if (q.scoreLevels.length < 2 || q.scoreLevels.length > 10) {
        list.push('Score requires between 2 and 10 levels.');
      }
    }
    if (list.length > 0) warnings[q.id] = list;
  }
  for (const [name, ids] of seen) {
    if (ids.length > 1) {
      for (const id of ids) {
        warnings[id] = [...(warnings[id] ?? []), `Duplicate question ID "${name}".`];
      }
    }
  }
  return warnings;
}

/** Convert visual Clairic draft fields + source into a Clairic JSON payload. */
export function buildClairicPayload(params: {
  stateText: string;
  referenceDate: string;
  includeQuantiles: boolean;
  returnStateText: boolean;
  fields: readonly FieldDraft[];
  attachments: readonly MediaAttachment[];
}): Record<string, unknown> {
  const fieldsObj: Record<string, Record<string, unknown>> = {};
  for (const field of params.fields) {
    const name = field.name.trim();
    if (name === '') continue;
    const spec: Record<string, unknown> = { type: field.type };
    if (field.instructions.trim() !== '') spec.instructions = field.instructions.trim();
    if (field.nullable) spec.nullable = true;
    if (supportsSourceOnly(field.type) && field.sourceOnly) spec.source_only = true;
    if (supportsKnowledge(field.type) && field.knowledge) spec.knowledge = true;

    if (supportsNumericOptions(field.type)) {
      if (field.integer) spec.integer = true;
      const decimals = parseOptionalNumber(field.decimals);
      if (decimals !== undefined) spec.decimals = decimals;
      if (field.type === 'duration') {
        spec.unit = field.durationUnit;
      } else if (field.unit.trim() !== '') {
        spec.unit = field.unit.trim();
      }
    }

    if (field.type === 'enum') {
      spec.enum = field.enumValues;
    } else if (field.type === 'object') {
      try {
        const parsed: unknown = JSON.parse(field.objectPropertiesJson);
        spec.properties = isPlainObject(parsed) ? parsed : {};
      } catch {
        spec.properties = {};
      }
    } else if (field.type === 'list') {
      if (field.listItemKind === 'object') {
        let props: Record<string, unknown> = {};
        try {
          const parsed: unknown = JSON.parse(field.listObjectPropertiesJson);
          if (isPlainObject(parsed)) props = parsed;
        } catch {
          // Keep empty object until fixed.
        }
        spec.items = { type: 'object', properties: props };
      } else {
        spec.items = field.listItemKind;
      }
      const maxItems = parseOptionalNumber(field.listMaxItems);
      if (maxItems !== undefined) spec.max_items = maxItems;
    } else if (field.type === 'estimate' || field.type === 'forecast') {
      const minVal = parseBoundValue(field.min);
      const maxVal = parseBoundValue(field.max);
      if (minVal !== undefined) spec.min = minVal;
      if (maxVal !== undefined) spec.max = maxVal;
      if (field.scale === 'log') spec.scale = 'log';
      if (field.unit.trim() !== '') spec.unit = field.unit.trim();
      if (field.type === 'forecast') {
        const horizon = parseOptionalNumber(field.horizon);
        spec.horizon = horizon ?? 3;
        spec.interval = field.interval;
      }
    }

    fieldsObj[name] = spec;
  }

  const payload: Record<string, unknown> = {};
  if (params.stateText.length > 0 || params.attachments.length === 0) {
    payload.state = params.stateText;
  }

  const images = params.attachments
    .filter((a) => a.kind === 'image')
    .map((a) => ({
      content_type: a.contentType,
      base64: a.base64,
      ...(a.rotation !== 0 ? { rotation: a.rotation } : {}),
    }));
  const documents = params.attachments
    .filter((a) => a.kind === 'pdf')
    .map((a) => ({
      content_type: 'application/pdf',
      base64: a.base64,
    }));

  if (images.length > 0) payload.images = images;
  if (documents.length > 0) payload.documents = documents;

  payload.fields = fieldsObj;

  if (params.referenceDate.trim() !== '') {
    payload.reference_date = params.referenceDate.trim();
  }

  const options: Record<string, boolean> = {};
  if (params.includeQuantiles) options.include_quantiles = true;
  if (params.returnStateText) options.return_state_text = true;
  if (Object.keys(options).length > 0) {
    payload.options = options;
  }

  return payload;
}

/** Convert visual CLEF questions + state into a CLEF JSON payload. */
export function buildClefPayload(params: {
  stateText: string;
  questions: readonly ClefQuestionDraft[];
}): Record<string, unknown> {
  let state: unknown = params.stateText;
  const trimmed = params.stateText.trim();
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      state = JSON.parse(trimmed);
    } catch {
      state = params.stateText;
    }
  }
  const questionsObj: Record<string, Record<string, unknown>> = {};
  for (const q of params.questions) {
    const name = q.name.trim();
    if (name === '') continue;
    const spec: Record<string, unknown> = { type: q.type };
    if (q.instructions.trim() !== '') spec.instructions = q.instructions.trim();
    if (q.type === 'noul') {
      const criteria: Record<string, string> = {};
      if (q.noulTrue.trim() !== '') criteria.true = q.noulTrue.trim();
      if (q.noulFalse.trim() !== '') criteria.false = q.noulFalse.trim();
      if (Object.keys(criteria).length > 0) spec.criteria = criteria;
    } else if (q.type === 'choice') {
      const criteria: Record<string, string> = {};
      for (const c of q.choices) {
        const key = c.key.trim();
        if (key !== '') criteria[key] = c.description.trim() || key;
      }
      spec.criteria = criteria;
    } else if (q.type === 'score') {
      spec.criteria = q.scoreLevels;
    }
    questionsObj[name] = spec;
  }
  return { state, questions: questionsObj };
}

function scalarToText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value === null || value === undefined) return '';
  return JSON.stringify(value);
}

/** Parse a raw Clairic payload object back into visual state + fields. */
export function parseClairicPayload(raw: Record<string, unknown>): {
  stateText: string;
  referenceDate: string;
  includeQuantiles: boolean;
  returnStateText: boolean;
  fields: FieldDraft[];
} {
  const stateText =
    typeof raw.state === 'string' ? raw.state : raw.state !== undefined ? JSON.stringify(raw.state, null, 2) : '';
  const referenceDate = typeof raw.reference_date === 'string' ? raw.reference_date : '';
  const opts = isPlainObject(raw.options) ? raw.options : {};
  const includeQuantiles = opts.include_quantiles === true;
  const returnStateText = opts.return_state_text === true;

  const fields: FieldDraft[] = [];
  if (isPlainObject(raw.fields)) {
    for (const [name, spec] of Object.entries(raw.fields)) {
      if (!isPlainObject(spec)) continue;
      const rawType = typeof spec.type === 'string' ? spec.type : 'text';
      const type = (CLAIRIC_FIELD_TYPES.some((item) => item.type === rawType) ? rawType : 'text') as ClairicFieldType;
      const draft = createEmptyFieldDraft(type, name);
      draft.instructions = typeof spec.instructions === 'string' ? spec.instructions : '';
      draft.nullable = spec.nullable === true;
      draft.sourceOnly = spec.source_only === true;
      draft.knowledge = spec.knowledge === true;
      draft.integer = spec.integer === true;
      draft.decimals = typeof spec.decimals === 'number' ? String(spec.decimals) : '';
      if (type === 'duration') {
        const u = typeof spec.unit === 'string' ? spec.unit : 'hours';
        draft.durationUnit = DURATION_UNITS.includes(u as ClairicDurationUnit) ? (u as ClairicDurationUnit) : 'hours';
      } else {
        draft.unit = typeof spec.unit === 'string' ? spec.unit : '';
      }
      if (Array.isArray(spec.enum)) {
        draft.enumValues = spec.enum.map((v) => String(v));
      }
      if (type === 'object' && isPlainObject(spec.properties)) {
        draft.objectPropertiesJson = JSON.stringify(spec.properties, null, 2);
      }
      if (type === 'list') {
        if (
          typeof spec.items === 'string' &&
          (spec.items === 'text' || spec.items === 'number' || spec.items === 'date')
        ) {
          draft.listItemKind = spec.items;
        } else if (isPlainObject(spec.items) && spec.items.type === 'object') {
          draft.listItemKind = 'object';
          draft.listObjectPropertiesJson = JSON.stringify(
            isPlainObject(spec.items.properties) ? spec.items.properties : {},
            null,
            2,
          );
        }
        draft.listMaxItems = typeof spec.max_items === 'number' ? String(spec.max_items) : '';
      }
      if (type === 'estimate' || type === 'forecast') {
        draft.min = scalarToText(spec.min);
        draft.max = scalarToText(spec.max);
        draft.scale = spec.scale === 'log' ? 'log' : 'linear';
        if (type === 'forecast') {
          draft.horizon = typeof spec.horizon === 'number' ? String(spec.horizon) : '3';
          const iv = typeof spec.interval === 'string' ? spec.interval : 'day';
          draft.interval = FORECAST_INTERVALS.includes(iv as ClairicForecastInterval)
            ? (iv as ClairicForecastInterval)
            : 'day';
        }
      }
      fields.push(draft);
    }
  }
  return { stateText, referenceDate, includeQuantiles, returnStateText, fields };
}

/** Parse a raw CLEF payload back into visual state + questions. */
export function parseClefPayload(raw: Record<string, unknown>): {
  stateText: string;
  questions: ClefQuestionDraft[];
} {
  const stateText =
    typeof raw.state === 'string' ? raw.state : raw.state !== undefined ? JSON.stringify(raw.state, null, 2) : '';
  const questions: ClefQuestionDraft[] = [];
  if (isPlainObject(raw.questions)) {
    for (const [name, q] of Object.entries(raw.questions)) {
      if (!isPlainObject(q)) continue;
      const type: ClefQuestionType = q.type === 'choice' || q.type === 'score' || q.type === 'noul' ? q.type : 'noul';
      const draft = createEmptyClefQuestion(type, name);
      draft.instructions =
        typeof q.instructions === 'string'
          ? q.instructions
          : q.instructions !== undefined
            ? JSON.stringify(q.instructions)
            : '';
      if (type === 'noul' && isPlainObject(q.criteria)) {
        draft.noulTrue = scalarToText(q.criteria.true);
        draft.noulFalse = scalarToText(q.criteria.false);
      } else if (type === 'choice' && isPlainObject(q.criteria)) {
        draft.choices = Object.entries(q.criteria).map(([key, value]) => ({
          id: nextDraftId('opt'),
          key,
          description: typeof value === 'string' ? value : JSON.stringify(value),
        }));
      } else if (type === 'score' && Array.isArray(q.criteria)) {
        draft.scoreLevels = q.criteria.map((item) => (typeof item === 'string' ? item : JSON.stringify(item)));
      }
      questions.push(draft);
    }
  }
  return { stateText, questions };
}

export interface SystemOnePreset {
  id: string;
  label: string;
  description: string;
  mode: SystemOneMode;
  needsSampleReceipt?: boolean;
  payload: Record<string, unknown>;
}

export const SYSTEM_ONE_PRESETS: readonly SystemOnePreset[] = [
  {
    id: 'invoice',
    label: 'Read an invoice',
    description: 'Pull vendor, total, date, line items, and optional PO number',
    mode: 'extract',
    payload: {
      state:
        'INVOICE #INV-2026-0842\nFrom: Northstar Robotics Supply Co.\nIssued: March 14, 2026\nBill To: Pacific Hardware Labs\n\nLine Items:\n- Stepper Motor NEMA-17 (x4): $148.00\n- Aluminum Extrusion 2020 1m (x6): $72.00\n- Silicone Thermal Pad 2mm (x2): $18.50\n\nSubtotal: $238.50\nTax (8.25%): $19.68\nTotal Due: $258.18\nPayment Terms: Net 30. Note: Customer has not provided a PO number.',
      reference_date: '2026-03-14',
      options: { return_state_text: true },
      fields: {
        vendor: { type: 'text', instructions: 'Issuing vendor company name' },
        total: { type: 'number', decimals: 2, unit: 'USD', instructions: 'Final total due including tax' },
        issued: { type: 'date', instructions: 'Invoice issue date' },
        items: { type: 'list', items: 'text', max_items: 8, instructions: 'Line item product names' },
        missing_po: { type: 'text', nullable: true, instructions: 'Customer purchase order number if present' },
      },
    },
  },
  {
    id: 'demand-forecast',
    label: 'Forecast demand',
    description: 'Estimate tomorrow and forecast the next 3 days from weekly orders',
    mode: 'extract',
    payload: {
      state:
        'Daily bakery croissant orders over the last 7 days:\nMon: 118\nTue: 124\nWed: 131\nThu: 129\nFri: 158\nSat: 212\nSun: 196\nWeather tomorrow (Mon): light rain, typical weekday commuter foot traffic.',
      reference_date: '2026-04-06',
      options: { include_quantiles: true, return_state_text: true },
      fields: {
        next_day: {
          type: 'estimate',
          min: 80,
          max: 260,
          unit: 'orders',
          instructions: 'Expected croissant orders on Monday',
        },
        next_three_days: {
          type: 'forecast',
          min: 80,
          max: 260,
          horizon: 3,
          interval: 'day',
          unit: 'orders',
          instructions: 'Forecasted daily croissant orders for Mon, Tue, Wed',
        },
      },
    },
  },
  {
    id: 'ticket-triage',
    label: 'Triage a support ticket',
    description: 'Classify severity, urgency, customer name, and response SLA',
    mode: 'extract',
    payload: {
      state:
        'From: Maya Lin <maya@cobaltgrid.io>\nSubject: Production webhooks returning 502 since 09:15 UTC\nBody: Our checkout pipeline depends on your order.paid webhook. Every delivery has failed with HTTP 502 for the past 45 minutes. Enterprise contract specifies a 2 hour response SLA for P1 outages.',
      options: { return_state_text: true },
      fields: {
        priority: {
          type: 'enum',
          enum: ['low', 'medium', 'high', 'critical'],
          instructions: 'Ticket severity level',
        },
        urgent: { type: 'boolean', instructions: 'True if production is currently impacted' },
        customer: { type: 'text', instructions: 'Customer full name' },
        sla: { type: 'duration', unit: 'hours', instructions: 'Contractual response SLA' },
      },
    },
  },
  {
    id: 'csv-table',
    label: 'Parse a CSV table',
    description: 'Turn raw CSV rows into structured objects and sum total units',
    mode: 'extract',
    payload: {
      state:
        'sku,product,qty,unit_price\nKB-75,Mechanical Keyboard 75%,3,149.00\nMS-PRO,Wireless Ergonomic Mouse,5,79.50\nHUB-C,USB-C 8-in-1 Dock,2,64.00',
      options: { return_state_text: true },
      fields: {
        rows: {
          type: 'list',
          max_items: 10,
          instructions: 'Each product row from the CSV',
          items: {
            type: 'object',
            properties: {
              sku: { type: 'text' },
              product: { type: 'text' },
              qty: { type: 'number', integer: true },
              unit_price: { type: 'number', decimals: 2 },
            },
          },
        },
        total_units: { type: 'compute', integer: true, instructions: 'Sum of qty across all rows' },
      },
    },
  },
  {
    id: 'receipt-vision',
    label: 'Scan a receipt image',
    description: 'Read merchant, date, total, and items from a store receipt photo',
    mode: 'extract',
    needsSampleReceipt: true,
    payload: {
      state: 'Extract key fields from the attached store receipt image.',
      options: { return_state_text: true },
      fields: {
        merchant: { type: 'text', instructions: 'Store or cafe name on the receipt' },
        date: { type: 'date', instructions: 'Purchase date' },
        total: { type: 'number', decimals: 2, unit: 'USD', instructions: 'Final total charged' },
        items: { type: 'list', items: 'text', max_items: 8, instructions: 'Purchased items' },
      },
    },
  },
  {
    id: 'clef-decision',
    label: 'Score a hiring decision',
    description: 'Rate a candidate with Yes/No, track choice, and 0–4 fit probabilities',
    mode: 'decide',
    payload: {
      state: {
        applicant: 'Avery Chen',
        role: 'Senior Systems Engineer',
        years_experience: 8,
        summary: 'Built low-latency Rust storage engines and Metal compute kernels. Strong open-source track record.',
        requested_level: 'L6',
      },
      questions: {
        advance_to_onsite: {
          type: 'noul',
          instructions: 'Should this candidate advance to the onsite loop?',
          criteria: {
            true: 'Strong match on systems experience and technical depth',
            false: 'Insufficient systems background for senior role',
          },
        },
        recommended_track: {
          type: 'choice',
          instructions: 'Pick the best interview track for this candidate',
          criteria: {
            gpu_kernels: 'Metal, CUDA, or ML inference kernel focus',
            distributed_storage: 'Storage engines, replication, and consensus',
            product_backend: 'APIs, web services, and data pipelines',
          },
        },
        fit_score: {
          type: 'score',
          instructions: 'Overall role fit from 0 to 4',
          criteria: ['Not a fit', 'Weak fit', 'Moderate fit', 'Strong fit', 'Exceptional fit'],
        },
      },
    },
  },
  {
    id: 'clef-moderation',
    label: 'Moderate user content',
    description: 'Gate a community post with Yes/No allow, policy category, and 0–3 severity',
    mode: 'decide',
    payload: {
      state: {
        author: 'user_9481',
        channel: 'product-feedback',
        post: 'Honestly this update broke our CI twice today. If you cannot ship tested builds, stop pushing on Fridays. Anyone know a working fork?',
        account_age_days: 412,
        prior_warnings: 0,
      },
      questions: {
        allow_post: {
          type: 'noul',
          instructions: 'Should this post be allowed to stay published?',
          criteria: {
            true: 'Frustrated or critical feedback without harassment, hate speech, or spam',
            false: 'Violates community policy (abuse, threats, scams, or spam)',
          },
        },
        violation_type: {
          type: 'choice',
          instructions: 'Select the closest moderation category',
          criteria: {
            none: 'No policy violation — acceptable product criticism',
            harassment: 'Personal attacks, insults, or threats directed at individuals',
            spam: 'Unsolicited promotion, phishing, or repetitive junk',
            off_topic: 'Unrelated content posted in the wrong channel',
          },
        },
        severity: {
          type: 'score',
          instructions: 'Moderation risk severity from 0 to 3',
          criteria: [
            'Safe — no action needed',
            'Borderline — harsh tone but within rules',
            'Moderate — needs moderator review or warning',
            'Severe — remove immediately',
          ],
        },
      },
    },
  },
  {
    id: 'clef-escalation',
    label: 'Triage a support escalation',
    description: 'Route a customer message by urgency, owning team, and 0–4 churn risk',
    mode: 'decide',
    payload: {
      state: {
        customer: 'Helios BioLabs (Enterprise · $180k ARR)',
        subject: 'Export jobs stuck for 4 hours ahead of FDA submission',
        message:
          'Three production export pipelines have been queued since 06:00 EST with no progress. Our regulatory filing deadline is at 17:00 today and our VP of Engineering is asking whether we need to invoke our SLA credit.',
        tier: 'enterprise',
      },
      questions: {
        page_oncall: {
          type: 'noul',
          instructions: 'Should this ticket page the on-call engineer immediately?',
          criteria: {
            true: 'Production outage, tight business deadline, or enterprise SLA breach',
            false: 'Non-blocking question or feature request that can wait for business hours',
          },
        },
        owner_team: {
          type: 'choice',
          instructions: 'Pick the primary engineering team to own this ticket',
          criteria: {
            data_pipelines: 'Batch exports, queues, ETL workers, and job scheduling',
            billing_account: 'Invoices, SLA credits, contracts, and account changes',
            auth_security: 'SSO, permissions, tokens, and compliance access',
          },
        },
        churn_risk: {
          type: 'score',
          instructions: 'Estimated customer escalation & churn risk from 0 to 4',
          criteria: [
            'None — routine inquiry',
            'Low — minor inconvenience',
            'Moderate — frustrated user, no immediate executive pressure',
            'High — deadline at risk, leadership involved',
            'Critical — active SLA breach with immediate renewal threat',
          ],
        },
      },
    },
  },
];

/** Generate a crisp synthetic receipt PNG on an offscreen `<canvas>` for 1-click Vision testing. */
export function generateSampleReceiptPng(): MediaAttachment | null {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = 420;
  canvas.height = 560;
  const ctx = canvas.getContext('2d');
  if (ctx === null) return null;

  ctx.fillStyle = '#faf8f3';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#d6d1c4';
  ctx.lineWidth = 2;
  ctx.strokeRect(14, 14, canvas.width - 28, canvas.height - 28);

  ctx.fillStyle = '#1f1e1c';
  ctx.font = 'bold 24px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('SUNSET COFFEE ROASTERS', canvas.width / 2, 64);

  ctx.font = '14px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#524f49';
  ctx.fillText('742 Valencia St, San Francisco, CA', canvas.width / 2, 90);
  ctx.fillText('Date: 2026-04-12  08:42 AM', canvas.width / 2, 114);
  ctx.fillText('Receipt #SCR-4918', canvas.width / 2, 136);

  const drawRule = (y: number): void => {
    ctx.strokeStyle = '#b9b3a4';
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(32, y);
    ctx.lineTo(canvas.width - 32, y);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  drawRule(158);

  const rows: ReadonlyArray<[string, string]> = [
    ['1x Oat Cortado', '$5.50'],
    ['1x Almond Croissant', '$4.75'],
    ['1x Kenya Single Origin 250g', '$19.00'],
    ['1x Cold Brew Bottle', '$5.25'],
  ];

  ctx.font = '16px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#1f1e1c';
  let y = 196;
  for (const [label, price] of rows) {
    ctx.textAlign = 'left';
    ctx.fillText(label, 36, y);
    ctx.textAlign = 'right';
    ctx.fillText(price, canvas.width - 36, y);
    y += 34;
  }

  drawRule(y + 6);
  y += 40;

  ctx.font = '15px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#524f49';
  ctx.textAlign = 'left';
  ctx.fillText('Subtotal', 36, y);
  ctx.textAlign = 'right';
  ctx.fillText('$34.50', canvas.width - 36, y);
  y += 28;
  ctx.textAlign = 'left';
  ctx.fillText('CA Sales Tax (8.625%)', 36, y);
  ctx.textAlign = 'right';
  ctx.fillText('$2.98', canvas.width - 36, y);
  y += 38;

  ctx.font = 'bold 20px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#111111';
  ctx.textAlign = 'left';
  ctx.fillText('TOTAL', 36, y);
  ctx.textAlign = 'right';
  ctx.fillText('$37.48', canvas.width - 36, y);

  drawRule(y + 24);
  ctx.font = '13px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#6b675e';
  ctx.textAlign = 'center';
  ctx.fillText('VISA **** 4242  AUTH 091284', canvas.width / 2, y + 58);
  ctx.fillText('Thank you for visiting Sunset Coffee!', canvas.width / 2, y + 84);

  const dataUrl = canvas.toDataURL('image/png');
  const comma = dataUrl.indexOf(',');
  const base64 = comma === -1 ? '' : dataUrl.slice(comma + 1);
  const sizeBytes = Math.floor((base64.length * 3) / 4);
  return {
    id: nextDraftId('img'),
    name: 'sample-receipt.png',
    contentType: 'image/png',
    sizeBytes,
    base64,
    dataUrl,
    rotation: 0,
    kind: 'image',
  };
}

/** Default local HTTP endpoint exposed by `mlx serve`. */
export const DEFAULT_SYSTEMONE_ENDPOINT = 'http://127.0.0.1:8080/v1/systemone';

/** Build a clean copyable cURL command for the current request. */
export function buildCurlCommand(
  model: string,
  payload: Record<string, unknown>,
  endpointUrl: string = DEFAULT_SYSTEMONE_ENDPOINT,
): string {
  const sanitized: Record<string, unknown> = { model, ...payload };
  if (Array.isArray(sanitized.images)) {
    sanitized.images = sanitized.images.map((item) => {
      if (isPlainObject(item) && typeof item.base64 === 'string' && item.base64.length > 64) {
        return { ...item, base64: `${item.base64.slice(0, 32)}...<base64>` };
      }
      return item;
    });
  }
  if (Array.isArray(sanitized.documents)) {
    sanitized.documents = sanitized.documents.map((item) => {
      if (isPlainObject(item) && typeof item.base64 === 'string' && item.base64.length > 64) {
        return { ...item, base64: `${item.base64.slice(0, 32)}...<base64>` };
      }
      return item;
    });
  }
  const body = JSON.stringify(sanitized, null, 2);
  const escaped = body.split("'").join("'\\''");
  return `curl -X POST ${endpointUrl} \\\n  -H 'Content-Type: application/json' \\\n  -d '${escaped}'`;
}

/** Compute summary stats from a Clairic or CLEF result for the KPI strip. */
export function summarizeResult(result: Record<string, unknown> | undefined): {
  extractedCount: number;
  nullCount: number;
  truncated: boolean;
  inputTokens: number | null;
  outputTokens: number | null;
  imageTokens: number | null;
  passes: number | null;
  meanConfidence: number | null;
} {
  if (result === undefined || !isPlainObject(result.answers)) {
    return {
      extractedCount: 0,
      nullCount: 0,
      truncated: false,
      inputTokens: null,
      outputTokens: null,
      imageTokens: null,
      passes: null,
      meanConfidence: null,
    };
  }
  let extractedCount = 0;
  let nullCount = 0;
  let confSum = 0;
  let confCount = 0;

  for (const ans of Object.values(result.answers)) {
    if (!isPlainObject(ans)) continue;
    if (ans.type === 'noul' && typeof ans.noul === 'number') {
      extractedCount += 1;
      confSum += Math.max(ans.noul, 1 - ans.noul);
      confCount += 1;
      continue;
    }
    if ((ans.type === 'choice' || ans.type === 'score') && typeof ans.confidence === 'number') {
      extractedCount += 1;
      confSum += ans.confidence;
      confCount += 1;
      continue;
    }
    if (ans.value === null || ans.value === undefined) {
      nullCount += 1;
    } else {
      extractedCount += 1;
    }
    if (typeof ans.confidence === 'number' && Number.isFinite(ans.confidence)) {
      confSum += ans.confidence;
      confCount += 1;
    }
  }

  const usage = isPlainObject(result.usage) ? result.usage : null;
  return {
    extractedCount,
    nullCount,
    truncated: result.truncated === true,
    inputTokens: typeof usage?.input_tokens === 'number' ? usage.input_tokens : null,
    outputTokens: typeof usage?.output_tokens === 'number' ? usage.output_tokens : null,
    imageTokens: typeof usage?.image_tokens === 'number' ? usage.image_tokens : null,
    passes: typeof usage?.passes === 'number' ? usage.passes : null,
    meanConfidence: confCount > 0 ? confSum / confCount : null,
  };
}

/** Palette colors for field highlights in Evidence & charts. */
export const FIELD_ACCENT_COLORS = [
  'var(--model-1)',
  'var(--model-3)',
  'var(--model-7)',
  'var(--model-2)',
  'var(--model-5)',
  'var(--model-4)',
  'var(--model-6)',
  'var(--model-8)',
] as const;

export function fieldAccentColor(index: number): string {
  return FIELD_ACCENT_COLORS[index % FIELD_ACCENT_COLORS.length]!;
}

const AUTO_DETECT_CATEGORIES = [
  'invoice',
  'receipt',
  'support_ticket',
  'email',
  'resume',
  'contract',
  'table',
  'chat_or_messages',
  'time_series_forecast',
  'id_or_form',
  'article_or_notes',
  'photo_or_image',
] as const;

type AutoDetectCategory = (typeof AUTO_DETECT_CATEGORIES)[number];

const CATEGORY_LABELS: Record<AutoDetectCategory, string> = {
  invoice: 'invoice',
  receipt: 'receipt',
  support_ticket: 'support ticket',
  email: 'email',
  resume: 'resume',
  contract: 'contract',
  table: 'table',
  chat_or_messages: 'chat',
  time_series_forecast: 'time series',
  id_or_form: 'form / ID',
  article_or_notes: 'article / notes',
  photo_or_image: 'image',
};

/** Build a fast 1-pass Clairic classification request to auto-detect useful extraction fields. */
export function buildAutoDetectPayload(params: {
  stateText: string;
  attachments: readonly MediaAttachment[];
}): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    state: params.stateText.trim() !== '' ? params.stateText : 'Classify the attached document or image.',
    fields: {
      document_type: {
        type: 'enum',
        enum: [...AUTO_DETECT_CATEGORIES],
        instructions: 'Closest category describing this document or image',
      },
      has_dates: {
        type: 'boolean',
        instructions: 'True if one or more dates appear',
      },
      has_money: {
        type: 'boolean',
        instructions: 'True if prices, totals, or currency amounts appear',
      },
      has_list_items: {
        type: 'boolean',
        instructions: 'True if there is a list of items, rows, or entries',
      },
    },
  };

  const images = params.attachments
    .filter((a) => a.kind === 'image')
    .map((a) => ({
      content_type: a.contentType,
      base64: a.base64,
      ...(a.rotation !== 0 ? { rotation: a.rotation } : {}),
    }));
  const documents = params.attachments
    .filter((a) => a.kind === 'pdf')
    .map((a) => ({
      content_type: 'application/pdf',
      base64: a.base64,
    }));

  if (images.length > 0) payload.images = images;
  if (documents.length > 0) payload.documents = documents;
  return payload;
}

/**
 * Scan numeric values out of plain text without regex literals.
 * Skips ISO dates (`YYYY-MM-DD`) and clock times (`HH:MM`) so forecast bounds are not skewed by years.
 */
export function scanNumbersInText(text: string): number[] {
  const nums: number[] = [];
  let i = 0;
  const len = text.length;
  while (i < len) {
    // Skip ISO date YYYY-MM-DD
    if (i + 10 <= len && isIsoDate(text.slice(i, i + 10))) {
      i += 10;
      continue;
    }
    const ch = text[i]!;
    const isNeg = ch === '-' && i + 1 < len && isAsciiDigit(text[i + 1]!);
    if (!isAsciiDigit(ch) && !isNeg) {
      i += 1;
      continue;
    }
    // Skip if preceded by identifier/date separator like `INV-2026-0842` or `:34`
    if (i > 0) {
      const prev = text[i - 1]!;
      const prevIsAlpha = (prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z');
      const prevIsDigit = isAsciiDigit(prev);
      if (prev === ':' || prev === '/' || prevIsAlpha || (isNeg && prevIsDigit)) {
        if (isNeg) i += 1;
        while (i < len && (isAsciiDigit(text[i]!) || text[i] === '.' || text[i] === ',')) i += 1;
        continue;
      }
    }
    const start = i;
    if (isNeg) i += 1;
    let sawDot = false;
    let sawComma = false;
    let digitsBeforeFirstComma = 0;
    while (i < len) {
      const c = text[i]!;
      if (isAsciiDigit(c)) {
        if (!sawComma && !sawDot) digitsBeforeFirstComma += 1;
        i += 1;
      } else if (
        c === ',' &&
        !sawDot &&
        (sawComma || (digitsBeforeFirstComma >= 1 && digitsBeforeFirstComma <= 3)) &&
        i + 3 < len &&
        isAsciiDigit(text[i + 1]!) &&
        isAsciiDigit(text[i + 2]!) &&
        isAsciiDigit(text[i + 3]!) &&
        (i + 4 >= len || !isAsciiDigit(text[i + 4]!))
      ) {
        // Distinguish thousands grouping (`1,440.50`, `1,234,567`) from tight CSV series (`100,110,120`):
        // if preceded by `,` (already inside a comma list) OR started with 3 digits + another `,DDD` without `$`, split.
        const precededByComma = start > 0 && text[start - 1] === ',';
        const startsLikeCsvSeries =
          !sawComma &&
          digitsBeforeFirstComma === 3 &&
          (start === 0 || text[start - 1] !== '$') &&
          i + 4 < len &&
          text[i + 4] === ',';
        if (precededByComma || startsLikeCsvSeries) break;
        sawComma = true;
        i += 4;
      } else if (c === '.' && !sawDot && i + 1 < len && isAsciiDigit(text[i + 1]!)) {
        sawDot = true;
        i += 1;
      } else {
        break;
      }
    }
    // Skip clock time prefix like `09:15` or date slashes like `07/01`
    if (i < len && (text[i] === ':' || text[i] === '/')) {
      while (i < len && (isAsciiDigit(text[i]!) || text[i] === ':' || text[i] === '/')) i += 1;
      continue;
    }
    const raw = text.slice(start, i).split(',').join('');
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) nums.push(parsed);
  }
  return nums;
}

function makeDraft(name: string, type: ClairicFieldType, patch: Partial<FieldDraft> = {}): FieldDraft {
  return { ...createEmptyFieldDraft(type, name), ...patch };
}

function extractAnswerValue(answers: Record<string, unknown>, key: string): unknown {
  const raw = answers[key];
  return isPlainObject(raw) ? raw.value : raw;
}

/** Map a Clairic auto-detect classification response into a tailored list of extraction fields. */
export function buildSuggestedFields(
  classifyResult: Record<string, unknown> | undefined,
  sourceText?: string,
): {
  categoryLabel: string;
  fields: FieldDraft[];
} {
  const answers = classifyResult !== undefined && isPlainObject(classifyResult.answers) ? classifyResult.answers : {};
  const rawCategory = extractAnswerValue(answers, 'document_type');
  const category: AutoDetectCategory =
    typeof rawCategory === 'string' && (AUTO_DETECT_CATEGORIES as readonly string[]).includes(rawCategory)
      ? (rawCategory as AutoDetectCategory)
      : 'article_or_notes';
  const hasDates = extractAnswerValue(answers, 'has_dates') === true;
  const hasMoney = extractAnswerValue(answers, 'has_money') === true;
  const hasListItems = extractAnswerValue(answers, 'has_list_items') === true;
  const scanned = sourceText !== undefined && sourceText.trim() !== '' ? scanNumbersInText(sourceText) : [];
  const rawMin = scanned.length > 0 ? Math.min(...scanned) : extractAnswerValue(answers, 'observed_min');
  const rawMax = scanned.length > 0 ? Math.max(...scanned) : extractAnswerValue(answers, 'observed_max');

  let fields: FieldDraft[];
  switch (category) {
    case 'invoice':
      fields = [
        makeDraft('vendor', 'text', { instructions: 'Issuing vendor or company name' }),
        makeDraft('invoice_number', 'text', { nullable: true, instructions: 'Invoice or reference number' }),
        makeDraft('issued_date', 'date', { nullable: true, instructions: 'Invoice issue date' }),
        makeDraft('due_date', 'date', { nullable: true, instructions: 'Payment due date if stated' }),
        makeDraft('total', 'number', { decimals: '2', instructions: 'Final total amount due' }),
        makeDraft('tax', 'number', { decimals: '2', nullable: true, instructions: 'Tax amount if stated' }),
        makeDraft('line_items', 'list', {
          listItemKind: 'text',
          listMaxItems: '10',
          instructions: 'Line item product or service descriptions',
        }),
      ];
      break;
    case 'receipt':
      fields = [
        makeDraft('merchant', 'text', { instructions: 'Store, restaurant, or merchant name' }),
        makeDraft('date', 'date', { nullable: true, instructions: 'Purchase date' }),
        makeDraft('total', 'number', { decimals: '2', instructions: 'Final total charged' }),
        makeDraft('tax', 'number', { decimals: '2', nullable: true, instructions: 'Sales tax if shown' }),
        makeDraft('payment_method', 'text', { nullable: true, instructions: 'Card or payment method if shown' }),
        makeDraft('items', 'list', {
          listItemKind: 'text',
          listMaxItems: '12',
          instructions: 'Purchased items listed on the receipt',
        }),
      ];
      break;
    case 'support_ticket':
      fields = [
        makeDraft('customer', 'text', { nullable: true, instructions: 'Customer name or organization' }),
        makeDraft('priority', 'enum', {
          enumValues: ['low', 'medium', 'high', 'critical'],
          instructions: 'Ticket severity level',
        }),
        makeDraft('urgent', 'boolean', { instructions: 'True if production or critical workflow is impacted' }),
        makeDraft('summary', 'text', { instructions: 'Short summary of the reported issue' }),
        makeDraft('sla', 'duration', {
          durationUnit: 'hours',
          nullable: true,
          instructions: 'Expected response or resolution SLA if mentioned',
        }),
      ];
      break;
    case 'email':
      fields = [
        makeDraft('sender', 'text', { nullable: true, instructions: 'Sender name or email address' }),
        makeDraft('subject', 'text', { nullable: true, instructions: 'Subject line or main topic' }),
        makeDraft('date', 'date', { nullable: true, instructions: 'Date mentioned or sent date' }),
        makeDraft('action_required', 'boolean', { instructions: 'True if a reply or action is requested' }),
        makeDraft('key_points', 'list', {
          listItemKind: 'text',
          listMaxItems: '6',
          instructions: 'Key facts or requests from the message',
        }),
      ];
      break;
    case 'resume':
      fields = [
        makeDraft('full_name', 'text', { instructions: 'Candidate full name' }),
        makeDraft('current_role', 'text', { nullable: true, instructions: 'Current or most recent job title' }),
        makeDraft('email', 'text', { nullable: true, instructions: 'Contact email address' }),
        makeDraft('years_experience', 'number', {
          integer: true,
          nullable: true,
          instructions: 'Total years of professional experience',
        }),
        makeDraft('skills', 'list', {
          listItemKind: 'text',
          listMaxItems: '10',
          instructions: 'Key technical or domain skills',
        }),
        makeDraft('education', 'text', { nullable: true, instructions: 'Highest degree and institution' }),
      ];
      break;
    case 'contract':
      fields = [
        makeDraft('parties', 'list', {
          listItemKind: 'text',
          listMaxItems: '6',
          instructions: 'Organizations or people entering the agreement',
        }),
        makeDraft('effective_date', 'date', { nullable: true, instructions: 'Agreement start or effective date' }),
        makeDraft('expiration_date', 'date', { nullable: true, instructions: 'End or renewal date if stated' }),
        makeDraft('total_value', 'number', {
          decimals: '2',
          nullable: true,
          instructions: 'Total contract value if stated',
        }),
        makeDraft('key_terms', 'list', {
          listItemKind: 'text',
          listMaxItems: '6',
          instructions: 'Key obligations, deliverables, or payment terms',
        }),
      ];
      break;
    case 'table':
      fields = [
        makeDraft('title', 'text', { nullable: true, instructions: 'Table title or subject' }),
        makeDraft('columns', 'list', {
          listItemKind: 'text',
          listMaxItems: '10',
          instructions: 'Column headers in the table',
        }),
        makeDraft('row_count', 'compute', { integer: true, instructions: 'Total number of data rows' }),
        makeDraft('key_entries', 'list', {
          listItemKind: 'text',
          listMaxItems: '10',
          instructions: 'Primary item names or row labels',
        }),
      ];
      break;
    case 'chat_or_messages':
      fields = [
        makeDraft('participants', 'list', {
          listItemKind: 'text',
          listMaxItems: '8',
          instructions: 'Names or handles of people in the conversation',
        }),
        makeDraft('topic', 'text', { instructions: 'Main topic of the conversation' }),
        makeDraft('time', 'time', { nullable: true, instructions: 'Timestamp shown in the conversation' }),
        makeDraft('messages', 'list', {
          listItemKind: 'text',
          listMaxItems: '8',
          instructions: 'Key messages or quotes from the chat',
        }),
        makeDraft('tone', 'enum', {
          enumValues: ['neutral', 'friendly', 'humorous', 'urgent', 'frustrated'],
          instructions: 'Overall tone of the conversation',
        }),
      ];
      break;
    case 'time_series_forecast': {
      const nMin = typeof rawMin === 'number' && Number.isFinite(rawMin) ? rawMin : 0;
      const nMax =
        typeof rawMax === 'number' && Number.isFinite(rawMax) && rawMax > nMin
          ? rawMax
          : nMin >= 0
            ? Math.max(100, nMin * 2)
            : nMin + Math.max(10, Math.abs(nMin) * 0.5);
      const span = Math.max(1, nMax - nMin);
      const rawLow = nMin >= 0 ? Math.max(0, nMin - span * 0.4) : nMin - span * 0.4;
      const lowBound = String(Math.floor(rawLow));
      const highBound = String(Math.ceil(nMax + span * 0.5));
      fields = [
        makeDraft('metric', 'text', { instructions: 'Name of the metric being tracked' }),
        makeDraft('latest_value', 'number', { instructions: 'Most recent observed value' }),
        makeDraft('next_period', 'estimate', {
          min: lowBound,
          max: highBound,
          instructions: 'Expected value for the next time period',
        }),
        makeDraft('next_three_periods', 'forecast', {
          min: lowBound,
          max: highBound,
          horizon: '3',
          interval: 'step',
          instructions: 'Forecast for the next 3 periods',
        }),
      ];
      break;
    }
    case 'id_or_form':
      fields = [
        makeDraft('document_title', 'text', { instructions: 'Form title or document type' }),
        makeDraft('full_name', 'text', { nullable: true, instructions: 'Person or entity name on the form' }),
        makeDraft('reference_number', 'text', { nullable: true, instructions: 'ID, account, or form number' }),
        makeDraft('date', 'date', { nullable: true, instructions: 'Issue, birth, or submission date' }),
        makeDraft('key_entries', 'list', {
          listItemKind: 'text',
          listMaxItems: '8',
          instructions: 'Filled-in values or notable items on the form',
        }),
      ];
      break;
    case 'photo_or_image':
      fields = [
        makeDraft('subject', 'text', { instructions: 'Main subject shown in the image' }),
        makeDraft('visible_text', 'list', {
          listItemKind: 'text',
          listMaxItems: '10',
          nullable: true,
          instructions: 'Readable text, labels, or headings visible in the image',
        }),
        makeDraft('brand_or_location', 'text', {
          nullable: true,
          instructions: 'Brand, place, or organization name if visible',
        }),
        makeDraft('key_details', 'list', {
          listItemKind: 'text',
          listMaxItems: '6',
          instructions: 'Important factual details visible in the image',
        }),
      ];
      break;
    case 'article_or_notes':
    default:
      fields = [
        makeDraft('title', 'text', { nullable: true, instructions: 'Title or main heading' }),
        makeDraft('summary', 'text', { instructions: 'One-sentence summary of the content' }),
        makeDraft('key_points', 'list', {
          listItemKind: 'text',
          listMaxItems: '6',
          instructions: 'Main facts or takeaways',
        }),
        makeDraft('entities', 'list', {
          listItemKind: 'text',
          listMaxItems: '8',
          instructions: 'People, companies, or products mentioned',
        }),
      ];
      break;
  }

  const hasDateField = fields.some((f) => f.type === 'date' || f.type === 'datetime');
  if (hasDates && !hasDateField && category !== 'time_series_forecast') {
    fields.push(makeDraft('date', 'date', { nullable: true, instructions: 'Primary date mentioned' }));
  }
  const hasMoneyField = fields.some((f) => f.name === 'total' || f.name === 'total_value' || f.name === 'total_amount');
  if (hasMoney && !hasMoneyField && category !== 'time_series_forecast') {
    fields.push(
      makeDraft('total_amount', 'number', {
        decimals: '2',
        nullable: true,
        instructions: 'Primary price or monetary total mentioned',
      }),
    );
  }
  const hasListField = fields.some((f) => f.type === 'list');
  if (hasListItems && !hasListField && category !== 'time_series_forecast') {
    fields.push(
      makeDraft('items', 'list', {
        listItemKind: 'text',
        listMaxItems: '8',
        instructions: 'List of items or entries mentioned',
      }),
    );
  }

  return { categoryLabel: CATEGORY_LABELS[category], fields };
}

/** Format a Clairic/CLEF answer value as compact text for diffs & tables. */
export function formatAnswerSummary(answer: unknown): string {
  if (!isPlainObject(answer)) return answer == null ? '—' : scalarToText(answer);
  if (answer.type === 'noul' && typeof answer.noul === 'number') {
    return `${answer.noul >= 0.5 ? 'true' : 'false'} (${Math.round(answer.noul * 100)}%)`;
  }
  if (answer.type === 'choice' && typeof answer.choice === 'string') {
    return answer.choice;
  }
  if (answer.type === 'score' && typeof answer.score === 'number') {
    return answer.score.toFixed(2);
  }
  if (isForecastAnswer(answer)) {
    return `[${answer.median.map((v) => String(v)).join(', ')}]`;
  }
  if (isEstimateAnswer(answer)) {
    return `${String(answer.median)} (${String(answer.p10)}–${String(answer.p90)})`;
  }
  if (answer.value === null || answer.value === undefined) return 'null';
  if (Array.isArray(answer.value)) {
    return answer.value
      .map((item) => (typeof item === 'object' && item !== null ? JSON.stringify(item) : scalarToText(item)))
      .join(', ');
  }
  return scalarToText(answer.value);
}

/* -------------------------------------------------------------------------- */
/* Decision Policy & Confidence Routing                                       */
/* -------------------------------------------------------------------------- */

export interface DecisionPolicy {
  /** Minimum confidence (0–1) to auto-accept an Extract field or Decide choice/rating. */
  minConfidence: number;
  /** Upper cutoff (0–1) on Yes/No probability to auto-approve (`Auto-Yes`). */
  yesThreshold: number;
  /** Lower cutoff (0–1) on Yes/No probability to auto-reject (`Auto-No`). */
  noThreshold: number;
}

export const DEFAULT_DECISION_POLICY: DecisionPolicy = {
  minConfidence: 0.85,
  yesThreshold: 0.75,
  noThreshold: 0.25,
};

export type RouteKind = 'auto-yes' | 'auto-no' | 'auto' | 'review' | 'missing';
export type RouteTone = 'success' | 'warning' | 'danger' | 'neutral' | 'info';

export interface AnswerRouteResult {
  route: RouteKind;
  label: string;
  tone: RouteTone;
  confidence: number | null;
  pNull: number | null;
  isNull: boolean;
}

export function clampUnit(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function normalizePolicy(raw: Partial<DecisionPolicy> | undefined): DecisionPolicy {
  const minConfidence = clampUnit(raw?.minConfidence ?? DEFAULT_DECISION_POLICY.minConfidence);
  let noThreshold = clampUnit(raw?.noThreshold ?? DEFAULT_DECISION_POLICY.noThreshold);
  let yesThreshold = clampUnit(raw?.yesThreshold ?? DEFAULT_DECISION_POLICY.yesThreshold);
  if (noThreshold > yesThreshold) {
    const mid = (noThreshold + yesThreshold) / 2;
    noThreshold = mid;
    yesThreshold = mid;
  }
  return { minConfidence, yesThreshold, noThreshold };
}

/** Route a single CLEF answer (`noul` / `choice` / `score`) using live decision thresholds. */
export function evaluateClefAnswerRoute(answer: ClefAnswer, policy: DecisionPolicy): AnswerRouteResult {
  if (answer.type === 'noul') {
    const p = clampUnit(answer.noul);
    if (p >= policy.yesThreshold) {
      return { route: 'auto-yes', label: 'Auto-Yes', tone: 'success', confidence: p, pNull: null, isNull: false };
    }
    if (p <= policy.noThreshold) {
      return { route: 'auto-no', label: 'Auto-No', tone: 'neutral', confidence: 1 - p, pNull: null, isNull: false };
    }
    return {
      route: 'review',
      label: 'Needs review',
      tone: 'warning',
      confidence: Math.max(p, 1 - p),
      pNull: null,
      isNull: false,
    };
  }
  const conf = clampUnit(answer.confidence);
  if (conf >= policy.minConfidence) {
    return { route: 'auto', label: 'Auto-route', tone: 'success', confidence: conf, pNull: null, isNull: false };
  }
  return { route: 'review', label: 'Needs review', tone: 'warning', confidence: conf, pNull: null, isNull: false };
}

/** Route a single Clairic extraction answer against the confidence cutoff. */
export function evaluateClairicAnswerRoute(rawAnswer: unknown, policy: DecisionPolicy): AnswerRouteResult {
  const answer = isPlainObject(rawAnswer) ? rawAnswer : { value: rawAnswer };
  const confidence =
    typeof answer.confidence === 'number' && Number.isFinite(answer.confidence) ? answer.confidence : null;
  const pNull = typeof answer.p_null === 'number' && Number.isFinite(answer.p_null) ? answer.p_null : null;
  const isNull =
    !isForecastAnswer(answer) &&
    !isEstimateAnswer(answer) &&
    !isListAnswer(answer) &&
    (answer.value === null || answer.value === undefined);

  if (answer.valid === false || answer.complete === false) {
    return { route: 'review', label: 'Needs review', tone: 'warning', confidence, pNull, isNull };
  }
  if (isNull) {
    const nullCert = pNull ?? confidence ?? 1;
    if (nullCert >= policy.minConfidence) {
      return { route: 'missing', label: 'Missing', tone: 'neutral', confidence, pNull, isNull: true };
    }
    return { route: 'review', label: 'Verify missing', tone: 'warning', confidence, pNull, isNull: true };
  }
  if (confidence !== null && confidence < policy.minConfidence) {
    return { route: 'review', label: 'Needs review', tone: 'warning', confidence, pNull, isNull: false };
  }
  return { route: 'auto', label: 'Auto-accept', tone: 'success', confidence, pNull, isNull: false };
}

/** Summarize routing counts across all fields/questions in a single System One result. */
export function summarizeRouting(
  result: Record<string, unknown> | undefined,
  policy: DecisionPolicy,
): { autoCount: number; reviewCount: number; missingCount: number; totalCount: number } {
  if (result === undefined || !isPlainObject(result.answers)) {
    return { autoCount: 0, reviewCount: 0, missingCount: 0, totalCount: 0 };
  }
  const clef = isClefResult(result);
  let autoCount = 0;
  let reviewCount = 0;
  let missingCount = 0;
  const values = Object.values(result.answers);
  for (const ans of values) {
    const routed = clef ? evaluateClefAnswerRoute(ans as ClefAnswer, policy) : evaluateClairicAnswerRoute(ans, policy);
    if (routed.route === 'review') reviewCount += 1;
    else if (routed.route === 'missing') missingCount += 1;
    else autoCount += 1;
  }
  return { autoCount, reviewCount, missingCount, totalCount: values.length };
}

/* -------------------------------------------------------------------------- */
/* Threshold-Aware TypeScript Code Generator                                  */
/* -------------------------------------------------------------------------- */

function sanitizePayloadForSnippet(payload: Record<string, unknown>): Record<string, unknown> {
  const sanitized: Record<string, unknown> = { ...payload };
  if (Array.isArray(sanitized.images)) {
    sanitized.images = sanitized.images.map((item) => {
      if (isPlainObject(item) && typeof item.base64 === 'string' && item.base64.length > 64) {
        return { ...item, base64: `${item.base64.slice(0, 24)}...<base64>` };
      }
      return item;
    });
  }
  if (Array.isArray(sanitized.documents)) {
    sanitized.documents = sanitized.documents.map((item) => {
      if (isPlainObject(item) && typeof item.base64 === 'string' && item.base64.length > 64) {
        return { ...item, base64: `${item.base64.slice(0, 24)}...<base64>` };
      }
      return item;
    });
  }
  return sanitized;
}

/** Generate copy-paste TypeScript calling `/v1/systemone` with the user's live threshold routing logic. */
export function buildTypeScriptSnippet(params: {
  model: string;
  mode: SystemOneMode;
  payload: Record<string, unknown>;
  policy: DecisionPolicy;
  endpointUrl?: string;
}): string {
  const endpoint = params.endpointUrl ?? DEFAULT_SYSTEMONE_ENDPOINT;
  const body = sanitizePayloadForSnippet({ model: params.model, ...params.payload });
  const payloadJson = JSON.stringify(body, null, 2);

  if (params.mode === 'decide') {
    const questions = isPlainObject(params.payload.questions) ? Object.entries(params.payload.questions) : [];
    const noulEntry = questions.find(([, q]) => isPlainObject(q) && q.type === 'noul');
    const choiceOrScore = questions.find(([, q]) => isPlainObject(q) && (q.type === 'choice' || q.type === 'score'));
    const lines: string[] = [
      `const YES_THRESHOLD = ${params.policy.yesThreshold.toFixed(2)};`,
      `const NO_THRESHOLD = ${params.policy.noThreshold.toFixed(2)};`,
      `const MIN_CONFIDENCE = ${params.policy.minConfidence.toFixed(2)};`,
      '',
      `const res = await fetch(${JSON.stringify(endpoint)}, {`,
      `  method: 'POST',`,
      `  headers: { 'Content-Type': 'application/json' },`,
      `  body: JSON.stringify(${payloadJson}),`,
      `});`,
      `const { answers } = await res.json();`,
    ];

    if (noulEntry !== undefined) {
      const key = noulEntry[0];
      const safeKey = JSON.stringify(key);
      lines.push(
        '',
        `// Calibrated 3-zone decision routing for ${safeKey}`,
        `const prob = answers[${safeKey}].noul;`,
        `const verdict = prob >= YES_THRESHOLD ? 'auto_yes' : prob <= NO_THRESHOLD ? 'auto_no' : 'human_review';`,
      );
    }
    if (choiceOrScore !== undefined) {
      const key = choiceOrScore[0];
      const safeKey = JSON.stringify(key);
      lines.push(
        '',
        `// Confidence-gated routing for ${safeKey}`,
        `const ans = answers[${safeKey}];`,
        `const needsReview = ans.confidence < MIN_CONFIDENCE;`,
      );
    }
    return lines.join('\n');
  }

  return [
    `const MIN_CONFIDENCE = ${params.policy.minConfidence.toFixed(2)};`,
    '',
    `const res = await fetch(${JSON.stringify(endpoint)}, {`,
    `  method: 'POST',`,
    `  headers: { 'Content-Type': 'application/json' },`,
    `  body: JSON.stringify(${payloadJson}),`,
    `});`,
    `const { answers } = await res.json();`,
    '',
    `// Route low-confidence fields to human review`,
    `const reviewFields = Object.entries(answers).filter(([, ans]) => {`,
    `  if (ans.valid === false || ans.complete === false) return true;`,
    `  if (ans.value === null) return (ans.p_null ?? ans.confidence ?? 1) < MIN_CONFIDENCE;`,
    `  return typeof ans.confidence === 'number' && ans.confidence < MIN_CONFIDENCE;`,
    `});`,
    `const action = reviewFields.length === 0 ? 'auto_accept' : 'human_review';`,
  ].join('\n');
}

/** Generate copy-paste Python (`requests`) calling `/v1/systemone` with live threshold routing. */
export function buildPythonSnippet(params: {
  model: string;
  mode: SystemOneMode;
  payload: Record<string, unknown>;
  policy: DecisionPolicy;
  endpointUrl?: string;
}): string {
  const endpoint = params.endpointUrl ?? DEFAULT_SYSTEMONE_ENDPOINT;
  const body = sanitizePayloadForSnippet({ model: params.model, ...params.payload });
  // Keep valid Python JSON loading via json.loads so booleans/nulls never mismatch Python syntax.
  const jsonStringLiteral = JSON.stringify(JSON.stringify(body, null, 2));

  if (params.mode === 'decide') {
    const questions = isPlainObject(params.payload.questions) ? Object.entries(params.payload.questions) : [];
    const noulEntry = questions.find(([, q]) => isPlainObject(q) && q.type === 'noul');
    const choiceOrScore = questions.find(([, q]) => isPlainObject(q) && (q.type === 'choice' || q.type === 'score'));
    const lines: string[] = [
      'import json',
      'import requests',
      '',
      `YES_THRESHOLD = ${params.policy.yesThreshold.toFixed(2)}`,
      `NO_THRESHOLD = ${params.policy.noThreshold.toFixed(2)}`,
      `MIN_CONFIDENCE = ${params.policy.minConfidence.toFixed(2)}`,
      '',
      `payload = json.loads(${jsonStringLiteral})`,
      `res = requests.post(${JSON.stringify(endpoint)}, json=payload, timeout=120)`,
      'res.raise_for_status()',
      'answers = res.json()["answers"]',
    ];

    if (noulEntry !== undefined) {
      const safeKey = JSON.stringify(noulEntry[0]);
      lines.push(
        '',
        `# Calibrated 3-zone decision routing for ${safeKey}`,
        `prob = answers[${safeKey}]["noul"]`,
        `verdict = "auto_yes" if prob >= YES_THRESHOLD else ("auto_no" if prob <= NO_THRESHOLD else "human_review")`,
      );
    }
    if (choiceOrScore !== undefined) {
      const safeKey = JSON.stringify(choiceOrScore[0]);
      lines.push(
        '',
        `# Confidence-gated routing for ${safeKey}`,
        `needs_review = answers[${safeKey}]["confidence"] < MIN_CONFIDENCE`,
      );
    }
    return lines.join('\n');
  }

  return [
    'import json',
    'import requests',
    '',
    `MIN_CONFIDENCE = ${params.policy.minConfidence.toFixed(2)}`,
    '',
    `payload = json.loads(${jsonStringLiteral})`,
    `res = requests.post(${JSON.stringify(endpoint)}, json=payload, timeout=120)`,
    'res.raise_for_status()',
    'answers = res.json()["answers"]',
    '',
    '# Route low-confidence fields to human review',
    'review_fields = [',
    '    k for k, ans in answers.items()',
    '    if ans.get("valid") is False or ans.get("complete") is False',
    '    or (ans.get("value") is None and (ans.get("p_null") if ans.get("p_null") is not None else ans.get("confidence", 1.0)) < MIN_CONFIDENCE)',
    '    or (ans.get("value") is not None and isinstance(ans.get("confidence"), (int, float)) and ans["confidence"] < MIN_CONFIDENCE)',
    ']',
    'action = "auto_accept" if not review_fields else "human_review"',
  ].join('\n');
}

/** Generate in-process Node/TypeScript snippet using `@mlx-node/lm` directly (no HTTP server required). */
export function buildNodeSdkSnippet(params: {
  modelPath: string;
  mode: SystemOneMode;
  payload: Record<string, unknown>;
  policy: DecisionPolicy;
}): string {
  const sanitized = sanitizePayloadForSnippet(params.payload);
  const payloadJson = JSON.stringify(sanitized, null, 2);

  if (params.mode === 'decide') {
    return [
      `import { ClefModel } from '@mlx-node/lm';`,
      '',
      `const model = await ClefModel.load(${JSON.stringify(params.modelPath)});`,
      `const { answers } = await model.decide(${payloadJson});`,
      '',
      `const YES_THRESHOLD = ${params.policy.yesThreshold.toFixed(2)};`,
      `const NO_THRESHOLD = ${params.policy.noThreshold.toFixed(2)};`,
      `console.log(answers);`,
    ].join('\n');
  }

  return [
    `import { ClairicModel } from '@mlx-node/lm';`,
    '',
    `const model = await ClairicModel.load(${JSON.stringify(params.modelPath)});`,
    `const { answers } = await model.extract(${payloadJson});`,
    '',
    `const MIN_CONFIDENCE = ${params.policy.minConfidence.toFixed(2)};`,
    `console.log(answers);`,
  ].join('\n');
}

/* -------------------------------------------------------------------------- */
/* Reusable Saved Schemas & Rubrics Library (localStorage)                    */
/* -------------------------------------------------------------------------- */

const SAVED_SPECS_STORAGE_KEY = 'mlx-node:systemone:saved-specs:v1';

export interface SavedSpec {
  id: string;
  name: string;
  mode: SystemOneMode;
  updatedAt: number;
  fields?: Record<string, unknown>;
  questions?: Record<string, unknown>;
  options?: {
    referenceDate?: string;
    includeQuantiles?: boolean;
    returnStateText?: boolean;
  };
  policy?: DecisionPolicy;
}

function isSavedSpec(value: unknown): value is SavedSpec {
  if (!isPlainObject(value)) return false;
  if (typeof value.id !== 'string' || value.id === '') return false;
  if (typeof value.name !== 'string' || value.name.trim() === '') return false;
  if (value.mode !== 'extract' && value.mode !== 'decide') return false;
  if (typeof value.updatedAt !== 'number' || !Number.isFinite(value.updatedAt)) return false;
  return true;
}

export function loadSavedSpecs(): SavedSpec[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(SAVED_SPECS_STORAGE_KEY);
    if (raw === null || raw === '') return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isSavedSpec);
  } catch {
    return [];
  }
}

export function persistSavedSpecs(specs: readonly SavedSpec[]): boolean {
  if (typeof localStorage === 'undefined') return false;
  try {
    localStorage.setItem(SAVED_SPECS_STORAGE_KEY, JSON.stringify(specs));
    return true;
  } catch {
    return false;
  }
}

/* -------------------------------------------------------------------------- */
/* Batch Evaluation Mode — Types, Samples, Import/Export & Agreement          */
/* -------------------------------------------------------------------------- */

export type BatchRowStatus = 'idle' | 'queued' | 'running' | 'done' | 'error' | 'cancelled';

export interface BatchRow {
  id: string;
  label: string;
  stateText: string;
  attachments: MediaAttachment[];
  /** Optional expected value for the primary field/question (enables live Agreement % scoring). */
  expected: string;
  status: BatchRowStatus;
  elapsedMs: number | null;
  result?: Record<string, unknown>;
  error?: string;
}

export function createBatchRow(patch: Partial<BatchRow> = {}): BatchRow {
  return {
    id: nextDraftId('case'),
    label: '',
    stateText: '',
    attachments: [],
    expected: '',
    status: 'idle',
    elapsedMs: null,
    ...patch,
  };
}

export interface BatchSampleDataset {
  id: string;
  label: string;
  description: string;
  mode: SystemOneMode;
  presetId: string;
  primaryField: string;
  rows: ReadonlyArray<{ label: string; stateText: string; expected: string }>;
}

export const SYSTEM_ONE_BATCH_SAMPLES: readonly BatchSampleDataset[] = [
  {
    id: 'batch-tickets',
    label: '4 Support tickets',
    description: 'Triage severity, urgency, customer, and SLA across 4 mixed tickets',
    mode: 'extract',
    presetId: 'ticket-triage',
    primaryField: 'priority',
    rows: [
      {
        label: 'P1 webhook outage',
        stateText:
          'From: Maya Lin <maya@cobaltgrid.io>\nSubject: Production webhooks returning 502 since 09:15 UTC\nBody: Every order.paid delivery has failed for 45 minutes. Enterprise SLA is 2 hours.',
        expected: 'critical',
      },
      {
        label: 'CSV export slow',
        stateText:
          'From: Daniel Park <dpark@vervehealth.com>\nSubject: Weekly CSV reports taking 10 minutes\nBody: Exports still complete but our analytics team is delayed. Standard 24 hour response SLA.',
        expected: 'medium',
      },
      {
        label: 'Dark mode request',
        stateText:
          'From: Sofia Alvarez <sofia@brightloop.co>\nSubject: Feature suggestion: keyboard shortcuts\nBody: Would love a shortcut to collapse the sidebar when reviewing dashboards. No rush at all.',
        expected: 'low',
      },
      {
        label: 'SSO login failing',
        stateText:
          'From: Owen Brooks <owen@meridianbank.io>\nSubject: Okta SAML redirect loop blocking 40 analysts\nBody: Nobody on our risk team can sign into production since 08:00. Contract SLA is 4 hours.',
        expected: 'high',
      },
    ],
  },
  {
    id: 'batch-invoices',
    label: '4 Vendor invoices',
    description: 'Extract vendor, total, date, line items, and optional PO across 4 invoices',
    mode: 'extract',
    presetId: 'invoice',
    primaryField: 'vendor',
    rows: [
      {
        label: 'Northstar Robotics',
        stateText:
          'INVOICE #INV-2026-0842\nFrom: Northstar Robotics Supply Co.\nIssued: March 14, 2026\nItems: Stepper Motor NEMA-17 (x4) $148.00; Aluminum Extrusion $72.00\nTotal Due: $238.50. No PO provided.',
        expected: 'Northstar Robotics Supply Co.',
      },
      {
        label: 'Pacific Cloud Hosting',
        stateText:
          'INVOICE #PC-99104\nVendor: Pacific Cloud Hosting LLC\nDate: 2026-04-01\nPO: PO-44812\nItems: GPU Cluster A100-80G (720 hrs) $1,440.00; Object Storage 4TB $92.00\nTotal: $1,532.00',
        expected: 'Pacific Cloud Hosting LLC',
      },
      {
        label: 'Harbor Freight Logistics',
        stateText:
          'Statement from Harbor Freight Logistics Inc.\nInvoice Date: 2026-04-09\nFreight Pallet SF-SEA: $640.00, Insurance: $45.00\nGrand Total: $685.00\nPO Number: PO-7731',
        expected: 'Harbor Freight Logistics Inc.',
      },
      {
        label: 'Lumen Optics Lab',
        stateText:
          'Lumen Optics Lab — Bill #LO-309\nIssued 2026-04-11\nLine items: Aspheric Lens 25mm ($190.00), Optical Mount ($60.00)\nTotal Due: $250.00. Customer did not supply a purchase order.',
        expected: 'Lumen Optics Lab',
      },
    ],
  },
  {
    id: 'batch-hiring',
    label: '5 Hiring candidates',
    description: 'Score clear-yes, borderline-review, and clear-no engineering profiles',
    mode: 'decide',
    presetId: 'clef-decision',
    primaryField: 'advance_to_onsite',
    rows: [
      {
        label: 'Avery Chen (strong)',
        stateText: JSON.stringify(
          {
            applicant: 'Avery Chen',
            role: 'Senior Systems Engineer',
            years_experience: 8,
            summary: 'Built Rust storage engines and Metal compute kernels. Strong open-source track record.',
            requested_level: 'L6',
          },
          null,
          2,
        ),
        expected: 'yes',
      },
      {
        label: 'Jordan Patel (borderline)',
        stateText: JSON.stringify(
          {
            applicant: 'Jordan Patel',
            role: 'Senior Systems Engineer',
            years_experience: 4,
            summary: 'Full-stack TypeScript and Go API engineer who recently wrote a hobby KV store in Rust.',
            requested_level: 'L6',
          },
          null,
          2,
        ),
        expected: 'review',
      },
      {
        label: 'Mina Cho (strong GPU)',
        stateText: JSON.stringify(
          {
            applicant: 'Mina Cho',
            role: 'Senior Systems Engineer',
            years_experience: 9,
            summary: 'Staff CUDA and Triton kernel author for quantized LLM inference; PhD in compilers.',
            requested_level: 'L6',
          },
          null,
          2,
        ),
        expected: 'yes',
      },
      {
        label: 'Sam Rivera (clear no)',
        stateText: JSON.stringify(
          {
            applicant: 'Sam Rivera',
            role: 'Senior Systems Engineer',
            years_experience: 1,
            summary: 'Recent bootcamp grad focused on WordPress sites and marketing landing pages.',
            requested_level: 'L6',
          },
          null,
          2,
        ),
        expected: 'no',
      },
      {
        label: 'Elena Novak (storage)',
        stateText: JSON.stringify(
          {
            applicant: 'Elena Novak',
            role: 'Senior Systems Engineer',
            years_experience: 7,
            summary: 'Core maintainer of a Raft-based distributed SQL engine in C++ and Rust.',
            requested_level: 'L6',
          },
          null,
          2,
        ),
        expected: 'yes',
      },
    ],
  },
  {
    id: 'batch-moderation',
    label: '5 Moderation posts',
    description: 'Test calibrated policy boundaries across criticism, spam, sarcasm, and abuse',
    mode: 'decide',
    presetId: 'clef-moderation',
    primaryField: 'allow_post',
    rows: [
      {
        label: 'Harsh bug feedback',
        stateText: JSON.stringify(
          {
            author: 'user_9481',
            post: 'This update broke our CI twice today. Please stop pushing untested Friday builds.',
          },
          null,
          2,
        ),
        expected: 'yes',
      },
      {
        label: 'Crypto promo spam',
        stateText: JSON.stringify(
          {
            author: 'bot_442',
            post: 'FREE AIRDROP 10X GUARANTEED!! Click bit.ly/claim-now-free-tokens before midnight!!',
          },
          null,
          2,
        ),
        expected: 'no',
      },
      {
        label: 'Borderline snark',
        stateText: JSON.stringify(
          {
            author: 'dev_219',
            post: 'Did anyone on the team even read the RFC or did you just let an intern ship this?',
          },
          null,
          2,
        ),
        expected: 'review',
      },
      {
        label: 'Direct harassment',
        stateText: JSON.stringify(
          {
            author: 'anon_77',
            post: '@alex you are completely incompetent and should be fired immediately, everyone hates working with you.',
          },
          null,
          2,
        ),
        expected: 'no',
      },
      {
        label: 'Helpful repro steps',
        stateText: JSON.stringify(
          {
            author: 'kyle_m',
            post: 'Reproduced on macOS 15.2 with Node 22. Downgrading to v0.4.1 works around the crash.',
          },
          null,
          2,
        ),
        expected: 'yes',
      },
    ],
  },
];

/** Pure RFC-4180 CSV parser (zero regex literals). Returns null on unclosed quotes. */
function parseCsvRows(input: string): string[][] | null {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;
  let i = 0;
  while (i < input.length) {
    const ch = input[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (input[i + 1] === '"') {
          cell += '"';
          i += 2;
        } else {
          inQuotes = false;
          i += 1;
        }
      } else {
        cell += ch;
        i += 1;
      }
      continue;
    }
    if (ch === '"' && cell.length === 0) {
      inQuotes = true;
      i += 1;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
      i += 1;
    } else if (ch === '\r') {
      if (input[i + 1] === '\n') i += 2;
      else i += 1;
      row.push(cell);
      if (row.some((c) => c.trim() !== '')) rows.push(row);
      row = [];
      cell = '';
    } else if (ch === '\n') {
      i += 1;
      row.push(cell);
      if (row.some((c) => c.trim() !== '')) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
      i += 1;
    }
  }
  if (inQuotes) return null;
  row.push(cell);
  if (row.some((c) => c.trim() !== '')) rows.push(row);
  return rows;
}

function objectToBatchEntry(item: unknown, idx: number): { label: string; stateText: string; expected: string } | null {
  if (typeof item === 'string') {
    const trimmed = item.trim();
    return trimmed === '' ? null : { label: `Case ${String(idx + 1)}`, stateText: trimmed, expected: '' };
  }
  if (!isPlainObject(item)) return null;
  const label =
    typeof item.label === 'string'
      ? item.label
      : typeof item.name === 'string'
        ? item.name
        : typeof item.id === 'string'
          ? item.id
          : `Case ${String(idx + 1)}`;
  const hasExplicitExpected =
    Object.hasOwn(item, 'expected') || Object.hasOwn(item, 'target') || Object.hasOwn(item, 'ground_truth');
  const rawExpected = Object.hasOwn(item, 'expected')
    ? item.expected
    : Object.hasOwn(item, 'target')
      ? item.target
      : item.ground_truth;
  const expected = !hasExplicitExpected
    ? ''
    : rawExpected === null
      ? 'null'
      : rawExpected === undefined
        ? ''
        : scalarToText(rawExpected);
  const rawState = item.state ?? item.input ?? item.text ?? item.document ?? item.content;
  const stateText =
    rawState === undefined
      ? JSON.stringify(item, null, 2)
      : typeof rawState === 'string'
        ? rawState
        : JSON.stringify(rawState, null, 2);
  return { label, stateText, expected };
}

/**
 * Parse pasted or uploaded batch cases from JSON array, JSONL, CSV, or plain lines.
 * Zero regex literals.
 */
export function parseBatchImportText(raw: string): Array<{ label: string; stateText: string; expected: string }> {
  const trimmed = raw.trim();
  if (trimmed === '') return [];

  // 1. JSON array
  if (trimmed.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        const out: Array<{ label: string; stateText: string; expected: string }> = [];
        parsed.forEach((item, idx) => {
          const entry = objectToBatchEntry(item, idx);
          if (entry !== null) out.push(entry);
        });
        if (out.length > 0) return out;
      }
    } catch {
      // Fall through to JSONL / CSV / plain lines.
    }
  }

  // 2. JSONL (every non-empty line is a JSON object)
  const rawLines = trimmed
    .split('\r\n')
    .join('\n')
    .split('\r')
    .join('\n')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '');
  if (rawLines.length > 0 && rawLines.every((line) => line.startsWith('{') && line.endsWith('}'))) {
    const out: Array<{ label: string; stateText: string; expected: string }> = [];
    let allParsed = true;
    rawLines.forEach((line, idx) => {
      try {
        const entry = objectToBatchEntry(JSON.parse(line) as unknown, idx);
        if (entry !== null) out.push(entry);
      } catch {
        allParsed = false;
      }
    });
    if (allParsed && out.length > 0) return out;
  }

  // 3. CSV with recognizable header row (require at least one known header so plain comma lines are not dropped)
  const csvRows = parseCsvRows(trimmed);
  if (csvRows !== null && csvRows.length >= 2) {
    const headers = csvRows[0]!.map((h) => h.trim().toLowerCase());
    const inputIdx = headers.findIndex((h) =>
      ['input', 'text', 'state', 'document', 'content', 'prompt', 'message'].includes(h),
    );
    const labelIdx = headers.findIndex((h) => ['label', 'name', 'id', 'title', 'case'].includes(h));
    const expectedIdx = headers.findIndex((h) => ['expected', 'target', 'ground_truth', 'answer'].includes(h));
    const hasRecognizedHeader = inputIdx !== -1 || labelIdx !== -1 || expectedIdx !== -1;
    if (hasRecognizedHeader) {
      const useInputIdx = inputIdx !== -1 ? inputIdx : 0;
      const out: Array<{ label: string; stateText: string; expected: string }> = [];
      for (let r = 1; r < csvRows.length; r += 1) {
        const cols = csvRows[r]!;
        const stateText = (cols[useInputIdx] ?? '').trim();
        if (stateText === '') continue;
        const label =
          labelIdx !== -1 && (cols[labelIdx] ?? '').trim() !== ''
            ? cols[labelIdx]!.trim()
            : `Case ${String(out.length + 1)}`;
        const expected = expectedIdx !== -1 ? (cols[expectedIdx] ?? '').trim() : '';
        out.push({ label, stateText, expected });
      }
      if (out.length > 0) return out;
    }
  }

  // 4. Plain newline-separated entries
  return rawLines.map((line, idx) => ({
    label: `Case ${String(idx + 1)}`,
    stateText: line,
    expected: '',
  }));
}

/** Compare a model answer against an optional expected string for batch Agreement %. */
export function compareExpectedAnswer(answer: unknown, expected: string, policy: DecisionPolicy): boolean | null {
  const target = expected.trim().toLowerCase();
  if (target === '') return null;
  if (!isPlainObject(answer)) {
    return scalarToText(answer).trim().toLowerCase() === target;
  }
  if (answer.type === 'noul' && typeof answer.noul === 'number') {
    const routed = evaluateClefAnswerRoute(answer as ClefAnswer, policy);
    if (target === 'review' || target === 'needs review') return routed.route === 'review';
    if (target === 'yes' || target === 'true' || target === 'auto-yes') return routed.route === 'auto-yes';
    if (target === 'no' || target === 'false' || target === 'auto-no') return routed.route === 'auto-no';
    return (answer.noul >= 0.5 ? 'yes' : 'no') === target;
  }
  if (answer.type === 'choice' && typeof answer.choice === 'string') {
    return answer.choice.trim().toLowerCase() === target;
  }
  if (answer.type === 'score' && typeof answer.score === 'number') {
    const rounded = String(Math.round(answer.score));
    const fixed = answer.score.toFixed(2);
    const legendLabel = isPlainObject(answer.legend) ? scalarToText(answer.legend[rounded]).toLowerCase() : '';
    return target === rounded || target === fixed || (legendLabel !== '' && legendLabel === target);
  }
  if (isEstimateAnswer(answer)) {
    const numTarget = Number(target);
    const candidates = [answer.value, answer.median];
    for (const cand of candidates) {
      const candText = scalarToText(cand).trim().toLowerCase();
      if (candText === target) return true;
      const numCand = Number(cand);
      if (Number.isFinite(numCand) && Number.isFinite(numTarget) && Math.abs(numCand - numTarget) < 1e-6) {
        return true;
      }
    }
    return false;
  }
  if (answer.value === null || answer.value === undefined) {
    return target === 'null' || target === 'missing' || target === 'none';
  }
  const actual = formatAnswerSummary(answer).trim().toLowerCase();
  if (actual === target) return true;
  const numActual = Number(actual);
  const numTarget = Number(target);
  if (Number.isFinite(numActual) && Number.isFinite(numTarget)) {
    return Math.abs(numActual - numTarget) < 1e-6;
  }
  return false;
}

function isFormulaPrefix(value: string): boolean {
  if (value.length === 0) return false;
  const first = value[0]!;
  if (first === '=' || first === '@' || first === '\t' || first === '\r') return true;
  if (first === '+' || first === '-') {
    // Allow plain numeric values like `-12.5` or `+3`, neutralize formula expressions like `-1+1` or `+CMD(...)`.
    const num = Number(value.trim());
    return !Number.isFinite(num);
  }
  return false;
}

function escapeCsvCell(value: string): string {
  const safe = isFormulaPrefix(value) ? `'${value}` : value;
  const needsQuote = safe.includes(',') || safe.includes('"') || safe.includes('\n') || safe.includes('\r');
  if (!needsQuote) return safe;
  return `"${safe.split('"').join('""')}"`;
}

/** Export a completed batch table to RFC-4180 CSV. */
export function exportBatchToCsv(params: {
  rows: readonly BatchRow[];
  columns: readonly string[];
  mode: SystemOneMode;
  policy: DecisionPolicy;
}): string {
  const header = ['case', 'input', 'expected', 'status', 'latency_ms'];
  for (const col of params.columns) {
    header.push(col, `${col}_route`);
  }
  const lines = [header.map(escapeCsvCell).join(',')];

  for (let i = 0; i < params.rows.length; i += 1) {
    const row = params.rows[i]!;
    const answers = isPlainObject(row.result) && isPlainObject(row.result.answers) ? row.result.answers : {};
    const cells: string[] = [
      row.label.trim() || `Case ${String(i + 1)}`,
      row.stateText !== '' ? row.stateText : row.attachments.map((a) => a.name).join('; '),
      row.expected,
      row.status,
      row.elapsedMs !== null ? String(Math.round(row.elapsedMs)) : '',
    ];
    for (const col of params.columns) {
      const ans = answers[col];
      if (ans === undefined) {
        cells.push('', '');
      } else {
        const routed =
          params.mode === 'decide' &&
          isPlainObject(ans) &&
          (ans.type === 'noul' || ans.type === 'choice' || ans.type === 'score')
            ? evaluateClefAnswerRoute(ans as ClefAnswer, params.policy)
            : evaluateClairicAnswerRoute(ans, params.policy);
        cells.push(formatAnswerSummary(ans), routed.label);
      }
    }
    lines.push(cells.map(escapeCsvCell).join(','));
  }

  return lines.join('\n');
}
