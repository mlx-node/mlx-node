/**
 * Pure TypeScript request, schema, and response types for Clairic
 * (System One: Extract).
 *
 * This module must remain 100% native-free (no runtime imports from
 * `@mlx-node/core`, `pdfjs-dist`, or `@napi-rs/canvas`).
 */

/** Maximum combined count of images and rendered PDF pages per request. */
export const CLAIRIC_MAX_IMAGES = 4;

/** Maximum decoded byte size for a single embedded image or PDF (4 MiB). */
export const CLAIRIC_MAX_MEDIA_BYTES = 4 * 1024 * 1024;

/** Maximum aggregate decoded byte size across all images and PDFs (8 MiB). */
export const CLAIRIC_MAX_TOTAL_MEDIA_BYTES = 8 * 1024 * 1024;

/** Maximum rendered pixel budget per PDF page (4,000,000 px). */
export const CLAIRIC_MAX_RENDER_PIXELS = 4_000_000;

/** Maximum decoded raster pixel budget for an image or embedded PDF image (16,000,000 px). */
export const CLAIRIC_MAX_PIXELS = 16_000_000;

/** Maximum render scale factor for PDF pages. */
export const CLAIRIC_MAX_RENDER_SCALE = 2.0;

/** Supported units for `duration` fields. */
export type ClairicDurationUnit = 'seconds' | 'minutes' | 'hours' | 'days' | 'weeks' | 'months' | 'years';

/** Supported scales for `estimate` and `forecast` fields. */
export type ClairicEstimateScale = 'linear' | 'log';

/** Step granularity for date-bounded `estimate` and `forecast` fields. */
export type ClairicDateEstimateStep = number | 'month' | 'year';

/** Supported time intervals for `forecast` fields. */
export type ClairicForecastInterval = 'step' | 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';

/** Scalar item types supported by `list` fields. */
export type ClairicListScalarItemType = 'text' | 'number' | 'date';

/** Clockwise rotation angles (in degrees) accepted for embedded images. */
export type ClairicImageRotation = 0 | 90 | 180 | 270;

/** Supported embedded image MIME types. */
export type ClairicImageContentType = 'image/png' | 'image/jpeg' | 'image/jpg' | 'image/webp';

/** Common options shared across all 13 Clairic field specifications. */
export interface ClairicFieldBase {
  instructions?: string;
  nullable?: boolean;
}

/** Shared numeric formatting options for `number`, `compute`, `duration`, and numeric `list` fields. */
export interface ClairicNumericOptions {
  integer?: boolean;
  decimals?: number;
  unit?: string;
}

/** JSON-Schema numeric grammar constraints supported on `number` fields. */
export interface ClairicNumberGrammarConstraints {
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  enum?: number[];
}

/** 1. `text` field specification. */
export interface ClairicTextFieldSpec extends ClairicFieldBase {
  type: 'text';
  knowledge?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  enum?: string[];
}

/** 2. `number` field specification. */
export interface ClairicNumberFieldSpec
  extends ClairicFieldBase, ClairicNumericOptions, ClairicNumberGrammarConstraints {
  type: 'number';
  knowledge?: boolean;
  source_only?: boolean;
}

/** 3. `compute` field specification (typed numeric head). */
export interface ClairicComputeFieldSpec extends ClairicFieldBase, ClairicNumericOptions {
  type: 'compute';
  null_threshold?: number;
}

/** 4. `date` field specification (ISO `YYYY-MM-DD`). */
export interface ClairicDateFieldSpec extends ClairicFieldBase {
  type: 'date';
  null_threshold?: number;
  source_only?: boolean;
}

/** 5. `time` field specification (`HH:MM` or `HH:MM:SS`). */
export interface ClairicTimeFieldSpec extends ClairicFieldBase {
  type: 'time';
  null_threshold?: number;
  source_only?: boolean;
}

/** 6. `datetime` field specification (`YYYY-MM-DDTHH:MM[:SS]`). */
export interface ClairicDateTimeFieldSpec extends ClairicFieldBase {
  type: 'datetime';
  null_threshold?: number;
  source_only?: boolean;
}

/** 7. `duration` field specification. */
export interface ClairicDurationFieldSpec extends ClairicFieldBase, Omit<ClairicNumericOptions, 'unit'> {
  type: 'duration';
  unit?: ClairicDurationUnit;
  null_threshold?: number;
}

/** 8. `boolean` field specification. */
export interface ClairicBooleanFieldSpec extends ClairicFieldBase {
  type: 'boolean';
}

/** 9. `enum` field specification. */
export interface ClairicEnumFieldSpec extends ClairicFieldBase {
  type: 'enum';
  enum: (string | number | boolean | null)[];
}

/** Property specifications allowed inside an `object` (row) field. */
export type ClairicObjectPropertySpec =
  | ClairicTextFieldSpec
  | ClairicNumberFieldSpec
  | ClairicBooleanFieldSpec
  | ClairicEnumFieldSpec;

/** 10. `object` field specification. */
export interface ClairicObjectFieldSpec extends ClairicFieldBase {
  type: 'object';
  properties: Record<string, ClairicObjectPropertySpec>;
}

/** 11. `list` field specification (scalar list or structured row list). */
export type ClairicListFieldSpec =
  | (ClairicFieldBase &
      ClairicNumericOptions & {
        type: 'list';
        items: ClairicListScalarItemType;
        max_items?: number;
        knowledge?: boolean;
        source_only?: boolean;
        enum?: (string | number)[];
      })
  | (ClairicFieldBase & {
      type: 'list';
      items: ClairicObjectFieldSpec;
      max_items?: number;
      source_only?: boolean;
    });

/** 12. `estimate` field specification (numeric or ISO date bounds). */
export type ClairicEstimateFieldSpec =
  | (ClairicFieldBase & {
      type: 'estimate';
      min: number;
      max: number;
      scale?: ClairicEstimateScale;
      step?: number;
      unit?: string;
    })
  | (ClairicFieldBase & {
      type: 'estimate';
      min: string;
      max: string;
      scale?: 'linear';
      step?: ClairicDateEstimateStep;
      unit?: string;
    });

/** 13. `forecast` field specification (multi-step numeric or ISO date horizon). */
export type ClairicForecastFieldSpec =
  | (ClairicFieldBase & {
      type: 'forecast';
      min: number;
      max: number;
      scale?: ClairicEstimateScale;
      step?: number;
      unit?: string;
      horizon: number;
      interval?: ClairicForecastInterval;
    })
  | (ClairicFieldBase & {
      type: 'forecast';
      min: string;
      max: string;
      scale?: 'linear';
      step?: ClairicDateEstimateStep;
      unit?: string;
      horizon: number;
      interval?: ClairicForecastInterval;
    });

/** Discriminated union of all 13 Clairic field specifications. */
export type ClairicFieldSpec =
  | ClairicTextFieldSpec
  | ClairicNumberFieldSpec
  | ClairicComputeFieldSpec
  | ClairicDateFieldSpec
  | ClairicTimeFieldSpec
  | ClairicDateTimeFieldSpec
  | ClairicDurationFieldSpec
  | ClairicBooleanFieldSpec
  | ClairicEnumFieldSpec
  | ClairicObjectFieldSpec
  | ClairicListFieldSpec
  | ClairicEstimateFieldSpec
  | ClairicForecastFieldSpec;

/** Literal union of all 13 Clairic field type discriminants. */
export type ClairicFieldType = ClairicFieldSpec['type'];

/** Structured embedded image object. */
export interface ClairicImageObject {
  content_type: ClairicImageContentType | (string & {});
  base64: string;
  rotation?: ClairicImageRotation;
}

/** Embedded image input: base64 data URI string or `{ content_type, base64, rotation? }`. */
export type ClairicImageInput = string | ClairicImageObject;

/** Structured embedded PDF document object. */
export interface ClairicDocumentObject {
  content_type: 'application/pdf';
  base64: string;
}

/** Embedded PDF document input: `data:application/pdf;base64,...` URI or `{ content_type, base64 }`. */
export type ClairicDocumentInput = string | ClairicDocumentObject;

/** Optional inference response flags. */
export interface ClairicOptions {
  include_quantiles?: boolean;
  return_state_text?: boolean;
}

/** Clairic extraction request payload. */
export interface ClairicRequest {
  model?: string | null;
  state?: string;
  images?: ClairicImageInput[];
  documents?: ClairicDocumentInput[];
  fields: Record<string, ClairicFieldSpec>;
  reference_date?: string | null;
  options?: ClairicOptions;
}

/** Token and pass accounting returned by Clairic. */
export interface ClairicUsage {
  state_tokens: number;
  image_tokens: number;
  schema_tokens: number;
  input_tokens: number;
  output_tokens: number;
  passes: number;
}

/** Character span `[start, end)` inside `state`. */
export type ClairicSpan = [number, number];

/** Source attribution for an extracted answer. */
export type ClairicAnswerSource = 'input' | 'image' | 'readout';

/** Scalar extraction answer (text, number, compute, date, time, datetime, duration, boolean, enum, object). */
export interface ClairicScalarAnswer {
  value: string | number | boolean | Record<string, unknown> | null;
  guess?: string | number | null;
  valid?: boolean;
  complete?: boolean;
  confidence?: number;
  p_null?: number;
  span?: ClairicSpan | null;
  source?: ClairicAnswerSource | (string & {});
  unit?: string;
}

/** List extraction answer. */
export interface ClairicListAnswer {
  value: (string | number | Record<string, unknown> | null)[];
  valid?: boolean;
  complete?: boolean;
  confidence?: number;
  confidences?: number[];
  p_null?: number;
  p_nulls?: number[];
  spans?: (ClairicSpan | null)[];
  source?: ClairicAnswerSource | (string & {});
}

/** Bounded distributional estimate answer. */
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

/** Multi-step forecast answer aggregated over horizon steps. */
export interface ClairicForecastAnswer {
  value: (number | string)[];
  median: (number | string)[];
  p10: (number | string)[];
  p90: (number | string)[];
  steps: ClairicEstimateAnswer[];
}

/** Union of all answer shapes returned in `ClairicResult.answers`. */
export type ClairicAnswer = ClairicScalarAnswer | ClairicListAnswer | ClairicEstimateAnswer | ClairicForecastAnswer;

/** Clairic extraction response payload. */
export interface ClairicResult {
  model: string;
  answers: Record<string, ClairicAnswer>;
  truncated: boolean;
  usage: ClairicUsage;
  state_text?: string;
}

/** Rasterized RGB24 PDF page handed off to the native Clairic vision encoder. */
export interface RenderedPdfPage {
  /** Tightly packed RGB24 pixels (`width * height * 3` bytes), composited onto white. */
  rgb: Buffer;
  width: number;
  height: number;
  documentIndex: number;
  pageIndex: number;
}

/** Runtime options for Clairic extraction calls. */
export interface ClairicExtractionOptions {
  signal?: AbortSignal;
}

/** Structural interface implemented by Clairic extraction models. */
export interface ExtractionModel {
  readonly kind: 'decision';
  /** Raw JSON retains the caller's field ordering. */
  extractRaw(request: string, options?: ClairicExtractionOptions): Promise<ClairicResult>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Structural type guard for Clairic extraction models. */
export function isClairicModel(model: unknown): model is ExtractionModel {
  return (
    model !== null &&
    typeof model === 'object' &&
    'extractRaw' in model &&
    typeof (model as { extractRaw?: unknown }).extractRaw === 'function'
  );
}

/** True when a parsed System One request payload targets Clairic (`fields` present). */
export function isClairicPayload(parsed: unknown): boolean {
  return isPlainObject(parsed) && Object.hasOwn(parsed, 'fields');
}

/**
 * Lightweight envelope validation for Clairic requests arriving on
 * `POST /v1/systemone`. Full schema and field validation stays in the native
 * Rust parser, which receives the untouched raw JSON string.
 */
export function validateSystemOneClairicEnvelope(
  parsed: unknown,
): asserts parsed is Record<string, unknown> & { model: string; fields: Record<string, unknown> } {
  if (!isPlainObject(parsed)) {
    throw new Error('Request body must be a JSON object');
  }
  if (typeof parsed.model !== 'string' || !parsed.model.trim()) {
    throw new Error('model is required');
  }
  if (Object.hasOwn(parsed, 'questions')) {
    throw new Error('Request cannot specify both questions and fields');
  }
  if (!isPlainObject(parsed.fields)) {
    throw new Error('fields must be a JSON object');
  }
}
