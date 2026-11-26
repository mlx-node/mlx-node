/**
 * Work: an MCP server's `_x.ai/mcp/elicit` request (`mcp_elicitation/types.rs`), read and checked; its form schema read
 * as the agent's own client reads it (`schema.rs`, same caps); the values the page sends checked as that client checks
 * them (`validate.rs`); a link request's address checked as that client checks it; and the reply. The agent passes the
 * content to the server unchecked, so the host is the one place that checks it. Typed values are never logged, stored
 * or put in an event or an error.
 */

import { redactText } from '../redact.js';
import type { McpField, McpFieldError, McpInteraction } from '../types.js';
import { shownCut } from './permissions.js';

export const FIELDS_MAX = 32;
const MESSAGE_MAX_CHARS = 4_096;
const NAME_MAX_CHARS = 64;
const TITLE_MAX_CHARS = 128;
const DESCRIPTION_MAX_CHARS = 512;
export const OPTIONS_MAX = 32;
const OPTION_MAX_CHARS = 128;
const SCHEMA_MAX_BYTES = 64 * 1024;
/** A typed value, and a default (a default is a value the user could type). */
export const VALUE_MAX_CHARS = 4_096;
/** The server's name as a card shows it. */
const SERVER_MAX_CHARS = 64;
/** A link request's address and id (`schema.rs`: url 2,048, elicitation id 128). */
export const URL_MAX_CHARS = 2_048;
const ELICITATION_ID_MAX_CHARS = 128;

/** The agent's request. `schema`: the form's `requestedSchema`; absent means an empty form. */
export type ElicitRequest = { sessionId: string; toolCallId: string; server: string; message: string } & (
  | { mode: 'form'; schema: unknown }
  | { mode: 'url'; url: string; elicitationId: string }
);

type TextFormat = Extract<McpField, { type: 'text' }>['format'];

interface Option {
  value: string;
  label: string;
}

/** One property of the schema; `options` hold the values the server reads, which never leave the host. */
export type FieldSpec = { name: string; title: string; description: string | null; required: boolean } & (
  | { type: 'text'; format: TextFormat; minLength: number | null; maxLength: number | null; initial: string | null }
  | { type: 'number'; min: number | null; max: number | null; initial: string | null }
  | { type: 'integer'; min: number | null; max: number | null; initial: string | null }
  | { type: 'boolean'; initial: boolean }
  | { type: 'choice'; options: Option[]; initial: number | null }
  | { type: 'choices'; options: Option[]; minItems: number | null; maxItems: number | null; initial: number[] }
  | { type: 'unsupported' }
);

/** A value the page sends for one shown field: text, a switch, an option index, option indexes; null for none. */
export type McpValue = string | boolean | number | number[] | null;

/** The page's answer to an MCP card. `values`: a form's, one per shown field, in order; a link's accept has none. */
export type McpAnswerInput = { mcp: 'accept'; values?: McpValue[] } | { mcp: 'decline' };

export const MCP_CANCEL = { outcome: 'cancel' } as const;
export const MCP_DECLINE = { outcome: 'decline' } as const;
/** A link's accept: the user opened the page. It carries no content. */
export const LINK_ACCEPT = { outcome: 'accept' } as const;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function own(record: Record<string, unknown>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

/** Characters as the agent counts them (Unicode scalar values). */
function chars(text: string): number {
  let n = 0;
  for (const _ of text) n += 1;
  return n;
}

/** The request, or null when its shape is not the agent's. */
export function readElicitRequest(params: unknown): ElicitRequest | null {
  if (!isPlainObject(params)) return null;
  const sessionId = own(params, 'sessionId');
  const toolCallId = own(params, 'toolCallId');
  const server = own(params, 'serverName');
  const message = own(params, 'message');
  const mode = own(params, 'mode');
  if (typeof sessionId !== 'string' || sessionId === '' || typeof toolCallId !== 'string' || toolCallId === '') {
    return null;
  }
  if (typeof server !== 'string' || typeof message !== 'string') return null;
  const base = { sessionId, toolCallId, server, message };
  if (mode === 'form') return { ...base, mode, schema: own(params, 'requestedSchema') };
  if (mode !== 'url') return null;
  const url = own(params, 'url');
  const elicitationId = own(params, 'elicitationId');
  if (typeof url !== 'string' || (elicitationId !== '' && !isElicitationId(elicitationId))) return null;
  return { ...base, mode, url, elicitationId: elicitationId as string };
}

function isElicitationId(value: unknown): value is string {
  return typeof value === 'string' && value !== '' && chars(value) <= ELICITATION_ID_MAX_CHARS;
}

/**
 * The agent's `_x.ai/mcp/elicit_complete` (`types.rs` `McpElicitCompletePayload`): the server says the browser part
 * of a link request is done. `server`: null when the agent left it out.
 */
export function readElicitComplete(
  params: unknown,
): { sessionId: string; elicitationId: string; server: string | null } | null {
  if (!isPlainObject(params)) return null;
  const sessionId = own(params, 'sessionId');
  const elicitationId = own(params, 'elicitationId');
  const server = own(params, 'serverName');
  if (typeof sessionId !== 'string' || sessionId === '' || !isElicitationId(elicitationId)) return null;
  if (server !== undefined && server !== null && typeof server !== 'string') return null;
  return { sessionId, elicitationId, server: typeof server === 'string' ? server : null };
}

/** A site name with a part in its `xn--` form: letters of another script, which can look like the ones expected. */
function lookalikeHost(host: string): boolean {
  for (const part of host.split('.')) if (part.toLowerCase().startsWith('xn--')) return true;
  return false;
}

/**
 * A link request's address as the agent's own client checks it (`check_elicit_url`): it parses, is plain `http:` or
 * `https:`, names a host and holds no user name or password. `url`: the parsed address as it is opened and shown;
 * `host`: its site, an international name in its `xn--` form, and then `lookalike`. Null when it is not a safe web
 * address.
 */
export function linkCheck(raw: string): { url: string; host: string; secure: boolean; lookalike?: true } | null {
  if (chars(raw) > URL_MAX_CHARS) return null;
  let parsed: URL;
  try {
    parsed = new URL(raw.trim());
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  if (parsed.username !== '' || parsed.password !== '') return null;
  if (parsed.hostname === '') return null;
  const link = { url: parsed.href, host: parsed.hostname, secure: parsed.protocol === 'https:' };
  return lookalikeHost(parsed.hostname) ? { ...link, lookalike: true } : link;
}

/** A JSON scalar as the agent's option value: a string as it is, a number or a boolean as its JSON text. */
function scalarText(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

/** A count bound (`minLength`, `maxItems`…): a whole number of 0 or more, else none. */
function countBound(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function numberBound(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** An integer field's bound; a fractional one is tightened inward to the nearest whole number. */
function integerBound(value: unknown, lower: boolean): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return lower ? Math.ceil(value) : Math.floor(value);
}

function textFormat(value: unknown): TextFormat {
  return value === 'email' || value === 'uri' || value === 'date' || value === 'date-time' ? value : null;
}

/** `const`/`title` entries; one with no string `const` is skipped. */
function constOptions(entries: readonly unknown[]): Option[] {
  const out: Option[] = [];
  for (const entry of entries) {
    if (!isPlainObject(entry)) continue;
    const value = own(entry, 'const');
    if (typeof value !== 'string') continue;
    const title = own(entry, 'title');
    out.push({ value, label: typeof title === 'string' ? title : value });
  }
  return out;
}

function optionsFit(options: readonly Option[]): boolean {
  if (options.length > OPTIONS_MAX) return false;
  return options.every((o) => chars(o.value) <= OPTION_MAX_CHARS && chars(o.label) <= OPTION_MAX_CHARS);
}

function indexOfValue(options: readonly Option[], value: string | null): number | null {
  if (value === null) return null;
  const at = options.findIndex((o) => o.value === value);
  return at === -1 ? null : at;
}

type KindOf<T> = T extends unknown ? Omit<T, 'name' | 'title' | 'description' | 'required'> : never;
type Kind = KindOf<FieldSpec>;

/** One property's kind; null when the whole form must be refused (over a cap). */
function kindOf(prop: Record<string, unknown>, initial: string | null): Kind | null {
  const enumValues = own(prop, 'enum');
  if (Array.isArray(enumValues)) {
    const names = own(prop, 'enumNames');
    const options: Option[] = [];
    enumValues.forEach((raw, i) => {
      const value = scalarText(raw);
      if (value === null) return;
      const name = Array.isArray(names) ? names[i] : undefined;
      options.push({ value, label: typeof name === 'string' ? name : value });
    });
    if (!optionsFit(options)) return null;
    return { type: 'choice', options, initial: indexOfValue(options, initial) };
  }
  const oneOf = own(prop, 'oneOf');
  if (Array.isArray(oneOf)) {
    const options = constOptions(oneOf);
    if (options.length > 0) {
      if (!optionsFit(options)) return null;
      return { type: 'choice', options, initial: indexOfValue(options, initial) };
    }
  }
  const type = own(prop, 'type');
  const kind = typeof type === 'string' ? type : 'string';
  switch (kind) {
    case 'string':
      return {
        type: 'text',
        format: textFormat(own(prop, 'format')),
        minLength: countBound(own(prop, 'minLength')),
        maxLength: countBound(own(prop, 'maxLength')),
        initial,
      };
    case 'number':
      return {
        type: 'number',
        min: numberBound(own(prop, 'minimum')),
        max: numberBound(own(prop, 'maximum')),
        initial,
      };
    case 'integer':
      return {
        type: 'integer',
        min: integerBound(own(prop, 'minimum'), true),
        max: integerBound(own(prop, 'maximum'), false),
        initial,
      };
    case 'boolean':
      return { type: 'boolean', initial: own(prop, 'default') === true };
    case 'array':
      return choicesOf(prop);
    default:
      return { type: 'unsupported' };
  }
}

/** A multi select: `items.enum`, or `items.anyOf` (or `oneOf`) of `const`/`title`; any other array is unsupported. */
function choicesOf(prop: Record<string, unknown>): Kind | null {
  const items = own(prop, 'items');
  if (items === undefined) return { type: 'unsupported' };
  const record = isPlainObject(items) ? items : {};
  const enumValues = own(record, 'enum');
  const titled = own(record, 'anyOf') ?? own(record, 'oneOf');
  let options: Option[];
  if (Array.isArray(enumValues)) {
    options = [];
    for (const raw of enumValues) {
      const value = scalarText(raw);
      if (value !== null) options.push({ value, label: value });
    }
  } else if (Array.isArray(titled)) {
    options = constOptions(titled);
  } else {
    return { type: 'unsupported' };
  }
  if (options.length === 0) return { type: 'unsupported' };
  if (!optionsFit(options)) return null;
  const defaults = own(prop, 'default');
  const initial: number[] = [];
  if (Array.isArray(defaults)) {
    for (const d of defaults) {
      if (typeof d !== 'string') continue;
      const at = options.findIndex((o) => o.value === d);
      if (at !== -1) initial.push(at);
    }
  }
  return {
    type: 'choices',
    options,
    minItems: countBound(own(prop, 'minItems')),
    maxItems: countBound(own(prop, 'maxItems')),
    initial,
  };
}

/** A default as the agent turns it into a draft: a string as it is, a number or a boolean as its JSON text. */
function defaultText(value: unknown): string | null {
  if (value === undefined) return null;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value) ?? null;
}

/**
 * The fields of a `requestedSchema`, in its property order, or null when the form cannot be used at all: not an
 * object, `type` other than `"object"`, no `properties`, over 64 KB, over 32 properties, or a name, title,
 * description, string default or option over its cap.
 */
export function parseForm(schema: unknown): FieldSpec[] | null {
  if (!isPlainObject(schema)) return null;
  const type = own(schema, 'type');
  if (typeof type === 'string' && type !== 'object') return null;
  const required = new Set<string>();
  const list = own(schema, 'required');
  if (Array.isArray(list)) for (const name of list) if (typeof name === 'string') required.add(name);
  const props = own(schema, 'properties');
  if (!isPlainObject(props)) return null;
  let bytes: number;
  try {
    bytes = Buffer.byteLength(JSON.stringify(schema), 'utf8');
  } catch {
    return null;
  }
  if (bytes > SCHEMA_MAX_BYTES) return null;
  const names = Object.keys(props);
  if (names.length > FIELDS_MAX) return null;
  const fields: FieldSpec[] = [];
  for (const name of names) {
    if (chars(name) > NAME_MAX_CHARS) return null;
    const raw = props[name];
    const prop = isPlainObject(raw) ? raw : {};
    const titleRaw = own(prop, 'title');
    const title = typeof titleRaw === 'string' ? titleRaw : name;
    if (chars(title) > TITLE_MAX_CHARS) return null;
    const descriptionRaw = own(prop, 'description');
    const description = typeof descriptionRaw === 'string' ? descriptionRaw : null;
    if (description !== null && chars(description) > DESCRIPTION_MAX_CHARS) return null;
    const fallback = own(prop, 'default');
    if (typeof fallback === 'string' && chars(fallback) > VALUE_MAX_CHARS) return null;
    const kind = kindOf(prop, defaultText(fallback));
    if (kind === null) return null;
    fields.push({ name, title, description, required: required.has(name), ...kind } as FieldSpec);
  }
  return fields;
}

/** The fields the card draws: every field but an unsupported one. */
export function shownSpecs(specs: readonly FieldSpec[]): FieldSpec[] {
  return specs.filter((spec) => spec.type !== 'unsupported');
}

function shownField(spec: FieldSpec, needles: readonly string[]): McpField | null {
  const base = {
    name: shownCut(spec.name, NAME_MAX_CHARS, needles),
    title: shownCut(spec.title, TITLE_MAX_CHARS, needles),
    description: spec.description === null ? null : shownCut(spec.description, DESCRIPTION_MAX_CHARS, needles),
    required: spec.required,
  };
  const labels = (options: readonly Option[]): Array<{ label: string }> =>
    options.map((o) => ({ label: shownCut(o.label, OPTION_MAX_CHARS, needles) }));
  const text = (initial: string | null): string =>
    initial === null ? '' : shownCut(initial, VALUE_MAX_CHARS, needles);
  switch (spec.type) {
    case 'text':
      return {
        ...base,
        type: 'text',
        format: spec.format,
        minLength: spec.minLength,
        maxLength: spec.maxLength,
        initial: text(spec.initial),
      };
    case 'number':
    case 'integer':
      return { ...base, type: spec.type, min: spec.min, max: spec.max, initial: text(spec.initial) };
    case 'boolean':
      return { ...base, type: 'boolean', initial: spec.initial };
    case 'choice':
      return { ...base, type: 'choice', options: labels(spec.options), initial: spec.initial };
    case 'choices':
      return {
        ...base,
        type: 'choices',
        options: labels(spec.options),
        minItems: spec.minItems,
        maxItems: spec.maxItems,
        initial: spec.initial,
      };
    case 'unsupported':
      return null;
  }
}

/**
 * The card of a form request: every text redacted, cut to its cap and with hidden characters shown. A schema the app
 * cannot read, or a needed field of a kind it cannot draw, makes the card `unusable` (only Decline).
 */
export function mcpInteraction(
  id: string,
  req: Extract<ElicitRequest, { mode: 'form' }>,
  specs: readonly FieldSpec[] | null,
  needles: readonly string[],
  openedAt: number,
): McpInteraction {
  const base = {
    id,
    kind: 'mcp-input' as const,
    toolCallId: req.toolCallId,
    openedAt,
    mode: 'form' as const,
    server: shownCut(req.server, SERVER_MAX_CHARS, needles),
    message: shownCut(req.message, MESSAGE_MAX_CHARS, needles),
  };
  if (specs === null) return { ...base, unusable: 'schema' };
  if (specs.some((spec) => spec.type === 'unsupported' && spec.required)) return { ...base, unusable: 'field' };
  const fields: McpField[] = [];
  for (const spec of specs) {
    const shown = shownField(spec, needles);
    if (shown !== null) fields.push(shown);
  }
  return { ...base, form: { fields, leftOut: specs.length - fields.length } };
}

/**
 * The card of a link request: the whole address, its site and whether it is encrypted. An address that is not a safe
 * web address ({@link linkCheck}), or that holds a key as the server sent it or as it opens (a site name is lower
 * case), makes the card `unusable` (only Decline): what is shown is always exactly what opens. So does an empty
 * elicitation id, which the agent passes on: the request is shown so it can be declined, not dropped unseen.
 */
export function linkInteraction(
  id: string,
  req: Extract<ElicitRequest, { mode: 'url' }>,
  needles: readonly string[],
  openedAt: number,
): McpInteraction {
  const base = {
    id,
    kind: 'mcp-input' as const,
    toolCallId: req.toolCallId,
    openedAt,
    mode: 'url' as const,
    server: shownCut(req.server, SERVER_MAX_CHARS, needles),
    message: shownCut(req.message, MESSAGE_MAX_CHARS, needles),
  };
  const link = linkCheck(req.url);
  const keyed = redactText(req.url, needles) !== req.url;
  if (req.elicitationId === '' || link === null || keyed || shownCut(link.url, URL_MAX_CHARS, needles) !== link.url) {
    return { ...base, unusable: 'address' };
  }
  return { ...base, link };
}

function isDigit(c: string | undefined): boolean {
  return c !== undefined && c >= '0' && c <= '9';
}

function digitsAt(text: string, from: number, n: number): number | null {
  let value = 0;
  for (let i = from; i < from + n; i += 1) {
    const c = text[i];
    if (!isDigit(c)) return null;
    value = value * 10 + (c!.charCodeAt(0) - 48);
  }
  return value;
}

function daysIn(year: number, month: number): number {
  if (month === 2) return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0 ? 29 : 28;
  return month === 4 || month === 6 || month === 9 || month === 11 ? 30 : 31;
}

/** `YYYY-MM-DD` at `from`, a real calendar day. */
function dateAt(text: string, from: number): boolean {
  const year = digitsAt(text, from, 4);
  const month = digitsAt(text, from + 5, 2);
  const day = digitsAt(text, from + 8, 2);
  if (year === null || month === null || day === null) return false;
  if (text[from + 4] !== '-' || text[from + 7] !== '-') return false;
  return month >= 1 && month <= 12 && day >= 1 && day <= daysIn(year, month);
}

/** An RFC 3339 full date: zero-padded and a real calendar day. */
export function isDate(text: string): boolean {
  return text.length === 10 && dateAt(text, 0);
}

/**
 * An RFC 3339 date and time, as the agent's client reads one: `T` (or `t`, or a space), seconds up to a leap 60, any
 * fraction, and `Z` or an offset of at most 23:59.
 */
export function isDateTime(text: string): boolean {
  if (text.length < 20 || !dateAt(text, 0)) return false;
  const sep = text[10];
  if (sep !== 'T' && sep !== 't' && sep !== ' ') return false;
  const hour = digitsAt(text, 11, 2);
  const minute = digitsAt(text, 14, 2);
  const second = digitsAt(text, 17, 2);
  if (hour === null || minute === null || second === null || text[13] !== ':' || text[16] !== ':') return false;
  if (hour > 23 || minute > 59 || second > 60) return false;
  let i = 19;
  if (text[i] === '.') {
    i += 1;
    const start = i;
    while (isDigit(text[i])) i += 1;
    if (i === start) return false;
  }
  const zone = text[i];
  if (zone === 'Z' || zone === 'z') return i + 1 === text.length;
  if (zone !== '+' && zone !== '-' && zone !== '−') return false;
  i += 1;
  const offsetHour = digitsAt(text, i, 2);
  const offsetMinute = digitsAt(text, i + 3, 2);
  if (offsetHour === null || offsetMinute === null || text[i + 2] !== ':') return false;
  return offsetHour <= 23 && offsetMinute <= 59 && i + 5 === text.length;
}

function isSpace(c: string): boolean {
  return c.trim() === '';
}

/** One `@`, a local part with no space, and a host name of two or more labels (`validate.rs` `is_plausible_email`). */
export function isEmail(text: string): boolean {
  const at = text.indexOf('@');
  if (at === -1) return false;
  const local = text.slice(0, at);
  const domain = text.slice(at + 1);
  if (local === '' || chars(local) > 64) return false;
  for (const c of local) if (c === '@' || isSpace(c)) return false;
  const labels = domain.split('.');
  if (labels.length < 2) return false;
  return labels.every((label) => {
    if (label === '' || label.length > 63 || label.startsWith('-') || label.endsWith('-')) return false;
    for (const c of label) {
      const ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c === '-';
      if (!ok) return false;
    }
    return true;
  });
}

/** Any absolute address with a scheme, not only http(s). */
export function isUri(text: string): boolean {
  return URL.canParse(text);
}

/** A decimal number as Rust reads an `f64`: a sign, digits with one point, an exponent; or `inf`, `infinity`, `nan`. */
function readNumber(text: string): number | null {
  let i = 0;
  let sign = 1;
  if (text[i] === '+' || text[i] === '-') {
    if (text[i] === '-') sign = -1;
    i += 1;
  }
  const word = text.slice(i).toLowerCase();
  if (word === 'inf' || word === 'infinity') return sign * Number.POSITIVE_INFINITY;
  if (word === 'nan') return Number.NaN;
  let digits = 0;
  while (isDigit(text[i])) {
    i += 1;
    digits += 1;
  }
  if (text[i] === '.') {
    i += 1;
    while (isDigit(text[i])) {
      i += 1;
      digits += 1;
    }
  }
  if (digits === 0) return null;
  if (text[i] === 'e' || text[i] === 'E') {
    i += 1;
    if (text[i] === '+' || text[i] === '-') i += 1;
    const start = i;
    while (isDigit(text[i])) i += 1;
    if (i === start) return null;
  }
  if (i !== text.length) return null;
  return Number(text);
}

const I64_MIN = -(2n ** 63n);
const I64_MAX = 2n ** 63n - 1n;

/** A whole number with an optional sign, within 64 bits; else null. */
function readInteger(text: string): bigint | null {
  let i = 0;
  if (text[i] === '+' || text[i] === '-') i += 1;
  if (i === text.length) return null;
  for (let k = i; k < text.length; k += 1) if (!isDigit(text[k])) return null;
  const value = BigInt(text[0] === '+' ? text.slice(1) : text);
  return value < I64_MIN || value > I64_MAX ? null : value;
}

type Checked = { ok: true; value: unknown } | { ok: true; omit: true } | { ok: false; error: McpFieldError };

const omit: Checked = { ok: true, omit: true };
const fail = (error: McpFieldError): Checked => ({ ok: false, error });

/** One value against its field; a value of the wrong shape for the field is null (the page sent what it must not). */
function checkValue(spec: FieldSpec, value: McpValue): Checked | null {
  switch (spec.type) {
    case 'unsupported':
      return spec.required ? fail({ code: 'required' }) : omit;
    case 'boolean':
      return typeof value === 'boolean' ? { ok: true, value } : null;
    case 'choice': {
      if (value === null) return spec.required ? fail({ code: 'required' }) : omit;
      if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value >= spec.options.length) {
        return null;
      }
      return { ok: true, value: spec.options[value]!.value };
    }
    case 'choices': {
      if (!Array.isArray(value)) return null;
      const seen = new Set<number>();
      for (const pick of value) {
        if (!Number.isInteger(pick) || pick < 0 || pick >= spec.options.length || seen.has(pick)) return null;
        seen.add(pick);
      }
      const picks = [...seen].sort((a, b) => a - b);
      if (spec.minItems !== null && picks.length < spec.minItems) return fail({ code: 'minItems', n: spec.minItems });
      if (spec.maxItems !== null && picks.length > spec.maxItems) return fail({ code: 'maxItems', n: spec.maxItems });
      if (picks.length === 0 && !spec.required) return omit;
      return { ok: true, value: picks.map((pick) => spec.options[pick]!.value) };
    }
    case 'text': {
      if (typeof value !== 'string') return null;
      if (value === '') return spec.required ? fail({ code: 'required' }) : omit;
      const n = chars(value);
      if (spec.minLength !== null && n < spec.minLength) return fail({ code: 'minLength', n: spec.minLength });
      if (spec.maxLength !== null && n > spec.maxLength) return fail({ code: 'maxLength', n: spec.maxLength });
      if (spec.format === 'email' && !isEmail(value)) return fail({ code: 'email' });
      if (spec.format === 'uri' && !isUri(value)) return fail({ code: 'uri' });
      if (spec.format === 'date' && !isDate(value)) return fail({ code: 'date' });
      if (spec.format === 'date-time' && !isDateTime(value)) return fail({ code: 'dateTime' });
      return { ok: true, value };
    }
    case 'integer': {
      if (typeof value !== 'string') return null;
      const text = value.trim();
      if (text === '') return spec.required ? fail({ code: 'required' }) : omit;
      const n = readInteger(text);
      if (n === null) return fail({ code: 'integer' });
      if (spec.min !== null && n < BigInt(spec.min)) return fail({ code: 'min', n: spec.min });
      if (spec.max !== null && n > BigInt(spec.max)) return fail({ code: 'max', n: spec.max });
      // A JSON number past 2^53 would reach the server changed.
      if (n > BigInt(Number.MAX_SAFE_INTEGER)) return fail({ code: 'max', n: Number.MAX_SAFE_INTEGER });
      if (n < BigInt(Number.MIN_SAFE_INTEGER)) return fail({ code: 'min', n: Number.MIN_SAFE_INTEGER });
      return { ok: true, value: Number(n) };
    }
    case 'number': {
      if (typeof value !== 'string') return null;
      const text = value.trim();
      if (text === '') return spec.required ? fail({ code: 'required' }) : omit;
      const n = readNumber(text);
      if (n === null) return fail({ code: 'number' });
      if (spec.min !== null && n < spec.min) return fail({ code: 'min', n: spec.min });
      if (spec.max !== null && n > spec.max) return fail({ code: 'max', n: spec.max });
      if (!Number.isFinite(n)) return fail({ code: 'number' });
      return { ok: true, value: n };
    }
  }
}

/** A text value the page may send: at most {@link VALUE_MAX_CHARS}. */
function valueFits(value: McpValue): boolean {
  return typeof value !== 'string' || value.length <= VALUE_MAX_CHARS;
}

/**
 * The page's values against the form: `content` (by field name, the server's option values) or one error per shown
 * field (null for a field that is fine). Null when the values do not fit the card: a count other than the shown
 * fields, or a value of the wrong shape.
 */
export function validateForm(
  specs: readonly FieldSpec[],
  values: readonly McpValue[],
): { content: Record<string, unknown> } | { errors: Array<McpFieldError | null> } | null {
  const shown = shownSpecs(specs);
  if (values.length !== shown.length || !values.every(valueFits)) return null;
  const content: Record<string, unknown> = {};
  const errors: Array<McpFieldError | null> = [];
  let failed = false;
  for (const spec of specs) {
    const at = shown.indexOf(spec);
    const checked = checkValue(spec, at === -1 ? null : values[at]!);
    if (checked === null) return null;
    if (!checked.ok) {
      failed = true;
      if (at !== -1) errors[at] = checked.error;
      continue;
    }
    if (at !== -1) errors[at] = null;
    if ('value' in checked) content[spec.name] = checked.value;
  }
  return failed ? { errors } : { content };
}

/** The agent's reply for an accepted form. */
export function acceptReply(content: Record<string, unknown>): { outcome: 'accept'; content: Record<string, unknown> } {
  return { outcome: 'accept', content };
}
