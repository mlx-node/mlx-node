/**
 * The MCP form reader and checker (`src/chat/agent/elicit.ts`), ported from the agent's own client
 * (`mcp_elicitation/schema.rs`, `validate.rs`); the cases are those of its `schema_tests.rs` and `validate_tests.rs`.
 */

import { describe, expect, it } from 'vite-plus/test';

import {
  type FieldSpec,
  isDate,
  isDateTime,
  isEmail,
  isUri,
  mcpInteraction,
  type McpValue,
  parseForm,
  readElicitRequest,
  validateForm,
} from '../src/chat/agent/elicit.js';

type Json = Record<string, unknown>;

function form(properties: Json, required: string[] = []): FieldSpec[] {
  const specs = parseForm({ type: 'object', properties, required });
  expect(specs).not.toBeNull();
  return specs!;
}

function content(specs: FieldSpec[], values: McpValue[]): Json | undefined {
  const checked = validateForm(specs, values);
  expect(checked).not.toBeNull();
  return checked !== null && 'content' in checked ? checked.content : undefined;
}

function errors(specs: FieldSpec[], values: McpValue[]): unknown {
  const checked = validateForm(specs, values);
  return checked !== null && 'errors' in checked ? checked.errors : null;
}

describe('parseForm (schema.rs)', () => {
  it('keeps the property order and reads required', () => {
    const specs = form({ zeta: { type: 'string' }, alpha: { type: 'string', format: 'email' } }, ['alpha']);
    expect(specs.map((s) => [s.name, s.required])).toEqual([
      ['zeta', false],
      ['alpha', true],
    ]);
    expect(specs[1]).toMatchObject({ type: 'text', format: 'email' });
  });

  it('every row of the field table', () => {
    const specs = form({
      a: { type: 'string', format: 'date-time', minLength: 2, maxLength: 9, default: 'x' },
      b: { type: 'number', minimum: 0.5, maximum: 2.5, default: 1.5 },
      c: { type: 'integer', minimum: 0.5, maximum: 4.5 },
      d: { type: 'boolean', default: true },
      e: { enum: ['r', 'b'], enumNames: ['Red', 'Blue'], default: 'b' },
      f: { oneOf: [{ const: 'prod', title: 'Production' }, { const: 'dev' }], default: 'dev' },
      g: { type: 'array', items: { enum: ['US', 'UK', 'DE'] }, minItems: 1, maxItems: 2, default: ['UK', 'XX'] },
      h: { type: 'array', items: { anyOf: [{ const: 'a', title: 'Alpha' }] } },
      i: { type: 'array', items: { oneOf: [{ const: 'z' }] } },
      j: { type: 'object' },
      k: { type: 'array', items: { type: 'object' } },
      l: { type: 'array' },
      m: {},
      n: { type: 'string', format: 'uri-reference' },
      o: { enum: [1, true, null, 'x'] },
    });
    expect(specs.map((s) => s.type)).toEqual([
      'text',
      'number',
      'integer',
      'boolean',
      'choice',
      'choice',
      'choices',
      'choices',
      'choices',
      'unsupported',
      'unsupported',
      'unsupported',
      'text',
      'text',
      'choice',
    ]);
    expect(specs[0]).toMatchObject({ format: 'date-time', minLength: 2, maxLength: 9, initial: 'x' });
    expect(specs[1]).toMatchObject({ min: 0.5, max: 2.5, initial: '1.5' });
    // A fractional integer bound is tightened inward.
    expect(specs[2]).toMatchObject({ min: 1, max: 4 });
    expect(specs[3]).toMatchObject({ initial: true });
    expect(specs[4]).toMatchObject({
      options: [
        { value: 'r', label: 'Red' },
        { value: 'b', label: 'Blue' },
      ],
      initial: 1,
    });
    expect(specs[5]).toMatchObject({
      options: [
        { value: 'prod', label: 'Production' },
        { value: 'dev', label: 'dev' },
      ],
      initial: 1,
    });
    expect(specs[6]).toMatchObject({ minItems: 1, maxItems: 2, initial: [1] });
    expect(specs[7]).toMatchObject({ options: [{ value: 'a', label: 'Alpha' }] });
    expect(specs[13]).toMatchObject({ format: null });
    expect(specs[14]).toMatchObject({
      options: [
        { value: '1', label: '1' },
        { value: 'true', label: 'true' },
        { value: 'x', label: 'x' },
      ],
    });
  });

  it('a form it cannot use at all is null', () => {
    expect(parseForm('nope')).toBeNull();
    expect(parseForm({ type: 'array', properties: {} })).toBeNull();
    expect(parseForm({ type: 'object' })).toBeNull();
    expect(parseForm({ properties: {} })).toEqual([]);
  });

  it('every cap', () => {
    const many = (n: number): Json =>
      Object.fromEntries(Array.from({ length: n }, (_, i) => [`f${i}`, { type: 'string' }]));
    expect(parseForm({ type: 'object', properties: many(32) })).toHaveLength(32);
    expect(parseForm({ type: 'object', properties: many(33) })).toBeNull();
    const one = (prop: Json, name = 'f'): FieldSpec[] | null =>
      parseForm({ type: 'object', properties: { [name]: prop } });
    expect(one({ type: 'string' }, 'n'.repeat(64))).not.toBeNull();
    expect(one({ type: 'string' }, 'n'.repeat(65))).toBeNull();
    expect(one({ type: 'string', title: 't'.repeat(128) })).not.toBeNull();
    expect(one({ type: 'string', title: 't'.repeat(129) })).toBeNull();
    expect(one({ type: 'string', description: 'd'.repeat(512) })).not.toBeNull();
    expect(one({ type: 'string', description: 'd'.repeat(513) })).toBeNull();
    // A default is a draft: the draft cap, not the description cap.
    expect(one({ type: 'string', default: 'd'.repeat(4_096) })).not.toBeNull();
    expect(one({ type: 'string', default: 'd'.repeat(4_097) })).toBeNull();
    const options = (n: number): string[] => Array.from({ length: n }, (_, i) => `v${i}`);
    expect(one({ enum: options(32) })).not.toBeNull();
    expect(one({ enum: options(33) })).toBeNull();
    expect(one({ enum: ['v'.repeat(129)] })).toBeNull();
    expect(one({ enum: ['v'], enumNames: ['l'.repeat(129)] })).toBeNull();
    expect(one({ type: 'array', items: { enum: options(33) } })).toBeNull();
    expect(one({ type: 'string', description: 'x'.repeat(500), default: 'y'.repeat(4_000) })).not.toBeNull();
    const big = Object.fromEntries(
      Array.from({ length: 20 }, (_, i) => [`f${i}`, { type: 'string', default: 'z'.repeat(4_000) }]),
    );
    expect(parseForm({ type: 'object', properties: big })).toBeNull();
    // Characters, not UTF-16 units.
    expect(one({ type: 'string', title: '😀'.repeat(128) })).not.toBeNull();
  });
});

describe('validateForm (validate.rs)', () => {
  it('a needed field left empty', () => {
    const specs = form({ email: { type: 'string' } }, ['email']);
    expect(errors(specs, [''])).toEqual([{ code: 'required' }]);
  });

  it('text goes as typed: spaces count toward lengths and stay', () => {
    const specs = form({ note: { type: 'string', minLength: 6 } }, ['note']);
    expect(content(specs, ['  ab  '])).toEqual({ note: '  ab  ' });
    expect(errors(specs, ['  ab'])).toEqual([{ code: 'minLength', n: 6 }]);
    expect(content(form({ sep: { type: 'string' } }), ['   '])).toEqual({ sep: '   ' });
    expect(errors(form({ s: { type: 'string', maxLength: 2 } }), ['😀😀😀'])).toEqual([{ code: 'maxLength', n: 2 }]);
    expect(content(form({ s: { type: 'string', maxLength: 2 } }), ['😀😀'])).toEqual({ s: '😀😀' });
  });

  it('emails', () => {
    for (const bad of [
      'not-an-email',
      'a b@example.com',
      'user@',
      'user@nodot',
      'user@@example.com',
      'user@-bad.com',
      'user@bad-.com',
      ' user@example.com',
    ]) {
      expect(isEmail(bad), bad).toBe(false);
    }
    for (const good of ['user@example.com', 'a.b+c@sub.example.co']) expect(isEmail(good), good).toBe(true);
    const specs = form({ e: { type: 'string', format: 'email' } });
    expect(errors(specs, ['nope'])).toEqual([{ code: 'email' }]);
  });

  it('uri: any absolute address with a scheme', () => {
    for (const good of [
      'https://example.com/a?b=1',
      'urn:isbn:0451450523',
      'mailto:user@example.com',
      'ftp://files.example.com/pub',
    ]) {
      expect(isUri(good), good).toBe(true);
    }
    for (const bad of ['not a uri', '/relative/only', 'http//missing-colon']) expect(isUri(bad), bad).toBe(false);
    expect(errors(form({ u: { type: 'string', format: 'uri' } }), ['/x'])).toEqual([{ code: 'uri' }]);
  });

  it('date: zero-padded and a real day', () => {
    for (const good of ['2024-02-29', '2026-12-31']) expect(isDate(good), good).toBe(true);
    for (const bad of ['2023-02-29', '2026-13-01', '2026-00-10', '2026-1-1', 'garbage', '2026-04-31']) {
      expect(isDate(bad), bad).toBe(false);
    }
    expect(errors(form({ d: { type: 'string', format: 'date' } }), ['2026-1-1'])).toEqual([{ code: 'date' }]);
  });

  it('date-time: RFC 3339', () => {
    for (const good of [
      '2026-08-19T10:00:00Z',
      '2026-08-19T10:00:00.123+02:00',
      '2026-08-19t10:00:00z',
      '2026-08-19 10:00:00-05:30',
      '2026-06-30T23:59:60Z',
    ]) {
      expect(isDateTime(good), good).toBe(true);
    }
    for (const bad of [
      '2026-08-19',
      '2026-08-19 10:00:00',
      '2026-08-19T25:00:00Z',
      '2026-08-19T10:00:00.Z',
      '2026-08-19T10:00:00+24:00',
      '2026-08-19T10:00:00+02:00x',
      '2026-08-19T10:60:00Z',
    ]) {
      expect(isDateTime(bad), bad).toBe(false);
    }
    expect(errors(form({ t: { type: 'string', format: 'date-time' } }), ['2026-08-19'])).toEqual([
      { code: 'dateTime' },
    ]);
  });

  it('an unknown format is not checked', () => {
    expect(content(form({ ref: { type: 'string', format: 'uri-reference' } }, ['ref']), ['/relative/path'])).toEqual({
      ref: '/relative/path',
    });
  });

  it('numbers: bounds, spaces around, not a number', () => {
    const specs = form({ ratio: { type: 'number', minimum: 0.5, maximum: 2.5 } }, ['ratio']);
    expect(content(specs, [' 1.25 '])).toEqual({ ratio: 1.25 });
    expect(errors(specs, ['0.1'])).toEqual([{ code: 'min', n: 0.5 }]);
    expect(errors(specs, ['nan'])).toEqual([{ code: 'number' }]);
    expect(errors(specs, ['1,5'])).toEqual([{ code: 'number' }]);
    const open = form({ x: { type: 'number' } });
    expect(content(open, ['-2e3'])).toEqual({ x: -2000 });
    expect(content(open, ['.5'])).toEqual({ x: 0.5 });
    expect(errors(open, ['inf'])).toEqual([{ code: 'number' }]);
    expect(errors(open, ['1e400'])).toEqual([{ code: 'number' }]);
    expect(content(open, [''])).toEqual({});
  });

  it('integers: no fraction, bounds, 64 bits; a value past 2^53 is refused', () => {
    const specs = form({ age: { type: 'integer', minimum: 0, maximum: 120 } }, ['age']);
    expect(content(specs, ['30'])).toEqual({ age: 30 });
    expect(content(specs, ['+30'])).toEqual({ age: 30 });
    expect(errors(specs, ['30.5'])).toEqual([{ code: 'integer' }]);
    expect(errors(specs, ['200'])).toEqual([{ code: 'max', n: 120 }]);
    expect(errors(specs, ['1e20'])).toEqual([{ code: 'integer' }]);
    expect(errors(specs, ['9223372036854775808'])).toEqual([{ code: 'integer' }]);
    const wide = form({ id: { type: 'integer' } }, ['id']);
    expect(content(wide, ['9007199254740991'])).toEqual({ id: 9007199254740991 });
    expect(errors(wide, ['9007199254740993'])).toEqual([{ code: 'max', n: Number.MAX_SAFE_INTEGER }]);
    const fractional = form({ n: { type: 'integer', minimum: 0.5, maximum: 4.5 } }, ['n']);
    expect(errors(fractional, ['0'])).toEqual([{ code: 'min', n: 1 }]);
    expect(content(fractional, ['4'])).toEqual({ n: 4 });
    expect(errors(fractional, ['5'])).toEqual([{ code: 'max', n: 4 }]);
  });

  it('single select: by index; none when needed', () => {
    const specs = form({ color: { type: 'string', enum: ['red', 'blue'] } }, ['color']);
    expect(content(specs, [1])).toEqual({ color: 'blue' });
    expect(errors(specs, [null])).toEqual([{ code: 'required' }]);
    expect(validateForm(specs, [2])).toBeNull();
    expect(content(form({ c: { enum: ['x'] } }), [null])).toEqual({});
  });

  it('multi select: items, min and max; [] for a needed one, nothing for an optional one', () => {
    const specs = form(
      { countries: { type: 'array', items: { type: 'string', enum: ['US', 'UK', 'DE'] }, minItems: 1, maxItems: 2 } },
      ['countries'],
    );
    expect(content(specs, [[2, 0]])).toEqual({ countries: ['US', 'DE'] });
    expect(errors(specs, [[]])).toEqual([{ code: 'minItems', n: 1 }]);
    expect(errors(specs, [[0, 1, 2]])).toEqual([{ code: 'maxItems', n: 2 }]);
    expect(validateForm(specs, [[0, 0]])).toBeNull();
    expect(content(form({ f: { type: 'array', items: { enum: ['a', 'b'] } } }), [[]])).toEqual({});
    expect(content(form({ tags: { type: 'array', items: { enum: ['x', 'y'] } } }, ['tags']), [[]])).toEqual({
      tags: [],
    });
  });

  it('a needed field of a kind it cannot draw fails; an optional one is left out', () => {
    expect(errors(form({ blob: { type: 'object' } }, ['blob']), [])).toEqual([]);
    expect(validateForm(form({ blob: { type: 'object' } }, ['blob']), [])).toEqual({ errors: [] });
    const specs = form({ blobs: { type: 'array', items: { type: 'object' } }, name: { type: 'string' } });
    expect(content(specs, ['n'])).toEqual({ name: 'n' });
  });

  it('booleans; values that do not fit the card are null', () => {
    const specs = form({ ok: { type: 'boolean', default: true } });
    expect(content(specs, [true])).toEqual({ ok: true });
    expect(validateForm(specs, ['true'])).toBeNull();
    expect(validateForm(specs, [])).toBeNull();
    expect(validateForm(form({ s: { type: 'string' } }), ['x'.repeat(4_097)])).toBeNull();
  });
});

describe('the request and its card', () => {
  it('reads both modes and refuses a wrong shape', () => {
    const base = { sessionId: 's', toolCallId: 'mcp-elicit-1', serverName: 'github', message: 'm' };
    expect(readElicitRequest({ ...base, mode: 'form', requestedSchema: { type: 'object' } })).toEqual({
      sessionId: 's',
      toolCallId: 'mcp-elicit-1',
      server: 'github',
      message: 'm',
      mode: 'form',
      schema: { type: 'object' },
    });
    expect(readElicitRequest({ ...base, mode: 'url', url: 'https://x', elicitationId: 'e' })).toMatchObject({
      mode: 'url',
      url: 'https://x',
      elicitationId: 'e',
    });
    expect(readElicitRequest({ ...base, mode: 'url', url: 'https://x' })).toBeNull();
    expect(readElicitRequest({ ...base, mode: 'other' })).toBeNull();
    expect(readElicitRequest({ ...base, sessionId: '', mode: 'form' })).toBeNull();
    expect(readElicitRequest({ ...base, serverName: 3, mode: 'form' })).toBeNull();
    expect(readElicitRequest(null)).toBeNull();
  });

  it('the card shows hidden characters, cuts the message, and is unusable for a needed field it cannot draw', () => {
    const req = {
      sessionId: 's',
      toolCallId: 't',
      server: 'git‮hub',
      message: 'm'.repeat(5_000),
      mode: 'form' as const,
      schema: undefined,
    };
    const card = mcpInteraction('i', req, form({ a: { type: 'string', title: 'A​' }, b: { type: 'null' } }), [], 1);
    expect(card.server).toBe('git\\u{202E}hub');
    expect(card.message).toHaveLength(4_096);
    expect(card.form).toEqual({
      fields: [
        {
          name: 'a',
          title: 'A\\u{200B}',
          description: null,
          required: false,
          type: 'text',
          format: null,
          minLength: null,
          maxLength: null,
          initial: '',
        },
      ],
      leftOut: 1,
    });
    expect(mcpInteraction('i', req, form({ b: { type: 'null' } }, ['b']), [], 1)).toMatchObject({ unusable: 'field' });
    expect(mcpInteraction('i', req, null, [], 1)).toMatchObject({ unusable: 'schema' });
    expect(mcpInteraction('i', req, null, [], 1).form).toBeUndefined();
  });
});
