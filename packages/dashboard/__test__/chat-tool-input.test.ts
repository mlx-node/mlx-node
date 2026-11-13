import { describe, expect, it } from 'vite-plus/test';

import {
  readToolInputText,
  TOOL_INPUT_TEXT_MAX,
  WEB_INPUT_KEYS,
  webInputKeys,
  type ReadSteps,
} from '../src/chat/agent/tool-input.js';
import { TOOL_DEFINITIONS } from '../src/chat/web/tools.js';

const FETCH = WEB_INPUT_KEYS.mlx_web__web_fetch;
const SEARCH = WEB_INPUT_KEYS.mlx_web__web_search;

function parsed(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

describe('readToolInputText', () => {
  it('reads the keys each web tool declares, and none for another name', () => {
    for (const def of TOOL_DEFINITIONS) {
      expect(webInputKeys(`mlx_web__${def.name}`)).toEqual(Object.keys(def.inputSchema.properties));
    }
    expect(webInputKeys('__proto__')).toEqual([]);
    expect(webInputKeys('constructor')).toEqual([]);
    expect(webInputKeys('mlx_web__web_crawl')).toEqual([]);
  });

  it('a text it reads as an object is the object a strict JSON reader gives', () => {
    for (const text of [
      '{"url": "https://a.example/"}',
      ' \t\r\n{ "url" : "https:\\/\\/a.example\\/p?q=\\"x\\"" }\n',
      '{"url": "https://a.example/\\u00fc\\ud83d\\ude00"}',
      '{"url": "https://a.example/😀"}',
      '{}',
    ]) {
      const read = readToolInputText(text, FETCH);
      expect([text, read.kind]).toEqual([text, 'object']);
      expect(read.kind === 'object' ? { ...read.value } : null).toEqual(parsed(text));
    }
    expect(readToolInputText('{"query": "q", "count": -1.5e2, "recency": "week"}', SEARCH)).toEqual({
      kind: 'object',
      value: { query: 'q', count: -150, recency: 'week' },
    });
  });

  it('a text that cannot start and end as an object is not one, for any JSON reader', () => {
    for (const text of [
      '',
      '   ',
      '{"query": "Anthropic S-1 prospectus filing',
      'Anthropic S-1 prospectus filing',
      '"{\\"url\\": \\"https://a.example/\\"}"',
      '[{"url": "https://a.example/"}]',
      '{"url": "https://a.example/"}x',
      '\u00a0{"url": "https://a.example/"}',
      '}',
      '{',
    ]) {
      expect([text, readToolInputText(text, FETCH).kind]).toEqual([text, 'not-object']);
      const value = parsed(text);
      expect([text, typeof value === 'object' && value !== null && !Array.isArray(value)]).toEqual([text, false]);
    }
  });

  it('a "{\u2026}" text that breaks the JSON grammar is not an object, whatever its keys and values', () => {
    for (const text of [
      "{'query': 'Anthropic S-1'}",
      '{"query": "Anthropic "S-1" filing"}',
      '{"query": "Anthropic S-1",}',
      '{query: "Anthropic S-1"}',
      "{'url': 'https://attacker.example/'}",
      '{"url": "https://a.example/", "url": \'https://b.example/\'}',
      '{"href": "https://a.example/",}',
      '{"url": ["https://a.example/",]}',
      '{"url": {"a": 1,}}',
      '{"url": {"a" 1}}',
      '{"url": tru}',
      '{"url": }',
      '{"url": NaN}',
      '{"url": 01}',
      '{"url": 1.}',
      '{"url": "a\tb"}',
      '{"url": "a\u0000"}',
      '{"url": "\\x41"}',
      '{"url": "\\u12"}',
      '{"url": "\\ud800\\u12"}',
      '{"url": "a"}}',
      '{"url": "a"} {"url": "b"}',
      '{"url": "a" "b"}',
      `{"url": ${'{'.repeat(1_000)}${'}'.repeat(1_001)}`,
    ]) {
      expect([text, readToolInputText(text, FETCH).kind]).toEqual([text, 'not-object']);
      expect([text, parsed(text)]).toEqual([text, undefined]);
    }
  });

  it('refuses a text in the grammar with two equal keys, other keys, other values, a lone surrogate or a long text', () => {
    for (const text of [
      '{"url": "https://a.example/", "url": "https://b.example/"}',
      '{"url": "https://a.example/", "u\\u0072l": "https://b.example/"}',
      '{"href": "https://a.example/"}',
      '{"url": ["https://a.example/"]}',
      '{"url": {"url": "https://a.example/"}}',
      `{"url": ${'['.repeat(2_000)}${']'.repeat(2_000)}}`,
      '{"url": true}',
      '{"url": null}',
      '{"url": 1e999}',
      '{"url": "\\udc00"}',
      '{"url": "\\ud800\\u0041"}',
      '{"url": "\\ud800\\n"}',
      '{"url": "a\ud800b"}',
      '{"url": "a\udc00b"}',
      `{"url": "${'a'.repeat(TOOL_INPUT_TEXT_MAX)}"}`,
      `{"url": "${'a'.repeat(TOOL_INPUT_TEXT_MAX)}",}`,
      '\ufeff{"url": "https://a.example/"}',
    ]) {
      expect([text.slice(0, 60), readToolInputText(text, FETCH).kind]).toEqual([text.slice(0, 60), 'unreadable']);
    }
  });

  it('every one-character change of a clean text: not-object only when JSON.parse fails, object only as it parses', () => {
    const marks = ['"', "'", ',', ':', '{', '}', '[', ']', '\\', ' ', '\t', 'x', '0', '-', '.', 'e', 'u', '\u0000'];
    for (const clean of ['{"url": "https://a.example/p?q=1", "url": "b"}', '{"query": "q \\"x\\"", "count": -1.5e2}']) {
      const keys = clean.includes('url') ? FETCH : SEARCH;
      const changed: string[] = [];
      for (let i = 0; i <= clean.length; i += 1) {
        changed.push(clean.slice(0, i) + clean.slice(i + 1));
        for (const mark of marks) changed.push(clean.slice(0, i) + mark + clean.slice(i));
      }
      for (const text of changed) {
        const read = readToolInputText(text, keys);
        const value = parsed(text);
        const isObject = typeof value === 'object' && value !== null && !Array.isArray(value);
        if (read.kind === 'not-object') expect([text, isObject]).toEqual([text, false]);
        if (read.kind === 'object') expect([text, { ...read.value }]).toEqual([text, value]);
      }
    }
  });

  it('reads each character at most once: the steps grow with the text, never with its nesting', () => {
    const count = (text: string): number => {
      const steps: ReadSteps = { count: 0 };
      readToolInputText(text, SEARCH, steps);
      return steps.count;
    };
    const long = `{"query": "${'a'.repeat(TOOL_INPUT_TEXT_MAX - 20)}"}`;
    expect(count(long)).toBeLessThanOrEqual(long.length + 2);
    expect(count(`{"query": ${'['.repeat(5_000)}${']'.repeat(5_000)}}`)).toBeLessThanOrEqual(12);
    expect(count(`{"query": ${'{'.repeat(1_000)}${'}'.repeat(1_001)}`)).toBeLessThanOrEqual(12);
    const deep = `{"query": ${'['.repeat(4_000)}${']'.repeat(4_000)}}`;
    expect(count(deep)).toBeLessThanOrEqual(deep.length + 2);
    expect(count(`{"query": "${'a'.repeat(1_000_000)}"}`)).toBe(0);
    expect(count(`${' '.repeat(1_000)}x`)).toBeLessThanOrEqual(1_001);
  });
});
