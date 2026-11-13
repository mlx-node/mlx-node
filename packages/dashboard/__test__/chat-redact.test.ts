import { describe, expect, it } from 'vite-plus/test';

import { needleList, REDACTED, redactCutText, redactDeep, redactText, StreamRedactor } from '../src/chat/redact.js';
import { promptTitle } from '../src/chat/titles.js';

const KEY = 'marker-KEY-12345678';
const OLD = 'retired-KEY-87654321';
const TOKEN = 'tok-INFERENCE-abcdefghijklmnop';

describe('redactText', () => {
  const needles = needleList([KEY, OLD, TOKEN, '']);

  it('replaces every occurrence of every needle, current and retired', () => {
    expect(redactText(`a ${KEY} b ${OLD} c ${KEY}${TOKEN}`, needles)).toBe(
      `a ${REDACTED} b ${REDACTED} c ${REDACTED}${REDACTED}`,
    );
  });

  it('leaves text without a needle unchanged and ignores empty needles', () => {
    expect(needleList(['', 'x', 'x'])).toEqual(['x']);
    expect(redactText('nothing here', needles)).toBe('nothing here');
    expect(redactText('', needles)).toBe('');
  });

  it('prefers the longest needle at a position', () => {
    expect(redactText('abcdef', needleList(['abc', 'abcdef']))).toBe(REDACTED);
  });

  it('redacts every string of a value, keys included', () => {
    const value = { [KEY]: [`x${KEY}`, 1, null, { deep: TOKEN }], n: 2 };
    expect(redactDeep(value, needles)).toEqual({ [REDACTED]: [`x${REDACTED}`, 1, null, { deep: REDACTED }], n: 2 });
  });
});

describe('StreamRedactor', () => {
  it('catches a secret split over two chunks by holding back L - 1 characters', () => {
    const r = new StreamRedactor(needleList([KEY]));
    const out = [r.push('The key is marker-'), r.push('KEY-12345678 done'), r.flush()].join('');
    expect(out).toBe(`The key is ${REDACTED} done`);
    expect(out).not.toContain('marker-');
  });

  it('sends nothing that could still become a secret', () => {
    const r = new StreamRedactor(needleList([KEY]));
    const first = r.push('hello marker-KEY-1234567');
    expect(first).toBe('hello ');
    expect(r.push('8!')).toBe(REDACTED);
    expect(r.flush()).toBe('!');
  });

  it('chunk by chunk gives the same text as redacting the whole', () => {
    const text = `x${KEY}yy${KEY}${KEY}z${OLD}end`;
    const needles = needleList([KEY, OLD]);
    for (const size of [1, 2, 3, 7, 19, 100]) {
      const r = new StreamRedactor(needles);
      let out = '';
      for (let i = 0; i < text.length; i += size) out += r.push(text.slice(i, i + size));
      out += r.flush();
      expect(out).toBe(redactText(text, needles));
    }
  });

  it('never splits a surrogate pair at the hold-back border', () => {
    const r = new StreamRedactor(needleList(['abc']));
    const out = r.push('x😀y');
    expect(out.endsWith('\uD83D')).toBe(false);
    expect(out + r.flush()).toBe('x😀y');
  });

  it('passes text straight through with no needles', () => {
    const r = new StreamRedactor([]);
    expect(r.push('abc')).toBe('abc');
    expect(r.flush()).toBe('');
  });
});

describe('redactCutText', () => {
  const key = 'fc-0123456789abcdef';
  it('drops the start of a key cut in two at the end', () => {
    expect(redactCutText(`output ${key.slice(0, 9)}`, [key])).toBe('output ');
    expect(redactCutText(`a ${key} b ${key.slice(0, 4)}`, [key])).toBe(`a ${REDACTED} b `);
  });

  it('keeps an end that cannot start a key', () => {
    expect(redactCutText('output done', [key])).toBe('output done');
    expect(redactCutText(`ends with ${key}`, [key])).toBe(`ends with ${REDACTED}`);
  });
});

describe('a prompt title cut inside a key', () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  it('shows no part of the key under a title key, at any depth', () => {
    const title = promptTitle(`use this key ${key} to call the API`);
    expect(title).toContain(key.slice(0, 20));
    const out = redactDeep({ title, rows: [{ title }], parent: { id: 'a', title } }, [key]);
    for (const shown of [out.title, out.rows[0]!.title, out.parent.title]) {
      expect(shown).toBe('use this key…');
    }
    expect(redactDeep({ title: `named ${key}` }, [key]).title).toBe(`named ${REDACTED}`);
    expect(redactDeep({ text: `cut s…` }, [key]).text).toBe('cut s…');
  });
});
