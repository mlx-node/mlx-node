import { Buffer } from 'node:buffer';
import { existsSync } from 'node:fs';

/**
 * Tests for {@link PrivacyFilter.redact}, which classifies and then
 * replaces each detected entity span.
 *
 * The model-backed end-to-end tests are gated on `PRIVACY_FILTER_MODEL_DIR`
 * so CI without weights stays green. The pure-function tests for
 * {@link redactImpl} run unconditionally — they cover the UTF-8 byte-offset
 * slicing contract and don't need the model.
 */
import type { Entity } from '@mlx-node/privacy';
import { PrivacyFilter } from '@mlx-node/privacy';
import { describe, expect, it } from 'vite-plus/test';

import { redactImpl } from '../src/redactor.js';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

/**
 * Helper: build an `Entity` whose `start`/`end` are UTF-8 byte offsets
 * computed from the substring's UTF-16 position in `text`. This mirrors
 * what the Rust classifier emits (HF tokenizer offsets are bytes).
 */
function entityFor(text: string, substring: string, label: Entity['label']): Entity {
  const idx = text.indexOf(substring);
  if (idx < 0) throw new Error(`substring ${JSON.stringify(substring)} not found in text`);
  const prefixBytes = Buffer.byteLength(text.slice(0, idx), 'utf8');
  const substringBytes = Buffer.byteLength(substring, 'utf8');
  return {
    start: prefixBytes,
    end: prefixBytes + substringBytes,
    label,
    score: 1.0,
    text: substring,
  };
}

describe('redactImpl (UTF-8 byte offsets)', () => {
  it('returns the input unchanged when there are no entities', () => {
    const text = 'No PII here, just text';
    const { redacted, entities } = redactImpl(text, []);
    expect(redacted).toBe(text);
    expect(entities).toEqual([]);
  });

  it('handles a leading emoji before an email entity', () => {
    // The smiley is 4 UTF-8 bytes but 2 UTF-16 code units. Slicing by
    // String.slice (UTF-16) with byte offsets would shift the cut and
    // leak part of the email; slicing the UTF-8 buffer is correct.
    const text = 'Hi 🙂 email harry@hogwarts.edu';
    const e = entityFor(text, 'harry@hogwarts.edu', 'private_email');
    const { redacted } = redactImpl(text, [e]);
    expect(redacted).toBe('Hi 🙂 email [private_email]');
    expect(redacted).not.toContain('harry@hogwarts.edu');
    // Sanity: the emoji must still be intact.
    expect(redacted).toContain('🙂');
  });

  it('handles a CJK preamble before a name-like entity', () => {
    // Each CJK ideograph is 3 UTF-8 bytes, 1 UTF-16 code unit — so the
    // UTF-8/UTF-16 offsets diverge by 2 per character.
    const text = '你好，我是 Harry Potter';
    const e = entityFor(text, 'Harry Potter', 'private_person');
    const { redacted } = redactImpl(text, [e]);
    expect(redacted).toBe('你好，我是 [private_person]');
    expect(redacted).not.toContain('Harry Potter');
    expect(redacted).toContain('你好，我是');
  });

  it('handles accented Latin (multi-byte UTF-8) before an email entity', () => {
    // 'é' is 2 UTF-8 bytes, 1 UTF-16 code unit.
    const text = 'Café for alice@example.com';
    const e = entityFor(text, 'alice@example.com', 'private_email');
    const { redacted } = redactImpl(text, [e]);
    expect(redacted).toBe('Café for [private_email]');
    expect(redacted).not.toContain('alice@example.com');
    expect(redacted).toContain('Café');
  });

  it('handles multiple non-ASCII entities in one input', () => {
    const text = '🎉 Email harry@hogwarts.edu and 你好 alice@example.com';
    const e1 = entityFor(text, 'harry@hogwarts.edu', 'private_email');
    const e2 = entityFor(text, 'alice@example.com', 'private_email');
    const { redacted } = redactImpl(text, [e1, e2]);
    expect(redacted).toBe('🎉 Email [private_email] and 你好 [private_email]');
  });

  it('filters by labels[] without breaking byte-offset slicing', () => {
    const text = 'Café 🙂 Harry Potter harry@hogwarts.edu';
    const person = entityFor(text, 'Harry Potter', 'private_person');
    const email = entityFor(text, 'harry@hogwarts.edu', 'private_email');
    const { redacted, entities } = redactImpl(text, [person, email], {
      labels: ['private_email'],
      replacement: '[REDACTED]',
    });
    expect(entities).toHaveLength(1);
    expect(entities[0]!.label).toBe('private_email');
    expect(redacted).toBe('Café 🙂 Harry Potter [REDACTED]');
  });
});

describe.skipIf(!modelAvailable)('PrivacyFilter.redact', () => {
  it('replaces with [label] by default', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { redacted } = await pf.redact("Hi, I'm Harry Potter, email: harry@hogwarts.edu", {
      replacement: 'label',
    });
    // The entity span may include a leading space (tokenizer offset
    // convention), so assert containment of both bracketed labels rather
    // than a brittle exact match on the surrounding whitespace.
    expect(redacted).toContain('[private_person]');
    expect(redacted).toContain('[private_email]');
    expect(redacted).not.toContain('Harry Potter');
    expect(redacted).not.toContain('harry@hogwarts.edu');
  });

  it('accepts a custom replacement function', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { redacted } = await pf.redact('Email me at foo@bar.com', {
      replacement: (e) => `<<${e.label}:${e.text.length}>>`,
    });
    expect(redacted).toContain('<<private_email:');
  });

  it('filters by labels[] so only matching entities are redacted', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { redacted, entities } = await pf.redact('Harry Potter — harry@hogwarts.edu', {
      labels: ['private_email'],
      replacement: '[REDACTED]',
    });
    expect(entities.every((e) => e.label === 'private_email')).toBe(true);
    // Person name is filtered out of the redaction set → still present.
    expect(redacted).toContain('Harry Potter');
    expect(redacted).toContain('[REDACTED]');
  });
});
