/**
 * Tests for {@link PrivacyFilter.redact}, which classifies and then
 * replaces each detected entity span.
 *
 * Gated on `PRIVACY_FILTER_MODEL_DIR` so CI without weights stays green.
 */
import { existsSync } from 'node:fs';

import { PrivacyFilter } from '@mlx-node/privacy';
import { describe, expect, it } from 'vite-plus/test';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

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
