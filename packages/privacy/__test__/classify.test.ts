/**
 * Tests for {@link PrivacyFilter.classify}, the public wrapper around the
 * native `PrivacyFilterModel.classify`.
 *
 * Gated on `PRIVACY_FILTER_MODEL_DIR` so CI without weights stays green.
 */
import { existsSync } from 'node:fs';

import { PrivacyFilter } from '@mlx-node/privacy';
import { describe, expect, it } from 'vite-plus/test';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

describe.skipIf(!modelAvailable)('PrivacyFilter.classify', () => {
  it('detects name and email in a short PII sentence', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { entities } = await pf.classify("Hi, I'm Harry Potter, email: harry@hogwarts.edu");

    const labels = entities.map((e) => e.label).sort();
    expect(labels).toContain('private_person');
    expect(labels).toContain('private_email');

    const person = entities.find((e) => e.label === 'private_person')!;
    // The tokenizer may include a leading space in the span; assert
    // containment rather than exact match so we're robust to that.
    expect(person.text).toContain('Harry Potter');

    const email = entities.find((e) => e.label === 'private_email')!;
    expect(email.text).toContain('harry@hogwarts.edu');
  });

  it('threshold filters low-confidence spans', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { entities } = await pf.classify('Hi, no PII here', { threshold: 0.99 });
    expect(entities).toEqual([]);
  });
});
