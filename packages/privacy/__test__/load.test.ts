/**
 * Smoke test for the `PrivacyFilterModel` NAPI surface.
 *
 * Gated on the `PRIVACY_FILTER_MODEL_DIR` environment variable so CI runs
 * without weights stay green — same pattern as other model-weight-dependent
 * tests under `__test__/models/`.
 */
import { existsSync } from 'node:fs';

import { PrivacyFilterModel } from '@mlx-node/core';
import { describe, it, expect } from 'vite-plus/test';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

describe.skipIf(!modelAvailable)('PrivacyFilterModel', () => {
  it('classify() detects person and email entities in a PII sentence', () => {
    const m = PrivacyFilterModel.load(MODEL_DIR!);
    const result = m.classify('Hi I am Alice Smith, email alice@example.com', {
      threshold: 0.5,
    });
    expect(result.entities.length).toBeGreaterThanOrEqual(2);
    const labels = result.entities.map((e) => e.label).sort();
    expect(labels).toContain('private_person');
    expect(labels).toContain('private_email');
  });

  it('returnTokens=true populates per-token tags', () => {
    const m = PrivacyFilterModel.load(MODEL_DIR!);
    const result = m.classify('Hi I am Alice Smith, email alice@example.com', {
      threshold: 0.5,
      returnTokens: true,
    });
    expect(result.tokens).toBeDefined();
    expect(result.tokens!.length).toBeGreaterThan(0);
    // At least one token should carry a non-O tag (we have entities).
    expect(result.tokens!.some((t) => t.tag !== 'O')).toBe(true);
  });
});
