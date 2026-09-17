import { describe, expect, it } from 'vite-plus/test';

import { decideAutoStart } from '../src/main/auto-start.js';

describe('decideAutoStart', () => {
  it('does not fork INFERENCE on a machine with no models', () => {
    // The fresh-install case: autoStartInference defaults on, zero downloads.
    // Without this gate every launch was a guaranteed NoModelsDiscoveredError.
    expect(decideAutoStart({ enabled: true, modelCount: 0 }).start).toBe(false);
  });

  it('starts when there is something to serve', () => {
    expect(decideAutoStart({ enabled: true, modelCount: 2 }).start).toBe(true);
  });

  it('respects the setting before anything else', () => {
    expect(decideAutoStart({ enabled: false, modelCount: 5 }).start).toBe(false);
    expect(decideAutoStart({ enabled: false, modelCount: 0 }).start).toBe(false);
  });

  it('fails open when discovery itself errored', () => {
    // An unreadable models dir is not proof of emptiness; the sidecar's own
    // discovery is authoritative and fails once, clearly.
    expect(decideAutoStart({ enabled: true, modelCount: null }).start).toBe(true);
  });
});
