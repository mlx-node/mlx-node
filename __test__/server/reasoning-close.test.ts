import { describe, expect, it } from 'vite-plus/test';

import { withoutReasoningClose } from '../../packages/server/src/mappers/reasoning-close.js';

describe('withoutReasoningClose', () => {
  it('removes the Qwen and Hy4 close tags', () => {
    expect(withoutReasoningClose('plan</think>')).toBe('plan');
    expect(withoutReasoningClose('</think>')).toBe('');
    expect(withoutReasoningClose('plan</think:6124c78e>\n')).toBe('plan\n');
    expect(withoutReasoningClose('</think:opensource>')).toBe('');
    expect(withoutReasoningClose('a</think>b</think:x>c')).toBe('abc');
  });

  it('keeps text that is not a close tag', () => {
    for (const text of ['</think:>', '</think: x>', '</think:a', '</thinking>', 'a </think b', '<think:x>', 'plan']) {
      expect(withoutReasoningClose(text)).toBe(text);
    }
    expect(withoutReasoningClose('</think:</think>')).toBe('</think:');
  });

  it('runs in linear time', () => {
    const text = '</think:x>'.repeat(200_000) + '</think:'.repeat(200_000);
    const started = performance.now();
    expect(withoutReasoningClose(text)).toBe('</think:'.repeat(200_000));
    expect(performance.now() - started).toBeLessThan(5_000);
  });
});
