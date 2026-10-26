import { describe, expect, it } from 'vite-plus/test';

import { hasAgentVersionBanner } from '../scripts/verify-cli.js';

describe('hasAgentVersionBanner', () => {
  it.each(['1.0.45', '0.0.17', '1.2.3-rc.1', '1.2.3.4', '  1.0.45  ', '1.0.45 (d49757ec37ce)'])(
    'accepts the legacy bare version in %j',
    (banner) => {
      expect(hasAgentVersionBanner(banner)).toBe(true);
    },
  );

  it.each(['grok 1.0.45 (d49757ec37ce)', 'grok 1.0.45', 'grok 0.0.17 (hash)'])(
    'accepts the grok-branded banner in %j',
    (banner) => {
      expect(hasAgentVersionBanner(banner)).toBe(true);
    },
  );

  it.each([
    '',
    '   ',
    'grok',
    'grok ',
    'grok\n1.0.45',
    'grok 1.0',
    'grok 1.0.x',
    'mlx-agent 1.0.45',
    'v1.0.45',
    '1.0',
    '1.0.x',
    'x1.0.45',
  ])('rejects malformed output in %j', (banner) => {
    expect(hasAgentVersionBanner(banner)).toBe(false);
  });
});
