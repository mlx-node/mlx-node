import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

const html = readFileSync(new URL('../ui/index.html', import.meta.url), 'utf8');

describe('Control Panel boot screen', () => {
  it('paints visible status before the JavaScript bundle runs', () => {
    expect(html).toContain('class="mlx-boot"');
    expect(html).toContain('role="status"');
    expect(html).toContain('Starting mlx-node…');
  });

  it('paints the boot screen in the window background of either OS scheme', () => {
    expect(html).toContain('light-dark(#ffffff, #141414)');
  });
});
