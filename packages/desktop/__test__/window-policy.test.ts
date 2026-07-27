import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { ADMIN_PORT_CHANNEL, ADMIN_READY_CHANNEL, ADMIN_URL, classifyNavigation } from '../src/main/window-policy.js';

describe('classifyNavigation', () => {
  it('allows the SPA to route within itself', () => {
    for (const url of [ADMIN_URL, 'app://mlx/sessions', 'app://mlx/sessions/abc-123?tab=metrics']) {
      expect(classifyNavigation(url), url).toBe('allow');
    }
  });

  // `app:` is a standard scheme only inside Chromium. To Node's URL parser —
  // which is what runs in MAIN and in this test — it is non-special, and a
  // non-special scheme's origin is the literal string "null". A check written as
  // `url.origin === 'app://mlx'` therefore refuses every real navigation AND
  // accepts this one, whose origin is also "null".
  it('refuses a look-alike host', () => {
    for (const url of ['app://mlx.evil/x', 'app://evil/x', 'app:/relative']) {
      expect(classifyNavigation(url), url).toBe('block');
    }
  });

  // Documentation, model cards and the sidecar's own port are all legitimate
  // links, and none of them may replace the shell window: there is no address
  // bar and no back button to leave with.
  it('sends web links to the browser', () => {
    for (const url of ['https://huggingface.co/mlx-node', 'http://127.0.0.1:51423/health', 'mailto:a@b.co']) {
      expect(classifyNavigation(url), url).toBe('external');
    }
  });

  // `javascript:` would run in the window's own context, next to the preload.
  // Default-deny is what makes this list complete rather than a guess at every
  // scheme that could exist.
  it('blocks everything else outright', () => {
    for (const url of [
      'javascript:alert(1)',
      'data:text/html,<script>1</script>',
      'file:///etc/passwd',
      'chrome://settings',
      'not a url',
      '',
    ]) {
      expect(classifyNavigation(url), url).toBe('block');
    }
  });
});

describe('the preload contract', () => {
  // A sandboxed preload may `require` only `electron` and a few Node builtins:
  // a relative `require` throws at load, and from the page's point of view the
  // failure is silent — the window renders and never receives its port. The
  // channel names are therefore duplicated as literals in the preload, and this
  // is the only thing keeping the two copies honest.
  const source = readFileSync(fileURLToPath(new URL('../src/preload/index.cts', import.meta.url)), 'utf8');

  it('hard-codes exactly the channel names main sends on', () => {
    expect(source).toContain(`'${ADMIN_PORT_CHANNEL}'`);
    expect(source).toContain(`'${ADMIN_READY_CHANNEL}'`);
  });

  // Same constraint, the other half: a relative import compiles to a relative
  // `require`, which throws at load in a sandboxed preload.
  it('imports nothing a sandboxed preload cannot resolve', () => {
    const specifiers = [...source.matchAll(/from '([^']+)'/gu)].map((match) => match[1]);
    expect(specifiers).toEqual(['electron']);
  });

  // The whole point of the preload: one port, no API. Anything exposed on
  // `window` is a capability the page can never give back.
  it('exposes no API to the page', () => {
    expect(source).not.toContain('exposeInMainWorld');
  });
});
