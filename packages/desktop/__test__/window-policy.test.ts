import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

// Not via `@mlx-node/dashboard`: the barrel re-exports the guards from
// `rpc/protocol.js` but not this constant, and the SPA reaches it by relative
// path too. Imported rather than regex-matched so a rename over there fails to
// compile here instead of quietly weakening the assertion.
import { DASHBOARD_PORT_MESSAGE } from '../../dashboard/src/rpc/protocol.js';
import {
  CONTROL_PANEL_PORT_CHANNEL,
  CONTROL_PANEL_READY_CHANNEL,
  CONTROL_PANEL_URL,
  classifyNavigation,
} from '../src/main/window-policy.js';

describe('classifyNavigation', () => {
  it('allows the SPA to route within itself', () => {
    for (const url of [CONTROL_PANEL_URL, 'app://mlx/sessions', 'app://mlx/sessions/abc-123?tab=metrics']) {
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
    expect(source).toContain(`'${CONTROL_PANEL_PORT_CHANNEL}'`);
    expect(source).toContain(`'${CONTROL_PANEL_READY_CHANNEL}'`);
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

  /*
   * The page-facing half of the hop. The SPA compares `event.data` against
   * `DASHBOARD_PORT_MESSAGE` — a bare string, not an envelope — and a mismatch is
   * completely silent on both sides: the window renders its "Connecting to the
   * mlx runtime…" placeholder forever and nothing is logged anywhere. That is
   * exactly what the previous literal did.
   */
  it('posts the tag the SPA is listening for, as a bare string', () => {
    expect(source).toContain(`'${DASHBOARD_PORT_MESSAGE}'`);
    expect(source).toContain(`window.postMessage(DASHBOARD_PORT_MESSAGE, '*', [port])`);
    // An object would satisfy `toContain` on the literal above while still never
    // matching the SPA's `event.data !== DASHBOARD_PORT_MESSAGE` check.
    expect(source).not.toContain(`window.postMessage({ type: CONTROL_PANEL_PORT_CHANNEL }`);
  });

  /*
   * Who starts the handshake. MAIN still answers rather than pushes — a
   * transferred port is consumed once, so pushing on `did-finish-load` races the
   * page's listener with no recovery — but the ASKING has to come from somewhere,
   * and the SPA never asks. The preload does, at `DOMContentLoaded`: deferred
   * module scripts (the SPA is one) all run before it, so the page's `message`
   * listener is provably installed by then.
   */
  it('asks for a port itself, once the page scripts have run', () => {
    expect(source).toContain(`ipcRenderer.send(CONTROL_PANEL_READY_CHANNEL)`);
    expect(source).toContain(`document.addEventListener('DOMContentLoaded', requestPort, { once: true })`);
    // The `readyState` branch matters: a preload that only subscribes would miss
    // a document that had already finished parsing.
    expect(source).toContain(`document.readyState === 'loading'`);
  });
});
