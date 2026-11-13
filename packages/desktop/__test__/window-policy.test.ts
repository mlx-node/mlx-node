import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

// Not via `@mlx-node/dashboard`: the barrel re-exports the guards from
// `rpc/protocol.js` but not this constant, and the SPA reaches it by relative
// path too. Imported rather than regex-matched so a rename over there fails to
// compile here instead of quietly weakening the assertion.
import { DASHBOARD_PORT_MESSAGE, DASHBOARD_UNRESPONSIVE_MESSAGE } from '../../dashboard/src/rpc/protocol.js';
import { FOLDER_PICKED_MESSAGE, PICK_FOLDER_MESSAGE } from '../../dashboard/ui/src/lib/pick-folder.js';
import {
  CONTROL_PANEL_FOLDER_CHANNEL,
  CONTROL_PANEL_PICK_FOLDER_CHANNEL,
  CONTROL_PANEL_PORT_CHANNEL,
  CONTROL_PANEL_READY_CHANNEL,
  CONTROL_PANEL_UNRESPONSIVE_CHANNEL,
  CONTROL_PANEL_URL,
  TRAFFIC_LIGHT_POSITION,
  WINDOW_BACKGROUND,
  classifyNavigation,
  createFolderPicker,
  FOLDER_DIALOG_OPTIONS,
  windowBackground,
} from '../src/main/window-policy.js';

describe('classifyNavigation', () => {
  it('allows the SPA to route within itself', () => {
    for (const url of [CONTROL_PANEL_URL, 'app://mlx/models', 'app://mlx/models/abc-123?tab=metrics']) {
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
    expect(source).toContain(`'${CONTROL_PANEL_UNRESPONSIVE_CHANNEL}'`);
    expect(CONTROL_PANEL_UNRESPONSIVE_CHANNEL).toBe(DASHBOARD_UNRESPONSIVE_MESSAGE);
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
   * The page-facing half of the hop. The tag and broker generation travel
   * together so a recovery report returns the identity of the exact channel
   * that observed the wedge.
   */
  it('posts the tag and validated broker generation the SPA is listening for', () => {
    expect(source).toContain(`'${DASHBOARD_PORT_MESSAGE}'`);
    expect(source).toContain(`window.postMessage({ type: DASHBOARD_PORT_MESSAGE, generation }, '*', [port])`);
    expect(source).toContain(`if (!isBrokerGeneration(generation))`);
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

  it('hard-codes the folder channels and the page messages of the folder dialog', () => {
    expect(source).toContain(`'${CONTROL_PANEL_PICK_FOLDER_CHANNEL}'`);
    expect(source).toContain(`'${CONTROL_PANEL_FOLDER_CHANNEL}'`);
    expect(source).toContain(`'${PICK_FOLDER_MESSAGE}'`);
    expect(source).toContain(`'${FOLDER_PICKED_MESSAGE}'`);
  });

  it('passes only the request id of a folder request to MAIN', () => {
    expect(source).toContain(`ipcRenderer.send(CONTROL_PANEL_PICK_FOLDER_CHANNEL, message.requestId)`);
    expect(source).toContain(`type === PICK_FOLDER_MESSAGE && isRequestId(message.requestId)`);
    expect(source).toContain(`window.postMessage({ type: FOLDER_PICKED_MESSAGE, requestId, path: folder }, '*')`);
  });

  it('maps the page recovery message to MAIN without exposing ipcRenderer', () => {
    expect(source).toContain(`type === CONTROL_PANEL_UNRESPONSIVE_CHANNEL && isBrokerGeneration(message.generation)`);
    expect(source).toContain(`recoverRuntime(message.generation)`);
    expect(source).toContain(`ipcRenderer.send(CONTROL_PANEL_UNRESPONSIVE_CHANNEL, generation)`);
  });
});

describe('window chrome', () => {
  it('paints the background the SPA boots on, per OS scheme', () => {
    expect(WINDOW_BACKGROUND).toEqual({ light: '#ffffff', dark: '#141414' });
    expect(windowBackground(false)).toBe('#ffffff');
    expect(windowBackground(true)).toBe('#141414');
  });

  it('centres the traffic lights in the 40 px title bar', () => {
    expect(TRAFFIC_LIGHT_POSITION).toEqual({ x: 14, y: 14 });
  });
});

describe('the folder dialog', () => {
  interface Sent {
    channel: string;
    args: unknown[];
  }
  function sender(log: Sent[]) {
    return { send: (channel: string, ...args: unknown[]) => log.push({ channel, args }) };
  }
  function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (error: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  function setup() {
    const sent: Sent[] = [];
    const contents = sender(sent);
    const win = { id: 'win' };
    const calls: Array<{ win: unknown; options: unknown }> = [];
    let next = deferred<{ canceled: boolean; filePaths: string[] }>();
    let current: { win: typeof win; contents: typeof contents } | null = { win, contents };
    const pick = createFolderPicker<typeof contents, typeof win>({
      window: () => current,
      showOpenDialog: (target, options) => {
        calls.push({ win: target, options });
        return next.promise;
      },
    });
    return {
      sent,
      contents,
      win,
      calls,
      pick,
      answer: (result: { canceled: boolean; filePaths: string[] }) => {
        const was = next;
        next = deferred();
        was.resolve(result);
      },
      fail: () => {
        const was = next;
        next = deferred();
        was.reject(new Error('no dialog'));
      },
      noWindow: () => {
        current = null;
      },
    };
  }

  const settle = () => new Promise((resolve) => setImmediate(resolve));

  it('opens the fixed dialog on our window and answers the asking sender with the chosen folder', async () => {
    const t = setup();
    t.pick({ sender: t.contents }, 'req-1');
    expect(t.calls).toEqual([{ win: t.win, options: FOLDER_DIALOG_OPTIONS }]);
    expect(FOLDER_DIALOG_OPTIONS).toEqual({
      title: 'Choose a folder for Work',
      buttonLabel: 'Choose',
      properties: ['openDirectory', 'createDirectory'],
    });
    t.answer({ canceled: false, filePaths: ['/Users/me/code/app'] });
    await settle();
    expect(t.sent).toEqual([{ channel: CONTROL_PANEL_FOLDER_CHANNEL, args: ['req-1', '/Users/me/code/app'] }]);
  });

  it('answers null on cancel, on a dialog failure, and to a second request while one is open', async () => {
    const t = setup();
    t.pick({ sender: t.contents }, 'a');
    t.pick({ sender: t.contents }, 'b');
    expect(t.calls).toHaveLength(1);
    expect(t.sent).toEqual([{ channel: CONTROL_PANEL_FOLDER_CHANNEL, args: ['b', null] }]);
    t.answer({ canceled: true, filePaths: ['/ignored'] });
    await settle();
    expect(t.sent.at(-1)).toEqual({ channel: CONTROL_PANEL_FOLDER_CHANNEL, args: ['a', null] });
    t.pick({ sender: t.contents }, 'c');
    t.fail();
    await settle();
    expect(t.sent.at(-1)).toEqual({ channel: CONTROL_PANEL_FOLDER_CHANNEL, args: ['c', null] });
    t.pick({ sender: t.contents }, 'd');
    t.answer({ canceled: false, filePaths: [] });
    await settle();
    expect(t.sent.at(-1)).toEqual({ channel: CONTROL_PANEL_FOLDER_CHANNEL, args: ['d', null] });
  });

  it('ignores another sender, a bad request id, and a request with no window', async () => {
    const t = setup();
    const otherSent: Sent[] = [];
    t.pick({ sender: sender(otherSent) }, 'x');
    for (const id of ['', 'x'.repeat(65), 42, null, { path: '/etc' }]) t.pick({ sender: t.contents }, id);
    t.noWindow();
    t.pick({ sender: t.contents }, 'y');
    await settle();
    expect(t.calls).toEqual([]);
    expect(t.sent).toEqual([]);
    expect(otherSent).toEqual([]);
  });
});
