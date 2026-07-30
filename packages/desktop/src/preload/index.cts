/**
 * The MessagePort hop, and the only code that runs in the Control Panel renderer's
 * process outside the page itself.
 *
 * It exposes NOTHING. No `contextBridge`, no `ipcRenderer` handle, no helper
 * object on `window`. The SPA's entire capability is one `MessagePort` to the
 * CONTROL PANEL utilityProcess, and a port cannot be widened by the page that holds it —
 * which is a much smaller thing to get right than an API surface.
 *
 * Two facts shape every line here:
 *
 *  1. **`.cts`, not `.ts`.** The package is `"type": "module"`, so a compiled
 *     `.ts` would be ESM, and a *sandboxed* preload must be CommonJS. `.cts`
 *     emits `dist/preload/index.cjs` with no bundler and no extra dependency.
 *  2. **No relative `require`.** A sandboxed preload may only `require`
 *     `electron` and a few Node builtins. The channel names below are therefore
 *     duplicated from `src/main/window-policy.ts` on purpose; a test reads this
 *     file and pins them.
 *
 * `contextIsolation` does not get in the way: the preload has its own JavaScript
 * context but the same DOM `window`, so `window.postMessage` with a transfer
 * list reaches the page — this is Electron's own documented MessagePort recipe.
 */

import { ipcRenderer, type IpcRendererEvent } from 'electron';

/** Keep in sync with the three CONTROL PANEL channel constants in `src/main/window-policy.ts`. */
const CONTROL_PANEL_PORT_CHANNEL = 'mlx:control-panel-port';
const CONTROL_PANEL_READY_CHANNEL = 'mlx:control-panel-ready';
const CONTROL_PANEL_UNRESPONSIVE_CHANNEL = 'mlx:control-panel-unresponsive';

/**
 * Keep in sync with `DASHBOARD_PORT_MESSAGE` in
 * `packages/dashboard/src/rpc/protocol.ts` — the tag the SPA compares `event.data`
 * against, as a bare string rather than an envelope.
 *
 * Third literal in this file for the same reason as the other two, and the one
 * that actually crosses a package boundary: `window-policy.test.ts` reads both
 * sources and pins them together, because a mismatch here is completely silent —
 * the page renders its "Connecting…" placeholder forever and nothing is logged
 * on either side.
 */
const DASHBOARD_PORT_MESSAGE = 'mlx:dashboard-port';

/** Ask MAIN for a port. Safe to call again: each request mints a fresh channel. */
function requestPort(): void {
  ipcRenderer.send(CONTROL_PANEL_READY_CHANNEL);
}

function isBrokerGeneration(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;
}

/** Ask MAIN to replace the process that owns this exact brokered generation. */
function recoverRuntime(generation: number): void {
  ipcRenderer.send(CONTROL_PANEL_UNRESPONSIVE_CHANNEL, generation);
}

/**
 * The handshake is started HERE, at `DOMContentLoaded`, rather than by the page.
 *
 * Pull, not push, is still the rule — MAIN answers a request instead of posting
 * blind, so a reload is ordinary and a missed port can always be replaced. What
 * this settles is who does the asking, and the preload can do it without a race:
 * a preload runs before any page script, and deferred module scripts (the SPA is
 * one) all execute BEFORE `DOMContentLoaded`. So by the time this fires, the
 * page's `message` listener is installed. Sending it any earlier is the exact
 * race the pull exists to avoid.
 */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', requestPort, { once: true });
} else {
  requestPort();
}

/**
 * The page may also ask, which is how it recovers if it ever loses the port
 * without navigating.
 *
 * Any script running in the page can send this. That is not a privilege boundary
 * being crossed: the page is our own bundle under a `script-src 'self'` CSP, and
 * anything executing there already has whatever the port grants.
 */
window.addEventListener('message', (event: MessageEvent) => {
  // An iframe's message arrives on this same window with a different `source`.
  // The SPA embeds none, and one that appeared should not be able to open a
  // channel to the runtime.
  if (event.source !== window) return;
  const data: unknown = event.data;
  if (typeof data !== 'object' || data === null) return;
  const message = data as { type?: unknown; generation?: unknown };
  const type = message.type;
  if (type === CONTROL_PANEL_READY_CHANNEL) requestPort();
  if (type === CONTROL_PANEL_UNRESPONSIVE_CHANNEL && isBrokerGeneration(message.generation)) {
    recoverRuntime(message.generation);
  }
});

ipcRenderer.on(CONTROL_PANEL_PORT_CHANNEL, (event: IpcRendererEvent, generation: unknown) => {
  const port = event.ports.at(0);
  if (port === undefined) return;
  if (!isBrokerGeneration(generation)) {
    // MAIN and the preload disagree about the envelope. Retire the transferred
    // capability rather than hand the page a port whose recovery identity is
    // unknowable.
    port.close();
    return;
  }
  // `'*'` rather than the real origin: a `targetOrigin` that fails to match is
  // dropped in silence, and the symptom — a window that renders and never
  // connects — is indistinguishable from the broker being down. There is no
  // third party to protect against here; the message goes to this window only.
  window.postMessage({ type: DASHBOARD_PORT_MESSAGE, generation }, '*', [port]);
});
