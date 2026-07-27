/**
 * The MessagePort hop, and the only code that runs in the Admin renderer's
 * process outside the page itself.
 *
 * It exposes NOTHING. No `contextBridge`, no `ipcRenderer` handle, no helper
 * object on `window`. The SPA's entire capability is one `MessagePort` to the
 * ADMIN utilityProcess, and a port cannot be widened by the page that holds it —
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

/** Keep in sync with `ADMIN_PORT_CHANNEL` / `ADMIN_READY_CHANNEL` in `src/main/window-policy.ts`. */
const ADMIN_PORT_CHANNEL = 'mlx:admin-port';
const ADMIN_READY_CHANNEL = 'mlx:admin-ready';

/**
 * The page tells us it has a `message` listener installed; we ask MAIN for a
 * port. Pull, not push: a transferred port is consumed on delivery, so a port
 * pushed before the page was listening is simply lost, and there is no way to
 * re-send it — only to build a new channel. Asking makes a reload ordinary.
 *
 * Any script running in the page can send this. That is not a privilege
 * boundary being crossed: the page is our own bundle under a `script-src 'self'`
 * CSP, and anything executing there already has whatever the port grants.
 */
window.addEventListener('message', (event: MessageEvent) => {
  // An iframe's message arrives on this same window with a different `source`.
  // The SPA embeds none, and one that appeared should not be able to open a
  // channel to the runtime.
  if (event.source !== window) return;
  const data: unknown = event.data;
  if (typeof data !== 'object' || data === null) return;
  if ((data as { type?: unknown }).type !== ADMIN_READY_CHANNEL) return;
  ipcRenderer.send(ADMIN_READY_CHANNEL);
});

ipcRenderer.on(ADMIN_PORT_CHANNEL, (event: IpcRendererEvent) => {
  const port = event.ports.at(0);
  if (port === undefined) return;
  // `'*'` rather than the real origin: a `targetOrigin` that fails to match is
  // dropped in silence, and the symptom — a window that renders and never
  // connects — is indistinguishable from the broker being down. There is no
  // third party to protect against here; the message goes to this window only.
  window.postMessage({ type: ADMIN_PORT_CHANNEL }, '*', [port]);
});
