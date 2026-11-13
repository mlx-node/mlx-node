/// <reference types="vite-plus/client" />
/**
 * Browser half of the dev harness: a MessageChannel whose far end rides Vite's HMR socket to the real runtime. It
 * hands `port1` to the page exactly as the desktop preload does, so `main.tsx` runs unchanged.
 */

import { DASHBOARD_PORT_MESSAGE } from '../../src/rpc/protocol.js';

const channel = crypto.randomUUID();
const { port1, port2 } = new MessageChannel();
port2.onmessage = (event) => import.meta.hot?.send('mlx:dev-rpc', { channel, message: event.data });
import.meta.hot?.on('mlx:dev-rpc-reply', (data: { channel?: unknown; message?: unknown }) => {
  if (data.channel === channel) port2.postMessage(data.message);
});
import.meta.hot?.send('mlx:dev-open', { channel });
addEventListener('pagehide', () => import.meta.hot?.send('mlx:dev-close', { channel }));

// The desktop preload's folder dialog: the harness answers with its throwaway Work folder.
addEventListener('message', (event: MessageEvent) => {
  if (event.source !== window) return;
  const data = event.data as { type?: unknown; requestId?: unknown } | null;
  if (typeof data !== 'object' || data === null || data.type !== 'mlx:pick-folder') return;
  if (typeof data.requestId === 'string') import.meta.hot?.send('mlx:dev-pick-folder', { requestId: data.requestId });
});
import.meta.hot?.on('mlx:dev-folder', (data: { requestId?: unknown; path?: unknown }) => {
  if (typeof data.requestId !== 'string') return;
  const path = typeof data.path === 'string' ? data.path : null;
  window.postMessage({ type: 'mlx:folder-picked', requestId: data.requestId, path }, '*');
});

const hand = (): void => window.postMessage({ type: DASHBOARD_PORT_MESSAGE, generation: 1 }, '*', [port1]);
// Wait for DOMContentLoaded even though a module script already runs at 'interactive': main.tsx installs its
// message listener only when its larger module graph has run, which can be after this script in dev.
if (document.readyState === 'complete') hand();
else document.addEventListener('DOMContentLoaded', hand, { once: true });
