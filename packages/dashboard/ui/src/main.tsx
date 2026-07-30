import { connectDashboardApi } from '@/lib/api';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { bindEventTargetPort } from '../../src/rpc/port.js';
import { DASHBOARD_PORT_MESSAGE, DASHBOARD_UNRESPONSIVE_MESSAGE } from '../../src/rpc/protocol.js';
import App from './App.tsx';
import './index.css';

const rootEl = document.getElementById('root');
if (rootEl === null) throw new Error('#root element not found');
const root = createRoot(rootEl);

/**
 * The port arrives from the preload as
 * `window.postMessage({ type: TAG, generation }, '*', [port])` — a MessagePort
 * cannot cross `contextBridge`, so this hop is the documented way in. The
 * listener is registered during module evaluation, before React renders,
 * because the main process sends the port once the page has loaded and a
 * listener attached later would simply miss it.
 *
 * `event.source === window` is the guard that matters: the preload posts into its
 * own window, so anything arriving from a different source is not the handshake
 * whatever tag it carries.
 */
window.addEventListener('message', (event: MessageEvent) => {
  if (event.source !== window || typeof event.data !== 'object' || event.data === null) return;
  const data = event.data as { type?: unknown; generation?: unknown };
  if (
    data.type !== DASHBOARD_PORT_MESSAGE ||
    typeof data.generation !== 'number' ||
    !Number.isSafeInteger(data.generation) ||
    data.generation < 1
  ) {
    return;
  }
  const port = event.ports[0];
  if (port === undefined) return;
  const generation = data.generation;
  connectDashboardApi(bindEventTargetPort(port), {
    onUnresponsive(): void {
      // The preload maps this narrow signal to MAIN's supervised restart path.
      // Asking for another port would reconnect to the SAME wedged process.
      window.postMessage({ type: DASHBOARD_UNRESPONSIVE_MESSAGE, generation }, '*');
    },
  });
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});

// Rendering the app before the port lands would make every page's first fetch
// fail; this placeholder is what the window shows for the few ms in between.
root.render(<div className="text-muted-foreground p-6 text-sm">Connecting to the mlx runtime…</div>);
