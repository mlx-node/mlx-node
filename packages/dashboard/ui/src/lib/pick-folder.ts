/**
 * The native folder dialog for Work. The page cannot open one: it posts a request id, the desktop preload (or the
 * dev harness bridge) asks MAIN, and the answer comes back as a window message with the folder the user chose. The
 * page sends nothing but the id, so it cannot pick the path, the title or the dialog's options. The Control Panel
 * checks the folder again when a conversation is created.
 */

/** Keep in sync with `packages/desktop/src/preload/index.cts` and `scripts/dev/bridge.ts`. */
export const PICK_FOLDER_MESSAGE = 'mlx:pick-folder';
export const FOLDER_PICKED_MESSAGE = 'mlx:folder-picked';

/** The dialog waits for the user; past this the answer is "nothing chosen". */
export const PICK_FOLDER_TIMEOUT_MS = 600_000;

/** The folder the user chose, or null (cancelled, no desktop app, timed out). */
export function pickFolder(timeoutMs: number = PICK_FOLDER_TIMEOUT_MS): Promise<string | null> {
  const requestId = crypto.randomUUID();
  return new Promise((resolve) => {
    const finish = (path: string | null): void => {
      window.removeEventListener('message', onMessage);
      clearTimeout(timer);
      resolve(path);
    };
    const onMessage = (event: MessageEvent): void => {
      if (event.source !== window) return;
      const data: unknown = event.data;
      if (typeof data !== 'object' || data === null) return;
      const message = data as { type?: unknown; requestId?: unknown; path?: unknown };
      if (message.type !== FOLDER_PICKED_MESSAGE || message.requestId !== requestId) return;
      finish(typeof message.path === 'string' && message.path !== '' ? message.path : null);
    };
    const timer = setTimeout(() => finish(null), timeoutMs);
    window.addEventListener('message', onMessage);
    window.postMessage({ type: PICK_FOLDER_MESSAGE, requestId }, '*');
  });
}
