/**
 * The Control Panel window's contract with its renderer: where it may navigate, and the
 * IPC channel names the preload hard-codes.
 *
 * Electron-free so both decisions are testable — `window.ts` holds the
 * `BrowserWindow` and nothing else.
 */

/** The one host `app://` serves the SPA from, so an `app://mlx.evil/` is not us. */
export const APP_SCHEME = 'app:';
export const APP_HOST = 'mlx';

/** What the Control Panel window loads. `BrowserRouter` deep links resolve under it — see `www.ts`. */
export const CONTROL_PANEL_URL = `${APP_SCHEME}//${APP_HOST}/`;

/**
 * MAIN → renderer, carrying one transferred `MessagePort` in `event.ports[0]`
 * and its positive broker generation as the IPC payload.
 *
 * **Duplicated as a string literal in `src/preload/index.cts`, which cannot
 * import it.** A sandboxed preload may `require` only `electron` and a handful
 * of Node builtins; a relative `require` fails at load, and the failure is
 * silent from the page's point of view — the window renders and simply never
 * receives its port. `window-policy.test.ts` reads the preload source and pins
 * both literals so a rename here cannot quietly break it.
 */
export const CONTROL_PANEL_PORT_CHANNEL = 'mlx:control-panel-port';

/**
 * Renderer → MAIN: "I am listening; send me a port."
 *
 * The direction matters. A port can be transferred exactly once, so posting it
 * on `did-finish-load` and hoping the page's listener is already installed is a
 * race with no recovery — a missed port cannot be re-sent, only replaced with a
 * new channel. Letting the renderer ask makes reloads and slow starts ordinary,
 * and lets MAIN answer with a fresh `MessageChannelMain` every time.
 */
export const CONTROL_PANEL_READY_CHANNEL = 'mlx:control-panel-ready';

/**
 * Renderer → MAIN: the CONTROL PANEL transport failed to acknowledge
 * cancellation within its hard bound.
 *
 * A fresh channel to the same process cannot recover a wedged event loop, so
 * this signal enters the broker's supervised kill/restart path instead.
 */
export const CONTROL_PANEL_UNRESPONSIVE_CHANNEL = 'mlx:control-panel-unresponsive';

/**
 * Renderer → MAIN: "let the user choose a Work folder", with a request id. MAIN answers on
 * {@link CONTROL_PANEL_FOLDER_CHANNEL}. Duplicated as a literal in the preload, pinned by `window-policy.test.ts`.
 */
export const CONTROL_PANEL_PICK_FOLDER_CHANNEL = 'mlx:control-panel-pick-folder';

/** MAIN → renderer: the request id and the folder the user chose in the native dialog, or `null`. */
export const CONTROL_PANEL_FOLDER_CHANNEL = 'mlx:control-panel-folder';

/** A request id from the page: a string of 1–64 characters, echoed back and nothing else. */
export function isFolderRequestId(value: unknown): value is string {
  return typeof value === 'string' && value.length >= 1 && value.length <= 64;
}

/**
 * The folder dialog. Every option is fixed here: the page sends only a request id, so it can never choose a path,
 * a title, a default folder or dialog flags. The answer is the path the user picked, or `null`.
 */
export const FOLDER_DIALOG_OPTIONS = {
  title: 'Choose a folder for Work',
  buttonLabel: 'Choose',
  properties: ['openDirectory', 'createDirectory'],
} as const;

/** What the folder handler needs from Electron: the sender check and the dialog. */
export interface FolderPickerDeps<Sender, Window> {
  /** The window allowed to ask, or null while there is none. */
  window(): { win: Window; contents: Sender } | null;
  showOpenDialog(win: Window, options: typeof FOLDER_DIALOG_OPTIONS): Promise<{ canceled: boolean; filePaths: string[] }>;
}

/**
 * The `ipcMain` handler for {@link CONTROL_PANEL_PICK_FOLDER_CHANNEL}: only our window may ask, one dialog at a time
 * (a second request while one is open is answered `null`), and the reply goes back to the asking sender only.
 */
export function createFolderPicker<Sender extends { send(channel: string, ...args: unknown[]): void }, Window>(
  deps: FolderPickerDeps<Sender, Window>,
): (event: { sender: Sender }, requestId: unknown) => void {
  let open = false;
  return (event, requestId) => {
    const current = deps.window();
    if (current === null || event.sender !== current.contents || !isFolderRequestId(requestId)) return;
    const reply = (path: string | null): void => {
      try {
        event.sender.send(CONTROL_PANEL_FOLDER_CHANNEL, requestId, path);
      } catch {
        // The window went away while the dialog was open.
      }
    };
    if (open) {
      reply(null);
      return;
    }
    open = true;
    deps.showOpenDialog(current.win, FOLDER_DIALOG_OPTIONS).then(
      (result) => {
        open = false;
        const path = result.canceled ? undefined : result.filePaths[0];
        reply(typeof path === 'string' && path !== '' ? path : null);
      },
      () => {
        open = false;
        reply(null);
      },
    );
  };
}

/**
 * `allow` — same-origin SPA navigation.
 * `external` — hand to the user's browser; never navigate the shell to it.
 * `block` — refuse, and do not open it anywhere.
 */
export type NavigationVerdict = 'allow' | 'external' | 'block';

/**
 * Decide what may happen to a URL the renderer asks for, whether by navigation
 * or by `window.open`.
 *
 * Default-deny, and the default is what this is for. Navigating the shell window
 * away from `app://mlx/` leaves the user inside a frameless-ish app window on a
 * page with no back button and no address bar, and if the target is remote it is
 * a full renderer running someone else's script against our preload. Links to
 * documentation, model cards and the local server are common and legitimate, so
 * they are opened in the browser rather than refused.
 */
export function classifyNavigation(target: string): NavigationVerdict {
  let url: URL;
  try {
    url = new URL(target);
  } catch {
    return 'block';
  }
  // NOT `url.origin`. `app:` is only a standard scheme inside Chromium; to
  // Node's URL parser — which is what runs in MAIN and in every test here — it
  // is a non-special scheme, and a non-special scheme's origin is the string
  // `"null"`. Comparing origins would therefore refuse every same-origin
  // navigation while quietly accepting `app://mlx.evil/`, whose origin is also
  // `"null"`. Verified on Node 24.
  if (url.protocol === APP_SCHEME && url.host === APP_HOST) return 'allow';
  // `javascript:`, `data:` and `file:` all land here. `javascript:` in
  // particular would otherwise execute in the window's own context.
  if (url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'mailto:') return 'external';
  return 'block';
}

/** Painted before the SPA's first frame; equals Arc `--background` in each scheme and the boot screen in `ui/index.html`. */
export const WINDOW_BACKGROUND = { light: '#ffffff', dark: '#141414' } as const;

export function windowBackground(dark: boolean): string {
  return dark ? WINDOW_BACKGROUND.dark : WINDOW_BACKGROUND.light;
}

/** Centres the 12 px traffic lights in the SPA's 40 px title bar. */
export const TRAFFIC_LIGHT_POSITION = { x: 14, y: 14 } as const;
