/**
 * The View menu's contents, as a pure decision.
 *
 * The reason this is not the `viewMenu` role: that role ALWAYS carries Reload
 * and Toggle Developer Tools, even in a signed production build, where a stray
 * ⌘⌥I opens a debugging surface on the user's own sessions and models. Zoom
 * and fullscreen are standard and harmless, so they ship unconditionally; the
 * developer trio appears only in unpackaged runs or under `MLX_DEVTOOLS=1` —
 * the same gate as the detached-DevTools hook in `window.ts`.
 *
 * Electron-free plain data (cast at the call site) so a Node test can hold the
 * gate: `Menu.buildFromTemplate` cannot resolve outside an Electron process.
 */

export interface ViewMenuEntry {
  role?: string;
  type?: string;
}

export interface ViewMenuTemplate {
  label: string;
  submenu: ViewMenuEntry[];
}

export function viewMenuTemplate(devAccess: boolean): ViewMenuTemplate {
  return {
    label: 'View',
    submenu: [
      ...(devAccess
        ? [{ role: 'reload' }, { role: 'forceReload' }, { role: 'toggleDevTools' }, { type: 'separator' }]
        : []),
      { role: 'resetZoom' },
      { role: 'zoomIn' },
      { role: 'zoomOut' },
      { type: 'separator' },
      { role: 'togglefullscreen' },
    ],
  };
}
