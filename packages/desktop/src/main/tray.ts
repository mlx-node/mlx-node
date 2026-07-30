/**
 * The menubar item: an icon, a status line, and the four things a user can do to
 * the sidecar.
 *
 * No decisions here — `tray-view.ts` turns a supervisor snapshot into what
 * should be on screen, and this file puts it there. `import { Tray } from
 * 'electron'` cannot resolve outside an Electron process, so anything written in
 * this file is unprovable.
 */

import { Menu, nativeImage, Tray, type MenuItemConstructorOptions } from 'electron';

import type { SupervisorSnapshot } from './supervisor/types.js';
import { presentTray, type TrayPresentation } from './tray-view.js';

export interface TrayActions {
  openControlPanel(): void;
  startInference(): void;
  stopInference(): void;
  restartInference(): void;
  /** Put a ready-to-paste Claude Code command carrying the URL and token on the clipboard. */
  copyClaudeConnectCommand(): void;
  /** Put a ready-to-paste Codex command carrying the URL and token on the clipboard. */
  copyCodexConnectCommand(): void;
  setShowInDock(next: boolean): void;
  quit(): void;
}

export interface TrayOptions {
  /**
   * Path to `iconTemplate.png`. The `Template` suffix is what makes macOS treat
   * it as a mask — black + alpha, recoloured for light/dark and inverted while
   * the menu is open — and `iconTemplate@2x.png` beside it is picked up by name
   * for Retina. {@link nativeImage.setTemplateImage} is set explicitly anyway,
   * because the filename convention is silently lost if the file is ever renamed
   * during packaging.
   */
  iconPath: string;
  showInDock(): boolean;
  actions: TrayActions;
}

export interface TrayController {
  update(snapshot: SupervisorSnapshot): void;
  dispose(): void;
}

export function createTray(options: TrayOptions): TrayController {
  const icon = nativeImage.createFromPath(options.iconPath);
  if (icon.isEmpty()) {
    // A tray built from an empty image is a menubar item with no icon and no
    // hit area: the app is running and there is nothing to click. Worth a loud
    // line, because nothing else in the app will look wrong.
    console.error(`[mlx] tray icon missing or unreadable: ${options.iconPath}`);
  }
  icon.setTemplateImage(true);

  // Held in the closure deliberately. A `Tray` that becomes unreachable is
  // garbage-collected and vanishes from the menubar — the single most common way
  // an Electron menubar app "randomly" loses its icon.
  const tray = new Tray(icon);
  tray.setIgnoreDoubleClickEvents(true);

  let rendered: string | null = null;

  function render(presentation: TrayPresentation): void {
    tray.setTitle(presentation.title);
    tray.setToolTip(presentation.tooltip);

    // Every `click` is wrapped rather than passed by reference: a method handed
    // straight to Electron is called with the menu item as `this`, and the
    // wrapper is also what keeps `actions` free to be reassigned later.
    const items: MenuItemConstructorOptions[] = [
      { label: presentation.statusLabel, enabled: false },
      ...(presentation.detail === null ? [] : [{ label: presentation.detail, enabled: false }]),
      { type: 'separator' },
      {
        label: 'Open Control Panel…',
        enabled: presentation.canOpenControlPanel,
        click: () => {
          options.actions.openControlPanel();
        },
      },
      // The URL alone is not a capability — every inference route is gated —
      // so both commands carry the token too. Client-specific rows matter:
      // Claude Code reads Anthropic environment variables while Codex requires
      // a custom Responses API provider.
      {
        label: 'Copy Claude Code Command',
        enabled: presentation.canCopyConnect,
        click: () => {
          options.actions.copyClaudeConnectCommand();
        },
      },
      {
        label: 'Copy Codex Command',
        enabled: presentation.canCopyConnect,
        click: () => {
          options.actions.copyCodexConnectCommand();
        },
      },
      { type: 'separator' },
      {
        label: 'Start Inference',
        enabled: presentation.canStart,
        click: () => {
          options.actions.startInference();
        },
      },
      {
        label: 'Stop Inference',
        enabled: presentation.canStop,
        click: () => {
          options.actions.stopInference();
        },
      },
      {
        label: 'Restart Inference',
        enabled: presentation.canRestart,
        click: () => {
          options.actions.restartInference();
        },
      },
      { type: 'separator' },
      {
        label: 'Show in Dock',
        type: 'checkbox',
        checked: options.showInDock(),
        click: (item) => {
          options.actions.setShowInDock(item.checked);
        },
      },
      { type: 'separator' },
      // The accelerator is display-only on a tray menu, which is why an
      // application menu with the standard roles is also installed — see
      // `index.ts`.
      {
        label: 'Quit mlx-node',
        accelerator: 'Command+Q',
        click: () => {
          options.actions.quit();
        },
      },
    ];

    tray.setContextMenu(Menu.buildFromTemplate(items));
  }

  return {
    update(snapshot: SupervisorSnapshot): void {
      if (tray.isDestroyed()) return;
      const presentation = presentTray(snapshot);
      // The supervisor emits a state event for every health poll and every
      // trace line. Rebuilding the menu closes it if it happens to be open, so
      // an unchanged presentation is not re-rendered.
      const key = `${JSON.stringify(presentation)}|${String(options.showInDock())}`;
      if (key === rendered) return;
      rendered = key;
      render(presentation);
    },
    dispose(): void {
      if (!tray.isDestroyed()) tray.destroy();
    },
  };
}
