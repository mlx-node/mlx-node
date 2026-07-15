/**
 * Surface the native inference-log path after Pi owns the terminal.
 *
 * The CLI must configure tracing before importing the native addon, but a
 * message printed at that point is erased when Pi starts its fullscreen TUI.
 * Announcing from `session_start` keeps the early subscriber setup while
 * placing the path in Pi's persistent chat/status area.
 */

import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';

export function createTraceNoticeExtension(logFile: string): InlineExtension {
  return {
    name: 'mlx-trace-notice',
    factory: (pi: ExtensionAPI) => {
      let pendingNotice: ReturnType<typeof setTimeout> | undefined;

      pi.on('session_start', (_event, ctx) => {
        if (pendingNotice !== undefined) clearTimeout(pendingNotice);
        if (ctx.mode !== 'tui') return;

        // Pi renders restored session messages immediately after extension
        // binding. Defer one task so the notice is appended after that history
        // instead of being buried above it in resumed sessions.
        pendingNotice = setTimeout(() => {
          pendingNotice = undefined;
          ctx.ui.notify(`mlx agent: inference log ${logFile}`, 'info');
        }, 0);
      });

      pi.on('session_shutdown', () => {
        if (pendingNotice !== undefined) clearTimeout(pendingNotice);
        pendingNotice = undefined;
      });
    },
  };
}
