/**
 * Keep pi's terminal title branded for the mlx CLI.
 *
 * Pi refreshes its own title after `session_start`, so startup/rebind updates
 * must run on the next task. Session-name changes are emitted after pi updates
 * its title and can be handled immediately.
 */

import { basename } from 'node:path';

import type { ExtensionAPI, ExtensionContext, InlineExtension } from '@earendil-works/pi-coding-agent';

function buildTerminalTitle(pi: ExtensionAPI, ctx: ExtensionContext): string {
  const cwd = basename(ctx.cwd);
  const context = pi.getSessionName() ?? ctx.model?.id;
  return context ? `mlx - ${context} - ${cwd}` : `mlx - ${cwd}`;
}

export function createTerminalTitleExtension(): InlineExtension {
  return {
    name: 'mlx-terminal-title',
    factory: (pi: ExtensionAPI) => {
      let pendingUpdate: ReturnType<typeof setTimeout> | undefined;

      const updateTitle = (ctx: ExtensionContext): void => {
        if (ctx.mode === 'tui') {
          ctx.ui.setTitle(buildTerminalTitle(pi, ctx));
        }
      };

      pi.on('session_start', (_event, ctx) => {
        if (pendingUpdate !== undefined) clearTimeout(pendingUpdate);
        pendingUpdate = setTimeout(() => {
          pendingUpdate = undefined;
          updateTitle(ctx);
        }, 0);
      });

      pi.on('session_info_changed', (_event, ctx) => {
        updateTitle(ctx);
      });

      pi.on('model_select', (_event, ctx) => {
        updateTitle(ctx);
      });

      pi.on('session_shutdown', () => {
        if (pendingUpdate !== undefined) clearTimeout(pendingUpdate);
        pendingUpdate = undefined;
      });
    },
  };
}
