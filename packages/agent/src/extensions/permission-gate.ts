/**
 * `createPermissionGateExtension` — pi has no permission system of its
 * own, so this inline extension is the product's v1 safety layer: every
 * `bash` / `write` / `edit` tool call must be approved before pi
 * executes it.
 *
 * Behavior (settled design):
 * - Interactive (`ctx.hasUI`): prompt via `ctx.ui.select` with the
 *   command (bash) or file path (write/edit) as the detail line.
 *   "Always (this session)" allow-lists the tool name in memory for the
 *   lifetime of this extension instance.
 * - Non-interactive: allow only when `MLX_AGENT_AUTO_APPROVE=1`,
 *   otherwise block with a reason naming the env var. Fail closed.
 *
 * Import discipline (load-bearing, same as the provider extension): pi
 * is import-order sensitive to its config env vars, so this module must
 * not runtime-import `@earendil-works/pi-coding-agent` at module top
 * level — type-only imports appear here, and the event input is
 * narrowed defensively by hand instead of via `isToolCallEventType`.
 */

import type { ExtensionAPI, InlineExtension, ToolCallEvent } from '@earendil-works/pi-coding-agent';

const GATED_TOOLS: ReadonlySet<string> = new Set(['bash', 'write', 'edit']);

const AUTO_APPROVE_ENV = 'MLX_AGENT_AUTO_APPROVE';

/**
 * Derive the human-readable detail line for the approval prompt.
 * Defensive on purpose: a malformed or missing `event.input` must never
 * throw — a handler error would fail closed upstream, but the prompt
 * should still render and let the user decide.
 */
function describeToolCall(toolName: string, event: ToolCallEvent): string {
  const rawInput: unknown = (event as { input?: unknown }).input;
  const input: Record<string, unknown> =
    typeof rawInput === 'object' && rawInput !== null ? (rawInput as Record<string, unknown>) : {};
  if (toolName === 'bash') {
    const command = input['command'];
    return typeof command === 'string' && command.length > 0 ? command : '(unknown command)';
  }
  // write/edit: pi's canonical field is `path`; `file_path` is the
  // compat alias pi's own renderers also accept.
  const path = typeof input['path'] === 'string' ? input['path'] : input['file_path'];
  return typeof path === 'string' && path.length > 0 ? path : '(unknown path)';
}

/**
 * Build the `mlx-permission-gate` inline extension. The per-session
 * allow list lives in the factory closure, so every extension load
 * (session start or `/reload`) starts with a clean slate.
 */
export function createPermissionGateExtension(): InlineExtension {
  return {
    name: 'mlx-permission-gate',
    factory: (pi: ExtensionAPI) => {
      const sessionAllowed = new Set<string>();

      pi.on('tool_call', async (event, ctx) => {
        const toolName: unknown = (event as { toolName?: unknown }).toolName;
        if (typeof toolName !== 'string' || !GATED_TOOLS.has(toolName)) {
          return undefined;
        }
        if (sessionAllowed.has(toolName)) {
          return undefined;
        }

        if (!ctx.hasUI) {
          if (process.env[AUTO_APPROVE_ENV] === '1') {
            return undefined;
          }
          return {
            block: true,
            reason: `Blocked ${toolName}: no interactive UI to approve it (set ${AUTO_APPROVE_ENV}=1 to auto-approve)`,
          };
        }

        const detail = describeToolCall(toolName, event);
        const choice = await ctx.ui.select(`Allow ${toolName}?\n\n  ${detail}`, ['Yes', 'Always (this session)', 'No']);

        if (choice === 'Yes') {
          return undefined;
        }
        if (choice === 'Always (this session)') {
          sessionAllowed.add(toolName);
          return undefined;
        }
        // 'No', a dismissed dialog (undefined), or anything unexpected:
        // fail closed.
        return { block: true, reason: 'Blocked by user' };
      });
    },
  };
}
