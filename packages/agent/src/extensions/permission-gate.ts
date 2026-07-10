/**
 * `createPermissionGateExtension` — pi has no permission system of its
 * own, so this inline extension is the product's v1 safety layer: every
 * `bash` / `write` / `edit` tool call must be approved before pi
 * executes it.
 *
 * Behavior (settled design):
 * - Interactive (`ctx.hasUI`): prompt via `ctx.ui.select` with the
 *   command (bash) or file path (write/edit) as the detail line, passed
 *   through `sanitizeDetail` (escape/control stripping + length cap).
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

/** Longest detail line (in chars) shown in the approval prompt. */
const DETAIL_MAX_CHARS = 500;
/** Most detail lines shown before truncation kicks in. */
const DETAIL_MAX_LINES = 6;
const TRUNCATION_MARKER = '… [truncated]';

/**
 * ANSI/VT escape sequences, stripped wholesale: CSI (`ESC [` or the bare
 * C1 CSI byte U+009B) with parameter/intermediate bytes and a final byte,
 * OSC (`ESC ]` … terminated by BEL or ST), and remaining two-byte
 * `ESC <Fe>` sequences. Anything malformed that this misses degrades to a
 * lone control character and is caught by CONTROL_CHAR_RE below.
 */
const ANSI_ESCAPE_RE =
  // eslint-disable-next-line no-control-regex
  /(?:\u001b\[|\u009b)[0-9;?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)?|\u001b[@-Z\\-_]?/g;

/**
 * C0 controls (except `\t` and `\n`), DEL, and C1 controls. These are
 * collapsed to a visible U+FFFD placeholder rather than removed, so the
 * user can see that the tool call contained something unprintable.
 */
// eslint-disable-next-line no-control-regex
const CONTROL_CHAR_RE = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g;

/**
 * Sanitize model-controlled text before it is embedded in the approval
 * prompt. The prompt is this product's only permission UI, so it must not
 * depend on pi's TUI stripping terminal escapes (it does not — pi-tui's
 * `wrapTextWithAnsi` deliberately preserves ANSI): a crafted bash command
 * could otherwise move the cursor, erase lines, or restyle the prompt to
 * disguise what is being approved.
 *
 * - ANSI escape sequences are stripped entirely.
 * - Other control characters (C0 except `\n`/`\t`, DEL, C1) become `�`.
 * - Output is capped at DETAIL_MAX_LINES lines and DETAIL_MAX_CHARS chars
 *   with a visible truncation marker, so a huge command cannot flood the
 *   prompt off the screen.
 */
function sanitizeDetail(text: string): string {
  let out = text.replace(ANSI_ESCAPE_RE, '').replace(CONTROL_CHAR_RE, '�');
  let truncated = false;

  const lines = out.split('\n');
  if (lines.length > DETAIL_MAX_LINES) {
    out = lines.slice(0, DETAIL_MAX_LINES).join('\n');
    truncated = true;
  }
  if (out.length > DETAIL_MAX_CHARS) {
    out = out.slice(0, DETAIL_MAX_CHARS);
    // Do not leave a lone high surrogate behind after the hard cut.
    const last = out.charCodeAt(out.length - 1);
    if (last >= 0xd800 && last <= 0xdbff) {
      out = out.slice(0, -1);
    }
    truncated = true;
  }
  if (truncated) {
    out += ` ${TRUNCATION_MARKER}`;
  }
  if (out.trim().length === 0 && text.length > 0) {
    // The input was entirely escape sequences / control characters; show
    // something rather than an approvable-looking blank line.
    return '(unprintable content)';
  }
  return out;
}

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

        // Defense in depth: the detail is model-controlled text and this
        // title is rendered by a third-party TUI that passes ANSI through.
        const detail = sanitizeDetail(describeToolCall(toolName, event));
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
