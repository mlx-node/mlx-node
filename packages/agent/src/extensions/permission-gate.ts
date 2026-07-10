/**
 * `createPermissionGateExtension` — pi has no permission system of its
 * own, so this inline extension is the product's v1 safety layer: every
 * `bash` / `write` / `edit` tool call must be approved before pi
 * executes it.
 *
 * Behavior (settled design):
 * - Interactive (`ctx.hasUI`): prompt via `ctx.ui.select` with the
 *   command (bash) or file path (write/edit) as the detail line, passed
 *   through `sanitizeDetail` (control-byte encoding + length cap).
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
 * Every character that must be rendered visibly instead of reaching the
 * terminal: C0 controls except `\n` and `\t`, DEL, and the C1 range
 * U+0080–U+009F (which contains the raw CSI/OSC/ST bytes U+009B, U+009D
 * and U+009C). Matched one character at a time — deliberately NOT as
 * multi-character escape "sequences": CSI parameters/finals and OSC
 * payloads are ordinary printable bytes that bash still parses and
 * executes, so any sequence-level deletion makes real shell syntax
 * invisible while it still runs (and CSI/OSC/ST termination is ambiguous
 * to parse in the first place — e.g. an unterminated OSC has no defined
 * end).
 */
// eslint-disable-next-line no-control-regex
const CONTROL_CHAR_RE = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g;

/** Render one control character as visible `\xNN` text (e.g. ESC → `\x1b`). */
function encodeControlChar(ch: string): string {
  return `\\x${ch.charCodeAt(0).toString(16).padStart(2, '0')}`;
}

/**
 * Sanitize model-controlled text before it is embedded in the approval
 * prompt. The prompt is this product's only permission UI, so it must not
 * depend on pi's TUI stripping terminal escapes (it does not — pi-tui's
 * `wrapTextWithAnsi` deliberately preserves ANSI): a crafted bash command
 * could otherwise move the cursor, erase lines, or restyle the prompt to
 * disguise what is being approved.
 *
 * Encode, never delete. An earlier deletion-based version stripped whole
 * "escape sequences", but the bytes inside a CSI/OSC-shaped span are
 * still shell syntax that bash executes — a command could display as a
 * safe-looking prefix while a pipe or a second command hid inside what
 * the sanitizer parsed as an escape payload. Instead:
 *
 * - Every printable character is preserved verbatim — nothing bash could
 *   interpret (letters, digits, shell metacharacters, spaces) is removed
 *   or altered.
 * - Every control character (C0 except `\n`/`\t`, DEL, C1) is rendered
 *   as visible `\xNN` text, so no byte that could drive the terminal
 *   survives, and no shell text can hide behind one.
 * - Output is capped at DETAIL_MAX_LINES lines and DETAIL_MAX_CHARS
 *   chars (counted after encoding) with a visible truncation marker, so
 *   a huge command cannot flood the prompt off the screen.
 */
function sanitizeDetail(text: string): string {
  let out = text.replace(CONTROL_CHAR_RE, encodeControlChar);
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
    // Control characters always encode to visible text, so this only
    // fires for whitespace-only input; show something rather than an
    // approvable-looking blank line.
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
