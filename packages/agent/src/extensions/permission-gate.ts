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

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import type { ExtensionAPI, ExtensionContext, InlineExtension, ToolCallEvent } from '@earendil-works/pi-coding-agent';

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
 * Read pi's effective bash `shellCommandPrefix` for one (cwd, trusted) pair
 * with a DIRECT, lock-free file read — deliberately NOT via pi's
 * `SettingsManager`, whose load takes a proper-lockfile lock (creating and
 * removing `<file>.lock`, so it is not read-only) and, under lock contention
 * or a non-writable dir, catches the failure in `tryLoadFromStorage` and
 * returns `{}` → an EMPTY prefix. pi bakes the REAL prefix into BashTool once
 * at boot and again on `/reload`; a later locked read that degraded to `{}`
 * would make the gate display an empty prefix while pi executes a non-empty
 * one (a security fail-open / under-disclosure). This reader has strictly
 * fewer failure modes than pi's locked read: it never takes a lock (no
 * contention failure, no `.lock` side effect) and treats every absent,
 * unreadable, or malformed file as `{}` — exactly as pi's own
 * `tryLoadFromStorage` does — so it can never show empty where pi baked
 * non-empty.
 *
 * The merge mirrors pi's `deepMergeSettings(global, project)` +
 * `getShellCommandPrefix()`: the project layer (`<cwd>/.pi/settings.json`) is
 * consulted ONLY when the project is trusted, and a project string overrides
 * the global one; otherwise the global (`<agentDir>/settings.json`) value is
 * used, else `''`. Verified against pi 0.80.6: no settings migration renames
 * `shellCommandPrefix`, so a direct read is complete.
 *
 * `getAgentDir`/`CONFIG_DIR_NAME` are imported at call time (deferred), so
 * this module keeps its "no pi runtime import before env seeding" discipline;
 * both are pure (the env is read lazily inside `getAgentDir`). NEVER throws
 * for a settings-file problem — each read is caught. May reject only if the
 * deferred pi import itself fails (post-boot it never does); callers wrap it.
 */
async function readShellPrefixNoLock(cwd: string, trusted: boolean): Promise<string> {
  const { getAgentDir, CONFIG_DIR_NAME } = await import('@earendil-works/pi-coding-agent');
  const readSettings = (path: string): Record<string, unknown> => {
    try {
      const parsed: unknown = JSON.parse(readFileSync(path, 'utf-8'));
      return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, unknown>) : {};
    } catch {
      // Absent, unreadable, or malformed — pi treats this identically to `{}`.
      return {};
    }
  };
  const pick = (settings: Record<string, unknown>): string | undefined =>
    typeof settings['shellCommandPrefix'] === 'string' ? (settings['shellCommandPrefix'] as string) : undefined;

  const globalSettings = readSettings(join(getAgentDir(), 'settings.json'));
  const projectSettings = trusted ? readSettings(join(cwd, CONFIG_DIR_NAME, 'settings.json')) : {};
  return pick(projectSettings) ?? pick(globalSettings) ?? '';
}

/**
 * Resolve the effective bash prefix for one context without ever throwing.
 * Wraps {@link readShellPrefixNoLock} plus the `ctx` getters (a hostile or
 * missing `ctx.isProjectTrusted` must not crash the gate) → `''` on any
 * failure. Used to snapshot at `session_start`, and only as a fallback for
 * the structurally-rare case that a bash approval precedes the first
 * `session_start`.
 */
async function resolveBashPrefix(ctx: ExtensionContext): Promise<string> {
  try {
    return await readShellPrefixNoLock(ctx.cwd, ctx.isProjectTrusted());
  } catch {
    return '';
  }
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

      // Snapshot of pi's effective bash `shellCommandPrefix`, captured at each
      // `session_start` — which fires at boot AFTER pi bakes the prefix into
      // BashTool and again on `/reload` AFTER the rebuild, i.e. the SAME
      // lifecycle instant pi bakes it. `undefined` means "not yet snapshotted"
      // (kept distinct from a snapshotted empty prefix), so the bash branch can
      // tell a real empty prefix apart from a missing snapshot. Snapshot-primary
      // is FAITHFUL: it shows exactly what pi baked, even after an edit-without-
      // reload where an on-demand re-read would drift from the executed value.
      let bashPrefix: string | undefined;

      pi.on('session_start', async (_event, ctx) => {
        bashPrefix = await resolveBashPrefix(ctx);
      });

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
        // For bash, prepend pi's effective `shellCommandPrefix` so the prompt
        // shows the full program pi will execute, not just the model's arg.
        const command = describeToolCall(toolName, event);
        let detailSource = command;
        if (toolName === 'bash') {
          // Snapshot is primary (faithful to pi's baked value). Only if a bash
          // approval somehow precedes the first session_start do we fall back
          // to a one-shot on-demand read (lock-free, never throws). `??` keeps
          // a snapshotted empty prefix (`''`) authoritative — it is not treated
          // as "missing" — so we never re-read over a deliberate empty bake.
          const prefix = bashPrefix ?? (await resolveBashPrefix(ctx));
          detailSource = prefix ? `${prefix}\n${command}` : command;
        }
        const detail = sanitizeDetail(detailSource);
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
