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

import { existsSync, readFileSync } from 'node:fs';
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
 * Per-layer snapshot of pi's `shellCommandPrefix`: the value contributed by the
 * global (`<agentDir>/settings.json`) and project (`<cwd>/.pi/settings.json`)
 * settings layers, each a `string` or `undefined` (that layer sets no prefix).
 * Kept per-layer — not pre-merged — so a `/reload` can update each layer with
 * pi's exact per-layer RETENTION semantics before re-merging.
 */
interface ShellPrefixLayers {
  global: string | undefined;
  project: string | undefined;
}

/**
 * Recompute ONE settings layer's `shellCommandPrefix` the way pi's
 * `SettingsManager.reload()` does (`tryLoadFromStorage` → `loadFromStorage` +
 * `withLock`, `dist/core/settings-manager.js`), given the layer's PRIOR value:
 * - `!active` (untrusted project) → `undefined` (pi's `loadFromStorage` returns
 *   `{}` for an untrusted project, no error → the layer is CLEARED).
 * - file ABSENT → `undefined` (`withLock` yields `current=undefined` → `{}`, no
 *   error → CLEARED).
 * - file present + parseable object → its `shellCommandPrefix` if a string, else
 *   `undefined` (REPLACE with the new value).
 * - file present but malformed / unreadable / a non-object (pi's
 *   `migrateSettings` does `"key" in settings`, which throws for a non-object) →
 *   `tryLoadFromStorage` returns an error → reload RETAINS the prior value.
 *
 * The retention branch is what closes the reload under-disclosure: on a FAILED
 * reload pi keeps baking the prior prefix into BashTool, so the gate must keep
 * showing it rather than degrade to empty. At the FIRST snapshot (boot) the
 * prior value is `undefined`, so a `retain` there yields empty — which matches
 * pi baking `{}` for a malformed-at-boot file. Same logic is correct at boot
 * and reload, so no boot/reload branching is needed.
 */
function resolveLayerPrefix(prior: string | undefined, path: string, active: boolean): string | undefined {
  if (!active) {
    return undefined; // untrusted project layer → dropped
  }
  if (!existsSync(path)) {
    return undefined; // absent → pi clears the layer
  }
  try {
    const content = readFileSync(path, 'utf-8');
    if (!content) {
      // Zero-byte file → pi's `loadFromStorage` short-circuits (`if (!content)
      // return {}`, no error) → the layer is CLEARED (not retained). `!content`
      // matches pi's exact truthiness, so whitespace-only (`" "`) is NOT empty:
      // it reaches JSON.parse, throws, and falls to the retain branch below —
      // exactly as pi errors and retains for it.
      return undefined;
    }
    const parsed: unknown = JSON.parse(content);
    if (typeof parsed !== 'object' || parsed === null) {
      // pi's migrateSettings throws on a non-object → reload retains the prior.
      return prior;
    }
    const value = (parsed as Record<string, unknown>)['shellCommandPrefix'];
    return typeof value === 'string' ? value : undefined;
  } catch {
    // Malformed JSON / unreadable (EACCES) → pi retains the prior layer value.
    return prior;
  }
}

/**
 * Recompute both settings layers for one context, applying {@link
 * resolveLayerPrefix} per layer with a DIRECT, lock-free read — deliberately
 * NOT via pi's `SettingsManager`, whose load takes a proper-lockfile lock
 * (creating/removing `<file>.lock`) and, under contention or a non-writable
 * dir, degrades to `{}`. This reader never takes a lock and MIRRORS pi's reload
 * retention, so it can never show empty where pi is still baking a non-empty
 * prefix. `prior` carries the last snapshot so the retain branch is faithful.
 *
 * `getAgentDir`/`CONFIG_DIR_NAME` are imported at call time (deferred), so this
 * module keeps its "no pi runtime import before env seeding" discipline; both
 * are pure (env read lazily inside `getAgentDir`). NEVER throws: a hostile
 * `ctx.isProjectTrusted` getter or a failed deferred import retains `prior`.
 */
async function resolveShellPrefixLayers(prior: ShellPrefixLayers, ctx: ExtensionContext): Promise<ShellPrefixLayers> {
  try {
    const { getAgentDir, CONFIG_DIR_NAME } = await import('@earendil-works/pi-coding-agent');
    const trusted = ctx.isProjectTrusted();
    return {
      global: resolveLayerPrefix(prior.global, join(getAgentDir(), 'settings.json'), true),
      project: resolveLayerPrefix(prior.project, join(ctx.cwd, CONFIG_DIR_NAME, 'settings.json'), trusted),
    };
  } catch {
    return prior;
  }
}

/**
 * Merge the two layers exactly as pi's `deepMergeSettings(global, project)` +
 * `getShellCommandPrefix()` do: a project string overrides the global one,
 * otherwise the global value is used, else `''`.
 */
function effectiveShellPrefix(layers: ShellPrefixLayers): string {
  return layers.project ?? layers.global ?? '';
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
 * Build the `mlx-permission-gate` inline extension. The per-session allow list
 * lives in the `factory` closure, so every extension load (session start or
 * `/reload`) starts with a clean slate. The bash-prefix snapshot, by contrast,
 * lives in THIS outer closure so it PERSISTS across factory reinvocations.
 */
export function createPermissionGateExtension(): InlineExtension {
  // Per-layer snapshot of pi's bash `shellCommandPrefix`, recomputed at each
  // `session_start` — which fires at boot AFTER pi bakes the prefix into
  // BashTool and again on `/reload` AFTER the rebuild, i.e. the SAME lifecycle
  // instant pi bakes it. Kept per-layer (not pre-merged) so each reload applies
  // pi's exact RETENTION rule: a failed layer reload keeps baking the prior
  // value, so we keep showing it. `snapshotted` tells a real empty snapshot
  // apart from "no session_start yet" (which falls back to a one-shot on-demand
  // read). Snapshot-primary is FAITHFUL: it shows exactly what pi baked, even
  // after an edit-without-reload where an on-demand re-read would drift.
  //
  // MUST live here, not in `factory`: pi re-invokes the inline extension factory
  // on every `/reload` (resource-loader `loadExtensionFactories`) BEFORE
  // emitting the reload `session_start`. If this state were reset per factory
  // run, a failed reload (malformed/unreadable file) would `retain` against a
  // freshly-reset `undefined` and drop pi's still-baked prefix — the exact
  // under-disclosure the retain rule exists to prevent.
  let layers: ShellPrefixLayers = { global: undefined, project: undefined };
  let snapshotted = false;

  return {
    name: 'mlx-permission-gate',
    factory: (pi: ExtensionAPI) => {
      const sessionAllowed = new Set<string>();

      pi.on('session_start', async (_event, ctx) => {
        layers = await resolveShellPrefixLayers(layers, ctx);
        snapshotted = true;
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
          // Snapshot is primary (faithful to pi's baked value, incl. reload
          // retention). Only if a bash approval somehow precedes the first
          // session_start do we fall back to a one-shot on-demand read from a
          // clean baseline (lock-free, never throws). A snapshotted empty prefix
          // is authoritative — it is NOT treated as "missing" — so we never
          // re-read over a deliberate empty bake.
          const prefix = snapshotted
            ? effectiveShellPrefix(layers)
            : effectiveShellPrefix(await resolveShellPrefixLayers({ global: undefined, project: undefined }, ctx));
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
