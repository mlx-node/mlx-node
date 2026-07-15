/**
 * `createPermissionGateExtension` — pi has no permission system of its
 * own, so this inline extension is the product's v1 safety layer: every
 * `bash` / `write` / `edit` / delegated `subagent` tool call must be approved before pi
 * executes it.
 *
 * Behavior (settled design):
 * - Interactive (`ctx.hasUI`): prompt via `ctx.ui.select` with the
 *   command (bash), file path (write/edit), or delegated task (subagent) as the detail line, passed
 *   through `sanitizeApprovalDetail` (control-byte encoding + length cap).
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

import { sanitizeApprovalDetail } from './approval-detail.js';
import { normalizeSubagentMode } from './subagent.js';

const GATED_TOOLS: ReadonlySet<string> = new Set(['bash', 'write', 'edit', 'subagent']);

const AUTO_APPROVE_ENV = 'MLX_AGENT_AUTO_APPROVE';

/**
 * Per-layer snapshot of pi's `shellCommandPrefix`: the RAW value contributed by
 * the global (`<agentDir>/settings.json`) and project (`<cwd>/.pi/settings.json`)
 * settings layers. pi does NO type validation — `getShellCommandPrefix()` just
 * returns the merged `settings.shellCommandPrefix` verbatim and `bash.js:213`
 * bakes `commandPrefix ? \`${commandPrefix}\n${command}\` : command` — so the
 * value can be ANY JSON type (`123`, `true`, `{…}`, `["a","b"]`, `""`, `null`).
 * Each field is the layer's raw value, or `undefined` when that layer sets no
 * prefix (absent key / cleared / dropped). `undefined` can only mean "layer
 * absent" — a JSON value is never `undefined` — which is exactly what the
 * presence-based merge below keys on. Kept per-layer (not pre-merged) so a
 * `/reload` can update each layer with pi's exact per-layer RETENTION semantics
 * before re-merging.
 */
interface ShellPrefixLayers {
  global: unknown;
  project: unknown;
}

/**
 * Recompute ONE settings layer's `shellCommandPrefix` the way pi's
 * `SettingsManager.reload()` does (`tryLoadFromStorage` → `loadFromStorage` +
 * `withLock`, `dist/core/settings-manager.js`), given the layer's PRIOR value:
 * - `!active` (untrusted project) → `undefined` (pi's `loadFromStorage` returns
 *   `{}` for an untrusted project, no error → the layer is CLEARED).
 * - file ABSENT → `undefined` (`withLock` yields `current=undefined` → `{}`, no
 *   error → CLEARED).
 * - file present + parseable object → its RAW `shellCommandPrefix` value if the
 *   key is present (ANY JSON type — pi does no validation), else `undefined`
 *   (REPLACE with the new value). Presence, not string-ness, decides: pi's
 *   merge keys on key-presence, and pi bakes any truthy value's `String(...)`.
 * - file present but malformed / unreadable / a non-object (pi's
 *   `migrateSettings` does `"key" in settings`, which throws for a non-object) →
 *   `tryLoadFromStorage` returns an error → reload RETAINS the prior value.
 *
 * The retention branch is what closes the reload under-disclosure: on a FAILED
 * reload pi keeps baking the prior prefix into BashTool, so the gate must keep
 * showing it rather than degrade to empty. Whether `prior` is a real prior
 * snapshot (reload) or a blank `undefined` baseline (a fresh lifecycle, which
 * builds a NEW `SettingsManager` with boot semantics — bad file → empty, no
 * retain) is decided by the CALLER (the `session_start` handler, on `reason`);
 * this per-layer rule is identical either way.
 */
function resolveLayerPrefix(prior: unknown, path: string, active: boolean): unknown {
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
    // Presence, NOT string-ness, decides — pi does no type validation and its
    // key-presence merge treats a present non-string / falsy value as "set".
    // Return the RAW value (any JSON type) when the key is present, else
    // `undefined` so the presence-based merge sees this layer as unset.
    return 'shellCommandPrefix' in parsed ? (parsed as Record<string, unknown>)['shellCommandPrefix'] : undefined;
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
 *
 * ACCEPTED RESIDUAL (do NOT try to fix): pi's LOCKED read also retains on lock
 * CONTENTION, which this lock-free reader cannot detect. So a `/reload` that
 * simultaneously (i) contends pi's settings lock and (ii) sees the file
 * concurrently reduced to empty could make pi retain a non-empty prefix while
 * this reader observes empty and under-discloses. That needs a concurrent
 * lock-holder emptying the file at the reload instant — an actor with settings
 * write access who could instead just inject a (disclosed) prefix directly, so
 * it grants no escalation. Left as an extreme-adversarial residual.
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
 * `getShellCommandPrefix()` do, returning the RAW merged value (any JSON type or
 * `undefined`). pi's `deepMergeSettings` overrides on KEY-PRESENCE: it iterates
 * `Object.keys(project)` and, for any `shellCommandPrefix` whose value is not
 * `undefined`, the project value wins — so a present project `""` / `0` /
 * `false` / `null` overrides the global one, and only an ABSENT project key
 * (our `undefined`) falls through to global. A `??` merge would be WRONG here:
 * it would let a present project `null` / `""` / `0` / `false` fall through to
 * global, disagreeing with pi. Coercion to a string (and the `''` empty default)
 * happens once at the call site, mirroring `bash.js:213`.
 */
function mergedPrefixRaw(layers: ShellPrefixLayers): unknown {
  return layers.project !== undefined ? layers.project : layers.global;
}

/**
 * `session_start` reasons that build a FRESH `SettingsManager` (boot semantics:
 * a bad/malformed file → empty, NEVER a retain). pi's `SettingsManager.reload()`
 * — reason `'reload'` — is its ONLY retain-on-error path, so ONLY a reload may
 * inherit the prior snapshot; these fresh lifecycles must restart from a blank
 * baseline or the gate would show a prior session's prefix that pi won't
 * execute (over-disclosure). Any unknown/missing reason falls OUTSIDE this set
 * and therefore retains — over-disclosing a stale prefix is safe; under-
 * disclosing an executed one is not. Verified against pi 0.80.6 `SessionStartEvent`.
 */
const FRESH_SESSION_REASONS: ReadonlySet<string> = new Set(['startup', 'new', 'resume', 'fork']);

/**
 * Derive the human-readable detail line for the approval prompt.
 * Defensive on purpose: a malformed or missing `event.input` must never
 * throw — a handler error would fail closed upstream, but the prompt
 * should still render and let the user decide.
 */
function describeToolCall(toolName: string, event: ToolCallEvent, defaultCwd?: string): string {
  const rawInput: unknown = (event as { input?: unknown }).input;
  const input: Record<string, unknown> =
    typeof rawInput === 'object' && rawInput !== null ? (rawInput as Record<string, unknown>) : {};
  if (toolName === 'bash') {
    const command = input['command'];
    return typeof command === 'string' && command.length > 0 ? command : '(unknown command)';
  }
  if (toolName === 'subagent') {
    const agent = typeof input['agent'] === 'string' ? input['agent'] : undefined;
    const task = typeof input['task'] === 'string' ? input['task'] : undefined;
    const scope = typeof input['agentScope'] === 'string' ? input['agentScope'] : 'user';
    const tasks = Array.isArray(input['tasks']) ? input['tasks'] : [];
    const chainItems = Array.isArray(input['chain']) ? input['chain'] : [];
    const { mode } = normalizeSubagentMode({ agent, task, tasks, chain: chainItems });
    const describeItems = (value: unknown): string[] =>
      Array.isArray(value)
        ? value.map((raw, index) => {
            const item = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
            const itemAgent = typeof item['agent'] === 'string' ? item['agent'] : '(unknown agent)';
            const itemTask = typeof item['task'] === 'string' ? item['task'] : '(unknown task)';
            const resolvedCwd = typeof item['cwd'] === 'string' ? item['cwd'] : defaultCwd;
            const cwd = resolvedCwd ? ` [cwd: ${resolvedCwd}]` : '';
            return `${index + 1}. ${itemAgent}: ${itemTask}${cwd}`;
          })
        : [];
    const items = describeItems(tasks);
    const chain = describeItems(chainItems);
    const singleCwd = typeof input['cwd'] === 'string' ? input['cwd'] : defaultCwd;
    const single = `${agent ?? '(unknown agent)'}: ${task ?? '(unknown task)'}${singleCwd ? ` [cwd: ${singleCwd}]` : ''}`;
    const summary =
      mode === 'chain'
        ? `Chain: ${chain.join(' | ')}`
        : mode === 'parallel'
          ? `Queued tasks: ${items.join(' | ')}`
          : single;
    return [
      'Delegated agent sessions may use bash/write/edit without further prompts.',
      `Execution mode: ${mode}`,
      `Agent scope: ${scope}`,
      summary,
    ].join('\n');
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

      pi.on('session_start', async (event, ctx) => {
        // Only a /reload (pi's sole retain-on-error path) may inherit the prior
        // snapshot. A fresh lifecycle (startup/new/resume/fork) builds a NEW
        // SettingsManager with boot semantics (bad file → empty), so it must
        // start from a blank baseline or we'd retain a prefix pi won't execute.
        // Unknown/missing reason → retain (over-disclose is safe; under-disclose
        // is not); read defensively since a hostile event need not be well-typed.
        const reason: unknown = (event as { reason?: unknown }).reason;
        const fresh = typeof reason === 'string' && FRESH_SESSION_REASONS.has(reason);
        const prior: ShellPrefixLayers = fresh ? { global: undefined, project: undefined } : layers;
        layers = await resolveShellPrefixLayers(prior, ctx);
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
        const command = describeToolCall(toolName, event, ctx.cwd);
        let detailSource = command;
        if (toolName === 'bash') {
          // Snapshot is primary (faithful to pi's baked value, incl. reload
          // retention). Only if a bash approval somehow precedes the first
          // session_start do we fall back to a one-shot on-demand read from a
          // clean baseline (lock-free, never throws). A snapshotted empty prefix
          // is authoritative — it is NOT treated as "missing" — so we never
          // re-read over a deliberate empty bake.
          //
          // Coerce ONCE here, byte-identical to pi's `bash.js:213`
          // (`commandPrefix ? \`${commandPrefix}\n${command}\` : command`): the
          // `raw ?` truthiness gate mirrors pi (falsy `0`/`""`/`false`/`null`/
          // absent → bare command), and `String(raw)` mirrors the template
          // coercion pi applies to any truthy value (`123`→`123`, `true`→`true`,
          // `{…}`→`[object Object]`, `["a","b"]`→`a,b`). Using `String(raw)`
          // rather than a bare `${raw}` keeps this well-typed on `unknown`.
          const raw = snapshotted
            ? mergedPrefixRaw(layers)
            : mergedPrefixRaw(await resolveShellPrefixLayers({ global: undefined, project: undefined }, ctx));
          // The `[object Object]` / `a,b` default stringification is DELIBERATE
          // here — it is exactly what pi's `${commandPrefix}` template bakes for a
          // non-string prefix, and disclosing pi's actual bytes is the whole point.
          // eslint-disable-next-line @typescript-eslint/no-base-to-string
          detailSource = raw ? `${String(raw)}\n${command}` : command;
        }
        const detail = sanitizeApprovalDetail(detailSource);
        const title = toolName === 'subagent' ? 'Allow delegated subagent tool access?' : `Allow ${toolName}?`;
        // Bind the dialog to the active agent operation. Pi's selector only
        // resolves on Ctrl+C/abort when the extension forwards this signal;
        // otherwise the UI can disappear while this awaited tool_call hook
        // remains pending and keeps the whole tool batch suspended.
        const choice = await ctx.ui.select(`${title}\n\n  ${detail}`, ['Yes', 'Always (this session)', 'No'], {
          signal: ctx.signal,
        });

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
