/**
 * `mlx agent` — boot the pi-based local coding agent on the in-process
 * mlx provider (fully offline).
 *
 * pi owns almost every flag, so this command never `parseArgs`es the
 * full argv: {@link scanAgentArgs} lifts out only what mlx handles
 * (`--models-dir`, tracing, help, the blocked `update` positional) and forwards
 * the rest verbatim. Boot discipline (spike-proven, see
 * `packages/agent/src/run-agent.ts`): pi may `process.exit()` inside
 * `runAgent`, print mode owns stdout/stdin, so nothing here runs after
 * the handoff and nothing here reads stdin.
 */

import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { MlxModelInfo } from '@mlx-node/agent';
// Native-free subpath: the help path must print without loading the addon, and
// the family list must have exactly ONE definition (the drift guard's).
import { coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';

export interface AgentArgScan {
  /** Value of `--models-dir` (the flag pair is removed from `passthrough`). */
  modelsDir?: string;
  /** `--models-dir` was present without a value — usage error. */
  modelsDirMissingValue: boolean;
  /** Enable bounded native inference diagnostics (`--trace-dir` implies this). */
  trace: boolean;
  /** Directory selected by `--trace-dir` (the flag pair is removed from `passthrough`). */
  traceDir?: string;
  /** `--trace-dir` was present without a value — usage error. */
  traceDirMissingValue: boolean;
  /**
   * Whether to persist the SSD paged cold tier (on by default; the mlx-owned
   * `--no-persist-cache` flag turns it off). A SINGLE process-wide boolean
   * applied to every family in `COLD_TIER_RESTORE_FAMILIES` — not to qwen3
   * alone. Lifted out of the argv like the other mlx flags — never forwarded
   * to pi.
   */
  persistPagedCache: boolean;
  /**
   * `-h`/`--help` seen and this is NOT a pi pass-through invocation
   * (`install`/`remove`/`uninstall`/`list`/`config` print their own
   * per-command help inside pi, so those pass through untouched).
   */
  help: boolean;
  /** Leading `update` positional — pi's npm self-update, always blocked. */
  update: boolean;
  /**
   * pi one-shot metadata argv: `--version`/`-v` in an option-NAME position,
   * or `--export` that consumed a value. pi answers these BEFORE model
   * resolution — `main()` prints VERSION / exports and `process.exit`s ahead
   * of session creation — so they need no local model: forward verbatim,
   * skipping discovery, the first-run wizard and default-model injection.
   * A bare trailing `--export` does NOT count: pi only consumes a following
   * value (`i + 1 < len`), otherwise it is an unknown flag. Neither does an
   * EMPTY value (`--export ''`): pi stores '' but its truthiness gate
   * (`if (parsed.export)`) skips the export path, so pi would fall into
   * normal session startup — that run must stay on the discovery path.
   */
  piOneShot: boolean;
  /** Args forwarded to pi in their original order. */
  passthrough: string[];
}

/**
 * Leading positionals that route into pi's own command handlers and stay
 * useful. pi recognizes them ONLY at `args[0]`: `parsePackageCommand`
 * matches exactly `install | remove | uninstall | update | list`
 * (`const [rawCommand] = args`) and `handleConfigCommand` matches
 * `config` (`const [command] = args`). So these must reach pi verbatim —
 * `update` (npm self-update) is the one member mlx blocks instead.
 */
const PI_PASSTHROUGH_COMMANDS: ReadonlySet<string> = new Set(['install', 'remove', 'uninstall', 'list', 'config']);

/**
 * Options that UNCONDITIONALLY consume the FOLLOWING token as their value —
 * mirrors pi 0.80.6 `cli/args.js` (`arg === X && i + 1 < len` → `args[++i]`).
 * re-verify on a pi bump (same bounded coupling as the {@link expandPiAgentDir}
 * / {@link readPersistedDefaultModel} shims). Shared by BOTH argv walks — the
 * {@link scanAgentArgs} option-name lift and the {@link withDefaultModel} model
 * scan — so they classify identically and a value token after a pi
 * value-consumer is never re-interpreted: `--system-prompt --models-dir` sets
 * systemPrompt="--models-dir" in pi, so mlx must forward that value verbatim,
 * not strip it as its own flag (the R3-2 / WB-5 sibling leaks). Both walks skip
 * the token after any of these. Deliberately EXCLUDES the conditional consumers
 * `-p`/`--print` and `--list-models` (pi only consumes a NON-dash next token for
 * those, so a `--`-leading sentinel is never swallowed) and the flag-only
 * carriers `-c`/`--continue`/`-r`/`--resume` (no value). The inline
 * `--opt=value` form is not modeled either — pi's exact-match parser does not
 * accept it.
 */
const VALUE_CONSUMING_ARGS: ReadonlySet<string> = new Set([
  '--mode',
  '--provider',
  '--model',
  '--api-key',
  '--system-prompt',
  '--append-system-prompt',
  '--name',
  '-n',
  '--session',
  '--session-id',
  '--fork',
  '--session-dir',
  '--models',
  '--tools',
  '-t',
  '--exclude-tools',
  '-xt',
  '--thinking',
  '--export',
  '--extension',
  '-e',
  '--skill',
  '--prompt-template',
  '--theme',
]);

/**
 * Pure manual scan of `mlx agent`'s argv — see {@link AgentArgScan}.
 *
 * ONE pi-parity, value-aware walk (sibling of {@link withDefaultModel}'s model
 * scan, sharing {@link VALUE_CONSUMING_ARGS}): mlx's own options
 * (`--models-dir`, `--trace`, `--trace-dir`, `-h`/`--help`) are recognized ONLY in an option-NAME
 * position. A token sitting in a pi value-consumer's value slot
 * (`--system-prompt --models-dir` → "--models-dir" is systemPrompt's value) is
 * forwarded verbatim, never hijacked as mlx's flag. Routing (`help`/`update`)
 * reads the value-aware passthrough head, so a stripped `--models-dir` pair
 * cannot mask what pi will see at args[0].
 */
export function scanAgentArgs(argv: string[]): AgentArgScan {
  const passthrough: string[] = [];
  let modelsDir: string | undefined;
  let modelsDirMissingValue = false;
  let trace = false;
  let traceDir: string | undefined;
  let traceDirMissingValue = false;
  let persistPagedCache = true;
  let helpSeen = false;
  let piOneShot = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;

    // pi value-consumer in an option-NAME position: the following token is its
    // VALUE (pi does `args[++i]`), never an option name. Forward BOTH verbatim
    // and skip the value so it is never interpreted as mlx's --models-dir/help.
    // This is checked FIRST, so a `--models-dir` (or `--help`) sitting in a
    // pi-consumer's value slot passes through untouched.
    if (VALUE_CONSUMING_ARGS.has(arg) && i + 1 < argv.length) {
      if (arg === '--export' && argv[i + 1]!.length > 0) {
        piOneShot = true;
      }
      passthrough.push(arg, argv[i + 1]!);
      i++;
      continue;
    }

    // mlx-only options, recognized ONLY here (an option-NAME position).
    if (arg === '--models-dir') {
      const next = argv[i + 1];
      // The SPACE-form value must be a real path token: absent, empty,
      // or option-looking (`-…`) values are usage errors. Consuming an
      // option here would swallow the next flag (`--models-dir --local`
      // must not create ./--local and turn an install global). Dash-
      // leading dirs must use the `--models-dir=<dir>` form.
      if (next === undefined || next.startsWith('-')) {
        modelsDirMissingValue = true;
      } else if (next.length === 0) {
        modelsDirMissingValue = true;
        i++; // the empty token was the (unusable) value — consume it
      } else {
        modelsDir = next;
        i++;
      }
      continue;
    }
    if (arg.startsWith('--models-dir=')) {
      const value = arg.slice('--models-dir='.length);
      if (value.length === 0) {
        modelsDirMissingValue = true;
      } else {
        modelsDir = value;
      }
      continue;
    }
    if (arg === '--trace') {
      trace = true;
      continue;
    }
    if (arg === '--no-persist-cache') {
      persistPagedCache = false;
      continue;
    }
    if (arg === '--trace-dir') {
      trace = true;
      const next = argv[i + 1];
      // Match --models-dir's value rules: never swallow the next option. A
      // dash-leading directory remains available through --trace-dir=<dir>.
      if (next === undefined || next.startsWith('-')) {
        traceDirMissingValue = true;
      } else if (next.length === 0) {
        traceDirMissingValue = true;
        i++;
      } else {
        traceDir = next;
        i++;
      }
      continue;
    }
    if (arg.startsWith('--trace-dir=')) {
      trace = true;
      const value = arg.slice('--trace-dir='.length);
      if (value.length === 0) {
        traceDirMissingValue = true;
      } else {
        traceDir = value;
      }
      continue;
    }
    if (arg === '-h' || arg === '--help') {
      helpSeen = true;
    }
    if (arg === '--version' || arg === '-v') {
      piOneShot = true;
    }
    passthrough.push(arg);
  }

  // Route on what pi will actually see at args[0] — the value-aware passthrough
  // head — so a preceding (stripped) `--models-dir` pair or a consumed value
  // cannot mask a pass-through command or the blocked `update`.
  return {
    modelsDir,
    modelsDirMissingValue,
    trace,
    traceDir,
    traceDirMissingValue,
    persistPagedCache,
    help: helpSeen && !PI_PASSTHROUGH_COMMANDS.has(passthrough[0] ?? ''),
    update: passthrough[0] === 'update',
    piOneShot,
    passthrough,
  };
}

const DEFAULT_AGENT_LOG_FILTER = 'mlx_core::inference=info,mlx_core::decode=info';

export interface AgentTracingSetupOptions {
  /** @internal Hermetic environment seam for tests. */
  env?: NodeJS.ProcessEnv;
  /** @internal Home-directory seam for tests. */
  homeDir?: string;
  /** @internal Clock seam for a deterministic per-run directory. */
  now?: Date;
  /** @internal Process-id seam for a deterministic per-run directory. */
  pid?: number;
  /** @internal Output seam. Production writes to stderr so pi keeps stdout. */
  announce?: (line: string) => void;
}

/**
 * Configure the native Rust `tracing` subscriber before `@mlx-node/agent`
 * imports the addon. N-API initializes the subscriber while loading the addon,
 * so the target filter and file must already be present in the environment.
 *
 * The diagnostics are opt-in through `--trace` or `--trace-dir`. Caller-supplied
 * `MLX_NODE_LOG` and `MLX_NODE_LOG_FILE` values always win. Otherwise each
 * process gets a bounded inference/decode filter and a fresh file below
 * `~/.mlx-node/logs/agent/` (or the explicit trace directory).
 */
export function configureAgentTracing(
  scan: Pick<AgentArgScan, 'trace' | 'traceDir'>,
  options: AgentTracingSetupOptions = {},
): string | undefined {
  const env = options.env ?? process.env;
  const requested = scan.trace || scan.traceDir !== undefined;
  if (!requested) return undefined;

  env.MLX_NODE_LOG ??= DEFAULT_AGENT_LOG_FILTER;

  const explicitFile = env.MLX_NODE_LOG_FILE;
  let traceDir: string | undefined;
  let logFile: string;
  if (explicitFile !== undefined && explicitFile.trim() !== '') {
    // Preserve the user's environment value verbatim; use its trimmed path for
    // pre-creation/announcement because the Rust subscriber applies the same
    // trim before opening it.
    logFile = explicitFile.trim();
  } else {
    traceDir = scan.traceDir
      ? resolve(scan.traceDir)
      : join(
          options.homeDir ?? homedir(),
          '.mlx-node',
          'logs',
          'agent',
          `${(options.now ?? new Date()).toISOString().replace(/[:.]/g, '-')}-pid-${options.pid ?? process.pid}`,
        );
    logFile = join(traceDir, 'inference.log');
    env.MLX_NODE_LOG_FILE = logFile;
  }

  // Diagnostics are never a launch prerequisite. Pre-create private paths when
  // possible; subscriber initialization remains responsible for opening its
  // actual writer after the deferred addon import.
  try {
    const parent = dirname(logFile);
    mkdirSync(parent, { recursive: true, mode: 0o700 });
    if (traceDir !== undefined) chmodSync(traceDir, 0o700);
    const fd = openSync(logFile, 'a', 0o600);
    closeSync(fd);
    chmodSync(logFile, 0o600);
  } catch {
    // Best effort by contract: diagnostics I/O must never prevent startup.
  }

  (options.announce ?? ((line: string) => console.error(line)))(`mlx agent: inference log ${logFile}`);
  return logFile;
}

/**
 * An explicit `--models` scope is authoritative: the user already named the
 * complete picker/cycling scope, so forward it verbatim. A concrete `--model`
 * is different: it chooses the active model but does NOT constrain pi's model
 * selector, so {@link withDefaultModel} still adds the local-only scope around
 * it.
 */
const FULL_SUPPRESS_ARGS: ReadonlySet<string> = new Set(['--models']);

/**
 * Session / provider carriers where a bare `--model mlx/<id>` injection is
 * WRONG but a local-only SCOPE is right: {@link withDefaultModel} prepends
 * `--models mlx/*` for these instead of a concrete `--model`.
 *
 * - session refs (`-c`/`--continue`/`-r`/`--resume`/`--session`/`--session-id`):
 *   a real existing session must restore its OWN saved model. `--models` is a
 *   scope, not a model — pi leaves `options.model` undefined for an existing
 *   session (saved model wins) and only picks `scopedModels[0]` (an mlx model)
 *   for a NEW / unknown / empty session, never a cloud default.
 * - `--provider <p>` (no `--model`): provider-only never selects a model in pi
 *   (`resolveCliModel` needs a `cliModel`), so it would fall through to
 *   `findInitialModel` and an ambient cloud default. `--models mlx/*` forces an
 *   mlx model at the CLI-arg layer, ABOVE settings (global + project) and above
 *   the cloud fallback — immune to a write-once seed failure, a project
 *   `.pi/settings.json` override, or ambient API keys. Injecting a bare
 *   `--model mlx/<id>` here would be WORSE: pi resolves it against `--provider
 *   <p>` via `buildFallbackModel`, minting a bogus CLOUD `<p>/mlx/<id>` custom
 *   model; a `--models` scope carries no provider and cannot. Verified against
 *   pi 0.80.6 `core/model-resolver.js` + `main.js` `buildSessionOptions`.
 * - `--fork`: like the other session carriers, pi restores the copied
 *   session's model because the session already contains messages; the scope
 *   only keeps the model selector/cycling list local and does not replace the
 *   restored model.
 */
const SESSION_PROVIDER_CARRIER_ARGS: ReadonlySet<string> = new Set([
  '--provider',
  '-c',
  '--continue',
  '-r',
  '--resume',
  '--session',
  '--session-id',
  '--fork',
]);

/** Collect real option names with the same unconditional value consumption as pi. */
function collectPiOptionNames(argv: readonly string[]): Set<string> {
  const optionNames = new Set<string>();
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i]!;
    optionNames.add(token);
    if (VALUE_CONSUMING_ARGS.has(token) && i + 1 < argv.length) {
      i++;
    }
  }
  return optionNames;
}

function injectsConcreteDefault(optionNames: ReadonlySet<string>): boolean {
  return (
    !optionNames.has('--models') &&
    !optionNames.has('--model') &&
    !Array.from(SESSION_PROVIDER_CARRIER_ARGS).some((arg) => optionNames.has(arg))
  );
}

/**
 * Keep a launch on-machine at the AUTHORITATIVE CLI-arg layer (above settings
 * and above pi's cloud fallback). Without this, ambient provider credentials
 * (e.g. a `GROQ_API_KEY` in the shell) make pi's "first available model"
 * fallback pick a CLOUD model over the local ones — the opposite of what
 * `mlx agent` promises. Three cases:
 * - explicit scope ({@link FULL_SUPPRESS_ARGS}) → forward as-is;
 * - session / provider / fork carrier ({@link SESSION_PROVIDER_CARRIER_ARGS}) →
 *   prepend `--models mlx/*` (a local-only scope that preserves session restore
 *   yet blocks a cloud default for a new / unknown / empty session);
 * - explicit model → prepend `--models mlx/*` so the picker starts in the
 *   local scope (the registry boundary separately keeps its `all` view local);
 * - plain fresh run → prepend both the local scope and
 *   `--model mlx/<default>`.
 * Pure function, exported for tests.
 */
export function withDefaultModel(passthrough: string[], defaultModelId: string): string[] {
  // Classify over OPTION NAMES, not raw tokens: a value token that merely looks
  // like an option (`--system-prompt --model` → "--model" is systemPrompt's
  // VALUE, not an option) must not drive the decision. Walk left→right and, just
  // like pi's parser (`args[++i]`), skip the token after any unconditional
  // value-consumer so only real option names remain. A sentinel that is itself
  // a consumer both classifies AND skips its own following value in this one
  // pass; a sentinel consumed as some earlier option's value is skipped here and
  // never classifies.
  const optionNames = collectPiOptionNames(passthrough);

  // Fail-closed default is to inject a LOCAL model and LOCAL picker scope.
  // Suppress only when the user supplied an explicit scope. A concrete model
  // still needs the scope: --model selects one model but pi otherwise exposes
  // every authenticated provider in /model.
  if (Array.from(FULL_SUPPRESS_ARGS).some((arg) => optionNames.has(arg))) {
    return passthrough;
  }
  if (optionNames.has('--model')) {
    return ['--models', 'mlx/*', ...passthrough];
  }
  if (Array.from(SESSION_PROVIDER_CARRIER_ARGS).some((arg) => optionNames.has(arg))) {
    return ['--models', 'mlx/*', ...passthrough];
  }
  return ['--models', 'mlx/*', '--model', `mlx/${defaultModelId}`, ...passthrough];
}

/** A `defaultProvider`/`defaultModel` pair persisted by pi's `/model`. */
export interface PersistedPiDefault {
  provider: string;
  modelId: string;
}

/**
 * Expand a `PI_CODING_AGENT_DIR` value exactly like pi 0.80.6 does
 * (`getAgentDir` → `expandTildePath` → `normalizePath` with default
 * options): a lone `~` or a leading `~/` (`~\` on Windows) becomes the
 * home directory, a `file://` URL becomes its path, and everything
 * else — including `~user` — passes through verbatim (no trim). Looser
 * or tighter rules would desync this reader from the settings.json pi
 * actually opens. `home` is a test seam (pi's `homeDir` option).
 */
export function expandPiAgentDir(dir: string, home: string = homedir()): string {
  if (dir === '~') {
    return home;
  }
  if (dir.startsWith('~/') || (process.platform === 'win32' && dir.startsWith('~\\'))) {
    return join(home, dir.slice(2));
  }
  // pi tests /^file:\/\//; startsWith is the identical predicate.
  if (dir.startsWith('file://')) {
    return fileURLToPath(dir);
  }
  return dir;
}

/**
 * Resolve pi's agent config home the way `runAgent`'s env seeding will:
 * an explicit `PI_CODING_AGENT_DIR` wins — run through {@link expandPiAgentDir}
 * to match pi's own tilde/file-URL expansion — else `~/.mlx-node/agent`
 * (the value `runAgent` seeds). An explicitly passed `agentDir` (test seam)
 * is used verbatim. May throw only via `expandPiAgentDir` (e.g. a malformed
 * `file://` URL); callers wrap this in their own try/catch. Shared by the
 * persisted-default reader and writer so both open the SAME settings.json.
 */
function resolvePiAgentDir(agentDir?: string): string {
  const envDir = process.env.PI_CODING_AGENT_DIR;
  return agentDir ?? (envDir ? expandPiAgentDir(envDir) : join(homedir(), '.mlx-node', 'agent'));
}

/**
 * Read pi's persisted `/model` default from the agent config home:
 * `<agentDir>/settings.json`, fields `defaultProvider` + `defaultModel`
 * (pi's `SettingsManager.setDefaultModelAndProvider` writes them to the
 * GLOBAL-scope file, i.e. this one). The dir mirrors what pi itself
 * will resolve after `runAgent`'s env seeding (see {@link resolvePiAgentDir}).
 * Absent, malformed, or unresolvable settings mean "no persisted default",
 * never an error.
 */
export function readPersistedDefaultModel(agentDir?: string): PersistedPiDefault | undefined {
  try {
    const dir = resolvePiAgentDir(agentDir);
    const parsed: unknown = JSON.parse(readFileSync(join(dir, 'settings.json'), 'utf8'));
    if (typeof parsed !== 'object' || parsed === null) {
      return undefined;
    }
    const { defaultProvider, defaultModel } = parsed as Record<string, unknown>;
    if (typeof defaultProvider !== 'string' || defaultProvider.length === 0) {
      return undefined;
    }
    if (typeof defaultModel !== 'string' || defaultModel.length === 0) {
      return undefined;
    }
    return { provider: defaultProvider, modelId: defaultModel };
  } catch {
    return undefined;
  }
}

/**
 * Persist an mlx `/model` default into `<agentDir>/settings.json`
 * (`defaultProvider` + `defaultModel`), PRESERVING every existing field.
 *
 * This is the belt half of the local-first guarantee for pi's session /
 * provider paths, which suppress the `--model` injection: pi's
 * `findInitialModel` consults this persisted default (step 3) ABOVE its
 * "first available cloud model" fallback (step 4) and BELOW a restored
 * session's own model, so seeding an mlx default keeps a `-c`/`--session`/
 * provider-only launch on-machine instead of silently picking a cloud
 * model that an ambient API key made available.
 *
 * Callers gate this to WRITE-ONCE (only when no default is persisted) so a
 * later user `/model` pick is never overwritten. Best-effort by contract:
 * ANY I/O failure is swallowed — a settings write must never crash the
 * launch (the `--model` injection on plain fresh runs remains the fallback).
 * The dir resolves IDENTICALLY to {@link readPersistedDefaultModel}.
 */
export function writePersistedDefaultModel(provider: string, modelId: string, agentDir?: string): void {
  try {
    const dir = resolvePiAgentDir(agentDir);
    const path = join(dir, 'settings.json');

    // Never CLOBBER a present-but-recoverable settings.json: pi treats a
    // malformed/unreadable file as a LOAD ERROR and refuses to overwrite it, so
    // the seed only fires on a genuinely-absent file or a valid settings object.
    // Split the read by failure reason:
    let existing: string;
    try {
      existing = readFileSync(path, 'utf8');
    } catch (readError) {
      // ENOENT = the file genuinely does not exist → seed a fresh, minimal file.
      // Any OTHER read error (EACCES/EISDIR/ENOTDIR/…) → a present file we cannot
      // read: leave it untouched rather than risk clobbering recoverable settings.
      if ((readError as NodeJS.ErrnoException).code === 'ENOENT') {
        mkdirSync(dir, { recursive: true });
        const fresh = { defaultProvider: provider, defaultModel: modelId };
        writeFileSync(path, `${JSON.stringify(fresh, null, 2)}\n`, 'utf8');
      }
      return;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(existing);
    } catch {
      // Present but not valid JSON → recoverable by the user; do NOT overwrite.
      console.error(`mlx agent: ${path} is not valid JSON; leaving it untouched (no default seeded)`);
      return;
    }
    // Valid JSON but not a plain settings object (array / string / number /
    // null) → treat as malformed and SKIP; only a real object is safe to merge.
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return;
    }

    const settings = parsed as Record<string, unknown>;
    settings['defaultProvider'] = provider;
    settings['defaultModel'] = modelId;
    mkdirSync(dir, { recursive: true });
    writeFileSync(path, `${JSON.stringify(settings, null, 2)}\n`, 'utf8');
  } catch {
    // Settings I/O failure must not abort the launch.
  }
}

/**
 * Pick the model id {@link withDefaultModel} injects on a fresh run. pi
 * only consults its persisted default AFTER CLI args, and `mlx agent`
 * always passes `--model` on fresh runs — so the injection itself must
 * honor the user's persisted `/model` pick:
 * - persisted `mlx/<id>` still discovered → inject that id;
 * - persisted `mlx/<id>` no longer discovered → first discovered model;
 * - persisted NON-mlx provider → deliberately overridden — this command
 *   is local-first/offline — but announced via `notice` (stderr), never
 *   silently.
 */
export function chooseDefaultModel(
  models: readonly MlxModelInfo[],
  persisted: PersistedPiDefault | undefined,
): { modelId: string; notice?: string } {
  const fallback = models[0]!.discovered.name;
  if (persisted === undefined) {
    return { modelId: fallback };
  }
  if (persisted.provider === 'mlx') {
    const match = models.find((model) => model.discovered.name === persisted.modelId);
    return { modelId: match ? match.discovered.name : fallback };
  }
  return {
    modelId: fallback,
    notice:
      `mlx agent: persisted default ${persisted.provider}/${persisted.modelId} is not a local mlx model; ` +
      `using mlx/${fallback} (this agent runs offline — pass --model to pick another local model)`,
  };
}

/**
 * mlx-side help text; pi's full flag list is appended by forwarding `--help`.
 * Exported so a test can assert the `--no-persist-cache` copy matches the
 * allowlist it actually applies to — the old wording named qwen3 alone and
 * promised "other families unaffected", which running it on qwen3_5_moe
 * disproved.
 */
export function agentPreambleText(): string {
  return `
mlx agent — local coding agent (pi) running fully offline on MLX

Usage:
  mlx agent [options] [@file ...]

mlx options (handled before pi sees the args):
  --models-dir <dir>        Local models directory (default: ~/.mlx-node/models;
                            also via MLX_MODELS_DIR or ~/.mlx-node/config.json).
                            Dash-leading paths need the --models-dir=<dir> form.
  --trace                   Enable bounded native inference diagnostics.
  --trace-dir <dir>         Write inference.log in this directory (implies
                            --trace; dash-leading paths need --trace-dir=<dir>).
  --no-persist-cache        Disable the on-by-default SSD cold tier for persisted
                            paged prefix blocks. One switch for ALL restore-eligible
                            families (${coldTierRestoreFamilyList().join(', ')});
                            every other family never persists, flag or not.

First run: when no local model exists, an interactive wizard offers a curated
download. Agent config home: ~/.mlx-node/agent (override: PI_CODING_AGENT_DIR).

Environment:
  MLX_AGENT_AUTO_APPROVE=1  Auto-approve bash/write/edit/subagent tool calls in headless
                            print/json runs — without an attached UI the
                            permission gate blocks them otherwise.
  MLX_AGENT_ENABLE_GEMMA_DRAFT=1
                            Use an embedded Gemma4 draft for flat speculative
                            decoding instead of the default paged AR path.
  MLX_NODE_LOG              Override the Rust tracing target filter used by --trace.
  MLX_NODE_LOG_FILE         Override the Rust tracing log file used by --trace.

Notes:
  The built-in subagent tool provides scout/planner/reviewer/worker. Each child
  is an isolated in-memory Pi session inside the parent process. Child sessions
  share one resident model/cache host; up to four tool loops may overlap while
  model inference is serialized. Separate mlx agent processes do not share
  model memory. --no-extensions (or -ne) disables subagents as well as
  discovered extensions.
  'mlx agent update' is disabled — update @mlx-node/cli via your package
  manager instead. 'install'/'remove'/'list' manage pi extensions, themes and
  skills under the agent config home; 'config' edits which are enabled.

pi options:`;
}

/** @internal Print {@link agentPreambleText} ahead of pi's own flag list. */
function printAgentPreamble(): void {
  console.log(agentPreambleText());
}

/**
 * Injectable seams for {@link run}'s argv-routing tests. Production
 * leaves them unset and fills each via the deferred dynamic imports;
 * types are `typeof import(...)` lookups (erased at compile time) so the
 * module stays importable without the native addon.
 */
export interface AgentRunDeps {
  resolveModelsDir?: (typeof import('@mlx-node/server/host/paths'))['resolveModelsDir'];
  discoverMlxModels?: (typeof import('@mlx-node/agent'))['discoverMlxModels'];
  runAgent?: (typeof import('@mlx-node/agent'))['runAgent'];
  /** Whole first-run wizard step (imports + IO wiring included). */
  wizard?: (modelsDir: string) => Promise<void>;
  /** Persisted-`/model` reader; production = {@link readPersistedDefaultModel}. */
  readPersistedDefault?: typeof readPersistedDefaultModel;
  /** Persisted-`/model` writer; production = {@link writePersistedDefaultModel}. */
  writePersistedDefault?: (provider: string, modelId: string) => void;
}

/** Production wizard step: interactive catalog pick + download. */
async function runProductionWizard(modelsDir: string): Promise<void> {
  const { runFirstRunWizard } = await import('./wizard.js');
  const { select } = await import('@inquirer/prompts');
  const { run: downloadModel } = await import('../download-model.js');
  await runFirstRunWizard({
    io: {
      select: (opts) => select(opts),
      isTTY: Boolean(process.stdin.isTTY && process.stdout.isTTY),
      log: (line) => console.log(line),
    },
    download: (downloadArgv) => downloadModel(downloadArgv),
    modelsDir,
  });
}

export async function run(argv: string[], deps: AgentRunDeps = {}): Promise<void> {
  const scan = scanAgentArgs(argv);

  if (scan.update) {
    console.error('mlx agent update is not supported; update @mlx-node/cli via your package manager instead');
    process.exitCode = 1;
    return;
  }

  if (scan.modelsDirMissingValue) {
    console.error('Missing value for --models-dir (a dash-leading path needs the --models-dir=<dir> form)');
    process.exitCode = 1;
    return;
  }

  if (scan.traceDirMissingValue) {
    console.error('Missing value for --trace-dir (a dash-leading path needs the --trace-dir=<dir> form)');
    process.exitCode = 1;
    return;
  }

  // Must run before the deferred `@mlx-node/agent` import below. The native
  // tracing subscriber is installed while the native addon initializes.
  const traceLogFile = configureAgentTracing(scan);

  // Deferred imports: `@mlx-node/agent` loads the native addon and the
  // pure `scanAgentArgs` export above must stay importable without it.
  const resolveModelsDir = deps.resolveModelsDir ?? (await import('@mlx-node/server/host/paths')).resolveModelsDir;
  const runAgent = deps.runAgent ?? (await import('@mlx-node/agent')).runAgent;

  const modelsDir = resolveModelsDir(scan.modelsDir);

  if (scan.help) {
    printAgentPreamble();
    // pi appends its full flag list and process.exit(0)s on this path.
    await runAgent({ modelsDir, models: [], argv: ['--help'], traceLogFile });
    return;
  }

  // Pass-through commands (install/remove/uninstall/list/config) must
  // reach pi with the command still at args[0] — pi's
  // `parsePackageCommand` and `handleConfigCommand` both read ONLY
  // args[0], so a prepended `--model` would knock them into the agent
  // prompt path. pi one-shots (`--version`/`-v`, `--export <file>`) exit
  // inside pi before any model resolution. Neither needs a model: skip
  // discovery, the first-run wizard and default-model injection, and
  // forward verbatim. `--list-models` is deliberately NOT routed here:
  // with zero models pi prints a dead-end "/login" hint (this agent is
  // offline-only, /login cannot produce a local model), while the wizard
  // path downloads a model in a TTY — then actually lists it — or prints
  // the exact `mlx download model` commands headless.
  if (PI_PASSTHROUGH_COMMANDS.has(scan.passthrough[0] ?? '') || scan.piOneShot) {
    await runAgent({ modelsDir, models: [], argv: scan.passthrough, traceLogFile });
    return;
  }

  const discoverMlxModels = deps.discoverMlxModels ?? (await import('@mlx-node/agent')).discoverMlxModels;
  let models = await discoverMlxModels(modelsDir);

  if (models.length === 0) {
    try {
      await (deps.wizard ?? runProductionWizard)(modelsDir);
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
      return;
    }

    models = await discoverMlxModels(modelsDir);
    if (models.length === 0) {
      console.error(`No usable model found in ${modelsDir} after the download.`);
      console.error('Expected a subdirectory with a config.json for a supported family (qwen3/qwen3.5/gemma4/lfm2).');
      console.error('Check the download output above, or point --models-dir at an existing models directory.');
      process.exitCode = 1;
      return;
    }
  }

  const persisted = (deps.readPersistedDefault ?? readPersistedDefaultModel)();
  const { modelId, notice } = chooseDefaultModel(models, persisted);
  // Write-once belt: seed an mlx default into pi's settings.json when none is
  // persisted yet. The AUTHORITATIVE guard for the session / provider carrier
  // paths is the `--models mlx/*` scope injected by withDefaultModel (CLI-arg
  // layer, above settings) — this seed is belt-and-suspenders: it gives pi's
  // `/model` a starting default and keeps `findInitialModel` step-3 on-machine
  // if the scope is ever bypassed. Only when unset, so a later user `/model`
  // pick is never overwritten; the writer swallows I/O failures, so a bad
  // settings file can't crash launch.
  if (persisted === undefined) {
    (deps.writePersistedDefault ?? writePersistedDefaultModel)('mlx', modelId);
  }
  const passthroughOptionNames = collectPiOptionNames(scan.passthrough);
  const agentArgv = withDefaultModel(scan.passthrough, modelId);
  // A notice only makes sense when a concrete default actually overrode a
  // non-mlx persisted default. Adding the local picker scope around an explicit
  // --model does not override that explicit choice and must stay silent.
  if (notice !== undefined && injectsConcreteDefault(passthroughOptionNames)) {
    console.error(notice);
  }

  await runAgent({ modelsDir, models, argv: agentArgv, traceLogFile, persistPagedCache: scan.persistPagedCache });
}
