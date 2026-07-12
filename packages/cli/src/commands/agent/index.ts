/**
 * `mlx agent` — boot the pi-based local coding agent on the in-process
 * mlx provider (fully offline).
 *
 * pi owns almost every flag, so this command never `parseArgs`es the
 * full argv: {@link scanAgentArgs} lifts out only what mlx handles
 * (`--models-dir`, help, the blocked `update` positional) and forwards
 * the rest verbatim. Boot discipline (spike-proven, see
 * `packages/agent/src/run-agent.ts`): pi may `process.exit()` inside
 * `runAgent`, print mode owns stdout/stdin, so nothing here runs after
 * the handoff and nothing here reads stdin.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { MlxModelInfo } from '@mlx-node/agent';

export interface AgentArgScan {
  /** Value of `--models-dir` (the flag pair is removed from `passthrough`). */
  modelsDir?: string;
  /** `--models-dir` was present without a value — usage error. */
  modelsDirMissingValue: boolean;
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
 * (`--models-dir`, `-h`/`--help`) are recognized ONLY in an option-NAME
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
    help: helpSeen && !PI_PASSTHROUGH_COMMANDS.has(passthrough[0] ?? ''),
    update: passthrough[0] === 'update',
    piOneShot,
    passthrough,
  };
}

/**
 * Args carrying an EXPLICIT model choice or a concrete session copy: ALL
 * default injection is suppressed and the argv is forwarded verbatim.
 * - `--model` / `--models` — the user already named a model or a scope; pi
 *   resolves it directly.
 * - `--fork` — copies the source session (messages + saved model) into a new
 *   one; injecting anything would fight the restore.
 */
const FULL_SUPPRESS_ARGS: ReadonlySet<string> = new Set(['--model', '--models', '--fork']);

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
 */
const SESSION_PROVIDER_CARRIER_ARGS: ReadonlySet<string> = new Set([
  '--provider',
  '-c',
  '--continue',
  '-r',
  '--resume',
  '--session',
  '--session-id',
]);

/**
 * Keep a launch on-machine at the AUTHORITATIVE CLI-arg layer (above settings
 * and above pi's cloud fallback). Without this, ambient provider credentials
 * (e.g. a `GROQ_API_KEY` in the shell) make pi's "first available model"
 * fallback pick a CLOUD model over the local ones — the opposite of what
 * `mlx agent` promises. Three cases:
 * - explicit model / scope / fork ({@link FULL_SUPPRESS_ARGS}) → forward as-is;
 * - session / provider carrier ({@link SESSION_PROVIDER_CARRIER_ARGS}) →
 *   prepend `--models mlx/*` (a local-only scope that preserves session restore
 *   yet blocks a cloud default for a new / unknown / empty session);
 * - plain fresh run → prepend `--model mlx/<default>`.
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
  const optionNames = new Set<string>();
  for (let i = 0; i < passthrough.length; i++) {
    const token = passthrough[i]!;
    optionNames.add(token);
    if (VALUE_CONSUMING_ARGS.has(token) && i + 1 < passthrough.length) {
      i++; // the next token is this option's value, never an option name
    }
  }

  // Fail-closed default is to inject a LOCAL model; suppress only on a real
  // explicit model/scope/fork, and scope (not bare --model) for a real carrier.
  if (Array.from(FULL_SUPPRESS_ARGS).some((arg) => optionNames.has(arg))) {
    return passthrough;
  }
  if (Array.from(SESSION_PROVIDER_CARRIER_ARGS).some((arg) => optionNames.has(arg))) {
    return ['--models', 'mlx/*', ...passthrough];
  }
  return ['--model', `mlx/${defaultModelId}`, ...passthrough];
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

/** mlx-side help; pi's full flag list is appended by forwarding `--help`. */
function printAgentPreamble(): void {
  console.log(`
mlx agent — local coding agent (pi) running fully offline on MLX

Usage:
  mlx agent [options] [@file ...]

mlx options (handled before pi sees the args):
  --models-dir <dir>        Local models directory (default: ~/.mlx-node/models;
                            also via MLX_MODELS_DIR or ~/.mlx-node/config.json).
                            Dash-leading paths need the --models-dir=<dir> form.

First run: when no local model exists, an interactive wizard offers a curated
download. Agent config home: ~/.mlx-node/agent (override: PI_CODING_AGENT_DIR).

Environment:
  MLX_AGENT_AUTO_APPROVE=1  Auto-approve bash/write/edit tool calls in headless
                            print/json runs — without an attached UI the
                            permission gate blocks them otherwise.

Notes:
  'mlx agent update' is disabled — update @mlx-node/cli via your package
  manager instead. 'install'/'remove'/'list' manage pi extensions, themes and
  skills under the agent config home; 'config' edits which are enabled.

pi options:`);
}

/**
 * Injectable seams for {@link run}'s argv-routing tests. Production
 * leaves them unset and fills each via the deferred dynamic imports;
 * types are `typeof import(...)` lookups (erased at compile time) so the
 * module stays importable without the native addon.
 */
export interface AgentRunDeps {
  resolveModelsDir?: (typeof import('../../config.js'))['resolveModelsDir'];
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

  // Deferred imports: `@mlx-node/agent` loads the native addon and the
  // pure `scanAgentArgs` export above must stay importable without it.
  const resolveModelsDir = deps.resolveModelsDir ?? (await import('../../config.js')).resolveModelsDir;
  const runAgent = deps.runAgent ?? (await import('@mlx-node/agent')).runAgent;

  const modelsDir = resolveModelsDir(scan.modelsDir);

  if (scan.help) {
    printAgentPreamble();
    // pi appends its full flag list and process.exit(0)s on this path.
    await runAgent({ modelsDir, models: [], argv: ['--help'] });
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
    await runAgent({ modelsDir, models: [], argv: scan.passthrough });
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
  const agentArgv = withDefaultModel(scan.passthrough, modelId);
  // A notice only makes sense when injection actually overrode a non-mlx
  // persisted default. An identity return (FULL_SUPPRESS: --model/--models/
  // --fork) overrode nothing, so the notice would be a lie; both the
  // `--models mlx/*` scope and the fresh-run `--model` injection return a new
  // array, so `agentArgv !== scan.passthrough` marks a real override.
  if (notice !== undefined && agentArgv !== scan.passthrough) {
    console.error(notice);
  }

  await runAgent({ modelsDir, models, argv: agentArgv });
}
