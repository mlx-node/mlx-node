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

import { readFileSync } from 'node:fs';
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

/** Pure manual scan of `mlx agent`'s argv — see {@link AgentArgScan}. */
export function scanAgentArgs(argv: string[]): AgentArgScan {
  const passthrough: string[] = [];
  let modelsDir: string | undefined;
  let modelsDirMissingValue = false;
  let helpSeen = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
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
    passthrough.push(arg);
  }

  // Route on what pi will actually see at args[0] — the passthrough head —
  // so a preceding (stripped) `--models-dir` pair cannot mask a pass-through
  // command or the blocked `update`.
  return {
    modelsDir,
    modelsDirMissingValue,
    help: helpSeen && !PI_PASSTHROUGH_COMMANDS.has(passthrough[0] ?? ''),
    update: passthrough[0] === 'update',
    passthrough,
  };
}

/**
 * Args that make pi resolve the model itself: an explicit choice
 * (`--model`/`--provider`) or a session reference whose saved model must
 * win (pi's `createAgentSession` restores the session's model ONLY when
 * no CLI `--model` was given). `--fork` belongs here too: it copies the
 * source session — messages and saved model included — into a new one.
 */
const MODEL_CARRYING_ARGS: ReadonlySet<string> = new Set([
  '--model',
  '--models',
  '--provider',
  '-c',
  '--continue',
  '-r',
  '--resume',
  '--session',
  '--session-id',
  '--fork',
]);

/**
 * Default fresh runs to the first discovered local model. Without this,
 * ambient provider credentials (e.g. a `GROQ_API_KEY` in the shell) make
 * pi's "first available model" fallback pick a CLOUD model over the
 * local ones — the opposite of what `mlx agent` promises. Pure function,
 * exported for tests.
 */
export function withDefaultModel(passthrough: string[], defaultModelId: string): string[] {
  if (passthrough.some((arg) => MODEL_CARRYING_ARGS.has(arg))) {
    return passthrough;
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
 * Read pi's persisted `/model` default from the agent config home:
 * `<agentDir>/settings.json`, fields `defaultProvider` + `defaultModel`
 * (pi's `SettingsManager.setDefaultModelAndProvider` writes them to the
 * GLOBAL-scope file, i.e. this one). The dir mirrors what pi itself
 * will resolve after `runAgent`'s env seeding: an explicit
 * `PI_CODING_AGENT_DIR` wins — run through {@link expandPiAgentDir},
 * matching pi's own tilde/file-URL expansion — else `~/.mlx-node/agent`
 * (the value `runAgent` seeds). An explicitly passed `agentDir` (test
 * seam) is used verbatim. Absent, malformed, or unresolvable settings
 * mean "no persisted default", never an error.
 */
export function readPersistedDefaultModel(agentDir?: string): PersistedPiDefault | undefined {
  try {
    const envDir = process.env.PI_CODING_AGENT_DIR;
    const dir = agentDir ?? (envDir ? expandPiAgentDir(envDir) : join(homedir(), '.mlx-node', 'agent'));
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
  // prompt path. They need no model either: skip discovery, the
  // first-run wizard and default-model injection, and forward verbatim.
  if (PI_PASSTHROUGH_COMMANDS.has(scan.passthrough[0] ?? '')) {
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

  const { modelId, notice } = chooseDefaultModel(models, (deps.readPersistedDefault ?? readPersistedDefaultModel)());
  const agentArgv = withDefaultModel(scan.passthrough, modelId);
  // The identity return means no injection happened (session/--model run)
  // — then nothing was overridden and the notice would be a lie.
  if (notice !== undefined && agentArgv !== scan.passthrough) {
    console.error(notice);
  }

  await runAgent({ modelsDir, models, argv: agentArgv });
}
