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

export interface AgentArgScan {
  /** Value of `--models-dir` (the flag pair is removed from `passthrough`). */
  modelsDir?: string;
  /** `--models-dir` was present without a value — usage error. */
  modelsDirMissingValue: boolean;
  /**
   * `-h`/`--help` seen and this is NOT a pi package-manager invocation
   * (`install`/`remove`/`uninstall`/`list` print their own per-command
   * help inside pi, so those pass through untouched).
   */
  help: boolean;
  /** Leading `update` positional — pi's npm self-update, always blocked. */
  update: boolean;
  /** Args forwarded to pi in their original order. */
  passthrough: string[];
}

/** Leading positionals that route into pi's package manager and stay useful. */
const PI_PACKAGE_COMMANDS: ReadonlySet<string> = new Set(['install', 'remove', 'uninstall', 'list']);

/** Pure manual scan of `mlx agent`'s argv — see {@link AgentArgScan}. */
export function scanAgentArgs(argv: string[]): AgentArgScan {
  const passthrough: string[] = [];
  let modelsDir: string | undefined;
  let modelsDirMissingValue = false;
  let helpSeen = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === '--models-dir') {
      if (i + 1 >= argv.length) {
        modelsDirMissingValue = true;
      } else {
        modelsDir = argv[++i]!;
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

  return {
    modelsDir,
    modelsDirMissingValue,
    help: helpSeen && !PI_PACKAGE_COMMANDS.has(argv[0] ?? ''),
    update: argv[0] === 'update',
    passthrough,
  };
}

/**
 * Args that make pi resolve the model itself: an explicit choice
 * (`--model`/`--provider`) or a session reference whose saved model must
 * win (a CLI `--model` overrides session restore in pi).
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

/** mlx-side help; pi's full flag list is appended by forwarding `--help`. */
function printAgentPreamble(): void {
  console.log(`
mlx agent — local coding agent (pi) running fully offline on MLX

Usage:
  mlx agent [options] [@file ...]

mlx options (handled before pi sees the args):
  --models-dir <dir>        Local models directory (default: ~/.mlx-node/models;
                            also via MLX_MODELS_DIR or ~/.mlx-node/config.json)

First run: when no local model exists, an interactive wizard offers a curated
download. Agent config home: ~/.mlx-node/agent (override: PI_CODING_AGENT_DIR).

Environment:
  MLX_AGENT_AUTO_APPROVE=1  Auto-approve bash/write/edit tool calls in headless
                            print/json runs — without an attached UI the
                            permission gate blocks them otherwise.

Notes:
  'mlx agent update' is disabled — update @mlx-node/cli via your package
  manager instead. 'install'/'remove'/'list' manage pi extensions, themes and
  skills under the agent config home.

pi options:`);
}

export async function run(argv: string[]): Promise<void> {
  const scan = scanAgentArgs(argv);

  if (scan.update) {
    console.error('mlx agent update is not supported; update @mlx-node/cli via your package manager instead');
    process.exitCode = 1;
    return;
  }

  if (scan.modelsDirMissingValue) {
    console.error('Missing value for --models-dir');
    process.exitCode = 1;
    return;
  }

  // Deferred imports: `@mlx-node/agent` loads the native addon and the
  // pure `scanAgentArgs` export above must stay importable without it.
  const { resolveModelsDir } = await import('../../config.js');
  const { discoverMlxModels, runAgent } = await import('@mlx-node/agent');

  const modelsDir = resolveModelsDir(scan.modelsDir);

  if (scan.help) {
    printAgentPreamble();
    // pi appends its full flag list and process.exit(0)s on this path.
    await runAgent({ modelsDir, models: [], argv: ['--help'] });
    return;
  }

  let models = await discoverMlxModels(modelsDir);

  if (models.length === 0) {
    const { runFirstRunWizard } = await import('./wizard.js');
    const { select } = await import('@inquirer/prompts');
    const { run: downloadModel } = await import('../download-model.js');
    try {
      await runFirstRunWizard({
        io: {
          select: (opts) => select(opts),
          isTTY: Boolean(process.stdin.isTTY && process.stdout.isTTY),
          log: (line) => console.log(line),
        },
        download: (downloadArgv) => downloadModel(downloadArgv),
        modelsDir,
      });
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

  await runAgent({
    modelsDir,
    models,
    argv: withDefaultModel(scan.passthrough, models[0]!.discovered.name),
  });
}
