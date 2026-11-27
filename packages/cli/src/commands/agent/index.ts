/**
 * `mlx agent` — boot the Grok Build coding agent over the local inference
 * host (fully offline).
 *
 * The agent binary (`mlx-agent`, built from the `mlx` branch of
 * github.com/mlx-node/grok-build) is a standalone Rust process that speaks
 * HTTP to `mlx serve`. This command's whole job:
 *
 *   1. lift mlx-owned flags out of argv (`--models-dir`, tracing, help,
 *      the mlx-owned `update` positional),
 *   2. make sure at least one local model exists (first-run wizard),
 *   3. spawn `mlx serve --port 0 --auth-token <tok>` + the agent binary,
 *      pointed at the real port via `GROK_*_BASE_URL`/`XAI_API_KEY`
 *      (see ./grok-build.ts for the env contract).
 *
 * The binary owns every other flag, so this command never `parseArgs`es the
 * full argv: {@link scanAgentArgs} lifts out only what mlx handles and
 * forwards the rest verbatim.
 */

import { totalmem } from 'node:os';

// Native-free subpaths: the help path must print without loading the addon, and
// the family list must have exactly ONE definition (the drift guard's).
import { type CatalogGpuInfo, CHAT_FAMILY_IDS } from '@mlx-node/agent/catalog';
import { DECODE_SHARE_ERROR } from '@mlx-node/server/host/decode-share';
import { IDLE_RELEASE_ERROR } from '@mlx-node/server/host/idle-release';
import { MAX_CONTEXT_ERROR } from '@mlx-node/server/host/max-context';
import { MAX_MEMORY_ERROR } from '@mlx-node/server/host/max-memory';

import {
  agentModelArg,
  agentOptionConsumesNext,
  agentPreambleText,
  chooseDefaultModel,
  configureAgentTracing,
  expandAgentDir,
  readPersistedDefaultModel,
  scanAgentArgs,
  withDefaultModel,
  withWebSearchDisabled,
  type AgentArgScan,
  type AgentTracingSetupOptions,
  type PersistedGrokDefault,
} from './args.js';
import {
  agentUpdateStatus,
  applyAgentUpdate,
  launchGrokAgent,
  type AgentUpdateDeps,
  type AgentUpdateStatus,
  type GrokAgentLaunchOptions,
} from './grok-build.js';

export {
  agentModelArg,
  agentOptionConsumesNext,
  agentPreambleText,
  chooseDefaultModel,
  configureAgentTracing,
  expandAgentDir,
  readPersistedDefaultModel,
  scanAgentArgs,
  withDefaultModel,
  withWebSearchDisabled,
  type AgentArgScan,
  type AgentTracingSetupOptions,
  type PersistedGrokDefault,
};

/** @internal Print {@link agentPreambleText} ahead of the binary's flag list. */
function printAgentPreamble(): void {
  console.log(agentPreambleText());
}

/**
 * Injectable seams for {@link run}'s argv-routing tests. Production leaves
 * them unset and fills each via the deferred imports; types are `typeof
 * import(...)` lookups (erased at compile time) so the module stays importable
 * without the native addon.
 */
export interface AgentRunDeps {
  resolveModelsDir?: (typeof import('@mlx-node/server/host/paths'))['resolveModelsDir'];
  /** Discover local chat models (native-free path via @mlx-node/lm). */
  discoverModels?: (dir: string) => Promise<{ name: string }[]>;
  /** Spawn the inference host + agent binary. */
  launch?: (opts: GrokAgentLaunchOptions) => Promise<number>;
  /** Whole first-run wizard step (imports + IO wiring included). */
  wizard?: (modelsDir: string) => Promise<void>;
  /** Persisted-default reader; production = {@link readPersistedDefaultModel}. */
  readPersistedDefault?: typeof readPersistedDefaultModel;
  /** `mlx agent update --check`: report-only probe (no download). */
  updateStatus?: (deps: AgentUpdateDeps) => Promise<AgentUpdateStatus>;
  /** `mlx agent update`: download + install into the managed cache. */
  applyUpdate?: (deps: AgentUpdateDeps) => Promise<{ status: AgentUpdateStatus; downloaded: boolean }>;
  /** Update-flow seams forwarded to both update functions (test-only). */
  updateDeps?: AgentUpdateDeps;
}

/** Probe host GPU capabilities lazily via `@mlx-node/lm` on non-darwin hosts (fails closed to `undefined`). */
async function probeRuntimeGpu(): Promise<CatalogGpuInfo | undefined> {
  if (process.platform === 'darwin') return undefined;
  try {
    const { getRuntimeCapabilities } = await import('@mlx-node/lm');
    const caps = getRuntimeCapabilities();
    return { backend: caps.backend, totalMemoryBytes: caps.totalMemoryBytes };
  } catch {
    return undefined;
  }
}

/** Production wizard step: interactive catalog pick + download. */
async function runProductionWizard(modelsDir: string): Promise<void> {
  const { runFirstRunWizard } = await import('./wizard.js');
  const { select } = await import('@inquirer/prompts');
  const { run: downloadModel } = await import('../download-model.js');
  const gpu = await probeRuntimeGpu();
  await runFirstRunWizard({
    io: {
      select: (opts) => select(opts),
      isTTY: Boolean(process.stdin.isTTY && process.stdout.isTTY),
      log: (line) => console.log(line),
    },
    download: (downloadArgv) => downloadModel(downloadArgv),
    modelsDir,
    totalMemBytes: totalmem(),
    gpu,
  });
}

/** Discover model names only (native-free); production = @mlx-node/lm discovery. */
async function discoverModelNames(modelsDir: string): Promise<{ name: string }[]> {
  const { discoverLocalChatModels } = await import('@mlx-node/lm/model-discovery');
  const models = await discoverLocalChatModels(modelsDir);
  return models.map((m) => ({ name: m.name }));
}

const PINNED_NOTICE: Readonly<Partial<Record<AgentUpdateStatus['pinned'], string>>> = {
  env: 'the installed release stays unused while MLX_AGENT_BIN points elsewhere',
  vendored: 'the installed release stays unused while grok-build/target/release/mlx-agent exists',
};

/**
 * `mlx agent update [--check]`: mlx-owned self-update. The fork binary's own
 * `update` subcommand was built for npm-style installs and can only corrupt a
 * download-managed binary, so the positional is intercepted here instead of
 * forwarded. `--check` emits the machine-readable status the desktop tray
 * consumes; an install run on a piped stdout also ends with the same status
 * JSON as its last line — without it the tray's parse resolves to {} and
 * reports "Up to Date" even when a fallback install still trails latest.
 */
async function runAgentUpdate(scan: AgentArgScan, deps: AgentRunDeps): Promise<void> {
  // scan.help stays false for subcommand argv, so read the passthrough directly.
  if (scan.passthrough.includes('--help') || scan.passthrough.includes('-h')) {
    console.log('Usage: mlx agent update [--check]\n\n  --check   print update status as JSON and exit');
    return;
  }
  const updateDeps = deps.updateDeps ?? {};
  const checkOnly = scan.passthrough.includes('--check');
  try {
    if (checkOnly) {
      const status = await (deps.updateStatus ?? agentUpdateStatus)(updateDeps);
      console.log(JSON.stringify(status));
      return;
    }
    const { status, downloaded } = await (deps.applyUpdate ?? applyAgentUpdate)(updateDeps);
    if (downloaded) {
      // current is the tag that actually installed — a candidate fallback
      // may have landed an older release than latest.
      console.log(`mlx agent: updated to ${status.current ?? status.latest} → ${status.managedPath}`);
      if (status.updateAvailable) {
        console.warn(`mlx agent: warning — latest ${status.latest} could not be fetched; installed ${status.current}`);
      }
    } else {
      console.log(`mlx agent: already up to date (${status.latest})`);
    }
    const pin = PINNED_NOTICE[status.pinned];
    if (pin !== undefined) console.warn(`mlx agent: warning — ${pin}`);
    // Piped consumers (the desktop updater) parse the LAST stdout line;
    // interactive terminals get the human lines above only.
    if (!process.stdout.isTTY) console.log(JSON.stringify(status));
  } catch (error) {
    console.error(`mlx agent: update failed — ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

export async function run(argv: string[], deps: AgentRunDeps = {}): Promise<void> {
  const scan = scanAgentArgs(argv);

  if (scan.update) {
    await runAgentUpdate(scan, deps);
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

  if (scan.kvFormatInvalid !== undefined) {
    console.error('--kv-format must be int8 or bf16');
    process.exitCode = 1;
    return;
  }

  if (scan.maxMemoryInvalid !== undefined) {
    console.error(MAX_MEMORY_ERROR);
    process.exitCode = 1;
    return;
  }

  if (scan.maxContextInvalid !== undefined) {
    console.error(MAX_CONTEXT_ERROR);
    process.exitCode = 1;
    return;
  }

  if (scan.decodeShareInvalid !== undefined) {
    console.error(DECODE_SHARE_ERROR);
    process.exitCode = 1;
    return;
  }

  if (scan.idleReleaseInvalid !== undefined) {
    console.error(IDLE_RELEASE_ERROR);
    process.exitCode = 1;
    return;
  }

  if (scan.cacheDiskError !== undefined) {
    console.error(scan.cacheDiskError);
    process.exitCode = 1;
    return;
  }

  // Must run before the spawned `mlx serve` starts so its native tracing
  // subscriber picks up the env we export here.
  configureAgentTracing(scan);

  const launch = deps.launch ?? launchGrokAgent;

  if (scan.help) {
    printAgentPreamble();
    // The binary prints its full flag list and exits; no server needed.
    process.exitCode = await launch({ argv: ['--help'], needsServer: false });
    return;
  }

  // One-shots and binary subcommands (models/sessions/export/…) resolve and
  // exit inside the binary — forward verbatim with no server and no wizard.
  if (scan.grokOneShot) {
    process.exitCode = await launch({ argv: scan.passthrough, needsServer: false });
    return;
  }

  const resolveModelsDir = deps.resolveModelsDir ?? (await import('@mlx-node/server/host/paths')).resolveModelsDir;
  const discover = deps.discoverModels ?? discoverModelNames;
  const modelsDir = resolveModelsDir(scan.modelsDir);

  let models = await discover(modelsDir);

  if (models.length === 0) {
    try {
      await (deps.wizard ?? runProductionWizard)(modelsDir);
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
      return;
    }

    models = await discover(modelsDir);
    if (models.length === 0) {
      console.error(`No usable model found in ${modelsDir} after the download.`);
      console.error(
        `Expected a subdirectory with a config.json for a supported family (${CHAT_FAMILY_IDS.join('/')}).`,
      );
      console.error('Check the download output above, or point --models-dir at an existing models directory.');
      process.exitCode = 1;
      return;
    }
  }

  const persisted = (deps.readPersistedDefault ?? readPersistedDefaultModel)();
  const { modelId } = chooseDefaultModel(
    models.map((m) => m.name),
    persisted,
  );
  const agentArgv = withWebSearchDisabled(withDefaultModel(scan.passthrough, modelId));
  const requestedModel = agentModelArg(agentArgv);
  const preloadModel =
    requestedModel !== undefined && models.some((m) => m.name === requestedModel) ? requestedModel : undefined;

  process.exitCode = await launch({
    argv: agentArgv,
    modelsDir,
    needsServer: true,
    ...(preloadModel !== undefined ? { preloadModel } : {}),
    ...(scan.kvFormat !== undefined ? { kvFormat: scan.kvFormat } : {}),
    ...(scan.maxMemoryBytes !== undefined ? { maxMemoryBytes: scan.maxMemoryBytes } : {}),
    ...(scan.maxContextTokens !== undefined ? { maxContextTokens: scan.maxContextTokens } : {}),
    ...(scan.decodeShare !== undefined ? { decodeShare: scan.decodeShare } : {}),
    ...(scan.idleReleaseSeconds !== undefined ? { idleReleaseSeconds: scan.idleReleaseSeconds } : {}),
    persistPagedCache: scan.persistPagedCache,
    ...(scan.cacheDisk !== undefined ? { cacheDisk: scan.cacheDisk } : {}),
    ...(scan.cacheDir !== undefined ? { cacheDir: scan.cacheDir } : {}),
  });
}
