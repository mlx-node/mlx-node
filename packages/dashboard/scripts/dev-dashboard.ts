/**
 * Dev harness: the REAL dashboard runtime on a generated fixture home, plus a Vite dev server for the SPA, in one
 * process. The tab talks to the runtime over Vite's HMR socket (`dev/harness-plugin.ts`). Dev only.
 *
 *   oxnode scripts/dev-dashboard.ts [--fresh] [--home <dir>] [--port <n>]
 *
 * `MLX_DEV_LOAD_MS`: how long the fake engine "loads" a model (default 4000). `MLX_DEV_LOAD_HIDDEN=1`: the heartbeat
 * does not report the load, as when a running answer holds the engine. `MLX_DEV_UNLOAD_BUSY=1`: the engine answers
 * every unload with 409. `MLX_DEV_ENGINE_STATES=1`: five more loaded models, one per state of the Engine pool section.
 * `MLX_DEV_TICKER_TPS` / `MLX_DEV_TICKER_MEM_GB`: fixed Output rate and MLX memory values in the heartbeat.
 * `MLX_DEV_FLIGHT=one-model`: both turns in flight on one model, one asked through the alias `mlx-local`.
 * `MLX_DEV_FLIGHT=phases`: three more turns before their prefill: model busy, waiting for a slot, preparing.
 * `MLX_DEV_LOAD_KERNELS=<done>/<steps>`: the heartbeat reports a load as "Preparing GPU kernels", done of steps.
 * `MLX_DEV_DISK_CACHE=native|off|older`: the disk cache the heartbeat reports (default: on, whole prompt).
 */

import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

import { createServer } from 'vite-plus';

import { HARNESS_LOAD_MS, startHarnessEngine } from '../__test__/helpers/fake-engine.js';
import { startHarnessProviders } from '../__test__/helpers/fake-providers.js';
import { createDashboardRuntime } from '../src/runtime.js';
import {
  assertFixtureTree,
  DEFAULT_HOME,
  FIXTURE_COMMAND,
  fixtureArgs,
  fixtureExists,
  fixturePaths,
  type LiveProcess,
  refreshRuntime,
  resetHome,
  writeFixtureHome,
  writeLiveFile,
} from './dev-fixture.js';
import { devHarness, harnessChatOptions, harnessTempRoot } from './dev/harness-plugin.js';

const UI_ROOT = join(import.meta.dirname, '../ui');
/** `MLX_DEV_WORK_DEEP=1`: the folder dialog answers a folder whose encoded path is too long for the agent. */
const DEEP_FOLDER = [
  'a-project-folder-with-a-long-name-one',
  'a-project-folder-with-a-long-name-two',
  'a-project-folder-with-a-long-name-three',
  'a-project-folder-with-a-long-name-four',
  'a-project-folder-with-a-long-name-five',
  'a-project-folder-with-a-long-name-six',
];
/** Rewrites the heartbeat well inside its promised 5 s interval, so the process reads as running. */
const HEARTBEAT_MS = 2_000;

function tickerValues(): LiveProcess['ticker'] {
  const tps = Number(process.env.MLX_DEV_TICKER_TPS ?? '');
  const memGb = Number(process.env.MLX_DEV_TICKER_MEM_GB ?? '');
  if (process.env.MLX_DEV_TICKER_TPS === undefined || !Number.isFinite(tps)) return undefined;
  const mem = process.env.MLX_DEV_TICKER_MEM_GB !== undefined && Number.isFinite(memGb) ? memGb : 27;
  return { decodeTps: tps, memActive: Math.round(mem * 1024 ** 3) };
}

function flightValue(value: string | undefined): LiveProcess['flight'] {
  return value === 'one-model' || value === 'phases' ? value : 'two-models';
}

/** `<done>/<steps>` of `MLX_DEV_LOAD_KERNELS`, or null. */
function kernelSteps(value: string | undefined): { stepsDone: number; steps: number } | null {
  if (value === undefined) return null;
  const [done, steps] = value.split('/').map((part) => Number.parseInt(part, 10));
  if (done === undefined || steps === undefined || !Number.isFinite(done) || !Number.isFinite(steps)) return null;
  return { stepsDone: done, steps };
}

function diskCacheValue(value: string | undefined): LiveProcess['diskCache'] {
  return value === 'native' || value === 'off' || value === 'older' ? value : 'whole-prompt';
}

async function main(): Promise<void> {
  const { home, fresh, port } = fixtureArgs(process.argv.slice(2), DEFAULT_HOME);
  if (fresh) resetHome(home);
  const paths = fixturePaths(home);
  const startedAt = Date.now() - 6 * 3_600_000;
  const engineStates = process.env.MLX_DEV_ENGINE_STATES === '1';
  const live: LiveProcess = {
    pid: process.pid,
    startedAt,
    engineStates,
    ticker: tickerValues(),
    flight: flightValue(process.env.MLX_DEV_FLIGHT),
    diskCache: diskCacheValue(process.env.MLX_DEV_DISK_CACHE),
  };
  if (fixtureExists(home)) {
    assertFixtureTree(home);
    refreshRuntime(home, live);
  } else {
    process.stdout.write(`writing fixture home ${home}\n`);
    const summary = writeFixtureHome(home, { engineStates });
    live.startedAt = summary.live.startedAt;
  }
  const heartbeat = setInterval(() => writeLiveFile(home, live), HEARTBEAT_MS);

  const providers = await startHarnessProviders();
  const loadMs = Number.parseInt(process.env.MLX_DEV_LOAD_MS ?? '', 10);
  const hidden = process.env.MLX_DEV_LOAD_HIDDEN === '1';
  const kernels = kernelSteps(process.env.MLX_DEV_LOAD_KERNELS);
  const engine = await startHarnessEngine({
    loadMs: Number.isFinite(loadMs) && loadMs >= 0 ? loadMs : HARNESS_LOAD_MS,
    unloadBusy: process.env.MLX_DEV_UNLOAD_BUSY === '1',
    onState: (state) => {
      live.loading =
        hidden || state.loading === null
          ? []
          : [
              kernels === null
                ? { ...state.loading, phase: 'weights', stepsDone: 0, steps: 0 }
                : { ...state.loading, phase: 'kernels', ...kernels },
            ];
      live.chatModels = state.resident === null ? [] : [state.resident];
      writeLiveFile(home, live);
    },
  });
  const chat = { ...harnessChatOptions(home, providers.baseUrls, engine.url), tempRoots: harnessTempRoot(home) };
  const workFolder = join(home, 'work', ...(process.env.MLX_DEV_WORK_DEEP === '1' ? DEEP_FOLDER : []));
  mkdirSync(workFolder, { recursive: true });
  const runtime = createDashboardRuntime({
    dbPath: paths.dbPath,
    modelsDir: paths.modelsDir,
    tracesDir: paths.tracesDir,
    runtimeDir: paths.runtimeDir,
    supervisorPath: paths.supervisorPath,
    cacheRoot: paths.cacheRoot,
    agentHome: paths.agentHome,
    codingAgents: { home, env: {} },
    prepareDelegation: async () => FIXTURE_COMMAND,
    chat,
  });

  let server: Awaited<ReturnType<typeof createServer>>;
  try {
    server = await createServer({
      root: UI_ROOT,
      configFile: join(UI_ROOT, 'vite.config.ts'),
      plugins: [devHarness(runtime, { chat, workFolder })],
      server: { host: '127.0.0.1', port, strictPort: true },
    });
    await server.listen();
  } catch (err) {
    clearInterval(heartbeat);
    await runtime.close();
    await providers.close();
    await engine.close();
    throw err;
  }
  process.stdout.write(`mlx-node Control Panel dev harness: http://127.0.0.1:${port}/  (home ${home})\n`);

  let stopping = false;
  const stop = async (): Promise<void> => {
    if (stopping) return;
    stopping = true;
    clearInterval(heartbeat);
    try {
      await server.close();
    } finally {
      await runtime.close();
      await providers.close();
      await engine.close();
      rmSync(join(paths.runtimeDir, `live-${process.pid}.json`), { force: true });
      process.exit(0);
    }
  };
  process.on('SIGINT', () => void stop());
  process.on('SIGTERM', () => void stop());
}

main().catch((err: unknown) => {
  process.stderr.write(`dev harness failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  process.exit(1);
});
