/**
 * `mlx serve` — run the inference host on its own, in the foreground.
 *
 * The same `createInferenceHost` the desktop app runs inside an Electron
 * `utilityProcess`, with nothing supervising it and stdout attached. That is
 * the point: when the sidecar wedges, this is how you reproduce it in a
 * terminal and watch it happen.
 */

import {
  createInferenceHost,
  LAUNCHER_ENGINE_POLICY,
  ModelNotFoundError,
  NoModelsDiscoveredError,
  WHOLE_PROMPT_CAPTURE_BLOCKS,
  WHOLE_PROMPT_CAPTURE_BUDGET_MS,
  type InferenceHost,
  type InferenceHostOptions,
} from '@mlx-node/server/host';
import { SERVER_CLOSE_MS } from '@mlx-node/server/supervision';

import { parseServeArgs, persistOption, type ServeArgs } from './serve-args.js';

export { parseServeArgs, type ServeArgs } from './serve-args.js';

function printHelp(): void {
  console.log(`
Serve local models over an Anthropic/OpenAI-compatible HTTP API

Usage:
  mlx serve [options]

Options:
  --port <n>          Port to listen on (default: auto-pick a free port;
                      0 binds an ephemeral port and prints the real one)
  --host <h>          Host to bind (default: 127.0.0.1). A non-loopback bind
                      exposes your local models to the network — pair it
                      with --auth-token.
  --models-dir <dir>  Directory to discover models from
                      (default: ~/.mlx-node/models; overridable via
                      MLX_MODELS_DIR env or ~/.mlx-node/config.json)
  --model <name>      Which discovered model is the default. Must match a
                      directory name under --models-dir. Defaults to
                      ANTHROPIC_MODEL env, else the first discovered model.
  --preload <name>    Start loading this discovered model at boot instead of
                      on its first request. A failed preload is logged and
                      the server keeps running.
  --model-residency <mode>  capability (default) or single
  --kv-format <fmt>   Target K/V cache format: int8 or bf16 (default: int8
                      where the int8 kernels serve the model, else bf16).
                      Sets the block-paged pool (qwen3_5, qwen3_5_moe,
                      qwen4_exp).
                      bf16 uses more memory.
  --max-memory <size> Metal memory budget, e.g. 28G; it can only lower the
                      automatic budget (default: auto). Qwen3.5 dense and
                      MoE loads size their KV pool and windows from it.
  --max-context <tokens>  Context token limit, e.g. 100K or 1M (K = 1024
                      tokens, M = 1000000 tokens), up to the model's limit,
                      at most 1048576; past its native window a model with
                      YaRN turns it on, which is static and changes answers
                      at every length, so pass more than the native window
                      only when a request needs it (default: auto, the
                      native window when memory holds it). Qwen3.5 dense,
                      MoE and Qwen3.8 Flash Next take it; a load whose
                      memory plan holds fewer tokens fails.
  --decode-share <share>  Decode time owed per unit of prefill time while
                      other requests generate (default: 0.5). Higher keeps
                      their output faster during a long prompt and slows
                      that prompt; 0 alternates one command each. Qwen3.5
                      dense and MoE take it.
  --activations <mode>  auto or bf16 (default: auto for Qwen3.5 dense with a
                      DFlash2 draft, bf16 for dense with none and for
                      MoE). On M5 and later Macs auto lets attention
                      over the int8 KV cache score int8 queries for every
                      token at position 7,168 or later, in prefill, verify
                      and decode alike: faster at depth. bf16 keeps every
                      query bf16. Both modes share one disk cache. Qwen3.5
                      dense and MoE take it.
  --idle-release <duration>  Time without a request before the engine unwires
                      its memory and frees the weights: seconds, or with an
                      s, m or h suffix, e.g. 30m (default: 10m); off keeps
                      both, though POST /v1/memory/release still frees the
                      weights. The next request loads the weights again.
  --persist-cache     Keep the paged KV disk cache across restarts, for every
                      family that has one (default: on, unless
                      MLX_PERSIST_PAGED_CACHE turns it off): the cold tier of
                      Qwen3, Gemma4, Muse-Glimmer and LFM2, and the SSD state
                      tier of Qwen3.5 dense and MoE.
  --no-persist-cache  Turn it off, whatever MLX_PERSIST_PAGED_CACHE says; the
                      Qwen3.5 state tier then keeps temporary files that go
                      when serve stops.
  --max-cache-disk <size>  SSD quota for the states of Qwen3.5 dense and MoE
                      loads: auto follows the volume, a quarter of its free
                      space beyond a floor of 10G (5% of a volume under
                      200G), at most 32G; a size such as 16G is held as
                      given while the volume keeps that floor; 0 disables
                      (default: auto).
  --cache-dir <dir>   Where the persistent Qwen3.5 state tier keeps its files
                      (default: ~/.mlx-node/cache/prefix-cache).
  --persist-whole-prompt  With the disk cache on, let one turn of Qwen3, Gemma4,
                      Muse-Glimmer and LFM2 write the whole prompt to the
                      cold tier: up to ${WHOLE_PROMPT_CAPTURE_BLOCKS} blocks within ${WHOLE_PROMPT_CAPTURE_BUDGET_MS} ms,
                      where MLX_COLD_CAPTURE_BLOCKS_PER_TURN /
                      MLX_COLD_CAPTURE_BUDGET_MS hold no number (default: 128
                      blocks, 250 ms); not the Qwen3.5 state tier. The turn
                      ends after the write. \`mlx agent\` passes it.
  --thinking-token-budget <n>  Thinking tokens of a thinking request that
                      sets no budget, after which its thinking closes
                      (default: no cap).
  --thinking-cycle-guard  Close a thought that repeats one exact block of
                      tokens: twice, over at least 512 tokens (default: off).
  --thinking-end-text <text>  Text written before the close when the budget or
                      the guard ends a thought, 1 to 512 characters (default:
                      the close alone). \`mlx agent\` passes all three.
  --thinking-levels   List each model family's thinking levels on
                      /v1/models and map a Messages request's
                      extra_body.reasoning_effort through them (default:
                      off, the field is used as it is). \`mlx agent\`
                      passes it.
  --rerank-config <file>    JSON { policy?, families?, service? } for rerank limits
  --auth-token <tok>  Require this secret on every route except /health,
                      as \`x-api-key\` or \`authorization: Bearer\`. Also
                      readable from MLX_SERVER_AUTH_TOKEN.
  -v, --verbose       Write every HTTP request/response to a log dir
  --log-dir <dir>     Override the verbose log directory (implies --verbose).
                      Default: ~/.mlx-node/logs/<ISO-timestamp>/.
  -h, --help          Show this help message

Models load lazily. Default residency: one chat/decision model, one ASR model,
one TTS model, and one reranker. Use --model-residency single for one model globally.

Examples:
  mlx serve
  mlx serve --port 8080
  mlx serve --port 8080 --model qwen3.5-9b
  mlx serve --host 0.0.0.0 --auth-token "$(openssl rand -hex 16)"
`);
}

/** Seams injected only by tests; production callers pass nothing. */
export interface ServeRunOptions {
  /** Merged over the flag-derived host options (store path, stub loader, …). */
  hostOverrides?: Partial<InferenceHostOptions>;
  /**
   * Invoked once the host is listening. When supplied, `run` resolves after
   * it settles instead of parking on signals — the signal handlers are still
   * installed first, so a test can grab and invoke the real one.
   */
  onReady?: (host: InferenceHost) => void | Promise<void>;
  /** Stands in for `process.exit` so a test can observe the code. */
  exit?: (code: number) => void;
}

export async function run(argv: string[], deps: ServeRunOptions = {}): Promise<void> {
  let args: ServeArgs;
  try {
    args = parseServeArgs(argv);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    printHelp();
    process.exit(1);
  }

  if (args.help) {
    printHelp();
    return;
  }

  let host: InferenceHost;
  try {
    host = await createInferenceHost({
      port: args.port,
      host: args.host,
      modelsDir: args.modelsDir,
      model: args.model,
      modelResidency: args.modelResidency,
      ...(args.kvFormat !== undefined ? { kvFormat: args.kvFormat } : {}),
      ...(args.maxMemoryBytes !== undefined ? { maxMemoryBytes: args.maxMemoryBytes } : {}),
      ...(args.maxContextTokens !== undefined ? { maxContextTokens: args.maxContextTokens } : {}),
      ...(args.decodeShare !== undefined ? { decodeShare: args.decodeShare } : {}),
      ...(args.activations !== undefined ? { activations: args.activations } : {}),
      ...(args.idleReleaseSeconds !== undefined ? { idleReleaseSeconds: args.idleReleaseSeconds } : {}),
      cacheDisk: args.cacheDisk,
      persistPagedCache: persistOption(args.persistPagedCache),
      ...(args.cacheDir !== undefined ? { cacheDir: args.cacheDir } : {}),
      ...(args.thinkingTokenBudget !== undefined ? { thinkingTokenBudget: args.thinkingTokenBudget } : {}),
      ...(args.persistWholePrompt === true ? { persistWholePrompt: true } : {}),
      ...(args.thinkingCycleGuard === true ? { thinkingCycleGuard: true } : {}),
      ...(args.thinkingEndText !== undefined ? { thinkingEndText: args.thinkingEndText } : {}),
      ...(args.thinkingLevels === true ? { thinkingLevels: true } : {}),
      ...(await readRerankConfig(args.rerankConfig)),
      authToken: args.authToken,
      logDir: args.logDir,
      // Same paged-prefill chunking `mlx launch claude` applies. A value
      // already in the environment wins; see `env-policy.ts`.
      enginePolicy: LAUNCHER_ENGINE_POLICY,
      hostKind: 'serve',
      ...deps.hostOverrides,
    });
  } catch (err) {
    if (err instanceof NoModelsDiscoveredError) {
      console.error(err.message);
      console.error('Run: mlx download model --model Qwen/Qwen3.5-9B');
      process.exit(1);
    }
    if (err instanceof ModelNotFoundError) {
      console.error(err.message);
      console.error('Discovered models:');
      for (const name of err.available) console.error(`  - ${name}`);
      process.exit(1);
    }
    throw err;
  }

  console.log(
    `[mlx] models dir: ${host.modelsDir} | listening on ${host.url} | discovered ${host.models.length + (host.audioModels?.length ?? 0) + (host.rerankModels?.length ?? 0)} model(s) | default: ${host.boundModel}`,
  );
  console.log(`[mlx] paged KV disk cache: ${host.persistPagedCache ? 'on' : 'off'}`);
  if (host.logDir !== null) console.log(`[mlx] verbose logging → ${host.logDir}`);
  if (host.coldCapture !== null)
    console.log(
      `[mlx] paged KV disk capture per turn: ${host.coldCapture.blocksPerTurn} blocks, ${host.coldCapture.budgetMs} ms`,
    );
  console.log('Press Ctrl+C to stop.');

  if (args.preload !== undefined) {
    const name = args.preload;
    void host.loadModel(name).catch((err: unknown) => {
      console.error(`[mlx] preload of "${name}" failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }

  const exit = deps.exit ?? ((code: number): void => process.exit(code));

  let shuttingDown = false;
  const shutdown = async (signal: NodeJS.Signals): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    // Detach first: a second Ctrl+C during the grace period should reach the
    // default handler and kill us outright rather than queue another close.
    process.removeListener('SIGINT', onSigint);
    process.removeListener('SIGTERM', onSigterm);
    console.log(`\nReceived ${signal}, shutting down mlx serve…`);
    try {
      await host.close({ timeoutMs: SERVER_CLOSE_MS });
    } catch {
      /* `close()` guards each disposer; never let shutdown itself throw */
    }
    exit(0);
  };
  const onSigint = (): void => void shutdown('SIGINT');
  const onSigterm = (): void => void shutdown('SIGTERM');
  process.on('SIGINT', onSigint);
  process.on('SIGTERM', onSigterm);

  if (deps.onReady !== undefined) await deps.onReady(host);
}

async function readRerankConfig(
  path?: string,
): Promise<Pick<import('@mlx-node/server/host').InferenceHostOptions, 'reranker' | 'rerank'>> {
  if (!path) return {};
  const { readFile } = await import('node:fs/promises');
  const value: unknown = JSON.parse(await readFile(path, 'utf8'));
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid rerank configuration');
  const config = value as Record<string, unknown>;
  for (const key of Object.keys(config))
    if (!['policy', 'families', 'service'].includes(key)) throw new Error(`Unknown rerank config key: ${key}`);
  return {
    reranker: {
      policy: config.policy as import('@mlx-node/rerank/types').RerankerFamilyOptions['policy'],
      families: config.families as import('@mlx-node/rerank/types').RerankerFamilyOptions['families'],
    },
    rerank: config.service as import('@mlx-node/server').RerankServiceOptions,
  };
}
