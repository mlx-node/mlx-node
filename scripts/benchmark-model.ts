#!/usr/bin/env oxnode

import { fork } from "node:child_process";
import {
  mkdir,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import {
  cpus,
  platform,
  release,
  totalmem,
  version as osVersion,
} from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import type { PerformanceMetrics, SessionCapableModel } from "@mlx-node/lm";

const DEFAULT_RUNS = 3;
const DEFAULT_MAX_NEW_TOKENS = 512;
const DEFAULT_COOLDOWN_SECONDS = 60;
const DEFAULT_WARMUP_RUNS = 0;
const MINIMUM_COMPLETION_RATIO = 0.95;

const SYSTEM_PROMPT =
  "You are a technical writer in a deterministic inference benchmark. Produce a continuous, detailed response and do not conclude early.";
const DEFAULT_PROMPT =
  "Write a comprehensive technical handbook chapter about building a production compiler. Cover lexical analysis, parsing, semantic analysis, type checking, intermediate representations, optimization, code generation, linking, testing, debugging, and deployment. Use detailed explanations and concrete examples. Do not summarize or conclude early; keep expanding the chapter until the generation limit is reached.";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const require = createRequire(import.meta.url);

interface CliOptions {
  modelPath: string;
  runs: number;
  maxNewTokens: number;
  cooldownSeconds: number;
  warmupRuns: number;
  prompt: string;
  outputPath: string | null;
  worker: boolean;
}

interface Sample {
  pid: number;
  loadMs: number;
  generationWallMs: number;
  totalWallMs: number;
  ttftMs: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
  promptTokens: number;
  generatedTokens: number;
  reasoningTokens: number;
  cachedTokens: number;
  finishReason: string;
  minimumGeneratedTokens: number;
}

interface RunReport extends Sample {
  phase: "warmup" | "measured";
  run: number;
}

interface WorkerMessage {
  type: "benchmark-model-sample";
  sample: Sample;
}

interface PackageMetadata {
  name: string;
  version: string;
}

async function readPackageFile(
  packageFile: string,
  packageName: string,
): Promise<PackageMetadata | null> {
  try {
    const parsed: unknown = JSON.parse(await readFile(packageFile, "utf8"));
    if (
      isRecord(parsed) &&
      parsed.name === packageName &&
      typeof parsed.version === "string" &&
      parsed.version.length > 0
    ) {
      return { name: packageName, version: parsed.version };
    }
  } catch {
    // Missing or unreadable package metadata is reported as null.
  }
  return null;
}

function usage(): string {
  return `Usage: oxnode scripts/benchmark-model.ts <model-path> [options]

Benchmarks one model artifact (a native model directory or directly loadable
GGUF file) in isolated child processes. Every sample includes a fresh model load
followed by one deterministic ChatSession generation.

Options:
  --runs <count>             Measured runs (default: ${DEFAULT_RUNS})
  --max-new-tokens <count>   Generation cap (default: ${DEFAULT_MAX_NEW_TOKENS})
  --cooldown <seconds>       Idle time between generations (default: ${DEFAULT_COOLDOWN_SECONDS})
  --warmup-runs <count>      Unmeasured runs, reported separately (default: ${DEFAULT_WARMUP_RUNS})
  --prompt <text>            Override the fixed long-form benchmark prompt
  --output <json-path>       Atomically write the final JSON summary
  -h, --help                 Show this help

The default matches the model-card convention: the median of three measured
512-token generations with a 60-second cooldown between generations. Warmups
are never included in medians.

Short plumbing check (still loads the selected model):
  oxnode scripts/benchmark-model.ts ./model --runs 1 --max-new-tokens 1 --cooldown 0
`;
}

function parseInteger(raw: string, name: string, minimum: number): number {
  if (!/^\d+$/.test(raw)) {
    throw new Error(`${name} must be an integer, got ${JSON.stringify(raw)}`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(
      `${name} must be at least ${minimum}, got ${JSON.stringify(raw)}`,
    );
  }
  return value;
}

function parseCooldown(raw: string): number {
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(
      `--cooldown must be a non-negative number of seconds, got ${JSON.stringify(raw)}`,
    );
  }
  return value;
}

function parseCli(): CliOptions | null {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    strict: true,
    options: {
      runs: { type: "string", default: String(DEFAULT_RUNS) },
      "max-new-tokens": {
        type: "string",
        default: String(DEFAULT_MAX_NEW_TOKENS),
      },
      cooldown: { type: "string", default: String(DEFAULT_COOLDOWN_SECONDS) },
      "warmup-runs": { type: "string", default: String(DEFAULT_WARMUP_RUNS) },
      prompt: { type: "string", default: DEFAULT_PROMPT },
      output: { type: "string" },
      help: { type: "boolean", short: "h", default: false },
      // Internal child-process mode. Kept on the same entrypoint so the
      // parent and worker always execute exactly the same checked-in code.
      worker: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    console.log(usage());
    return null;
  }
  if (positionals.length !== 1) {
    throw new Error(`Expected exactly one model directory.\n\n${usage()}`);
  }

  const prompt = values.prompt.trim();
  if (prompt.length === 0) throw new Error("--prompt must not be empty");
  const output = values.output?.trim();
  if (output !== undefined && output.length === 0)
    throw new Error("--output must not be empty");

  return {
    modelPath: resolve(positionals[0]!),
    runs: parseInteger(values.runs, "--runs", 1),
    maxNewTokens: parseInteger(values["max-new-tokens"], "--max-new-tokens", 1),
    cooldownSeconds: parseCooldown(values.cooldown),
    warmupRuns: parseInteger(values["warmup-runs"], "--warmup-runs", 0),
    prompt,
    outputPath: output === undefined ? null : resolve(output),
    worker: values.worker,
  };
}

function assertMetric(
  value: unknown,
  name: keyof PerformanceMetrics,
  allowZero = false,
): asserts value is number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    (allowZero ? value < 0 : value <= 0)
  ) {
    throw new Error(
      `Missing or invalid performance metric ${name}: ${String(value)}`,
    );
  }
}

async function runWorker(options: CliOptions): Promise<Sample> {
  // Deliberately delayed until worker mode so --help and argument validation
  // never initialize the native binding or load a checkpoint.
  const { ChatSession, HarrierModel, loadModel } = await import("@mlx-node/lm");

  const totalStartedAt = performance.now();
  const loadStartedAt = totalStartedAt;
  const model = await loadModel(options.modelPath);
  const loadedAt = performance.now();

  if (model instanceof HarrierModel) {
    throw new Error(
      `Expected a generative model, got an embedding model: ${options.modelPath}`,
    );
  }

  const session = new ChatSession(model as unknown as SessionCapableModel, {
    system: SYSTEM_PROMPT,
  });
  const generationStartedAt = performance.now();
  const result = await session.send(options.prompt, {
    config: {
      maxNewTokens: options.maxNewTokens,
      temperature: 0,
      reasoningEffort: "none",
      reportPerformance: true,
    },
  });
  const completedAt = performance.now();

  const metrics = result.performance;
  if (metrics === undefined) {
    throw new Error(
      "Generation returned no performance metrics despite reportPerformance: true",
    );
  }
  assertMetric(metrics.ttftMs, "ttftMs");
  assertMetric(metrics.prefillTokensPerSecond, "prefillTokensPerSecond");
  // A one-token plumbing check has no decode interval by definition, so zero
  // is valid only for that short-validation case. Normal benchmark runs must
  // provide a positive decode throughput.
  assertMetric(
    metrics.decodeTokensPerSecond,
    "decodeTokensPerSecond",
    options.maxNewTokens === 1,
  );

  const minimumGeneratedTokens = Math.max(
    1,
    Math.ceil(options.maxNewTokens * MINIMUM_COMPLETION_RATIO),
  );
  if (result.numTokens < minimumGeneratedTokens) {
    throw new Error(
      `Generation ended materially early: got ${result.numTokens}/${options.maxNewTokens} tokens ` +
        `(finishReason=${JSON.stringify(result.finishReason)}, minimum=${minimumGeneratedTokens})`,
    );
  }

  return {
    pid: process.pid,
    loadMs: loadedAt - loadStartedAt,
    generationWallMs: completedAt - generationStartedAt,
    totalWallMs: completedAt - totalStartedAt,
    ttftMs: metrics.ttftMs,
    prefillTokensPerSecond: metrics.prefillTokensPerSecond,
    decodeTokensPerSecond: metrics.decodeTokensPerSecond,
    promptTokens: result.promptTokens,
    generatedTokens: result.numTokens,
    reasoningTokens: result.reasoningTokens,
    cachedTokens: result.cachedTokens,
    finishReason: result.finishReason,
    minimumGeneratedTokens,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isWorkerMessage(value: unknown): value is WorkerMessage {
  if (
    !isRecord(value) ||
    value.type !== "benchmark-model-sample" ||
    !isRecord(value.sample)
  )
    return false;
  const sample = value.sample;
  return (
    typeof sample.pid === "number" &&
    typeof sample.loadMs === "number" &&
    typeof sample.ttftMs === "number" &&
    typeof sample.prefillTokensPerSecond === "number" &&
    typeof sample.decodeTokensPerSecond === "number" &&
    typeof sample.generatedTokens === "number"
  );
}

async function sendWorkerMessage(message: WorkerMessage): Promise<void> {
  if (process.send === undefined || process.disconnect === undefined)
    throw new Error("Benchmark worker was started without an IPC channel");
  await new Promise<void>((resolvePromise, rejectPromise) => {
    process.send!(message, (error) => {
      if (error === null) resolvePromise();
      else rejectPromise(error);
    });
  });
  process.disconnect!();
}

function workerArgs(options: CliOptions): string[] {
  return [
    options.modelPath,
    "--worker",
    "--runs",
    String(options.runs),
    "--max-new-tokens",
    String(options.maxNewTokens),
    "--cooldown",
    String(options.cooldownSeconds),
    "--warmup-runs",
    String(options.warmupRuns),
    "--prompt",
    options.prompt,
  ];
}

function runIsolatedSample(options: CliOptions): Promise<Sample> {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = fork(SCRIPT_PATH, workerArgs(options), {
      execArgv: process.execArgv,
      stdio: ["ignore", "inherit", "inherit", "ipc"],
    });
    let sample: Sample | undefined;
    let protocolError: Error | undefined;

    child.on("message", (message) => {
      if (isWorkerMessage(message)) sample = message.sample;
      else
        protocolError = new Error(
          "Benchmark worker returned an invalid IPC message",
        );
    });
    child.once("error", rejectPromise);
    child.once("exit", (code, signal) => {
      if (protocolError !== undefined) {
        rejectPromise(protocolError);
      } else if (code !== 0) {
        rejectPromise(
          new Error(
            `Benchmark worker failed (${signal === null ? `exit ${code}` : `signal ${signal}`})`,
          ),
        );
      } else if (sample === undefined) {
        rejectPromise(
          new Error("Benchmark worker exited without returning a sample"),
        );
      } else {
        resolvePromise(sample);
      }
    });
  });
}

function formatSample(report: RunReport): string {
  const label =
    report.phase === "warmup" ? `warmup ${report.run}` : `run ${report.run}`;
  return (
    `${label}: load ${report.loadMs.toFixed(0)} ms | TTFT ${report.ttftMs.toFixed(0)} ms | ` +
    `prefill ${report.prefillTokensPerSecond.toFixed(1)} tok/s | ` +
    `decode ${report.decodeTokensPerSecond.toFixed(1)} tok/s | ` +
    `${report.generatedTokens} tokens | ${report.finishReason}`
  );
}

function median(values: number[]): number {
  if (values.length === 0)
    throw new Error("Cannot take the median of an empty sample set");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function medians(reports: RunReport[]) {
  return {
    loadMs: median(reports.map((report) => report.loadMs)),
    generationWallMs: median(reports.map((report) => report.generationWallMs)),
    totalWallMs: median(reports.map((report) => report.totalWallMs)),
    ttftMs: median(reports.map((report) => report.ttftMs)),
    prefillTokensPerSecond: median(
      reports.map((report) => report.prefillTokensPerSecond),
    ),
    decodeTokensPerSecond: median(
      reports.map((report) => report.decodeTokensPerSecond),
    ),
    promptTokens: median(reports.map((report) => report.promptTokens)),
    generatedTokens: median(reports.map((report) => report.generatedTokens)),
  };
}

async function readPackageMetadata(
  packageName: string,
): Promise<PackageMetadata | null> {
  let entryPath: string;
  try {
    entryPath = require.resolve(packageName);
  } catch {
    const workspacePackageFile =
      packageName === "@mlx-node/lm"
        ? resolve(dirname(SCRIPT_PATH), "../packages/lm/package.json")
        : packageName === "@mlx-node/core"
          ? resolve(dirname(SCRIPT_PATH), "../packages/core/package.json")
          : null;
    return workspacePackageFile === null
      ? null
      : await readPackageFile(workspacePackageFile, packageName);
  }

  let currentDirectory = dirname(entryPath);
  for (let depth = 0; depth < 5; depth++) {
    const metadata = await readPackageFile(
      resolve(currentDirectory, "package.json"),
      packageName,
    );
    if (metadata !== null) return metadata;
    // Keep walking: the resolved entry is commonly inside a dist/ folder.
    const parentDirectory = dirname(currentDirectory);
    if (parentDirectory === currentDirectory) break;
    currentDirectory = parentDirectory;
  }
  return null;
}

function nativePackageCandidates(): string[] {
  if (process.platform === "darwin" && process.arch === "arm64")
    return ["@mlx-node/core-darwin-arm64"];
  if (process.platform === "linux" && process.arch === "arm64") {
    return [
      "@mlx-node/core-linux-arm64-gnu",
      "@mlx-node/core-linux-arm64-musl",
    ];
  }
  return [];
}

async function environmentMetadata() {
  let nativePackage: PackageMetadata | null = null;
  for (const candidate of nativePackageCandidates()) {
    nativePackage = await readPackageMetadata(candidate);
    if (nativePackage !== null) break;
  }

  return {
    platform: platform(),
    arch: process.arch,
    osRelease: release(),
    osVersion: osVersion(),
    cpu: cpus()[0]?.model ?? null,
    logicalCpus: cpus().length,
    totalMemoryBytes: totalmem(),
    nodeVersion: process.version,
    packages: {
      lm: await readPackageMetadata("@mlx-node/lm"),
      core: await readPackageMetadata("@mlx-node/core"),
      native: nativePackage,
    },
  };
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolvePromise) =>
    setTimeout(resolvePromise, milliseconds),
  );
}

async function writeJsonAtomically(
  outputPath: string,
  contents: string,
): Promise<void> {
  const outputDirectory = dirname(outputPath);
  await mkdir(outputDirectory, { recursive: true });
  const temporaryPath = `${outputPath}.${process.pid}.${Date.now()}.tmp`;

  try {
    await writeFile(temporaryPath, `${contents}\n`, {
      encoding: "utf8",
      flag: "wx",
    });
    await rename(temporaryPath, outputPath);
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined);
    throw new Error(
      `Failed to write JSON summary to ${JSON.stringify(outputPath)}`,
      { cause: error },
    );
  }
}

async function assertModelPath(modelPath: string): Promise<void> {
  let modelStat;
  try {
    modelStat = await stat(modelPath);
  } catch (error) {
    throw new Error(
      `Cannot access model directory ${JSON.stringify(modelPath)}`,
      { cause: error },
    );
  }
  if (!modelStat.isDirectory() && !modelStat.isFile()) {
    throw new Error(`Model path is neither a directory nor a file: ${modelPath}`);
  }
}

async function runParent(options: CliOptions): Promise<void> {
  await assertModelPath(options.modelPath);
  const startedAt = new Date();
  const environment = await environmentMetadata();
  const warmups: RunReport[] = [];
  const runs: RunReport[] = [];
  const totalSamples = options.warmupRuns + options.runs;
  let completedSamples = 0;

  console.log(`Benchmarking ${options.modelPath}`);
  console.log(
    `${options.runs} measured x ${options.maxNewTokens} tokens; ` +
      `${options.warmupRuns} warmup; ${options.cooldownSeconds}s cooldown; temperature 0`,
  );
  console.log("Each sample uses a fresh child process and model load.");

  const executePhase = async (
    phase: RunReport["phase"],
    count: number,
    destination: RunReport[],
  ) => {
    for (let index = 1; index <= count; index++) {
      console.log(
        `\n${phase === "warmup" ? "Warmup" : "Run"} ${index}/${count}: starting`,
      );
      const report: RunReport = {
        ...(await runIsolatedSample(options)),
        phase,
        run: index,
      };
      destination.push(report);
      console.log(formatSample(report));

      completedSamples++;
      if (completedSamples < totalSamples && options.cooldownSeconds > 0) {
        console.log(`Cooling down for ${options.cooldownSeconds}s...`);
        await sleep(options.cooldownSeconds * 1000);
      }
    }
  };

  await executePhase("warmup", options.warmupRuns, warmups);
  await executePhase("measured", options.runs, runs);

  const summary = {
    schemaVersion: 1,
    modelPath: options.modelPath,
    startedAt: startedAt.toISOString(),
    completedAt: new Date().toISOString(),
    environment,
    configuration: {
      runs: options.runs,
      maxNewTokens: options.maxNewTokens,
      cooldownSeconds: options.cooldownSeconds,
      warmupRuns: options.warmupRuns,
      temperature: 0,
      reasoningEffort: "none",
      reportPerformance: true,
      minimumCompletionRatio: MINIMUM_COMPLETION_RATIO,
      systemPrompt: SYSTEM_PROMPT,
      prompt: options.prompt,
      isolation: "fresh child process and model load per sample",
    },
    warmups,
    runs,
    medians: medians(runs),
  };

  const summaryJson = JSON.stringify(summary, null, 2);
  if (options.outputPath !== null) {
    await writeJsonAtomically(options.outputPath, summaryJson);
    console.log(`\nWrote JSON summary to ${options.outputPath}`);
  }
  console.log("\nJSON summary:");
  console.log(summaryJson);
}

async function main(): Promise<void> {
  const options = parseCli();
  if (options === null) return;

  if (options.worker) {
    const sample = await runWorker(options);
    await sendWorkerMessage({ type: "benchmark-model-sample", sample });
    return;
  }

  await runParent(options);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});
