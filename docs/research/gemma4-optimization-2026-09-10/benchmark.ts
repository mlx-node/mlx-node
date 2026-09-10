import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  closeSync,
  mkdirSync,
  openSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const modelDir = resolve(root, ".cache/models/gemma-4-12b-it-qat-q4_0-gguf");
const modelFile = resolve(modelDir, "gemma-4-12b-it-qat-q4_0.gguf");
const llamaDir = "/Users/brooklyn/workspace/github/llama.cpp";
const llamaServer = resolve(llamaDir, "build-codex-bench/bin/llama-server");
const dataDir = resolve(
  root,
  ".cache/benchmarks/gemma4-optimization-2026-09-10",
);
const cases = ["diff-review", "test-review", "final-review"];
const outputTokens = 256;
const cooldownSeconds = 20;
const env = {
  ...process.env,
  MLX_AGENT_METRICS: "0",
  MLX_PERSIST_PAGED_CACHE: "0",
  MLX_PAGED_PREFILL_CHUNK_SIZE: "512",
};
const sha = (s: string | Buffer) =>
  createHash("sha256").update(s).digest("hex");
const core = () =>
  import(resolve(root, process.env.BENCH_CORE || "packages/core/index.cjs"));
const variant = process.env.BENCH_VARIANT || "optimized";
const save = (path: string, value: unknown) =>
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
const input = (name: string) =>
  JSON.parse(
    readFileSync(
      resolve(root, ".cache/benchmarks/gemma4-agent-2026-09-10/inputs.json"),
      "utf8",
    ),
  ).find((x: any) => x.name === name);

async function mlx(name: string, run: string) {
  const planLog = resolve(dataDir, `raw/${variant}-${name}-${run}.plans.jsonl`);
  process.env.MLX_NODE_LOG = "mlx_core::decode_tuning=info";
  process.env.MLX_NODE_LOG_FILE = planLog;
  const data = input(name);
  const n = data.promptTokens;
  const { Gemma4Model, getMemorySnapshot, resetPeakMemory } = await core();
  const started = performance.now();
  const model = await Gemma4Model.load(modelFile);
  const loadMs = performance.now() - started;
  const memoryAfterLoad = getMemorySnapshot();
  const config = {
    temperature: 0,
    topK: 1,
    topP: 1,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    reasoningEffort: "high",
    tools: data.tools,
    enableMtp: false,
    reuseCache: true,
    reportPerformance: true,
  };
  await model.chatSessionStart(input(cases[0]).messages, {
    ...config,
    maxNewTokens: 32,
  });
  await model.resetCaches();
  resetPeakMemory();
  const memoryBeforeRequest = getMemorySnapshot();
  const start = performance.now();
  const result = await model.chatSessionStart(data.messages, {
    ...config,
    maxNewTokens: outputTokens,
  });
  const wallMs = performance.now() - start;
  const memoryAfterRequest = getMemorySnapshot();
  const p = result.performance;
  if (
    result.promptTokens !== n ||
    result.cachedTokens !== 0 ||
    result.numTokens !== outputTokens
  ) {
    throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
  }
  const decodeMs = ((result.numTokens - 1) / p.decodeTokensPerSecond) * 1000;
  const sample = {
    runtime: variant,
    env: Object.fromEntries(
      Object.entries(process.env).filter(([key]) => key.startsWith("MLX_")),
    ),
    addonSha256: sha(
      readFileSync(
        resolve(
          root,
          process.env.BENCH_CORE
            ? dirname(process.env.BENCH_CORE)
            : "packages/core",
          "mlx-core.darwin-arm64.node",
        ),
      ),
    ),
    name,
    n,
    run,
    loadMs,
    memoryAfterLoad,
    memoryBeforeRequest,
    memoryAfterRequest,
    decodePlans: readFileSync(planLog, "utf8").trim().split("\n")
      .filter(Boolean).map((line) => JSON.parse(line))
      .filter((event) => event.event === "gemma4_decode_tuned")
      .map(({ context_bucket, stage, early_layers, grouped_stripes, samples,
        candidate_early_layers, candidate_stripes }) => ({
        contextBucket: context_bucket, stage, earlyLayers: early_layers,
        groupedStripes: grouped_stripes, samples: JSON.parse(samples),
        candidateEarlyLayers: JSON.parse(candidate_early_layers),
        candidateStripes: JSON.parse(candidate_stripes),
      })),
    wallMs,
    prefillMs: p.ttftMs,
    prefillTps: p.prefillTokensPerSecond,
    decodeMs,
    decodeTps: p.decodeTokensPerSecond,
    accountedMs: p.ttftMs + decodeMs,
    promptTokens: result.promptTokens,
    generatedTokens: result.numTokens,
    cachedTokens: result.cachedTokens,
    textSha256: sha(result.rawText),
    inputSha256: data.sha256,
    result,
  };
  save(resolve(dataDir, `raw/${variant}-${name}-${run}.json`), sample);
  console.log(JSON.stringify({ ...sample, result: undefined }));
  await model.resetCaches();
}

async function llama(name: string, run: string) {
  const data = input(name);
  const n = data.promptTokens;
  const port = 19090;
  const args = [
    "-m",
    modelFile,
    "-ngl",
    "999",
    "-fa",
    "on",
    "-ctk",
    "bf16",
    "-ctv",
    "bf16",
    "-c",
    "73728",
    "-b",
    "2048",
    "-ub",
    "512",
    "-t",
    "6",
    "-np",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--no-webui",
    "--no-context-shift",
  ];
  const fd = openSync(
    resolve(dataDir, `raw/llama-${name}-${run}.stderr.txt`),
    "w",
  );
  const started = performance.now();
  const server = spawn(llamaServer, args, { stdio: ["ignore", fd, fd], env });
  closeSync(fd);
  const exit = new Promise<void>((res) => server.once("exit", () => res()));
  let ready = false;
  try {
    for (let i = 0; i < 240; i++) {
      if (server.exitCode !== null)
        throw new Error(`llama-server exited: ${server.exitCode}`);
      try {
        if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) {
          ready = true;
          break;
        }
      } catch {}
      await sleep(250);
    }
    if (!ready) throw new Error("llama-server readiness timeout");
    const loadMs = performance.now() - started;
    async function post(path: string, body: unknown) {
      const r = await fetch(`http://127.0.0.1:${port}/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`${path}: ${await r.text()}`);
      return r.json();
    }
    const tokenized = await post("tokenize", {
      content: data.rendered,
      add_special: false,
      parse_special: true,
    });
    if (JSON.stringify(tokenized.tokens) !== JSON.stringify(data.tokenIds))
      throw new Error("Tokenizer mismatch");
    const request = {
      prompt: data.tokenIds,
      temperature: 0,
      top_k: 1,
      top_p: 1,
      min_p: 0,
      repeat_penalty: 1,
      presence_penalty: 0,
      frequency_penalty: 0,
      seed: 1234,
      cache_prompt: false,
      stream: false,
      return_tokens: true,
      ignore_eos: false,
    };
    await post("completion", {
      ...request,
      prompt: input(cases[0]).tokenIds,
      n_predict: 32,
    });
    const start = performance.now();
    const result = await post("completion", {
      ...request,
      n_predict: outputTokens,
    });
    const wallMs = performance.now() - start;
    const t = result.timings;
    if (
      t.prompt_n !== n ||
      t.predicted_n !== outputTokens ||
      t.cache_n !== 0 ||
      result.truncated
    ) {
      throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
    }
    const sample = {
      runtime: "llama",
      name,
      n,
      run,
      args,
      loadMs,
      wallMs,
      prefillMs: t.prompt_ms,
      prefillTps: t.prompt_per_second,
      decodeMs: t.predicted_ms,
      decodeTps: t.predicted_per_second,
      accountedMs: t.prompt_ms + t.predicted_ms,
      promptTokens: t.prompt_n,
      generatedTokens: t.predicted_n,
      cachedTokens: t.cache_n,
      textSha256: sha(result.content),
      inputSha256: data.sha256,
      result,
    };
    save(resolve(dataDir, `raw/llama-${name}-${run}.json`), sample);
    console.log(
      JSON.stringify({ ...sample, args: undefined, result: undefined }),
    );
  } finally {
    server.kill("SIGTERM");
    await Promise.race([exit, sleep(5000)]);
    if (server.exitCode === null) {
      server.kill("SIGKILL");
      await exit;
    }
  }
}

async function runAll() {
  for (let run = 1; run <= 3; run++) {
    for (const name of cases) {
      const order = run % 2 ? ["mlx", "llama"] : ["llama", "mlx"];
      for (const runtime of order) {
        console.log(new Date().toISOString(), "START", runtime, name, run);
        const child = spawn(
          "oxnode",
          [fileURLToPath(import.meta.url), runtime, name, String(run)],
          {
            env,
            stdio: "inherit",
          },
        );
        const code = await new Promise<number | null>((res) =>
          child.once("exit", res),
        );
        if (code !== 0)
          throw new Error(`${runtime} ${name} run ${run}: exit ${code}`);
        await sleep(cooldownSeconds * 1000);
      }
    }
  }
}

const [mode, n, run] = process.argv.slice(2);
mkdirSync(resolve(dataDir, "raw"), { recursive: true });
if (mode === "mlx") await mlx(n!, run!);
else if (mode === "llama") await llama(n!, run!);
else if (mode === "run") await runAll();
else
  throw new Error(
    "Usage: oxnode benchmark.ts run | mlx|llama <case-name> <run-id>",
  );
