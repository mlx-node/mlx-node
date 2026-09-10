/// <reference types="node" />
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const dataDir = resolve(root, ".cache/benchmarks/gemma4-path-audit-2026-09-10");
const inputs = JSON.parse(
  readFileSync(
    resolve(root, ".cache/benchmarks/gemma4-agent-2026-09-10/inputs.json"),
    "utf8",
  ),
);
const hash = (value: string) =>
  createHash("sha256").update(value).digest("hex");
const [mode, variant = "baseline", name = "diff-review", run = "1"] =
  process.argv.slice(2);

async function worker(warmRepeats = false) {
  const core = await import(resolve(root, "packages/core/index.cjs"));
  const data = inputs.find((x: any) => x.name === name);
  if (!data) throw new Error(`Unknown fixture ${name}`);
  const t0 = performance.now();
  const model = await core.Gemma4Model.load(resolve(dataDir, variant));
  const loadMs = performance.now() - t0;
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
  await model.chatSessionStart(inputs[0].messages, {
    ...config,
    maxNewTokens: 32,
  });
  await model.resetCaches();
  core.setProfilingEnabled(true);
  for (
    let sampleIndex = 0;
    sampleIndex < (warmRepeats ? 4 : 1);
    sampleIndex++
  ) {
    core.resetProfilingData();
    const start = performance.now();
    const result = await model.chatSessionStart(data.messages, {
      ...config,
      maxNewTokens: 256,
    });
    const wallMs = performance.now() - start;
    if (
      result.promptTokens !== data.promptTokens ||
      (sampleIndex === 0 && result.cachedTokens !== 0) ||
      result.numTokens !== 256
    )
      throw new Error("Workload count mismatch");
    const label = sampleIndex === 0 ? run : `${run}-reuse${sampleIndex}`;
    const sample = {
      variant,
      name,
      run: label,
      cachePolicy:
        sampleIndex === 0
          ? "cold"
          : "replay identical full history with prefix reuse",
      promptTokens: data.promptTokens,
      generatedTokens: result.numTokens,
      cachedTokens: result.cachedTokens,
      loadMs,
      wallMs,
      prefillMs: result.performance.ttftMs,
      prefillTps: result.performance.prefillTokensPerSecond,
      decodeTps: result.performance.decodeTokensPerSecond,
      outputSha256: hash(result.rawText),
      inputSha256: data.sha256,
      env: Object.fromEntries(
        Object.entries(process.env).filter(([k]) => k.startsWith("MLX_")),
      ),
      profiles: core.getProfilingData(),
    };
    writeFileSync(
      resolve(dataDir, `raw/${variant}-${name}-${label}.json`),
      JSON.stringify(sample, null, 2) + "\n",
    );
    console.log(JSON.stringify({ ...sample, profiles: undefined }));
    if (sampleIndex > 0 && result.cachedTokens < data.promptTokens / 2)
      throw new Error("Insufficient prefix reuse for warm decode screen");
    if (warmRepeats && sampleIndex < 3) await sleep(20_000);
  }
  await model.resetCaches();
}

if (mode === "worker") await worker();
else if (mode === "warm-worker") await worker(true);
else if (mode === "pairs") {
  for (let repeat = 1; repeat <= 3; repeat++) {
    for (const arm of repeat % 2
      ? ["baseline", "hoisted"]
      : ["hoisted", "baseline"]) {
      console.log(new Date().toISOString(), "START", arm, name, repeat);
      const child = spawn(
        "oxnode",
        [fileURLToPath(import.meta.url), "worker", arm, name, String(repeat)],
        {
          stdio: "inherit",
          env: {
            ...process.env,
            MLX_AGENT_METRICS: "0",
            MLX_PERSIST_PAGED_CACHE: "0",
            MLX_PAGED_PREFILL_CHUNK_SIZE: "512",
            MLX_NODE_LOG: "info",
          },
        },
      );
      const code = await new Promise<number | null>((res) =>
        child.once("exit", res),
      );
      if (code !== 0) throw new Error(`Worker exited ${code}`);
      await sleep(20_000);
    }
  }
} else if (mode === "routes") {
  for (const route of ["sdpa", "grouped"]) {
    console.log(new Date().toISOString(), "START", route, "final-review");
    const child = spawn(
      "oxnode",
      [
        fileURLToPath(import.meta.url),
        "worker",
        "hoisted",
        "final-review",
        `${route}-long`,
      ],
      {
        stdio: "inherit",
        env: {
          ...process.env,
          MLX_AGENT_METRICS: "0",
          MLX_PERSIST_PAGED_CACHE: "0",
          MLX_PAGED_PREFILL_CHUNK_SIZE: "512",
          MLX_NODE_LOG: "info",
          MLX_GEMMA4_PAGED_DECODE_ROUTE: route === "sdpa" ? "sdpa" : "auto",
          MLX_PAGED_GROUPED_D512: route === "grouped" ? "force" : "auto",
        },
      },
    );
    const code = await new Promise<number | null>((res) =>
      child.once("exit", res),
    );
    if (code !== 0) throw new Error(`Route worker exited ${code}`);
    await sleep(20_000);
  }
} else if (mode === "reverse-routes") {
  for (const route of ["grouped", "auto"]) {
    console.log(
      new Date().toISOString(),
      "START",
      route,
      "final-review",
      "reverse",
    );
    const child = spawn(
      "oxnode",
      [
        fileURLToPath(import.meta.url),
        "worker",
        "hoisted",
        "final-review",
        `${route}-long-2`,
      ],
      {
        stdio: "inherit",
        env: {
          ...process.env,
          MLX_AGENT_METRICS: "0",
          MLX_PERSIST_PAGED_CACHE: "0",
          MLX_PAGED_PREFILL_CHUNK_SIZE: "512",
          MLX_NODE_LOG: "info",
          MLX_GEMMA4_PAGED_DECODE_ROUTE: "auto",
          MLX_PAGED_GROUPED_D512: route === "grouped" ? "force" : "auto",
        },
      },
    );
    const code = await new Promise<number | null>((res) =>
      child.once("exit", res),
    );
    if (code !== 0) throw new Error(`Reverse route worker exited ${code}`);
    await sleep(20_000);
  }
} else
  throw new Error(
    "Usage: probe.ts worker <baseline|hoisted> <fixture> <label> | pairs unused <fixture> | routes",
  );
