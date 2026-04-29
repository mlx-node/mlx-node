import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

import { chromium } from "playwright";

export const WEBGPU_CHROME_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,SharedArrayBuffer",
  "--disable-dawn-features=disallow_unsafe_apis",
];

async function waitForExit(proc, timeoutMs) {
  if (!proc || proc.exitCode !== null || proc.killed) return;
  await Promise.race([
    once(proc, "exit"),
    delay(timeoutMs).then(() => {
      throw new Error(`Chrome did not exit within ${timeoutMs}ms`);
    }),
  ]);
}

export async function launchWebGPUContext({
  headed = false,
  channel = process.env.PLAYWRIGHT_CHROME_CHANNEL || undefined,
  closeTimeoutMs = 5_000,
} = {}) {
  const userDataDir = await mkdtemp(join(tmpdir(), "mlx-webgpu-playwright-"));
  let context;
  let browserProcess;
  let cleaned = false;

  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      headless: !headed,
      channel,
      args: WEBGPU_CHROME_ARGS,
    });
    browserProcess = context.browser()?.process?.() ?? null;
  } catch (error) {
    await rm(userDataDir, { recursive: true, force: true });
    throw error;
  }

  async function cleanup() {
    if (cleaned) return;
    cleaned = true;

    if (context) {
      await Promise.race([
        context.close(),
        delay(closeTimeoutMs).then(() => {
          throw new Error(`context.close timed out after ${closeTimeoutMs}ms`);
        }),
      ]).catch((error) => {
        console.warn(`[cleanup] Chrome close failed: ${error.message}`);
      });
    }

    if (
      browserProcess &&
      browserProcess.exitCode === null &&
      !browserProcess.killed
    ) {
      browserProcess.kill("SIGTERM");
      await waitForExit(browserProcess, closeTimeoutMs).catch(() => {
        browserProcess.kill("SIGKILL");
      });
    }

    await rm(userDataDir, { recursive: true, force: true }).catch((error) => {
      console.warn(
        `[cleanup] Failed to remove ${userDataDir}: ${error.message}`,
      );
    });
  }

  return { context, cleanup, userDataDir };
}

export async function withWebGPUPage(options, fn) {
  const session = await launchWebGPUContext(options);
  const page = await session.context.newPage();

  const exitAfterCleanup = (exitCode) => {
    session.cleanup().finally(() => {
      process.exitCode = exitCode;
      process.exit();
    });
  };
  const handleSigint = () => exitAfterCleanup(130);
  const handleSigterm = () => exitAfterCleanup(143);

  process.once("SIGINT", handleSigint);
  process.once("SIGTERM", handleSigterm);

  try {
    return await fn(page, session.context, session);
  } finally {
    process.off("SIGINT", handleSigint);
    process.off("SIGTERM", handleSigterm);
    await session.cleanup();
  }
}
