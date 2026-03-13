import { setProfilingEnabled, isProfilingEnabled, getProfilingData, resetProfilingData } from '@mlx-node/core';
import { writeFile } from 'node:fs/promises';

const ENV_VAR = 'MLX_PROFILE_DECODE';
const envVarSet = !!process.env[ENV_VAR];
let exitHandlerRegistered = false;

// Auto-enable if env var set
if (envVarSet) {
  setProfilingEnabled(true);
  registerExitHandler();
}

/**
 * Enable profiling programmatically.
 *
 * When enabled, all subsequent model generate/chat calls will record
 * timing, memory, and throughput data. Call `disableProfiling()` to
 * stop recording and write the report.
 *
 * If `MLX_PROFILE_DECODE` env var is set, this is a no-op (env var
 * takes precedence).
 */
export function enableProfiling(): void {
  if (envVarSet) {
    console.warn(`Warning: env var ${ENV_VAR} is set, ignoring explicit profiling API calls`);
    return;
  }
  setProfilingEnabled(true);
  resetProfilingData();
  registerExitHandler();
}

/**
 * Disable profiling and write the collected data to a JSON file.
 *
 * Returns the path to the written file, or empty string if no data
 * was collected. If `MLX_PROFILE_DECODE` env var is set, this is a
 * no-op (env var controls the lifecycle).
 */
export async function disableProfiling(): Promise<string> {
  if (envVarSet) {
    console.warn(`Warning: env var ${ENV_VAR} is set, ignoring explicit profiling API calls`);
    return '';
  }
  setProfilingEnabled(false);
  return writeProfilingReport();
}

async function writeProfilingReport(): Promise<string> {
  const data = getProfilingData();
  if (data.generations.length === 0) return '';
  const path = `mlx-profile-${Date.now()}.json`;
  await writeFile(path, JSON.stringify(data, null, 2));
  console.info(`Profiling report written to ${path}`);
  return path;
}

function registerExitHandler(): void {
  if (exitHandlerRegistered) return;
  exitHandlerRegistered = true;
  process.on('beforeExit', async () => {
    if (isProfilingEnabled()) {
      setProfilingEnabled(false);
      await writeProfilingReport();
    }
  });
}
