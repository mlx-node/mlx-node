/// <reference types="node" />

import { execFile, spawnSync } from 'node:child_process';
import { appendFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { promisify } from 'node:util';

import { hostSnapshot, sh, sleep } from '../common.js';
import type { Policy, Progress } from './suite.js';

const execFileAsync = promisify(execFile);
const IOREG = ['-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'];
const BUSY_TOOLS = ['/zig', '/rustc', '/clang', '/llama-bench'];
const LLM_COMMANDS = [
  'llama-server',
  'llama-bench',
  'llama-cli',
  'ollama',
  'mlx_lm',
  'vllm',
  'llmster',
  'lmstudio',
  'lm studio',
  'mlx-worker.ts',
  'mlx-node serve',
];
const LLM_FILES = ['/mlx/', 'libmlx.', 'mlx-core.', 'libggml', 'libllama', '.safetensors', '.gguf'];

function gpuStats(ioreg: string): Record<string, number> {
  return Object.fromEntries([...ioreg.matchAll(/"([^"\n]*Utilization %)"=(\d+)/g)].map((m) => [m[1], Number(m[2])]));
}

export type Snapshot = ReturnType<typeof snapshot>;

export function snapshot() {
  return {
    ...hostSnapshot(),
    gpu: gpuStats(sh('ioreg', IOREG)),
    memory: sh('vm_stat', []),
    processes: sh('ps', ['-axo', 'pid,ppid,pcpu,pmem,comm']),
  };
}

interface ProcessEntry {
  pid: number;
  ppid: number;
  cpu: number;
  executable: string;
  command: string;
}

const rows = (columns: string) =>
  sh('ps', ['-axo', columns])
    .split('\n')
    .map((line) => /^\s*(\d+)\s+(.+)$/.exec(line))
    .filter((match): match is RegExpExecArray => match !== null);

// `comm` is the executable path, spaces included; `command` is argv.
export function processTable(): ProcessEntry[] {
  const executables = new Map(rows('pid=,comm=').map((match) => [Number(match[1]), match[2]]));
  return rows('pid=,ppid=,pcpu=,command=')
    .map((match) => [Number(match[1]), /^(\d+)\s+([\d.]+)\s+(.+)$/.exec(match[2])] as const)
    .filter((row): row is [number, RegExpExecArray] => row[1] !== null)
    .map(([pid, match]) => ({
      pid,
      ppid: Number(match[1]),
      cpu: Number(match[2]),
      executable: executables.get(pid) ?? match[3].split(' ')[0],
      command: match[3],
    }));
}

// The executable decides, or for an interpreter its argv: a shell, editor or
// `tail` that only names a runtime is not one. An idle runtime still holds memory.
export function isLocalLlm({ executable, command }: ProcessEntry): boolean {
  const text = /^(node|python[\d.]*)$/i.test(basename(executable)) ? command : executable;
  return LLM_COMMANDS.some((marker) => text.toLowerCase().includes(marker));
}

// `oxnode` runs a script in a child node process, so the PID returned by
// spawn() is not the process that loads the model; match by descent.
export function descendants(table: readonly ProcessEntry[], root: number): Set<number> {
  const tree = new Set([root]);
  for (let grew = true; grew;) {
    grew = false;
    for (const row of table) {
      if (tree.has(row.ppid) && !tree.has(row.pid)) {
        tree.add(row.pid);
        grew = true;
      }
    }
  }
  return tree;
}

export function otherLocalLlms(): { pid: number; cpu: number; command: string }[] {
  const table = processTable();
  const own = descendants(table, process.pid);
  const active = [];
  for (const row of table) {
    const { pid, cpu, command } = row;
    if (own.has(pid)) continue;
    const lower = command.toLowerCase();
    if (isLocalLlm(row)) {
      active.push({ pid, cpu, command });
    } else if (cpu > 10 && (lower.includes('python') || lower.includes('/node '))) {
      const opened = spawnSync('lsof', ['-p', String(pid), '-Fn'], { encoding: 'utf8', maxBuffer: 256 << 20 });
      if (LLM_FILES.some((marker) => (opened.stdout ?? '').toLowerCase().includes(marker))) {
        active.push({ pid, cpu, command });
      }
    }
  }
  return active;
}

// Five consecutive readings 5 s apart: 20 s at <=5% GPU with no busy compiler.
async function gpuIdle(progress: Progress, label: string, seconds: number): Promise<Snapshot> {
  progress.emit({ event: 'cooldown', label, seconds, policy: 'gpu-idle' });
  await sleep(seconds * 1000);
  let quiet = 0;
  let lastNotice = -Infinity;
  for (;;) {
    const state = snapshot();
    const busy = state.processes
      .split('\n')
      .slice(1)
      .filter((line) => {
        const fields = line.trim().split(/\s+/);
        return fields.length >= 5 && Number(fields[2]) > 20 && BUSY_TOOLS.some((tool) => line.includes(tool));
      });
    const utilization = state.gpuUtilization;
    quiet = busy.length === 0 && utilization <= 5 ? quiet + 1 : 0;
    if (quiet >= 5) {
      progress.emit({ event: 'gpu_idle_confirmed', label, utilization, quietSeconds: 20 });
      return state;
    }
    if (performance.now() - lastNotice >= 30_000) {
      progress.emit({ event: 'waiting_for_gpu_idle', label, utilization, busyCompilers: busy.length });
      lastNotice = performance.now();
    }
    await sleep(5000);
  }
}

async function noOtherLocalLlm(progress: Progress, label: string, seconds: number): Promise<Snapshot> {
  const policy = 'no-other-local-llm';
  progress.emit({ event: 'cooldown', label, seconds, policy });
  await sleep(seconds * 1000);
  for (;;) {
    const active = otherLocalLlms();
    if (!active.length) {
      const state = snapshot();
      progress.emit({ event: 'llm_isolation_confirmed', label, utilization: state.gpuUtilization, policy });
      return state;
    }
    progress.emit({ event: 'waiting_for_other_llm', label, active });
    await sleep(10_000);
  }
}

export function admit(policy: Policy, progress: Progress, label: string, seconds: number): Promise<Snapshot> {
  return policy === 'gpu-idle' ? gpuIdle(progress, label, seconds) : noOtherLocalLlm(progress, label, seconds);
}

// Asynchronous so a reading never stalls the in-process llama.cpp stream reader.
export function recordGpu(path: string, phase: () => unknown, intervalMs = 10_000): () => Promise<void> {
  let pending = Promise.resolve();
  const tick = () => {
    pending = pending.then(async () => {
      let line: Record<string, unknown>;
      try {
        line = { gpu: gpuStats((await execFileAsync('ioreg', IOREG)).stdout) };
      } catch (error) {
        line = { error: error instanceof Error ? error.message : String(error) };
      }
      await appendFile(path, `${JSON.stringify({ timestamp: new Date().toISOString(), ...line, phase: phase() })}\n`);
    });
  };
  tick();
  const timer = setInterval(tick, intervalMs);
  return async () => {
    clearInterval(timer);
    await pending;
  };
}
