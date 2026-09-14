import { execFile } from 'node:child_process';
import { appendFile } from 'node:fs/promises';
import { promisify } from 'node:util';

const exec = promisify(execFile);

export function physicalFootprint(text: string): number | undefined {
  const match = text.match(/^Physical footprint(?: \(peak\))?:\s*([\d.]+)([KMGT]?)\b/gm);
  if (!match) return undefined;
  return Math.max(
    ...match.map((line) => {
      const [, size, unit] = line.match(/:\s*([\d.]+)([KMGT]?)\b/)!;
      return Number(size) * 1024 ** ' KMGT'.indexOf(unit || ' ');
    }),
  );
}

/** Out-of-process observations only. Never import the addon or alter child settings. */
export function observeProcess(pid: number, path: string) {
  const summary = { peakRssBytes: 0, peakPhysicalBytes: undefined as number | undefined };
  let phase = 'startup';
  let pending: Promise<void> | undefined;
  const sample = async (): Promise<void> => {
    const at = new Date().toISOString();
    const label = phase;
    try {
      const { stdout } = await exec('ps', ['-o', 'rss=', '-p', String(pid)], { timeout: 4_000 });
      const rssBytes = Number(stdout.trim()) * 1024;
      if (!Number.isFinite(rssBytes)) throw new Error('Invalid RSS measurement.');
      let physicalBytes: number | undefined;
      let physicalError: string | undefined;
      if (process.platform === 'darwin') {
        try {
          const vm = await exec('vmmap', ['-summary', String(pid)], { timeout: 4_000, maxBuffer: 2 * 1024 ** 2 });
          physicalBytes = physicalFootprint(vm.stdout);
        } catch (error) {
          physicalError = String(error);
        }
      }
      summary.peakRssBytes = Math.max(summary.peakRssBytes, rssBytes);
      if (physicalBytes !== undefined)
        summary.peakPhysicalBytes = Math.max(summary.peakPhysicalBytes ?? 0, physicalBytes);
      await appendFile(path, JSON.stringify({ at, phase: label, pid, rssBytes, physicalBytes, physicalError }) + '\n');
    } catch (error) {
      await appendFile(path, JSON.stringify({ at, phase: label, error: String(error) }) + '\n');
    }
  };
  const tick = (): void => {
    if (!pending)
      pending = sample().finally(() => {
        pending = undefined;
      });
  };
  tick();
  const timer = setInterval(tick, 5_000);
  timer.unref();
  return {
    summary,
    phase(value: string) {
      phase = value;
    },
    async stop() {
      clearInterval(timer);
      await pending;
    },
  };
}
