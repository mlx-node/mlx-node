/**
 * `ROOT/work-settings.json` (0600): the app-wide Work settings. `claudeCode`: Work loads the user's Claude Code setup,
 * as `mlx agent` in a terminal does (on by default). A missing or unreadable file reads as the defaults.
 */

import { readFileSync } from 'node:fs';
import { chmod, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

import { writeFileAtomic } from './paths.js';

export interface WorkSettings {
  claudeCode: boolean;
}

export function defaultWorkSettings(): WorkSettings {
  return { claudeCode: true };
}

/** Only `version: 1` with a boolean `claudeCode` is read; anything else is the defaults. */
export function parseWorkSettings(text: string): WorkSettings {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return defaultWorkSettings();
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return defaultWorkSettings();
  const r = raw as Record<string, unknown>;
  if (r.version !== 1 || typeof r.claudeCode !== 'boolean') return defaultWorkSettings();
  return { claudeCode: r.claudeCode };
}

export function readWorkSettingsSync(path: string): WorkSettings {
  try {
    return parseWorkSettings(readFileSync(path, 'utf8'));
  } catch {
    return defaultWorkSettings();
  }
}

export async function writeWorkSettings(path: string, settings: WorkSettings): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await chmod(dirname(path), 0o700);
  await writeFileAtomic(path, `${JSON.stringify({ version: 1, claudeCode: settings.claudeCode })}\n`);
}
