/**
 * One color per model on every page: the slot follows the model's registration order, which is
 * persisted and append-only, so a filter, a pruned trace file or a deleted model never repaints the others.
 */

import type { ChartSeries } from '../components/arc-x/chart/types';

export interface ModelColor {
  /** 0-based registration order; -1 for a name that is not registered ('unknown', or past the registry cap). */
  ordinal: number;
  /** CSS color expression: 'var(--model-3)' or 'var(--model-other)'. */
  color: string;
  /** solid = ordinals 0-7, alt = 8-15 (same hue, second encoding), other = 16+. */
  pattern: 'solid' | 'alt' | 'other';
}

export const MODEL_COLORS_STORAGE_KEY = 'mlx-node:model-colors:v1';
const MAX_MODELS = 256;
const SLOTS = 8;
const UNKNOWN = 'unknown';
const OTHER: ModelColor = { ordinal: -1, color: 'var(--model-other)', pattern: 'other' };

const names: string[] = [];
const ordinals = new Map<string, number>();
let writeQueued = false;

function load(): void {
  try {
    const raw = globalThis.localStorage?.getItem(MODEL_COLORS_STORAGE_KEY);
    if (raw === null || raw === undefined) return;
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    for (const entry of parsed) {
      if (typeof entry !== 'string' || entry.length === 0 || entry === UNKNOWN || ordinals.has(entry)) continue;
      if (names.length >= MAX_MODELS) break;
      ordinals.set(entry, names.length);
      names.push(entry);
    }
  } catch {
    // Unreadable storage starts an empty registry.
  }
}

function persist(): void {
  if (writeQueued) return;
  writeQueued = true;
  queueMicrotask(() => {
    writeQueued = false;
    try {
      globalThis.localStorage?.setItem(MODEL_COLORS_STORAGE_KEY, JSON.stringify(names));
    } catch {
      // The registry still holds for this window.
    }
  });
}

load();

/** The grouping key. Same string the API groups by (`model_key` = modelId ?? model ?? 'unknown'). 'unknown' is always `other`. */
export function modelKey(name: string): string {
  return name.length === 0 ? UNKNOWN : name;
}

function register(name: string): boolean {
  const key = modelKey(name);
  if (key === UNKNOWN || ordinals.has(key) || names.length >= MAX_MODELS) return false;
  ordinals.set(key, names.length);
  names.push(key);
  return true;
}

/** Register names not seen before, in the order given. Idempotent. Safe to call during render. */
export function ensureModelColors(list: readonly string[]): void {
  let changed = false;
  for (const name of list) changed = register(name) || changed;
  if (changed) persist();
}

function colorOf(ordinal: number | undefined): ModelColor {
  if (ordinal === undefined || ordinal >= SLOTS * 2) return ordinal === undefined ? OTHER : { ...OTHER, ordinal };
  const slot = (ordinal % SLOTS) + 1;
  return { ordinal, color: `var(--model-${slot})`, pattern: ordinal < SLOTS ? 'solid' : 'alt' };
}

export function modelColor(name: string): ModelColor {
  ensureModelColors([name]);
  return colorOf(ordinals.get(modelKey(name)));
}

/** Series for charts: registers `names` (unseen ones sorted by code unit) and maps each to a ChartSeries. */
export function modelSeries(list: readonly string[], labels?: ReadonlyMap<string, string>): ChartSeries[] {
  const unseen: string[] = [];
  for (const name of list) {
    const key = modelKey(name);
    if (!ordinals.has(key) && !unseen.includes(key)) unseen.push(key);
  }
  unseen.sort();
  ensureModelColors(unseen);
  return list.map((name) => {
    const color = colorOf(ordinals.get(modelKey(name)));
    return {
      key: name,
      label: labels?.get(name) ?? name,
      color: color.color,
      pattern: color.pattern === 'alt' ? 'alt' : 'solid',
    };
  });
}

/** Palette command "Reset model colors" + tests. */
export function resetModelColors(): void {
  names.length = 0;
  ordinals.clear();
  try {
    globalThis.localStorage?.removeItem(MODEL_COLORS_STORAGE_KEY);
  } catch {
    // Nothing stored to remove.
  }
}
