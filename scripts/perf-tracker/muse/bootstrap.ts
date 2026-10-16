#!/usr/bin/env oxnode
/// <reference types="node" />

// Every process that loads mlx-core starts here, so it binds the frozen addon:
//   bootstrap.ts prewarm <model.gguf>
//   bootstrap.ts prepare|worker <runner flags>   (scripts/benchmark-muse-gguf.ts, unchanged)
import { realpathSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { ROOT } from '../common.js';
import { RUNNER } from './suite.js';

const variant = process.env.MUSE_NATIVE_VARIANT;
if (!variant) throw new Error('MUSE_NATIVE_VARIANT must name the frozen addon');
if (process.env.NAPI_RS_NATIVE_LIBRARY_PATH) throw new Error('NAPI_RS_NATIVE_LIBRARY_PATH must not be inherited');
const frozen = realpathSync(variant);

// Set only around this import: at process start it would also redirect
// oxnode's own NAPI parser addon. Later imports of the same index.cjs, by the
// runner and by packages/lm, reuse this instance.
process.env.NAPI_RS_NATIVE_LIBRARY_PATH = frozen;
let core: { prepareMuseGlimmerGguf(path: string): Promise<string> };
try {
  core = await import(pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href);
} finally {
  delete process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
}

if (process.argv[2] === 'prewarm') {
  const model = process.argv[3];
  if (!model) throw new Error('Usage: bootstrap.ts prewarm <model.gguf>');
  const started = performance.now();
  const prepared = await core.prepareMuseGlimmerGguf(model);
  console.log(JSON.stringify({ prepared, elapsedMs: performance.now() - started }));
} else {
  await import(pathToFileURL(RUNNER).href);
}

// Checked after the runner finishes, so it cannot touch the timed region.
// Exactly one mlx-core binding may be loaded: the frozen copy. common's
// assertLoadedAddon matches only the release file names, not current.node.
const { sharedObjects } = process.report.getReport() as unknown as { sharedObjects: string[] };
const bindings = sharedObjects
  .filter((path) => path.endsWith('.node'))
  .map((path) => realpathSync(path))
  .filter((path) => path === frozen || /\/mlx-core\.[^/]*\.node$/.test(path));
if (bindings.length !== 1 || bindings[0] !== frozen) {
  throw new Error(`Expected only the frozen addon ${frozen}; loaded: ${bindings.join(', ') || 'none'}`);
}
