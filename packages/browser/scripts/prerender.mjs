// scripts/prerender.mjs — postbuild prerender orchestrator (plain ESM, run by
// plain `node`, e.g. `node scripts/prerender.mjs`).
//
// 1. Verifies a built dist/client/index.html exists (run `vite build` first).
// 2. esbuild-bundles scripts/prerender.entry.tsx -> dist/.prerender/entry.cjs,
//    stubbing all asset imports and aliasing @ and @mlx-node/lm/tools to match
//    vite.config.ts. react/react-dom are external (resolved by node at runtime).
// 3. Runs the bundle in a clean node subprocess, which does the actual SSR and
//    writes dist/client/chapters/<id>/index.html, the hub, the rewritten
//    landing head, sitemap.xml and robots.txt.
//
// Wired into package.json `build` after `vite build`, so deploy:void (which
// calls `yarn build`) picks it up automatically.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, '..');
const distClient = path.join(packageRoot, 'dist', 'client');
const shellPath = path.join(distClient, 'index.html');

if (!fs.existsSync(shellPath)) {
  console.error(`[prerender] dist/client/index.html not found at ${shellPath}.`);
  console.error('[prerender] Run `vite build` first (or `yarn build`, which runs it before this step).');
  process.exit(1);
}

const entry = path.join(scriptDir, 'prerender.entry.tsx');
const prerenderDir = path.join(packageRoot, 'dist', '.prerender');
const outfile = path.join(prerenderDir, 'entry.cjs');
fs.mkdirSync(prerenderDir, { recursive: true });

await build({
  entryPoints: [entry],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node18',
  jsx: 'automatic',
  logLevel: 'warning',
  outfile,
  absWorkingDir: packageRoot,
  // Resolved by node at runtime from node_modules — keep them out of the bundle.
  external: ['react', 'react-dom', 'react-dom/server', 'react/jsx-runtime', 'react/jsx-dev-runtime'],
  alias: {
    '@': path.join(packageRoot, 'demo'),
    '@mlx-node/lm/tools': path.join(packageRoot, '..', 'lm', 'src', 'tools', 'index.ts'),
  },
  loader: {
    '.css': 'empty',
    '.png': 'empty',
    '.jpg': 'empty',
    '.jpeg': 'empty',
    '.gif': 'empty',
    '.svg': 'empty',
    '.woff': 'empty',
    '.woff2': 'empty',
    '.ttf': 'empty',
    '.glsl': 'text',
    '.wgsl': 'text',
    '.glb': 'empty',
    '.bin': 'empty',
    '.mp4': 'empty',
    '.webm': 'empty',
  },
});

execFileSync(process.execPath, [outfile], {
  stdio: 'inherit',
  cwd: packageRoot,
  env: { ...process.env, PRERENDER_DIST_CLIENT: distClient },
});
