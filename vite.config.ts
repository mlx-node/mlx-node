import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Plugin } from 'vite';
import { defineConfig } from 'vite-plus';

const __dirname = dirname(fileURLToPath(import.meta.url));

/** Condition a workspace package declares for its own TypeScript source. */
const SOURCE_CONDITION = '@mlx-node/source';

/**
 * Resolve `@mlx-node/*` specifiers to the TypeScript source each package declares,
 * so a dev server and the test run exercise `src` while `tsc`, `oxnode` and
 * published consumers keep reading `dist`.
 *
 * The mapping is not a list: it is read from each package's own `exports` map,
 * where a subpath and its `@mlx-node/source` target sit together, so a new subpath
 * needs no change here. That is what replaced the hand-maintained
 * `resolve.alias` table, whose subpath list was a second copy of `exports` that
 * nothing validated.
 *
 * Deliberately not `resolve.conditions` / `ssr.resolve.conditions`: Vitest mirrors
 * those onto its Node processes as real `--conditions` flags, which makes Node
 * itself resolve workspace packages to TypeScript. Node can only run the
 * TypeScript it can strip, so every Node-side load dies on the first parameter
 * property (`packages/server/src/host/index.ts` has one) — measured on a worker
 * thread and on the forked desktop sidecar. Keeping the mapping inside Vite
 * leaves Node, `oxnode` and the published map untouched.
 */
function workspaceSource(): Plugin {
  const packages = new Map<string, { dir: string; exports: Record<string, string | Record<string, string>> }>();
  const packagesRoot = resolve(__dirname, 'packages');

  for (const entry of readdirSync(packagesRoot)) {
    const manifestPath = join(packagesRoot, entry, 'package.json');
    if (!existsSync(manifestPath)) continue;
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8')) as {
      name?: string;
      exports?: Record<string, string | Record<string, string>>;
    };
    if (manifest.name === undefined || manifest.exports === undefined) continue;
    packages.set(manifest.name, { dir: dirname(manifestPath), exports: manifest.exports });
  }

  return {
    name: 'mlx-node:workspace-source',
    enforce: 'pre',
    resolveId(source) {
      const parts = source.split('/');
      if (parts.length < 2 || parts[0] !== '@mlx-node') return null;
      const pkg = packages.get(`${parts[0]}/${parts[1]}`);
      if (pkg === undefined) return null;
      const entry = pkg.exports[parts.length > 2 ? `./${parts.slice(2).join('/')}` : '.'];
      const target = typeof entry === 'string' ? undefined : entry?.[SOURCE_CONDITION];
      return target === undefined ? null : resolve(pkg.dir, target);
    },
    // The map is built once, at config load, and these manifests are not config
    // files: Vite restarts on a `vite.config.ts` edit but not on a package.json
    // edit, so a watch session would keep resolving a subpath that moved, lost its
    // source entry, or fall back to `dist` for one that is new.
    configureServer(server) {
      const restart = (file: string): void => {
        if (file.startsWith(`${packagesRoot}/`) && file.endsWith('package.json')) void server.restart();
      };
      for (const event of ['add', 'change', 'unlink'] as const) server.watcher.on(event, restart);
    },
  };
}

export default defineConfig({
  fmt: {
    printWidth: 120,
    tabWidth: 2,
    singleQuote: true,
    sortPackageJson: true,
    sortImports: {
      groups: [
        ['type-import'],
        ['type-builtin', 'value-builtin'],
        ['type-external', 'value-external', 'type-internal', 'value-internal'],
        ['type-parent', 'type-sibling', 'type-index', 'value-parent', 'value-sibling', 'value-index'],
        ['unknown'],
      ],
      newlinesBetween: true,
      order: 'asc',
    },
    ignorePatterns: [
      '**/dist/**',
      'packages/dashboard/web/**',
      '**/tests/**',
      '**/generated/**',
      '**/fixtures/**',
      '.yarn/**',
      'index.d.cts',
      'index.cjs',
      '/trl',
      '/transformers',
      '/mlx-lm',
      '/mlx-rs',
      '/crates/mlx-sys/mlx',
      '**/*.metal.inc',
    ],
  },
  lint: {
    options: {
      typeAware: true,
      typeCheck: true,
    },
  },
  staged: {
    '*': ['vp check --fix'],
    '*.rs': ['cargo fmt --all --'],
  },
  test: {
    globals: true,
    environment: 'node',
    // Tests must never write inference telemetry to the developer's real
    // ~/.mlx-node. metrics-trace.test.ts manages this var itself (temp dirs).
    env: { MLX_AGENT_METRICS: '0' },
    maxConcurrency: 1,
    watch: false,
    testTimeout: 120000, // 2 minutes
    maxWorkers: 1,
    include: [
      '__test__/**/*.{test,spec}.ts',
      'examples/**/*.{test,spec}.ts',
      'packages/*/__test__/**/*.{test,spec}.ts',
    ],
  },
  plugins: [workspaceSource()],
  resolve: {
    alias: {
      // Dashboard SPA's own `@/` alias (packages/dashboard/ui), repeated for tests; no package boundary to cross.
      '@/': `${resolve(__dirname, './packages/dashboard/ui/src')}/`,
    },
  },
});
