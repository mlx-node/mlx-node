import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite-plus';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Condition that resolves a workspace package to its TypeScript source instead
 * of its published `dist` output. Each package declares it in its own `exports`
 * map, so the subpath list lives in exactly one place: the package that owns it.
 */
const SOURCE_CONDITION = '@mlx-node/source';

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
  // The test/Vitest environment resolves server-side, where `ssr.resolve`
  // replaces Vite's defaults rather than extending them — so the defaults are
  // repeated here on purpose. Dropping 'node'/'module' would change how
  // third-party packages resolve inside the test environment, and this is the
  // only key that reaches the test resolver: `resolve.conditions` configures the
  // client environment, which tests do not use.
  //
  // Vitest also mirrors this list onto its Node processes as real `--conditions`
  // flags, so the condition must stay out of anything Node loads as JavaScript:
  // a spawn that inherits those flags would resolve workspace packages to
  // TypeScript, which Node can only strip when the source happens to be
  // erasable. Spawns pass `execArgv: []` for that reason (see
  // `packages/dashboard/src/worker/client.ts`).
  ssr: {
    resolve: {
      conditions: ['module', 'node', 'development|production', SOURCE_CONDITION],
    },
  },
  resolve: {
    alias: {
      // Dashboard SPA's own `@/` alias (packages/dashboard/ui), repeated for tests; no package boundary to cross.
      '@/': `${resolve(__dirname, './packages/dashboard/ui/src')}/`,
    },
  },
});
