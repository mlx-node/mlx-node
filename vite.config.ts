import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite-plus';

const __dirname = dirname(fileURLToPath(import.meta.url));

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
  resolve: {
    alias: {
      '@mlx-node/core': resolve(__dirname, './packages/core/index.cjs'),
      '@mlx-node/lm': resolve(__dirname, './packages/lm/src/index.ts'),
      '@mlx-node/agent/catalog': resolve(__dirname, './packages/agent/src/catalog.ts'),
      '@mlx-node/agent': resolve(__dirname, './packages/agent/src/index.ts'),
      '@mlx-node/privacy': resolve(__dirname, './packages/privacy/src/index.ts'),
      '@mlx-node/trl': resolve(__dirname, './packages/trl/src/index.ts'),
      '@mlx-node/server': resolve(__dirname, './packages/server/src/index.ts'),
      '@mlx-node/dashboard': resolve(__dirname, './packages/dashboard/src/index.ts'),
      // The dashboard SPA's own `@/…` alias (packages/dashboard/ui/vite.config.ts),
      // repeated here so a test can import a page component. Keyed with the
      // trailing slash so it can never swallow an `@mlx-node/…` specifier.
      '@/': `${resolve(__dirname, './packages/dashboard/ui/src')}/`,
    },
  },
});
