import { defineConfig, type ViteUserConfig } from 'vite-plus';
import { resolve } from 'node:path';

const regularTestConfig: ViteUserConfig['test'] = {
  globals: true,
  environment: 'node',
  watch: false,
  testTimeout: 120000, // 2 minutes
  include: ['__test__/**/*.{test,spec}.ts', 'examples/**/*.{test,spec}.ts'],
  exclude: ['**/node_modules/**', '**/.git/**', '**/trainers/**', '**/__test__/dist/**', '**/target/**'],
};

// Run trainer tests sequentially (single worker) due to GPU memory constraints
const trainerTestConfig: ViteUserConfig['test'] = {
  ...regularTestConfig,
  maxWorkers: 1,
  exclude: ['**/node_modules/**', '**/.git/**', '**/__test__/dist/**', '**/target/**'],
  include: ['__test__/trainers/*.{test,spec}.ts'],
};

export default defineConfig({
  fmt: {
    printWidth: 120,
    tabWidth: 2,
    singleQuote: true,
    ignorePatterns: [
      '**/dist/**',
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
    ],
  },
  test: process.env.TEST_TRAINER ? trainerTestConfig : regularTestConfig,
  resolve: {
    alias: {
      '@mlx-node/core': resolve(__dirname, './packages/core/index.cjs'),
      '@mlx-node/lm': resolve(__dirname, './packages/lm/src/index.ts'),
      '@mlx-node/trl': resolve(__dirname, './packages/trl/src/index.ts'),
    },
  },
});
