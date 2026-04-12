import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite-plus';
import { playwright } from 'vite-plus/test/browser-playwright';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: __dirname,
  test: {
    name: 'browser',
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: {
          channel: 'chrome',
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
          ],
        },
      }),
      headless: true,
      instances: [{ browser: 'chromium' }],
    },
    fileParallelism: false,
    isolate: false,
    testTimeout: 120_000,
    include: ['src/__tests__/**/*.test.ts'],
    globals: true,
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  plugins: [
    {
      name: 'coep-headers',
      enforce: 'pre' as const,
      configureServer: (server) => {
        server.middlewares.use((_req, res, next) => {
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          next();
        });
      },
    },
  ],
  publicDir: resolve(__dirname, 'demo/public'),
  optimizeDeps: {
    exclude: ['@napi-rs/wasm-runtime'],
  },
});
