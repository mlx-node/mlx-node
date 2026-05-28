import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import tailwindcss from '@tailwindcss/vite';
import { tanstackRouter } from '@tanstack/router-plugin/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';
import { voidPlugin } from 'void';

const dirname = path.dirname(fileURLToPath(import.meta.url));

function browserDeployAssets(): Plugin {
  return {
    name: 'browser-deploy-assets',
    apply: 'build' as const,
    buildStart() {
      const assets = [
        ['_headers', path.resolve(dirname, 'demo/public/_headers')],
        ['capybara.png', path.resolve(dirname, 'demo/public/capybara.png')],
        ['og-image.png', path.resolve(dirname, 'demo/public/og-image.png')],
        ['og-image.svg', path.resolve(dirname, 'demo/public/og-image.svg')],
        ['mlx-core.opt.wasm', path.resolve(dirname, '../core/mlx-core.wasm32-wasi.opt.wasm')],
      ] as const;

      for (const [fileName, filePath] of assets) {
        if (!fs.existsSync(filePath)) {
          throw new Error(
            `Missing browser deploy asset ${fileName}: ${filePath}. Run the WASM build before building the browser playground.`,
          );
        }

        this.emitFile({
          type: 'asset',
          fileName,
          source: fs.readFileSync(filePath),
        });
      }
    },
  };
}

// SharedArrayBuffer (required by the MLX worker for both wasm threads and the
// chat/inspector SAB ring) is only enabled when the document is served
// cross-origin-isolated. That means every response must carry COOP, COEP, and
// CORP — production sets these via void.json's `routing.headers`; locally we
// need them on both the dev and preview servers so `yarn dev` / `yarn preview`
// can actually load the model. Without CORP, sibling responses (e.g. the
// chunk-split worker bundle) get blocked by the COEP check even though the
// document itself is isolated.
const CROSS_ORIGIN_ISOLATION_HEADERS: Record<string, string> = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'cross-origin',
};

export default defineConfig(({ command }) => ({
  root: 'demo',
  publicDir: command === 'build' ? false : 'public',
  resolve: {
    alias: {
      '@mlx-node/lm/tools': path.resolve(dirname, '../lm/src/tools/index.ts'),
      '@': path.resolve(dirname, 'demo'),
    },
  },
  server: {
    headers: CROSS_ORIGIN_ISOLATION_HEADERS,
  },
  preview: {
    headers: CROSS_ORIGIN_ISOLATION_HEADERS,
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  environments: {
    void_worker: {
      build: {
        outDir: '../dist/ssr',
      },
    },
  },
  plugins: [
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: false,
      routesDirectory: path.resolve(dirname, 'demo/routes'),
      generatedRouteTree: path.resolve(dirname, 'demo/routeTree.gen.ts'),
    }),
    voidPlugin(),
    react(),
    tailwindcss(),
    browserDeployAssets(),
    {
      // Belt-and-suspenders: `server.headers` / `preview.headers` cover most
      // responses, but some intermediate Vite/void plugins write the response
      // headers manually and bypass that map. A middleware that fires before
      // anything else sets the three headers on every response — including
      // chunk-split worker URLs, the HMR client, and the SSR worker preview —
      // so the cross-origin-isolation check stays green across the board.
      name: 'configure-response-headers',
      enforce: 'pre',
      configureServer: (server) => {
        server.middlewares.use((_req, res, next) => {
          for (const [k, v] of Object.entries(CROSS_ORIGIN_ISOLATION_HEADERS)) {
            res.setHeader(k, v);
          }
          next();
        });
      },
      configurePreviewServer: (server) => {
        server.middlewares.use((_req, res, next) => {
          for (const [k, v] of Object.entries(CROSS_ORIGIN_ISOLATION_HEADERS)) {
            res.setHeader(k, v);
          }
          next();
        });
      },
    },
  ],
  optimizeDeps: {
    exclude: ['@napi-rs/wasm-runtime'],
  },
}));
