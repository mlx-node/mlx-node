import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import { defineConfig, type Plugin } from "vite";
import { voidPlugin } from "void";

const dirname = path.dirname(fileURLToPath(import.meta.url));

function browserDeployAssets(): Plugin {
  return {
    name: "browser-deploy-assets",
    apply: "build" as const,
    buildStart() {
      const assets = [
        ["_headers", path.resolve(dirname, "demo/public/_headers")],
        ["capybara.png", path.resolve(dirname, "demo/public/capybara.png")],
        ["og-image.png", path.resolve(dirname, "demo/public/og-image.png")],
        ["og-image.svg", path.resolve(dirname, "demo/public/og-image.svg")],
        [
          "mlx-core.opt.wasm",
          path.resolve(dirname, "../core/mlx-core.wasm32-wasi.opt.wasm"),
        ],
      ] as const;

      for (const [fileName, filePath] of assets) {
        if (!fs.existsSync(filePath)) {
          throw new Error(
            `Missing browser deploy asset ${fileName}: ${filePath}. Run the WASM build before building the browser playground.`,
          );
        }

        this.emitFile({
          type: "asset",
          fileName,
          source: fs.readFileSync(filePath),
        });
      }
    },
  };
}

export default defineConfig(({ command }) => ({
  root: "demo",
  publicDir: command === "build" ? false : "public",
  resolve: {
    alias: {
      "@mlx-node/lm/tools": path.resolve(dirname, "../lm/src/tools/index.ts"),
      "@": path.resolve(dirname, "demo"),
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  environments: {
    void_worker: {
      build: {
        outDir: "../dist/ssr",
      },
    },
  },
  plugins: [
    tanstackRouter({
      target: "react",
      autoCodeSplitting: false,
      routesDirectory: path.resolve(dirname, "demo/routes"),
      generatedRouteTree: path.resolve(dirname, "demo/routeTree.gen.ts"),
    }),
    voidPlugin(),
    react(),
    tailwindcss(),
    browserDeployAssets(),
    {
      name: "configure-response-headers",
      enforce: "pre",
      configureServer: (server) => {
        server.middlewares.use((_req, res, next) => {
          res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
          res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
          next();
        });
      },
    },
  ],
  optimizeDeps: {
    exclude: ["@napi-rs/wasm-runtime"],
  },
}));
