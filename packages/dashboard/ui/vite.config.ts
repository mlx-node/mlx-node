import { resolve } from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// No dev-server `/api` proxy: the SPA reaches the runtime over a MessagePort
// handed to it by the desktop app's preload, not over HTTP. There is no port to
// proxy to, and `vite dev` alone therefore has no backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': resolve(import.meta.dirname, 'src') } },
  build: { outDir: '../web', emptyOutDir: true },
});
