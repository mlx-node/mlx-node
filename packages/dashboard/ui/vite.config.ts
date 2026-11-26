import { resolve } from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite-plus';

import { pdfjsAssets } from './pdfjs-assets';

/** markstream's optional peers, loaded by bare dynamic imports; subpaths first, the first match wins. */
const MARKSTREAM_OFF = resolve(import.meta.dirname, 'src/lib/markstream-off.ts');

// No dev-server `/api` proxy: the SPA reaches the runtime over a MessagePort
// handed to it by the desktop app's preload, not over HTTP. There is no port to
// proxy to, and `vite dev` alone therefore has no backend.
export default defineConfig({
  plugins: [react(), tailwindcss(), pdfjsAssets()],
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, 'src'),
      'katex/contrib/mhchem': MARKSTREAM_OFF,
      katex: MARKSTREAM_OFF,
      mermaid: MARKSTREAM_OFF,
      'stream-diffs/markstream': MARKSTREAM_OFF,
      '@terrastruct/d2': MARKSTREAM_OFF,
      '@antv/infographic': MARKSTREAM_OFF,
    },
  },
  build: {
    outDir: '../web',
    emptyOutDir: true,
    target: 'chrome140',
    cssTarget: 'chrome140',
    // markstream loads its language icons and optional peers with `import()`; the SPA ships as one chunk.
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
