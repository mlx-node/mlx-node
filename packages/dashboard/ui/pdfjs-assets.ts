/**
 * The pdf.js files the page loads by URL at run time: the worker, the JS decoders it imports in place of WASM, the
 * CMaps and the standard fonts, with their licences. Vite does not bundle them, so they ship as plain copies under
 * `assets/pdfjs-<version>/`: there a missing file is a 404 (anywhere else the app answers `index.html` with 200), and
 * the version in the folder name keeps the year-long cache of `/assets/` true.
 */

import { copyFileSync, createReadStream, mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';

import type { Plugin } from 'vite-plus';

export const PDFJS_WORKER = 'pdf.worker.min.mjs';

const SINGLE_FILES: ReadonlyArray<readonly [published: string, source: string]> = [
  [PDFJS_WORKER, 'build/pdf.worker.min.mjs'],
  ['wasm/jbig2_nowasm_fallback.js', 'wasm/jbig2_nowasm_fallback.js'],
  ['wasm/openjpeg_nowasm_fallback.js', 'wasm/openjpeg_nowasm_fallback.js'],
  ['LICENSE', 'LICENSE'],
  ['wasm/LICENSE_JBIG2', 'wasm/LICENSE_JBIG2'],
  ['wasm/LICENSE_OPENJPEG', 'wasm/LICENSE_OPENJPEG'],
  ['wasm/LICENSE_PDFJS_JBIG2', 'wasm/LICENSE_PDFJS_JBIG2'],
  ['wasm/LICENSE_PDFJS_OPENJPEG', 'wasm/LICENSE_PDFJS_OPENJPEG'],
];
const WHOLE_DIRS = ['cmaps', 'standard_fonts'] as const;

/** The pdfjs-dist this package resolves, the one the page bundles. */
export function pdfjsRoot(): string {
  return dirname(createRequire(import.meta.url).resolve('pdfjs-dist/package.json'));
}

/** `assets/pdfjs-<version>`, the folder under the web root. */
export function pdfjsBase(root: string = pdfjsRoot()): string {
  const { version } = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as { version: string };
  return `assets/pdfjs-${version}`;
}

/** Each file as `[path under the folder, path inside the package]`. */
export function pdfjsFiles(root: string = pdfjsRoot()): Array<readonly [string, string]> {
  const out: Array<readonly [string, string]> = [...SINGLE_FILES];
  for (const dir of WHOLE_DIRS) {
    for (const entry of readdirSync(join(root, dir), { withFileTypes: true })) {
      if (entry.isFile()) out.push([`${dir}/${entry.name}`, `${dir}/${entry.name}`]);
    }
  }
  return out;
}

/** Copies the files into `webRoot/assets/pdfjs-<version>/`. */
export function copyPdfjs(webRoot: string, root: string = pdfjsRoot()): void {
  const base = join(webRoot, pdfjsBase(root));
  for (const [published, source] of pdfjsFiles(root)) {
    const to = join(base, published);
    mkdirSync(dirname(to), { recursive: true });
    copyFileSync(join(root, source), to);
  }
}

function contentType(path: string): string {
  return path.endsWith('.mjs') || path.endsWith('.js') ? 'text/javascript' : 'application/octet-stream';
}

/** Copies the files after a build, and serves the same paths from the package in the dev server. */
export function pdfjsAssets(): Plugin {
  const root = pdfjsRoot();
  const prefix = `/${pdfjsBase(root)}/`;
  const served = new Map(pdfjsFiles(root));
  let outDir = '';
  return {
    name: 'mlx-node-pdfjs-assets',
    configResolved(config) {
      // Vite names the folder relative to the root, which is not the shell's folder when the root is passed.
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const url = request.url ?? '';
        const end = url.indexOf('?');
        const path = end === -1 ? url : url.slice(0, end);
        const source = path.startsWith(prefix) ? served.get(path.slice(prefix.length)) : undefined;
        if (request.method !== 'GET' || source === undefined) {
          next();
          return;
        }
        response.statusCode = 200;
        response.setHeader('Content-Type', contentType(source));
        createReadStream(join(root, source)).on('error', next).pipe(response);
      });
    },
    writeBundle() {
      copyPdfjs(outDir, root);
    },
  };
}
