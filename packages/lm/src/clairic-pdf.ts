/**
 * Validate embedded PDF documents in Clairic requests and rasterize each page
 * to tightly packed RGB24 buffers using `pdfjs-dist` and `@napi-rs/canvas`.
 *
 * Non-PDF requests exit immediately on the fast path without loading either
 * dependency.
 */

import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

import {
  CLAIRIC_MAX_IMAGES,
  CLAIRIC_MAX_MEDIA_BYTES,
  CLAIRIC_MAX_PIXELS,
  CLAIRIC_MAX_RENDER_PIXELS,
  CLAIRIC_MAX_RENDER_SCALE,
  CLAIRIC_MAX_TOTAL_MEDIA_BYTES,
  type RenderedPdfPage,
} from './clairic-types.js';

const PDF_DATA_URI_RE = /^data:application\/pdf;base64,(.+)$/is;
const IMAGE_DATA_URI_RE = /^data:[^;,]+;base64,(.*)$/is;
const PDF_MAGIC = Buffer.from('%PDF-', 'ascii');

interface PdfjsAssetUrls {
  standardFontDataUrl: string;
  cMapUrl: string;
  wasmUrl: string;
}

type CanvasModule = typeof import('@napi-rs/canvas');

let cachedAssetUrls: PdfjsAssetUrls | undefined;
let cachedCanvasModule: CanvasModule | undefined;

/**
 * Load `@napi-rs/canvas` via CJS `require` while temporarily hiding
 * `NAPI_RS_NATIVE_LIBRARY_PATH`.
 *
 * Hosts such as the desktop inference sidecar set `NAPI_RS_NATIVE_LIBRARY_PATH`
 * for `@mlx-node/core`. Because `@napi-rs/canvas` is also a NAPI-RS addon that
 * checks the same generic variable first, leaving it visible causes Canvas to
 * `require()` `mlx-core.*.node` instead of its own `skia.*.node` binary.
 * Synchronous `createRequire` avoids any async window where concurrent work
 * could observe the temporarily cleared env var.
 */
function loadCanvasModule(): CanvasModule {
  if (cachedCanvasModule !== undefined) {
    return cachedCanvasModule;
  }
  const require = createRequire(import.meta.url);
  const hadOverride = Object.hasOwn(process.env, 'NAPI_RS_NATIVE_LIBRARY_PATH');
  const savedOverride = process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
  if (hadOverride) {
    delete process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
  }
  try {
    cachedCanvasModule = require('@napi-rs/canvas') as CanvasModule;
    return cachedCanvasModule;
  } finally {
    if (hadOverride) {
      process.env.NAPI_RS_NATIVE_LIBRARY_PATH = savedOverride;
    }
  }
}

type PdfjsModule = typeof import('pdfjs-dist/legacy/build/pdf.mjs');

let cachedPdfjsModule: PdfjsModule | undefined;

function isElectronNonBrowserProcess(): boolean {
  const processType: unknown = Reflect.get(process, 'type');
  return Boolean(process.versions.electron) && Boolean(processType) && processType !== 'browser';
}

/**
 * Make `target[key]` read as `undefined` and return the function that puts the
 * original back, or `undefined` when the property cannot be hidden.
 */
function hideProperty(target: object, key: string): (() => void) | undefined {
  const saved = Object.getOwnPropertyDescriptor(target, key);
  let restore: () => void;
  if (saved === undefined) {
    if (!Object.isExtensible(target)) {
      return undefined;
    }
    Object.defineProperty(target, key, { value: undefined, writable: true, configurable: true, enumerable: false });
    restore = () => {
      Reflect.deleteProperty(target, key);
    };
  } else if (saved.configurable === true) {
    Object.defineProperty(target, key, {
      value: undefined,
      writable: true,
      configurable: true,
      enumerable: saved.enumerable,
    });
    restore = () => {
      Object.defineProperty(target, key, saved);
    };
  } else if (saved.writable === true) {
    Reflect.set(target, key, undefined);
    restore = () => {
      Reflect.set(target, key, saved.value);
    };
  } else {
    return undefined;
  }
  if (Reflect.get(target, key) !== undefined) {
    restore();
    return undefined;
  }
  return restore;
}

/**
 * Load `pdfjs-dist` so that it configures itself for Node.
 *
 * pdfjs decides "is this Node?" once, at module evaluation, and answers no in
 * any Electron process whose `process.type` is not `'browser'` (the desktop
 * app's utility process included). It then takes DOM defaults and
 * `getDocument` throws `document is not defined`. In such a process the
 * Electron marker is hidden while both pdfjs modules evaluate: the main module
 * and the worker module, which has its own copy of the check and is what the
 * in-process fake worker runs (it publishes `globalThis.pdfjsWorker`, which
 * pdfjs uses instead of importing the worker again). The load is a synchronous
 * `require` of ESM so no other code can run while the marker is hidden. This
 * file must stay the only place that imports pdfjs: a module evaluated
 * elsewhere would cache the DOM configuration for the whole process.
 */
async function loadPdfjsModule(): Promise<PdfjsModule> {
  if (cachedPdfjsModule !== undefined) {
    return cachedPdfjsModule;
  }
  if (!isElectronNonBrowserProcess()) {
    cachedPdfjsModule = await import('pdfjs-dist/legacy/build/pdf.mjs');
    return cachedPdfjsModule;
  }
  const require = createRequire(import.meta.url);
  const restore = hideProperty(process, 'type') ?? hideProperty(process.versions, 'electron');
  if (restore === undefined) {
    throw new Error('PDF rendering is unavailable in this Electron process');
  }
  try {
    const pdfjs = require('pdfjs-dist/legacy/build/pdf.mjs') as PdfjsModule;
    require('pdfjs-dist/legacy/build/pdf.worker.mjs');
    cachedPdfjsModule = pdfjs;
    return cachedPdfjsModule;
  } finally {
    restore();
  }
}

function toTrailingSlashPath(dir: string): string {
  const normalized = dir.replaceAll('\\', '/');
  return normalized.endsWith('/') ? normalized : `${normalized}/`;
}

function resolvePdfjsAssetUrls(): PdfjsAssetUrls {
  if (cachedAssetUrls !== undefined) {
    return cachedAssetUrls;
  }
  const require = createRequire(import.meta.url);
  const pdfjsRoot = dirname(require.resolve('pdfjs-dist/package.json'));
  cachedAssetUrls = {
    standardFontDataUrl: toTrailingSlashPath(join(pdfjsRoot, 'standard_fonts')),
    cMapUrl: toTrailingSlashPath(join(pdfjsRoot, 'cmaps')),
    wasmUrl: toTrailingSlashPath(join(pdfjsRoot, 'wasm')),
  };
  return cachedAssetUrls;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeBase64(encoded: string): string | null {
  const stripped = encoded
    .replace(/[\t\n\r ]+/g, '')
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  if (stripped.length === 0) return null;

  let padCount = 0;
  while (padCount < stripped.length && stripped.charCodeAt(stripped.length - 1 - padCount) === 0x3d) {
    padCount++;
  }
  if (padCount > 2) return null;

  const dataLen = stripped.length - padCount;
  if (dataLen === 0) return null;

  if (padCount > 0) {
    if (stripped.length % 4 !== 0) return null;
  } else if (dataLen % 4 === 1) {
    return null;
  }

  for (let i = 0; i < dataLen; i++) {
    const c = stripped.charCodeAt(i);
    const valid =
      (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) || (c >= 0x30 && c <= 0x39) || c === 0x2b || c === 0x2f;
    if (!valid) return null;
  }

  if (padCount === 0) {
    const rem = dataLen % 4;
    if (rem === 2) return `${stripped}==`;
    if (rem === 3) return `${stripped}=`;
  }
  return stripped;
}

function strictDecodeBase64(encoded: string, errorMessage: string): Buffer {
  const normalized = normalizeBase64(encoded);
  if (normalized === null) {
    throw new Error(errorMessage);
  }
  return Buffer.from(normalized, 'base64');
}

function decodedBase64ByteLength(encoded: string): number {
  if (encoded.length === 0) {
    return 0;
  }
  const normalized = normalizeBase64(encoded);
  if (normalized === null) {
    throw new Error('image base64 is malformed');
  }
  let padding = 0;
  if (normalized.endsWith('==')) {
    padding = 2;
  } else if (normalized.endsWith('=')) {
    padding = 1;
  }
  return (normalized.length / 4) * 3 - padding;
}

function parseDocumentBytes(entry: unknown): Buffer {
  let base64: string;
  if (typeof entry === 'string') {
    const match = PDF_DATA_URI_RE.exec(entry);
    if (match === null) {
      throw new Error('document must be an embedded PDF data URI');
    }
    base64 = match[1];
  } else if (isPlainRecord(entry)) {
    const keys = Object.keys(entry);
    if (keys.length !== 2 || !Object.hasOwn(entry, 'content_type') || !Object.hasOwn(entry, 'base64')) {
      throw new Error('document object must contain content_type and base64');
    }
    if (entry.content_type !== 'application/pdf' || typeof entry.base64 !== 'string') {
      throw new Error('document content type must be application/pdf');
    }
    base64 = entry.base64;
  } else {
    throw new Error('document object must contain content_type and base64');
  }

  const decoded = strictDecodeBase64(base64, 'document base64 is malformed');
  if (
    decoded.length < PDF_MAGIC.length ||
    !decoded.subarray(0, PDF_MAGIC.length).equals(PDF_MAGIC) ||
    decoded.length > CLAIRIC_MAX_MEDIA_BYTES
  ) {
    throw new Error('document must be a PDF within the decoded byte limit');
  }
  return decoded;
}

function measureImageDecodedBytes(entry: unknown): number {
  if (typeof entry === 'string') {
    const match = IMAGE_DATA_URI_RE.exec(entry);
    if (match === null) {
      throw new Error('image must be an embedded base64 data URI');
    }
    return decodedBase64ByteLength(match[1]);
  }
  if (isPlainRecord(entry) && typeof entry.base64 === 'string') {
    return decodedBase64ByteLength(entry.base64);
  }
  throw new Error('image object has invalid fields');
}

function rgbaToRgbOnWhite(rgba: Uint8ClampedArray, width: number, height: number): Buffer {
  const rgb = Buffer.allocUnsafe(width * height * 3);
  for (let src = 0, dst = 0; src < rgba.length; src += 4, dst += 3) {
    const alpha = rgba[src + 3];
    if (alpha === 255) {
      rgb[dst] = rgba[src];
      rgb[dst + 1] = rgba[src + 1];
      rgb[dst + 2] = rgba[src + 2];
    } else if (alpha === 0) {
      rgb[dst] = 255;
      rgb[dst + 1] = 255;
      rgb[dst + 2] = 255;
    } else {
      const inv = 255 - alpha;
      rgb[dst] = Math.round((rgba[src] * alpha + 255 * inv) / 255);
      rgb[dst + 1] = Math.round((rgba[src + 1] * alpha + 255 * inv) / 255);
      rgb[dst + 2] = Math.round((rgba[src + 2] * alpha + 255 * inv) / 255);
    }
  }
  return rgb;
}

function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) {
    return true;
  }
  return error instanceof Error && (error.name === 'AbortError' || error.name === 'RenderingCancelledException');
}

/**
 * Validate and rasterize embedded PDF documents in a parsed Clairic request.
 *
 * Returns `undefined` immediately when `parsedRequest.documents` is absent or
 * empty, without importing `pdfjs-dist` or `@napi-rs/canvas`.
 */
export async function renderClairicDocuments(
  parsedRequest: Record<string, unknown>,
  options?: { signal?: AbortSignal },
): Promise<RenderedPdfPage[] | undefined> {
  const rawDocuments = parsedRequest.documents;
  if (rawDocuments === undefined || rawDocuments === null) {
    return undefined;
  }
  if (!Array.isArray(rawDocuments)) {
    throw new Error('documents must be an array');
  }
  if (rawDocuments.length === 0) {
    return undefined;
  }

  const signal = options?.signal;
  signal?.throwIfAborted();

  let imageCount = 0;
  let totalDecodedBytes = 0;

  const rawImages = parsedRequest.images;
  if (rawImages !== undefined && rawImages !== null) {
    if (!Array.isArray(rawImages)) {
      throw new Error('images must be an array');
    }
    imageCount = rawImages.length;
    if (imageCount > CLAIRIC_MAX_IMAGES) {
      throw new Error('request exceeds the image count limit');
    }
    for (const imageEntry of rawImages) {
      totalDecodedBytes += measureImageDecodedBytes(imageEntry);
    }
  }

  if (imageCount + rawDocuments.length > CLAIRIC_MAX_IMAGES) {
    throw new Error(`PDF pages and images together must not exceed ${CLAIRIC_MAX_IMAGES}`);
  }

  const decodedDocs: Buffer[] = [];
  for (const docEntry of rawDocuments) {
    const docBytes = parseDocumentBytes(docEntry);
    totalDecodedBytes += docBytes.length;
    decodedDocs.push(docBytes);
  }

  if (totalDecodedBytes > CLAIRIC_MAX_TOTAL_MEDIA_BYTES) {
    throw new Error('images exceed the aggregate decoded byte limit');
  }

  signal?.throwIfAborted();

  // Load `@napi-rs/canvas` BEFORE `pdfjs-dist`: pdfjs internally requires
  // `@napi-rs/canvas` during module initialization to polyfill DOMMatrix/Path2D,
  // so priming the module cache with `NAPI_RS_NATIVE_LIBRARY_PATH` hidden ensures
  // both pdfjs and our renderer get Skia rather than `@mlx-node/core`.
  const { createCanvas } = loadCanvasModule();
  const pdfjs = await loadPdfjsModule();

  signal?.throwIfAborted();
  const assetUrls = resolvePdfjsAssetUrls();

  const loadingTasks: ReturnType<typeof pdfjs.getDocument>[] = [];
  const openedDocs: Awaited<ReturnType<typeof pdfjs.getDocument>['promise']>[] = [];
  try {
    // Open all PDFs and enforce the aggregate page + image limit BEFORE rendering any page.
    let totalPdfPages = 0;
    for (const docBytes of decodedDocs) {
      signal?.throwIfAborted();
      const loadingTask = pdfjs.getDocument({
        data: Uint8Array.from(docBytes),
        useSystemFonts: false,
        disableFontFace: true,
        isOffscreenCanvasSupported: false,
        isImageDecoderSupported: false,
        disableWorkerRendering: true,
        maxImageSize: CLAIRIC_MAX_PIXELS,
        canvasMaxAreaInBytes: CLAIRIC_MAX_RENDER_PIXELS * 4,
        verbosity: pdfjs.VerbosityLevel.ERRORS,
        standardFontDataUrl: assetUrls.standardFontDataUrl,
        cMapUrl: assetUrls.cMapUrl,
        cMapPacked: true,
        wasmUrl: assetUrls.wasmUrl,
      });
      loadingTasks.push(loadingTask);

      const onAbortLoad = () => {
        void loadingTask.destroy();
      };
      signal?.addEventListener('abort', onAbortLoad, { once: true });

      let pdfDoc: Awaited<typeof loadingTask.promise>;
      try {
        pdfDoc = await loadingTask.promise;
      } catch (error) {
        if (isAbortError(error, signal)) {
          signal?.throwIfAborted();
        }
        throw new Error('PDF cannot be opened or rendered; encrypted PDFs are unsupported', { cause: error });
      } finally {
        signal?.removeEventListener('abort', onAbortLoad);
      }

      openedDocs.push(pdfDoc);
      if (pdfDoc.numPages <= 0) {
        throw new Error(`PDF pages and images together must not exceed ${CLAIRIC_MAX_IMAGES}`);
      }
      totalPdfPages += pdfDoc.numPages;
      if (imageCount + totalPdfPages > CLAIRIC_MAX_IMAGES) {
        throw new Error(`PDF pages and images together must not exceed ${CLAIRIC_MAX_IMAGES}`);
      }
    }

    const renderedPages: RenderedPdfPage[] = [];

    for (let documentIndex = 0; documentIndex < openedDocs.length; documentIndex++) {
      const pdfDoc = openedDocs[documentIndex];
      for (let pageIndex = 0; pageIndex < pdfDoc.numPages; pageIndex++) {
        signal?.throwIfAborted();
        const page = await pdfDoc.getPage(pageIndex + 1);
        try {
          const viewport1 = page.getViewport({ scale: 1.0 });
          const area1 = viewport1.width * viewport1.height;
          if (!Number.isFinite(area1) || viewport1.width <= 0 || viewport1.height <= 0) {
            throw new Error('PDF page has invalid dimensions');
          }

          const scale = Math.min(CLAIRIC_MAX_RENDER_SCALE, Math.sqrt(CLAIRIC_MAX_RENDER_PIXELS / area1));
          const viewport = page.getViewport({ scale });
          const width = Math.max(1, Math.floor(viewport.width));
          const height = Math.max(1, Math.floor(viewport.height));

          const canvas = createCanvas(width, height);
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, width, height);

          const renderTask = page.render({
            canvas: null,
            canvasContext: ctx as unknown as CanvasRenderingContext2D,
            viewport,
            background: 'rgb(255,255,255)',
          });

          const onAbortRender = () => {
            renderTask.cancel();
          };
          signal?.addEventListener('abort', onAbortRender, { once: true });

          try {
            await renderTask.promise;
          } catch (error) {
            if (isAbortError(error, signal)) {
              signal?.throwIfAborted();
            }
            throw new Error('PDF cannot be opened or rendered; encrypted PDFs are unsupported', { cause: error });
          } finally {
            signal?.removeEventListener('abort', onAbortRender);
          }

          signal?.throwIfAborted();

          const imageData = ctx.getImageData(0, 0, width, height);
          const rgb = rgbaToRgbOnWhite(imageData.data, width, height);

          renderedPages.push({
            rgb,
            width,
            height,
            documentIndex,
            pageIndex,
          });
        } finally {
          page.cleanup();
        }
      }
    }

    return renderedPages;
  } finally {
    await Promise.allSettled(openedDocs.map((doc) => doc.cleanup()));
    await Promise.allSettled(loadingTasks.map((task) => task.destroy()));
  }
}
