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

const PDF_DATA_URI_RE = /^data:application\/pdf;base64,([A-Za-z0-9+/]*={0,2})$/i;
const IMAGE_DATA_URI_RE = /^data:[^;,]+;base64,(.*)$/s;
const BASE64_CHARS_RE = /^[A-Za-z0-9+/]*={0,2}$/;
const PDF_MAGIC = Buffer.from('%PDF-', 'ascii');

interface PdfjsAssetUrls {
  standardFontDataUrl: string;
  cMapUrl: string;
  wasmUrl: string;
}

let cachedAssetUrls: PdfjsAssetUrls | undefined;

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

function isValidStrictBase64(encoded: string): boolean {
  return encoded.length % 4 === 0 && BASE64_CHARS_RE.test(encoded);
}

function strictDecodeBase64(encoded: string, errorMessage: string): Buffer {
  if (!isValidStrictBase64(encoded)) {
    throw new Error(errorMessage);
  }
  return Buffer.from(encoded, 'base64');
}

function decodedBase64ByteLength(encoded: string): number {
  if (!isValidStrictBase64(encoded)) {
    throw new Error('image base64 is malformed');
  }
  if (encoded.length === 0) {
    return 0;
  }
  let padding = 0;
  if (encoded.endsWith('==')) {
    padding = 2;
  } else if (encoded.endsWith('=')) {
    padding = 1;
  }
  return (encoded.length / 4) * 3 - padding;
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

  const [pdfjs, canvasModule] = await Promise.all([
    import('pdfjs-dist/legacy/build/pdf.mjs'),
    import('@napi-rs/canvas'),
  ]);

  signal?.throwIfAborted();

  const { createCanvas } = canvasModule;
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
