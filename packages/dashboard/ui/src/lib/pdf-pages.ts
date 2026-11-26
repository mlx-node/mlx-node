/**
 * The pdf.js side of a PDF the person adds: the worker, a document opened from its bytes, a page's size, the page
 * drawn on white and encoded as JPEG. The read itself (`readPdf`) sees only {@link PdfCodec}, so it runs on a fake.
 *
 * Settings: the bytes are passed as `data` (no URL fetch); no WASM, so the worker imports its JS decoders and never
 * tries a compile the CSP blocks; no system fonts, so a file draws the same on every Mac; no XFA; no annotation or
 * text layer, so `enableScripting` is never read. pdf.js loads only when the first PDF is opened.
 */

export type PdfOpenFailure = 'locked' | 'damaged';

export class PdfOpenError extends Error {
  constructor(readonly reason: PdfOpenFailure) {
    super(reason === 'locked' ? 'The PDF needs a password.' : 'The PDF could not be opened.');
    this.name = 'PdfOpenError';
  }
}

/** A page drawn at its final size, encoded as often as the quality ladder asks. */
export interface DrawnPage {
  encode(quality: number): Promise<Blob>;
  close(): void;
}

export interface PdfPage {
  /** In points, the page's rotation applied. */
  readonly width: number;
  readonly height: number;
  /** The page on white at `width` x `height` pixels. */
  draw(width: number, height: number): Promise<DrawnPage>;
  close(): void;
}

export interface PdfDocument {
  readonly pages: number;
  /** Page `n`, from 1. */
  page(n: number): Promise<PdfPage>;
  close(): Promise<void>;
}

export interface PdfCodec {
  /** Rejects with {@link PdfOpenError}. */
  open(data: ArrayBuffer): Promise<PdfDocument>;
}

type PdfJs = typeof import('pdfjs-dist');

/** `/assets/pdfjs-<version>/`: the build copies the worker and its files there. */
function assetBase(lib: PdfJs): string {
  return `/assets/pdfjs-${lib.version}/`;
}

/**
 * The module worker, once it said it is ready; null when it failed to start. The page starts it itself and hands it
 * over as the port: pdf.js's own start wraps a worker it judges cross-origin in a `blob:` script, which the CSP blocks.
 */
function moduleWorker(url: string): Promise<Worker | null> {
  return new Promise((resolve) => {
    let worker: Worker;
    try {
      worker = new Worker(url, { type: 'module' });
    } catch {
      resolve(null);
      return;
    }
    const settle = (ready: boolean): void => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      if (!ready) worker.terminate();
      resolve(ready ? worker : null);
    };
    const onMessage = (): void => settle(true);
    const onError = (): void => settle(false);
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
  });
}

let loading: Promise<PdfJs> | null = null;

/** pdf.js with its worker. A worker that fails leaves `workerSrc` set, and pdf.js then loads it on this thread. */
function pdfjs(): Promise<PdfJs> {
  loading ??= import('pdfjs-dist').then(async (lib) => {
    const src = `${assetBase(lib)}pdf.worker.min.mjs`;
    lib.GlobalWorkerOptions.workerSrc = src;
    const worker = await moduleWorker(src);
    if (worker !== null) lib.GlobalWorkerOptions.workerPort = worker;
    return lib;
  });
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

function release(canvas: HTMLCanvasElement): void {
  canvas.width = 0;
  canvas.height = 0;
}

function jpeg(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob === null ? reject(new Error('Could not encode')) : resolve(blob)),
      'image/jpeg',
      quality,
    );
  });
}

export const browserPdfCodec: PdfCodec = {
  open: async (data) => {
    const lib = await pdfjs();
    const base = assetBase(lib);
    const task = lib.getDocument({
      data: new Uint8Array(data),
      useWasm: false,
      wasmUrl: `${base}wasm/`,
      cMapUrl: `${base}cmaps/`,
      standardFontDataUrl: `${base}standard_fonts/`,
      useSystemFonts: false,
      enableXfa: false,
      stopAtErrors: false,
      disableWorkerRendering: true,
    });
    let doc: Awaited<typeof task.promise>;
    try {
      doc = await task.promise;
    } catch (err) {
      await task.destroy().catch(() => undefined);
      const locked = err instanceof Error && err.name === 'PasswordException';
      throw new PdfOpenError(locked ? 'locked' : 'damaged');
    }
    return {
      pages: doc.numPages,
      page: async (n) => {
        const page = await doc.getPage(n);
        const viewport = page.getViewport({ scale: 1 });
        return {
          width: viewport.width,
          height: viewport.height,
          draw: async (width, height) => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            try {
              await page.render({
                canvas,
                viewport,
                transform: [width / viewport.width, 0, 0, height / viewport.height, 0, 0],
                background: '#ffffff',
              }).promise;
            } catch (err) {
              release(canvas);
              throw err;
            }
            return { encode: (quality) => jpeg(canvas, quality), close: () => release(canvas) };
          },
          close: () => {
            page.cleanup();
          },
        };
      },
      close: () => task.destroy(),
    };
  },
};
