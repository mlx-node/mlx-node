import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { renderClairicDocuments } from '../../packages/lm/src/clairic-pdf.js';
import {
  CLAIRIC_MAX_MEDIA_BYTES,
  CLAIRIC_MAX_RENDER_PIXELS,
  CLAIRIC_MAX_RENDER_SCALE,
  CLAIRIC_MAX_TOTAL_MEDIA_BYTES,
} from '../../packages/lm/src/clairic-types.js';

interface SyntheticPageSpec {
  width: number;
  height: number;
  content: string;
}

/**
 * Build a minimal, spec-compliant PDF 1.4 byte buffer in pure ASCII with
 * accurate byte offsets in the xref table.
 */
function buildSyntheticPdf(pages: SyntheticPageSpec[]): Buffer {
  const objects: string[] = [];

  // 1: Catalog, 2: Pages, 3: Helvetica Type1 font
  const pageObjNums: number[] = [];
  for (let i = 0; i < pages.length; i++) {
    pageObjNums.push(4 + i * 2);
  }

  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  objects.push(`<< /Type /Pages /Kids [${pageObjNums.map((n) => `${n} 0 R`).join(' ')}] /Count ${pages.length} >>`);
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');

  for (let i = 0; i < pages.length; i++) {
    const { width, height, content } = pages[i];
    const contentObjNum = 5 + i * 2;
    const streamBytes = Buffer.byteLength(content, 'ascii');
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] /Resources << /Font << /F1 3 0 R >> >> /Contents ${contentObjNum} 0 R >>`,
    );
    objects.push(`<< /Length ${streamBytes} >>\nstream\n${content}\nendstream`);
  }

  let pdf = '%PDF-1.4\n';
  const offsets: number[] = [0];
  for (let i = 0; i < objects.length; i++) {
    offsets.push(Buffer.byteLength(pdf, 'ascii'));
    pdf += `${i + 1} 0 obj\n${objects[i]}\nendobj\n`;
  }

  const xrefOffset = Buffer.byteLength(pdf, 'ascii');
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += '0000000000 65535 f \n';
  for (let i = 1; i <= objects.length; i++) {
    pdf += `${String(offsets[i]).padStart(10, '0')} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(pdf, 'ascii');
}

function makeOnePagePdf(width = 200, height = 100): Buffer {
  // Draw a dark blue rectangle and black text on a white page.
  const content = ['0 0 0.8 rg', '20 20 80 40 re f', '0 0 0 rg', 'BT /F1 18 Tf 20 70 Td (Clairic PDF Test) Tj ET'].join(
    '\n',
  );
  return buildSyntheticPdf([{ width, height, content }]);
}

function makeTwoPagePdf(): Buffer {
  return buildSyntheticPdf([
    {
      width: 200,
      height: 100,
      content: '0.8 0 0 rg\n10 10 60 60 re f\nBT /F1 14 Tf 80 50 Td (Page 1) Tj ET',
    },
    {
      width: 150,
      height: 200,
      content: '0 0.7 0 rg\n15 25 70 70 re f\nBT /F1 14 Tf 20 150 Td (Page 2) Tj ET',
    },
  ]);
}

function makeTinyPngDataUri(): string {
  // Minimal 1x1 PNG header bytes (valid base64, small payload)
  const pngHeader = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d]);
  return `data:image/png;base64,${pngHeader.toString('base64')}`;
}

interface ElectronMarkerShape {
  type: PropertyDescriptor;
  electron: PropertyDescriptor;
}

/**
 * Render a one-page PDF `renders` times in a fresh Node process whose
 * `process.type` / `process.versions.electron` are defined, before any import,
 * the way an Electron utility process has them. Returns the child's stdout.
 */
function runInSimulatedElectronUtility(shape: ElectronMarkerShape, renders: number): string {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
  const pdfModulePath = resolve(root, 'packages/lm/src/clairic-pdf.ts');
  const docUri = `data:application/pdf;base64,${makeOnePagePdf().toString('base64')}`;
  const probe = `
    Object.defineProperty(process.versions, 'electron', ${JSON.stringify(shape.electron)});
    Object.defineProperty(process, 'type', ${JSON.stringify(shape.type)});
    const describeMarkers = () =>
      JSON.stringify([
        Object.getOwnPropertyDescriptor(process, 'type'),
        Object.getOwnPropertyDescriptor(process.versions, 'electron'),
      ]);
    const before = describeMarkers();
    const { registerHooks } = await import('node:module');
    registerHooks({
      resolve(specifier, context, nextResolve) {
        if (specifier === './clairic-types.js') {
          return nextResolve('./clairic-types.ts', context);
        }
        return nextResolve(specifier, context);
      },
    });
    const { renderClairicDocuments } = await import(${JSON.stringify(pdfModulePath)});
    for (let i = 0; i < ${renders}; i++) {
      let pages;
      try {
        pages = await renderClairicDocuments({ documents: [${JSON.stringify(docUri)}], fields: {} });
      } catch (error) {
        console.log('RENDER-ERROR ' + error.message);
        break;
      }
      if (!Array.isArray(pages) || pages.length !== 1) throw new Error('expected 1 rendered page');
      const { rgb, width, height } = pages[0];
      if (width <= 0 || height <= 0 || rgb.length !== width * height * 3) throw new Error('invalid RGB24 buffer');
      let dark = 0;
      for (let p = 0; p < rgb.length; p += 3) {
        if (rgb[p] < 64 && rgb[p + 1] < 64 && rgb[p + 2] < 64) dark++;
      }
      if (dark === 0) throw new Error('no dark text pixels');
      console.log('RENDER-OK ' + i);
    }
    if (describeMarkers() !== before) throw new Error('Electron markers changed: ' + describeMarkers() + ' vs ' + before);
    if (process.type !== 'utility' || process.versions.electron !== '45.2.3') throw new Error('Electron markers not restored');
    console.log('MARKERS-RESTORED');
  `;
  return execFileSync(process.execPath, ['--input-type=module', '-e', probe], { encoding: 'utf-8' });
}

const configurableElectronVersion: PropertyDescriptor = {
  value: '45.2.3',
  writable: false,
  enumerable: true,
  configurable: true,
};

function countNonWhitePixels(rgb: Buffer): number {
  let nonWhite = 0;
  for (let i = 0; i < rgb.length; i += 3) {
    if (rgb[i] < 250 || rgb[i + 1] < 250 || rgb[i + 2] < 250) {
      nonWhite++;
    }
  }
  return nonWhite;
}

describe('renderClairicDocuments', () => {
  it('returns undefined immediately when documents is absent or empty without loading pdfjs-dist', async () => {
    expect(await renderClairicDocuments({ state: 'hello', fields: { a: { type: 'text' } } })).toBeUndefined();
    expect(
      await renderClairicDocuments({ state: 'hello', documents: [], fields: { a: { type: 'text' } } }),
    ).toBeUndefined();
  });

  it('fast-path never imports pdfjs-dist or @napi-rs/canvas in a clean child process', () => {
    const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
    const pdfModulePath = resolve(root, 'packages/lm/src/clairic-pdf.ts');
    const probe = `
      const { registerHooks } = await import('node:module');
      registerHooks({
        resolve(specifier, context, nextResolve) {
          if (specifier.includes('pdfjs-dist') || specifier.includes('@napi-rs/canvas') || specifier.includes('@mlx-node/core')) {
            throw new Error('unexpected import: ' + specifier);
          }
          if (specifier === './clairic-types.js') {
            return nextResolve('./clairic-types.ts', context);
          }
          return nextResolve(specifier, context);
        },
      });
      const { renderClairicDocuments } = await import(${JSON.stringify(pdfModulePath)});
      const r1 = await renderClairicDocuments({ state: 'x', fields: {} });
      const r2 = await renderClairicDocuments({ state: 'x', documents: [], fields: {} });
      if (r1 !== undefined || r2 !== undefined) throw new Error('expected undefined');
      console.log('FAST-PATH-OK');
    `;
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
      encoding: 'utf-8',
    });
    expect(output).toContain('FAST-PATH-OK');
  });

  it('renders a valid 1-page PDF from a data URI with exact scale and RGB24 buffer', async () => {
    const pdfBytes = makeOnePagePdf(200, 100);
    const dataUri = `data:application/pdf;base64,${pdfBytes.toString('base64')}`;

    const pages = await renderClairicDocuments({
      documents: [dataUri],
      fields: { title: { type: 'text' } },
    });

    expect(pages).toBeDefined();
    expect(pages).toHaveLength(1);
    const page = pages![0];

    const expectedScale = Math.min(CLAIRIC_MAX_RENDER_SCALE, Math.sqrt(CLAIRIC_MAX_RENDER_PIXELS / (200 * 100)));
    expect(expectedScale).toBe(2.0);
    expect(page.width).toBe(Math.floor(200 * expectedScale));
    expect(page.height).toBe(Math.floor(100 * expectedScale));
    expect(page.documentIndex).toBe(0);
    expect(page.pageIndex).toBe(0);
    expect(Buffer.isBuffer(page.rgb)).toBe(true);
    expect(page.rgb.length).toBe(page.width * page.height * 3);

    // Must have both white background pixels and non-white rendered content (rectangle + text).
    const nonWhite = countNonWhitePixels(page.rgb);
    expect(nonWhite).toBeGreaterThan(1000);
    expect(nonWhite).toBeLessThan(page.width * page.height);
    // Top-left corner (0, 0) should be white background.
    expect([page.rgb[0], page.rgb[1], page.rgb[2]]).toEqual([255, 255, 255]);
  });

  it('renders a valid 1-page PDF from a { content_type, base64 } object', async () => {
    const pdfBytes = makeOnePagePdf(200, 100);
    const pages = await renderClairicDocuments({
      documents: [
        {
          content_type: 'application/pdf',
          base64: pdfBytes.toString('base64'),
        },
      ],
      fields: { title: { type: 'text' } },
    });

    expect(pages).toHaveLength(1);
    expect(pages![0].width).toBe(400);
    expect(pages![0].height).toBe(200);
    expect(pages![0].rgb.length).toBe(400 * 200 * 3);
    expect(countNonWhitePixels(pages![0].rgb)).toBeGreaterThan(1000);
  });

  it('clamps render scale to sqrt(4_000_000 / (w * h)) for large pages', async () => {
    const widthPt = 2500;
    const heightPt = 2500;
    const pdfBytes = makeOnePagePdf(widthPt, heightPt);
    const expectedScale = Math.min(
      CLAIRIC_MAX_RENDER_SCALE,
      Math.sqrt(CLAIRIC_MAX_RENDER_PIXELS / (widthPt * heightPt)),
    );
    expect(expectedScale).toBeCloseTo(0.8, 5);

    const pages = await renderClairicDocuments({
      documents: [`data:application/pdf;base64,${pdfBytes.toString('base64')}`],
      fields: {},
    });

    expect(pages).toHaveLength(1);
    expect(pages![0].width).toBe(Math.floor(widthPt * expectedScale));
    expect(pages![0].height).toBe(Math.floor(heightPt * expectedScale));
    expect(pages![0].width * pages![0].height).toBeLessThanOrEqual(CLAIRIC_MAX_RENDER_PIXELS);
    expect(pages![0].rgb.length).toBe(pages![0].width * pages![0].height * 3);
  });

  it('enforces combined images + PDF page limit (<= 4 passes, > 4 rejects)', async () => {
    const twoPagePdf = makeTwoPagePdf();
    const docUri = `data:application/pdf;base64,${twoPagePdf.toString('base64')}`;
    const imgUri = makeTinyPngDataUri();

    // 2 images + 2-page PDF = 4 total -> passes
    const okPages = await renderClairicDocuments({
      images: [imgUri, imgUri],
      documents: [docUri],
      fields: {},
    });
    expect(okPages).toHaveLength(2);
    expect(okPages![0].documentIndex).toBe(0);
    expect(okPages![0].pageIndex).toBe(0);
    expect(okPages![0].width).toBe(400);
    expect(okPages![0].height).toBe(200);
    expect(okPages![1].documentIndex).toBe(0);
    expect(okPages![1].pageIndex).toBe(1);
    expect(okPages![1].width).toBe(300);
    expect(okPages![1].height).toBe(400);

    // 3 images + 2-page PDF = 5 total -> rejects
    await expect(
      renderClairicDocuments({
        images: [imgUri, imgUri, imgUri],
        documents: [docUri],
        fields: {},
      }),
    ).rejects.toThrow(/must not exceed 4/);

    // 2 two-page PDFs = 4 total -> passes with proper documentIndex/pageIndex
    const multiDocPages = await renderClairicDocuments({
      documents: [docUri, docUri],
      fields: {},
    });
    expect(multiDocPages).toHaveLength(4);
    expect(multiDocPages!.map((p) => [p.documentIndex, p.pageIndex])).toEqual([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);

    // 3 two-page PDFs = 6 total -> rejects
    await expect(
      renderClairicDocuments({
        documents: [docUri, docUri, docUri],
        fields: {},
      }),
    ).rejects.toThrow(/must not exceed 4/);
  });

  it('rejects non-PDF bytes, remote URLs, wrong MIME types, malformed base64, and extra keys', async () => {
    // Missing %PDF- magic header
    const notPdf = Buffer.from('Hello, this is plain text, not a PDF', 'utf-8').toString('base64');
    await expect(
      renderClairicDocuments({
        documents: [`data:application/pdf;base64,${notPdf}`],
        fields: {},
      }),
    ).rejects.toThrow(/document must be a PDF within the decoded byte limit/);

    // Remote URL rejected
    await expect(
      renderClairicDocuments({
        documents: ['https://example.com/invoice.pdf'],
        fields: {},
      }),
    ).rejects.toThrow(/embedded PDF data URI/);

    // Wrong MIME in data URI
    const pdfB64 = makeOnePagePdf().toString('base64');
    await expect(
      renderClairicDocuments({
        documents: [`data:image/png;base64,${pdfB64}`],
        fields: {},
      }),
    ).rejects.toThrow(/embedded PDF data URI/);

    // Wrong MIME in object
    await expect(
      renderClairicDocuments({
        documents: [{ content_type: 'image/png', base64: pdfB64 }],
        fields: {},
      }),
    ).rejects.toThrow(/content type must be application\/pdf/);

    // Malformed base64
    await expect(
      renderClairicDocuments({
        documents: [{ content_type: 'application/pdf', base64: '!!!not-valid-base64!!!' }],
        fields: {},
      }),
    ).rejects.toThrow(/base64 is malformed/);

    // Extra field on document object
    await expect(
      renderClairicDocuments({
        documents: [{ content_type: 'application/pdf', base64: pdfB64, rotation: 90 }],
        fields: {},
      }),
    ).rejects.toThrow(/must contain content_type and base64/);

    // Valid %PDF- prefix but corrupted PDF structure
    const corruptPdf = Buffer.from('%PDF-1.4\nnot a valid pdf body', 'ascii').toString('base64');
    await expect(
      renderClairicDocuments({
        documents: [`data:application/pdf;base64,${corruptPdf}`],
        fields: {},
      }),
    ).rejects.toThrow(/PDF cannot be opened or rendered/);
  });

  it('enforces 4 MiB single PDF limit and 8 MiB combined images + documents limit', async () => {
    // Single PDF > 4 MiB
    const oversizedPdf = Buffer.alloc(CLAIRIC_MAX_MEDIA_BYTES + 1, 0x20);
    oversizedPdf.write('%PDF-1.4\n', 0, 'ascii');
    await expect(
      renderClairicDocuments({
        documents: [`data:application/pdf;base64,${oversizedPdf.toString('base64')}`],
        fields: {},
      }),
    ).rejects.toThrow(/within the decoded byte limit/);

    // 3 PDFs of 3 MiB each = 9 MiB (> 8 MiB aggregate)
    const threeMiBPdf = Buffer.alloc(3 * 1024 * 1024, 0x20);
    threeMiBPdf.write('%PDF-1.4\n', 0, 'ascii');
    const threeMiBUri = `data:application/pdf;base64,${threeMiBPdf.toString('base64')}`;
    await expect(
      renderClairicDocuments({
        documents: [threeMiBUri, threeMiBUri, threeMiBUri],
        fields: {},
      }),
    ).rejects.toThrow(/aggregate decoded byte limit/);

    // 2 images of 3 MiB each + 1 PDF of 3 MiB = 9 MiB (> 8 MiB aggregate)
    const threeMiBImage = Buffer.alloc(3 * 1024 * 1024, 0x00);
    const threeMiBImageUri = `data:image/png;base64,${threeMiBImage.toString('base64')}`;
    expect(3 * 3 * 1024 * 1024).toBeGreaterThan(CLAIRIC_MAX_TOTAL_MEDIA_BYTES);
    await expect(
      renderClairicDocuments({
        images: [threeMiBImageUri, threeMiBImageUri],
        documents: [threeMiBUri],
        fields: {},
      }),
    ).rejects.toThrow(/aggregate decoded byte limit/);
  });

  it('respects AbortSignal cancellation', async () => {
    const controller = new AbortController();
    controller.abort(new DOMException('cancelled by test', 'AbortError'));
    const docUri = `data:application/pdf;base64,${makeOnePagePdf().toString('base64')}`;

    await expect(
      renderClairicDocuments({ documents: [docUri], fields: {} }, { signal: controller.signal }),
    ).rejects.toThrow(/cancelled by test/);
  });

  it('loads @napi-rs/canvas cleanly even when NAPI_RS_NATIVE_LIBRARY_PATH points at another addon', () => {
    const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
    const pdfModulePath = resolve(root, 'packages/lm/src/clairic-pdf.ts');
    const fakeAddon = resolve(root, 'packages/lm/package.json');
    const docUri = `data:application/pdf;base64,${makeOnePagePdf().toString('base64')}`;
    const probe = `
      const { registerHooks } = await import('node:module');
      registerHooks({
        resolve(specifier, context, nextResolve) {
          if (specifier === './clairic-types.js') {
            return nextResolve('./clairic-types.ts', context);
          }
          return nextResolve(specifier, context);
        },
      });
      const { renderClairicDocuments } = await import(${JSON.stringify(pdfModulePath)});
      const before = process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
      const pages = await renderClairicDocuments({ documents: [${JSON.stringify(docUri)}], fields: {} });
      if (!Array.isArray(pages) || pages.length !== 1) throw new Error('expected 1 rendered page');
      if (pages[0].width <= 0 || pages[0].height <= 0 || pages[0].rgb.length !== pages[0].width * pages[0].height * 3) {
        throw new Error('invalid RGB24 buffer');
      }
      if (process.env.NAPI_RS_NATIVE_LIBRARY_PATH !== before) {
        throw new Error('NAPI_RS_NATIVE_LIBRARY_PATH was not restored');
      }
      console.log('CANVAS-OVERRIDE-ISOLATED-OK');
    `;
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
      encoding: 'utf-8',
      env: { ...process.env, NAPI_RS_NATIVE_LIBRARY_PATH: fakeAddon },
    });
    expect(output).toContain('CANVAS-OVERRIDE-ISOLATED-OK');
  });

  it('keeps clairic-types.ts 100% native-free', () => {
    const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
    const typesModulePath = resolve(root, 'packages/lm/src/clairic-types.ts');
    const probe = `
      const { registerHooks } = await import('node:module');
      registerHooks({
        resolve(specifier, context, nextResolve) {
          if (
            specifier === '@mlx-node/core' ||
            specifier.startsWith('@mlx-node/core/') ||
            specifier.includes('pdfjs-dist') ||
            specifier.includes('@napi-rs/canvas') ||
            specifier.endsWith('.node')
          ) {
            throw new Error('forbidden import: ' + specifier);
          }
          return nextResolve(specifier, context);
        },
      });
      const mod = await import(${JSON.stringify(typesModulePath)});
      if (mod.CLAIRIC_MAX_IMAGES !== 4) throw new Error('unexpected constant');
      console.log('CLAIRIC-TYPES-NATIVE-FREE-OK');
    `;
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
      encoding: 'utf-8',
    });
    expect(output).toContain('CLAIRIC-TYPES-NATIVE-FREE-OK');
  });

  describe('in an Electron utility process', () => {
    it('renders twice when process.type is read-only but configurable', () => {
      const output = runInSimulatedElectronUtility(
        {
          type: { value: 'utility', writable: false, enumerable: true, configurable: true },
          electron: configurableElectronVersion,
        },
        2,
      );
      expect(output).toContain('RENDER-OK 0');
      expect(output).toContain('RENDER-OK 1');
      expect(output).toContain('MARKERS-RESTORED');
    });

    it('renders when process.type is writable but not configurable', () => {
      const output = runInSimulatedElectronUtility(
        {
          type: { value: 'utility', writable: true, enumerable: true, configurable: false },
          electron: configurableElectronVersion,
        },
        1,
      );
      expect(output).toContain('RENDER-OK 0');
      expect(output).toContain('MARKERS-RESTORED');
    });

    it('hides process.versions.electron when process.type can be neither redefined nor written', () => {
      const output = runInSimulatedElectronUtility(
        {
          type: { value: 'utility', writable: false, enumerable: true, configurable: false },
          electron: configurableElectronVersion,
        },
        2,
      );
      expect(output).toContain('RENDER-OK 0');
      expect(output).toContain('RENDER-OK 1');
      expect(output).toContain('MARKERS-RESTORED');
    });

    it('fails with a clear message when no Electron marker can be hidden', () => {
      const output = runInSimulatedElectronUtility(
        {
          type: { value: 'utility', writable: false, enumerable: true, configurable: false },
          electron: { value: '45.2.3', writable: false, enumerable: true, configurable: false },
        },
        1,
      );
      expect(output).toContain('RENDER-ERROR PDF rendering is unavailable in this Electron process');
      expect(output).toContain('MARKERS-RESTORED');
    });
  });
});
