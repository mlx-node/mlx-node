/**
 * PP-StructureV3 Document Understanding Pipeline
 *
 * Combines PP-DocLayoutV3 (layout detection) with PP-OCRv5 (text detection + recognition)
 * for fast, accurate document understanding without a VLM.
 *
 * Pipeline:
 *   1. DocLayoutModel detects layout elements (titles, text, tables, figures...)
 *   2. For text/title/list elements: TextDetModel detects text lines → TextRecModel recognizes text
 *   3. For table/formula/chart: VLM fallback (optional)
 *   4. Results assembled into structured markdown in reading order
 *
 * @example
 * ```typescript
 * import { StructureV3Pipeline } from '@mlx-node/vlm';
 *
 * const pipeline = StructureV3Pipeline.load({
 *   layoutModelPath: './models/PP-DocLayoutV3',
 *   textDetModelPath: './models/PP-OCRv5_server_det',
 *   textRecModelPath: './models/PP-OCRv5_server_rec',
 *   dictPath: './models/PP-OCRv5_server_rec/ppocr_keys_v1.txt',
 * });
 *
 * const result = pipeline.analyze('./document.png');
 * console.log(result.markdown);
 * ```
 */

import { readFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtempSync, rmSync } from 'node:fs';
import { DocLayoutModel, TextDetModel, TextRecModel, type LayoutElement } from '@mlx-node/core';

// ============================================================================
// Types
// ============================================================================

/** Configuration for loading the StructureV3 pipeline. */
export interface StructureV3Config {
  /** Path to PP-DocLayoutV3 model directory */
  layoutModelPath: string;
  /** Path to PP-OCRv5 text detection model directory */
  textDetModelPath: string;
  /** Path to PP-OCRv5 text recognition model directory */
  textRecModelPath: string;
  /** Path to character dictionary file (e.g., ppocr_keys_v1.txt) */
  dictPath: string;
}

/** Options for document analysis. */
export interface AnalyzeOptions {
  /** Layout detection confidence threshold (default: 0.5) */
  layoutThreshold?: number;
  /** Text detection confidence threshold (default: 0.3) */
  textDetThreshold?: number;
  /** Whether to include element-level details in output (default: false) */
  includeDetails?: boolean;
}

/** A recognized text line within a layout element. */
export interface TextLine {
  /** Bounding box [x1, y1, x2, y2] relative to the element crop */
  bbox: number[];
  /** Recognized text */
  text: string;
  /** Recognition confidence */
  score: number;
}

/** A structured document element with recognized content. */
export interface StructuredElement {
  /** Element type from layout detection */
  label: string;
  /** Detection confidence */
  score: number;
  /** Bounding box [x1, y1, x2, y2] in original image coordinates */
  bbox: number[];
  /** Reading order index */
  order: number;
  /** Recognized text content */
  text: string;
  /** Individual text lines (if includeDetails is true) */
  lines?: TextLine[];
}

/** Result of document analysis. */
export interface StructuredDocument {
  /** Structured elements in reading order */
  elements: StructuredElement[];
  /** Assembled markdown output */
  markdown: string;
}

// ============================================================================
// Element type sets
// ============================================================================

/** Elements that contain text and should be processed with OCR */
const TEXT_LABELS = new Set([
  'title',
  'doc_title',
  'paragraph_title',
  'text',
  'abstract',
  'list',
  'table_caption',
  'table_footnote',
  'figure_caption',
  'chart_caption',
  'formula_caption',
  'code_txt',
  'header',
  'footer',
  'footnote',
  'margin_note',
  'reference',
  'content',
  'index',
  'handwriting',
]);

/** Elements that are non-content (skipped) */
const SKIP_LABELS = new Set(['abandon']);

// ============================================================================
// Pipeline
// ============================================================================

/**
 * PP-StructureV3 document understanding pipeline.
 *
 * Uses dedicated OCR models (TextDet + TextRec) instead of a VLM,
 * providing ~4-5x faster text extraction with ~6x lower memory usage.
 */
export class StructureV3Pipeline {
  private layout: DocLayoutModel;
  private textDet: TextDetModel;
  private textRec: TextRecModel;

  private constructor(layout: DocLayoutModel, textDet: TextDetModel, textRec: TextRecModel) {
    this.layout = layout;
    this.textDet = textDet;
    this.textRec = textRec;
  }

  /**
   * Load all models and create the pipeline.
   */
  static load(config: StructureV3Config): StructureV3Pipeline {
    const layout = DocLayoutModel.load(config.layoutModelPath);
    const textDet = TextDetModel.load(config.textDetModelPath);
    const textRec = TextRecModel.load(config.textRecModelPath, config.dictPath);
    return new StructureV3Pipeline(layout, textDet, textRec);
  }

  /**
   * Analyze a document image and extract structured content.
   *
   * @param imagePath - Path to the document image
   * @param options - Analysis options
   * @returns Structured document with elements and markdown
   */
  async analyze(imagePath: string, options: AnalyzeOptions = {}): Promise<StructuredDocument> {
    const { layoutThreshold = 0.5, textDetThreshold, includeDetails = false } = options;

    // Step 1: Layout detection
    const layoutElements = this.layout.detect(imagePath, layoutThreshold);

    if (layoutElements.length === 0) {
      return { elements: [], markdown: '' };
    }

    // Step 2: Process each element
    const imageBuffer = readFileSync(imagePath);
    const tmpDir = mkdtempSync(join(tmpdir(), 'structv3-'));

    try {
      const elements: StructuredElement[] = [];

      for (const el of layoutElements) {
        const label = el.labelName;

        if (SKIP_LABELS.has(label)) {
          continue;
        }

        if (TEXT_LABELS.has(label)) {
          // OCR path: detect text lines then recognize
          const cropPath = await this.cropElement(imageBuffer, el, tmpDir);
          const textLines = await this.ocrRegion(cropPath, tmpDir, textDetThreshold);

          const fullText = textLines.map((l) => l.text).join('\n');

          elements.push({
            label,
            score: el.score,
            bbox: el.bbox,
            order: el.order,
            text: fullText,
            lines: includeDetails ? textLines : undefined,
          });
        } else if (label === 'table') {
          // Table: detect text lines in each cell (simplified - full table parsing in Phase 4)
          const cropPath = await this.cropElement(imageBuffer, el, tmpDir);
          const textLines = await this.ocrRegion(cropPath, tmpDir, textDetThreshold);
          const fullText = textLines.map((l) => l.text).join('\n');

          elements.push({
            label,
            score: el.score,
            bbox: el.bbox,
            order: el.order,
            text: fullText,
            lines: includeDetails ? textLines : undefined,
          });
        } else if (label === 'isolate_formula') {
          // Formula: basic OCR for now (Phase 5 adds LaTeX recognition)
          const cropPath = await this.cropElement(imageBuffer, el, tmpDir);
          const textLines = await this.ocrRegion(cropPath, tmpDir, textDetThreshold);
          const fullText = textLines.map((l) => l.text).join(' ');

          elements.push({
            label,
            score: el.score,
            bbox: el.bbox,
            order: el.order,
            text: fullText,
          });
        } else {
          // figure, chart, seal, etc. - placeholder
          elements.push({
            label,
            score: el.score,
            bbox: el.bbox,
            order: el.order,
            text: '',
          });
        }
      }

      // Step 3: Assemble markdown
      const markdown = assembleMarkdown(elements);

      return { elements, markdown };
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  }

  /**
   * Run text detection + recognition on a single image (no layout detection).
   *
   * Useful for processing pre-cropped text regions.
   */
  async ocrImage(imagePath: string, textDetThreshold?: number): Promise<TextLine[]> {
    const tempDir = mkdtempSync(join(tmpdir(), 'ocr-'));
    try {
      return await this.ocrRegion(imagePath, tempDir, textDetThreshold);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }

  /**
   * Detect text lines and recognize text in a cropped region.
   *
   * For each detected text line bounding box, sub-crops that line from the
   * crop image and passes the individual line image to text recognition.
   */
  private async ocrRegion(imagePath: string, tmpDir: string, textDetThreshold?: number): Promise<TextLine[]> {
    // Detect text lines within the crop
    const textBoxes = this.textDet.detect(imagePath, textDetThreshold);

    if (textBoxes.length === 0) {
      // Fall back to recognizing the entire crop as one text line
      const result = this.textRec.recognize(imagePath);
      if (result.text.trim()) {
        return [{ bbox: [0, 0, 0, 0], text: result.text, score: result.score }];
      }
      return [];
    }

    // Sort text boxes by vertical position (top to bottom, left to right)
    const sorted = [...textBoxes].sort((a, b) => {
      const yDiff = a.bbox[1] - b.bbox[1];
      if (Math.abs(yDiff) > 10) return yDiff;
      return a.bbox[0] - b.bbox[0];
    });

    // Single detected line: recognize the full crop directly (no sub-crop needed)
    if (sorted.length === 1) {
      const result = this.textRec.recognize(imagePath);
      return [
        {
          bbox: sorted[0].bbox,
          text: result.text,
          score: result.score,
        },
      ];
    }

    // Multiple detected lines: sub-crop each line from the crop image
    const cropBuffer = readFileSync(imagePath);
    const { Transformer } = await import('@napi-rs/image');

    const linePaths: string[] = [];
    for (let i = 0; i < sorted.length; i++) {
      const [x1, y1, x2, y2] = sorted[i].bbox;
      const x = Math.max(0, Math.round(x1));
      const y = Math.max(0, Math.round(y1));
      const w = Math.max(1, Math.round(x2 - x1));
      const h = Math.max(1, Math.round(y2 - y1));

      const linePng = await new Transformer(cropBuffer).crop(x, y, w, h).png();
      const linePath = join(tmpDir, `line_${i}.png`);
      await writeFile(linePath, linePng);
      linePaths.push(linePath);
    }

    const results = this.textRec.recognizeBatch(linePaths);

    return sorted.map((box, i) => ({
      bbox: box.bbox,
      text: results[i]?.text ?? '',
      score: results[i]?.score ?? 0,
    }));
  }

  /**
   * Crop a layout element from the source image and save to a temporary file.
   */
  private async cropElement(imageBuffer: Buffer, el: LayoutElement, tmpDir: string): Promise<string> {
    const [x1, y1, x2, y2] = el.bbox;
    const x = Math.max(0, Math.round(x1));
    const y = Math.max(0, Math.round(y1));
    const w = Math.max(1, Math.round(x2 - x1));
    const h = Math.max(1, Math.round(y2 - y1));

    // Use @napi-rs/image for cropping
    const { Transformer } = await import('@napi-rs/image');
    const cropped = await new Transformer(imageBuffer).crop(x, y, w, h).png();
    const cropPath = join(tmpDir, `crop_${el.order}.png`);
    await writeFile(cropPath, cropped);
    return cropPath;
  }
}

// ============================================================================
// Markdown assembly
// ============================================================================

/** Format a single element as markdown. */
function formatElement(label: string, text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return '';

  switch (label) {
    case 'doc_title':
      return `# ${trimmed}\n`;
    case 'title':
      return `## ${trimmed}\n`;
    case 'paragraph_title':
      return `### ${trimmed}\n`;
    case 'abstract':
      return `> ${trimmed}\n`;
    case 'table':
      return `${trimmed}\n`;
    case 'table_caption':
    case 'figure_caption':
    case 'chart_caption':
    case 'formula_caption':
      return `*${trimmed}*\n`;
    case 'isolate_formula':
      return `$$\n${trimmed}\n$$\n`;
    case 'code_txt':
      return `\`\`\`\n${trimmed}\n\`\`\`\n`;
    case 'figure':
    case 'chart':
      return trimmed ? `[${label}: ${trimmed}]\n` : `[${label}]\n`;
    case 'header':
    case 'footer':
      return `<!-- ${label}: ${trimmed} -->\n`;
    case 'footnote':
    case 'table_footnote':
      return `[^note]: ${trimmed}\n`;
    case 'list':
      return `${trimmed}\n`;
    case 'seal':
      return `[seal: ${trimmed}]\n`;
    default:
      return `${trimmed}\n`;
  }
}

/** Assemble structured elements into markdown. */
function assembleMarkdown(elements: StructuredElement[]): string {
  const parts: string[] = [];

  for (const el of elements) {
    const formatted = formatElement(el.label, el.text);
    if (formatted) {
      parts.push(formatted);
    }
  }

  return parts.join('\n');
}
