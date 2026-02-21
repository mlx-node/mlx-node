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
import {
  DocLayoutModel,
  TextDetModel,
  TextRecModel,
  DocOrientationModel,
  DocUnwarpModel,
  type LayoutElement,
} from '@mlx-node/core';

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
  /** Path to doc orientation classification model directory (optional) */
  docOrientationModelPath?: string;
  /** Path to doc unwarping model directory (optional) */
  docUnwarpModelPath?: string;
}

/** Options for document analysis. */
export interface AnalyzeOptions {
  /** Layout detection confidence threshold (default: 0.5) */
  layoutThreshold?: number;
  /** Text detection confidence threshold (default: 0.3) */
  textDetThreshold?: number;
  /** Whether to include element-level details in output (default: false) */
  includeDetails?: boolean;
  /** Whether to run document orientation classification (default: true if model loaded) */
  useDocOrientationClassify?: boolean;
  /** Whether to run document unwarping (default: true if model loaded) */
  useDocUnwarping?: boolean;
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
  private docOrientation: DocOrientationModel | null;
  private docUnwarp: DocUnwarpModel | null;

  private constructor(
    layout: DocLayoutModel,
    textDet: TextDetModel,
    textRec: TextRecModel,
    docOrientation: DocOrientationModel | null,
    docUnwarp: DocUnwarpModel | null,
  ) {
    this.layout = layout;
    this.textDet = textDet;
    this.textRec = textRec;
    this.docOrientation = docOrientation;
    this.docUnwarp = docUnwarp;
  }

  /**
   * Load all models and create the pipeline.
   */
  static load(config: StructureV3Config): StructureV3Pipeline {
    const layout = DocLayoutModel.load(config.layoutModelPath);
    const textDet = TextDetModel.load(config.textDetModelPath);
    const textRec = TextRecModel.load(config.textRecModelPath, config.dictPath);

    const docOrientation = config.docOrientationModelPath
      ? DocOrientationModel.load(config.docOrientationModelPath)
      : null;
    const docUnwarp = config.docUnwarpModelPath ? DocUnwarpModel.load(config.docUnwarpModelPath) : null;

    return new StructureV3Pipeline(layout, textDet, textRec, docOrientation, docUnwarp);
  }

  /**
   * Analyze a document image and extract structured content.
   *
   * @param imageData - Buffer with encoded image bytes, or a file path string
   * @param options - Analysis options
   * @returns Structured document with elements and markdown
   */
  async analyze(imageData: Buffer | string, options: AnalyzeOptions = {}): Promise<StructuredDocument> {
    const { layoutThreshold = 0.5, textDetThreshold, includeDetails = false } = options;

    let imageBuffer: Buffer = typeof imageData === 'string' ? readFileSync(imageData) : imageData;

    // Step 0a: Document orientation correction
    if (this.docOrientation && (options.useDocOrientationClassify ?? true)) {
      const rotateResult = this.docOrientation.classifyAndRotate(imageBuffer);
      if (rotateResult.angle !== 0) {
        imageBuffer = Buffer.from(rotateResult.image);
      }
    }

    // Step 0b: Document unwarping
    if (this.docUnwarp && (options.useDocUnwarping ?? true)) {
      const unwarpResult = this.docUnwarp.unwarp(imageBuffer);
      imageBuffer = Buffer.from(unwarpResult.image);
    }

    // Step 1: Layout detection (on preprocessed image)
    const layoutElements = this.layout.detect(imageBuffer, layoutThreshold);

    if (layoutElements.length === 0) {
      return { elements: [], markdown: '' };
    }

    // Step 2: Process each element
    const elements: StructuredElement[] = [];

    for (const el of layoutElements) {
      const label = el.labelName;

      if (SKIP_LABELS.has(label)) {
        continue;
      }

      if (TEXT_LABELS.has(label)) {
        // OCR path: detect text lines then recognize
        const cropBuffer = await this.cropElement(imageBuffer, el);
        const textLines = await this.ocrRegion(cropBuffer, textDetThreshold);

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
        const cropBuffer = await this.cropElement(imageBuffer, el);
        const textLines = await this.ocrRegion(cropBuffer, textDetThreshold);
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
        const cropBuffer = await this.cropElement(imageBuffer, el);
        const textLines = await this.ocrRegion(cropBuffer, textDetThreshold);
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
  }

  /**
   * Run text detection + recognition on a single image (no layout detection).
   *
   * Useful for processing pre-cropped text regions.
   */
  async ocrImage(imageData: Buffer, textDetThreshold?: number): Promise<TextLine[]> {
    return this.ocrRegion(imageData, textDetThreshold);
  }

  /**
   * Detect text lines and recognize text in a cropped region.
   *
   * For each detected text line bounding box, sub-crops that line from the
   * crop image and passes the individual line image to text recognition.
   */
  private async ocrRegion(imageData: Buffer, textDetThreshold?: number): Promise<TextLine[]> {
    // Detect text lines within the crop
    const textBoxes = this.textDet.detect(imageData, textDetThreshold);

    if (textBoxes.length === 0) {
      // Fall back to recognizing the entire crop as one text line
      const result = this.textRec.recognize(imageData);
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
      const result = this.textRec.recognize(imageData);
      return [
        {
          bbox: sorted[0].bbox,
          text: result.text,
          score: result.score,
        },
      ];
    }

    // Multiple detected lines: sub-crop each line from the crop image
    const { Transformer } = await import('@napi-rs/image');

    const lineBuffers: Buffer[] = [];
    for (let i = 0; i < sorted.length; i++) {
      const [x1, y1, x2, y2] = sorted[i].bbox;
      const x = Math.max(0, Math.round(x1));
      const y = Math.max(0, Math.round(y1));
      const w = Math.max(1, Math.round(x2 - x1));
      const h = Math.max(1, Math.round(y2 - y1));

      const linePng = await new Transformer(imageData).crop(x, y, w, h).png();
      lineBuffers.push(Buffer.from(linePng));
    }

    const results = this.textRec.recognizeBatch(lineBuffers);

    return sorted.map((box, i) => ({
      bbox: box.bbox,
      text: results[i]?.text ?? '',
      score: results[i]?.score ?? 0,
    }));
  }

  /**
   * Crop a layout element from the source image and return PNG bytes.
   */
  private async cropElement(imageBuffer: Buffer, el: LayoutElement): Promise<Buffer> {
    const [x1, y1, x2, y2] = el.bbox;
    const x = Math.max(0, Math.round(x1));
    const y = Math.max(0, Math.round(y1));
    const w = Math.max(1, Math.round(x2 - x1));
    const h = Math.max(1, Math.round(y2 - y1));

    const { Transformer } = await import('@napi-rs/image');
    const cropped = await new Transformer(imageBuffer).crop(x, y, w, h).png();
    return Buffer.from(cropped);
  }
}

// ============================================================================
// Markdown assembly
// ============================================================================

/** Format a single element as markdown. */
function formatElement(label: string, text: string, order: number): string {
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
      return `[^note-${order}]: ${trimmed}\n`;
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
    const formatted = formatElement(el.label, el.text, el.order);
    if (formatted) {
      parts.push(formatted);
    }
  }

  return parts.join('\n');
}
