/**
 * Document Layout Analysis + OCR Pipeline
 *
 * Combines PP-DocLayoutV3 (layout detection) with PaddleOCR-VL-1.5 (OCR)
 * to extract structured markdown from document images.
 *
 * Pipeline:
 *   0. (Optional) Preprocessing: orientation correction + unwarping
 *   1. DocLayoutModel detects layout elements (titles, text, tables, figures...)
 *   2. Each element is cropped from the source image
 *   3. VLModel OCRs each cropped region with type-appropriate prompts
 *   4. Results are assembled into formatted markdown following reading order
 *
 * Usage:
 *   oxnode examples/vlm-inference.ts <image_path> [--threshold 0.5] [--vlm-only] [--layout-only]
 *
 * Examples:
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png                # Full pipeline
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png --vlm-only     # VLM OCR only
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png --layout-only  # Layout detection only
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png --threshold 0.3
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png --orient       # With orientation correction
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png --unwarp       # With document unwarping
 */
import { readFileSync, existsSync } from 'node:fs';
import { ChatRole, DocOrientationModel, DocUnwarpModel, type LayoutElement } from '@mlx-node/core';
import { DocLayoutModel, VLModel, parsePaddleResponse, OutputFormat } from '@mlx-node/vlm';
import { Transformer } from '@napi-rs/image';

// --- CLI args ---
const args = process.argv.slice(2);
const imagePath = args.find((a) => !a.startsWith('--'));
const vlmOnly = args.includes('--vlm-only');
const layoutOnly = args.includes('--layout-only');
const useOrient = args.includes('--orient');
const useUnwarp = args.includes('--unwarp');
const thresholdIdx = args.indexOf('--threshold');
const threshold =
  thresholdIdx !== -1 && !isNaN(parseFloat(args[thresholdIdx + 1])) ? parseFloat(args[thresholdIdx + 1]) : 0.5;

const vlmModelPath = '.cache/models/PaddleOCR-VL-1.5-mlx';
// Clone from HuggingFace: PaddlePaddle/PP-DocLayoutV3_safetensors (NOT PaddlePaddle/PP-DocLayoutV3)
const layoutModelPath = '.cache/models/PP-DocLayoutV3';
const orientModelPath = '.cache/models/PP-LCNet_x1_0_doc_ori-mlx';
const unwarpModelPath = '.cache/models/UVDoc-mlx';

function tryLoadOrient(): DocOrientationModel | null {
  if (!useOrient) return null;
  if (!existsSync(orientModelPath)) {
    console.error(`Error: Orientation model not found at ${orientModelPath}`);
    console.error('  Download and convert it first:');
    console.error(
      '    oxnode scripts/download-model.ts -m PaddlePaddle/PP-LCNet_x1_0_doc_ori -o .cache/models/PP-LCNet_x1_0_doc_ori',
    );
    console.error(
      '    oxnode scripts/convert-model.ts -m pp-lcnet-ori -i .cache/models/PP-LCNet_x1_0_doc_ori -o .cache/models/PP-LCNet_x1_0_doc_ori-mlx',
    );
    process.exit(1);
  }
  return DocOrientationModel.load(orientModelPath);
}

function tryLoadUnwarp(): DocUnwarpModel | null {
  if (!useUnwarp) return null;
  if (!existsSync(unwarpModelPath)) {
    console.error(`Error: Unwarp model not found at ${unwarpModelPath}`);
    console.error('  Download and convert it first:');
    console.error('    oxnode scripts/download-model.ts -m PaddlePaddle/UVDoc -o .cache/models/UVDoc');
    console.error('    oxnode scripts/convert-model.ts -m uvdoc -i .cache/models/UVDoc -o .cache/models/UVDoc-mlx');
    process.exit(1);
  }
  return DocUnwarpModel.load(unwarpModelPath);
}

if (!imagePath) {
  console.log('Usage: oxnode examples/vlm-inference.ts <image_path> [options]');
  console.log('Options:');
  console.log('  --vlm-only       VLM OCR only (no layout detection)');
  console.log('  --layout-only    Layout detection only (no OCR)');
  console.log('  --orient         Enable document orientation correction (0/90/180/270)');
  console.log('  --unwarp         Enable document unwarping (curved/distorted pages)');
  console.log('  --threshold N    Detection confidence threshold (default: 0.5)');
  process.exit(1);
}

// --- VLM-only mode (original behavior) ---
if (vlmOnly) {
  console.log('Loading VLM...');
  console.time('VLM load');
  const vlm = await VLModel.load(vlmModelPath);
  console.timeEnd('VLM load');

  console.log(`\n--- OCR: ${imagePath} ---`);
  console.time('OCR');
  const imageBuffer = readFileSync(imagePath);
  const result = vlm.chat([{ role: ChatRole.User, content: 'Extract the text in this image' }], {
    images: [imageBuffer],
  });
  console.timeEnd('OCR');

  const formatted = parsePaddleResponse(result.text, { format: OutputFormat.Markdown });
  console.log(`\n${formatted}`);
  process.exit(0);
}

// --- Layout-only mode ---
if (layoutOnly) {
  console.log('Loading models...');
  console.time('Models load');
  const layout = DocLayoutModel.load(layoutModelPath);
  const orientModel = tryLoadOrient();
  const unwarpModel = tryLoadUnwarp();
  console.timeEnd('Models load');

  let imageData: Buffer = readFileSync(imagePath);

  if (orientModel) {
    console.log('\n--- Orientation Classification ---');
    const rotateResult = orientModel.classifyAndRotate(imageData);
    console.log(`  Detected: ${rotateResult.angle}° (score: ${rotateResult.score.toFixed(3)})`);
    if (rotateResult.angle !== 0) {
      imageData = Buffer.from(rotateResult.image);
      console.log(`  Corrected to upright`);
    }
  }

  if (unwarpModel) {
    console.log('\n--- Document Unwarping ---');
    const unwarpResult = unwarpModel.unwarp(imageData);
    imageData = Buffer.from(unwarpResult.image);
    console.log(`  Unwarped`);
  }

  console.log(`\n--- Layout Detection: ${imagePath} (threshold=${threshold}) ---`);
  console.time('Detection');
  const elements = layout.detect(imageData, threshold);
  console.timeEnd('Detection');

  console.log(`\nDetected ${elements.length} elements:\n`);
  for (const el of elements) {
    const [x1, y1, x2, y2] = el.bbox;
    console.log(
      `  [${el.order}] ${el.labelName} (${el.label}) - score: ${el.score.toFixed(3)} ` +
        `bbox: [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}]`,
    );
  }

  process.exit(0);
}

// --- Full pipeline: Layout + OCR → Markdown ---
console.log('Loading models...');
console.time('Models load');
const layout = DocLayoutModel.load(layoutModelPath);
const vlm = await VLModel.load(vlmModelPath);
const orientModel = tryLoadOrient();
const unwarpModel = tryLoadUnwarp();
console.timeEnd('Models load');

// Step 0: Preprocessing (orientation correction + unwarping)
let processedImage: Buffer = readFileSync(imagePath);

if (orientModel) {
  console.log('\n--- Orientation Classification ---');
  console.time('Orientation');
  const rotateResult = orientModel.classifyAndRotate(processedImage);
  console.timeEnd('Orientation');
  console.log(`  Detected: ${rotateResult.angle}° (score: ${rotateResult.score.toFixed(3)})`);

  if (rotateResult.angle !== 0) {
    processedImage = Buffer.from(rotateResult.image);
    console.log(`  Corrected to upright`);
  }
}

if (unwarpModel) {
  console.log('\n--- Document Unwarping ---');
  console.time('Unwarp');
  const unwarpResult = unwarpModel.unwarp(processedImage);
  processedImage = Buffer.from(unwarpResult.image);
  console.timeEnd('Unwarp');
  console.log(`  Unwarped`);
}

// Step 1: Detect layout
console.log(`\n--- Layout Detection (threshold=${threshold}) ---`);
console.time('Detection');
const elements = layout.detect(processedImage, threshold);
console.timeEnd('Detection');
console.log(`Detected ${elements.length} elements`);

if (elements.length === 0) {
  console.log('No elements detected. Try lowering the threshold with --threshold 0.3');
  process.exit(0);
}

// Step 2: Crop regions and OCR each element
async function cropElement(el: LayoutElement): Promise<Buffer> {
  const [x1, y1, x2, y2] = el.bbox;
  const x = Math.max(0, Math.round(x1));
  const y = Math.max(0, Math.round(y1));
  const w = Math.max(1, Math.round(x2 - x1));
  const h = Math.max(1, Math.round(y2 - y1));

  return Buffer.from(await new Transformer(processedImage).crop(x, y, w, h).png());
}

/** Get OCR prompt based on element type */
function getPrompt(labelName: string): string {
  switch (labelName) {
    case 'table':
      return 'Extract this table as a markdown table. Preserve all rows and columns.';
    case 'isolate_formula':
      return 'Extract this mathematical formula in LaTeX notation.';
    case 'code_txt':
      return 'Extract this code exactly as shown, preserving indentation.';
    case 'chart':
    case 'figure':
      return 'Describe what is shown in this image briefly.';
    default:
      return 'Extract the text in this image exactly as shown.';
  }
}

/** Elements that should be OCR'd */
const ocrLabels = new Set([
  'title',
  'doc_title',
  'paragraph_title',
  'text',
  'abstract',
  'list',
  'table',
  'table_caption',
  'table_footnote',
  'figure_caption',
  'chart_caption',
  'isolate_formula',
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

/** Format OCR text based on element type */
function formatElement(labelName: string, text: string, order: number): string {
  const trimmed = text.trim();
  if (!trimmed) return '';

  switch (labelName) {
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
      return `[${labelName}: ${trimmed}]\n`;
    case 'header':
    case 'footer':
      return `<!-- ${labelName}: ${trimmed} -->\n`;
    case 'footnote':
    case 'table_footnote':
      return `[^note-${order}]: ${trimmed}\n`;
    case 'list':
      return `${trimmed}\n`;
    default:
      return `${trimmed}\n`;
  }
}

// Step 3: Crop all elements and prepare batch OCR items
type OcrItem = { index: number; label: string; cropBuffer: Buffer; prompt: string; order: number };
const ocrItems: OcrItem[] = [];
const nonOcrParts: Map<number, string> = new Map(); // index -> markdown for non-OCR elements

let idx = 0;
for (const el of elements) {
  const label = el.labelName;

  // Skip non-content elements
  if (label === 'abandon' || label === 'seal') {
    continue;
  }

  // For figures/charts without text, just note their position
  if (label === 'figure' || label === 'chart') {
    nonOcrParts.set(idx, `[${label}]\n`);
    idx++;
    continue;
  }

  if (!ocrLabels.has(label)) {
    continue;
  }

  const cropBuffer = await cropElement(el);
  const prompt = getPrompt(label);
  console.log(`  [${el.order}] ${label} (${el.score.toFixed(2)})`);
  ocrItems.push({ index: idx, label, cropBuffer, prompt, order: el.order });
  idx++;
}

// Step 4: Batch OCR all cropped elements
console.log(`\n--- Batch OCR (${ocrItems.length} elements) ---`);
console.time('Batch OCR');

const batchItems = ocrItems.map((item) => ({
  messages: [{ role: ChatRole.User as const, content: item.prompt }],
  images: [item.cropBuffer],
}));
const batchResults = vlm.batch(batchItems);

console.timeEnd('Batch OCR');

// Step 5: Assemble markdown in reading order
const markdownParts: string[] = [];
let ocrResultIdx = 0;

for (let i = 0; i < idx; i++) {
  if (nonOcrParts.has(i)) {
    markdownParts.push(nonOcrParts.get(i)!);
  } else {
    const item = ocrItems[ocrResultIdx];
    const result = batchResults[ocrResultIdx];
    const text = parsePaddleResponse(result.text, { format: OutputFormat.Markdown });
    const formatted = formatElement(item.label, text, item.order);
    if (formatted) {
      markdownParts.push(formatted);
    }
    ocrResultIdx++;
  }
}

// Output
const markdown = markdownParts.join('\n');
console.log('\n--- Result ---\n');
console.log(markdown);
