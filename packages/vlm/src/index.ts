/**
 * @mlx-node/vlm - Vision Language Model support for MLX-Node
 *
 * This package provides VLM capabilities including:
 * - VLModel for OCR and document understanding tasks
 *
 * @example
 * ```typescript
 * import { VLModel } from '@mlx-node/vlm';
 *
 * // Load a model
 * const model = await VLModel.load('./models/paddleocr-vl');
 *
 * // Chat with images
 * const imageBuffer = readFileSync('./photo.jpg');
 * const result = model.chat(
 *   [{ role: 'user', content: 'What is in this image?' }],
 *   { images: [imageBuffer] }
 * );
 * console.log(result.text);
 *
 * // Simple OCR
 * const text = model.ocr(readFileSync('./document.jpg'));
 *
 * // Batch OCR (multiple images)
 * const texts = model.ocrBatch([readFileSync('page1.jpg'), readFileSync('page2.jpg')]);
 * ```
 */

// ============== PUBLIC API ==============

// Main model class and factory functions (exposed directly from Rust)
export { VLModel, createPaddleocrVlConfig } from '@mlx-node/core';

// Document layout analysis model
export { DocLayoutModel, type LayoutElement } from '@mlx-node/core';

// Text detection and recognition models (PP-OCRv5)
export { TextDetModel, type TextBox, TextRecModel, type RecResult } from '@mlx-node/core';

// Document preprocessing models
export {
  DocOrientationModel,
  type OrientationResult,
  type ClassifyRotateResult,
  DocUnwarpModel,
  type UnwarpResult,
} from '@mlx-node/core';

// Document understanding pipeline (PP-StructureV3)
export {
  StructureV3Pipeline,
  type StructureV3Config,
  type AnalyzeOptions,
  type StructuredElement,
  type StructuredDocument,
  type TextLine,
} from './pipeline/structure-v3.js';

// Configuration types
export type {
  VisionConfig,
  TextConfig,
  ModelConfig,
  VlmChatConfig,
  VlmChatMessage,
  VlmBatchItem,
} from '@mlx-node/core';

// Model-specific configs
export { PADDLEOCR_VL_CONFIGS, type PaddleOCRVLConfig } from './models/paddleocr-vl-configs.js';

// Chat result type
export { VlmChatResult, type VLMChatResult } from '@mlx-node/core';

// Output parsing and formatting (Rust implementation)
export {
  parsePaddleResponse,
  parseVlmOutput,
  formatDocument,
  type ParsedDocument,
  type DocumentElement,
  type Table,
  type TableRow,
  type TableCell,
  type Paragraph,
  type ParserConfig,
  OutputFormat,
} from '@mlx-node/core';

// XLSX export (Rust implementation)
export { documentToXlsx, saveToXlsx } from '@mlx-node/core';

// Re-export shared utilities
export { Qwen3Tokenizer as Tokenizer } from '@mlx-node/lm';
export { MxArray, type DType } from '@mlx-node/core';
