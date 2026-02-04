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
 * const result = model.chat(
 *   [{ role: 'user', content: 'What is in these images?' }],
 *   { imagePaths: ['./photo1.jpg', './photo2.jpg'] }
 * );
 * console.log(result.text);
 *
 * // Simple OCR
 * const text = model.ocr('./document.jpg');
 * ```
 */

// ============== PUBLIC API ==============

// Main model class and factory functions (exposed directly from Rust)
export { VLModel, createPaddleocrVlConfig } from '@mlx-node/core';

// Configuration types
export type { VisionConfig, TextConfig, ModelConfig, VlmChatConfig, VlmChatMessage } from '@mlx-node/core';

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
