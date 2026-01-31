/**
 * Vision utilities exports
 *
 * Re-exports the Rust ImageProcessor from @mlx-node/core with
 * additional TypeScript convenience functions.
 */

// Re-export from the Rust bindings (via image-processor.ts wrapper)
export {
  ImageProcessor,
  ProcessedImage,
  smartResize,
  preprocessImage,
  preprocessImageBytes,
  DEFAULT_IMAGE_CONFIG,
  type ImageProcessorConfig,
} from './image-processor.js';

// Note: smartResizeJs is kept in image-processor.ts but not exported publicly
// since the Rust smartResize is the canonical implementation
