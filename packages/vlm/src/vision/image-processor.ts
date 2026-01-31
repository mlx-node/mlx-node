/**
 * Image Processor for VLM models
 *
 * Re-exports the Rust ImageProcessor from @mlx-node/core and provides
 * TypeScript utility functions for image processing.
 */

// Re-export the Rust implementation from core
export { ImageProcessor, ProcessedImage, smartResize, type ImageProcessorConfig } from '@mlx-node/core';

import type { ImageProcessorConfig } from '@mlx-node/core';
import { ImageProcessor, ProcessedImage } from '@mlx-node/core';

/**
 * Smart resize to maintain aspect ratio within pixel bounds
 *
 * This is a pure TypeScript implementation for environments where
 * the native module is not available.
 *
 * @param height - Original image height
 * @param width - Original image width
 * @param factor - Resize factor (patchSize * mergeSize)
 * @param minPixels - Minimum total pixels
 * @param maxPixels - Maximum total pixels
 * @returns [newHeight, newWidth] resized dimensions
 */
export function smartResizeJs(
  height: number,
  width: number,
  factor: number,
  minPixels: number,
  maxPixels: number,
): [number, number] {
  // Validate inputs to prevent division by zero
  if (height <= 0) {
    throw new Error(`height must be positive, got ${height}`);
  }
  if (width <= 0) {
    throw new Error(`width must be positive, got ${width}`);
  }
  if (factor <= 0) {
    throw new Error(`factor must be positive, got ${factor}`);
  }

  // Ensure minimum dimensions
  let h = height;
  let w = width;

  if (h < factor) {
    w = Math.round((w * factor) / h);
    h = factor;
  }
  if (w < factor) {
    h = Math.round((h * factor) / w);
    w = factor;
  }

  // Check aspect ratio
  const aspectRatio = Math.max(h, w) / Math.min(h, w);
  if (aspectRatio > 200) {
    throw new Error(`Absolute aspect ratio must be smaller than 200, got ${aspectRatio}`);
  }

  // Round to factor
  let hBar = Math.round(h / factor) * factor;
  let wBar = Math.round(w / factor) * factor;

  // Adjust to fit within pixel bounds
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((h * w) / maxPixels);
    hBar = Math.floor(h / beta / factor) * factor;
    wBar = Math.floor(w / beta / factor) * factor;
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (h * w));
    hBar = Math.ceil((h * beta) / factor) * factor;
    wBar = Math.ceil((w * beta) / factor) * factor;
  }

  return [hBar, wBar];
}

/**
 * Default image processing configuration
 */
export const DEFAULT_IMAGE_CONFIG: Required<ImageProcessorConfig> = {
  minPixels: 147384,
  maxPixels: 2822400,
  patchSize: 14,
  temporalPatchSize: 1,
  mergeSize: 2,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  doRescale: true,
  doNormalize: true,
};

/**
 * Convenience function to preprocess an image with default settings
 *
 * @param imagePath - Path to the image file
 * @param config - Optional configuration overrides
 * @returns Processed image ready for model input
 */
export function preprocessImage(imagePath: string, config?: ImageProcessorConfig): ProcessedImage {
  const processor = new ImageProcessor(config);
  return processor.processFile(imagePath);
}

/**
 * Convenience function to preprocess image bytes with default settings
 *
 * @param imageData - Image data as Buffer
 * @param config - Optional configuration overrides
 * @returns Processed image ready for model input
 */
export function preprocessImageBytes(imageData: Buffer, config?: ImageProcessorConfig): ProcessedImage {
  const processor = new ImageProcessor(config);
  return processor.processBytes(imageData);
}
