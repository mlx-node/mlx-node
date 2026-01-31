/**
 * Vision types for VLM processing
 */

/**
 * Supported image input formats
 */
export type ImageInput = Buffer | Uint8Array | string;

/**
 * Image processing configuration
 */
export interface ImageConfig {
  /** Minimum pixels for image (default: 147384) */
  minPixels?: number;
  /** Maximum pixels for image (default: 2822400) */
  maxPixels?: number;
  /** Patch size for vision transformer (default: 14) */
  patchSize?: number;
  /** Temporal patch size for video (default: 1) */
  temporalPatchSize?: number;
  /** Spatial merge size for token reduction (default: 2) */
  mergeSize?: number;
  /** Image normalization mean (default: [0.5, 0.5, 0.5]) */
  imageMean?: [number, number, number];
  /** Image normalization std (default: [0.5, 0.5, 0.5]) */
  imageStd?: [number, number, number];
  /** Whether to rescale pixel values to [0, 1] (default: true) */
  doRescale?: boolean;
  /** Whether to normalize pixel values (default: true) */
  doNormalize?: boolean;
  /** Whether to convert to RGB (default: true) */
  doConvertRgb?: boolean;
}

/**
 * Processed image ready for model input
 */
export interface ProcessedImage {
  /** Pixel values as Float32Array, shape [batch, seq, channels, patch_h, patch_w] */
  pixelValues: Float32Array;
  /** Shape of the pixel values array */
  pixelValuesShape: bigint[];
  /** Grid dimensions [temporal, height, width] for each image */
  imageGridThw: Int32Array;
  /** Shape of the grid array */
  imageGridThwShape: bigint[];
}
