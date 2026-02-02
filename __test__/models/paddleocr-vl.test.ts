/**
 * PaddleOCR-VL NAPI Binding Smoke Tests
 *
 * Minimal tests to verify NAPI bindings work correctly.
 * Full unit tests are in Rust: crates/mlx-core/src/models/paddleocr_vl/
 */
import { describe, expect, it } from 'vite-plus/test';
import { VLModel, createPaddleocrVlConfig } from '@mlx-node/vlm';

describe('PaddleOCR-VL NAPI Bindings', () => {
  it('should create config via NAPI', () => {
    const config = createPaddleocrVlConfig();
    expect(config.modelType).toBe('paddleocr_vl');
    expect(config.imageTokenId).toBe(100295);
  });

  it('should create model via NAPI', () => {
    const config = createPaddleocrVlConfig();
    const model = new VLModel(config);
    expect(model.isInitialized).toBe(false);
  });
});
