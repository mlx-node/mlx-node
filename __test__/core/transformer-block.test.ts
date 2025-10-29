import { describe, it, expect } from 'vitest';
import { TransformerBlock, KVCache, MxArray } from '@mlx-node/core';
import { shape, assertShape, assertFinite } from '../test-utils';

describe('TransformerBlock', () => {
  describe('Construction', () => {
    it('should create transformer block with standard config', () => {
      const block = new TransformerBlock(
        512, // hidden_size
        8, // num_heads
        8, // num_kv_heads
        2048, // intermediate_size
        1e-5, // rms_norm_eps
      );
      expect(block).toBeDefined();
    });

    it('should create transformer block with GQA', () => {
      const block = new TransformerBlock(
        512, // hidden_size
        8, // num_heads
        2, // num_kv_heads (GQA)
        2048, // intermediate_size
        1e-6, // rms_norm_eps
      );
      expect(block).toBeDefined();
    });

    it('should create transformer block with custom RoPE theta', () => {
      const block = new TransformerBlock(
        256,
        4,
        4,
        1024,
        1e-5,
        100000, // rope_theta
      );
      expect(block).toBeDefined();
    });

    it('should create transformer block with QK normalization', () => {
      const block = new TransformerBlock(
        512,
        8,
        8,
        2048,
        1e-5,
        10000, // rope_theta
        true, // use_qk_norm
      );
      expect(block).toBeDefined();
    });

    it('should create transformer block with custom head_dim', () => {
      const block = new TransformerBlock(
        512,
        8,
        8,
        2048,
        1e-5,
        10000,
        false,
        80, // head_dim
      );
      expect(block).toBeDefined();
    });
  });

  describe('Forward Pass - Shape Verification', () => {
    it('should maintain correct output shape', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const x = MxArray.randomNormal(shape(2, 10, 256), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 10, 256]); // Same as input
      assertFinite(output);
    });

    it('should handle single token', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(1, 1, 128), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 1, 128]);
      assertFinite(output);
    });

    it('should handle batch size of 1', () => {
      const block = new TransformerBlock(512, 8, 8, 2048, 1e-5);
      const x = MxArray.randomNormal(shape(1, 20, 512), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 20, 512]);
      assertFinite(output);
    });

    it('should handle larger batch sizes', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const x = MxArray.randomNormal(shape(8, 16, 256), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [8, 16, 256]);
      assertFinite(output);
    });

    it('should handle different sequence lengths', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);

      const lengths = [1, 5, 16, 32, 64];
      for (const seq_len of lengths) {
        const x = MxArray.randomNormal(shape(2, seq_len, 256), 0, 0.02);
        const output = block.forward(x, null, null);
        assertShape(output, [2, seq_len, 256]);
        assertFinite(output);
      }
    });
  });

  describe('Residual Connections', () => {
    it('should apply residual connections correctly', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(1, 8, 128), 0, 0.02);

      const output = block.forward(x, null, null);

      // Output should be different from input (due to transformations)
      assertShape(output, [1, 8, 128]);

      const x_data = x.toFloat32();
      const out_data = output.toFloat32();

      // Should not be identical (residual adds to transformed values)
      const areIdentical = Array.from(x_data).every((v, i) => Math.abs(v - out_data[i]) < 1e-8);
      expect(areIdentical).toBe(false);
    });

    it('should preserve information through residual connections', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const x = MxArray.randomNormal(shape(2, 10, 256), 0, 0.5);

      const output = block.forward(x, null, null);

      assertFinite(output);

      // Output should have reasonable magnitude (not exploding/vanishing)
      const out_data = output.toFloat32();
      const mean = Array.from(out_data).reduce((a, b) => a + b, 0) / out_data.length;
      const variance = Array.from(out_data).reduce((a, b) => a + (b - mean) ** 2, 0) / out_data.length;

      expect(Math.abs(mean)).toBeLessThan(2.0); // Reasonable mean
      expect(variance).toBeGreaterThan(0.01); // Not collapsed
      expect(variance).toBeLessThan(10.0); // Not exploded
    });
  });

  describe('KV Cache Support', () => {
    it('should work with KV cache on first call', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const cache = new KVCache();
      const x = MxArray.randomNormal(shape(1, 5, 256), 0, 0.02);

      const output = block.forward(x, null, cache);

      assertShape(output, [1, 5, 256]);
      expect(cache.getOffset()).toBe(5);
      assertFinite(output);
    });

    it('should accumulate tokens with KV cache', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const cache = new KVCache();

      // First call: process 10 tokens
      const x1 = MxArray.randomNormal(shape(1, 10, 256), 0, 0.02);
      const output1 = block.forward(x1, null, cache);
      assertShape(output1, [1, 10, 256]);
      expect(cache.getOffset()).toBe(10);

      // Second call: process 1 more token (incremental)
      const x2 = MxArray.randomNormal(shape(1, 1, 256), 0, 0.02);
      const output2 = block.forward(x2, null, cache);
      assertShape(output2, [1, 1, 256]);
      expect(cache.getOffset()).toBe(11);
    });

    it('should handle incremental generation', () => {
      const block = new TransformerBlock(128, 2, 2, 512, 1e-5);
      const cache = new KVCache();

      // Initial prompt
      let x = MxArray.randomNormal(shape(1, 8, 128), 0, 0.02);
      let output = block.forward(x, null, cache);
      expect(cache.getOffset()).toBe(8);

      // Generate 5 tokens one by one
      for (let i = 0; i < 5; i++) {
        x = MxArray.randomNormal(shape(1, 1, 128), 0, 0.02);
        output = block.forward(x, null, cache);
        expect(cache.getOffset()).toBe(8 + i + 1);
        assertShape(output, [1, 1, 128]);
      }

      expect(cache.getOffset()).toBe(13);
    });

    it('should work with KV cache and GQA', () => {
      const block = new TransformerBlock(512, 8, 2, 2048, 1e-5); // GQA
      const cache = new KVCache();

      const x1 = MxArray.randomNormal(shape(1, 10, 512), 0, 0.02);
      const output1 = block.forward(x1, null, cache);
      assertShape(output1, [1, 10, 512]);
      expect(cache.getOffset()).toBe(10);

      const x2 = MxArray.randomNormal(shape(1, 1, 512), 0, 0.02);
      const output2 = block.forward(x2, null, cache);
      assertShape(output2, [1, 1, 512]);
      expect(cache.getOffset()).toBe(11);
    });

    it('should reset cache between sequences', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const cache = new KVCache();

      // First sequence
      const x1 = MxArray.randomNormal(shape(1, 10, 128), 0, 0.02);
      block.forward(x1, null, cache);
      expect(cache.getOffset()).toBe(10);

      // Reset and start new sequence
      cache.reset();
      expect(cache.getOffset()).toBe(0);

      // Second sequence
      const x2 = MxArray.randomNormal(shape(1, 5, 128), 0, 0.02);
      block.forward(x2, null, cache);
      expect(cache.getOffset()).toBe(5);
    });
  });

  describe('Different Configurations', () => {
    it('should work with small dimensions', () => {
      const block = new TransformerBlock(64, 2, 2, 256, 1e-5);
      const x = MxArray.randomNormal(shape(1, 8, 64), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 8, 64]);
      assertFinite(output);
    });

    it('should work with large dimensions', () => {
      const block = new TransformerBlock(1024, 16, 16, 4096, 1e-5);
      const x = MxArray.randomNormal(shape(1, 4, 1024), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 4, 1024]);
      assertFinite(output);
    });

    it('should work with QK normalization enabled', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5, 10000, true);
      const x = MxArray.randomNormal(shape(2, 10, 256), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 10, 256]);
      assertFinite(output);
    });

    it('should work with custom RoPE theta', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5, 100000);
      const x = MxArray.randomNormal(shape(1, 16, 256), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 16, 256]);
      assertFinite(output);
    });

    it('should work with non-standard head dimension', () => {
      const block = new TransformerBlock(512, 8, 8, 2048, 1e-5, 10000, false, 80);
      const x = MxArray.randomNormal(shape(1, 10, 512), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 10, 512]);
      assertFinite(output);
    });
  });

  describe('Multiple Forward Passes', () => {
    it('should be consistent across multiple calls', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(1, 10, 128), 0, 0.02);

      const output1 = block.forward(x, null, null);
      const output2 = block.forward(x, null, null);

      assertShape(output1, [1, 10, 128]);
      assertShape(output2, [1, 10, 128]);

      // Results should be identical (deterministic)
      const data1 = output1.toFloat32();
      const data2 = output2.toFloat32();

      expect(Array.from(data1)).toEqual(Array.from(data2));
    });

    it('should handle different inputs sequentially', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);

      const x1 = MxArray.randomNormal(shape(2, 8, 256), 0, 0.02);
      const x2 = MxArray.randomNormal(shape(3, 12, 256), 0, 0.02);
      const x3 = MxArray.randomNormal(shape(1, 5, 256), 0, 0.02);

      const out1 = block.forward(x1, null, null);
      const out2 = block.forward(x2, null, null);
      const out3 = block.forward(x3, null, null);

      assertShape(out1, [2, 8, 256]);
      assertShape(out2, [3, 12, 256]);
      assertShape(out3, [1, 5, 256]);
    });
  });

  describe('Stacking Multiple Blocks', () => {
    it('should stack 2 transformer blocks', () => {
      const block1 = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const block2 = new TransformerBlock(256, 4, 4, 1024, 1e-5);

      let x = MxArray.randomNormal(shape(2, 10, 256), 0, 0.02);
      x = block1.forward(x, null, null);
      x = block2.forward(x, null, null);

      assertShape(x, [2, 10, 256]);
      assertFinite(x);
    });

    it('should stack 4 transformer blocks', () => {
      const blocks = [
        new TransformerBlock(128, 4, 4, 512, 1e-5),
        new TransformerBlock(128, 4, 4, 512, 1e-5),
        new TransformerBlock(128, 4, 4, 512, 1e-5),
        new TransformerBlock(128, 4, 4, 512, 1e-5),
      ];

      let x = MxArray.randomNormal(shape(1, 16, 128), 0, 0.02);

      for (const block of blocks) {
        x = block.forward(x, null, null);
        assertShape(x, [1, 16, 128]);
        assertFinite(x);
      }
    });

    it('should stack blocks with KV cache', () => {
      const blocks = [new TransformerBlock(128, 4, 4, 512, 1e-5), new TransformerBlock(128, 4, 4, 512, 1e-5)];
      const caches = [new KVCache(), new KVCache()];

      // Initial prompt
      let x = MxArray.randomNormal(shape(1, 8, 128), 0, 0.02);
      for (let i = 0; i < blocks.length; i++) {
        x = blocks[i].forward(x, null, caches[i]);
      }

      caches.forEach((cache) => expect(cache.getOffset()).toBe(8));

      // Incremental generation
      x = MxArray.randomNormal(shape(1, 1, 128), 0, 0.02);
      for (let i = 0; i < blocks.length; i++) {
        x = blocks[i].forward(x, null, caches[i]);
      }

      caches.forEach((cache) => expect(cache.getOffset()).toBe(9));
      assertShape(x, [1, 1, 128]);
    });
  });

  describe('Attention Masks', () => {
    it('should work with causal mask', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const seq_len = 10;
      const x = MxArray.randomNormal(shape(1, seq_len, 256), 0, 0.02);

      // Create causal mask
      const mask_data = new Float32Array(seq_len * seq_len);
      for (let i = 0; i < seq_len; i++) {
        for (let j = 0; j < seq_len; j++) {
          mask_data[i * seq_len + j] = j > i ? -Infinity : 0.0;
        }
      }
      const mask = MxArray.fromFloat32(mask_data, shape(1, 1, seq_len, seq_len));

      const output = block.forward(x, mask, null);

      assertShape(output, [1, seq_len, 256]);
      assertFinite(output);
    });

    it('should work with mask and KV cache', () => {
      const block = new TransformerBlock(256, 4, 4, 1024, 1e-5);
      const cache = new KVCache();

      // First call with causal mask
      const seq_len1 = 8;
      const x1 = MxArray.randomNormal(shape(1, seq_len1, 256), 0, 0.02);
      const mask_data1 = new Float32Array(seq_len1 * seq_len1);
      for (let i = 0; i < seq_len1; i++) {
        for (let j = 0; j < seq_len1; j++) {
          mask_data1[i * seq_len1 + j] = j > i ? -Infinity : 0.0;
        }
      }
      const mask1 = MxArray.fromFloat32(mask_data1, shape(1, 1, seq_len1, seq_len1));

      const output1 = block.forward(x1, mask1, cache);
      assertShape(output1, [1, seq_len1, 256]);
      expect(cache.getOffset()).toBe(8);

      // Second call with single token
      const x2 = MxArray.randomNormal(shape(1, 1, 256), 0, 0.02);
      const mask2 = MxArray.zeros(shape(1, 1, 1, 9));
      const output2 = block.forward(x2, mask2, cache);
      assertShape(output2, [1, 1, 256]);
      expect(cache.getOffset()).toBe(9);
    });
  });

  describe('Sequence Length Variations (MLX-RS compatibility)', () => {
    it('should handle sequence length 63', () => {
      const block = new TransformerBlock(512, 8, 8, 2048, 1e-5);
      const x = MxArray.randomNormal(shape(2, 63, 512), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 63, 512]);
      assertFinite(output);
    });

    it('should handle sequence length 129', () => {
      const block = new TransformerBlock(512, 8, 8, 2048, 1e-5);
      const x = MxArray.randomNormal(shape(2, 129, 512), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 129, 512]);
      assertFinite(output);
    });

    it('should handle sequence length 400', () => {
      const block = new TransformerBlock(512, 8, 8, 2048, 1e-5);
      const x = MxArray.randomNormal(shape(2, 400, 512), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 400, 512]);
      assertFinite(output);
    });
  });

  describe('Large Head Configurations', () => {
    it('should work with 24 heads', () => {
      const block = new TransformerBlock(1536, 24, 24, 6144, 1e-5);
      const x = MxArray.randomNormal(shape(2, 16, 1536), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 16, 1536]);
      assertFinite(output);
    });

    it('should work with 32 heads', () => {
      const block = new TransformerBlock(2048, 32, 32, 8192, 1e-5);
      const x = MxArray.randomNormal(shape(1, 16, 2048), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 16, 2048]);
      assertFinite(output);
    });

    it('should work with 24 query heads and 8 KV heads (3x GQA)', () => {
      const block = new TransformerBlock(1536, 24, 8, 6144, 1e-5);
      const x = MxArray.randomNormal(shape(2, 16, 1536), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [2, 16, 1536]);
      assertFinite(output);
    });
  });

  describe('Numerical Properties', () => {
    it('should produce non-zero outputs', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(1, 10, 128), 0, 0.5);

      const output = block.forward(x, null, null);

      const data = output.toFloat32();
      const hasNonZero = Array.from(data).some((v) => Math.abs(v) > 1e-6);
      expect(hasNonZero).toBe(true);
    });

    it('should handle zero input', () => {
      const block = new TransformerBlock(64, 2, 2, 256, 1e-5);
      const x = MxArray.zeros(shape(1, 8, 64));

      const output = block.forward(x, null, null);

      assertShape(output, [1, 8, 64]);
      assertFinite(output);
    });

    it('should maintain stable outputs across blocks', () => {
      const blocks = [
        new TransformerBlock(256, 4, 4, 1024, 1e-5),
        new TransformerBlock(256, 4, 4, 1024, 1e-5),
        new TransformerBlock(256, 4, 4, 1024, 1e-5),
      ];

      let x = MxArray.randomNormal(shape(1, 10, 256), 0, 0.02);

      for (const block of blocks) {
        x = block.forward(x, null, null);
        assertFinite(x);

        // Check for numerical stability (no explosion/vanishing)
        const data = x.toFloat32();
        const mean = Array.from(data).reduce((a, b) => a + b, 0) / data.length;
        const variance = Array.from(data).reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;

        expect(Math.abs(mean)).toBeLessThan(5.0);
        expect(variance).toBeGreaterThan(0.001);
        expect(variance).toBeLessThan(50.0);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long sequences', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(1, 512, 128), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 512, 128]);
      assertFinite(output);
    });

    it('should handle minimal configuration', () => {
      const block = new TransformerBlock(32, 2, 2, 128, 1e-5);
      const x = MxArray.randomNormal(shape(1, 4, 32), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 4, 32]);
      assertFinite(output);
    });

    it('should handle large batch sizes', () => {
      const block = new TransformerBlock(128, 4, 4, 512, 1e-5);
      const x = MxArray.randomNormal(shape(32, 8, 128), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [32, 8, 128]);
      assertFinite(output);
    });

    it('should handle single batch, single token', () => {
      const block = new TransformerBlock(64, 2, 2, 256, 1e-5);
      const x = MxArray.randomNormal(shape(1, 1, 64), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 1, 64]);
      assertFinite(output);
    });

    it('should handle extreme GQA ratio in block', () => {
      const block = new TransformerBlock(1024, 16, 1, 4096, 1e-5);
      const x = MxArray.randomNormal(shape(1, 8, 1024), 0, 0.02);

      const output = block.forward(x, null, null);

      assertShape(output, [1, 8, 1024]);
      assertFinite(output);
    });
  });
});
