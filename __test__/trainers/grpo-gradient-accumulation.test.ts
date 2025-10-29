import { describe, it, expect } from 'vitest';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import { shape } from '../test-utils';

describe('GRPO Gradient Accumulation', () => {
  describe('Gradient Accumulation Math', () => {
    it('should correctly accumulate gradients using Qwen3Model.accumulateGradients', () => {
      // Test gradient accumulation helper function
      const grad1 = {
        param1: MxArray.fromFloat32(new Float32Array([1.0, 2.0]), shape(2)),
        param2: MxArray.fromFloat32(new Float32Array([3.0]), shape(1)),
      };

      const grad2 = {
        param1: MxArray.fromFloat32(new Float32Array([0.5, 1.5]), shape(2)),
        param2: MxArray.fromFloat32(new Float32Array([1.0]), shape(1)),
      };

      const accumulated = Qwen3Model.accumulateGradients(grad1, grad2);

      // Check param1: [1.0, 2.0] + [0.5, 1.5] = [1.5, 3.5]
      const param1Values = accumulated['param1'].toFloat32();
      expect(param1Values[0]).toBeCloseTo(1.5, 5);
      expect(param1Values[1]).toBeCloseTo(3.5, 5);

      // Check param2: [3.0] + [1.0] = [4.0]
      const param2Values = accumulated['param2'].toFloat32();
      expect(param2Values[0]).toBeCloseTo(4.0, 5);
    });

    it('should handle initial accumulation (null accumulated gradients)', () => {
      const grad1 = {};

      const grad2 = {
        param1: MxArray.fromFloat32(new Float32Array([1.0, 2.0]), shape(2)),
      };

      const accumulated = Qwen3Model.accumulateGradients(grad1, grad2);

      // First accumulation should just copy grad2
      const param1Values = accumulated['param1'].toFloat32();
      expect(param1Values[0]).toBe(1.0);
      expect(param1Values[1]).toBe(2.0);
    });
  });

  describe('Gradient Magnitude with Accumulation Steps', () => {
    it('accumulated gradient magnitude scales linearly with N steps', () => {
      // Simulate N micro-batches with identical gradients
      const N = 4;
      const baseGrad = {
        weight: MxArray.fromFloat32(new Float32Array([1.0, 2.0, 3.0, 4.0]), shape(2, 2)),
      };

      // Accumulate N identical gradients
      let accumulated: Record<string, MxArray> = {};
      for (let i = 0; i < N; i++) {
        accumulated = Qwen3Model.accumulateGradients(accumulated, baseGrad);
      }

      // Sum should be N times the base gradient
      const accValues = accumulated['weight'].toFloat32();
      const baseValues = baseGrad.weight.toFloat32();

      for (let i = 0; i < accValues.length; i++) {
        expect(accValues[i]).toBeCloseTo(N * baseValues[i], 5);
      }
    });

    it('accumulated gradients preserve shape through accumulation', () => {
      const testCases = [
        { sh: shape(10), size: 10 }, // 1D
        { sh: shape(4, 8), size: 32 }, // 2D
        { sh: shape(2, 3, 4), size: 24 }, // 3D
      ];

      for (const { sh, size } of testCases) {
        const data = new Float32Array(size).fill(1.0);

        const grad1 = { param: MxArray.fromFloat32(data, sh) };
        const grad2 = { param: MxArray.fromFloat32(data, sh) };

        const accumulated = Qwen3Model.accumulateGradients(grad1, grad2);
        const accShape = accumulated['param'].shape();

        // Convert BigInt64Array to regular numbers for comparison
        expect(Array.from(accShape).map(Number)).toEqual(Array.from(sh).map(Number));
      }
    });
  });

  describe('Learning Rate Scaling Equivalence', () => {
    it('lr/N with summed grads equals lr with averaged grads', () => {
      // Mathematical proof:
      // param = param - (lr/N) * sum(grads)
      // is equivalent to:
      // param = param - lr * mean(grads)

      const N = 4;
      const lr = 0.1;

      // Create some gradients
      const grads = [
        new Float32Array([1.0, 2.0]),
        new Float32Array([3.0, 4.0]),
        new Float32Array([5.0, 6.0]),
        new Float32Array([7.0, 8.0]),
      ];

      // Method 1: Sum gradients, scale lr by N
      const sum = new Float32Array(2);
      for (const g of grads) {
        sum[0] += g[0];
        sum[1] += g[1];
      }
      const update1 = [(lr / N) * sum[0], (lr / N) * sum[1]];

      // Method 2: Average gradients, use full lr
      const avg = [sum[0] / N, sum[1] / N];
      const update2 = [lr * avg[0], lr * avg[1]];

      // Both should be equal
      expect(update1[0]).toBeCloseTo(update2[0], 10);
      expect(update1[1]).toBeCloseTo(update2[1], 10);

      // Verify the actual values
      // sum = [16, 20], avg = [4, 5]
      // update1 = (0.1/4) * [16, 20] = [0.4, 0.5]
      // update2 = 0.1 * [4, 5] = [0.4, 0.5]
      expect(update1[0]).toBeCloseTo(0.4, 10);
      expect(update1[1]).toBeCloseTo(0.5, 10);
    });

    it('MxArray operations verify lr scaling equivalence', () => {
      const N = 4;
      const lr = 0.01;

      // Create initial parameter
      const param = MxArray.fromFloat32(new Float32Array([10.0, 20.0]), shape(2));

      // Create gradients to accumulate
      const grads = [
        MxArray.fromFloat32(new Float32Array([1.0, 2.0]), shape(2)),
        MxArray.fromFloat32(new Float32Array([2.0, 3.0]), shape(2)),
        MxArray.fromFloat32(new Float32Array([3.0, 4.0]), shape(2)),
        MxArray.fromFloat32(new Float32Array([4.0, 5.0]), shape(2)),
      ];

      // Method 1: Accumulate and apply with scaled LR (what grpo-trainer does)
      let accumulated: Record<string, MxArray> = {};
      for (const g of grads) {
        accumulated = Qwen3Model.accumulateGradients(accumulated, { w: g });
      }
      const scaledLr = MxArray.fromFloat32(new Float32Array([lr / N]), shape(1));
      const scaledGrad1 = scaledLr.mul(accumulated['w']);
      const updated1 = param.sub(scaledGrad1);

      // Method 2: Average first, then apply with full LR
      const sumGrad = accumulated['w'];
      const avgGrad = sumGrad.divScalar(N);
      const fullLr = MxArray.fromFloat32(new Float32Array([lr]), shape(1));
      const scaledGrad2 = fullLr.mul(avgGrad);
      const updated2 = param.sub(scaledGrad2);

      // Both methods should produce identical results
      const vals1 = updated1.toFloat32();
      const vals2 = updated2.toFloat32();

      expect(vals1[0]).toBeCloseTo(vals2[0], 5);
      expect(vals1[1]).toBeCloseTo(vals2[1], 5);
    });
  });
});
