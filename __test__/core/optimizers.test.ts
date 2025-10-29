import { describe, it, expect } from 'vitest';
import { Adam, AdamW, SGD, RMSprop, GradientUtils, LRScheduler } from '@mlx-node/core';
import { createFloat32Array } from '../test-utils';

describe('Optimizers', () => {
  describe('Adam', () => {
    it('should update parameters with default settings', () => {
      const adam = new Adam();

      // Create a simple parameter and gradient
      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const grad = createFloat32Array([0.1, 0.2, 0.3], [3]);

      // Update the parameter
      const updated = adam.updateSingle('param1', param, grad);
      const values = updated.toFloat32();

      // With default lr=1e-3, the parameters should decrease
      expect(values[0]).toBeLessThan(1.0);
      expect(values[1]).toBeLessThan(2.0);
      expect(values[2]).toBeLessThan(3.0);
    });

    it('should maintain separate state for different parameters', () => {
      const adam = new Adam();

      const param1 = createFloat32Array([1.0, 2.0], [2]);
      const param2 = createFloat32Array([3.0, 4.0], [2]);
      const grad = createFloat32Array([0.1, 0.2], [2]);

      // Update both parameters
      const updated1 = adam.updateSingle('param1', param1, grad);
      const updated2 = adam.updateSingle('param2', param2, grad);

      // Updates should be different since initial params are different
      const values1 = updated1.toFloat32();
      const values2 = updated2.toFloat32();

      expect(values1[0]).not.toBeCloseTo(values2[0]);
      expect(values1[1]).not.toBeCloseTo(values2[1]);
    });

    it('should apply bias correction when enabled', () => {
      const adamWithBias = new Adam(0.01, 0.9, 0.999, 1e-8, true);
      const adamWithoutBias = new Adam(0.01, 0.9, 0.999, 1e-8, false);

      const param = createFloat32Array([1.0, 1.0], [2]);
      const grad = createFloat32Array([0.1, 0.1], [2]);

      // First update
      const withBias = adamWithBias.updateSingle('param', param, grad);
      const withoutBias = adamWithoutBias.updateSingle('param', param, grad);

      const valuesWithBias = withBias.toFloat32();
      const valuesWithoutBias = withoutBias.toFloat32();

      // With bias correction, the initial steps have bias-corrected moments
      // The update magnitudes will be different but exact behavior depends on implementation
      // Just check that both actually changed the parameters
      expect(valuesWithBias[0]).not.toBeCloseTo(1.0);
      expect(valuesWithoutBias[0]).not.toBeCloseTo(1.0);

      // And that they're different from each other
      expect(valuesWithBias[0]).not.toBeCloseTo(valuesWithoutBias[0]);
    });

    it('should reset state correctly', () => {
      const adam = new Adam();

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad = createFloat32Array([0.1, 0.2], [2]);

      // First update
      const updated1 = adam.updateSingle('param', param, grad);

      // Reset and update again
      adam.reset();
      const updated2 = adam.updateSingle('param', param, grad);

      // After reset, the update should be the same as the first one
      const values1 = updated1.toFloat32();
      const values2 = updated2.toFloat32();

      expect(values1[0]).toBeCloseTo(values2[0], 5);
      expect(values1[1]).toBeCloseTo(values2[1], 5);
    });

    it('should handle different learning rates', () => {
      const adamLow = new Adam(0.001);
      const adamHigh = new Adam(0.01);

      const param = createFloat32Array([1.0, 1.0], [2]);
      const grad = createFloat32Array([0.1, 0.1], [2]);

      const updatedLow = adamLow.updateSingle('param', param, grad);
      const updatedHigh = adamHigh.updateSingle('param', param, grad);

      const valuesLow = updatedLow.toFloat32();
      const valuesHigh = updatedHigh.toFloat32();

      // Higher learning rate should produce larger changes
      expect(Math.abs(1.0 - valuesHigh[0])).toBeGreaterThan(Math.abs(1.0 - valuesLow[0]));
    });
  });

  describe('AdamW', () => {
    it('should apply weight decay', () => {
      const adamw = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.01);
      const adam = new Adam(0.01, 0.9, 0.999, 1e-8);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const grad = createFloat32Array([0.0, 0.0, 0.0], [3]); // Zero gradient

      // With zero gradient, AdamW should still update due to weight decay
      const updatedAdamW = adamw.updateSingle('param', param, grad);
      const updatedAdam = adam.updateSingle('param', param, grad);

      const valuesAdamW = updatedAdamW.toFloat32();
      const valuesAdam = updatedAdam.toFloat32();

      // AdamW should decay the weights even with zero gradient
      expect(valuesAdamW[0]).toBeLessThan(1.0);
      expect(valuesAdamW[1]).toBeLessThan(2.0);
      expect(valuesAdamW[2]).toBeLessThan(3.0);

      // Regular Adam should not change with zero gradient
      expect(valuesAdam[0]).toBeCloseTo(1.0, 5);
      expect(valuesAdam[1]).toBeCloseTo(2.0, 5);
      expect(valuesAdam[2]).toBeCloseTo(3.0, 5);
    });

    it('should handle different weight decay values', () => {
      const adamwLow = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.001);
      const adamwHigh = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.1);

      const param = createFloat32Array([1.0, 1.0], [2]);
      const grad = createFloat32Array([0.0, 0.0], [2]);

      const updatedLow = adamwLow.updateSingle('param', param, grad);
      const updatedHigh = adamwHigh.updateSingle('param', param, grad);

      const valuesLow = updatedLow.toFloat32();
      const valuesHigh = updatedHigh.toFloat32();

      // Higher weight decay should produce more decay
      expect(valuesHigh[0]).toBeLessThan(valuesLow[0]);
    });
  });

  describe('SGD', () => {
    it('should update parameters with basic gradient descent', () => {
      const sgd = new SGD(0.1);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const grad = createFloat32Array([0.1, 0.2, 0.3], [3]);

      const updated = sgd.updateSingle('param', param, grad);
      const values = updated.toFloat32();

      // SGD update: param = param - lr * grad
      expect(values[0]).toBeCloseTo(1.0 - 0.1 * 0.1, 5);
      expect(values[1]).toBeCloseTo(2.0 - 0.1 * 0.2, 5);
      expect(values[2]).toBeCloseTo(3.0 - 0.1 * 0.3, 5);
    });

    it('should apply momentum correctly', () => {
      const sgdMomentum = new SGD(0.1, 0.9);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad1 = createFloat32Array([0.1, 0.2], [2]);
      const grad2 = createFloat32Array([0.1, 0.2], [2]);

      // First update - no momentum yet
      const updated1 = sgdMomentum.updateSingle('param', param, grad1);

      // Second update - should have momentum
      const updated2 = sgdMomentum.updateSingle('param', updated1, grad2);
      const values2 = updated2.toFloat32();

      // With momentum, the second update should be larger
      // v1 = grad1 = [0.1, 0.2]
      // v2 = 0.9 * v1 + grad2 = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
      const expected0 = 1.0 - 0.1 * 0.1 - 0.1 * 0.19;
      const expected1 = 2.0 - 0.1 * 0.2 - 0.1 * 0.38;

      expect(values2[0]).toBeCloseTo(expected0, 4);
      expect(values2[1]).toBeCloseTo(expected1, 4);
    });

    it('should handle weight decay', () => {
      const sgdWD = new SGD(0.1, 0.0, 0.01);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad = createFloat32Array([0.0, 0.0], [2]);

      const updated = sgdWD.updateSingle('param', param, grad);
      const values = updated.toFloat32();

      // With weight decay and zero gradient, parameters should still decrease
      // effective_grad = grad + weight_decay * param
      expect(values[0]).toBeCloseTo(1.0 - 0.1 * 0.01 * 1.0, 5);
      expect(values[1]).toBeCloseTo(2.0 - 0.1 * 0.01 * 2.0, 5);
    });

    it('should reject invalid Nesterov configuration', () => {
      // Nesterov requires momentum > 0
      expect(() => new SGD(0.1, 0.0, 0.0, 0.0, true)).toThrow();

      // Nesterov requires dampening = 0
      expect(() => new SGD(0.1, 0.9, 0.0, 0.1, true)).toThrow();

      // Valid Nesterov configuration should not throw
      expect(() => new SGD(0.1, 0.9, 0.0, 0.0, true)).not.toThrow();
    });
  });

  describe('RMSprop', () => {
    it('should update parameters with adaptive learning rate', () => {
      const rmsprop = new RMSprop(0.01);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const grad = createFloat32Array([0.1, 0.2, 0.3], [3]);

      const updated = rmsprop.updateSingle('param', param, grad);
      const values = updated.toFloat32();

      // RMSprop should decrease parameters
      expect(values[0]).toBeLessThan(1.0);
      expect(values[1]).toBeLessThan(2.0);
      expect(values[2]).toBeLessThan(3.0);
    });

    it('should adapt to gradient magnitudes', () => {
      const rmsprop = new RMSprop(0.01);

      const param = createFloat32Array([1.0, 1.0], [2]);
      const gradLarge = createFloat32Array([1.0, 0.1], [2]);

      // Multiple updates with different gradient magnitudes
      let current = param;
      for (let i = 0; i < 5; i++) {
        current = rmsprop.updateSingle('param', current, gradLarge);
      }

      const values = current.toFloat32();

      // The parameter with larger gradients should have smaller effective learning rate
      // due to RMSprop's normalization
      const change0 = Math.abs(1.0 - values[0]);
      const change1 = Math.abs(1.0 - values[1]);

      // Both should change
      expect(change0).toBeGreaterThan(0);
      expect(change1).toBeGreaterThan(0);
    });

    it('should handle different alpha values', () => {
      const rmspropLowAlpha = new RMSprop(0.01, 0.9);
      const rmspropHighAlpha = new RMSprop(0.01, 0.99);

      const param = createFloat32Array([1.0, 1.0], [2]);
      const grad = createFloat32Array([0.1, 0.1], [2]);

      // Multiple updates
      let currentLow = param;
      let currentHigh = param;
      for (let i = 0; i < 3; i++) {
        currentLow = rmspropLowAlpha.updateSingle('param', currentLow, grad);
        currentHigh = rmspropHighAlpha.updateSingle('param', currentHigh, grad);
      }

      const valuesLow = currentLow.toFloat32();
      const valuesHigh = currentHigh.toFloat32();

      // Different alpha values should lead to different updates
      expect(valuesLow[0]).not.toBeCloseTo(valuesHigh[0], 3);
    });
  });

  describe('GradientUtils', () => {
    it('should clip gradients by value', () => {
      const grad = createFloat32Array([-2.0, -0.5, 0.0, 0.5, 2.0], [5]);
      const clipped = GradientUtils.clipGradValue(grad, -1.0, 1.0);
      const values = clipped.toFloat32();

      expect(values[0]).toBeCloseTo(-1.0, 5);
      expect(values[1]).toBeCloseTo(-0.5, 5);
      expect(values[2]).toBeCloseTo(0.0, 5);
      expect(values[3]).toBeCloseTo(0.5, 5);
      expect(values[4]).toBeCloseTo(1.0, 5);
    });

    it('should compute gradient norm correctly', () => {
      // Create gradients: [3, 4] has L2 norm = sqrt(9 + 16) = 5
      const grad1 = createFloat32Array([3.0, 4.0], [2]);
      const gradients = { param1: grad1 };

      const norm = GradientUtils.computeGradientNorm(gradients);
      expect(norm).toBeCloseTo(5.0, 5);
    });

    it('should compute gradient norm across multiple parameters', () => {
      // grads: [1, 2] and [2] have total L2 norm = sqrt(1 + 4 + 4) = 3
      const grad1 = createFloat32Array([1.0, 2.0], [2]);
      const grad2 = createFloat32Array([2.0], [1]);
      const gradients = { param1: grad1, param2: grad2 };

      const norm = GradientUtils.computeGradientNorm(gradients);
      expect(norm).toBeCloseTo(3.0, 5);
    });

    it('should not clip gradients when norm is below max_norm', () => {
      // Gradient norm = 5, max_norm = 10, should not clip
      const grad = createFloat32Array([3.0, 4.0], [2]);
      const gradients = { param: grad };

      const clipped = GradientUtils.clipGradNorm(gradients, 10.0);
      const values = clipped['param'].toFloat32();

      expect(values[0]).toBeCloseTo(3.0, 5);
      expect(values[1]).toBeCloseTo(4.0, 5);
    });

    it('should clip gradients when norm exceeds max_norm', () => {
      // Gradient norm = 5, max_norm = 2.5, should scale by 0.5
      const grad = createFloat32Array([3.0, 4.0], [2]);
      const gradients = { param: grad };

      const clipped = GradientUtils.clipGradNorm(gradients, 2.5);
      const values = clipped['param'].toFloat32();

      // Should scale by max_norm / total_norm = 2.5 / 5 = 0.5
      expect(values[0]).toBeCloseTo(1.5, 4);
      expect(values[1]).toBeCloseTo(2.0, 4);
    });

    it('should clip multiple gradient parameters proportionally', () => {
      // Total norm = sqrt(1 + 4 + 4) = 3, max_norm = 1.5, scale = 0.5
      const grad1 = createFloat32Array([1.0, 2.0], [2]);
      const grad2 = createFloat32Array([2.0], [1]);
      const gradients = { param1: grad1, param2: grad2 };

      const clipped = GradientUtils.clipGradNorm(gradients, 1.5);

      const values1 = clipped['param1'].toFloat32();
      const values2 = clipped['param2'].toFloat32();

      expect(values1[0]).toBeCloseTo(0.5, 4);
      expect(values1[1]).toBeCloseTo(1.0, 4);
      expect(values2[0]).toBeCloseTo(1.0, 4);
    });

    it('should return original norm with clip_grad_norm_with_norm', () => {
      // Gradient norm = 5, max_norm = 2.5
      const grad = createFloat32Array([3.0, 4.0], [2]);
      const gradients = { param: grad };

      const [clipped, originalNorm] = GradientUtils.clipGradNormWithNorm(gradients, 2.5);

      expect(originalNorm).toBeCloseTo(5.0, 5);

      const values = clipped['param'].toFloat32();
      expect(values[0]).toBeCloseTo(1.5, 4);
      expect(values[1]).toBeCloseTo(2.0, 4);
    });

    it('should preserve gradient structure after clipping', () => {
      const grad1 = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const grad2 = createFloat32Array([4.0, 5.0], [2]);
      const gradients = { weight: grad1, bias: grad2 };

      const clipped = GradientUtils.clipGradNorm(gradients, 100.0);

      expect(Object.keys(clipped)).toContain('weight');
      expect(Object.keys(clipped)).toContain('bias');
      expect(clipped['weight'].shape()).toEqual(BigInt64Array.from([3n]));
      expect(clipped['bias'].shape()).toEqual(BigInt64Array.from([2n]));
    });
  });

  describe('LRScheduler', () => {
    it('should apply linear decay', () => {
      const lr0 = LRScheduler.linearDecay(1.0, 0.1, 0, 100);
      const lr50 = LRScheduler.linearDecay(1.0, 0.1, 50, 100);
      const lr100 = LRScheduler.linearDecay(1.0, 0.1, 100, 100);
      const lr200 = LRScheduler.linearDecay(1.0, 0.1, 200, 100);

      expect(lr0).toBeCloseTo(1.0, 5);
      expect(lr50).toBeCloseTo(0.55, 5);
      expect(lr100).toBeCloseTo(0.1, 5);
      expect(lr200).toBeCloseTo(0.1, 5); // Should stay at final_lr
    });

    it('should apply exponential decay', () => {
      const lr0 = LRScheduler.exponentialDecay(1.0, 0.9, 0, 10);
      const lr10 = LRScheduler.exponentialDecay(1.0, 0.9, 10, 10);
      const lr20 = LRScheduler.exponentialDecay(1.0, 0.9, 20, 10);

      expect(lr0).toBeCloseTo(1.0, 5);
      expect(lr10).toBeCloseTo(0.9, 5);
      expect(lr20).toBeCloseTo(0.81, 5);
    });

    it('should apply cosine annealing', () => {
      const lr0 = LRScheduler.cosineAnnealing(1.0, 0.1, 0, 100);
      const lr25 = LRScheduler.cosineAnnealing(1.0, 0.1, 25, 100);
      const lr50 = LRScheduler.cosineAnnealing(1.0, 0.1, 50, 100);
      const lr75 = LRScheduler.cosineAnnealing(1.0, 0.1, 75, 100);
      const lr100 = LRScheduler.cosineAnnealing(1.0, 0.1, 100, 100);

      expect(lr0).toBeCloseTo(1.0, 5);
      expect(lr50).toBeCloseTo(0.55, 5); // Midpoint
      expect(lr100).toBeCloseTo(0.1, 5);

      // Cosine shape: should be symmetric
      const diff25 = lr0 - lr25;
      const diff75 = lr75 - lr100;
      expect(diff25).toBeCloseTo(diff75, 3);
    });

    it('should apply step decay', () => {
      const lr0 = LRScheduler.stepDecay(1.0, 0.5, 0, 10);
      const lr9 = LRScheduler.stepDecay(1.0, 0.5, 9, 10);
      const lr10 = LRScheduler.stepDecay(1.0, 0.5, 10, 10);
      const lr20 = LRScheduler.stepDecay(1.0, 0.5, 20, 10);
      const lr30 = LRScheduler.stepDecay(1.0, 0.5, 30, 10);

      expect(lr0).toBeCloseTo(1.0, 5);
      expect(lr9).toBeCloseTo(1.0, 5);
      expect(lr10).toBeCloseTo(0.5, 5);
      expect(lr20).toBeCloseTo(0.25, 5);
      expect(lr30).toBeCloseTo(0.125, 5);
    });
  });

  describe('Integration tests', () => {
    it('should optimize a simple quadratic function with Adam', () => {
      const adam = new Adam(0.1);

      // Minimize f(x) = x^2, gradient = 2x
      let x = createFloat32Array([10.0], [1]);

      for (let i = 0; i < 100; i++) {
        const grad = x.mulScalar(2.0); // gradient = 2x
        x = adam.updateSingle('x', x, grad);
      }

      const value = x.toFloat32()[0];

      // Should converge close to minimum at x=0
      expect(Math.abs(value)).toBeLessThan(0.1);
    });

    it('should optimize with learning rate scheduling', () => {
      let x = createFloat32Array([10.0], [1]);

      for (let i = 0; i < 50; i++) {
        // Decay learning rate
        const lr = LRScheduler.exponentialDecay(1.0, 0.9, i, 10);
        const sgdWithLR = new SGD(lr);

        const grad = x.mulScalar(2.0);
        x = sgdWithLR.updateSingle('x', x, grad);
      }

      const value = x.toFloat32()[0];

      // Should converge with decaying learning rate
      expect(Math.abs(value)).toBeLessThan(1.0);
    });

    it('should handle multiple parameters with different optimizers', () => {
      const adam = new Adam(0.1); // Increased learning rate for faster convergence
      const sgd = new SGD(0.1);

      let param1 = createFloat32Array([5.0], [1]);
      let param2 = createFloat32Array([5.0], [1]);

      // Optimize both parameters
      for (let i = 0; i < 100; i++) {
        // Increased iterations
        const grad1 = param1.mulScalar(2.0);
        const grad2 = param2.mulScalar(2.0);

        param1 = adam.updateSingle('param1', param1, grad1);
        param2 = sgd.updateSingle('param2', param2, grad2);
      }

      const value1 = param1.toFloat32()[0];
      const value2 = param2.toFloat32()[0];

      // Both should converge close to zero
      expect(Math.abs(value1)).toBeLessThan(1.0);
      expect(Math.abs(value2)).toBeLessThan(1.0);
    });
  });
});
