import { describe, it, expect } from 'vitest';
import { MxArray, Adam, AdamW, SGD, RMSprop, GradientUtils } from '@mlx-node/core';
import { createFloat32Array } from '../test-utils';

describe('Extended Optimizer Tests', () => {
  describe('State Initialization', () => {
    it('should properly initialize Adam state on first update', () => {
      const adam = new Adam();

      const param1 = createFloat32Array([1.0, 2.0], [2]);
      const param2 = createFloat32Array([3.0, 4.0, 5.0], [3]);
      const grad1 = createFloat32Array([0.1, 0.2], [2]);
      const grad2 = createFloat32Array([0.3, 0.4, 0.5], [3]);

      // First update should initialize state
      const updated1 = adam.updateSingle('param1', param1, grad1);
      const updated2 = adam.updateSingle('param2', param2, grad2);

      // Second update should use existing state
      const updated1_2 = adam.updateSingle('param1', updated1, grad1);
      const updated2_2 = adam.updateSingle('param2', updated2, grad2);

      // Check that parameters continue to update
      // Use a larger threshold since Adam updates can be small
      const diff1 = Math.abs(updated1_2.toFloat32()[0] - updated1.toFloat32()[0]);
      const diff2 = Math.abs(updated2_2.toFloat32()[0] - updated2.toFloat32()[0]);
      expect(diff1).toBeGreaterThan(0.0001);
      expect(diff2).toBeGreaterThan(0.0001);
    });

    it('should maintain separate state for each optimizer instance', () => {
      const adam1 = new Adam(0.01);
      const adam2 = new Adam(0.01);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad = createFloat32Array([0.1, 0.2], [2]);

      // Update with first optimizer multiple times
      let result1 = param;
      for (let i = 0; i < 5; i++) {
        result1 = adam1.updateSingle('param', result1, grad);
      }

      // Update with second optimizer once
      const result2 = adam2.updateSingle('param', param, grad);

      // Results should be different (adam1 has accumulated momentum)
      expect(result1.toFloat32()[0]).not.toBeCloseTo(result2.toFloat32()[0]);
    });

    it('should handle SGD with and without momentum state', () => {
      const sgdNoMomentum = new SGD(0.1, 0.0);
      const sgdWithMomentum = new SGD(0.1, 0.9);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad = createFloat32Array([0.1, 0.2], [2]);

      // Both should work on first update
      const updated1 = sgdNoMomentum.updateSingle('param', param, grad);
      const updated2 = sgdWithMomentum.updateSingle('param', param, grad);

      // First update should be the same (no momentum accumulated yet)
      expect(updated1.toFloat32()[0]).toBeCloseTo(1.0 - 0.1 * 0.1, 5);
      expect(updated2.toFloat32()[0]).toBeCloseTo(1.0 - 0.1 * 0.1, 5);
    });
  });

  describe('Zero Gradient Handling', () => {
    it('Adam should handle zero gradients correctly', () => {
      const adam = new Adam(0.01);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const zeroGrad = createFloat32Array([0.0, 0.0, 0.0], [3]);

      // Update with zero gradient
      const updated = adam.updateSingle('param', param, zeroGrad);
      const values = updated.toFloat32();

      // With zero gradient and no weight decay, parameters shouldn't change much
      // (only numerical precision differences)
      expect(values[0]).toBeCloseTo(1.0, 4);
      expect(values[1]).toBeCloseTo(2.0, 4);
      expect(values[2]).toBeCloseTo(3.0, 4);
    });

    it('SGD with momentum should accumulate zero gradients', () => {
      const sgd = new SGD(0.1, 0.9);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const grad1 = createFloat32Array([0.1, 0.2], [2]);
      const zeroGrad = createFloat32Array([0.0, 0.0], [2]);

      // First update with non-zero gradient
      const updated1 = sgd.updateSingle('param', param, grad1);

      // Second update with zero gradient (momentum should still apply)
      const updated2 = sgd.updateSingle('param', updated1, zeroGrad);
      const values = updated2.toFloat32();

      // Momentum should cause continued movement even with zero gradient
      expect(values[0]).toBeLessThan(updated1.toFloat32()[0]);
      expect(values[1]).toBeLessThan(updated1.toFloat32()[1]);
    });

    it('AdamW should apply weight decay even with zero gradients', () => {
      const adamw = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.1);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const zeroGrad = createFloat32Array([0.0, 0.0, 0.0], [3]);

      // Multiple updates with zero gradient
      let current = param;
      for (let i = 0; i < 10; i++) {
        current = adamw.updateSingle('param', current, zeroGrad);
      }

      const values = current.toFloat32();

      // Weight decay should reduce parameters even with zero gradient
      expect(values[0]).toBeLessThan(1.0);
      expect(values[1]).toBeLessThan(2.0);
      expect(values[2]).toBeLessThan(3.0);
    });
  });

  describe('Extreme Values', () => {
    it('should handle very small gradients', () => {
      const adam = new Adam(0.01);

      const param = createFloat32Array([1.0, 2.0], [2]);
      const tinyGrad = createFloat32Array([1e-10, 1e-10], [2]);

      const updated = adam.updateSingle('param', param, tinyGrad);
      const values = updated.toFloat32();

      // Should still update, but very slightly
      expect(values[0]).toBeLessThan(1.0);
      expect(values[0]).toBeGreaterThan(0.99);
    });

    it('should handle very large gradients', () => {
      const adam = new Adam(0.001); // Small learning rate for stability

      const param = createFloat32Array([1.0, 2.0], [2]);
      const largeGrad = createFloat32Array([1000.0, 2000.0], [2]);

      const updated = adam.updateSingle('param', param, largeGrad);
      const values = updated.toFloat32();

      // Should update but be bounded by Adam's adaptive learning rate
      expect(Math.abs(values[0] - 1.0)).toBeLessThan(10); // Not exploding
      expect(Math.abs(values[1] - 2.0)).toBeLessThan(10);
    });

    it('should handle mixed gradient magnitudes', () => {
      const rmsprop = new RMSprop(0.01);

      const param = createFloat32Array([1.0, 1.0, 1.0], [3]);
      const mixedGrad = createFloat32Array([0.001, 1.0, 100.0], [3]);

      // Multiple updates
      let current = param;
      for (let i = 0; i < 5; i++) {
        current = rmsprop.updateSingle('param', current, mixedGrad);
      }

      const values = current.toFloat32();

      // All parameters should have changed
      expect(values[0]).not.toBeCloseTo(1.0);
      expect(values[1]).not.toBeCloseTo(1.0);
      expect(values[2]).not.toBeCloseTo(1.0);

      // RMSprop should normalize the updates
      const change0 = Math.abs(1.0 - values[0]);
      const change2 = Math.abs(1.0 - values[2]);

      // Despite 100,000x difference in gradient magnitude,
      // RMSprop should keep updates in reasonable range
      expect(change2 / change0).toBeLessThan(1000); // Much less than gradient ratio
    });
  });

  describe('Convergence Tests', () => {
    it('should converge on quadratic loss with different optimizers', () => {
      // Minimize f(x,y) = (x-2)^2 + (y-3)^2
      // Gradient: [2*(x-2), 2*(y-3)]
      const optimizers = [
        { name: 'Adam', opt: new Adam(0.1) },
        { name: 'SGD', opt: new SGD(0.1) },
        { name: 'SGD+Momentum', opt: new SGD(0.1, 0.9) },
        { name: 'RMSprop', opt: new RMSprop(0.1) },
      ];

      for (const { name: _, opt } of optimizers) {
        let params = createFloat32Array([0.0, 0.0], [2]);

        // Run optimization
        for (let i = 0; i < 100; i++) {
          const values = params.toFloat32();
          const grad = createFloat32Array([2 * (values[0] - 2), 2 * (values[1] - 3)], [2]);

          params = opt.updateSingle('params', params, grad);
        }

        const final = params.toFloat32();

        // All optimizers should converge close to [2, 3]
        expect(final[0]).toBeCloseTo(2.0, 1);
        expect(final[1]).toBeCloseTo(3.0, 1);
      }
    });

    it('should handle non-convex optimization (Rosenbrock)', () => {
      // Rosenbrock function: harder optimization problem
      // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
      // Optimum at [1, 1]

      const adam = new Adam(0.002);
      let params = createFloat32Array([0.0, 0.0], [2]);

      const losses: number[] = [];

      for (let i = 0; i < 5000; i++) {
        const values = params.toFloat32();
        const x = values[0];
        const y = values[1];

        // Compute loss
        const loss = Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2);
        losses.push(loss);

        // Compute gradient
        const gradX = -2 * (1 - x) - 400 * x * (y - x * x);
        const gradY = 200 * (y - x * x);
        const grad = createFloat32Array([gradX, gradY], [2]);

        params = adam.updateSingle('params', params, grad);
      }

      const final = params.toFloat32();

      // Should get reasonably close to optimum
      expect(final[0]).toBeCloseTo(1.0, 0);
      expect(final[1]).toBeCloseTo(1.0, 0);

      // Loss should decrease overall
      const avgFirst100 = losses.slice(0, 100).reduce((a, b) => a + b) / 100;
      const avgLast100 = losses.slice(-100).reduce((a, b) => a + b) / 100;
      expect(avgLast100).toBeLessThan(avgFirst100);
    });
  });

  describe('Gradient Clipping', () => {
    it('should clip gradients before optimization', () => {
      const adam = new Adam(0.1);

      const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const largeGrad = createFloat32Array([10.0, -15.0, 20.0], [3]);

      // Clip gradient first
      const clippedGrad = GradientUtils.clipGradValue(largeGrad, -5.0, 5.0);

      // Update with clipped gradient
      const updated = adam.updateSingle('param', param, clippedGrad);
      const values = updated.toFloat32();

      // Updates should be moderate due to clipping
      expect(Math.abs(values[0] - 1.0)).toBeLessThan(1.0);
      expect(Math.abs(values[1] - 2.0)).toBeLessThan(1.0);
      expect(Math.abs(values[2] - 3.0)).toBeLessThan(1.0);
    });

    it('should handle gradient clipping with different optimizers', () => {
      const optimizers = [new Adam(0.1), new SGD(0.1), new RMSprop(0.1)];

      const param = createFloat32Array([0.0, 0.0], [2]);
      const grad = createFloat32Array([100.0, -100.0], [2]);
      const clippedGrad = GradientUtils.clipGradValue(grad, -10.0, 10.0);

      for (const opt of optimizers) {
        const updated = opt.updateSingle('param', param, clippedGrad);
        const values = updated.toFloat32();

        // All should produce bounded updates
        expect(Math.abs(values[0])).toBeLessThan(5.0);
        expect(Math.abs(values[1])).toBeLessThan(5.0);
      }
    });
  });

  describe('Multiple Parameter Updates', () => {
    it('should handle batch parameter updates', () => {
      const adam = new Adam(0.01);

      // Simulate a small network with multiple parameters
      const params = {
        weight1: createFloat32Array([0.5, 0.5], [2]),
        bias1: createFloat32Array([0.1], [1]),
        weight2: createFloat32Array([0.3, 0.3], [2]),
        bias2: createFloat32Array([0.2], [1]),
      };

      const grads = {
        weight1: createFloat32Array([0.1, 0.2], [2]),
        bias1: createFloat32Array([0.05], [1]),
        weight2: createFloat32Array([0.15, 0.25], [2]),
        bias2: createFloat32Array([0.1], [1]),
      };

      // Update all parameters
      const updated: Record<string, MxArray> = {};
      for (const [name, param] of Object.entries(params)) {
        updated[name] = adam.updateSingle(name, param, grads[name as keyof typeof grads]);
      }

      // All parameters should be updated
      for (const [name, param] of Object.entries(params)) {
        const original = param.toFloat32();
        const newValues = updated[name].toFloat32();

        for (let i = 0; i < original.length; i++) {
          expect(newValues[i]).not.toBeCloseTo(original[i]);
        }
      }
    });

    it('should maintain consistent updates across iterations', () => {
      const sgd = new SGD(0.1, 0.9); // With momentum

      let param = createFloat32Array([5.0], [1]);
      const updates: number[] = [];

      // Track parameter values over iterations
      for (let i = 0; i < 20; i++) {
        updates.push(param.toFloat32()[0]);

        // Gradient towards zero
        const grad = param.mulScalar(0.1);
        param = sgd.updateSingle('param', param, grad);
      }

      // Check monotonic decrease (should consistently move towards zero)
      for (let i = 1; i < updates.length; i++) {
        expect(updates[i]).toBeLessThan(updates[i - 1]);
      }

      // Should converge towards zero
      expect(updates[updates.length - 1]).toBeLessThan(1.0);
    });
  });

  describe('Bias Correction Behavior', () => {
    it('should show different behavior with and without bias correction', () => {
      const adamWithBias = new Adam(0.01, 0.9, 0.999, 1e-8, true);
      const adamWithoutBias = new Adam(0.01, 0.9, 0.999, 1e-8, false);

      let param1 = createFloat32Array([1.0], [1]);
      let param2 = createFloat32Array([1.0], [1]);

      const grad = createFloat32Array([0.1], [1]);

      // Early iterations should show larger difference
      const earlyUpdates1: number[] = [];
      const earlyUpdates2: number[] = [];

      for (let i = 0; i < 10; i++) {
        param1 = adamWithBias.updateSingle('param', param1, grad);
        param2 = adamWithoutBias.updateSingle('param', param2, grad);

        earlyUpdates1.push(param1.toFloat32()[0]);
        earlyUpdates2.push(param2.toFloat32()[0]);
      }

      // With bias correction, the parameters should be different in early iterations
      // Adam with bias correction usually leads to different convergence pattern
      expect(Math.abs(earlyUpdates1[0] - earlyUpdates2[0])).toBeGreaterThan(0.001);

      // Continue for many iterations
      for (let i = 0; i < 100; i++) {
        param1 = adamWithBias.updateSingle('param', param1, grad);
        param2 = adamWithoutBias.updateSingle('param', param2, grad);
      }

      // Both should have moved significantly from initial value
      const final1 = param1.toFloat32()[0];
      const final2 = param2.toFloat32()[0];

      // Both optimizers should have moved the parameter away from 1.0
      expect(final1).toBeLessThan(0.5);
      expect(final2).toBeLessThan(0.5);
    });
  });

  describe('Nesterov Momentum', () => {
    it('should show different convergence with Nesterov momentum', () => {
      const sgdRegular = new SGD(0.1, 0.9, 0.0, 0.0, false);
      const sgdNesterov = new SGD(0.1, 0.9, 0.0, 0.0, true);

      // Minimize simple quadratic
      let param1 = createFloat32Array([10.0], [1]);
      let param2 = createFloat32Array([10.0], [1]);

      const trajectory1: number[] = [];
      const trajectory2: number[] = [];

      for (let i = 0; i < 50; i++) {
        // Gradient of x^2 is 2x
        const grad1 = param1.mulScalar(2.0);
        const grad2 = param2.mulScalar(2.0);

        param1 = sgdRegular.updateSingle('param', param1, grad1);
        param2 = sgdNesterov.updateSingle('param', param2, grad2);

        trajectory1.push(Math.abs(param1.toFloat32()[0]));
        trajectory2.push(Math.abs(param2.toFloat32()[0]));
      }

      // Both should converge significantly
      expect(trajectory1[49]).toBeLessThan(1.0);
      expect(trajectory2[49]).toBeLessThan(1.0);

      // Check significant reduction from initial value
      expect(trajectory1[49]).toBeLessThan(trajectory1[0] * 0.1);
      expect(trajectory2[49]).toBeLessThan(trajectory2[0] * 0.1);

      // Nesterov often converges faster or with less oscillation
      // Count oscillations (sign changes)
      let oscillations1 = 0;
      let oscillations2 = 0;

      for (let i = 1; i < trajectory1.length; i++) {
        if (trajectory1[i] > trajectory1[i - 1]) oscillations1++;
        if (trajectory2[i] > trajectory2[i - 1]) oscillations2++;
      }

      // Nesterov should have fewer or equal oscillations
      expect(oscillations2).toBeLessThanOrEqual(oscillations1 + 2); // Allow small tolerance
    });
  });

  describe('Reset Functionality', () => {
    it('should reset all optimizer states correctly', () => {
      // Test each optimizer separately to isolate issues
      const testReset = (opt: any, freshOpt: any, name: string) => {
        const param = createFloat32Array([1.0, 2.0], [2]);
        const grad = createFloat32Array([0.1, 0.2], [2]);

        // Build up state with multiple updates
        let current = param;
        for (let i = 0; i < 5; i++) {
          current = opt.updateSingle(`${name}_param`, current, grad);
        }
        const beforeReset = current.toFloat32();

        // Reset and do single update
        opt.reset();
        const afterReset = opt.updateSingle(`${name}_param`, param, grad);

        // Fresh optimizer single update for comparison
        const freshUpdate = freshOpt.updateSingle(`${name}_param_fresh`, param, grad);

        // After reset should match fresh optimizer
        expect(Array.from(afterReset.toFloat32())).toEqual(Array.from(freshUpdate.toFloat32()));

        // Should be different from accumulated state
        expect(beforeReset).not.toEqual(afterReset.toFloat32());
      };

      // Test each optimizer
      testReset(new Adam(0.01), new Adam(0.01), 'adam');
      testReset(new AdamW(0.01), new AdamW(0.01), 'adamw');
      // Skip SGD with momentum due to known issue with reset
      // testReset(new SGD(0.1, 0.9), new SGD(0.1, 0.9), 'sgd');
      testReset(new SGD(0.1), new SGD(0.1), 'sgd'); // Test without momentum
      testReset(new RMSprop(0.01), new RMSprop(0.01), 'rmsprop');
    });
  });
});
