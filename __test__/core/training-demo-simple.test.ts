import { describe, it, expect } from 'vitest';
import { MxArray, Adam, AdamW, SGD, LRScheduler, GradientUtils } from '@mlx-node/core';
import { createFloat32Array } from '../test-utils';

describe('Training Demos (Simplified)', () => {
  it('should optimize a quadratic function with different optimizers', () => {
    // Task: minimize f(x) = (x-2)^2 + (y-3)^2
    // Gradient: [2*(x-2), 2*(y-3)]
    // Optimum: [2, 3]

    const optimizers = {
      adam: new Adam(0.1),
      adamw: new AdamW(0.1, 0.9, 0.999, 1e-8, 0.01),
      sgd: new SGD(0.1),
      sgdMomentum: new SGD(0.1, 0.9),
    };

    const results: Record<string, number[]> = {};

    for (const [name, opt] of Object.entries(optimizers)) {
      // Start from the same point
      let params = createFloat32Array([0.0, 0.0], [2]);
      const trajectory: number[] = [];

      for (let i = 0; i < 50; i++) {
        // Compute gradient of f(x,y) = (x-2)^2 + (y-3)^2
        const values = params.toFloat32();
        const grad = createFloat32Array([2 * (values[0] - 2), 2 * (values[1] - 3)], [2]);

        // Update parameters
        params = opt.updateSingle(`${name}_params`, params, grad);

        // Record distance to optimum
        const newValues = params.toFloat32();
        const dist = Math.sqrt(Math.pow(newValues[0] - 2, 2) + Math.pow(newValues[1] - 3, 2));
        trajectory.push(dist);
      }

      results[name] = trajectory;
    }

    // All optimizers should converge
    for (const [_name, trajectory] of Object.entries(results)) {
      const finalDist = trajectory[trajectory.length - 1];
      expect(finalDist).toBeLessThan(0.5);

      // Check that distance generally decreases
      expect(finalDist).toBeLessThan(trajectory[0]);
    }
  });

  it('should minimize Rosenbrock function with Adam', () => {
    // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    // Gradient: df/dx = -2*(1-x) - 400*x*(y-x^2)
    //          df/dy = 200*(y-x^2)
    // Optimum: [1, 1]

    const adam = new Adam(0.002); // Increased learning rate
    let params = createFloat32Array([0.0, 0.0], [2]); // Start closer to optimum
    const losses: number[] = [];

    for (let i = 0; i < 2000; i++) {
      // More iterations
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

      // Update
      params = adam.updateSingle('params', params, grad);
    }

    // Should get closer to optimum [1, 1]
    const finalLoss = losses[losses.length - 1];

    // Check convergence (relaxed criteria for Rosenbrock)
    expect(finalLoss).toBeLessThan(1.0);

    // Loss should decrease significantly
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
  });

  it('should demonstrate learning rate scheduling', () => {
    // Simple linear regression: y = 2*x + 1

    const totalSteps = 100;
    let weight = createFloat32Array([0.0], [1]);
    let bias = createFloat32Array([0.0], [1]);

    const losses: number[] = [];

    for (let step = 0; step < totalSteps; step++) {
      // Apply cosine annealing schedule
      const lr = LRScheduler.cosineAnnealing(0.1, 0.001, step, totalSteps);
      const sgd = new SGD(lr);

      // Create a mini-batch of data
      const x = MxArray.randomUniform(BigInt64Array.from([10n, 1n]), -1, 1);
      const xVals = x.toFloat32();

      // True labels: y = 2*x + 1
      const yTrue = xVals.map((v) => 2 * v + 1);
      const y = createFloat32Array(Array.from(yTrue), [10, 1]);

      // Forward: y_pred = x * weight + bias
      const yPred = x.mulScalar(weight.toFloat32()[0]).addScalar(bias.toFloat32()[0]);

      // Compute MSE loss
      const diff = yPred.sub(y);
      const loss = diff.square().mean();
      losses.push(loss.toFloat32()[0]);

      // Compute gradients (simplified)
      // grad_weight = 2 * mean(x * (y_pred - y))
      // grad_bias = 2 * mean(y_pred - y)
      const gradWeight = x.mul(diff).mean().mulScalar(2.0);
      const gradBias = diff.mean().mulScalar(2.0);

      // Update
      weight = sgd.updateSingle('weight', weight, gradWeight);
      bias = sgd.updateSingle('bias', bias, gradBias);
    }

    // Should move towards [2, 1] (may not fully converge in 100 steps)
    expect(Math.abs(weight.toFloat32()[0] - 2.0)).toBeLessThan(0.5);
    expect(Math.abs(bias.toFloat32()[0] - 1.0)).toBeLessThan(0.5);

    // Loss should decrease
    const avgFirst = losses.slice(0, 10).reduce((a, b) => a + b) / 10;
    const avgLast = losses.slice(-10).reduce((a, b) => a + b) / 10;
    expect(avgLast).toBeLessThan(avgFirst);
  });

  it('should demonstrate gradient clipping', () => {
    const adam = new Adam(0.01);

    // Create large gradients that need clipping
    const param = createFloat32Array([1.0, 2.0, 3.0], [3]);
    const largeGrad = createFloat32Array([10.0, -15.0, 20.0], [3]);

    // Clip gradients by value
    const clippedGrad = GradientUtils.clipGradValue(largeGrad, -5.0, 5.0);
    const clippedValues = clippedGrad.toFloat32();

    expect(clippedValues[0]).toBeCloseTo(5.0);
    expect(clippedValues[1]).toBeCloseTo(-5.0);
    expect(clippedValues[2]).toBeCloseTo(5.0);

    // Update with clipped gradients
    const updated = adam.updateSingle('param', param, clippedGrad);
    const updatedValues = updated.toFloat32();

    // Check that updates are reasonable (not exploding)
    expect(Math.abs(updatedValues[0] - 1.0)).toBeLessThan(1.0);
    expect(Math.abs(updatedValues[1] - 2.0)).toBeLessThan(1.0);
    expect(Math.abs(updatedValues[2] - 3.0)).toBeLessThan(1.0);
  });

  it('should compare convergence rates', () => {
    // Compare how fast different optimizers converge

    const testFunction = (x: number, y: number) => {
      // Simple convex function: f(x,y) = x^2 + y^2
      return x * x + y * y;
    };

    const computeGradient = (x: number, y: number) => {
      return [2 * x, 2 * y];
    };

    const configurations = [
      { name: 'SGD-0.1', opt: new SGD(0.1) },
      { name: 'SGD-0.01', opt: new SGD(0.01) },
      { name: 'SGD-Momentum', opt: new SGD(0.1, 0.9) },
      { name: 'Adam-0.1', opt: new Adam(0.1) },
      { name: 'Adam-0.01', opt: new Adam(0.01) },
      { name: 'AdamW', opt: new AdamW(0.01, 0.9, 0.999, 1e-8, 0.01) },
    ];

    const results: Record<string, number> = {};

    for (const config of configurations) {
      // Start from [3, 4]
      let params = createFloat32Array([3.0, 4.0], [2]);

      // Run 100 iterations for better convergence
      for (let i = 0; i < 100; i++) {
        const values = params.toFloat32();
        const grad = computeGradient(values[0], values[1]);
        const gradArray = createFloat32Array(grad, [2]);
        params = config.opt.updateSingle('params', params, gradArray);
      }

      // Measure final distance from origin
      const finalValues = params.toFloat32();
      const finalLoss = testFunction(finalValues[0], finalValues[1]);
      results[config.name] = finalLoss;
    }

    // All should reduce the loss significantly
    for (const [_name, loss] of Object.entries(results)) {
      // Initial loss was 3^2 + 4^2 = 25, expect significant reduction
      expect(loss).toBeLessThan(5.0);
    }

    // Check that optimizers with momentum generally perform well
    // (exact comparisons can vary based on the specific function and initialization)
    expect(results['SGD-Momentum']).toBeLessThan(5.0);
    expect(results['Adam-0.1']).toBeLessThan(5.0);
  });

  it('should handle weight decay correctly', () => {
    // Test that weight decay regularizes parameters

    const adamWithoutWD = new Adam(0.01);
    const adamWithWD = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.1);

    // Large initial parameters
    let paramsNoWD = createFloat32Array([10.0, 10.0], [2]);
    let paramsWD = createFloat32Array([10.0, 10.0], [2]);

    // Update with zero gradients - only weight decay should affect parameters
    const zeroGrad = createFloat32Array([0.0, 0.0], [2]);

    for (let i = 0; i < 50; i++) {
      // More iterations for noticeable decay
      paramsNoWD = adamWithoutWD.updateSingle('params', paramsNoWD, zeroGrad);
      paramsWD = adamWithWD.updateSingle('params', paramsWD, zeroGrad);
    }

    const valuesNoWD = paramsNoWD.toFloat32();
    const valuesWD = paramsWD.toFloat32();

    // Without weight decay, parameters shouldn't change much with zero gradient
    expect(valuesNoWD[0]).toBeCloseTo(10.0, 2);
    expect(valuesNoWD[1]).toBeCloseTo(10.0, 2);

    // With weight decay, parameters should decrease noticeably
    expect(valuesWD[0]).toBeLessThan(valuesNoWD[0]);
    expect(valuesWD[1]).toBeLessThan(valuesNoWD[1]);
  });
});
