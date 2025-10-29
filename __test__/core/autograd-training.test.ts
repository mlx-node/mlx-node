import { describe, it, expect } from 'vitest';
import { MxArray, Adam } from '@mlx-node/core';
import { shape, float32 } from '../test-utils';

describe('Autograd Training Examples', () => {
  describe('Simple Linear Regression', () => {
    it('should train a linear model with autograd (simulated)', () => {
      // This test demonstrates the pattern for using MLX autograd
      // Once value_and_grad is exposed to TypeScript, we can use it directly

      // Simple linear model: y = w * x + b
      // We want to learn w=2, b=3 from data

      // Training data
      const xData = [1.0, 2.0, 3.0, 4.0];
      const yData = [5.0, 7.0, 9.0, 11.0]; // y = 2x + 3

      // Initialize parameters
      let w = MxArray.fromFloat32(float32(0.0), shape(1));
      let b = MxArray.fromFloat32(float32(0.0), shape(1));

      const learningRate = 0.01;
      const epochs = 200;

      let initialAvgLoss: number | null = null;
      let finalAvgLoss: number | null = null;

      for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;

        for (let i = 0; i < xData.length; i++) {
          const x = MxArray.fromFloat32(float32(xData[i]), shape(1));
          const yTrue = MxArray.fromFloat32(float32(yData[i]), shape(1));

          // Forward pass: y_pred = w * x + b
          const wx = w.mul(x);
          const yPred = wx.add(b);

          // Loss: MSE = (y_pred - y_true)^2
          const diff = yPred.sub(yTrue);
          const loss = diff.square();
          loss.eval();

          const lossValue = loss.toFloat32()[0];
          totalLoss += lossValue;

          // Manual gradient computation (in real code, this would use value_and_grad)
          // d(loss)/d(w) = 2 * (y_pred - y_true) * x
          // d(loss)/d(b) = 2 * (y_pred - y_true)

          diff.eval();
          const diffValue = diff.toFloat32()[0];

          const gradW = MxArray.fromFloat32(float32(2 * diffValue * xData[i]), shape(1));
          const gradB = MxArray.fromFloat32(float32(2 * diffValue), shape(1));

          // Update parameters: w -= lr * grad_w, b -= lr * grad_b
          const wUpdate = gradW.mulScalar(learningRate);
          const bUpdate = gradB.mulScalar(learningRate);

          w = w.sub(wUpdate);
          b = b.sub(bUpdate);

          w.eval();
          b.eval();
        }

        // Track convergence
        const avgLoss = totalLoss / xData.length;
        if (epoch === 0) {
          initialAvgLoss = avgLoss;
        }
        if (epoch === epochs - 1) {
          finalAvgLoss = avgLoss;
        }
      }

      // Verify loss decreased during training
      expect(initialAvgLoss).not.toBeNull();
      expect(finalAvgLoss).not.toBeNull();
      expect(finalAvgLoss!).toBeLessThan(initialAvgLoss!);

      // Check final parameters
      const finalW = w.toFloat32()[0];
      const finalB = b.toFloat32()[0];

      // Should be close to w=2, b=3 (relaxed tolerance for manual gradient descent)
      expect(finalW).toBeCloseTo(2.0, 0);
      expect(finalB).toBeCloseTo(3.0, 0);
    });
  });

  describe('Gradient Descent with Adam', () => {
    it('should optimize quadratic function with manual gradients', () => {
      // Minimize f(x) = (x - 5)^2
      // Gradient: f'(x) = 2(x - 5)
      // Minimum at x = 5

      let x = MxArray.fromFloat32(float32(0.0), shape(1));

      const learningRate = 0.1;
      const adam = new Adam(learningRate);
      const iterations = 50;

      let initialLoss: number | null = null;
      let finalLoss: number | null = null;

      for (let i = 0; i < iterations; i++) {
        // Forward: loss = (x - 5)^2
        const xMinus5 = x.subScalar(5.0);
        const loss = xMinus5.square();
        loss.eval();

        const lossVal = loss.toFloat32()[0];
        if (i === 0) {
          initialLoss = lossVal;
        }
        if (i === iterations - 1) {
          finalLoss = lossVal;
        }

        // Gradient: 2(x - 5)
        xMinus5.eval();
        const xMinus5Val = xMinus5.toFloat32()[0];
        const gradX = MxArray.fromFloat32(float32(2 * xMinus5Val), shape(1));

        // Update with Adam
        x = adam.updateSingle('x', x, gradX);
        x.eval();
      }

      // Verify loss decreased
      expect(initialLoss).not.toBeNull();
      expect(finalLoss).not.toBeNull();
      expect(finalLoss!).toBeLessThan(initialLoss!);

      const finalX = x.toFloat32()[0];

      // Should converge to x = 5 (relaxed tolerance)
      expect(finalX).toBeCloseTo(5.0, 0);
    });
  });

  describe('Multi-parameter Optimization', () => {
    it('should optimize multi-parameter function', () => {
      // Minimize f(a, b) = (a - 3)^2 + (b - 4)^2
      // Gradients: df/da = 2(a - 3), df/db = 2(b - 4)
      // Minimum at a=3, b=4

      let a = MxArray.fromFloat32(float32(0.0), shape(1));
      let b = MxArray.fromFloat32(float32(0.0), shape(1));

      const learningRate = 0.1;
      const adam = new Adam(learningRate);
      const iterations = 50;

      let initialLoss: number | null = null;
      let finalLoss: number | null = null;

      for (let i = 0; i < iterations; i++) {
        // Forward: loss = (a - 3)^2 + (b - 4)^2
        const aMinus3 = a.subScalar(3.0);
        const bMinus4 = b.subScalar(4.0);
        const lossA = aMinus3.square();
        const lossB = bMinus4.square();
        const loss = lossA.add(lossB);
        loss.eval();

        const lossVal = loss.toFloat32()[0];
        if (i === 0) {
          initialLoss = lossVal;
        }
        if (i === iterations - 1) {
          finalLoss = lossVal;
        }

        // Gradients
        aMinus3.eval();
        bMinus4.eval();
        const aMinus3Val = aMinus3.toFloat32()[0];
        const bMinus4Val = bMinus4.toFloat32()[0];

        const gradA = MxArray.fromFloat32(float32(2 * aMinus3Val), shape(1));
        const gradB = MxArray.fromFloat32(float32(2 * bMinus4Val), shape(1));

        // Update with Adam
        a = adam.updateSingle('a', a, gradA);
        b = adam.updateSingle('b', b, gradB);
        a.eval();
        b.eval();
      }

      // Verify loss decreased
      expect(initialLoss).not.toBeNull();
      expect(finalLoss).not.toBeNull();
      expect(finalLoss!).toBeLessThan(initialLoss!);

      const finalA = a.toFloat32()[0];
      const finalB = b.toFloat32()[0];

      // Should converge to a=3, b=4 (relaxed tolerance)
      expect(finalA).toBeCloseTo(3.0, 0);
      expect(finalB).toBeCloseTo(4.0, 0);
    });
  });
});
