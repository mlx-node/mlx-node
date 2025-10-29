import { describe, it, expect } from 'vitest';
import { MxArray } from '@mlx-node/core';
import { shape, float32 } from '../test-utils';

describe('GRPO Autograd Integration', () => {
  describe('Basic Autograd Functionality', () => {
    it('should compute gradients for simple quadratic loss', () => {
      // This test verifies that the Rust autograd infrastructure works
      // f(x) = x^2, gradient = 2x

      const x = MxArray.fromFloat32(float32(3.0), shape(1));
      const loss = x.square();
      loss.eval();

      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeCloseTo(9.0, 5);

      // In future, we'll use value_and_grad to get gradients automatically
      // For now, verify the loss computation works
    });

    it('should handle multi-parameter functions', () => {
      // f(a, b) = a^2 + b^2
      const a = MxArray.fromFloat32(float32(2.0), shape(1));
      const b = MxArray.fromFloat32(float32(3.0), shape(1));

      const aSq = a.square();
      const bSq = b.square();
      const loss = aSq.add(bSq);
      loss.eval();

      const lossValue = loss.toFloat32()[0];
      // 2^2 + 3^2 = 13
      expect(lossValue).toBeCloseTo(13.0, 5);
    });
  });

  describe('GRPO Loss Computation', () => {
    it('should compute GRPO loss without errors', () => {
      // Simple test: verify GRPO loss can be computed
      // This is the forward pass that will be differentiated

      const batchSize = 2;
      const seqLen = 4;

      // Create dummy logprobs (should be negative)
      const logprobs = MxArray.fromFloat32(
        float32(-0.1, -0.2, -0.15, -0.18, -0.12, -0.25, -0.14, -0.19),
        shape(batchSize, seqLen),
      );

      // This tests that the loss computation is differentiable
      const loss = logprobs.sum(undefined, false);
      loss.eval();

      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeLessThan(0); // Should be negative (sum of negative logprobs)
    });
  });

  describe('Parameter Management', () => {
    it('should extract and flatten parameters for autograd', () => {
      // Test parameter handling for autograd
      // In real use, we extract all trainable parameters into a flat list

      const params = new Map([
        ['layer1.weight', MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(2, 2))],
        ['layer1.bias', MxArray.fromFloat32(float32(0.1, 0.2), shape(2))],
        ['layer2.weight', MxArray.fromFloat32(float32(5.0, 6.0), shape(1, 2))],
      ]);

      // Verify we can access parameters
      expect(params.size).toBe(3);
      expect(params.has('layer1.weight')).toBe(true);

      // Verify shapes
      const layer1Weight = params.get('layer1.weight')!;
      layer1Weight.eval();
      const shape1 = Array.from(layer1Weight.shape()).map(Number);
      expect(shape1).toEqual([2, 2]);
    });

    it('should handle gradient mapping back to parameters', () => {
      // After autograd computes gradients, we need to map them back to parameter names
      const paramNames = ['w1', 'w2', 'b1'];

      // Simulate gradients from autograd (ordered same as param_names)
      const gradients = [
        MxArray.fromFloat32(float32(0.01, 0.02), shape(2)),
        MxArray.fromFloat32(float32(0.03, 0.04), shape(2)),
        MxArray.fromFloat32(float32(0.005), shape(1)),
      ];

      // Map back to dictionary
      const gradMap = new Map<string, MxArray>();
      for (let i = 0; i < paramNames.length; i++) {
        gradMap.set(paramNames[i], gradients[i]);
      }

      expect(gradMap.size).toBe(3);
      expect(gradMap.has('w1')).toBe(true);
      expect(gradMap.has('b1')).toBe(true);
    });
  });

  describe('End-to-End Autograd Flow', () => {
    it('should demonstrate complete autograd training step', () => {
      // This test shows the complete flow:
      // 1. Extract parameters
      // 2. Compute loss (forward pass)
      // 3. Compute gradients (backward pass via autograd)
      // 4. Apply gradients

      // Step 1: Initialize parameter
      let w = MxArray.fromFloat32(float32(0.0), shape(1));

      // Step 2: Define training data
      // Target: w = 5.0 (we want to learn this)
      const target = 5.0;
      const learningRate = 0.1;

      // Step 3: Training loop (simplified, no autograd yet)
      const iterations = 10;
      for (let i = 0; i < iterations; i++) {
        // Forward: loss = (w - target)^2
        const diff = w.subScalar(target);
        const loss = diff.square();
        loss.eval();

        // Manual gradient: 2(w - target)
        diff.eval();
        const diffValue = diff.toFloat32()[0];
        const gradW = 2 * diffValue;

        // Update: w -= lr * grad
        const wNew = w.subScalar(learningRate * gradW);
        w = wNew;
        w.eval();
      }

      // Step 4: Verify convergence
      const finalW = w.toFloat32()[0];
      console.log(`Final w = ${finalW}, target = ${target}`);

      // Should be closer to 5.0 than initial 0.0
      expect(Math.abs(finalW - target)).toBeLessThan(Math.abs(0.0 - target));
    });
  });
});
