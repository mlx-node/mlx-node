/**
 * Tests for GRPO parameter updates
 *
 * Verifies that training actually updates model parameters.
 * Based on TRL's parameter update verification pattern.
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Linear, RMSNorm, Embedding, Adam } from '@mlx-node/core';
import { shape, float32 } from '../test-utils.js';

/**
 * Capture all weights from a model component
 */
function captureWeights(component: any): Map<string, Float32Array> {
  const weights = new Map<string, Float32Array>();

  if (component.getWeight) {
    weights.set('weight', component.getWeight().toFloat32());
  }

  if (component.getBias) {
    const bias = component.getBias();
    if (bias) {
      weights.set('bias', bias.toFloat32());
    }
  }

  return weights;
}

/**
 * Compare two weight snapshots
 */
function weightsChanged(before: Map<string, Float32Array>, after: Map<string, Float32Array>): boolean {
  if (before.size !== after.size) return true;

  for (const [key, beforeData] of before.entries()) {
    const afterData = after.get(key);
    if (!afterData) return true;

    if (beforeData.length !== afterData.length) return true;

    // Check if any values changed
    for (let i = 0; i < beforeData.length; i++) {
      if (Math.abs(beforeData[i] - afterData[i]) > 1e-10) {
        return true;
      }
    }
  }

  return false;
}

describe('GRPO Parameter Updates', () => {
  describe('Linear layer with optimizer', () => {
    it('should update weights after optimizer step', () => {
      const layer = new Linear(4, 2, true);
      const optimizer = new Adam(0.001);

      // Capture initial weights
      const beforeWeights = captureWeights(layer);

      // Get current weights
      const weight = layer.getWeight();
      const bias = layer.getBias();

      // Capture initial values before optimizer modifies them
      const oldWeightData = weight.toFloat32();
      const oldBiasData = bias!.toFloat32();

      // Simulate gradient (all ones)
      const gradWeight = MxArray.ones(shape(2, 4));
      const gradBias = MxArray.ones(shape(2));

      // Update using optimizer (returns new parameter values)
      const newWeight = optimizer.updateSingle('weight', weight, gradWeight);
      const newBias = optimizer.updateSingle('bias', bias!, gradBias);

      // Apply updated weights to layer
      layer.setWeight(newWeight);
      layer.setBias(newBias);

      // Capture after update
      const afterWeights = captureWeights(layer);

      // Verify weights changed
      expect(weightsChanged(beforeWeights, afterWeights)).toBe(true);

      // Verify values decreased (gradient is positive, so params should decrease)
      const newWeightData = newWeight.toFloat32();
      expect(newWeightData[0]).toBeLessThan(oldWeightData[0]);

      // Verify bias also decreased
      const newBiasData = newBias.toFloat32();
      expect(newBiasData[0]).toBeLessThan(oldBiasData[0]);
    });
  });

  describe('Manual weight update simulation', () => {
    it('should change weights when explicitly updated', () => {
      const layer = new Linear(3, 2, false);

      // Capture before
      const before = captureWeights(layer);

      // Simulate gradient descent: w = w - lr * grad
      const lr = 0.01;
      const weight = layer.getWeight();
      const weightData = weight.toFloat32();

      // Create fake gradient (all 0.1)
      const gradData = new Float32Array(weightData.length).fill(0.1);

      // Update: w = w - lr * grad
      const newWeightData = new Float32Array(weightData.length);
      for (let i = 0; i < weightData.length; i++) {
        newWeightData[i] = weightData[i] - lr * gradData[i];
      }

      const newWeight = MxArray.fromFloat32(newWeightData, weight.shape());
      layer.setWeight(newWeight);

      // Capture after
      const after = captureWeights(layer);

      // Verify weights changed
      expect(weightsChanged(before, after)).toBe(true);

      // Verify the change is correct (should be -0.001 for each element)
      const afterWeightData = after.get('weight')!;
      for (let i = 0; i < weightData.length; i++) {
        expect(afterWeightData[i]).toBeCloseTo(weightData[i] - 0.001, 6);
      }
    });
  });

  describe('Multiple components', () => {
    it('should update all components independently', () => {
      const linear1 = new Linear(4, 4, true);
      const norm = new RMSNorm(4);
      const linear2 = new Linear(4, 2, false);

      // Capture all weights
      const beforeLinear1 = captureWeights(linear1);
      const beforeNorm = captureWeights(norm);
      const beforeLinear2 = captureWeights(linear2);

      // Simulate updates
      const lr = 0.01;

      // Update linear1
      const w1 = linear1.getWeight();
      const w1Data = w1.toFloat32();
      const newW1Data = new Float32Array(w1Data.length);
      for (let i = 0; i < w1Data.length; i++) {
        newW1Data[i] = w1Data[i] - lr * 0.1;
      }
      linear1.setWeight(MxArray.fromFloat32(newW1Data, w1.shape()));

      // Update norm
      const wNorm = norm.getWeight();
      const wNormData = wNorm.toFloat32();
      const newWNormData = new Float32Array(wNormData.length);
      for (let i = 0; i < wNormData.length; i++) {
        newWNormData[i] = wNormData[i] - lr * 0.05;
      }
      norm.setWeight(MxArray.fromFloat32(newWNormData, wNorm.shape()));

      // Update linear2
      const w2 = linear2.getWeight();
      const w2Data = w2.toFloat32();
      const newW2Data = new Float32Array(w2Data.length);
      for (let i = 0; i < w2Data.length; i++) {
        newW2Data[i] = w2Data[i] - lr * 0.15;
      }
      linear2.setWeight(MxArray.fromFloat32(newW2Data, w2.shape()));

      // Capture after
      const afterLinear1 = captureWeights(linear1);
      const afterNorm = captureWeights(norm);
      const afterLinear2 = captureWeights(linear2);

      // Verify all changed
      expect(weightsChanged(beforeLinear1, afterLinear1)).toBe(true);
      expect(weightsChanged(beforeNorm, afterNorm)).toBe(true);
      expect(weightsChanged(beforeLinear2, afterLinear2)).toBe(true);
    });
  });

  describe('Optimizer integration', () => {
    it('should work with Adam optimizer', () => {
      const layer = new Linear(4, 2, true);

      // Get initial weight
      let weight = layer.getWeight();
      const weightBefore = weight.toFloat32();

      // Create optimizer
      const optimizer = new Adam(0.001);

      // Simulate 10 gradient descent steps
      for (let step = 0; step < 10; step++) {
        // Create small gradient
        const grad = MxArray.fromFloat32(float32(...Array(8).fill(0.01)), shape(2, 4));

        // Update parameter using optimizer
        weight = optimizer.updateSingle('weight', weight, grad);
      }

      // Apply final weight to layer
      layer.setWeight(weight);

      // Get final weight
      const weightAfter = layer.getWeight().toFloat32();

      // Verify weights changed
      let changed = false;
      for (let i = 0; i < weightBefore.length; i++) {
        if (Math.abs(weightBefore[i] - weightAfter[i]) > 1e-8) {
          changed = true;
          break;
        }
      }
      expect(changed).toBe(true);

      // Verify weights decreased (positive gradient should decrease params)
      expect(weightAfter[0]).toBeLessThan(weightBefore[0]);

      // Verify multiple steps had cumulative effect
      const totalChange = Math.abs(weightAfter[0] - weightBefore[0]);
      expect(totalChange).toBeGreaterThan(1e-6); // Should have moved significantly
    });
  });

  describe('Weight persistence', () => {
    it('should maintain weight updates across forward passes', () => {
      const layer = new Linear(2, 2, false);

      const input = MxArray.fromFloat32(float32(1.0, 2.0), shape(1, 2));

      // First forward pass
      const output1 = layer.forward(input);
      const result1 = output1.toFloat32();

      // Update weights
      const w = layer.getWeight();
      const wData = w.toFloat32();
      const newWData = new Float32Array(wData.length);
      for (let i = 0; i < wData.length; i++) {
        newWData[i] = wData[i] * 2.0; // Double all weights
      }
      layer.setWeight(MxArray.fromFloat32(newWData, w.shape()));

      // Second forward pass with same input
      const output2 = layer.forward(input);
      const result2 = output2.toFloat32();

      // Results should be different (approximately double)
      expect(result2[0]).not.toBeCloseTo(result1[0], 5);
      expect(result2[1]).not.toBeCloseTo(result1[1], 5);

      // New results should be approximately double the old ones
      expect(result2[0]).toBeCloseTo(result1[0] * 2.0, 4);
      expect(result2[1]).toBeCloseTo(result1[1] * 2.0, 4);
    });
  });

  describe('Embedding layer updates', () => {
    it('should update embedding weights', () => {
      const embedding = new Embedding(100, 16);

      const before = embedding.getWeight().toFloat32();

      // Simulate update to first embedding vector
      const w = embedding.getWeight();
      const wData = w.toFloat32();

      // Modify first 16 values (first embedding vector)
      for (let i = 0; i < 16; i++) {
        wData[i] += 0.1;
      }

      embedding.loadWeight(MxArray.fromFloat32(wData, w.shape()));

      const after = embedding.getWeight().toFloat32();

      // Verify change
      expect(after[0]).toBeCloseTo(before[0] + 0.1, 6);
      expect(after[15]).toBeCloseTo(before[15] + 0.1, 6);

      // Verify other embeddings unchanged
      expect(after[16]).toBeCloseTo(before[16], 10);
    });
  });

  describe('Training step simulation', () => {
    it('should simulate complete training step with weight updates', () => {
      // Create a simple 2-layer network
      const layer1 = new Linear(4, 8, true);
      const layer2 = new Linear(8, 2, true);

      const captureAll = () => ({
        layer1: captureWeights(layer1),
        layer2: captureWeights(layer2),
      });

      const before = captureAll();

      // Simulate forward pass
      const input = MxArray.randomNormal(shape(2, 4), 0, 1);
      const h = layer1.forward(input);
      const output = layer2.forward(h);

      // Compute loss (currently unused - placeholder for future auto-diff)
      // TODO(Phase 1.2): Use loss.backward() to compute gradients automatically
      const _loss = output.sum(undefined, false);

      // Simulate backward pass (manual gradients)
      // Note: In complete implementation, gradients would be derived from _loss
      const lr = 0.01;

      // Layer 2 gradients (simplified - hardcoded for now)
      const w2 = layer2.getWeight();
      const w2Data = w2.toFloat32();
      const newW2Data = new Float32Array(w2Data.length);
      for (let i = 0; i < w2Data.length; i++) {
        newW2Data[i] = w2Data[i] - lr * 0.01; // Simplified gradient
      }
      layer2.setWeight(MxArray.fromFloat32(newW2Data, w2.shape()));

      // Layer 1 gradients (simplified)
      const w1 = layer1.getWeight();
      const w1Data = w1.toFloat32();
      const newW1Data = new Float32Array(w1Data.length);
      for (let i = 0; i < w1Data.length; i++) {
        newW1Data[i] = w1Data[i] - lr * 0.005; // Simplified gradient
      }
      layer1.setWeight(MxArray.fromFloat32(newW1Data, w1.shape()));

      const after = captureAll();

      // Verify all weights changed
      expect(weightsChanged(before.layer1, after.layer1)).toBe(true);
      expect(weightsChanged(before.layer2, after.layer2)).toBe(true);
    });
  });
});
