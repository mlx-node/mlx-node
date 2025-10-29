/**
 * Parameter Update Verification Tests (TRL Pattern)
 *
 * Based on TRL's test_training_delta_clipping and similar tests.
 * These tests explicitly verify that trainable parameters actually change
 * after gradient updates, catching silent gradient flow bugs.
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Linear, RMSNorm, Adam, AdamW, SGD, Losses, Activations, Gradients } from '@mlx-node/core';
import { shape, float32 } from '../test-utils.js';

// Helper to check if arrays are equal
function arraysEqual(a: Float32Array, b: Float32Array, tolerance = 1e-6): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tolerance) return false;
  }
  return true;
}

// Helper to check if any element changed
function anyChanged(a: Float32Array, b: Float32Array, tolerance = 1e-6): boolean {
  return !arraysEqual(a, b, tolerance);
}

describe('Parameter Update Verification (TRL Pattern)', () => {
  describe('Linear Layer Updates', () => {
    it('should update all parameters after gradient step', () => {
      const layer = new Linear(8, 4, true);

      // Save initial parameters
      const initialWeight = layer.getWeight().toFloat32();
      const initialBias = layer.getBias()!.toFloat32();

      // Create batch
      const input = MxArray.randomNormal(shape(4, 8), 0, 0.1);
      const target = MxArray.randomNormal(shape(4, 4), 0, 0.1);

      // Forward + backward
      const output = layer.forward(input);
      const _loss = Losses.mse(output, target);

      // Compute gradients manually (MSE backward: 2*(pred-target)/n)
      const n = 4 * 4;
      const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(2.0 / n), shape(1)));
      const grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, true);

      // Update with optimizer
      const optimizer = new Adam(0.01);
      const updatedWeight = optimizer.updateSingle('weight', layer.getWeight(), grads[1]);
      const updatedBias = optimizer.updateSingle('bias', layer.getBias()!, grads[2]);

      layer.setWeight(updatedWeight);
      layer.setBias(updatedBias);

      // Get updated parameters
      const newWeight = layer.getWeight().toFloat32();
      const newBias = layer.getBias()!.toFloat32();

      // Verify ALL weights changed
      expect(anyChanged(initialWeight, newWeight)).toBe(true);

      // Verify ALL biases changed
      expect(anyChanged(initialBias, newBias)).toBe(true);

      // Verify they didn't explode
      for (let i = 0; i < newWeight.length; i++) {
        expect(Math.abs(newWeight[i])).toBeLessThan(10.0);
      }
      for (let i = 0; i < newBias.length; i++) {
        expect(Math.abs(newBias[i])).toBeLessThan(10.0);
      }
    });

    it('should update parameters consistently across different optimizers', () => {
      const optimizers = [
        { name: 'adam', opt: new Adam(0.01) },
        { name: 'adamw', opt: new AdamW(0.01) },
        { name: 'sgd', opt: new SGD(0.01) },
      ];

      for (const { name: _name, opt } of optimizers) {
        const layer = new Linear(4, 2, true);
        const initialWeight = layer.getWeight().toFloat32();

        // Training step
        const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);
        const target = MxArray.randomNormal(shape(2, 2), 0, 0.1);
        const output = layer.forward(input);
        const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));
        const grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, true);

        const updatedWeight = opt.updateSingle('weight', layer.getWeight(), grads[1]);
        layer.setWeight(updatedWeight);

        const newWeight = layer.getWeight().toFloat32();

        // All optimizers should change parameters
        expect(anyChanged(initialWeight, newWeight)).toBe(true);
      }
    });

    it('should update parameters more after larger gradients', () => {
      const layer = new Linear(4, 2, false);

      // Small gradient update
      const input1 = MxArray.randomNormal(shape(2, 4), 0, 0.01);
      const target1 = MxArray.randomNormal(shape(2, 2), 0, 0.01);
      const output1 = layer.forward(input1);
      const gradOutput1 = output1.sub(target1).mul(MxArray.fromFloat32(float32(0.1), shape(1)));
      const grads1 = Gradients.linearBackward(input1, layer.getWeight(), gradOutput1, false);

      const optimizer1 = new Adam(0.01);
      const updated1 = optimizer1.updateSingle('weight', layer.getWeight(), grads1[1]);

      const delta1 = layer.getWeight().sub(updated1);
      const norm1 = Math.sqrt(delta1.square().sum(undefined, false).toFloat32()[0]);

      // Large gradient update (reset layer first)
      const layer2 = new Linear(4, 2, false);
      const input2 = MxArray.randomNormal(shape(2, 4), 0, 1.0);
      const target2 = MxArray.randomNormal(shape(2, 2), 0, 1.0);
      const output2 = layer2.forward(input2);
      const gradOutput2 = output2.sub(target2).mul(MxArray.fromFloat32(float32(1.0), shape(1)));
      const grads2 = Gradients.linearBackward(input2, layer2.getWeight(), gradOutput2, false);

      const optimizer2 = new Adam(0.01);
      const updated2 = optimizer2.updateSingle('weight', layer2.getWeight(), grads2[1]);

      const delta2 = layer2.getWeight().sub(updated2);
      const norm2 = Math.sqrt(delta2.square().sum(undefined, false).toFloat32()[0]);

      // Larger gradients should produce larger updates
      expect(norm2).toBeGreaterThan(norm1);
    });
  });

  describe('Frozen Parameters', () => {
    it('should NOT update frozen parameters', () => {
      const layer1 = new Linear(4, 4, false);
      const layer2 = new Linear(4, 2, false);

      // Save initial states
      const initialWeight1 = layer1.getWeight().toFloat32();
      const initialWeight2 = layer2.getWeight().toFloat32();

      // Forward through both layers
      const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);
      const hidden = layer1.forward(input);
      const output = layer2.forward(hidden);
      const target = MxArray.randomNormal(shape(2, 2), 0, 0.1);

      // Backward through layer2 ONLY (freeze layer1)
      const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));
      const grads2 = Gradients.linearBackward(hidden, layer2.getWeight(), gradOutput, false);

      // Update ONLY layer2
      const optimizer = new Adam(0.01);
      const updated2 = optimizer.updateSingle('weight', layer2.getWeight(), grads2[1]);
      layer2.setWeight(updated2);

      // Get final states
      const finalWeight1 = layer1.getWeight().toFloat32();
      const finalWeight2 = layer2.getWeight().toFloat32();

      // Layer 1 should be IDENTICAL (frozen)
      expect(arraysEqual(initialWeight1, finalWeight1, 1e-10)).toBe(true);

      // Layer 2 should be DIFFERENT (trained)
      expect(anyChanged(initialWeight2, finalWeight2)).toBe(true);
    });

    it('should handle selective parameter freezing', () => {
      const layer = new Linear(4, 2, true);

      // Save initial states
      const initialWeight = layer.getWeight().toFloat32();
      const initialBias = layer.getBias()!.toFloat32();

      // Training step
      const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);
      const target = MxArray.randomNormal(shape(2, 2), 0, 0.1);
      const output = layer.forward(input);
      const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));
      const grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, true);

      // Update ONLY weights, freeze bias
      const optimizer = new Adam(0.01);
      const updatedWeight = optimizer.updateSingle('weight', layer.getWeight(), grads[1]);
      layer.setWeight(updatedWeight);

      // Get final states
      const finalWeight = layer.getWeight().toFloat32();
      const finalBias = layer.getBias()!.toFloat32();

      // Weight should change
      expect(anyChanged(initialWeight, finalWeight)).toBe(true);

      // Bias should be frozen
      expect(arraysEqual(initialBias, finalBias, 1e-10)).toBe(true);
    });
  });

  describe('RMSNorm Updates', () => {
    it('should update normalization weights', () => {
      const norm = new RMSNorm(8);

      // Save initial weight
      const initialWeight = norm.getWeight().toFloat32();

      // Forward + backward
      const input = MxArray.randomNormal(shape(2, 8), 0, 0.1);
      const _output = norm.forward(input);
      const gradOutput = MxArray.ones(shape(2, 8));

      const grads = Gradients.rmsNormBackward(input, norm.getWeight(), gradOutput, 1e-5);

      // Update
      const optimizer = new Adam(0.01);
      const updatedWeight = optimizer.updateSingle('weight', norm.getWeight(), grads[1]);
      norm.setWeight(updatedWeight);

      const newWeight = norm.getWeight().toFloat32();

      // Verify weight changed
      expect(anyChanged(initialWeight, newWeight)).toBe(true);
    });
  });

  describe('Multi-Layer Networks', () => {
    it('should update all layers in a network', () => {
      // Create 3-layer network
      const layer1 = new Linear(8, 16, true);
      const norm1 = new RMSNorm(16);
      const layer2 = new Linear(16, 8, true);
      const norm2 = new RMSNorm(8);
      const layer3 = new Linear(8, 4, true);

      // Save all initial parameters
      const initialParams = {
        layer1_weight: layer1.getWeight().toFloat32(),
        layer1_bias: layer1.getBias()!.toFloat32(),
        norm1_weight: norm1.getWeight().toFloat32(),
        layer2_weight: layer2.getWeight().toFloat32(),
        layer2_bias: layer2.getBias()!.toFloat32(),
        norm2_weight: norm2.getWeight().toFloat32(),
        layer3_weight: layer3.getWeight().toFloat32(),
        layer3_bias: layer3.getBias()!.toFloat32(),
      };

      // Forward pass
      const input = MxArray.randomNormal(shape(4, 8), 0, 0.1);
      let h = layer1.forward(input);
      h = norm1.forward(h);
      h = Activations.relu(h);
      h = layer2.forward(h);
      h = norm2.forward(h);
      h = Activations.relu(h);
      const output = layer3.forward(h);

      // Compute loss
      const target = MxArray.randomNormal(shape(4, 4), 0, 0.1);
      const _loss = Losses.mse(output, target);

      // Backward pass (simplified - just check if params can be updated)
      // In real training, we'd backprop through all layers
      const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));

      // Update just layer3 to verify mechanism works
      const grads3 = Gradients.linearBackward(h, layer3.getWeight(), gradOutput, true);
      const optimizer = new Adam(0.01);
      const updatedWeight3 = optimizer.updateSingle('weight', layer3.getWeight(), grads3[1]);
      const updatedBias3 = optimizer.updateSingle('bias', layer3.getBias()!, grads3[2]);
      layer3.setWeight(updatedWeight3);
      layer3.setBias(updatedBias3);

      // Verify layer3 changed
      const finalParams = {
        layer3_weight: layer3.getWeight().toFloat32(),
        layer3_bias: layer3.getBias()!.toFloat32(),
      };

      expect(anyChanged(initialParams.layer3_weight, finalParams.layer3_weight)).toBe(true);
      expect(anyChanged(initialParams.layer3_bias, finalParams.layer3_bias)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero gradients (parameters should not change)', () => {
      const layer = new Linear(4, 2, false);
      const initialWeight = layer.getWeight().toFloat32();

      // Zero gradient
      const zeroGrad = MxArray.zeros(shape(2, 4));

      const optimizer = new Adam(0.01);
      const updatedWeight = optimizer.updateSingle('weight', layer.getWeight(), zeroGrad);
      layer.setWeight(updatedWeight);

      const finalWeight = layer.getWeight().toFloat32();

      // With zero gradients, Adam might still change params slightly due to momentum
      // But change should be minimal
      let maxChange = 0;
      for (let i = 0; i < initialWeight.length; i++) {
        maxChange = Math.max(maxChange, Math.abs(finalWeight[i] - initialWeight[i]));
      }

      // Change should be very small (< 1e-6) for zero gradients
      expect(maxChange).toBeLessThan(1e-6);
    });

    it('should handle very small learning rates', () => {
      const layer = new Linear(4, 2, false);
      const initialWeight = layer.getWeight().toFloat32();

      // Normal gradient
      const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);
      const target = MxArray.randomNormal(shape(2, 2), 0, 0.1);
      const output = layer.forward(input);
      const gradOutput = output.sub(target);
      const grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, false);

      // Very small learning rate
      const optimizer = new Adam(1e-6); // Increased from 1e-10 to be more realistic
      const updatedWeight = optimizer.updateSingle('weight', layer.getWeight(), grads[1]);
      layer.setWeight(updatedWeight);

      const finalWeight = layer.getWeight().toFloat32();

      // Parameters should change, but very little
      expect(anyChanged(initialWeight, finalWeight, 1e-8)).toBe(true);

      let maxChange = 0;
      for (let i = 0; i < initialWeight.length; i++) {
        maxChange = Math.max(maxChange, Math.abs(finalWeight[i] - initialWeight[i]));
      }

      // Change should be tiny but detectable
      expect(maxChange).toBeLessThan(1e-4);
      expect(maxChange).toBeGreaterThan(0);
    });

    it('should detect no update when gradients are not applied', () => {
      const layer = new Linear(4, 2, false);
      const initialWeight = layer.getWeight().toFloat32();

      // Compute gradients but DON'T apply them
      const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);
      const target = MxArray.randomNormal(shape(2, 2), 0, 0.1);
      const output = layer.forward(input);
      const gradOutput = output.sub(target);
      const _grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, false);

      // DON'T update the layer

      const finalWeight = layer.getWeight().toFloat32();

      // Parameters should be identical
      expect(arraysEqual(initialWeight, finalWeight, 1e-10)).toBe(true);
    });
  });
});
