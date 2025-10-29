/**
 * Tests for manual backpropagation through model components
 *
 * This tests our ability to manually chain gradient computations
 * for training a language model.
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Linear, RMSNorm, Activations, Gradients } from '@mlx-node/core';
import { shape, float32 } from '../test-utils.js';

describe('Manual Backpropagation', () => {
  it('should compute gradients for a simple Linear layer', () => {
    // Create a simple linear layer
    const layer = new Linear(4, 2, true); // 4 inputs -> 2 outputs

    // Create input
    const input = MxArray.randomNormal(shape(1, 4), 0, 0.1);

    // Forward pass
    const output = layer.forward(input);

    // Compute loss (placeholder for auto-diff - shows what should drive gradients)
    // TODO(Phase 1.2): Implement loss.backward() to derive gradOutput automatically
    const _loss = output.sum(undefined, false);

    // Gradient of loss w.r.t. output: ∂(sum)/∂output = ones
    // In complete implementation: gradOutput = _loss.backward()
    const gradOutput = MxArray.ones(shape(1, 2));

    // Backward pass
    const grads = Gradients.linearBackward(input, layer.getWeight(), gradOutput, true);

    // Check we got 3 gradients: input, weight, bias
    expect(grads.length).toBe(3);

    const [gradInput, gradWeight, gradBias] = grads;

    // Check shapes
    expect(Array.from(gradInput.shape())).toEqual([1n, 4n]);
    expect(Array.from(gradWeight.shape())).toEqual([2n, 4n]); // weight is [out_features, in_features]
    expect(Array.from(gradBias.shape())).toEqual([2n]);

    // Check gradients are non-zero
    const gradWeightSum = Math.abs(gradWeight.sum(undefined, false).toFloat32()[0]);
    const gradBiasSum = Math.abs(gradBias.sum(undefined, false).toFloat32()[0]);

    expect(gradWeightSum).toBeGreaterThan(0);
    expect(gradBiasSum).toBeGreaterThan(0);
  });

  it('should compute gradients for Linear + RMSNorm chain', () => {
    const hiddenSize = 8;

    // Create components
    const linear = new Linear(hiddenSize, hiddenSize, true);
    const rmsNorm = new RMSNorm(hiddenSize);

    // Create input
    const input = MxArray.randomNormal(shape(2, hiddenSize), 0, 0.1);

    // Forward pass
    const linearOut = linear.forward(input);
    const normalizedOut = rmsNorm.forward(linearOut);

    // Compute loss (placeholder for auto-diff)
    // TODO(Phase 1.2): Use loss.backward() to chain gradients automatically
    const _loss = normalizedOut.sum(undefined, false);

    // Manual gradient for testing backward pass
    // In complete implementation: gradOutput = _loss.backward()
    const gradOutput = MxArray.ones(shape(2, hiddenSize));
    const rmsNormGrads = Gradients.rmsNormBackward(linearOut, rmsNorm.getWeight(), gradOutput, 1e-5);

    const [gradLinearOut, gradRmsWeight] = rmsNormGrads;

    // Backward through Linear
    const linearGrads = Gradients.linearBackward(input, linear.getWeight(), gradLinearOut, true);

    const [gradInput, gradLinearWeight, gradLinearBias] = linearGrads;

    // Check all gradients are non-zero
    expect(Math.abs(gradInput.sum(undefined, false).toFloat32()[0])).toBeGreaterThan(0);
    expect(Math.abs(gradLinearWeight.sum(undefined, false).toFloat32()[0])).toBeGreaterThan(0);
    expect(Math.abs(gradLinearBias.sum(undefined, false).toFloat32()[0])).toBeGreaterThan(0);
    expect(Math.abs(gradRmsWeight.sum(undefined, false).toFloat32()[0])).toBeGreaterThan(0);
  });

  it('should compute gradients for SiLU activation', () => {
    const input = MxArray.fromFloat32(float32(0.0, 1.0, -1.0, 2.0), shape(4));

    // Forward pass
    const output = Activations.silu(input);

    // Verify forward pass shape
    expect(Array.from(output.shape())).toEqual([4n]);

    // Verify output is non-zero (SiLU(x) = x * sigmoid(x))
    const outputData = output.toFloat32();
    expect(outputData.length).toBe(4);
    expect(Math.abs(outputData[1])).toBeGreaterThan(0); // SiLU(1.0) should be non-zero

    // Assume gradient from next layer is all ones
    const gradOutput = MxArray.ones(shape(4));

    // Backward
    const gradInput = Gradients.siluBackward(input, gradOutput);

    // Check gradient shape matches input
    expect(Array.from(gradInput.shape())).toEqual([4n]);

    // Check gradients are reasonable
    const gradData = gradInput.toFloat32();
    expect(gradData.length).toBe(4);

    // SiLU gradient should be positive for positive inputs
    // and depend on the sigmoid
    for (let i = 0; i < gradData.length; i++) {
      expect(Math.abs(gradData[i])).toBeGreaterThan(0);
    }
  });

  it('should verify gradient computation with finite differences', () => {
    // Simple test: verify linear backward is correct using finite differences
    const eps = 1e-4;

    const linear = new Linear(3, 2, false); // no bias for simplicity
    const input = MxArray.fromFloat32(float32(1.0, 2.0, 3.0), shape(1, 3));

    // Forward
    const output = linear.forward(input);
    const loss = output.sum(undefined, false);
    const lossValue = loss.toFloat32()[0];

    // Analytical gradient
    const gradOutput = MxArray.ones(shape(1, 2));
    const grads = Gradients.linearBackward(input, linear.getWeight(), gradOutput, false);
    const gradWeight = grads[1];
    const gradWeightData = gradWeight.toFloat32();

    // Finite difference gradient
    const weight = linear.getWeight();
    const weightShape = weight.shape();
    const originalWeightData = weight.toFloat32();
    const fdGrads = new Float32Array(originalWeightData.length);

    for (let i = 0; i < originalWeightData.length; i++) {
      // Copy weight data
      const weightData = new Float32Array(originalWeightData);

      // Perturb weight
      weightData[i] += eps;
      const newWeight = MxArray.fromFloat32(weightData, weightShape);
      linear.setWeight(newWeight);

      // Compute perturbed loss
      const newOutput = linear.forward(input);
      const newLoss = newOutput.sum(undefined, false);
      const newLossValue = newLoss.toFloat32()[0];

      // Compute finite difference
      fdGrads[i] = (newLossValue - lossValue) / eps;
    }

    // Restore original weight
    const originalWeight = MxArray.fromFloat32(originalWeightData, weightShape);
    linear.setWeight(originalWeight);

    // Compare analytical and finite difference gradients
    for (let i = 0; i < fdGrads.length; i++) {
      const diff = Math.abs(gradWeightData[i] - fdGrads[i]);
      const relError = diff / (Math.abs(fdGrads[i]) + 1e-8);

      // Should match within 1% relative error
      expect(relError).toBeLessThan(0.01);
    }
  });
});
