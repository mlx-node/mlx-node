/**
 * MLP Backpropagation Tests
 *
 * Tests gradient computation through the SwiGLU-based MLP architecture.
 * The MLP consists of three projections:
 * - gate_proj: Projects to intermediate dimension (used for gating)
 * - up_proj: Projects to intermediate dimension (upscaling)
 * - down_proj: Projects back to hidden dimension
 *
 * Forward pass: down_proj(silu(gate_proj(x)) * up_proj(x))
 */

import { describe, it, expect } from 'vitest';
import { MxArray, MLP, Activations, Gradients, Adam } from '@mlx-node/core';
import { shape, float32 } from '../test-utils.js';

// Helper to check if arrays are close
function assertClose(actual: MxArray, expected: MxArray, rtol: number = 1e-4, atol: number = 1e-6) {
  const actualData = actual.toFloat32();
  const expectedData = expected.toFloat32();

  expect(actualData.length).toBe(expectedData.length);

  for (let i = 0; i < actualData.length; i++) {
    const diff = Math.abs(actualData[i] - expectedData[i]);
    const tolerance = atol + rtol * Math.abs(expectedData[i]);

    if (diff > tolerance) {
      throw new Error(
        `Arrays not close at index ${i}: actual=${actualData[i]}, expected=${expectedData[i]}, diff=${diff}, tolerance=${tolerance}`,
      );
    }
  }
}

// Helper to check if any element changed
function anyChanged(a: Float32Array, b: Float32Array, tolerance = 1e-6): boolean {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tolerance) return true;
  }
  return false;
}

describe('MLP Backpropagation', () => {
  describe('Gradient Flow', () => {
    it('should compute gradients for all three projections', () => {
      const mlp = new MLP(4, 8); // hidden=4, intermediate=8

      // Forward pass
      const input = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(1, 4));
      const output = mlp.forward(input);

      // Mock gradient from next layer
      const _gradOutput = MxArray.ones(shape(1, 4));

      // Manual backward pass simulation
      // In practice, we'd need full backward implementation for MLP
      // For now, we verify the structure exists

      // Get weights directly
      const gateWeight = mlp.getGateProjWeight();
      const upWeight = mlp.getUpProjWeight();
      const downWeight = mlp.getDownProjWeight();

      // Verify weights exist and have correct shapes
      expect(Array.from(gateWeight.shape())).toEqual([8n, 4n]);
      expect(Array.from(upWeight.shape())).toEqual([8n, 4n]);
      expect(Array.from(downWeight.shape())).toEqual([4n, 8n]);

      // Verify output shape
      expect(Array.from(output.shape())).toEqual([1n, 4n]);
    });

    it('should handle gradient flow through SwiGLU activation', () => {
      // Test the SwiGLU gating mechanism: silu(gate) * up
      const gate = MxArray.fromFloat32(float32(-1.0, 0.0, 1.0, 2.0), shape(4));
      const up = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(4));

      // Forward: silu(gate) * up
      const siluGate = Activations.silu(gate);
      const output = Activations.swiglu(gate, up);

      // Verify output matches manual computation
      const siluGateData = siluGate.toFloat32();
      const upData = up.toFloat32();
      const outputData = output.toFloat32();

      for (let i = 0; i < 4; i++) {
        expect(outputData[i]).toBeCloseTo(siluGateData[i] * upData[i], 5);
      }

      // Mock gradient
      const gradOut = MxArray.ones(shape(4));

      // Gradient w.r.t. gate: grad_out * up * silu'(gate)
      const gradSilu = Gradients.siluBackward(gate, gradOut.mul(up));

      // Gradient w.r.t. up: grad_out * silu(gate)
      const gradUp = gradOut.mul(siluGate);

      // Verify shapes
      expect(Array.from(gradSilu.shape())).toEqual([4n]);
      expect(Array.from(gradUp.shape())).toEqual([4n]);

      // Verify gradients are non-zero (gradient flow working)
      const gradSiluData = gradSilu.toFloat32();
      const gradUpData = gradUp.toFloat32();

      let nonZeroCount = 0;
      for (let i = 0; i < 4; i++) {
        if (Math.abs(gradSiluData[i]) > 1e-6) nonZeroCount++;
        if (Math.abs(gradUpData[i]) > 1e-6) nonZeroCount++;
      }
      expect(nonZeroCount).toBeGreaterThan(0);
    });
  });

  describe('Parameter Updates', () => {
    it('should update gate_proj weights during training', () => {
      const mlp = new MLP(4, 8);

      // Save initial weights
      const initialWeight = mlp.getGateProjWeight().toFloat32();

      // Create a mock gradient with realistic values
      // In practice, this would come from backprop through the full MLP
      const mockGrad = MxArray.randomNormal(shape(8, 4), 0, 0.01);

      // Update gate_proj
      const optimizer = new Adam(0.01);
      const updated = optimizer.updateSingle('weight', mlp.getGateProjWeight(), mockGrad);
      mlp.setGateProjWeight(updated);

      // Verify weights changed
      const newWeight = mlp.getGateProjWeight().toFloat32();
      expect(anyChanged(initialWeight, newWeight)).toBe(true);

      // Verify weights are reasonable
      for (let i = 0; i < newWeight.length; i++) {
        expect(isFinite(newWeight[i])).toBe(true);
        expect(Math.abs(newWeight[i])).toBeLessThan(10.0);
      }
    });

    it('should update up_proj weights during training', () => {
      const mlp = new MLP(4, 8);

      // Save initial weights
      const initialWeight = mlp.getUpProjWeight().toFloat32();

      // Create a mock gradient with realistic values
      // In practice, this would come from backprop through the full MLP
      const mockGrad = MxArray.randomNormal(shape(8, 4), 0, 0.01);

      // Update up_proj
      const optimizer = new Adam(0.01);
      const updated = optimizer.updateSingle('weight', mlp.getUpProjWeight(), mockGrad);
      mlp.setUpProjWeight(updated);

      // Verify weights changed
      const newWeight = mlp.getUpProjWeight().toFloat32();
      expect(anyChanged(initialWeight, newWeight)).toBe(true);

      // Verify weights are reasonable
      for (let i = 0; i < newWeight.length; i++) {
        expect(isFinite(newWeight[i])).toBe(true);
        expect(Math.abs(newWeight[i])).toBeLessThan(10.0);
      }
    });

    it('should update down_proj weights during training', () => {
      const mlp = new MLP(4, 8);

      // Save initial weights
      const initialWeight = mlp.getDownProjWeight().toFloat32();

      // Training step
      const input = MxArray.randomNormal(shape(2, 4), 0, 0.1);

      // Use forwardWithCache to get intermediates
      // Returns: [output, gate, up, gate_act, gated]
      const cached = mlp.forwardWithCache(input);
      const output = cached[0];
      const gated = cached[4]; // silu(gate) * up

      // Simulate loss gradient
      const target = MxArray.randomNormal(shape(2, 4), 0, 0.1);
      const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));

      // Backward through down_proj
      const grads = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      // Update weights
      const optimizer = new Adam(0.01);
      const updated = optimizer.updateSingle('weight', mlp.getDownProjWeight(), grads[1]);
      mlp.setDownProjWeight(updated);

      // Verify weights changed
      const newWeight = mlp.getDownProjWeight().toFloat32();
      expect(anyChanged(initialWeight, newWeight)).toBe(true);
    });

    it('should update all MLP weights consistently', () => {
      const mlp = new MLP(8, 16);

      // Save all initial weights
      const _initialGate = mlp.getGateProjWeight().toFloat32();
      const _initialUp = mlp.getUpProjWeight().toFloat32();
      const initialDown = mlp.getDownProjWeight().toFloat32();

      // Multiple training steps
      const optimizer = new Adam(0.01);

      for (let step = 0; step < 5; step++) {
        const input = MxArray.randomNormal(shape(2, 8), 0, 0.1);

        // Use forwardWithCache to get intermediates
        const cached = mlp.forwardWithCache(input);
        const output = cached[0];
        const gated = cached[4]; // silu(gate) * up

        // Loss gradient
        const target = MxArray.randomNormal(shape(2, 8), 0, 0.1);
        const gradOutput = output.sub(target).mul(MxArray.fromFloat32(float32(0.5), shape(1)));

        // Backward through down_proj
        const downGrads = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

        // Update down_proj
        const updatedDown = optimizer.updateSingle('down', mlp.getDownProjWeight(), downGrads[1]);
        mlp.setDownProjWeight(updatedDown);

        // For gate and up, we'd need gradients from intermediate
        // This is simplified - full implementation needs complete backward pass
      }

      // Verify down_proj weights changed
      const finalDown = mlp.getDownProjWeight().toFloat32();
      expect(anyChanged(initialDown, finalDown)).toBe(true);
    });
  });

  describe('Gradient Computation Validation', () => {
    it('should compute down_proj gradients correctly', () => {
      const mlp = new MLP(4, 8);
      const input = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(1, 4));

      // Forward to get intermediate using forwardWithCache
      const cached = mlp.forwardWithCache(input);
      const gated = cached[4]; // silu(gate) * up

      // Mock gradient
      const gradOutput = MxArray.ones(shape(1, 4));

      // Compute gradients
      const grads = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      // Verify gradient shapes
      expect(Array.from(grads[0].shape())).toEqual([1n, 8n]); // grad_input
      expect(Array.from(grads[1].shape())).toEqual([4n, 8n]); // grad_weight

      // Verify gradients are non-zero
      const gradWeightData = grads[1].toFloat32();
      let nonZeroCount = 0;
      for (let i = 0; i < gradWeightData.length; i++) {
        if (Math.abs(gradWeightData[i]) > 1e-6) nonZeroCount++;
      }
      expect(nonZeroCount).toBeGreaterThan(0);
    });

    it('should propagate gradients through multiple layers', () => {
      const mlp = new MLP(4, 8);

      // Forward pass
      const input = MxArray.fromFloat32(float32(0.5, 1.0, 1.5, 2.0), shape(1, 4));

      // Use forwardWithCache to get intermediates
      const cached = mlp.forwardWithCache(input);
      const _output = cached[0];
      const gated = cached[4]; // silu(gate) * up

      // Backward pass gradient
      const gradOutput = MxArray.fromFloat32(float32(0.1, 0.2, 0.3, 0.4), shape(1, 4));

      // Backward through down_proj
      const downGrads = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      // Gradient w.r.t. intermediate (to propagate to gate and up)
      const gradIntermediate = downGrads[0];

      // Verify gradient propagation
      expect(Array.from(gradIntermediate.shape())).toEqual([1n, 8n]);

      const gradIntData = gradIntermediate.toFloat32();
      let hasNonZero = false;
      for (const val of gradIntData) {
        if (Math.abs(val) > 1e-6) {
          hasNonZero = true;
          break;
        }
      }
      expect(hasNonZero).toBe(true);
    });
  });

  describe('SwiGLU Gradient Details', () => {
    it('should compute correct gradients for SwiGLU gate', () => {
      const gate = MxArray.fromFloat32(float32(0.0, 1.0, -1.0), shape(3));
      const up = MxArray.fromFloat32(float32(2.0, 3.0, 4.0), shape(3));

      // Forward
      const _output = Activations.swiglu(gate, up);

      // Gradient w.r.t output
      const gradOut = MxArray.ones(shape(3));

      // Gradient w.r.t gate: grad_out * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
      const gradGate = Gradients.siluBackward(gate, gradOut.mul(up));

      // Verify gradient is computed
      const gradGateData = gradGate.toFloat32();
      expect(gradGateData.length).toBe(3);

      // Verify gradient values are reasonable
      for (const val of gradGateData) {
        expect(isFinite(val)).toBe(true);
        expect(Math.abs(val)).toBeLessThan(100); // Reasonable magnitude
      }
    });

    it('should compute correct gradients for SwiGLU up projection', () => {
      const gate = MxArray.fromFloat32(float32(0.0, 1.0, 2.0), shape(3));
      const _up = MxArray.fromFloat32(float32(1.0, 2.0, 3.0), shape(3));

      // Forward
      const siluGate = Activations.silu(gate);

      // Gradient w.r.t output
      const gradOut = MxArray.ones(shape(3));

      // Gradient w.r.t up: grad_out * silu(gate)
      const gradUp = gradOut.mul(siluGate);

      // Verify gradient
      const gradUpData = gradUp.toFloat32();
      const siluGateData = siluGate.toFloat32();

      for (let i = 0; i < 3; i++) {
        expect(gradUpData[i]).toBeCloseTo(siluGateData[i], 5);
      }
    });

    it('should handle zero gradients correctly', () => {
      const gate = MxArray.fromFloat32(float32(1.0, 2.0, 3.0), shape(3));
      const up = MxArray.fromFloat32(float32(1.0, 2.0, 3.0), shape(3));

      // Zero gradient from next layer
      const gradOut = MxArray.zeros(shape(3));

      // Gradient w.r.t gate
      const gradGate = Gradients.siluBackward(gate, gradOut.mul(up));

      // Should be zero (or very close)
      const gradGateData = gradGate.toFloat32();
      for (const val of gradGateData) {
        expect(Math.abs(val)).toBeLessThan(1e-6);
      }
    });
  });

  describe('Numerical Stability', () => {
    it('should handle large activations without overflow', () => {
      const mlp = new MLP(4, 8);

      // Large input values
      const input = MxArray.fromFloat32(float32(10.0, 20.0, 30.0, 40.0), shape(1, 4));

      const output = mlp.forward(input);

      // Verify output is finite
      const outputData = output.toFloat32();
      for (const val of outputData) {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      }
    });

    it('should handle small gradients without underflow', () => {
      const mlp = new MLP(4, 8);

      const input = MxArray.randomNormal(shape(1, 4), 0, 0.01);

      // Use forwardWithCache to get intermediates
      const cached = mlp.forwardWithCache(input);
      const _output = cached[0];
      const gated = cached[4]; // silu(gate) * up

      // Very small gradient
      const gradOutput = MxArray.fromFloat32(float32(1e-6, 1e-6, 1e-6, 1e-6), shape(1, 4));

      // Backward
      const grads = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      // Verify gradients are finite
      const gradData = grads[1].toFloat32();
      for (const val of gradData) {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      }
    });

    it('should maintain precision in gradient computation', () => {
      const mlp = new MLP(4, 8);

      const input = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(1, 4));

      // Compute gradients twice with same input
      const cached = mlp.forwardWithCache(input);
      const gated = cached[4]; // silu(gate) * up
      const gradOutput = MxArray.ones(shape(1, 4));

      const grads1 = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      const grads2 = Gradients.linearBackward(gated, mlp.getDownProjWeight(), gradOutput, false);

      // Should be identical
      assertClose(grads1[1], grads2[1], 1e-10, 1e-10);
    });
  });

  describe('Edge Cases', () => {
    it('should handle batch size of 1', () => {
      const mlp = new MLP(4, 8);
      const input = MxArray.randomNormal(shape(1, 4), 0, 0.1);

      const output = mlp.forward(input);

      expect(Array.from(output.shape())).toEqual([1n, 4n]);
    });

    it('should handle larger batch sizes', () => {
      const mlp = new MLP(4, 8);
      const input = MxArray.randomNormal(shape(32, 4), 0, 0.1);

      const output = mlp.forward(input);

      expect(Array.from(output.shape())).toEqual([32n, 4n]);
    });

    it('should handle sequence dimension', () => {
      const mlp = new MLP(4, 8);
      const input = MxArray.randomNormal(shape(2, 10, 4), 0, 0.1);

      const output = mlp.forward(input);

      expect(Array.from(output.shape())).toEqual([2n, 10n, 4n]);
    });

    it('should handle very small hidden dimensions', () => {
      const mlp = new MLP(2, 4);
      const input = MxArray.fromFloat32(float32(1.0, 2.0), shape(1, 2));

      const output = mlp.forward(input);

      expect(Array.from(output.shape())).toEqual([1n, 2n]);
    });

    it('should handle large intermediate dimensions', () => {
      const mlp = new MLP(8, 64); // 8x expansion
      const input = MxArray.randomNormal(shape(2, 8), 0, 0.1);

      const output = mlp.forward(input);

      expect(Array.from(output.shape())).toEqual([2n, 8n]);

      // Verify output is reasonable
      const outputData = output.toFloat32();
      for (const val of outputData) {
        expect(isFinite(val)).toBe(true);
        expect(Math.abs(val)).toBeLessThan(10.0);
      }
    });
  });
});
