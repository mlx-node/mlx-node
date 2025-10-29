/**
 * Shared test utilities for MLX tests
 */

import { expect } from 'vitest';

import { MxArray as MxArrayClass, type MxArray } from '@mlx-node/core';

/**
 * Helper function to create MxArray from regular arrays (float32)
 * @param data - Regular JavaScript array of numbers or Float32Array
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @returns MxArray instance
 */
export function createFloat32Array(data: number[] | Float32Array, shape: number[] | BigInt64Array): MxArray {
  const float32Data = data instanceof Float32Array ? data : new Float32Array(data);
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return MxArrayClass.fromFloat32(float32Data, int64Shape);
}

/**
 * Helper function to create MxArray from regular arrays (int32)
 * @param data - Regular JavaScript array of numbers or Int32Array
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @returns MxArray instance
 */
export function createInt32Array(data: number[] | Int32Array, shape: number[] | BigInt64Array): MxArray {
  const int32Data = data instanceof Int32Array ? data : new Int32Array(data);
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return MxArrayClass.fromInt32(int32Data, int64Shape);
}

// Import Tensor class for factory methods
const { Tensor: TensorClass } = require('@mlx-node/core');

/**
 * Helper to create Int32Array from regular array
 */
export function int32(...values: number[]): Int32Array {
  return new Int32Array(values);
}

/**
 * Helper to create BigInt64Array for shapes from regular array
 */
export function shape(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map((d) => BigInt(d)));
}

/**
 * Helper to create Float32Array from regular array
 */
export function float32(...values: number[]): Float32Array {
  return new Float32Array(values);
}

/**
 * Helper to create Float64Array from regular array
 */
export function float64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

/**
 * Helper function to create Tensor from regular arrays (float32)
 * @param data - Regular JavaScript array of numbers
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param requiresGrad - Whether the tensor requires gradients
 * @returns Tensor instance
 */
export function createFloat32Tensor(
  data: number[],
  shape: number[] | BigInt64Array,
  requiresGrad: boolean = false,
): any {
  const float32Data = new Float32Array(data);
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return TensorClass.fromFloat32(float32Data, int64Shape, requiresGrad);
}

/**
 * Helper function to create Tensor from regular arrays (int32)
 * @param data - Regular JavaScript array of numbers
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param requiresGrad - Whether the tensor requires gradients
 * @returns Tensor instance
 */
export function createInt32Tensor(data: number[], shape: number[] | BigInt64Array, requiresGrad: boolean = false): any {
  const int32Data = new Int32Array(data);
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return TensorClass.fromInt32(int32Data, int64Shape, requiresGrad);
}

/**
 * Helper function to create zeros array
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param dtype - Optional data type
 * @returns MxArray instance
 */
export function createZerosArray(shape: number[] | BigInt64Array, dtype?: any): MxArray {
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return MxArrayClass.zeros(int64Shape, dtype);
}

/**
 * Helper function to create ones array
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param dtype - Optional data type
 * @returns MxArray instance
 */
export function createOnesArray(shape: number[] | BigInt64Array, dtype?: any): MxArray {
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return MxArrayClass.ones(int64Shape, dtype);
}

/**
 * Helper function to create full array
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param fillValue - Value to fill with
 * @param dtype - Optional data type
 * @returns MxArray instance
 */
export function createFullArray(shape: number[] | BigInt64Array, fillValue: number | MxArray, dtype?: any): MxArray {
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return MxArrayClass.full(int64Shape, fillValue, dtype);
}

/**
 * Helper function to create zeros tensor
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param dtype - Optional data type
 * @param requiresGrad - Whether the tensor requires gradients
 * @returns Tensor instance
 */
export function createZerosTensor(shape: number[] | BigInt64Array, dtype?: any, requiresGrad: boolean = false): any {
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return TensorClass.zeros(int64Shape, dtype, requiresGrad);
}

/**
 * Helper function to create ones tensor
 * @param shape - Regular JavaScript array of shape dimensions or BigInt64Array
 * @param dtype - Optional data type
 * @param requiresGrad - Whether the tensor requires gradients
 * @returns Tensor instance
 */
export function createOnesTensor(shape: number[] | BigInt64Array, dtype?: any, requiresGrad: boolean = false): any {
  const int64Shape = shape instanceof BigInt64Array ? shape : BigInt64Array.from(shape.map((x) => BigInt(x)));
  return TensorClass.ones(int64Shape, dtype, requiresGrad);
}

/**
 * Check if two arrays are equal within tolerance
 * @param {MxArray} a - First array
 * @param {MxArray} b - Second array
 * @param {number} atol - Absolute tolerance
 * @returns {boolean} True if arrays are equal within tolerance
 */
export function arrayEqual(a: MxArray, b: MxArray, atol: number = 1e-5): boolean {
  const aData = a.toFloat32();
  const bData = b.toFloat32();

  if (aData.length !== bData.length) return false;

  for (let i = 0; i < aData.length; i++) {
    if (Math.abs(aData[i] - bData[i]) > atol) {
      console.log(`Mismatch at index ${i}: ${aData[i]} vs ${bData[i]}, diff=${Math.abs(aData[i] - bData[i])}`);
      return false;
    }
  }
  return true;
}

/**
 * Assert that two arrays are close within tolerance
 * @param {MxArray} actual - Actual array
 * @param {MxArray} expected - Expected array
 * @param {number} atol - Absolute tolerance
 * @param {number} rtol - Relative tolerance
 * @param {string} message - Optional error message
 */
export function assertArrayClose(
  actual: MxArray,
  expected: MxArray,
  atol: number = 1e-5,
  rtol: number = 1e-5,
  message: string = '',
): void {
  const actualData = actual.toFloat32();
  const expectedData = expected.toFloat32();

  // Check shape
  const actualShape = Array.from(actual.shape()).map((x) => Number(x));
  const expectedShape = Array.from(expected.shape()).map((x) => Number(x));
  expect(actualShape, `Shape mismatch ${message}`).toEqual(expectedShape);

  // Check values
  for (let i = 0; i < actualData.length; i++) {
    const diff = Math.abs(actualData[i] - expectedData[i]);
    const tol = atol + rtol * Math.abs(expectedData[i]);
    if (diff > tol) {
      const errorMsg = `${message} - Mismatch at index ${i}: ${actualData[i]} vs ${expectedData[i]}, diff=${diff}, tol=${tol}`;
      console.log(errorMsg);
      console.log('Actual:', actualData);
      console.log('Expected:', expectedData);
      expect(diff, errorMsg).toBeLessThanOrEqual(tol);
    }
  }
}

/**
 * Assert that an array has expected shape
 * @param {MxArray} array - Array to check
 * @param {number[]} expectedShape - Expected shape
 */
export function assertShape(array: MxArray, expectedShape: number[]): void {
  const actualShape = Array.from(array.shape()).map((x) => Number(x));
  expect(actualShape).toEqual(expectedShape);
}

/**
 * Assert that an array sums to approximately a value
 * @param {MxArray} array - Array to sum
 * @param {number} expectedSum - Expected sum
 * @param {number} tol - Tolerance
 */
export function assertSum(array: MxArray, expectedSum: number, tol: number = 1e-5): void {
  const data = array.toFloat32();
  const sum = data.reduce((a, b) => a + b, 0);
  expect(Math.abs(sum - expectedSum)).toBeLessThan(tol);
}

/**
 * Assert that an array has approximately zero mean
 * @param {MxArray} array - Array to check
 * @param {number} tol - Tolerance
 */
export function assertZeroMean(array: MxArray, tol: number = 1e-5): void {
  const data = array.toFloat32();
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  expect(Math.abs(mean)).toBeLessThan(tol);
}

/**
 * Assert that an array has approximately unit variance
 * @param {MxArray} array - Array to check
 * @param {number} tol - Tolerance
 */
export function assertUnitVariance(array: MxArray, tol: number = 1e-4): void {
  const data = array.toFloat32();
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const variance = data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / data.length;
  expect(Math.abs(variance - 1.0)).toBeLessThan(tol);
}

/**
 * Check if all values in array are finite (no NaN or Inf)
 * @param {MxArray} array - Array to check
 */
export function assertFinite(array: MxArray): void {
  const data = array.toFloat32();
  data.forEach((val, i) => {
    expect(isFinite(val), `Non-finite value at index ${i}: ${val}`).toBe(true);
    expect(isNaN(val), `NaN value at index ${i}`).toBe(false);
  });
}

/**
 * Create a one-hot encoded array
 * @param {number[]} indices - Indices to set to 1
 * @param {number} numClasses - Total number of classes
 * @returns {Float32Array} One-hot encoded array
 */
export function oneHot(indices: number[], numClasses: number): Float32Array {
  const result = new Float32Array(indices.length * numClasses);
  indices.forEach((idx, i) => {
    result[i * numClasses + idx] = 1.0;
  });
  return result;
}

/**
 * Numerical gradient check using finite differences
 * @param {Function} fn - Function to compute gradient for
 * @param {MxArray} x - Input array
 * @param {number} epsilon - Small perturbation for finite differences
 * @returns {Float32Array} Numerical gradient
 */
export function numericalGradient(fn: (x: MxArray) => MxArray, x: MxArray, epsilon: number = 1e-4): Float32Array {
  const xData = x.toFloat32();
  const grad = new Float32Array(xData.length);

  for (let i = 0; i < xData.length; i++) {
    // Save original value
    const orig = xData[i];

    // Compute f(x + eps)
    xData[i] = orig + epsilon;
    const fPlus = fn(x).toFloat32()[0];

    // Compute f(x - eps)
    xData[i] = orig - epsilon;
    const fMinus = fn(x).toFloat32()[0];

    // Restore original value
    xData[i] = orig;

    // Compute gradient
    grad[i] = (fPlus - fMinus) / (2 * epsilon);
  }

  return grad;
}
