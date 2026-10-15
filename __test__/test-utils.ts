/**
 * Shared test utilities for MLX tests
 */

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
