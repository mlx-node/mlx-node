/**
 * Tests for the profiling infrastructure.
 *
 * Tests the NAPI profiling API (global store, enable/disable, data retrieval)
 * and the DecodeProfiler integration (simulated via the store API).
 */

import { describe, it, expect, beforeEach } from 'vite-plus/test';
import { setProfilingEnabled, isProfilingEnabled, getProfilingData, resetProfilingData } from '@mlx-node/core';

describe('Profiling Infrastructure', () => {
  beforeEach(() => {
    // Clean state for each test
    setProfilingEnabled(false);
    resetProfilingData();
  });

  describe('setProfilingEnabled / isProfilingEnabled', () => {
    it('starts disabled', () => {
      expect(isProfilingEnabled()).toBe(false);
    });

    it('can be enabled and disabled', () => {
      setProfilingEnabled(true);
      expect(isProfilingEnabled()).toBe(true);
      setProfilingEnabled(false);
      expect(isProfilingEnabled()).toBe(false);
    });

    it('toggling multiple times is idempotent', () => {
      setProfilingEnabled(true);
      setProfilingEnabled(true);
      expect(isProfilingEnabled()).toBe(true);
      setProfilingEnabled(false);
      setProfilingEnabled(false);
      expect(isProfilingEnabled()).toBe(false);
    });
  });

  describe('getProfilingData', () => {
    it('returns empty session when no data collected', () => {
      const data = getProfilingData();

      expect(data.generations).toEqual([]);
      expect(data.totalDurationMs).toBeGreaterThanOrEqual(0);
      expect(data.gpuInfo).toBeDefined();
      expect(data.gpuInfo.architectureGen).toBeGreaterThanOrEqual(0);

      // Summary should be zeroed
      expect(data.summary.totalTokens).toBe(0);
      expect(data.summary.totalPromptTokens).toBe(0);
      expect(data.summary.avgTokensPerSecond).toBe(0);
      expect(data.summary.avgTimeToFirstTokenMs).toBe(0);
      expect(data.summary.avgPrefillMs).toBe(0);
    });

    it('returns valid GPU info', () => {
      const data = getProfilingData();
      // architectureGen is a non-negative integer:
      // Apple Silicon: M1=13, M2=14, M3=15, M4=16, M5=17
      // CI/VM without Metal may return a lower value (e.g. 2)
      expect(data.gpuInfo.architectureGen).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(data.gpuInfo.architectureGen)).toBe(true);
    });

    it('totalDurationMs increases over time', async () => {
      resetProfilingData();
      // Small delay
      await new Promise((resolve) => setTimeout(resolve, 10));
      const data = getProfilingData();
      expect(data.totalDurationMs).toBeGreaterThanOrEqual(5);
    });
  });

  describe('resetProfilingData', () => {
    it('clears all data and resets session timer', async () => {
      // Let some time pass
      await new Promise((resolve) => setTimeout(resolve, 10));
      const before = getProfilingData();
      expect(before.totalDurationMs).toBeGreaterThanOrEqual(5);

      resetProfilingData();
      const after = getProfilingData();
      // Timer should be reset, so duration should be very small
      expect(after.totalDurationMs).toBeLessThan(before.totalDurationMs);
      expect(after.generations).toEqual([]);
    });
  });

  describe('ProfilingSession structure', () => {
    it('has correct shape with all required fields', () => {
      const data = getProfilingData();

      // Top-level fields
      expect(typeof data.totalDurationMs).toBe('number');
      expect(Array.isArray(data.generations)).toBe(true);
      expect(typeof data.gpuInfo).toBe('object');
      expect(typeof data.summary).toBe('object');

      // GpuInfo
      expect(typeof data.gpuInfo.architectureGen).toBe('number');

      // Summary
      expect(typeof data.summary.totalTokens).toBe('number');
      expect(typeof data.summary.totalPromptTokens).toBe('number');
      expect(typeof data.summary.avgTokensPerSecond).toBe('number');
      expect(typeof data.summary.avgTimeToFirstTokenMs).toBe('number');
      expect(typeof data.summary.avgPrefillMs).toBe('number');
    });
  });

  describe('JSON serialization', () => {
    it('serializes to valid JSON', () => {
      const data = getProfilingData();
      const json = JSON.stringify(data, null, 2);
      const parsed = JSON.parse(json);

      expect(parsed.gpuInfo).toBeDefined();
      expect(parsed.generations).toBeDefined();
      expect(parsed.summary).toBeDefined();
      expect(parsed.totalDurationMs).toBeDefined();
    });

    it('uses camelCase keys from NAPI', () => {
      const data = getProfilingData();
      const json = JSON.stringify(data);

      // Verify NAPI auto-converts snake_case to camelCase
      expect(json).toContain('gpuInfo');
      expect(json).toContain('architectureGen');
      expect(json).toContain('totalDurationMs');
      expect(json).toContain('totalTokens');
      expect(json).toContain('totalPromptTokens');
      expect(json).toContain('avgTokensPerSecond');
      expect(json).toContain('avgTimeToFirstTokenMs');
      expect(json).toContain('avgPrefillMs');

      // Should NOT have snake_case
      expect(json).not.toContain('gpu_info');
      expect(json).not.toContain('architecture_gen');
      expect(json).not.toContain('total_duration_ms');
    });
  });
});
