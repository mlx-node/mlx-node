/**
 * Pins for the CONVERT_DETECT table.
 *
 * 1. Row ORDER: a raw model_type row that precedes an architecture-probe row
 *    must win even when the probe also matches.
 * 2. Cross-language PARITY: every convertible model_type the native recipe
 *    registry accepts must be producible by some CONVERT_DETECT row — otherwise
 *    a checkpoint of that family with `-m` omitted silently converts through
 *    the generic family-less pass and produces unloadable output.
 */
import { convertibleModelTypes } from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import { CONVERT_DETECT, detectConvertModelType } from '../src/commands/convert-detect.js';

describe('CONVERT_DETECT overlap precedence', () => {
  it("resolves model_type 'gemma4_text' WITH a Gemma4Unified architecture to 'gemma4'", () => {
    // The gemma4|gemma4_text row precedes the unified-architecture row. A
    // loop that probed architectures first (as the runtime loader does)
    // would resolve 'gemma4_unified' and silently reroute the checkpoint:
    // the E2B prequantized-importer gate keys on the exact string 'gemma4'.
    expect(
      detectConvertModelType({
        model_type: 'gemma4_text',
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
      }),
    ).toBe('gemma4');
  });

  it("resolves model_type 'gemma4' WITH a Gemma4Unified architecture to 'gemma4'", () => {
    expect(
      detectConvertModelType({
        model_type: 'gemma4',
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
      }),
    ).toBe('gemma4');
  });

  it('the NemotronH architecture beats any stale recognized model_type', () => {
    // The architecture row is FIRST: it is authoritative over a
    // stale-but-recognized model_type, matching the native parser and the
    // runtime registry probe.
    for (const stale of ['qwen3_5', 'gemma4_unified', 'muse_glimmer', 'paddleocr_vl']) {
      expect(
        detectConvertModelType({
          model_type: stale,
          architectures: ['NemotronHForCausalLM'],
        }),
      ).toBe('nemotron_h');
    }
  });

  it("a recognized model_type beats the Gemma4Unified architecture probe ('qwen3_5' stays 'qwen3_5')", () => {
    expect(
      detectConvertModelType({
        model_type: 'qwen3_5',
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
      }),
    ).toBe('qwen3_5');
  });

  it("an UNRECOGNIZED model_type falls through to the Gemma4Unified architecture probe -> 'gemma4_unified'", () => {
    expect(
      detectConvertModelType({
        model_type: 'gemma_new',
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
      }),
    ).toBe('gemma4_unified');
  });
});

describe('CONVERT_DETECT row semantics', () => {
  it.each([
    ['paddleocr_vl', 'paddleocr-vl'],
    ['internvl_chat', 'qianfan-ocr'],
    ['qianfan-ocr', 'qianfan-ocr'],
    ['qwen3_asr', 'qwen3_asr'],
    ['qwen3_5_moe', 'qwen3_5_moe'],
    ['qwen3_5', 'qwen3_5'],
    ['gemma4', 'gemma4'],
    ['gemma4_text', 'gemma4'],
    ['gemma4_unified', 'gemma4_unified'],
    ['lfm2_moe', 'lfm2_moe'],
    ['lfm2', 'lfm2'],
    ['openai_privacy_filter', 'privacy-filter'],
    ['muse_glimmer', 'muse_glimmer'],
    ['nemotron_h', 'nemotron_h'],
  ])("maps raw model_type '%s' to '%s'", (raw, out) => {
    expect(detectConvertModelType({ model_type: raw })).toBe(out);
  });

  it.each([
    ['Gemma4UnifiedForConditionalGeneration', 'gemma4_unified'],
    ['NemotronHForCausalLM', 'nemotron_h'],
  ])("detects an architecture-only config with '%s' as '%s'", (architecture, out) => {
    expect(detectConvertModelType({ architectures: [architecture] })).toBe(out);
  });

  it('yields undefined for an unmatched model_type (deliberate generic conversion, e.g. qwen3)', () => {
    expect(detectConvertModelType({ model_type: 'qwen3' })).toBeUndefined();
    expect(detectConvertModelType({ model_type: 'qwen3', architectures: ['Qwen3ForCausalLM'] })).toBeUndefined();
    expect(detectConvertModelType({})).toBeUndefined();
    expect(detectConvertModelType({ architectures: 'NemotronHForCausalLM' })).toBeUndefined();
  });
});

describe('CONVERT_DETECT native parity', () => {
  // Native convertible model_types with no auto-detect row. Empty today:
  // every recipe family is detectable from config.json. (pp-lcnet-ori and
  // uvdoc are flag-only, but they are foreign-weight conversions outside the
  // native recipe registry, so they never appear in convertibleModelTypes().)
  const MANUAL_ONLY: readonly string[] = [];

  const producible = new Set(
    CONVERT_DETECT.flatMap((row) => (row.out !== undefined ? [row.out] : [...(row.rawModelTypes ?? [])])),
  );

  it('every native convertible model_type is producible by a detect row', () => {
    const native = convertibleModelTypes();
    expect(native.length).toBeGreaterThan(0);
    const undetectable = native.filter((id) => !producible.has(id) && !MANUAL_ONLY.includes(id));
    expect(undetectable, 'recipe families with no CLI auto-detect row (convert would silently go generic)').toEqual([]);
  });

  it('every detect-row output is a native convertible model_type', () => {
    // The reverse direction: a row emitting an id the native registry does
    // not accept would auto-detect its checkpoints into the hard
    // "Unknown model type" dispatch error.
    const native = new Set(convertibleModelTypes());
    const unknownOutputs = [...producible].filter((id) => !native.has(id));
    expect(unknownOutputs).toEqual([]);
  });
});
