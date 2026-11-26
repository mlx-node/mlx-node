import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  __getResolvedNativeBase64DecoderForTest,
  __setNativeBase64DecoderForTest,
  ALLOWED_IMAGE_MEDIA_TYPES,
  decodeBase64Strict,
  isAllowedImageMediaType,
  MAX_REQUEST_IMAGES,
} from '../../packages/server/src/base64.js';
import { mapAnthropicRequest } from '../../packages/server/src/mappers/anthropic-request.js';
import { mapRequest } from '../../packages/server/src/mappers/request.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const VALID_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
const EXPECTED_PNG = new Uint8Array(Buffer.from(VALID_PNG, 'base64'));
const EXPECTED_HELLO = new Uint8Array([72, 101, 108, 108, 111]);

const INVALID_BASE64_CASES: readonly [string, string][] = [
  ['empty string', ''],
  ['whitespace only', '   \r\n\t '],
  ['leading form feed', '\fSGVsbG8='],
  ['middle form feed', 'SGVs\fbG8='],
  ['trailing form feed', 'SGVsbG8=\f'],
  ['vertical tab', 'SGVs\vbG8='],
  ['non-breaking space', 'SGVs\u00a0bG8='],
  ['NUL byte', 'SGVs\0bG8='],
  ['invalid characters', '@@@'],
  ['non-ASCII character', 'SGVs\u00e9bG8='],
  ['len % 4 === 1 (single char)', 'A'],
  ['len % 4 === 1 (five chars)', 'AAAAA'],
  ['interior padding A=BC', 'A=BC'],
  ['interior padding AA==AA==', 'AA==AA=='],
  ['interior padding with whitespace AA== AA==', 'AA== AA=='],
  ['data char after single pad AA=A', 'AA=A'],
  ['single char + single pad A=', 'A='],
  ['single char + double pad A==', 'A=='],
  ['single char + spaced double pad A= =', 'A= ='],
  ['partial padding AA=', 'AA='],
  ['partial padding with trailing space AA= ', 'AA= '],
  ['excess padding AAA==', 'AAA=='],
  ['triple padding A===', 'A==='],
  ['spaced triple padding AA== =', 'AA== ='],
  ['padding only ====', '===='],
  ['excess padding AAAA=', 'AAAA='],
  ['excess padding with trailing space AAAA= ', 'AAAA= '],
  ['excess padding AAAA==', 'AAAA=='],
  ['excess padding AAAA====', 'AAAA===='],
  ['single =', '='],
  ['single = with trailing space', '= '],
  ['double ==', '=='],
  ['double == with trailing space', '== '],
  ['triple ===', '==='],
  ['mixed standard + URL-safe +-', '+-'],
  ['mixed standard + URL-safe /_', '/_'],
  ['mixed standard + URL-safe AB+_', 'AB+_'],
  ['mixed standard + URL-safe /-', '/-'],
  ['mixed standard + URL-safe -+', '-+'],
  // RFC 4648 §3.5: non-canonical non-zero trailing bits are strictly rejected
  // by both `base64_simd`'s non-forgiving engines and the fallback decoder.
  ['non-canonical trailing bits padded AB==', 'AB=='],
  ['non-canonical trailing bits unpadded AB', 'AB'],
  ['non-canonical trailing bits 3-char padded AAB=', 'AAB='],
  ['non-canonical trailing bits 3-char unpadded AAB', 'AAB'],
  ['non-canonical trailing bits standard +/==', '+/=='],
  ['non-canonical trailing bits standard +/', '+/'],
  ['non-canonical trailing bits URL-safe -_==', '-_=='],
  ['non-canonical trailing bits URL-safe -_', '-_'],
];

const EDGE_VALID_CASES: readonly [string, string, number[]][] = [
  ['whitespace between padding chars AA= =', 'AA= =', [0]],
  ['newline between padding chars AA=\\n=', 'AA=\n=', [0]],
  ['whitespace before padding AA ==', 'AA ==', [0]],
  ['trailing whitespace after single pad AAA= ', 'AAA= ', [0, 0]],
  ['URL-safe whitespace between padding -w= =', '-w= =', [0xfb]],
];

const MULTI_BLOCK_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 32, 64, 127, 128, 129, 256, 513] as const;

function withFallbackDecoder<T>(fn: () => T): T {
  __setNativeBase64DecoderForTest(null);
  try {
    return fn();
  } finally {
    __setNativeBase64DecoderForTest(undefined);
  }
}

function assertPlainTightUint8Array(result: Uint8Array, expectedLength?: number): void {
  expect(result instanceof Uint8Array).toBe(true);
  expect(Buffer.isBuffer(result)).toBe(false);
  expect(Object.getPrototypeOf(result)).toBe(Uint8Array.prototype);
  expect(result.byteOffset).toBe(0);
  expect(result.buffer.byteLength).toBe(result.byteLength);
  if (expectedLength !== undefined) {
    expect(result.byteLength).toBe(expectedLength);
  }
}

function defineDecodeSuite(label: string, useFallback: boolean): void {
  describe(`decodeBase64Strict (${label})`, () => {
    beforeEach(() => {
      __setNativeBase64DecoderForTest(useFallback ? null : undefined);
    });

    afterEach(() => {
      __setNativeBase64DecoderForTest(undefined);
    });

    it('exercises the expected native or fallback path', () => {
      const resolved = __getResolvedNativeBase64DecoderForTest();
      if (useFallback) {
        expect(resolved).toBeNull();
      } else {
        expect(typeof resolved).toBe('function');
      }
    });

    it('decodes standard padded and unpadded base64', () => {
      expect(decodeBase64Strict('SGVsbG8=')).toEqual(EXPECTED_HELLO);
      expect(decodeBase64Strict('SGVsbG8')).toEqual(EXPECTED_HELLO);
      expect(decodeBase64Strict('+w==')).toEqual(new Uint8Array([0xfb]));
      expect(decodeBase64Strict('+w')).toEqual(new Uint8Array([0xfb]));
      expect(decodeBase64Strict('+/A=')).toEqual(new Uint8Array([0xfb, 0xf0]));
      expect(decodeBase64Strict('+/A')).toEqual(new Uint8Array([0xfb, 0xf0]));
      expect(decodeBase64Strict(VALID_PNG)).toEqual(EXPECTED_PNG);
      expect(decodeBase64Strict(VALID_PNG.replace(/=+$/, ''))).toEqual(EXPECTED_PNG);
    });

    it('decodes URL-safe padded and unpadded base64', () => {
      expect(decodeBase64Strict('-w==')).toEqual(new Uint8Array([0xfb]));
      expect(decodeBase64Strict('-w')).toEqual(new Uint8Array([0xfb]));
      expect(decodeBase64Strict('-_A=')).toEqual(new Uint8Array([0xfb, 0xf0]));
      expect(decodeBase64Strict('-_A')).toEqual(new Uint8Array([0xfb, 0xf0]));

      const urlSafePadded = VALID_PNG.replace(/\+/g, '-').replace(/\//g, '_');
      const urlSafeUnpadded = urlSafePadded.replace(/=+$/, '');
      expect(decodeBase64Strict(urlSafePadded)).toEqual(EXPECTED_PNG);
      expect(decodeBase64Strict(urlSafeUnpadded)).toEqual(EXPECTED_PNG);
    });

    it('ignores ASCII whitespace wrapping (space, tab, LF, CR)', () => {
      expect(decodeBase64Strict(' \t\n\r SGVs\r\n\tbG8= \n')).toEqual(EXPECTED_HELLO);
      expect(decodeBase64Strict('  SGVs bG8\t\r\n')).toEqual(EXPECTED_HELLO);
      expect(decodeBase64Strict('\t\r\n -w== \n ')).toEqual(new Uint8Array([0xfb]));
      expect(decodeBase64Strict('  -w\r\n\t')).toEqual(new Uint8Array([0xfb]));
    });

    it.each(EDGE_VALID_CASES)('handles edge case: %s (%j)', (_caseLabel, input, expected) => {
      const result = decodeBase64Strict(input);
      expect(result).toEqual(new Uint8Array(expected));
      assertPlainTightUint8Array(result, expected.length);
    });

    it('returns a plain Uint8Array with a tight ArrayBuffer rather than a Node Buffer', () => {
      for (const sample of ['SGVsbG8=', 'SGVsbG8', '  SGVs\r\nbG8=  ', '-w==', '-w', VALID_PNG]) {
        assertPlainTightUint8Array(decodeBase64Strict(sample));
      }
    });

    it('decodes multi-block and large SIMD-lane payloads in all variants', () => {
      for (const len of MULTI_BLOCK_LENGTHS) {
        // Include bytes >= 0xf8 so standard (`+`/`/`) and URL-safe (`-`/`_`) characters appear.
        const raw = Uint8Array.from({ length: len }, (_, i) => (0xfb - i * 17) & 0xff);
        const stdPadded = Buffer.from(raw).toString('base64');
        const stdUnpadded = stdPadded.replace(/=+$/, '');
        const urlPadded = stdPadded.replace(/\+/g, '-').replace(/\//g, '_');
        const urlUnpadded = stdUnpadded.replace(/\+/g, '-').replace(/\//g, '_');
        const mid = Math.floor(stdPadded.length / 2);
        const wrapped = ` \t${stdPadded.slice(0, mid)}\r\n${stdPadded.slice(mid)} \n`;

        for (const encoded of [stdPadded, stdUnpadded, urlPadded, urlUnpadded, wrapped]) {
          const decoded = decodeBase64Strict(encoded);
          expect(decoded).toEqual(raw);
          assertPlainTightUint8Array(decoded, raw.byteLength);
        }
      }
    });

    it('propagates a custom errorMessage as an Error on failure', () => {
      for (const [input, msg] of [
        ['@@@', 'Custom decode error'],
        ['', 'Empty payload'],
        ['SGVs\fbG8=', 'Form feed rejected'],
        ['AB==', 'Non-canonical trailing bits rejected'],
        [null as unknown as string, 'Non-string rejected'],
      ] as const) {
        let caught: unknown;
        try {
          decodeBase64Strict(input, msg);
        } catch (err) {
          caught = err;
        }
        expect(caught).toBeInstanceOf(Error);
        expect((caught as Error).message).toBe(msg);
      }
    });

    it.each(INVALID_BASE64_CASES)('rejects invalid input: %s (%j)', (_caseLabel, bad) => {
      expect(() => decodeBase64Strict(bad)).toThrow('Could not process image');
    });

    it.each([null, undefined, 123, {}, []])('rejects non-string runtime input %p', (bad) => {
      expect(() => decodeBase64Strict(bad as unknown as string)).toThrow('Could not process image');
    });

    it('rejects invalid characters and mixed alphabets inside large SIMD-lane payloads', () => {
      const raw = Uint8Array.from({ length: 128 }, (_, i) => (0xfb - i * 17) & 0xff);
      const std = Buffer.from(raw).toString('base64');
      expect(std.includes('+') || std.includes('/')).toBe(true);

      const withInvalidChar = `${std.slice(0, 40)}@${std.slice(41)}`;
      const withFormFeed = `${std.slice(0, 40)}\f${std.slice(41)}`;
      const withMixedUrlChar = `${std.slice(0, 32)}-${std.slice(33)}`;

      expect(() => decodeBase64Strict(withInvalidChar)).toThrow('Could not process image');
      expect(() => decodeBase64Strict(withFormFeed)).toThrow('Could not process image');
      expect(() => decodeBase64Strict(withMixedUrlChar)).toThrow('Could not process image');
    });
  });
}

defineDecodeSuite('native SIMD', false);
defineDecodeSuite('stale-addon fallback', true);

describe('decodeBase64Strict native SIMD vs fallback parity', () => {
  afterEach(() => {
    __setNativeBase64DecoderForTest(undefined);
  });

  it('produces identical bytes and tight plain Uint8Arrays across valid inputs and lengths', () => {
    expect(typeof __getResolvedNativeBase64DecoderForTest()).toBe('function');

    const samples: string[] = [
      VALID_PNG,
      VALID_PNG.replace(/=+$/, ''),
      VALID_PNG.replace(/\+/g, '-').replace(/\//g, '_'),
      VALID_PNG.replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_'),
      ...EDGE_VALID_CASES.map(([, input]) => input),
    ];

    for (const len of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 128, 513]) {
      const raw = Uint8Array.from({ length: len }, (_, i) => (0xff - i * 29) & 0xff);
      const stdPadded = Buffer.from(raw).toString('base64');
      const stdUnpadded = stdPadded.replace(/=+$/, '');
      const urlPadded = stdPadded.replace(/\+/g, '-').replace(/\//g, '_');
      const urlUnpadded = stdUnpadded.replace(/\+/g, '-').replace(/\//g, '_');
      const split = Math.floor(urlPadded.length / 2);
      const wrapped = `  ${urlPadded.slice(0, split)}\r\n\t${urlPadded.slice(split)} `;
      samples.push(stdPadded, stdUnpadded, urlPadded, urlUnpadded, wrapped);
    }

    for (const sample of samples) {
      const nativeResult = decodeBase64Strict(sample);
      const fallbackResult = withFallbackDecoder(() => decodeBase64Strict(sample));
      expect(fallbackResult).toEqual(nativeResult);
      assertPlainTightUint8Array(nativeResult);
      assertPlainTightUint8Array(fallbackResult, nativeResult.byteLength);
    }
  });

  it('rejects all invalid inputs with identical error messages on both paths', () => {
    for (const [, bad] of INVALID_BASE64_CASES) {
      expect(() => decodeBase64Strict(bad)).toThrow('Could not process image');
      expect(() => withFallbackDecoder(() => decodeBase64Strict(bad))).toThrow('Could not process image');
      expect(() => decodeBase64Strict(bad, 'Custom parity error')).toThrow('Custom parity error');
      expect(() => withFallbackDecoder(() => decodeBase64Strict(bad, 'Custom parity error'))).toThrow(
        'Custom parity error',
      );
    }
  });

  it('decodes the same PNG through mapRequest and mapAnthropicRequest with native and fallback paths', () => {
    const responsesReq = {
      model: 'test-model',
      input: [
        {
          type: 'message' as const,
          role: 'user' as const,
          content: [{ type: 'input_image' as const, image_url: `data:image/png;base64,${VALID_PNG}` }],
        },
      ],
    };

    const anthropicReq = {
      model: 'local',
      max_tokens: 64,
      messages: [
        {
          role: 'user' as const,
          content: [
            {
              type: 'image' as const,
              source: { type: 'base64' as const, media_type: 'image/png', data: VALID_PNG },
            },
          ],
        },
      ],
    };

    expect(typeof __getResolvedNativeBase64DecoderForTest()).toBe('function');
    const responsesNative = mapRequest(responsesReq);
    const anthropicNative = mapAnthropicRequest(anthropicReq);

    const { responsesFallback, anthropicFallback } = withFallbackDecoder(() => {
      expect(__getResolvedNativeBase64DecoderForTest()).toBeNull();
      return {
        responsesFallback: mapRequest(responsesReq),
        anthropicFallback: mapAnthropicRequest(anthropicReq),
      };
    });

    expect(responsesNative.messages[0]?.images?.[0]).toEqual(EXPECTED_PNG);
    expect(responsesFallback.messages[0]?.images?.[0]).toEqual(EXPECTED_PNG);
    expect(anthropicNative.messages[0]?.images?.[0]).toEqual(EXPECTED_PNG);
    expect(anthropicFallback.messages[0]?.images?.[0]).toEqual(EXPECTED_PNG);
    assertPlainTightUint8Array(responsesNative.messages[0]!.images![0]!, EXPECTED_PNG.byteLength);
    assertPlainTightUint8Array(responsesFallback.messages[0]!.images![0]!, EXPECTED_PNG.byteLength);
    assertPlainTightUint8Array(anthropicNative.messages[0]!.images![0]!, EXPECTED_PNG.byteLength);
    assertPlainTightUint8Array(anthropicFallback.messages[0]!.images![0]!, EXPECTED_PNG.byteLength);
  });
});

describe('offline / missing-addon compatibility in a fresh Node process', () => {
  const anthropicDist = resolve(ROOT, 'packages/server/dist/mappers/anthropic-request.js');
  const responsesDist = resolve(ROOT, 'packages/server/dist/mappers/request.js');
  const coreEntry = resolve(ROOT, 'packages/core/index.cjs');
  const missingAddonEnv = {
    ...process.env,
    NAPI_RS_NATIVE_LIBRARY_PATH: '/nonexistent/mlx-core.node',
  };

  it('guards the guard: NAPI_RS_NATIVE_LIBRARY_PATH=/nonexistent/mlx-core.node fails @mlx-node/core load', () => {
    const probe = `
      import { createRequire } from 'node:module';
      const req = createRequire(import.meta.url);
      let msg = '';
      try {
        req(${JSON.stringify(coreEntry)});
      } catch (err) {
        msg = err instanceof Error ? err.message : String(err);
      }
      console.log(msg);
    `;
    const out = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
      encoding: 'utf-8',
      env: missingAddonEnv,
    });
    expect(out).toContain('Cannot find native binding');
  });

  it.skipIf(!existsSync(anthropicDist) || !existsSync(responsesDist))(
    'imports mappers, maps offline text requests, and decodes PNGs without a loadable native addon',
    () => {
      const probe = `
        import assert from 'node:assert/strict';
        const { mapAnthropicRequest } = await import(${JSON.stringify(anthropicDist)});
        const { mapRequest } = await import(${JSON.stringify(responsesDist)});

        const mappedBefore = process.report.getReport().sharedObjects.filter((s) => s.includes('mlx-core'));
        assert.deepEqual(mappedBefore, []);

        // Offline text-only replay path (no base64 decode, no addon load).
        const textAnthropic = mapAnthropicRequest({
          model: 'local',
          max_tokens: 64,
          messages: [{ role: 'user', content: 'Hello offline replay' }],
        });
        assert.equal(textAnthropic.messages[0].content, 'Hello offline replay');

        const textResponses = mapRequest({
          model: 'test-model',
          input: 'Hello offline responses',
        });
        assert.equal(textResponses.messages[0].content, 'Hello offline responses');

        // Image path with un-loadable native addon falls back cleanly.
        const png = ${JSON.stringify(VALID_PNG)};
        const expected = ${JSON.stringify(Array.from(EXPECTED_PNG))};

        const imgAnthropic = mapAnthropicRequest({
          model: 'local',
          max_tokens: 64,
          messages: [
            {
              role: 'user',
              content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } }],
            },
          ],
        });
        const imgResponses = mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'user',
              content: [{ type: 'input_image', image_url: 'data:image/png;base64,' + png }],
            },
          ],
        });

        for (const decoded of [imgAnthropic.messages[0].images[0], imgResponses.messages[0].images[0]]) {
          assert.equal(decoded instanceof Uint8Array, true);
          assert.equal(Buffer.isBuffer(decoded), false);
          assert.equal(Object.getPrototypeOf(decoded), Uint8Array.prototype);
          assert.equal(decoded.byteOffset, 0);
          assert.equal(decoded.buffer.byteLength, decoded.byteLength);
          assert.deepEqual(Array.from(decoded), expected);
        }

        const mappedAfter = process.report.getReport().sharedObjects.filter((s) => s.includes('mlx-core'));
        assert.deepEqual(mappedAfter, []);
        console.log('OFFLINE_OK');
      `;
      const out = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
        encoding: 'utf-8',
        env: missingAddonEnv,
      });
      expect(out.trim()).toBe('OFFLINE_OK');
    },
  );
});

describe('isAllowedImageMediaType and image constants', () => {
  it('exposes the expected image cap and MIME allowlist', () => {
    expect(MAX_REQUEST_IMAGES).toBe(64);
    expect(ALLOWED_IMAGE_MEDIA_TYPES).toEqual(new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp']));
  });

  it.each([
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
    'IMAGE/PNG',
    '  Image/Jpeg  ',
    '\timage/WEBP\n',
    '  IMAGE/GIF ',
  ])('accepts allowed media type %j', (mediaType) => {
    expect(isAllowedImageMediaType(mediaType)).toBe(true);
  });

  it.each([
    'image/heic',
    'image/bmp',
    'image/tiff',
    'image/svg+xml',
    'application/pdf',
    'text/plain',
    '',
    'image/png; charset=utf-8',
  ])('rejects disallowed media type %j', (mediaType) => {
    expect(isAllowedImageMediaType(mediaType)).toBe(false);
  });
});
