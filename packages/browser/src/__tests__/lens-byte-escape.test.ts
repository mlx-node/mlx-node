// Unit tests for lens-byte-escape.ts — the byte-level-BPE fragment escaper used
// by the /jspace lens (worker live path + the baked-data migration).
//
// This is a Vitest *browser* suite (Playwright/Chromium, no node:fs), so the
// "real tokenizer" case builds `idToBytes` from a tiny synthetic `model.vocab`
// whose piece strings were extracted VERBATIM from the shipped Qwen3.5 tokenizer
// (demo/public/model/tokenizer.json) and verified with `oxnode`:
//   id 133    piece "É"    → bytes [0xC9]        (lone 2-byte lead → fragment)
//   id 169822 piece "á½"   → bytes [0xE1, 0xBD]  (truncated 3-byte char → fragment)
//   id 49247  piece "ï¿½s" → bytes [0xEF,0xBF,0xBD,0x73] (a GENUINE U+FFFD + 's')
// The end-to-end run over the FULL 12.8MB tokenizer.json lives in the migration
// script (scripts/jlens/fix-byte-escapes.mts), which runs under oxnode/node.

import { describe, expect, it } from 'vite-plus/test';

import {
  buildByteDecoder,
  buildIdToBytes,
  bytesToUnicode,
  escapeByteFragments,
} from '../lens-byte-escape';

describe('bytesToUnicode / byte decoder', () => {
  it('round-trips all 256 bytes through the inverse decoder', () => {
    const cs = bytesToUnicode();
    const decoder = buildByteDecoder();
    expect(cs).toHaveLength(256);
    // The forward map is a bijection: every codepoint is distinct and inverts.
    expect(new Set(cs).size).toBe(256);
    for (let b = 0; b < 256; b++) {
      expect(decoder.get(cs[b]!)).toBe(b);
    }
    expect(decoder.size).toBe(256);
  });

  it('pins the GPT-2 table anchors: U+0100→0x00 and U+0120→0x20', () => {
    const cs = bytesToUnicode();
    // byte → codepoint
    expect(cs[0x00]).toBe(0x100); // U+0100
    expect(cs[0x20]).toBe(0x120); // U+0120 (Ġ)
    // codepoint → byte (inverse)
    const decoder = buildByteDecoder();
    expect(decoder.get(0x100)).toBe(0x00);
    expect(decoder.get(0x120)).toBe(0x20);
  });
});

describe('escapeByteFragments (stubbed idToBytes)', () => {
  // Deterministic stub: the escaper never inspects the id beyond passing it here.
  const stub = (bytesById: Record<number, number[]>) => (id: number) =>
    id in bytesById ? Uint8Array.from(bytesById[id]!) : null;

  it('escapes a lone fragment byte: "�" + [0xE5] → "‹E5›"', () => {
    const idToBytes = stub({ 1: [0xe5] });
    expect(escapeByteFragments('�', 1, idToBytes)).toBe('‹E5›');
  });

  it('keeps a valid trailing byte: "�s" + [0xE5,0x73] → "‹E5›s"', () => {
    const idToBytes = stub({ 2: [0xe5, 0x73] });
    expect(escapeByteFragments('�s', 2, idToBytes)).toBe('‹E5›s');
  });

  it('leaves a normal token (no U+FFFD) unchanged without consulting bytes', () => {
    let called = false;
    const idToBytes = (_id: number) => {
      called = true;
      return null;
    };
    expect(escapeByteFragments('hello', 42, idToBytes)).toBe('hello');
    expect(called).toBe(false); // fast path never resolves bytes
  });

  it('leaves text unchanged when the id has no reconstructable bytes', () => {
    const idToBytes = stub({}); // resolver returns null (e.g. an added token)
    expect(escapeByteFragments('�', 99, idToBytes)).toBe('�');
  });

  it('escapes each byte of a truncated multi-byte fragment: [0xE1,0xBD] → "‹E1›‹BD›"', () => {
    const idToBytes = stub({ 3: [0xe1, 0xbd] });
    expect(escapeByteFragments('�', 3, idToBytes)).toBe('‹E1›‹BD›');
  });
});

describe('escapeByteFragments (real Qwen3.5 vocab piece strings)', () => {
  // A synthetic model.vocab using the EXACT piece strings from the shipped
  // tokenizer (see header). This exercises the real buildIdToBytes path: parse
  // model.vocab → invert to id→piece → map code units through the byte decoder.
  const tokenizerJson = JSON.stringify({
    model: {
      type: 'BPE',
      vocab: {
        É: 133, // 0xC9
        'á½': 169822, // 0xE1 0xBD
        'ï¿½s': 49247, // 0xEF 0xBF 0xBD 0x73 (genuine U+FFFD + 's')
      },
    },
  });
  const idToBytes = buildIdToBytes(tokenizerJson);

  it('escapes a real lossy fragment id 133 "�" → "‹C9›"', () => {
    const escaped = escapeByteFragments('�', 133, idToBytes);
    expect(escaped).toBe('‹C9›');
    expect(escaped).not.toContain('�');
    expect(escaped).toMatch(/‹[0-9A-F]{2}›/);
  });

  it('escapes a real truncated 3-byte fragment id 169822 "�" → "‹E1›‹BD›"', () => {
    const escaped = escapeByteFragments('�', 169822, idToBytes);
    expect(escaped).toBe('‹E1›‹BD›');
    expect(escaped).not.toContain('�');
  });

  it('PRESERVES a genuine U+FFFD token id 49247 "�s" (bytes EF BF BD 73 are valid UTF-8)', () => {
    // This token really encodes a replacement-character glyph; escaping it would
    // corrupt faithful data. It must pass through unchanged.
    expect(escapeByteFragments('�s', 49247, idToBytes)).toBe('�s');
  });

  it('returns null bytes for an out-of-vocab / added-token id', () => {
    expect(idToBytes(999999)).toBeNull();
    expect(escapeByteFragments('�', 999999, idToBytes)).toBe('�');
  });
});
