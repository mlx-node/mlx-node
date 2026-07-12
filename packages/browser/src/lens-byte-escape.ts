// lens-byte-escape.ts — reconstruct faithful byte escapes for byte-level-BPE
// fragment tokens in the /jspace logit / Jacobian lens.
//
// WHY: the lens shows ONE decoded BPE token per (layer, position) cell. Qwen3.5
// is byte-level BPE, so a single vocab token can be a *fragment* of a multi-byte
// UTF-8 character (e.g. the first byte of a 2-byte `é`). The native readout
// decodes each id STANDALONE (`tokenizer.decode_sync(&[id], false)`), whose HF
// `ByteLevel` decoder runs `String::from_utf8_lossy`, so a fragment collapses to
// the Unicode replacement char U+FFFD `�`. The raw byte is destroyed inside the
// WASM; only the token id + the `�` survive into JS.
//
// This module reconstructs the token's RAW bytes from its id using the GPT-2
// byte-level vocab (`model.vocab` piece string → bytes) and re-renders any lossy
// fragment as a faithful hex escape like `‹E5›`, while keeping every byte that
// still forms a VALID UTF-8 sequence. A token whose bytes genuinely encode a real
// U+FFFD (bytes `EF BF BD`, e.g. a `�` literal in the training data) is a VALID
// sequence and is therefore preserved unchanged — only lossy fragments escape.
//
// Pure TS, no DOM, no node deps: imported by the browser worker (mlx-worker.ts,
// live path) AND by node scripts run with `oxnode` (bake.mts, fix-byte-escapes.mts).

/** Bracket glyphs wrapping a hex byte escape. Named + swappable by design so the
 *  render form (`‹E5›`) can change in one place. Both are single-guillemet
 *  characters (U+2039 / U+203A) that `cleanupTokenText` leaves untouched. */
export const OPEN = '‹'; // ‹
export const CLOSE = '›'; // ›

/** The Unicode replacement character produced by `from_utf8_lossy` in the WASM. */
export const REPLACEMENT_CHAR = '�'; // �

/**
 * GPT-2 / HF `ByteLevel` `bytes_to_unicode` table.
 *
 * Returns an array indexed by byte value `0..255` giving the printable codepoint
 * that byte maps to in a byte-level BPE piece string. The printable set
 * `bs = [33..126] ∪ [161..172] ∪ [174..255]` maps to itself; every OTHER byte
 * `b` (in ascending `b`) maps to `256 + n` with `n` incrementing from 0.
 *
 * Anchors (pinned by the unit test): byte `0x00` → U+0100, byte `0x20` → U+0120 (Ġ).
 */
export function bytesToUnicode(): number[] {
  const bs: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b); // 33..126  '!'..'~'
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b); // 161..172 '¡'..'¬'
  for (let b = 0xae; b <= 0xff; b++) bs.push(b); // 174..255 '®'..'ÿ'
  const inSet = new Set(bs);
  const cs: number[] = Array.from({ length: 256 }, () => 0);
  for (const b of bs) cs[b] = b; // printable bytes map to themselves
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!inSet.has(b)) {
      cs[b] = 256 + n;
      n += 1;
    }
  }
  return cs;
}

/** Inverse of {@link bytesToUnicode}: `Map<codepoint, byte>`. Exact and total
 *  over all 256 bytes (the forward map is a bijection onto its image). */
export function buildByteDecoder(): Map<number, number> {
  const cs = bytesToUnicode();
  const decoder = new Map<number, number>();
  for (let b = 0; b < 256; b++) decoder.set(cs[b]!, b);
  return decoder;
}

/**
 * Parse a `tokenizer.json` string and return a memoized `id → bytes` resolver.
 *
 * The resolver inverts `model.vocab` (piece string → id) into `id → piece` and
 * maps each code UNIT of the piece back through the byte decoder to the raw byte.
 * Base vocab ids only: `added_tokens` (literal special tokens, never byte
 * fragments, never yield U+FFFD) are NOT in `model.vocab`, so the resolver
 * returns `null` for them and the caller keeps the original text.
 *
 * The heavy inversion is lazy (built on the first lookup) and memoized, and each
 * id → bytes result is cached, so calling this at model load is cheap and the
 * first lens transform pays the one-time parse.
 */
export function buildIdToBytes(tokenizerJsonText: string): (id: number) => Uint8Array | null {
  const decoder = buildByteDecoder();
  let idToPiece: Map<number, string> | null = null;
  const ensureIdToPiece = (): Map<number, string> => {
    if (idToPiece) return idToPiece;
    const map = new Map<number, string>();
    try {
      const parsed = JSON.parse(tokenizerJsonText) as { model?: { vocab?: Record<string, number> } };
      const vocab = parsed?.model?.vocab;
      if (vocab) {
        for (const piece in vocab) map.set(vocab[piece]!, piece);
      }
    } catch {
      // Malformed tokenizer.json → empty map → every id resolves to null → the
      // caller leaves texts untouched. Never throw from a display transform.
    }
    idToPiece = map;
    return map;
  };

  const cache = new Map<number, Uint8Array | null>();
  return (id: number): Uint8Array | null => {
    const cached = cache.get(id);
    if (cached !== undefined || cache.has(id)) return cached ?? null;
    const piece = ensureIdToPiece().get(id);
    let result: Uint8Array | null = null;
    if (piece !== undefined) {
      const bytes = new Uint8Array(piece.length);
      let ok = true;
      for (let i = 0; i < piece.length; i++) {
        const byte = decoder.get(piece.charCodeAt(i));
        if (byte === undefined) {
          ok = false; // piece contains a char outside the byte-level image
          break;
        }
        bytes[i] = byte;
      }
      result = ok ? bytes : null;
    }
    cache.set(id, result);
    return result;
  };
}

/** Decode the single UTF-8 sequence starting at `bytes[i]`, validating length,
 *  continuation bytes, overlong forms, surrogates and the U+10FFFF ceiling.
 *  Returns the decoded char + its byte length, or `null` if `bytes[i]` is not a
 *  valid, complete lead of a well-formed sequence. */
function decodeUtf8At(bytes: Uint8Array, i: number): { char: string; len: number } | null {
  const b0 = bytes[i]!;
  // 1-byte ASCII.
  if (b0 < 0x80) return { char: String.fromCodePoint(b0), len: 1 };
  // 2-byte (reject 0xC0/0xC1 overlong leads by starting at 0xC2).
  if (b0 >= 0xc2 && b0 <= 0xdf) {
    const b1 = bytes[i + 1];
    if (b1 === undefined || (b1 & 0xc0) !== 0x80) return null;
    const cp = ((b0 & 0x1f) << 6) | (b1 & 0x3f);
    return { char: String.fromCodePoint(cp), len: 2 };
  }
  // 3-byte.
  if (b0 >= 0xe0 && b0 <= 0xef) {
    const b1 = bytes[i + 1];
    const b2 = bytes[i + 2];
    if (b1 === undefined || b2 === undefined) return null;
    if ((b1 & 0xc0) !== 0x80 || (b2 & 0xc0) !== 0x80) return null;
    const cp = ((b0 & 0x0f) << 12) | ((b1 & 0x3f) << 6) | (b2 & 0x3f);
    if (cp < 0x800) return null; // overlong
    if (cp >= 0xd800 && cp <= 0xdfff) return null; // UTF-16 surrogate
    return { char: String.fromCodePoint(cp), len: 3 };
  }
  // 4-byte.
  if (b0 >= 0xf0 && b0 <= 0xf4) {
    const b1 = bytes[i + 1];
    const b2 = bytes[i + 2];
    const b3 = bytes[i + 3];
    if (b1 === undefined || b2 === undefined || b3 === undefined) return null;
    if ((b1 & 0xc0) !== 0x80 || (b2 & 0xc0) !== 0x80 || (b3 & 0xc0) !== 0x80) return null;
    const cp = ((b0 & 0x07) << 18) | ((b1 & 0x3f) << 12) | ((b2 & 0x3f) << 6) | (b3 & 0x3f);
    if (cp < 0x10000) return null; // overlong
    if (cp > 0x10ffff) return null; // above the Unicode ceiling
    return { char: String.fromCodePoint(cp), len: 4 };
  }
  // 0x80..0xC1 (stray continuation / overlong lead) and 0xF5..0xFF: invalid lead.
  return null;
}

/** Format one raw byte as `‹E5›` (2-digit UPPERCASE hex, wrapped). */
function escapeByte(byte: number): string {
  return OPEN + byte.toString(16).toUpperCase().padStart(2, '0') + CLOSE;
}

/**
 * Rewrite a decoded lens token so any lossy byte fragment renders as a faithful
 * hex escape instead of `�`.
 *
 * - Fast path: text without U+FFFD is returned unchanged (the common case).
 * - If the id has no reconstructable bytes (e.g. an added/special token, or the
 *   resolver is unavailable), the original text is returned unchanged.
 * - Otherwise the raw bytes are walked left-to-right, greedily emitting the
 *   longest VALID UTF-8 sequence at each position (so genuine multi-byte chars —
 *   including a real U+FFFD encoded as `EF BF BD` — survive) and emitting `‹XX›`
 *   for each byte that cannot start a well-formed sequence.
 *
 * Example: bytes `[0xE5]` → `‹E5›`; `[0xE5, 0x73]` → `‹E5›s`.
 */
export function escapeByteFragments(
  text: string,
  id: number,
  idToBytes: (id: number) => Uint8Array | null,
): string {
  if (!text.includes(REPLACEMENT_CHAR)) return text; // fast path
  const bytes = idToBytes(id);
  if (!bytes || bytes.length === 0) return text;
  let out = '';
  let i = 0;
  while (i < bytes.length) {
    const decoded = decodeUtf8At(bytes, i);
    if (decoded) {
      out += decoded.char;
      i += decoded.len;
    } else {
      out += escapeByte(bytes[i]!);
      i += 1;
    }
  }
  return out;
}
