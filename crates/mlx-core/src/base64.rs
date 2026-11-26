//! Strict SIMD-accelerated base64 decoding exported to NAPI (`decodeBase64Strict`).
//!
//! Accepts standard (`+`/`/`) and URL-safe (`-`/`_`) base64, padded or unpadded,
//! with optional ASCII whitespace wrapping (`' '`, `'\t'`, `'\n'`, `'\r'`).
//!
//! Strictness contract:
//! - Rejects empty or whitespace-only input.
//! - Rejects form feed (`\x0c`), vertical tab (`\x0b`), NUL (`\0`), non-ASCII,
//!   and any non-alphabet bytes (only `' '`, `'\t'`, `'\n'`, `'\r'` are stripped).
//! - Rejects mixed standard (`+`/`/`) and URL-safe (`-`/`_`) alphabets.
//! - Rejects malformed, partial, interior, or excess `=` padding and `len % 4 == 1`.
//! - Rejects non-canonical non-zero trailing bits per RFC 4648 §3.5 (e.g.
//!   `"AB=="`, `"AB"`, `"AAB="`, `"AAB"`, `"+/=="`, `"-_=="`), enforced natively
//!   by `base64_simd`'s non-forgiving engines (`STANDARD`, `STANDARD_NO_PAD`,
//!   `URL_SAFE`, `URL_SAFE_NO_PAD`).

use napi::bindgen_prelude::{Error, Result, Status, Uint8Array};
use napi_derive::napi;

const DEFAULT_ERROR_MESSAGE: &str = "Could not process image";

/// Error returned when strict base64 decoding fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Base64DecodeError;

impl std::fmt::Display for Base64DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid base64 input")
    }
}

impl std::error::Error for Base64DecodeError {}

#[inline]
fn has_ignorable_whitespace(bytes: &[u8]) -> bool {
    memchr::memchr3(b' ', b'\n', b'\r', bytes).is_some() || memchr::memchr(b'\t', bytes).is_some()
}

#[inline]
fn is_ignorable_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn decode_clean_slice(clean: &[u8]) -> std::result::Result<Vec<u8>, Base64DecodeError> {
    if clean.is_empty() {
        return Err(Base64DecodeError);
    }

    // If standard (`+`/`/`) and URL-safe (`-`/`_`) alphabets are mixed, selecting
    // the URL-safe engine makes `base64_simd` reject `+` and `/` natively.
    let url_safe = memchr::memchr2(b'-', b'_', clean).is_some();
    let padded = clean.ends_with(b"=");
    let engine = match (url_safe, padded) {
        (false, true) => base64_simd::STANDARD,
        (false, false) => base64_simd::STANDARD_NO_PAD,
        (true, true) => base64_simd::URL_SAFE,
        (true, false) => base64_simd::URL_SAFE_NO_PAD,
    };

    let decoded = engine.decode_to_vec(clean).map_err(|_| Base64DecodeError)?;
    if decoded.is_empty() {
        return Err(Base64DecodeError);
    }
    Ok(decoded)
}

/// Strictly decode a base64 string into raw bytes using SIMD acceleration.
///
/// Accepts ASCII whitespace `' '`/`'\t'`/`'\n'`/`'\r'`, standard or URL-safe
/// alphabets (never mixed), padded or unpadded, and enforces canonical zero
/// trailing bits per RFC 4648 §3.5. Rejects empty input, form feed (`\x0c`),
/// vertical tab (`\x0b`), malformed padding, and invalid characters.
pub fn decode_base64_strict_bytes(
    encoded: &str,
) -> std::result::Result<Vec<u8>, Base64DecodeError> {
    let bytes = encoded.as_bytes();
    if has_ignorable_whitespace(bytes) {
        let mut cleaned = Vec::with_capacity(bytes.len());
        cleaned.extend(
            bytes
                .iter()
                .copied()
                .filter(|&b| !is_ignorable_whitespace(b)),
        );
        decode_clean_slice(&cleaned)
    } else {
        decode_clean_slice(bytes)
    }
}

/// Strictly decode base64 using SIMD acceleration into a plain, tightly sized
/// `Uint8Array` (exported to JS as `decodeBase64Strict`).
///
/// Accepts ASCII whitespace (space, tab, LF, CR), unpadded or properly padded
/// standard base64, and URL-safe base64. Rejects empty input, form feed (0x0c),
/// vertical tab (0x0b), mixed alphabets, malformed padding, invalid characters,
/// and non-canonical trailing bits per RFC 4648 §3.5.
#[napi]
pub fn decode_base64_strict(encoded: String, error_message: Option<String>) -> Result<Uint8Array> {
    let mut buf = encoded.into_bytes();
    if has_ignorable_whitespace(&buf) {
        buf.retain(|&b| !is_ignorable_whitespace(b));
    }
    match decode_clean_slice(&buf) {
        Ok(decoded) => Ok(Uint8Array::new(decoded)),
        Err(Base64DecodeError) => Err(Error::new(
            Status::GenericFailure,
            error_message.as_deref().unwrap_or(DEFAULT_ERROR_MESSAGE),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    fn make_payload(len: usize) -> Vec<u8> {
        // Starts at 0xfb with step -17 so standard (`+`/`/`) and URL-safe (`-`/`_`)
        // characters appear even on short payloads.
        (0..len)
            .map(|i| 0xfbu8.wrapping_sub((i as u8).wrapping_mul(17)))
            .collect()
    }

    #[test]
    fn decodes_standard_padded_and_unpadded() {
        assert_eq!(decode_base64_strict_bytes("SGVsbG8=").unwrap(), b"Hello");
        assert_eq!(decode_base64_strict_bytes("SGVsbG8").unwrap(), b"Hello");
        assert_eq!(decode_base64_strict_bytes("+w==").unwrap(), vec![0xfb]);
        assert_eq!(decode_base64_strict_bytes("+w").unwrap(), vec![0xfb]);
        assert_eq!(
            decode_base64_strict_bytes("+/A=").unwrap(),
            vec![0xfb, 0xf0]
        );
        assert_eq!(decode_base64_strict_bytes("+/A").unwrap(), vec![0xfb, 0xf0]);
        assert_eq!(decode_base64_strict_bytes("AA==").unwrap(), vec![0]);
        assert_eq!(decode_base64_strict_bytes("AA").unwrap(), vec![0]);
        assert_eq!(decode_base64_strict_bytes("AAA=").unwrap(), vec![0, 0]);
        assert_eq!(decode_base64_strict_bytes("AAA").unwrap(), vec![0, 0]);
        assert_eq!(decode_base64_strict_bytes("AAAA").unwrap(), vec![0, 0, 0]);

        let expected_png = base64_simd::STANDARD.decode_to_vec(VALID_PNG_B64).unwrap();
        assert_eq!(
            decode_base64_strict_bytes(VALID_PNG_B64).unwrap(),
            expected_png
        );
        assert_eq!(
            decode_base64_strict_bytes(VALID_PNG_B64.trim_end_matches('=')).unwrap(),
            expected_png
        );
    }

    #[test]
    fn decodes_url_safe_padded_and_unpadded() {
        assert_eq!(decode_base64_strict_bytes("-w==").unwrap(), vec![0xfb]);
        assert_eq!(decode_base64_strict_bytes("-w").unwrap(), vec![0xfb]);
        assert_eq!(
            decode_base64_strict_bytes("-_A=").unwrap(),
            vec![0xfb, 0xf0]
        );
        assert_eq!(decode_base64_strict_bytes("-_A").unwrap(), vec![0xfb, 0xf0]);

        let expected_png = base64_simd::STANDARD.decode_to_vec(VALID_PNG_B64).unwrap();
        let url_padded = VALID_PNG_B64.replace('+', "-").replace('/', "_");
        let url_unpadded = url_padded.trim_end_matches('=');
        assert_eq!(
            decode_base64_strict_bytes(&url_padded).unwrap(),
            expected_png
        );
        assert_eq!(
            decode_base64_strict_bytes(url_unpadded).unwrap(),
            expected_png
        );
    }

    #[test]
    fn decodes_multi_block_and_simd_lane_payloads() {
        // Cover short lengths 1..=12 plus larger lengths >= 64 that cross
        // base64-simd's 32-byte SIMD loop threshold (n >= 38).
        let lengths: Vec<usize> = (1..=12)
            .chain([
                24, 28, 29, 30, 31, 32, 48, 63, 64, 65, 127, 128, 129, 256, 513,
            ])
            .collect();

        for len in lengths {
            let raw = make_payload(len);
            let std_padded = base64_simd::STANDARD.encode_to_string(&raw);
            let std_unpadded = base64_simd::STANDARD_NO_PAD.encode_to_string(&raw);
            let url_padded = base64_simd::URL_SAFE.encode_to_string(&raw);
            let url_unpadded = base64_simd::URL_SAFE_NO_PAD.encode_to_string(&raw);
            let split = std_padded.len() / 2;
            let wrapped_std = format!(" \t{}\r\n{} \n", &std_padded[..split], &std_padded[split..]);
            let wrapped_url = format!(
                "\r\n{}\t {}\n",
                &url_unpadded[..url_unpadded.len() / 2],
                &url_unpadded[url_unpadded.len() / 2..]
            );

            for encoded in [
                std_padded.as_str(),
                std_unpadded.as_str(),
                url_padded.as_str(),
                url_unpadded.as_str(),
                wrapped_std.as_str(),
                wrapped_url.as_str(),
            ] {
                let decoded = decode_base64_strict_bytes(encoded).unwrap_or_else(|err| {
                    panic!("expected valid decode for len={len}, encoded={encoded:?}: {err}")
                });
                assert_eq!(decoded, raw, "mismatch for len={len}");
                assert_eq!(
                    decoded.len(),
                    decoded.capacity(),
                    "decoded Vec should be tightly sized for len={len}"
                );
            }
        }
    }

    #[test]
    fn ignores_allowed_ascii_whitespace() {
        assert_eq!(
            decode_base64_strict_bytes(" \t\n\r SGVs\r\n\tbG8= \n").unwrap(),
            b"Hello"
        );
        assert_eq!(
            decode_base64_strict_bytes("  SGVs bG8\t\r\n").unwrap(),
            b"Hello"
        );
        assert_eq!(
            decode_base64_strict_bytes("\t\r\n -w== \n ").unwrap(),
            vec![0xfb]
        );
        assert_eq!(
            decode_base64_strict_bytes("  -w\r\n\t").unwrap(),
            vec![0xfb]
        );
        // Whitespace between and around padding characters
        assert_eq!(decode_base64_strict_bytes("AA= =").unwrap(), vec![0]);
        assert_eq!(decode_base64_strict_bytes("AA=\n=").unwrap(), vec![0]);
        assert_eq!(decode_base64_strict_bytes("AA ==").unwrap(), vec![0]);
        assert_eq!(decode_base64_strict_bytes("AAA= ").unwrap(), vec![0, 0]);
        assert_eq!(decode_base64_strict_bytes("-w= =").unwrap(), vec![0xfb]);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let invalid_cases: &[(&str, &str)] = &[
            ("empty string", ""),
            ("whitespace only", "   \r\n\t "),
            ("leading form feed", "\x0cSGVsbG8="),
            ("middle form feed", "SGVs\x0cbG8="),
            ("trailing form feed", "SGVsbG8=\x0c"),
            ("vertical tab", "SGVs\x0bbG8="),
            ("non-breaking space", "SGVs\u{00a0}bG8="),
            ("NUL byte", "SGVs\0bG8="),
            ("invalid characters", "@@@"),
            ("non-ASCII character", "SGVs\u{00e9}bG8="),
            ("len % 4 == 1 (single char)", "A"),
            ("len % 4 == 1 (five chars)", "AAAAA"),
            ("interior padding A=BC", "A=BC"),
            ("interior padding AA==AA==", "AA==AA=="),
            ("interior padding with whitespace AA== AA==", "AA== AA=="),
            ("data char after single pad AA=A", "AA=A"),
            ("single char + single pad A=", "A="),
            ("single char + double pad A==", "A=="),
            ("single char + spaced double pad A= =", "A= ="),
            ("partial padding AA=", "AA="),
            ("partial padding with trailing space AA= ", "AA= "),
            ("excess padding AAA==", "AAA=="),
            ("triple padding A===", "A==="),
            ("spaced triple padding AA== =", "AA== ="),
            ("padding only ====", "===="),
            ("excess padding AAAA=", "AAAA="),
            ("excess padding with trailing space AAAA= ", "AAAA= "),
            ("excess padding AAAA==", "AAAA=="),
            ("excess padding AAAA====", "AAAA===="),
            ("single =", "="),
            ("single = with trailing space", "= "),
            ("double ==", "=="),
            ("double == with trailing space", "== "),
            ("triple ===", "==="),
            ("mixed standard + URL-safe +-", "+-"),
            ("mixed standard + URL-safe /_", "/_"),
            ("mixed standard + URL-safe AB+_", "AB+_"),
            ("mixed standard + URL-safe /-", "/-"),
            ("mixed standard + URL-safe -+", "-+"),
        ];

        for &(label, input) in invalid_cases {
            assert!(
                decode_base64_strict_bytes(input).is_err(),
                "expected Err for {label}: {input:?}"
            );
        }
    }

    #[test]
    fn rejects_non_canonical_trailing_bits() {
        // `base64_simd` non-forgiving engines (`STANDARD`, `STANDARD_NO_PAD`,
        // `URL_SAFE`, `URL_SAFE_NO_PAD`) enforce RFC 4648 §3.5 canonical zero
        // trailing bits:
        // - 2-char tail: (y2 & 0x0f) == 0  (so 'B'=1 and '/'/'_'=63 are rejected)
        // - 3-char tail: (y3 & 0x03) == 0  (so 'B'=1 is rejected)
        let non_canonical = [
            "AB==", "AB", "AAB=", "AAB", "+/==", "+/", "-_==", "-_", "-_= =",
        ];

        for input in non_canonical {
            assert!(
                decode_base64_strict_bytes(input).is_err(),
                "expected non-canonical trailing bits to be rejected: {input:?}"
            );
        }
    }

    #[test]
    fn rejects_invalid_and_mixed_chars_inside_simd_blocks() {
        let raw = make_payload(96);
        let mut std_encoded = base64_simd::STANDARD.encode_to_string(&raw);
        // Inject an invalid byte in the middle of the SIMD vector region (offset 40).
        let mut bad_bytes = std_encoded.clone().into_bytes();
        bad_bytes[40] = b'@';
        let bad_str = String::from_utf8(bad_bytes).unwrap();
        assert!(decode_base64_strict_bytes(&bad_str).is_err());

        // Inject form feed inside a SIMD-sized payload.
        let mut ff_bytes = std_encoded.clone().into_bytes();
        ff_bytes[40] = b'\x0c';
        let ff_str = String::from_utf8(ff_bytes).unwrap();
        assert!(decode_base64_strict_bytes(&ff_str).is_err());

        // Mix URL-safe '-' into a long standard payload that contains '+'/'/'.
        assert!(std_encoded.contains('+') || std_encoded.contains('/'));
        std_encoded.replace_range(32..33, "-");
        assert!(decode_base64_strict_bytes(&std_encoded).is_err());
    }
}
