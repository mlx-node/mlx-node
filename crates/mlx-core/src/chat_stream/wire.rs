//! Binary wire format for the SharedArrayBuffer (SAB) ring-buffer streaming
//! path.
//!
//! This module is the **pure format layer**. It knows how to lay out the
//! 32-byte ring header, how to serialise a [`ChatStreamChunk`] into one record,
//! and how to parse a record back. It performs no atomic operations, owns no
//! memory beyond local scratch, and compiles on every target (native, WASM).
//!
//! ## Ring-buffer layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ 32-byte header                                           │
//! │   offset 0  (u32 LE, atomic-intended): seq               │
//! │   offset 4  (u32 LE, atomic-intended): write_cur         │
//! │   offset 8  (u32 LE, atomic-intended): read_cur          │
//! │   offset 12 (u32 LE, atomic-intended): cancelled (0|1)   │
//! │   offset 16..32: reserved, zeroed                        │
//! ├──────────────────────────────────────────────────────────┤
//! │ body (SAB_len − 32 bytes)                                │
//! │   records written sequentially, wrapping at body_len     │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Record layout
//!
//! ```text
//! ┌──────────┬──────────┬────────────┬──────────────────┐
//! │ len (u16)│ kind (u8)│ flags (u8) │ payload (len B)  │
//! └──────────┴──────────┴────────────┴──────────────────┘
//! ```
//!
//! Total record size = `RECORD_HEADER_BYTES + len`.
//!
//! ### Kinds
//! * `KIND_TEXT (0)` — fast-path raw UTF-8 text. Used when `text` is the only
//!   meaningful field in the chunk. The `is_reasoning` flag is encoded in
//!   `FLAG_IS_REASONING` (bit 0) of the flags byte so the fast path is not
//!   lost for reasoning-phase tokens.
//! * `KIND_JSON (1)` — full JSON encoding of [`WireChunk`] (a serde-capable
//!   mirror of [`ChatStreamChunk`]).
//! * `KIND_ERROR (2)` — UTF-8 error message. Decoded as `Err(...)`.
//!
//! ### Wrap-around
//! Records are written sequentially. When a record would extend past
//! `body_len`, it wraps to offset 0. There is no pad-to-end marker; the
//! 4-byte record header carried with the payload tells the reader the record
//! size. Task 4 (the atomic SabSink) handles copying wrapped ranges into a
//! scratch buffer before calling [`read_record`].

use crate::models::qwen3_5::model::ChatStreamChunk;
use crate::tools::ToolCallResult;

// ---------------------------------------------------------------------------
// Ring-header constants
// ---------------------------------------------------------------------------

/// Size of the fixed ring-buffer header in bytes.
pub const SAB_HEADER_BYTES: usize = 32;

/// Byte offset of the `seq` counter (u32 LE) in the header.
pub const SAB_HEADER_OFFSET_SEQ: usize = 0;

/// Byte offset of `write_cur` (u32 LE) in the header.
pub const SAB_HEADER_OFFSET_WRITE_CUR: usize = 4;

/// Byte offset of `read_cur` (u32 LE) in the header.
pub const SAB_HEADER_OFFSET_READ_CUR: usize = 8;

/// Byte offset of `cancelled` (u32 LE, 0 or 1) in the header.
pub const SAB_HEADER_OFFSET_CANCELLED: usize = 12;

// ---------------------------------------------------------------------------
// Record constants
// ---------------------------------------------------------------------------

/// Size of the record header in bytes: `len(u16) | kind(u8) | flags(u8)`.
pub const RECORD_HEADER_BYTES: usize = 4;

/// Record kind: fast-path raw UTF-8 text chunk.
pub const KIND_TEXT: u8 = 0;

/// Record kind: full JSON-encoded [`ChatStreamChunk`] (via [`WireChunk`]).
pub const KIND_JSON: u8 = 1;

/// Record kind: UTF-8 error message; decoded as `Err(napi::Error)`.
pub const KIND_ERROR: u8 = 2;

/// Flags bit 0 — set when `is_reasoning` is `Some(true)` in a `KIND_TEXT`
/// record. Ignored for `KIND_JSON` and `KIND_ERROR`.
pub const FLAG_IS_REASONING: u8 = 0x01;

/// Maximum payload bytes in one record (dictated by the u16 length field).
pub const MAX_PAYLOAD_BYTES: usize = u16::MAX as usize;

// ---------------------------------------------------------------------------
// Local serde mirror types
// ---------------------------------------------------------------------------
// `ChatStreamChunk` is annotated with `#[napi(object)]` and does not derive
// `Serialize` / `Deserialize`. `PerformanceMetrics` is in the same situation.
// We define lightweight local mirrors so we can JSON-encode any chunk without
// touching the canonical type definitions.

/// Serde-capable mirror of [`crate::profiling::PerformanceMetrics`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WirePerf {
    ttft_ms: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
}

impl From<&crate::profiling::PerformanceMetrics> for WirePerf {
    fn from(p: &crate::profiling::PerformanceMetrics) -> Self {
        Self {
            ttft_ms: p.ttft_ms,
            prefill_tokens_per_second: p.prefill_tokens_per_second,
            decode_tokens_per_second: p.decode_tokens_per_second,
        }
    }
}

impl From<WirePerf> for crate::profiling::PerformanceMetrics {
    fn from(w: WirePerf) -> Self {
        Self {
            ttft_ms: w.ttft_ms,
            prefill_tokens_per_second: w.prefill_tokens_per_second,
            decode_tokens_per_second: w.decode_tokens_per_second,
        }
    }
}

/// Serde-capable mirror of [`ChatStreamChunk`] used for `KIND_JSON` records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WireChunk {
    text: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    performance: Option<WirePerf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_reasoning: Option<bool>,
}

impl From<&ChatStreamChunk> for WireChunk {
    fn from(c: &ChatStreamChunk) -> Self {
        Self {
            text: c.text.clone(),
            done: c.done,
            finish_reason: c.finish_reason.clone(),
            tool_calls: c.tool_calls.clone(),
            thinking: c.thinking.clone(),
            num_tokens: c.num_tokens,
            raw_text: c.raw_text.clone(),
            performance: c.performance.as_ref().map(WirePerf::from),
            is_reasoning: c.is_reasoning,
        }
    }
}

impl From<WireChunk> for ChatStreamChunk {
    fn from(w: WireChunk) -> Self {
        Self {
            text: w.text,
            done: w.done,
            finish_reason: w.finish_reason,
            tool_calls: w.tool_calls,
            thinking: w.thinking,
            num_tokens: w.num_tokens,
            raw_text: w.raw_text,
            performance: w.performance.map(Into::into),
            is_reasoning: w.is_reasoning,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoded payload
// ---------------------------------------------------------------------------

/// Classified record payload produced by [`classify_payload`] and consumed by
/// [`write_record`].
#[derive(Debug, Clone)]
pub enum EncodedPayload<'a> {
    /// Fast-path: raw UTF-8 text bytes and the `is_reasoning` flag.
    Text {
        bytes: &'a [u8],
        is_reasoning: bool,
    },
    /// Full JSON encoding of the chunk (allocated by [`classify_payload`]).
    Json(Vec<u8>),
    /// UTF-8 error message.
    Error(String),
}

// ---------------------------------------------------------------------------
// Parsed record
// ---------------------------------------------------------------------------

/// A record parsed from the ring body by [`read_record`].
pub struct ParsedRecord<'a> {
    pub kind: u8,
    pub flags: u8,
    pub payload: &'a [u8],
    /// `RECORD_HEADER_BYTES + payload.len()`
    pub total_len: usize,
}

// ---------------------------------------------------------------------------
// Fast-path predicate
// ---------------------------------------------------------------------------

/// Returns `true` when the chunk carries only `text` plus a concrete
/// `is_reasoning: Some(_)` boolean — the fast path encodes the boolean in
/// one flag bit. `is_reasoning == None` falls through to JSON so the three
/// states (`None`, `Some(false)`, `Some(true)`) round-trip losslessly.
///
/// Note: if `text` is longer than [`MAX_PAYLOAD_BYTES`] the caller falls back
/// to JSON even when this predicate returns true.
fn is_text_fast_path(chunk: &ChatStreamChunk) -> bool {
    !chunk.text.is_empty()
        && !chunk.done
        && chunk.finish_reason.is_none()
        && chunk.tool_calls.is_none()
        && chunk.thinking.is_none()
        && chunk.num_tokens.is_none()
        && chunk.raw_text.is_none()
        && chunk.performance.is_none()
        && chunk.is_reasoning.is_some()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Classify a `napi::Result<ChatStreamChunk>` into the [`EncodedPayload`] that
/// should land in the ring.
///
/// # Fast path
/// When the chunk carries only `text` (and optionally `is_reasoning`) the
/// result is [`EncodedPayload::Text`] — a zero-copy borrow of the chunk's
/// UTF-8 bytes. This is the hot path for every normal decode token.
///
/// # JSON fallback
/// Used when any "structured" field is set (`done`, `finish_reason`,
/// `tool_calls`, `thinking`, `num_tokens`, `raw_text`, `performance`), or
/// when `text` is empty, or when `text.len() > MAX_PAYLOAD_BYTES` (a token
/// that large is impossible in practice but we handle it defensively).
///
/// # Error path
/// `Err(e)` maps to [`EncodedPayload::Error`] carrying the reason string.
pub fn classify_payload<'a>(
    result: &'a napi::Result<ChatStreamChunk>,
) -> napi::Result<EncodedPayload<'a>> {
    match result {
        Err(e) => Ok(EncodedPayload::Error(e.reason.clone())),
        Ok(chunk) => {
            if is_text_fast_path(chunk) && chunk.text.len() <= MAX_PAYLOAD_BYTES {
                // is_text_fast_path guarantees is_reasoning.is_some()
                Ok(EncodedPayload::Text {
                    bytes: chunk.text.as_bytes(),
                    is_reasoning: chunk.is_reasoning.unwrap_or(false),
                })
            } else {
                // JSON fallback — also handles text.len() > MAX_PAYLOAD_BYTES
                // because serde_json itself will produce a payload that exceeds
                // the u16 limit, which write_record will reject with an error.
                let wire = WireChunk::from(chunk);
                let bytes = serde_json::to_vec(&wire)
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                Ok(EncodedPayload::Json(bytes))
            }
        }
    }
}

/// Serialise one record into `out` and return the total number of bytes
/// written (`RECORD_HEADER_BYTES + payload.len()`).
///
/// Returns `Err` if:
/// * `out` is too small to hold the full record, or
/// * the payload exceeds [`MAX_PAYLOAD_BYTES`] (u16 overflow).
pub fn write_record(out: &mut [u8], payload: &EncodedPayload<'_>) -> napi::Result<usize> {
    let (kind, flags, data): (u8, u8, &[u8]) = match payload {
        EncodedPayload::Text { bytes, is_reasoning } => {
            let flags = if *is_reasoning { FLAG_IS_REASONING } else { 0 };
            (KIND_TEXT, flags, bytes)
        }
        EncodedPayload::Json(bytes) => (KIND_JSON, 0, bytes.as_slice()),
        EncodedPayload::Error(msg) => (KIND_ERROR, 0, msg.as_bytes()),
    };

    let payload_len = data.len();
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err(napi::Error::from_reason(format!(
            "wire: payload too large ({} > {})",
            payload_len, MAX_PAYLOAD_BYTES
        )));
    }

    let total = RECORD_HEADER_BYTES + payload_len;
    if out.len() < total {
        return Err(napi::Error::from_reason(format!(
            "wire: output buffer too small ({} < {})",
            out.len(),
            total
        )));
    }

    // len (u16 LE)
    let len_bytes = (payload_len as u16).to_le_bytes();
    out[0] = len_bytes[0];
    out[1] = len_bytes[1];
    out[2] = kind;
    out[3] = flags;
    out[RECORD_HEADER_BYTES..total].copy_from_slice(data);

    Ok(total)
}

/// Parse one record from `input`. Returns `None` if `input` is truncated (no
/// full record header, or header announces more bytes than available).
///
/// On success the returned [`ParsedRecord`] borrows from `input`.
pub fn read_record(input: &[u8]) -> Option<ParsedRecord<'_>> {
    if input.len() < RECORD_HEADER_BYTES {
        return None;
    }
    let payload_len = u16::from_le_bytes([input[0], input[1]]) as usize;
    let total = RECORD_HEADER_BYTES + payload_len;
    if input.len() < total {
        return None;
    }
    Some(ParsedRecord {
        kind: input[2],
        flags: input[3],
        payload: &input[RECORD_HEADER_BYTES..total],
        total_len: total,
    })
}

/// Decode a [`ParsedRecord`] back to a `napi::Result<ChatStreamChunk>`.
///
/// This is the inverse of the `classify_payload` + `write_record` pair.
/// The browser JS reader re-implements decoding in TypeScript; this function
/// is used by unit tests and any native fallback path.
///
/// * `KIND_TEXT` — reconstructs a minimal chunk with `text` set and all other
///   fields at their zero/`None` value. `is_reasoning` is derived from
///   `FLAG_IS_REASONING`.
/// * `KIND_JSON` — deserialises via [`WireChunk`] then converts.
/// * `KIND_ERROR` — returns `Err(napi::Error)` with the UTF-8 payload as the
///   reason string.
pub fn decode_record(rec: &ParsedRecord<'_>) -> napi::Result<ChatStreamChunk> {
    match rec.kind {
        KIND_TEXT => {
            let text = String::from_utf8(rec.payload.to_vec())
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            let is_reasoning = (rec.flags & FLAG_IS_REASONING) != 0;
            Ok(ChatStreamChunk {
                text,
                done: false,
                finish_reason: None,
                tool_calls: None,
                thinking: None,
                num_tokens: None,
                raw_text: None,
                performance: None,
                is_reasoning: Some(is_reasoning),
            })
        }
        KIND_JSON => {
            let wire: WireChunk = serde_json::from_slice(rec.payload)
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(ChatStreamChunk::from(wire))
        }
        KIND_ERROR => {
            let msg = String::from_utf8(rec.payload.to_vec())
                .unwrap_or_else(|_| "<invalid utf-8 in error record>".to_string());
            Err(napi::Error::from_reason(msg))
        }
        other => Err(napi::Error::from_reason(format!(
            "wire: unknown record kind {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::model::ChatStreamChunk;
    use crate::tools::ToolCallResult;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimal chunk that lands on the text fast path. `is_reasoning` is
    /// `Some(false)` — the fast path requires a concrete boolean.
    fn plain_text_chunk(text: &str) -> ChatStreamChunk {
        ChatStreamChunk {
            text: text.to_string(),
            done: false,
            finish_reason: None,
            tool_calls: None,
            thinking: None,
            num_tokens: None,
            raw_text: None,
            performance: None,
            is_reasoning: Some(false),
        }
    }

    fn round_trip(chunk: &ChatStreamChunk) -> napi::Result<ChatStreamChunk> {
        let result = Ok(chunk.clone());
        let payload = classify_payload(&result)?;
        let mut buf = vec![0u8; RECORD_HEADER_BYTES + MAX_PAYLOAD_BYTES];
        let written = write_record(&mut buf, &payload)?;
        let rec = read_record(&buf[..written]).expect("read_record should succeed");
        decode_record(&rec)
    }

    fn compare_chunks(a: &ChatStreamChunk, b: &ChatStreamChunk) {
        // Normalise via WireChunk JSON so we don't need PartialEq on the
        // original type (it is not derived).
        let ja = serde_json::to_string(&WireChunk::from(a)).unwrap();
        let jb = serde_json::to_string(&WireChunk::from(b)).unwrap();
        assert_eq!(ja, jb, "round-tripped chunk does not match original");
    }

    // -----------------------------------------------------------------------
    // Test 1: plain text → Text fast path, flag = 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_plain_text_is_fast_path() {
        let chunk = plain_text_chunk("hello");
        let result = Ok(chunk.clone());
        let payload = classify_payload(&result).unwrap();
        match payload {
            EncodedPayload::Text { bytes, is_reasoning } => {
                assert_eq!(bytes, b"hello");
                assert!(!is_reasoning);
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: reasoning text → Text fast path, FLAG_IS_REASONING set
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_reasoning_text_sets_flag() {
        let mut chunk = plain_text_chunk("thinking...");
        chunk.is_reasoning = Some(true);
        let result = Ok(chunk.clone());
        let payload = classify_payload(&result).unwrap();
        match payload {
            EncodedPayload::Text { bytes, is_reasoning } => {
                assert_eq!(bytes, b"thinking...");
                assert!(is_reasoning, "FLAG_IS_REASONING should be set");
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: done=true, finish_reason, tool_calls, thinking, num_tokens all
    //         force the JSON path.
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_done_is_json() {
        let mut chunk = plain_text_chunk("final");
        chunk.done = true;
        let result = Ok(chunk);
        match classify_payload(&result).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json for done=true, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_finish_reason_is_json() {
        let mut chunk = plain_text_chunk("done");
        chunk.finish_reason = Some("stop".to_string());
        match classify_payload(&Ok(chunk)).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_tool_calls_is_json() {
        let mut chunk = plain_text_chunk("");
        // empty text but tool_calls is Some → JSON
        chunk.tool_calls = Some(vec![]);
        match classify_payload(&Ok(chunk)).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_thinking_is_json() {
        let mut chunk = plain_text_chunk("text");
        chunk.thinking = Some("hmm".to_string());
        match classify_payload(&Ok(chunk)).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_num_tokens_is_json() {
        let mut chunk = plain_text_chunk("text");
        chunk.num_tokens = Some(42);
        match classify_payload(&Ok(chunk)).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Err(e) → Error payload
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_error() {
        let err: napi::Result<ChatStreamChunk> =
            Err(napi::Error::from_reason("boom".to_string()));
        match classify_payload(&err).unwrap() {
            EncodedPayload::Error(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: write_record + read_record round-trip for each kind
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_read_text_roundtrip() {
        let payload = EncodedPayload::Text {
            bytes: b"world",
            is_reasoning: false,
        };
        let mut buf = vec![0u8; 64];
        let written = write_record(&mut buf, &payload).unwrap();
        assert_eq!(written, RECORD_HEADER_BYTES + 5);

        let rec = read_record(&buf[..written]).unwrap();
        assert_eq!(rec.kind, KIND_TEXT);
        assert_eq!(rec.flags, 0);
        assert_eq!(rec.payload, b"world");
        assert_eq!(rec.total_len, written);
    }

    #[test]
    fn test_write_read_json_roundtrip() {
        let chunk = ChatStreamChunk {
            text: "".to_string(),
            done: true,
            finish_reason: Some("stop".to_string()),
            tool_calls: None,
            thinking: None,
            num_tokens: Some(7),
            raw_text: None,
            performance: None,
            is_reasoning: None,
        };
        let result = Ok(chunk);
        let payload = classify_payload(&result).unwrap();
        let mut buf = vec![0u8; 4096];
        let written = write_record(&mut buf, &payload).unwrap();

        let rec = read_record(&buf[..written]).unwrap();
        assert_eq!(rec.kind, KIND_JSON);
        assert_eq!(rec.flags, 0);
        assert_eq!(rec.total_len, written);
        assert_eq!(rec.total_len, RECORD_HEADER_BYTES + rec.payload.len());
    }

    #[test]
    fn test_write_read_error_roundtrip() {
        let payload = EncodedPayload::Error("something went wrong".to_string());
        let mut buf = vec![0u8; 64];
        let written = write_record(&mut buf, &payload).unwrap();
        assert_eq!(written, RECORD_HEADER_BYTES + "something went wrong".len());

        let rec = read_record(&buf[..written]).unwrap();
        assert_eq!(rec.kind, KIND_ERROR);
        assert_eq!(rec.flags, 0);
        assert_eq!(rec.payload, b"something went wrong");
        assert_eq!(rec.total_len, written);
    }

    // -----------------------------------------------------------------------
    // Test 6: full classify → write → read → decode round-trips
    // -----------------------------------------------------------------------
    #[test]
    fn test_decode_plain_text_chunk() {
        let chunk = plain_text_chunk("token");
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    #[test]
    fn test_decode_reasoning_text_chunk() {
        let mut chunk = plain_text_chunk("step one");
        chunk.is_reasoning = Some(true);
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    #[test]
    fn test_decode_done_chunk() {
        let chunk = ChatStreamChunk {
            text: "".to_string(),
            done: true,
            finish_reason: Some("stop".to_string()),
            tool_calls: None,
            thinking: None,
            num_tokens: Some(10),
            raw_text: Some("raw".to_string()),
            performance: None,
            is_reasoning: None,
        };
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    #[test]
    fn test_decode_tool_calls_chunk() {
        let tc = ToolCallResult::ok(
            "my_tool".to_string(),
            serde_json::json!({"key": "val"}),
            "raw".to_string(),
        );
        let chunk = ChatStreamChunk {
            text: "".to_string(),
            done: true,
            finish_reason: Some("tool_calls".to_string()),
            tool_calls: Some(vec![tc.clone()]),
            thinking: None,
            num_tokens: Some(3),
            raw_text: None,
            performance: None,
            is_reasoning: None,
        };
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    #[test]
    fn test_decode_performance_chunk() {
        let perf = crate::profiling::PerformanceMetrics {
            ttft_ms: 12.5,
            prefill_tokens_per_second: 100.0,
            decode_tokens_per_second: 80.0,
        };
        let chunk = ChatStreamChunk {
            text: "".to_string(),
            done: true,
            finish_reason: Some("stop".to_string()),
            tool_calls: None,
            thinking: Some("thought".to_string()),
            num_tokens: Some(5),
            raw_text: Some("raw_text".to_string()),
            performance: Some(perf),
            is_reasoning: None,
        };
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    // -----------------------------------------------------------------------
    // Drift-guard: maximally populated chunk. The destructuring below forces
    // a compile error if a new field is added to ChatStreamChunk without
    // updating this test — which is the cheapest mechanical way to keep the
    // WireChunk mirror and the From impls in sync.
    // -----------------------------------------------------------------------
    #[test]
    fn test_decode_maximally_populated_chunk() {
        let perf = crate::profiling::PerformanceMetrics {
            ttft_ms: 1.5,
            prefill_tokens_per_second: 512.0,
            decode_tokens_per_second: 45.5,
        };
        let tc = ToolCallResult::ok(
            "search".to_string(),
            serde_json::json!({"q": "rust"}),
            "raw".to_string(),
        );
        let chunk = ChatStreamChunk {
            text: "full".to_string(),
            done: true,
            finish_reason: Some("stop".to_string()),
            tool_calls: Some(vec![tc]),
            thinking: Some("thinking text".to_string()),
            num_tokens: Some(99),
            raw_text: Some("raw output".to_string()),
            performance: Some(perf),
            is_reasoning: Some(true),
        };
        // Destructuring ensures a new field on ChatStreamChunk won't compile
        // without updating this test (and hence WireChunk + From impls).
        let ChatStreamChunk {
            text: _,
            done: _,
            finish_reason: _,
            tool_calls: _,
            thinking: _,
            num_tokens: _,
            raw_text: _,
            performance: _,
            is_reasoning: _,
        } = &chunk;
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
    }

    // -----------------------------------------------------------------------
    // is_reasoning: None round-trips via JSON (not the text fast path)
    // so the three states (None, Some(false), Some(true)) are lossless.
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_is_reasoning_none_falls_through_to_json() {
        let mut chunk = plain_text_chunk("word");
        chunk.is_reasoning = None;
        match classify_payload(&Ok(chunk)).unwrap() {
            EncodedPayload::Json(_) => {}
            other => panic!("expected Json for is_reasoning=None, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_is_reasoning_none_roundtrip() {
        let mut chunk = plain_text_chunk("word");
        chunk.is_reasoning = None;
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
        assert_eq!(decoded.is_reasoning, None);
    }

    #[test]
    fn test_decode_is_reasoning_some_false_roundtrip() {
        let chunk = plain_text_chunk("word"); // is_reasoning = Some(false)
        let decoded = round_trip(&chunk).unwrap();
        compare_chunks(&chunk, &decoded);
        assert_eq!(decoded.is_reasoning, Some(false));
    }

    // -----------------------------------------------------------------------
    // Test 7: read_record on truncated input returns None
    // -----------------------------------------------------------------------
    #[test]
    fn test_read_record_truncated_header() {
        // Fewer than RECORD_HEADER_BYTES bytes available.
        assert!(read_record(&[0u8, 0u8, 0u8]).is_none());
    }

    #[test]
    fn test_read_record_truncated_payload() {
        // Header present (claims payload_len = 10) but only 3 bytes of payload.
        let mut buf = [0u8; 7];
        buf[0] = 10; // payload_len low byte
        buf[1] = 0;  // payload_len high byte
        buf[2] = KIND_TEXT;
        buf[3] = 0;
        // buf[4..7] — only 3 bytes where 10 are needed
        assert!(read_record(&buf).is_none());
    }

    // -----------------------------------------------------------------------
    // Test 8: write_record into a buffer that is too small returns Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_record_buffer_too_small() {
        let payload = EncodedPayload::Text {
            bytes: b"hello world",
            is_reasoning: false,
        };
        // Buffer big enough for the header (4) but not for the payload (11).
        let mut buf = [0u8; 8];
        let result = write_record(&mut buf, &payload);
        assert!(result.is_err(), "expected Err for undersized buffer");
    }

    // -----------------------------------------------------------------------
    // Test 9: text longer than MAX_PAYLOAD_BYTES falls through to Json
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_giant_text_falls_back_to_json() {
        // Construct a chunk whose text exceeds u16::MAX bytes.
        let big_text = "x".repeat(MAX_PAYLOAD_BYTES + 1);
        let chunk = plain_text_chunk(&big_text);
        let result = Ok(chunk);
        let payload = classify_payload(&result).unwrap();
        match &payload {
            EncodedPayload::Json(_) => {}
            EncodedPayload::Text { .. } => {
                panic!("expected Json fallback for oversized text, got Text")
            }
            other => panic!("expected Json, got {:?}", other),
        }
        // The JSON payload itself is > MAX_PAYLOAD_BYTES, so write_record
        // must reject with a "payload too large" error rather than panic.
        let mut buf = vec![0u8; RECORD_HEADER_BYTES + MAX_PAYLOAD_BYTES + 64];
        let res = write_record(&mut buf, &payload);
        assert!(
            res.is_err(),
            "write_record should reject oversized JSON payload"
        );
        let err_msg = res.unwrap_err().reason;
        assert!(
            err_msg.contains("payload too large"),
            "unexpected error: {err_msg}"
        );
    }
}
