//! Gemma4 model output parsing.
//!
//! Gemma4 emits structured outputs using three families of delimiter tokens:
//!
//! * `<|channel>thought\n...\n<channel|>` — reasoning/"chain-of-thought" block
//! * `<|tool_call>call:NAME{K:V,K:V,...}<tool_call|>` — zero or more tool calls
//! * `<|tool_response>...<tool_response|>`, `<|tool>...<tool|>` — input-only
//!   tokens that should not appear on the model's output; this module still
//!   strips them defensively if it ever sees them so they never leak through
//!   as user-visible content.
//!
//! The tool-call arguments are **NOT** JSON. They use Gemma4's own DSL — see
//! [`crate::models::gemma4::model::json_args_to_gemma4_dsl`] for the reverse
//! direction. This module owns the inverse parser; `parse_gemma4_dsl_args`
//! MUST be the inverse of `json_args_to_gemma4_dsl` for any fixture that
//! passes through the encoder.
//!
//! Two entry points:
//!
//! * [`parse_gemma4_output`] — offline parse of a completed decode string
//!   (used by the non-streaming `ChatResult` construction sites).
//! * [`Gemma4StreamParser`] — incremental parser driven by the streaming
//!   decode loop. It buffers bytes that might be the prefix of a delimiter
//!   and surfaces segments as text / reasoning deltas (or, for tool calls,
//!   a fully-parsed terminal ToolCall once the closing tag arrives).

use crate::tools::ToolCallResult;
use serde_json::Value;

/// True iff the ASCII byte pattern `needle` starts at byte offset `pos` of
/// `src`. Unlike `src[pos..].starts_with(needle)`, this is safe even when
/// `pos` lands mid-UTF-8 multi-byte char — we never take a `&str` slice at
/// a non-boundary position. All DSL delimiters (`<|"|>`, `<|channel>`, …) are
/// pure ASCII, so a match at any byte position is always a valid char
/// boundary (ASCII bytes never appear as UTF-8 continuation bytes).
fn bytes_starts_with(src: &str, pos: usize, needle: &str) -> bool {
    let b = src.as_bytes();
    let n = needle.as_bytes();
    b.len() >= pos.saturating_add(n.len()) && &b[pos..pos + n.len()] == n
}

// ---------------------------------------------------------------------------
// Delimiter constants
// ---------------------------------------------------------------------------

const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";
const TOOL_CALL_OPEN: &str = "<|tool_call>";
const TOOL_CALL_CLOSE: &str = "<tool_call|>";
/// Input-only token — stripped defensively on output.
const TOOL_RESPONSE_OPEN: &str = "<|tool_response>";
/// Input-only token — stripped defensively on output.
const TOOL_RESPONSE_CLOSE: &str = "<tool_response|>";
/// Input-only token — stripped defensively on output.
const TOOL_OPEN: &str = "<|tool>";
/// Input-only token — stripped defensively on output.
const TOOL_CLOSE: &str = "<tool|>";
/// String-delimiter used inside the DSL: `<|"|>str<|"|>`.
const DSL_STRING_DELIM: &str = "<|\"|>";

/// The set of opening delimiters the stream parser must watch for. Ordered
/// longest-first so an in-progress prefix of a longer token is preferred
/// over a shorter prefix that is always ambiguous (e.g. a literal `<`).
const OPEN_MARKERS: &[&str] = &[TOOL_RESPONSE_OPEN, TOOL_CALL_OPEN, CHANNEL_OPEN, TOOL_OPEN];

// ---------------------------------------------------------------------------
// DSL parsing
// ---------------------------------------------------------------------------

/// Error kind produced by [`parse_gemma4_dsl_args`]. Only surfaced in debug
/// paths — the public API falls back to a raw-string representation when a
/// parse fails so a best-effort tool call is still emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DslParseError(pub String);

impl std::fmt::Display for DslParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DslParseError {}

/// Parse Gemma4's `{k:v,k:v,...}` argument DSL into a JSON `Value`.
///
/// Inverse of [`crate::models::gemma4::model::json_args_to_gemma4_dsl`].
/// Accepts input with or without surrounding braces — `json_args_to_gemma4_dsl`
/// emits the inner pairs only, so callers of this function typically pass
/// the inner slice. Both shapes round-trip.
pub(crate) fn parse_gemma4_dsl_args(s: &str) -> Result<Value, DslParseError> {
    let trimmed = s.trim();
    // The encoder emits the inner (`k:v,k:v`) form without braces — accept
    // either shape transparently.
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .unwrap_or(trimmed);
    let mut parser = DslParser::new(inner);
    let val = parser.parse_object_body()?;
    parser.skip_ws();
    if !parser.at_end() {
        return Err(DslParseError(format!(
            "trailing bytes in DSL at position {}: {:?}",
            parser.pos,
            &parser.src[parser.pos..]
        )));
    }
    Ok(val)
}

/// Recursive-descent parser over a UTF-8 byte slice. Position indices point
/// at byte offsets, which is safe because all DSL structural bytes
/// (`{`, `}`, `[`, `]`, `,`, `:`) are ASCII. Arbitrary UTF-8 text inside
/// values is preserved verbatim.
struct DslParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> DslParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek_byte() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Parse the body of an object — `k:v,k:v,...` — up to end-of-input
    /// (used for the top-level un-braced form) or a `}` (used for nested
    /// objects; the closing brace is NOT consumed here, the caller does).
    fn parse_object_body(&mut self) -> Result<Value, DslParseError> {
        let mut map = serde_json::Map::new();
        self.skip_ws();
        // Empty object body — {} or `` — is valid.
        if self.at_end() || self.peek_byte() == Some(b'}') {
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_key()?;
            self.skip_ws();
            if self.peek_byte() != Some(b':') {
                return Err(DslParseError(format!(
                    "expected ':' after key {:?} at position {}",
                    key, self.pos
                )));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.peek_byte() {
                Some(b',') => {
                    self.pos += 1;
                    continue;
                }
                Some(b'}') | None => break,
                Some(other) => {
                    return Err(DslParseError(format!(
                        "unexpected byte {:?} after value at position {}",
                        other as char, self.pos
                    )));
                }
            }
        }
        Ok(Value::Object(map))
    }

    /// Parse an object key. Keys are bare identifier-like tokens terminated
    /// by `:`. The encoder emits only ascii keys from the JSON input, so we
    /// accept anything up to the next `:`.
    fn parse_key(&mut self) -> Result<String, DslParseError> {
        // Accept optional `<|"|>...<|"|>` wrapping for keys defensively.
        if bytes_starts_with(self.src, self.pos, DSL_STRING_DELIM) {
            return self.parse_quoted_string();
        }
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if matches!(b, b':' | b',' | b'}' | b']') {
                break;
            }
            self.pos += 1;
        }
        if start == self.pos {
            return Err(DslParseError(format!("empty key at position {}", self.pos)));
        }
        Ok(self.src[start..self.pos].trim().to_string())
    }

    fn parse_value(&mut self) -> Result<Value, DslParseError> {
        self.skip_ws();
        match self.peek_byte() {
            Some(b'{') => {
                self.pos += 1;
                let inner = self.parse_object_body()?;
                self.skip_ws();
                if self.peek_byte() != Some(b'}') {
                    return Err(DslParseError(format!(
                        "unterminated object at position {}",
                        self.pos
                    )));
                }
                self.pos += 1;
                Ok(inner)
            }
            Some(b'[') => {
                self.pos += 1;
                self.parse_array_body()
            }
            _ if bytes_starts_with(self.src, self.pos, DSL_STRING_DELIM) => {
                self.parse_quoted_string().map(Value::String)
            }
            _ => self.parse_bare_value(),
        }
    }

    fn parse_array_body(&mut self) -> Result<Value, DslParseError> {
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek_byte() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(arr));
        }
        loop {
            self.skip_ws();
            let v = self.parse_value()?;
            arr.push(v);
            self.skip_ws();
            match self.peek_byte() {
                Some(b',') => {
                    self.pos += 1;
                    continue;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(DslParseError(format!(
                        "unterminated array at position {}",
                        self.pos
                    )));
                }
            }
        }
        Ok(Value::Array(arr))
    }

    fn parse_quoted_string(&mut self) -> Result<String, DslParseError> {
        // Consume opening <|"|>. Caller guarantees we're at an ASCII
        // boundary (the `<` of the delimiter), so byte slicing is safe.
        let delim_len = DSL_STRING_DELIM.len();
        self.pos += delim_len;
        let start = self.pos;
        // Search for the closing delimiter by bytes — safe through
        // multi-byte UTF-8 content because `<|"|>` is pure ASCII and
        // an ASCII byte never appears as a UTF-8 continuation byte.
        let haystack = self.src.as_bytes();
        let needle = DSL_STRING_DELIM.as_bytes();
        let close = find_subslice(&haystack[start..], needle).ok_or_else(|| {
            DslParseError(format!("unterminated quoted string at position {}", start))
        })?;
        let body = &self.src[start..start + close];
        self.pos = start + close + delim_len;
        Ok(body.to_string())
    }

    /// Parse a bare value — number, bool, null, or bare string. A bare
    /// string runs up to the next `,`, `}`, or `]` at the current nesting
    /// level (which is always 0 inside this helper, since nested objects /
    /// arrays are handled via their own parsers before reaching here).
    /// Whitespace is trimmed from both ends of the slice before value-type
    /// detection.
    fn parse_bare_value(&mut self) -> Result<Value, DslParseError> {
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if matches!(b, b',' | b'}' | b']') {
                break;
            }
            self.pos += 1;
        }
        let raw = self.src[start..self.pos].trim();
        if raw.is_empty() {
            // Edge case: `key:,...` — treat as empty string to match the
            // best-effort contract. The encoder never produces this.
            return Ok(Value::String(String::new()));
        }
        Ok(bare_scalar_to_json(raw))
    }
}

fn bare_scalar_to_json(raw: &str) -> Value {
    match raw {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ => {
            // Try integer first to preserve precision for large ints.
            if let Ok(n) = raw.parse::<i64>() {
                return Value::from(n);
            }
            if let Ok(n) = raw.parse::<u64>() {
                return Value::from(n);
            }
            if let Ok(f) = raw.parse::<f64>()
                && let Some(n) = serde_json::Number::from_f64(f)
            {
                return Value::Number(n);
            }
            Value::String(raw.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Block-level output parser (offline)
// ---------------------------------------------------------------------------

/// Structured outcome from parsing a complete Gemma4 model output string
/// (with special tokens preserved — i.e. `decode_sync(tokens, false)`).
pub(crate) struct Gemma4ParsedOutput {
    /// User-visible message text with channel/tool-call markers stripped.
    pub text: String,
    /// Reasoning content from the `<|channel>thought\n...<channel|>` block.
    /// `None` when the model produced no channel block.
    pub thinking: Option<String>,
    /// Tool calls extracted from `<|tool_call>...<tool_call|>` blocks.
    pub tool_calls: Vec<ToolCallResult>,
}

/// Offline parse of a complete decoded string. Invariant: whatever chain of
/// markers the model emitted, this function returns cleaned text, joined
/// thinking, and all parsed tool calls in order.
pub(crate) fn parse_gemma4_output(text: &str) -> Gemma4ParsedOutput {
    let mut parser = Gemma4StreamParser::new();
    let mut segments = parser.feed(text);
    segments.extend(parser.flush());

    let mut cleaned = String::new();
    for seg in segments {
        if let StreamSegment::Text(t) = seg {
            cleaned.push_str(&t);
        }
    }

    Gemma4ParsedOutput {
        text: cleaned,
        thinking: parser.thinking(),
        tool_calls: parser.tool_calls(),
    }
}

// ---------------------------------------------------------------------------
// Stream parser
// ---------------------------------------------------------------------------

/// Output unit from [`Gemma4StreamParser::feed`] / `flush`. The caller maps
/// each segment to a `ChatStreamChunk`:
///
/// * `Text`     → `{text, is_reasoning: Some(false)}`
/// * `Reasoning`→ `{text, is_reasoning: Some(true)}`
/// * `ToolCall` → accumulate, only emit on the terminal `done=true` chunk.
#[derive(Debug, Clone)]
pub(crate) enum StreamSegment {
    Text(String),
    Reasoning(String),
    /// Payload is read by tests and by `parser.tool_calls()`; the wire
    /// dispatcher ignores it because tool calls only land on the terminal
    /// chunk. Marked `allow(dead_code)` so we can keep the payload for
    /// test ergonomics without tripping `-D warnings`.
    #[allow(dead_code)]
    ToolCall(ToolCallResult),
}

/// Current position of the stream parser inside Gemma4's output grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    /// Plain message text. Bytes flow through to the consumer as `Text`
    /// segments (minus any prefix that might be the start of an opening
    /// marker — that prefix is buffered in `pending`).
    Message,
    /// Inside a `<|channel>...<channel|>` block. Bytes flow out as
    /// `Reasoning` segments and accumulate into `thinking`.
    Channel,
    /// Inside a `<|tool_call>...<tool_call|>` block. Bytes are buffered in
    /// `tool_call_buf` until the closing marker is seen, then parsed and
    /// emitted as a single `ToolCall` segment.
    ToolCall,
    /// Inside a `<|tool_response>...<tool_response|>` or `<|tool>...<tool|>`
    /// block. Input-only markers that should not appear on output, but we
    /// swallow their contents defensively.
    Swallow { close: &'static str },
}

pub(crate) struct Gemma4StreamParser {
    state: StreamState,
    /// Buffer for bytes that might be a prefix of an opening/closing marker.
    /// Emitted verbatim once we know they aren't.
    pending: String,
    /// Accumulator for reasoning content (full channel body, across feeds).
    thinking: String,
    /// True once we've entered a channel block at least once.
    saw_channel: bool,
    /// Has the `thought\n` channel label been stripped from the current
    /// channel block yet? Set to `false` on each `<|channel>` open and
    /// flipped to `true` on the first reasoning emit that consumes the
    /// label. Subsequent emits inside the same block pass through
    /// unchanged. See `strip_channel_label` for the full rationale —
    /// the template re-emits its own `thought\n` label on echo, so the
    /// client-visible `thinking` MUST NOT carry it.
    channel_label_stripped: bool,
    /// Raw bytes of the current tool-call block (everything between the
    /// opening and closing markers).
    tool_call_buf: String,
    /// All tool calls parsed so far across all feeds.
    tool_calls: Vec<ToolCallResult>,
}

impl Default for Gemma4StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4StreamParser {
    pub(crate) fn new() -> Self {
        Self {
            state: StreamState::Message,
            pending: String::new(),
            thinking: String::new(),
            saw_channel: false,
            channel_label_stripped: true,
            tool_call_buf: String::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Accumulated reasoning content, or `None` if no channel block was
    /// ever observed. Trimmed of leading/trailing whitespace to match
    /// the common "body of the thought" interpretation.
    ///
    /// The `thought\n` channel label is stripped incrementally during
    /// streaming (see `channel_label_stripped` / the first-emit path in
    /// `drain_until_close`) so the accumulated buffer already reflects
    /// the label-less body. The Gemma4 chat template re-emits its own
    /// `thought\n` label when echoing `reasoning_content` back on a
    /// subsequent turn — if we left the label in `thinking` the
    /// rendered prompt would double it (`thought\nthought\n{body}`),
    /// diverging from the bytes the model originally generated and
    /// zeroing `verify_cache_prefix`'s prefix match. See
    /// `strip_channel_label` for the full rationale.
    pub(crate) fn thinking(&self) -> Option<String> {
        if self.saw_channel {
            let t = self.thinking.trim();
            if t.is_empty() {
                Some(String::new())
            } else {
                Some(t.to_string())
            }
        } else {
            None
        }
    }

    /// Tool calls parsed across the full stream.
    pub(crate) fn tool_calls(&self) -> Vec<ToolCallResult> {
        self.tool_calls.clone()
    }

    /// Feed decoded text. Returns any segments ready to surface on this
    /// turn. Partial markers are buffered internally and emitted as text
    /// (or swallowed, depending on context) once resolved.
    pub(crate) fn feed(&mut self, chunk: &str) -> Vec<StreamSegment> {
        let mut out = Vec::new();
        self.pending.push_str(chunk);
        self.drain(&mut out, /*flushing=*/ false);
        out
    }

    /// Final flush. Drains any buffered bytes — in `Message`/`Channel`
    /// states, those surface as a final `Text`/`Reasoning` segment; in
    /// `ToolCall`/`Swallow` states, they are discarded (the model never
    /// closed the block, which is a malformed output; we refuse to
    /// hallucinate a partial tool call).
    pub(crate) fn flush(&mut self) -> Vec<StreamSegment> {
        let mut out = Vec::new();
        self.drain(&mut out, /*flushing=*/ true);
        out
    }

    /// Core state machine loop. Repeatedly consumes `self.pending` until
    /// either it's empty or no progress can be made (an ambiguous marker
    /// prefix is sitting at the tail of the buffer).
    ///
    /// `flushing` controls whether partial prefixes at end-of-buffer are
    /// held back (`false` — mid-stream, might complete on the next feed)
    /// or surfaced (`true` — end-of-generation, nothing more is coming).
    fn drain(&mut self, out: &mut Vec<StreamSegment>, flushing: bool) {
        loop {
            match self.state {
                StreamState::Message => {
                    if !self.drain_until_open_marker(out, flushing) {
                        return;
                    }
                }
                StreamState::Channel => {
                    if !self.drain_until_close(
                        out,
                        CHANNEL_CLOSE,
                        /*reasoning=*/ true,
                        flushing,
                    ) {
                        return;
                    }
                }
                StreamState::ToolCall => {
                    if !self.drain_tool_call(out, flushing) {
                        return;
                    }
                }
                StreamState::Swallow { close } => {
                    if !self.drain_until_close(out, close, /*reasoning=*/ false, flushing) {
                        return;
                    }
                }
            }
        }
    }

    /// Message state: emit plain text up to the next opening marker, or
    /// buffer a possible partial marker at the tail. Returns `true` if
    /// the state transitioned (another iteration of the outer loop is
    /// needed) or the buffer is fully drained; `false` if we're waiting
    /// for more input.
    fn drain_until_open_marker(&mut self, out: &mut Vec<StreamSegment>, flushing: bool) -> bool {
        // Search for the earliest opening marker.
        let mut best: Option<(usize, &'static str)> = None;
        for marker in OPEN_MARKERS {
            if let Some(idx) = self.pending.find(marker) {
                match best {
                    Some((bi, _)) if bi <= idx => {}
                    _ => best = Some((idx, marker)),
                }
            }
        }

        if let Some((idx, marker)) = best {
            if idx > 0 {
                let prefix: String = self.pending.drain(..idx).collect();
                out.push(StreamSegment::Text(prefix));
            }
            // Consume the marker.
            self.pending.drain(..marker.len());
            self.state = match marker {
                m if m == CHANNEL_OPEN => {
                    self.saw_channel = true;
                    // Fresh channel block — arm the label stripper so
                    // the first reasoning emit consumes the hardcoded
                    // `thought\n` label the chat template re-emits on
                    // echo. See `strip_channel_label`.
                    self.channel_label_stripped = false;
                    StreamState::Channel
                }
                m if m == TOOL_CALL_OPEN => {
                    self.tool_call_buf.clear();
                    StreamState::ToolCall
                }
                m if m == TOOL_RESPONSE_OPEN => StreamState::Swallow {
                    close: TOOL_RESPONSE_CLOSE,
                },
                m if m == TOOL_OPEN => StreamState::Swallow { close: TOOL_CLOSE },
                _ => unreachable!("OPEN_MARKERS entry without matching branch"),
            };
            return true;
        }

        // No complete marker. If we're flushing, dump everything. Otherwise,
        // hold back any tail that could be the prefix of a marker.
        if flushing {
            if !self.pending.is_empty() {
                let text = std::mem::take(&mut self.pending);
                out.push(StreamSegment::Text(text));
            }
            return false;
        }
        let hold = longest_prefix_hold(&self.pending, OPEN_MARKERS);
        if hold > 0 {
            let emit_len = self.pending.len() - hold;
            if emit_len > 0 {
                let emit: String = self.pending.drain(..emit_len).collect();
                out.push(StreamSegment::Text(emit));
            }
        } else if !self.pending.is_empty() {
            let emit = std::mem::take(&mut self.pending);
            out.push(StreamSegment::Text(emit));
        }
        false
    }

    /// Emit a reasoning chunk, stripping the hardcoded `thought\n`
    /// channel label from the first emit of a block. Always the only
    /// path that feeds `self.thinking` + pushes a `Reasoning` segment
    /// so label stripping stays in sync across the accumulated buffer
    /// and the streaming deltas.
    ///
    /// See `strip_channel_label` for why the label must be stripped
    /// eagerly: the Gemma4 chat template re-emits `thought\n` on echo,
    /// so the client-visible reasoning body MUST NOT carry it or the
    /// re-rendered prefix diverges from the bytes the model originally
    /// generated and KV-cache reuse misses on every turn.
    fn emit_reasoning(&mut self, out: &mut Vec<StreamSegment>, body: String) {
        if body.is_empty() {
            return;
        }
        let stripped = if !self.channel_label_stripped {
            self.channel_label_stripped = true;
            // `strip_channel_label` is byte-safe: the label is ASCII-only,
            // so stripping it never lands mid-UTF-8. Falls through to
            // the original body when the first chunk doesn't start with
            // the expected label (unseen channel variants — pass-through
            // keeps those correct even if they won't re-render).
            strip_channel_label(&body).to_string()
        } else {
            body
        };
        if stripped.is_empty() {
            return;
        }
        self.thinking.push_str(&stripped);
        out.push(StreamSegment::Reasoning(stripped));
    }

    /// Drain bytes in states that end on a fixed close marker: `Channel`
    /// (emit as `Reasoning`) or `Swallow` (discard). Mirrors the guard
    /// contract of `drain_until_open_marker`.
    fn drain_until_close(
        &mut self,
        out: &mut Vec<StreamSegment>,
        close: &'static str,
        reasoning: bool,
        flushing: bool,
    ) -> bool {
        // Buffer the `thought\n` label across feed boundaries so the
        // stripper has enough bytes to make a definitive match. The
        // label is 8 bytes; hold back up to that many while we wait for
        // more input (non-flushing path). The close marker also needs
        // to be held back — use the max of the two to avoid racing
        // either guard.
        if reasoning && !self.channel_label_stripped && !flushing {
            const LABEL_LEN: usize = "thought\n".len();
            // Only force-buffer while the pending is short AND the close
            // marker hasn't landed yet. If the close marker is already
            // visible, fall through to the normal drain — the label
            // stripper will run inside `emit_reasoning` on the short
            // body slice.
            if self.pending.len() < LABEL_LEN && !self.pending.contains(close) {
                return false;
            }
        }

        if let Some(idx) = self.pending.find(close) {
            let body: String = self.pending.drain(..idx).collect();
            // Drop the close marker.
            self.pending.drain(..close.len());
            if reasoning {
                self.emit_reasoning(out, body);
            }
            // For Swallow we intentionally discard `body`.
            self.state = StreamState::Message;
            return true;
        }

        // No close yet. Emit / swallow everything except a possible partial
        // close marker at the tail (when not flushing).
        if flushing {
            if reasoning && !self.pending.is_empty() {
                let body = std::mem::take(&mut self.pending);
                self.emit_reasoning(out, body);
            } else {
                self.pending.clear();
            }
            // Reset to Message so callers don't stay wedged in a broken
            // state if flush is ever called on a non-terminated block.
            self.state = StreamState::Message;
            return false;
        }
        let hold = longest_prefix_hold(&self.pending, &[close]);
        if hold == self.pending.len() {
            return false;
        }
        let emit_len = self.pending.len() - hold;
        let body: String = self.pending.drain(..emit_len).collect();
        if reasoning {
            self.emit_reasoning(out, body);
        }
        false
    }

    /// Drain bytes in the tool-call state: accumulate into `tool_call_buf`
    /// until the closing marker is seen, then parse and emit a single
    /// `ToolCall` segment. Bytes do NOT stream out incrementally — a
    /// partial tool call is useless without its closing brace.
    fn drain_tool_call(&mut self, out: &mut Vec<StreamSegment>, flushing: bool) -> bool {
        if let Some(idx) = self.pending.find(TOOL_CALL_CLOSE) {
            let body: String = self.pending.drain(..idx).collect();
            self.pending.drain(..TOOL_CALL_CLOSE.len());
            self.tool_call_buf.push_str(&body);
            let raw = std::mem::take(&mut self.tool_call_buf);
            let tc = parse_tool_call_body(&raw);
            self.tool_calls.push(tc.clone());
            out.push(StreamSegment::ToolCall(tc));
            self.state = StreamState::Message;
            return true;
        }

        if flushing {
            // Unterminated tool call at end-of-stream — drop it. We don't
            // emit a broken ToolCall because we cannot reconstruct the name.
            self.tool_call_buf.clear();
            self.pending.clear();
            self.state = StreamState::Message;
            return false;
        }
        let hold = longest_prefix_hold(&self.pending, &[TOOL_CALL_CLOSE]);
        if hold == self.pending.len() {
            return false;
        }
        let emit_len = self.pending.len() - hold;
        let body: String = self.pending.drain(..emit_len).collect();
        self.tool_call_buf.push_str(&body);
        false
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip the leading `thought\n` channel label from a reasoning body so
/// the returned slice is pure body text that round-trips byte-for-byte
/// through the Gemma4 chat template.
///
/// Gemma4's channel grammar is `<|channel>{label}\n{body}\n<channel|>`.
/// The shipped chat template (`chat_template.jinja` line 238) hardcodes
/// `thought` as the only label it re-emits:
///
/// ```text
/// '<|channel>thought\n' + thinking_text + '\n<channel|>'
/// ```
///
/// Callers must therefore store `thinking` as the body only, without
/// the label, so that when a downstream consumer echoes it back as
/// `reasoning_content` and the template re-renders it, the output
/// matches the bytes the model originally generated. Any other channel
/// label (future models may use `deliberation`, `scratchpad`, etc.)
/// passes through unchanged — the template wouldn't know how to echo
/// those anyway, and preserving the raw body lets the caller make an
/// informed choice.
fn strip_channel_label(body: &str) -> &str {
    const THOUGHT_LABEL: &str = "thought\n";
    body.strip_prefix(THOUGHT_LABEL).unwrap_or(body)
}

/// Longest suffix of `buf` that is a non-empty prefix of any marker in
/// `markers`. Used by the stream parser to hold back ambiguous tail bytes
/// across feed boundaries.
fn longest_prefix_hold(buf: &str, markers: &[&str]) -> usize {
    let max_marker = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    let upper = max_marker.min(buf.len());
    for len in (1..=upper).rev() {
        // Ensure we're on a UTF-8 boundary — special tokens are ASCII, so
        // this is belt-and-suspenders, but protects against pathological
        // inputs where a partial UTF-8 sequence happens to sit at the tail.
        let start = buf.len() - len;
        if !buf.is_char_boundary(start) {
            continue;
        }
        let tail = &buf[start..];
        if markers.iter().any(|m| m.starts_with(tail)) {
            return len;
        }
    }
    0
}

/// Parse the body of a tool-call block: `call:NAME{K:V,...}`.
///
/// Returns a best-effort `ToolCallResult` — if the arguments body fails
/// to parse as DSL, the raw string is preserved via
/// [`ToolCallResult::parse_error`] so the user still sees SOME tool call
/// rather than having it silently dropped.
fn parse_tool_call_body(raw: &str) -> ToolCallResult {
    let trimmed = raw.trim();
    // Expected shape: `call:NAME{...}`
    let after_call = trimmed.strip_prefix("call:").unwrap_or(trimmed);
    // Split at the first `{` — everything before is the name, everything
    // between that `{` and its matching `}` is the argument DSL.
    let (name, args_region) = match after_call.find('{') {
        Some(i) => (after_call[..i].trim().to_string(), &after_call[i..]),
        None => (after_call.trim().to_string(), ""),
    };

    if name.is_empty() {
        return ToolCallResult::missing_name(raw.to_string());
    }

    // Slice out the balanced argument region. The encoder emits exactly
    // one top-level `{...}` so a naive depth counter suffices.
    let args_body: Option<String> = if args_region.is_empty() {
        Some(String::new())
    } else {
        extract_balanced_braces(args_region)
    };

    let Some(args_inner) = args_body else {
        // Unbalanced braces — preserve raw as parse_error.
        return ToolCallResult::parse_error(
            name,
            args_region.to_string(),
            "unbalanced braces in tool call arguments".to_string(),
            raw.to_string(),
        );
    };

    if args_inner.is_empty() {
        return ToolCallResult::ok(name, Value::Object(serde_json::Map::new()), raw.to_string());
    }

    match parse_gemma4_dsl_args(&args_inner) {
        Ok(v) => ToolCallResult::ok(name, v, raw.to_string()),
        Err(e) => ToolCallResult::parse_error(name, args_inner, e.to_string(), raw.to_string()),
    }
}

/// Given a slice that starts with `{`, return the content between the `{`
/// and its matching `}` (a depth-tracked scan that correctly handles
/// nested `{...}` / `[...]` inside). Returns `None` if the braces are
/// unbalanced.
fn extract_balanced_braces(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first().copied() != Some(b'{') {
        return None;
    }
    let delim = DSL_STRING_DELIM.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    let mut in_dsl_string = false;
    // Walk entirely in byte-space — DSL delimiters and structural tokens
    // (`{`, `}`) are all ASCII, so comparing bytes is safe even when the
    // enclosed string contains multi-byte UTF-8 (e.g. an em-dash). The
    // final `s[1..i]` slice lands on ASCII boundaries (`{`, `}`) which
    // are always valid char boundaries.
    while i < bytes.len() {
        // DSL-string delimiter — 5 bytes (`<|"|>`). Strings may contain
        // `{`/`}` bytes that must NOT be counted as depth markers, so
        // flip a flag as we cross the delimiter.
        if bytes.len() >= i + delim.len() && &bytes[i..i + delim.len()] == delim {
            in_dsl_string = !in_dsl_string;
            i += delim.len();
            continue;
        }
        if in_dsl_string {
            // Advance one byte at a time through arbitrary UTF-8 payload.
            // Safe: we only compare against ASCII bytes above, and the
            // loop never slices `&str` at `i` while `in_dsl_string` is
            // true.
            i += 1;
            continue;
        }
        let b = bytes[i];
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                // Return the inner slice (exclusive of the outer braces).
                // `i` is at a `}` which is ASCII → safe char boundary.
                return Some(s[1..i].to_string());
            }
        }
        i += 1;
    }
    None
}

/// Byte-level `memmem` replacement — finds the first occurrence of
/// `needle` in `haystack`. Used so we can search for ASCII delimiters
/// through UTF-8 payloads without ever having to take a `&str` slice at
/// a non-boundary position. Returns the byte offset into `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gemma4::model::json_args_to_gemma4_dsl_for_test;

    #[test]
    fn parse_plain_text_no_markers() {
        let parsed = parse_gemma4_output("hello world, how are you?");
        assert_eq!(parsed.text, "hello world, how are you?");
        assert!(parsed.thinking.is_none());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_channel_only() {
        // `thought\n` is the Gemma4 channel label, stripped by the
        // parser so the body round-trips byte-for-byte through the
        // chat template (see `strip_channel_label` for the full rationale).
        let parsed = parse_gemma4_output("<|channel>thought\nREASON\n<channel|>");
        assert_eq!(parsed.text, "");
        assert_eq!(parsed.thinking.as_deref(), Some("REASON"));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_tool_call_simple() {
        let parsed = parse_gemma4_output("<|tool_call>call:bash{command:ls}<tool_call|>");
        assert_eq!(parsed.text, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.status, "ok");
        let args = tc.arguments.as_object().unwrap();
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn parse_tool_call_with_array() {
        let parsed = parse_gemma4_output(
            "<|tool_call>call:edit{path:/a,edits:[{oldText:x,newText:y}]}<tool_call|>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        assert_eq!(tc.name, "edit");
        assert_eq!(tc.status, "ok");
        let args = tc.arguments.as_object().unwrap();
        assert_eq!(args.get("path").and_then(|v| v.as_str()), Some("/a"));
        let edits = args.get("edits").unwrap().as_array().unwrap();
        assert_eq!(edits.len(), 1);
        let edit = edits[0].as_object().unwrap();
        assert_eq!(edit.get("oldText").and_then(|v| v.as_str()), Some("x"));
        assert_eq!(edit.get("newText").and_then(|v| v.as_str()), Some("y"));
    }

    #[test]
    fn parse_channel_and_tool_call() {
        let input = "before<|channel>thought\nREASON\n<channel|>middle<|tool_call>call:bash{command:ls}<tool_call|>after";
        let parsed = parse_gemma4_output(input);
        assert_eq!(parsed.text, "beforemiddleafter");
        // `thought\n` label stripped — see `strip_channel_label`.
        assert_eq!(parsed.thinking.as_deref(), Some("REASON"));
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.status, "ok");
    }

    #[test]
    fn parse_task_description_invariant() {
        // The invariant from the task spec, verbatim. `thinking`
        // carries only the body — the `thought\n` label is stripped to
        // keep re-renders byte-equal (see `strip_channel_label`).
        let parsed = parse_gemma4_output(
            "<|channel>thought\nREASON\n<channel|>message<|tool_call>call:bash{command:ls}<tool_call|>",
        );
        assert_eq!(parsed.text, "message");
        assert_eq!(parsed.thinking.as_deref(), Some("REASON"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        let args = parsed.tool_calls[0].arguments.as_object().unwrap();
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn parse_malformed_tool_call_falls_back_to_raw_string_args() {
        // Truly malformed — closing brace before opening brace in the args.
        let parsed = parse_gemma4_output("<|tool_call>call:bash}garbage<tool_call|>");
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        // Name parses as "bash}garbage" because there's no `{` separator.
        assert!(tc.name.starts_with("bash"));
    }

    #[test]
    fn parse_tool_call_unbalanced_braces_preserves_raw() {
        // No closing brace inside the tool-call region.
        let parsed = parse_gemma4_output("<|tool_call>call:bash{command:ls<tool_call|>");
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.status, "parse_error");
        // Raw arguments string preserved on the `arguments` field.
        assert!(tc.arguments.is_string());
    }

    #[test]
    fn parse_quoted_strings_encoder_compatible() {
        // Values wrapped in `<|"|>...<|"|>` — the encoder's canonical string form.
        let parsed = parse_gemma4_output(
            "<|tool_call>call:weather{location:<|\"|>Paris<|\"|>,units:<|\"|>celsius<|\"|>}<tool_call|>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        let args = parsed.tool_calls[0].arguments.as_object().unwrap();
        assert_eq!(args.get("location").and_then(|v| v.as_str()), Some("Paris"));
        assert_eq!(args.get("units").and_then(|v| v.as_str()), Some("celsius"));
    }

    #[test]
    fn dsl_args_roundtrip_with_encoder() {
        let fixtures = [
            r#"{"command":"ls -R"}"#,
            r#"{"location":"Paris","units":"celsius"}"#,
            r#"{"count":5,"active":true,"meta":null}"#,
            r#"{"path":"/a/b.txt","edits":[{"oldText":"foo","newText":"bar"}]}"#,
            r#"{"tags":["x","y","z"]}"#,
        ];
        for raw in fixtures {
            let dsl = json_args_to_gemma4_dsl_for_test(raw);
            let parsed = parse_gemma4_dsl_args(&dsl)
                .unwrap_or_else(|e| panic!("failed to parse {:?}: {}", dsl, e));
            let original: Value = serde_json::from_str(raw).unwrap();
            assert_eq!(
                parsed, original,
                "roundtrip mismatch for fixture {:?}: DSL={:?}",
                raw, dsl
            );
        }
    }

    #[test]
    fn stream_parser_splits_across_feeds() {
        let mut parser = Gemma4StreamParser::new();
        let mut all = Vec::new();
        all.extend(parser.feed("<|cha"));
        all.extend(parser.feed("nnel>thought\nreas"));
        all.extend(parser.feed("on\n<channel|>plaintext"));
        all.extend(parser.flush());

        // Text segments are everything NOT inside the channel block.
        let text: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Text(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        let reasoning: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Reasoning(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "plaintext");
        // Streaming deltas AND the accumulated `thinking()` both have
        // the `thought\n` channel label stripped — kept in sync so the
        // client-visible body matches the re-render contract byte for
        // byte. See `emit_reasoning` / `strip_channel_label`.
        assert_eq!(reasoning, "reason\n");
        assert_eq!(parser.thinking().as_deref(), Some("reason"));
    }

    #[test]
    fn stream_parser_tool_call_spans_feeds() {
        let mut parser = Gemma4StreamParser::new();
        let mut all = Vec::new();
        all.extend(parser.feed("before<|tool_c"));
        all.extend(parser.feed("all>call:bash{comm"));
        all.extend(parser.feed("and:ls -R}<tool_c"));
        all.extend(parser.feed("all|>after"));
        all.extend(parser.flush());

        let text: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Text(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "beforeafter");

        let tool_calls: Vec<_> = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::ToolCall(tc) = s {
                    Some(tc.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "bash");
        let args = tool_calls[0].arguments.as_object().unwrap();
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls -R"));

        // The aggregated accessor reports the same list.
        assert_eq!(parser.tool_calls().len(), 1);
    }

    #[test]
    fn stream_parser_flush_empties_buffer() {
        // Pure trailing text that was never followed by a marker should
        // show up on flush().
        let mut parser = Gemma4StreamParser::new();
        let feed = parser.feed("hello ");
        // May be empty or may emit partial bytes; both fine — the union of
        // feed+flush is what matters.
        let mut all = feed;
        all.extend(parser.feed("world"));
        all.extend(parser.flush());
        let text: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Text(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn stream_parser_flush_drops_unterminated_tool_call() {
        // An un-closed tool_call block at end-of-stream is malformed.
        // Reconstructing a partial ToolCall with no closing brace would
        // be lossy, so we drop it — the test pins that behavior.
        let mut parser = Gemma4StreamParser::new();
        let _ = parser.feed("prefix<|tool_call>call:bash{command:ls");
        let _ = parser.flush();
        assert!(parser.tool_calls().is_empty());
    }

    /// Label stripping must survive the pathological case where the
    /// model's output gets tokenized into bytes that arrive one-at-a-time
    /// through the streaming decode loop. This exercises the force-buffer
    /// gate inside `drain_until_close` — pending bytes are held back
    /// until the parser has seen enough to confirm or reject the
    /// `thought\n` label, then the first emit strips it.
    ///
    /// Without the force-buffer gate, the first feed could emit a single
    /// `t` as a Reasoning segment (and push it onto `thinking`), the
    /// label-stripper would never match because the `thought\n` bytes
    /// are now split across two different segments, and the accumulated
    /// `thinking` would still carry the label — the exact regression
    /// that zeroed Gemma4's cache reuse under pi-mono.
    /// Exact reproduction of what the Gemma4 tokenizer's `step_decode_stream`
    /// actually emits in production: one feed per generated token. For the
    /// opening of a reasoning block the tokens arrive as `<|channel>`
    /// (one feed), then `thought` (one feed), then `\n` (one feed), then
    /// the body tokens. The force-buffer gate in `drain_until_close`
    /// MUST hold back the 7-byte `thought` chunk until `\n` arrives,
    /// otherwise the label leaks as a standalone `Reasoning('thought')`
    /// delta — exactly what the live log showed.
    #[test]
    fn stream_parser_reproduces_production_token_boundaries() {
        let mut parser = Gemma4StreamParser::new();
        let mut all = Vec::new();
        // Mirror the exact production chunk sequence observed in
        // `.logging-gemma/requests.ndjson` — first delta was `'thought'`
        // (7 bytes, no newline), second was `'\n'`, third was `'The'`.
        for chunk in [
            "<|channel>",
            "thought",
            "\n",
            "The user wants me to upgrade",
            "\n<channel|>",
            "rest",
        ] {
            all.extend(parser.feed(chunk));
        }
        all.extend(parser.flush());
        let reasoning: Vec<String> = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Reasoning(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !reasoning.iter().any(|s| s == "thought"),
            "leaked `thought` label as a standalone delta: {reasoning:?}",
        );
        assert!(
            !reasoning.iter().any(|s| s.starts_with("thought\n")),
            "any delta prefixed with `thought\\n` means the label wasn't stripped: {reasoning:?}",
        );
        assert_eq!(
            parser.thinking().as_deref(),
            Some("The user wants me to upgrade"),
        );
    }

    #[test]
    fn stream_parser_strips_channel_label_split_across_single_byte_feeds() {
        let mut parser = Gemma4StreamParser::new();
        let mut all = Vec::new();
        for ch in "<|channel>thought\nREASON\n<channel|>after".chars() {
            all.extend(parser.feed(&ch.to_string()));
        }
        all.extend(parser.flush());

        let reasoning: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Reasoning(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        // Streaming deltas must not contain the `thought\n` label — if
        // they did, the server would accumulate it into the outgoing
        // reasoning_content and the next turn's template re-render
        // would double it.
        assert!(
            !reasoning.starts_with("thought\n"),
            "streaming reasoning must not leak the `thought\\n` channel label: {reasoning:?}",
        );
        assert!(
            !reasoning.contains("thought\nthought"),
            "no doubled label should ever slip through: {reasoning:?}",
        );
        assert_eq!(parser.thinking().as_deref(), Some("REASON"));
    }

    /// Partial channel block where the body never even reaches the full
    /// label length before the close marker lands. The stripper must
    /// fall through without mangling short bodies (e.g. a malformed
    /// model emission like `<|channel>tho<channel|>`).
    #[test]
    fn stream_parser_passes_through_channel_body_shorter_than_label() {
        let mut parser = Gemma4StreamParser::new();
        let mut all = parser.feed("<|channel>tho<channel|>next");
        all.extend(parser.flush());
        let reasoning: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Reasoning(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            reasoning, "tho",
            "short body that isn't the full label passes through unchanged",
        );
        assert_eq!(parser.thinking().as_deref(), Some("tho"));
    }

    #[test]
    fn stream_parser_flush_surfaces_unterminated_channel_as_reasoning() {
        // "half-written thought" doesn't start with `thought\n`, so the
        // label stripper passes it through unchanged — the content is
        // exactly what the model emitted.
        let mut parser = Gemma4StreamParser::new();
        let mut all = parser.feed("<|channel>half-written thought");
        all.extend(parser.flush());
        let reasoning: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Reasoning(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(reasoning, "half-written thought");
        assert_eq!(parser.thinking().as_deref(), Some("half-written thought"));
    }

    #[test]
    fn stream_parser_swallows_tool_response_block_defensively() {
        // These markers are input-only, but if the model ever emits one,
        // the content must not leak into the user-visible text.
        let mut parser = Gemma4StreamParser::new();
        let mut all = parser.feed("keep<|tool_response>drop<tool_response|>keep2");
        all.extend(parser.flush());
        let text: String = all
            .iter()
            .filter_map(|s| {
                if let StreamSegment::Text(t) = s {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "keepkeep2");
    }

    #[test]
    fn parse_dsl_args_numbers_and_bools() {
        let v = parse_gemma4_dsl_args("count:5,active:true,flag:false,empty:null").unwrap();
        let m = v.as_object().unwrap();
        assert_eq!(m.get("count").and_then(|v| v.as_i64()), Some(5));
        assert_eq!(m.get("active").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(m.get("flag").and_then(|v| v.as_bool()), Some(false));
        assert!(m.get("empty").unwrap().is_null());
    }

    #[test]
    fn parse_dsl_args_with_surrounding_braces() {
        let v = parse_gemma4_dsl_args("{command:ls}").unwrap();
        let m = v.as_object().unwrap();
        assert_eq!(m.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn parse_dsl_args_empty_object() {
        let v = parse_gemma4_dsl_args("").unwrap();
        assert!(v.as_object().unwrap().is_empty());
        let v = parse_gemma4_dsl_args("{}").unwrap();
        assert!(v.as_object().unwrap().is_empty());
    }

    /// Regression test for a panic observed in a live Gemma4 session:
    ///   `start byte index 281 is not a char boundary; it is inside '—'`
    /// The model emitted a tool-call whose DSL-string value contained an
    /// em-dash (`—`, 3 bytes in UTF-8). The previous `extract_balanced_braces`
    /// walked byte-by-byte inside a `<|"|>...<|"|>` payload and then tried to
    /// `&str`-slice at `s[i..].starts_with(...)`, landing mid-char and
    /// panicking. After the fix the call should parse (or at minimum the
    /// stream parser should not abort).
    #[test]
    fn stream_parser_survives_multibyte_inside_dsl_string() {
        let mut p = Gemma4StreamParser::new();
        let chunk = concat!(
            "before ",
            "<|tool_call>",
            "call:bash{command:<|\"|># I don't have firecrawl-scrape as a tool",
            " — wait, let me check the skills.\n",
            "echo 'done'<|\"|>}",
            "<tool_call|>",
            " after",
        );
        // Must not panic. Also must recover the enclosing text around the
        // tool call block.
        let _segs = p.feed(chunk);
        let _ = p.flush();
        let calls = p.tool_calls();
        assert_eq!(calls.len(), 1, "should parse exactly one tool call");
        assert_eq!(calls[0].name, "bash");
        // The command string is preserved whole (em-dash included).
        let cmd = calls[0]
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            cmd.contains('—'),
            "em-dash must survive the round trip, got: {cmd:?}",
        );
    }

    /// Direct unit test on `extract_balanced_braces` — the original panic
    /// site — to pin the char-boundary safety in place going forward.
    #[test]
    fn extract_balanced_braces_handles_multibyte_inside_dsl_string() {
        let inner_body = "<|\"|>foo — bar<|\"|>";
        let s = format!("{{key:{}}}", inner_body);
        let extracted = extract_balanced_braces(&s).expect("must not panic and must extract");
        // The inner (brace-stripped) body should include both the key, the
        // DSL delimiters, and the em-dash-bearing payload verbatim.
        assert!(
            extracted.contains("<|\"|>foo — bar<|\"|>"),
            "got: {extracted:?}"
        );
        assert!(extracted.starts_with("key:"));
    }
}
