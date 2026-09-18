//! Tool call parsing utilities
//!
//! Extracts structured tool calls from model-generated text.
//! Supports JSON format (Qwen3), function/parameter format (Qwen3.5), XML format (legacy),
//! and the pythonic sentinel format (LFM2/LFM2.5):
//! `<|tool_call_start|>[func(arg='v')]<|tool_call_end|>`.
//!
//! Uses simple string-based parsers instead of regex for clarity and debuggability.

use napi_derive::napi;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

/// Structured tool call with parsed arguments
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallResult {
    /// Unique identifier for this tool call (format: call_<uuid>)
    pub id: String,
    /// Name of the tool/function to call
    pub name: String,
    /// Parsed arguments as native object (serde_json::Value -> JS object)
    ///
    /// When status is "ok", this contains the parsed arguments object.
    /// When status is "parse_error", this contains the original unparsed string.
    /// Otherwise, this is an empty object {}.
    #[napi(ts_type = "Record<string, unknown> | string")]
    pub arguments: Value,
    /// Parsing status: "ok" | "invalid_json" | "missing_name" | "parse_error"
    ///
    /// - "ok": Successfully parsed tool call
    /// - "invalid_json": The tool_call tag content was not valid JSON
    /// - "missing_name": Valid JSON but no "name" field
    /// - "parse_error": Valid JSON but the "arguments" string field couldn't be parsed as JSON
    pub status: String,
    /// Error message if status != "ok"
    pub error: Option<String>,
    /// Raw content from the tool-call markup (preserved for debugging/persistence):
    /// the `<tool_call>…</tool_call>` block, or for LFM2 the whole
    /// `<|tool_call_start|>…<|tool_call_end|>` sentinel block.
    /// Defaults to empty string for backward compatibility with older JSON
    #[serde(default)]
    pub raw_content: String,
}

impl ToolCallResult {
    /// Create a successful tool call result
    pub fn ok(name: String, arguments: Value, raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments,
            status: "ok".to_string(),
            error: None,
            raw_content,
        }
    }

    /// Create a tool call result with invalid JSON arguments
    pub fn invalid_json(name: String, error_msg: String, raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments: Value::Object(serde_json::Map::new()),
            status: "invalid_json".to_string(),
            error: Some(error_msg),
            raw_content,
        }
    }

    /// Create a tool call result where the arguments string failed to parse
    ///
    /// This is distinct from `invalid_json` - it means the outer tool call JSON was valid,
    /// but the arguments field contained a string that couldn't be parsed as JSON.
    pub fn parse_error(
        name: String,
        raw_arguments: String,
        error_msg: String,
        raw_content: String,
    ) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            // Store the original string in arguments as a fallback
            arguments: Value::String(raw_arguments),
            status: "parse_error".to_string(),
            error: Some(error_msg),
            raw_content,
        }
    }

    /// Create a tool call result with missing name
    pub fn missing_name(raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name: String::new(),
            arguments: Value::Object(serde_json::Map::new()),
            status: "missing_name".to_string(),
            error: Some(format!("Tool call missing name: {raw_content}")),
            raw_content,
        }
    }
}

/// Generate a unique tool call ID in OpenAI format: call_<uuid>
fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Tag extraction helpers — replace all regex with simple string scanning
// ---------------------------------------------------------------------------

/// Extract all blocks between `<open_tag>` and `</close_tag>`.
/// Returns Vec of (start_of_open_tag, end_of_close_tag, inner_content).
fn extract_tag_blocks<'a>(
    text: &'a str,
    open_tag: &str,
    close_tag: &str,
) -> Vec<(usize, usize, &'a str)> {
    let mut results = Vec::new();
    let mut search_from = 0;

    while search_from < text.len() {
        let Some(open_start) = text[search_from..].find(open_tag) else {
            break;
        };
        let open_start = search_from + open_start;
        let content_start = open_start + open_tag.len();

        let Some(close_start) = text[content_start..].find(close_tag) else {
            break;
        };
        let close_start = content_start + close_start;
        let close_end = close_start + close_tag.len();

        let inner = &text[content_start..close_start];
        results.push((open_start, close_end, inner));

        search_from = close_end;
    }

    results
}

/// Remove all occurrences of `<open_tag>...</close_tag>` from text.
fn strip_tag_blocks(text: &str, open_tag: &str, close_tag: &str) -> String {
    let blocks = extract_tag_blocks(text, open_tag, close_tag);
    if blocks.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (start, end, _) in &blocks {
        result.push_str(&text[last_end..*start]);
        last_end = *end;
    }
    result.push_str(&text[last_end..]);
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// JSON sanitizer (for LLM-generated JSON with raw control characters)
// ---------------------------------------------------------------------------

/// Sanitize JSON string by escaping raw control characters inside string values.
///
/// LLMs often generate JSON with raw newlines inside strings for readability.
/// This function escapes control characters (`\u0000-\u001F`) found inside
/// quoted string values so that standard JSON parsers can handle them.
fn sanitize_json_string(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 64);
    let mut in_string = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if in_string {
            if c == '\\' {
                // Escaped character - copy the backslash and the next char as-is
                result.push(c);
                if let Some(next) = chars.next() {
                    result.push(next);
                }
            } else if c == '"' {
                // End of string
                in_string = false;
                result.push(c);
            } else if c.is_ascii_control() {
                // Control character inside string - escape it
                match c {
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    '\x08' => result.push_str("\\b"),
                    '\x0C' => result.push_str("\\f"),
                    _ => {
                        // Other control characters as \uXXXX
                        result.push_str(&format!("\\u{:04x}", c as u32));
                    }
                }
            } else {
                result.push(c);
            }
        } else {
            // Not in a string
            if c == '"' {
                in_string = true;
            }
            result.push(c);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Individual format parsers
// ---------------------------------------------------------------------------

/// Parse a JSON format tool call (Qwen3)
///
/// Format: `{"name": "func", "arguments": {...}}`
fn parse_json_tool_call(json_str: &str, raw_content: &str) -> ToolCallResult {
    let sanitized = sanitize_json_string(json_str);
    match serde_json::from_str::<Value>(&sanitized) {
        Ok(parsed) => {
            let name = parsed
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            match name {
                Some(name) if !name.is_empty() => {
                    let arguments = parsed
                        .get("arguments")
                        .cloned()
                        .unwrap_or(Value::Object(serde_json::Map::new()));

                    // If arguments is a string, try to parse it as JSON
                    match &arguments {
                        Value::String(s) => match serde_json::from_str::<Value>(s) {
                            Ok(parsed_args) => {
                                ToolCallResult::ok(name, parsed_args, raw_content.to_string())
                            }
                            Err(e) => ToolCallResult::parse_error(
                                name,
                                s.clone(),
                                format!("Failed to parse arguments string as JSON: {}", e),
                                raw_content.to_string(),
                            ),
                        },
                        _ => ToolCallResult::ok(name, arguments, raw_content.to_string()),
                    }
                }
                _ => ToolCallResult::missing_name(raw_content.to_string()),
            }
        }
        Err(e) => ToolCallResult::invalid_json(
            String::new(),
            format!("Invalid JSON: {}", e),
            raw_content.to_string(),
        ),
    }
}

/// Parse a Qwen3.5/Qwen3-Coder function-parameter format tool call
///
/// Format: `<function=func_name>\n<parameter=key>\nvalue\n</parameter>\n</function>`
fn parse_function_tool_call(inner: &str, raw_content: &str) -> ToolCallResult {
    // inner is everything between <tool_call> and </tool_call>, e.g.:
    // "\n<function=fetch_url>\n<parameter=url>\nhttps://...\n</parameter>\n</function>\n"
    let inner = inner.trim();

    // Extract <function=NAME>...</function>
    let Some(func_start) = inner.find("<function=") else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };
    let after_prefix = &inner[func_start + "<function=".len()..];

    let Some(name_end) = after_prefix.find('>') else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };
    let function_name = after_prefix[..name_end].trim().to_string();
    if function_name.is_empty() {
        return ToolCallResult::missing_name(raw_content.to_string());
    }

    // Find </function> to get parameter section
    let params_start = name_end + 1;
    let params_section = if let Some(func_end) = after_prefix[params_start..].find("</function>") {
        &after_prefix[params_start..params_start + func_end]
    } else {
        &after_prefix[params_start..]
    };

    // Extract all <parameter=key>value</parameter> pairs
    let mut param_map = serde_json::Map::new();
    for (_, _, param_inner) in extract_tag_blocks(params_section, "<parameter=", "</parameter>") {
        // param_inner is "key>\nvalue"
        if let Some(idx) = param_inner.find('>') {
            let param_name = param_inner[..idx].trim();
            let mut param_value = &param_inner[idx + 1..];

            // Strip leading/trailing newlines (matches Python mlx-lm behavior)
            if param_value.starts_with('\n') {
                param_value = &param_value[1..];
            }
            if param_value.ends_with('\n') {
                param_value = &param_value[..param_value.len() - 1];
            }

            // Qwen3.5+ chat template emits non-string argument values via `| tojson`,
            // so arrays/objects land here as raw JSON text inside the <parameter> block.
            // Parse them back to Value::Array / Value::Object so the schema on the
            // consumer side (e.g. pi's `edit` tool expecting `edits: array`) validates.
            //
            // We only parse when the value STARTS with `[` or `{`. The template emits
            // string-typed args as bare text (no quotes), so `"5"` and `5` are
            // indistinguishable at this layer; treating bare values as strings is the
            // safe choice — schema consumers already know how to coerce.
            let trimmed = param_value.trim();
            let parsed_value = match trimmed.chars().next() {
                Some('[') | Some('{') => serde_json::from_str::<Value>(trimmed)
                    .unwrap_or_else(|_| Value::String(param_value.to_string())),
                _ => Value::String(param_value.to_string()),
            };
            param_map.insert(param_name.to_string(), parsed_value);
        }
    }

    ToolCallResult::ok(
        function_name,
        Value::Object(param_map),
        raw_content.to_string(),
    )
}

/// Parse an XML format tool call (legacy/training)
///
/// Format: `<name>func</name><arguments>{...}</arguments>`
fn parse_xml_tool_call(inner: &str, raw_content: &str) -> ToolCallResult {
    // Extract <name>...</name>
    let name_blocks = extract_tag_blocks(inner, "<name>", "</name>");
    let Some((_, _, name_content)) = name_blocks.first() else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };

    let name = name_content.trim().to_string();
    if name.is_empty() {
        return ToolCallResult::missing_name(raw_content.to_string());
    }

    // Extract <arguments>...</arguments> (optional)
    let args_blocks = extract_tag_blocks(inner, "<arguments>", "</arguments>");
    match args_blocks.first() {
        Some((_, _, args_content)) => {
            let args_str = args_content.trim();
            if args_str.is_empty() {
                ToolCallResult::ok(
                    name,
                    Value::Object(serde_json::Map::new()),
                    raw_content.to_string(),
                )
            } else {
                match serde_json::from_str::<Value>(args_str) {
                    Ok(args) => ToolCallResult::ok(name, args, raw_content.to_string()),
                    Err(e) => ToolCallResult::invalid_json(
                        name,
                        format!("Invalid arguments JSON: {}", e),
                        raw_content.to_string(),
                    ),
                }
            }
        }
        None => ToolCallResult::ok(
            name,
            Value::Object(serde_json::Map::new()),
            raw_content.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Detect which format a <tool_call> block uses
// ---------------------------------------------------------------------------

/// Determine the format of tool call content and parse accordingly.
fn classify_and_parse_tool_call(inner: &str, raw_content: &str) -> Option<ToolCallResult> {
    let trimmed = inner.trim();

    // JSON format (Qwen3): starts with `{`
    if trimmed.starts_with('{') {
        return Some(parse_json_tool_call(trimmed, raw_content));
    }

    // Function format (Qwen3.5): contains `<function=`
    if trimmed.contains("<function=") {
        return Some(parse_function_tool_call(inner, raw_content));
    }

    // XML format (legacy): contains `<name>`
    if trimmed.contains("<name>") {
        return Some(parse_xml_tool_call(inner, raw_content));
    }

    // Unrecognized content — not a tool call
    None
}

// ---------------------------------------------------------------------------
// LFM2 pythonic tool calls — <|tool_call_start|>[fn(kw=literal)]<|tool_call_end|>
// ---------------------------------------------------------------------------
//
// Mirrors vLLM `tool_parsers/lfm2_tool_parser.py::Lfm2ToolParser`
// (`extract_tool_calls`, non-streaming) semantics:
//   - Only the FIRST sentinel block is parsed; LFM2 frequently re-emits the
//     call body after the first `<|tool_call_end|>` capped by a second end
//     sentinel, so everything through the LAST orphan end is dropped
//     (`_strip_echo`). A second real sentinel block is indistinguishable
//     from that echo and is dropped the same way — LFM2 packs parallel
//     calls into ONE bracket list, so nothing real is lost.
//   - The call body must be a non-empty bracketed list whose every element
//     is a function call `name(kw=literal, ...)`. Dotted names
//     (`a.b.c(...)`) are preserved verbatim.
//   - Keyword args only: positionals are ignored, `**kwargs` rejects the
//     block (vLLM `handle_single_tool` iterates `call.keywords` only and
//     `arguments[None]` fails `json.dumps`).
//   - Argument values are Python literals: strings (single/double/triple
//     quotes, `r`/`u`/`f` prefixes, escapes; `f` rejected when it carries a
//     `{...}` placeholder), numbers (dec/hex/oct/bin ints, floats, unary
//     +/-), True/False/None plus the JSON spellings true/false/null, lists,
//     tuples and sets (→ arrays), dicts with literal keys (non-string keys
//     are stringified like `json.dumps`). Anything else rejects the block.
//   - On ANY parse failure the raw text is returned verbatim as content
//     (vLLM returns `content=model_output`, `tools_called=False`) — a
//     malformed call is model output the user should see, not markup to
//     silently delete.
//
// The sentinels are non-special added tokens in the LFM2 tokenizers, so
// `skip_special_tokens=true` does NOT strip them — this parser sees them
// in the default decode with no flag changes needed.

const LFM2_TOOL_CALL_START: &str = "<|tool_call_start|>";
const LFM2_TOOL_CALL_END: &str = "<|tool_call_end|>";

/// vLLM `_strip_echo`: drop everything through the last orphan
/// `<|tool_call_end|>` in the post-call text — the echoed body is capped by
/// a second end sentinel, so the last end marks where real content resumes.
fn strip_lfm2_echo(raw_after: &str) -> &str {
    match raw_after.rfind(LFM2_TOOL_CALL_END) {
        Some(idx) => &raw_after[idx + LFM2_TOOL_CALL_END.len()..],
        None => raw_after,
    }
}

/// Recursive-descent parser for the pythonic call list. Byte-indexed
/// scanner: every syntax character we look for is ASCII, so UTF-8 string
/// contents pass through safely.
struct PyLiteralParser<'a> {
    s: &'a [u8],
    pos: usize,
    /// `parse_literal` nesting depth — containers and unary ops recurse, so
    /// pathological input (`[[[[…`, `x=----…1`) must fail closed instead of
    /// overflowing the stack (vLLM's `ast.parse` raises `RecursionError`,
    /// which its caller catches → raw text).
    depth: u32,
}

/// Hard cap on literal nesting — far beyond any real tool argument.
const MAX_PY_LITERAL_DEPTH: u32 = 64;

/// Python reserved words that can never be a name — a call segment
/// (`for(x)`), a comprehension target (`for not in y`), or a lambda
/// parameter (`lambda True: x`) is a SyntaxError, and `None(x)` parses to
/// a `Constant` with no `.func.id` under vLLM's AST extraction. Soft
/// keywords (`match`, `case`, `type`) stay usable as names.
fn reserved_ident(id: &[u8]) -> bool {
    matches!(
        id,
        b"and"
            | b"as"
            | b"assert"
            | b"async"
            | b"await"
            | b"break"
            | b"class"
            | b"continue"
            | b"def"
            | b"del"
            | b"elif"
            | b"else"
            | b"except"
            | b"finally"
            | b"for"
            | b"from"
            | b"global"
            | b"if"
            | b"import"
            | b"in"
            | b"is"
            | b"lambda"
            | b"nonlocal"
            | b"not"
            | b"or"
            | b"pass"
            | b"raise"
            | b"return"
            | b"try"
            | b"while"
            | b"with"
            | b"yield"
            | b"True"
            | b"False"
            | b"None"
    )
}

impl<'a> PyLiteralParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            s: text.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ()> {
        self.skip_ws();
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(())
        }
    }

    fn eof(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.s.len()
    }

    fn ident(&mut self) -> Result<&'a str, ()> {
        self.skip_ws();
        let start = self.pos;
        self.pos = ident_end(self.s, self.pos);
        if start == self.pos || !ident_char_at(self.s, start, true) {
            return Err(());
        }
        std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| ())
    }

    /// `name` or `a.b.c` — dotted attribute chains are preserved.
    fn dotted_name(&mut self) -> Result<String, ()> {
        let first = self.ident()?;
        if reserved_ident(first.as_bytes()) {
            return Err(());
        }
        let mut name: String = first.nfkc().collect();
        loop {
            self.skip_ws();
            if self.peek() != Some(b'.') {
                break;
            }
            self.pos += 1;
            let seg = self.ident()?;
            if reserved_ident(seg.as_bytes()) {
                return Err(());
            }
            name.push('.');
            name.extend(seg.nfkc());
        }
        Ok(name)
    }

    /// Skip a quoted string starting at `pos` (single or triple quoted);
    /// backslash escapes the next byte. Unterminated → Err.
    fn skip_quoted(&mut self, f_mode: bool, bytes_mode: bool, raw_mode: bool) -> Result<(), ()> {
        let quote = self.s[self.pos];
        self.pos += 1;
        let triple = self.peek() == Some(quote) && self.s.get(self.pos + 1) == Some(&quote);
        if triple {
            self.pos += 2;
        }
        while let Some(&b) = self.s.get(self.pos) {
            if bytes_mode && b >= 0x80 {
                return Err(());
            }
            if f_mode && b == b'{' {
                if self.s.get(self.pos + 1) == Some(&b) {
                    self.pos += 2;
                } else {
                    self.pos += 1;
                    self.skip_f_replacement(1)?;
                }
                continue;
            }
            if f_mode && b == b'}' {
                if self.s.get(self.pos + 1) != Some(&b) {
                    return Err(());
                }
                self.pos += 2;
                continue;
            }
            if b == b'\\' {
                if !raw_mode {
                    // Mandatory escapes must be well-formed or `ast.parse`
                    // raises and vLLM keeps the whole block verbatim:
                    // `\x` needs 2 hex digits, `\u`/`\U` need 4/8, and `\N`
                    // is rejected outright (see below). `\u`/`\U`/`\N` are
                    // NOT escapes in bytes literals, so they stay literal
                    // there (only `\x` is checked).
                    let escape = self.s.get(self.pos + 1).copied();
                    let hex_run = |start: usize, len: usize| {
                        (0..len)
                            .all(|i| self.s.get(start + i).is_some_and(|c| c.is_ascii_hexdigit()))
                    };
                    let hex_value = |start: usize, len: usize| {
                        (0..len).fold(0u32, |acc, i| {
                            acc * 16 + (self.s[start + i] as char).to_digit(16).unwrap_or(0)
                        })
                    };
                    match escape {
                        Some(b'x') if !hex_run(self.pos + 2, 2) => return Err(()),
                        Some(b'u') if !bytes_mode && !hex_run(self.pos + 2, 4) => {
                            return Err(());
                        }
                        // `\U` is also range-checked: past 0x10FFFF is
                        // "illegal Unicode character" (surrogates are fine).
                        Some(b'U')
                            if !bytes_mode
                                && (!hex_run(self.pos + 2, 8)
                                    || hex_value(self.pos + 2, 8) > 0x10FFFF) =>
                        {
                            return Err(());
                        }
                        // `\N` is a recognized escape introducer in str: a
                        // bare `\N` is a hard SyntaxError, and `\N{NAME}`
                        // needs a Unicode-name table we don't carry, so an
                        // unknown name cannot be told from a known one.
                        // Reject both rather than promote a call whose
                        // `ast.parse` failed — the same reject-on-
                        // non-representable stance as keyword `\N` and
                        // non-JSON literals. `b'\N'` is only a warning, so
                        // bytes skip this.
                        Some(b'N') if !bytes_mode => return Err(()),
                        _ => {}
                    }
                }
                self.pos += if f_mode && matches!(self.s.get(self.pos + 1), Some(b'{' | b'}')) {
                    1
                } else {
                    2
                };
                continue;
            }
            if b == quote {
                if triple {
                    if self.s.get(self.pos + 1) == Some(&quote)
                        && self.s.get(self.pos + 2) == Some(&quote)
                    {
                        self.pos += 3;
                        return Ok(());
                    }
                    self.pos += 1;
                    continue;
                }
                self.pos += 1;
                return Ok(());
            }
            self.pos += 1;
        }
        Err(())
    }

    fn validate_f_expression(&self, expression: &[u8]) -> Result<(), ()> {
        let expression = std::str::from_utf8(expression).map_err(|_| ())?.trim();
        if expression.is_empty() {
            return Err(());
        }
        let wrapped = format!("({expression}),");
        let mut parser = PyLiteralParser::new(&wrapped);
        parser.depth = self.depth;
        parser.skip_expression(false)?;
        parser.expect(b',')?;
        if parser.eof() { Ok(()) } else { Err(()) }
    }

    fn skip_f_replacement(&mut self, nesting: u32) -> Result<(), ()> {
        if nesting > 3 {
            return Err(());
        }
        let start = self.pos;
        let mut stack = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(());
            };
            match b {
                b'\'' | b'"' => {
                    let prefix_end = self.pos;
                    let mut prefix_start = prefix_end;
                    while prefix_start > start
                        && prefix_end - prefix_start < 2
                        && matches!(
                            self.s[prefix_start - 1],
                            b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B'
                        )
                    {
                        prefix_start -= 1;
                    }
                    let prefix = &self.s[prefix_start..prefix_end];
                    let valid_prefix = match prefix.len() {
                        0 => true,
                        1 => matches!(prefix[0].to_ascii_lowercase(), b'r' | b'u' | b'b' | b'f'),
                        2 => matches!(
                            (
                                prefix[0].to_ascii_lowercase(),
                                prefix[1].to_ascii_lowercase()
                            ),
                            (b'r', b'b') | (b'b', b'r') | (b'r', b'f') | (b'f', b'r')
                        ),
                        _ => false,
                    };
                    self.skip_quoted(
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'f')),
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'b')),
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'r')),
                    )?;
                }
                b'(' | b'[' | b'{' => {
                    stack.push(b);
                    self.pos += 1;
                }
                b')' | b']' | b'}' => {
                    if b == b'}' && stack.is_empty() {
                        break;
                    }
                    let Some(open) = stack.pop() else {
                        return Err(());
                    };
                    if !matches!((open, b), (b'(', b')') | (b'[', b']') | (b'{', b'}')) {
                        return Err(());
                    }
                    self.pos += 1;
                }
                b'!' if stack.is_empty() && self.s.get(self.pos + 1) != Some(&b'=') => break,
                b':' if stack.is_empty() => break,
                b'=' if stack.is_empty()
                    && self.s.get(self.pos + 1) != Some(&b'=')
                    && (self.pos == start
                        || !matches!(self.s[self.pos - 1], b'!' | b'<' | b'>' | b'=')) =>
                {
                    break;
                }
                _ => {
                    let len = utf8_len(b);
                    if self.pos + len > self.s.len() {
                        return Err(());
                    }
                    self.pos += len;
                }
            }
        }
        let end = self.pos;
        self.validate_f_expression(&self.s[start..end])?;
        if self.peek() == Some(b'=') {
            self.pos += 1;
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }
        if self.peek() == Some(b'!') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b's' | b'r' | b'a')) {
                return Err(());
            }
            self.pos += 1;
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }
        if self.peek() == Some(b':') {
            self.pos += 1;
            self.skip_f_format_spec(nesting)?;
        }
        if self.peek() != Some(b'}') {
            return Err(());
        }
        self.pos += 1;
        Ok(())
    }

    fn skip_f_format_spec(&mut self, nesting: u32) -> Result<(), ()> {
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(());
            };
            if b == b'}' {
                return Ok(());
            }
            if b == b'{' {
                self.pos += 1;
                self.skip_f_replacement(nesting + 1)?;
            } else {
                let len = utf8_len(b);
                if self.pos + len > self.s.len() {
                    return Err(());
                }
                self.pos += len;
            }
        }
    }

    /// Skip one positional argument's expression text. vLLM ignores
    /// `call.args` entirely, so the content is dropped — but the skipped
    /// text must still be a plausible Python expression: anything
    /// `ast.parse` would `SyntaxError` (`@@@`, `1 +`, `a b`, `f(,)`,
    /// unbalanced closers, bare `=`/`!`) rejects the WHOLE block verbatim
    /// instead of promoting an `ok` call built from the remaining kwargs.
    ///
    /// The scan is a small operand/operator automaton, not a full grammar:
    /// `Need` requires a value (identifier, literal, number, string,
    /// container opener, unary `- + ~ not`, `*` spread); `Have` requires a
    /// binary operator, a postfix (`(` `[` `.`), or the top-level `,`/`)`
    /// argument boundary. `Lambda` covers `lambda …:` parameter lists.
    /// Deliberate residuals (rare invalid forms still accepted, e.g.
    /// `a ~ b`, `0xg`): skipping is a boundary check, not codegen — the
    /// goal is keeping obviously-malformed text from promoting a call.
    /// `starred` marks the expression as an already-consumed `*a`
    /// unpacking — a `for` at the top level then makes it a comp
    /// element (`*a for a in b` is a SyntaxError).
    fn skip_expression(&mut self, starred: bool) -> Result<(), ()> {
        self.skip_expression_until(starred, false)
    }

    fn skip_expression_until(&mut self, starred: bool, lambda_default: bool) -> Result<(), ()> {
        self.depth += 1;
        if self.depth > MAX_PY_LITERAL_DEPTH {
            self.depth -= 1;
            return Err(());
        }
        let result = self.skip_expression_inner(starred, lambda_default);
        self.depth -= 1;
        result
    }

    fn skip_expression_inner(&mut self, starred: bool, lambda_default: bool) -> Result<(), ()> {
        #[derive(Clone, Copy, PartialEq)]
        enum St {
            Need,
            Have,
            Lambda,
        }
        #[derive(Clone, Copy, PartialEq)]
        enum NeedCtx {
            Expr,
            Inversion,
            Restricted,
        }
        #[derive(Clone, Copy, PartialEq)]
        enum LambdaMarker {
            None,
            Star,
            DoubleStar,
            VarargsDone,
            KwargsDone,
            Slash,
        }
        // Identifiers that can never appear where an operand is expected.
        fn reject_keyword(id: &[u8]) -> bool {
            matches!(
                id,
                b"and"
                    | b"or"
                    | b"in"
                    | b"is"
                    | b"if"
                    | b"else"
                    | b"for"
                    | b"elif"
                    | b"while"
                    | b"return"
                    | b"def"
                    | b"class"
                    | b"import"
                    | b"from"
                    | b"as"
                    | b"with"
                    | b"try"
                    | b"except"
                    | b"finally"
                    | b"raise"
                    | b"pass"
                    | b"break"
                    | b"continue"
                    | b"global"
                    | b"nonlocal"
                    | b"del"
                    | b"assert"
                    | b"yield"
                    | b"await"
                    | b"async"
            )
        }
        // Python string-literal prefixes: r/u/f/b alone, or the r-pairs
        // rb/br/rf/fr — anything else (`rr`, `uf`, `bu` …) is a plain
        // identifier, so a glued quote after it is a SyntaxError.
        fn string_prefix(id: &[u8]) -> bool {
            match id.len() {
                1 => matches!(id[0].to_ascii_lowercase(), b'r' | b'u' | b'b' | b'f'),
                2 => matches!(
                    (id[0].to_ascii_lowercase(), id[1].to_ascii_lowercase()),
                    (b'r', b'b') | (b'b', b'r') | (b'r', b'f') | (b'f', b'r')
                ),
                _ => false,
            }
        }
        let mut st = St::Need;
        let mut need_ctx = NeedCtx::Expr;
        let mut stack: Vec<u8> = Vec::new(); // open brackets — closers must match
        // A postfix `x[…]` opens a subscript — the only bracket where `:`
        // is slice syntax (`x[1:2]`, `x[:]`, `x[::]`). `:` inside a list
        // display, tuple, or call is a SyntaxError; inside `{}` it is a
        // dict separator that requires a value.
        const SUBSCRIPT: u8 = 0;
        // A postfix `f(` opens a call — keyword `x=1` arguments are legal
        // inside it but not inside a grouping `(`, list display, dict, or
        // subscript.
        const CALLPAREN: u8 = 1;
        // `{}` resolves to a set display on its first bare element (`,` /
        // walrus / `for`) and to a dict on a key `:` — mixing the two
        // (`{1, 2:3}`, `{1:2, 3}`) is a SyntaxError.
        const SET: u8 = 2;
        const DICT: u8 = 3;
        fn close_match(o: u8, b: u8) -> bool {
            match b {
                b')' => matches!(o, b'(' | CALLPAREN),
                b']' => matches!(o, b'[' | SUBSCRIPT),
                b'}' => matches!(o, b'{' | SET | DICT),
                _ => false,
            }
        }
        fn enter_target_receiver(
            depth: usize,
            island_stack: &mut Vec<usize>,
            target_group_depths: &mut Vec<usize>,
            target_star_depths: &mut Vec<usize>,
            target_comma_depths: &mut Vec<usize>,
            target_unassignable_depths: &mut Vec<usize>,
            target_receiver_depths: &mut Vec<usize>,
        ) {
            target_group_depths.retain(|&d| d != depth);
            target_star_depths.retain(|&d| d != depth);
            target_comma_depths.retain(|&d| d != depth);
            target_unassignable_depths.retain(|&d| d != depth);
            island_stack.push(depth - 1);
            target_receiver_depths.push(depth);
        }
        fn enter_direct_target_receiver(
            depth: usize,
            island_stack: &mut Vec<usize>,
            target_star_depths: &mut Vec<usize>,
            target_comma_depths: &mut Vec<usize>,
            target_unassignable_depths: &mut Vec<usize>,
            target_direct_receiver_depths: &mut Vec<usize>,
        ) {
            target_star_depths.retain(|&d| d != depth);
            target_comma_depths.retain(|&d| d != depth);
            target_unassignable_depths.retain(|&d| d != depth);
            island_stack.push(depth);
            target_direct_receiver_depths.push(depth);
        }
        // `not` consumed as an operator (`x not in y`): `in`/`is` may follow.
        let mut after_not_op = false;
        // Ternary `x if y else z` needs its `else`: `a if b` is a
        // SyntaxError. Comprehension `if`s (`[x for i in y if z]`) are
        // filters — no `else` — tracked per bracket depth so nested
        // comprehensions and outer clauses survive inner closes.
        let mut pending_ifs: Vec<usize> = Vec::new();
        let mut comp_depths: Vec<usize> = Vec::new();
        // Every `for` is a comprehension clause that owes an `in` at the
        // same bracket depth (`[x for x]` is a SyntaxError). An `in`
        // inside deeper brackets belongs to the target or iterable
        // expression and must not satisfy it.
        let mut for_depths: Vec<usize> = Vec::new();
        // Subscript brackets opened while a `for` target is open
        // (`for a[x + 1] in` — a subscript value is a real expression).
        // Recorded as the stack depth at which they were pushed; inside
        // them the target grammar does not apply.
        let mut island_stack: Vec<usize> = Vec::new();
        let mut target_group_depths: Vec<usize> = Vec::new();
        let mut target_star_depths: Vec<usize> = Vec::new();
        let mut target_comma_depths: Vec<usize> = Vec::new();
        let mut target_unassignable_depths: Vec<usize> = Vec::new();
        let mut target_receiver_depths: Vec<usize> = Vec::new();
        let mut target_direct_receiver_depths: Vec<usize> = Vec::new();
        // A `for` at top level is a genexpr — valid only as the sole
        // argument, so a `,` boundary after it is a SyntaxError.
        let mut top_genexpr = false;
        // Lambda parameter tracking: at a param start (entry / after
        // `,`) only a name, `*`/`/` markers, `:` (zero params), or `,`
        // after a marker are legal — `lambda 1: x`, `lambda 'a': x`,
        // `lambda ,: x`, `lambda =x: y` are all SyntaxErrors. Default
        // expressions after `=` are scanned loosely like the rest.
        let mut param_start = false;
        let mut lambda_marker = LambdaMarker::None;
        // A completed param name admits only `,` `=` or the ending `:` —
        // `lambda a b` / `lambda a.b` / `lambda a(` are SyntaxErrors.
        let mut param_done = false;
        let mut lambda_depth = 0;
        let mut lambda_default_seen = false;
        let mut lambda_keyword_only = false;
        let mut param_has_default = false;
        let mut lambda_positional_seen = false;
        let mut lambda_slash_seen = false;
        let mut lambda_bare_star_pending = false;
        // Implicit-concat / string-prefix tracking: a quote after an
        // operand is legal only directly glued to a prefix identifier
        // (`rb"x"` — one literal) or after a string literal (`"a" "b"`).
        let mut last_str = false;
        let mut prefix_end = usize::MAX;
        let mut prefix_f = false;
        let mut prefix_b = false;
        let mut prefix_r = false;
        // `*a`/`**a` consumed at an element start inside a bracket —
        // recorded as (depth, bracket). A `for` at that depth makes it a
        // comprehension element where unpacking is a SyntaxError
        // (`[*a for x in y]`, `f(*a for x in y)`); a `)` closing a `(`
        // group with no `,` is `(*a)` — also a SyntaxError.
        let mut starred_depths: Vec<(usize, u8)> = Vec::new();
        if starred {
            starred_depths.push((0, 0));
        }
        // `,` inside a bracket — a comprehension is single-element, so a
        // `for` after any element comma is a SyntaxError (`[a, x for x
        // in y]`).
        let mut comma_depths: Vec<usize> = Vec::new();
        // Depths where a dict `,` was seen — the next operand is a key
        // that owes a `:` (`{k:v, k2}` is a SyntaxError). A `}` in Have
        // state with a key pending rejects; in Need it's a legal
        // trailing comma.
        let mut dict_key_next: Vec<usize> = Vec::new();
        let mut slice_colons: Vec<(usize, u8)> = Vec::new();
        let mut await_operand = false;
        let mut yield_depths: Vec<usize> = Vec::new();
        let mut yield_from_depths: Vec<usize> = Vec::new();
        // `*a`/`**a` unpacking is legal only at an element start — after
        // `,`, a bracket open, or the expression start. After an
        // operator, `:`, `=`, or a unary prefix it is a SyntaxError
        // (`f(1 + *a)`).
        let mut elem_start = !lambda_default;
        // Argument ordering inside a nested call: once a `x=1` kwarg is
        // consumed at a CALLPAREN depth, later elements must be `*a` or
        // kwargs — a bare positional (`helper(x=1, 2)`) is a SyntaxError.
        // `kwarg_seen` persists per depth; `kwarg_elem`/`star_elem`
        // describe the element currently closing and reset at `,`.
        let mut kwarg_seen_depths: Vec<usize> = Vec::new();
        let mut kw_unpack_depths: Vec<usize> = Vec::new();
        let mut kwarg_elem_depths: Vec<usize> = Vec::new();
        let mut star_elem_depths: Vec<usize> = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(()); // ran out of text — unterminated
            };
            // Inside a `for` target only target grammar is legal — names,
            // `*` starred items, `(`/`[` target groups, `.`/`[` postfix,
            // `,` separators, and the terminating `in`. Binary operators,
            // literals, and calls are all SyntaxErrors (`[x for x + y
            // in z]`). A subscript island suspends the target grammar
            // until it closes (`for a[x + 1] in` is valid).
            let strict_target = for_depths.last().is_some_and(|target_depth| {
                island_stack
                    .last()
                    .is_none_or(|island_depth| target_depth > island_depth)
            });
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                // Top-level argument boundary — valid only after an operand
                // (not inside `lambda` params, where `,` separates names).
                b')' | b',' | b':'
                    if stack.is_empty() && st != St::Lambda && (b != b':' || lambda_default) =>
                {
                    return if st == St::Have
                        && pending_ifs.is_empty()
                        && for_depths.is_empty()
                        && !(top_genexpr && (b == b',' || lambda_default))
                    {
                        Ok(())
                    } else {
                        Err(())
                    };
                }
                _ => match st {
                    St::Need => {
                        if after_not_op {
                            // A `not` after an operand only ever forms
                            // `x not in y` — anything else next (`is`, a
                            // value, a group) is a SyntaxError.
                            let id_start = self.pos;
                            self.pos = ident_end(self.s, self.pos);
                            if self.pos > id_start && &self.s[id_start..self.pos] == b"in" {
                                after_not_op = false;
                                need_ctx = NeedCtx::Restricted;
                                continue; // still Need — the operand follows
                            }
                            return Err(());
                        }
                        if await_operand
                            && !matches!(b, b'(' | b'[' | b'{' | b'\'' | b'"' | b'.')
                            && !b.is_ascii_digit()
                            && !ident_char_at(self.s, self.pos, true)
                        {
                            return Err(());
                        }
                        match b {
                            // A `for` target admits only names, `*` starred
                            // items, and `(`/`[` target groups — literals,
                            // unary ops, `{}`, and every other first are
                            // SyntaxErrors.
                            _ if strict_target => match b {
                                _ if !target_group_depths.contains(&stack.len())
                                    && (matches!(b, b'\'' | b'"' | b'{' | b'.')
                                        || b.is_ascii_digit()) =>
                                {
                                    enter_direct_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_direct_receiver_depths,
                                    );
                                    need_ctx = NeedCtx::Expr;
                                    continue;
                                }
                                _ if target_group_depths.contains(&stack.len())
                                    && !matches!(b, b'*' | b'(' | b'[' | b')' | b']')
                                    && !ident_char_at(self.s, self.pos, true) =>
                                {
                                    enter_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_group_depths,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_receiver_depths,
                                    );
                                    need_ctx = NeedCtx::Expr;
                                    continue;
                                }
                                b'*' => {
                                    if self.s.get(self.pos + 1) == Some(&b'*') {
                                        return Err(()); // `**a` is no target
                                    }
                                    target_star_depths.push(stack.len());
                                    self.pos += 1;
                                }
                                b'(' | b'[' => {
                                    stack.push(b);
                                    target_group_depths.push(stack.len());
                                    self.pos += 1;
                                }
                                b')' | b']' => {
                                    let depth = stack.len();
                                    let Some(&open) = stack.last() else {
                                        return Err(());
                                    };
                                    let mut previous = self.pos;
                                    while previous > 0
                                        && matches!(
                                            self.s[previous - 1],
                                            b' ' | b'\t' | b'\n' | b'\r'
                                        )
                                    {
                                        previous -= 1;
                                    }
                                    let empty = previous > 0
                                        && matches!(
                                            (open, self.s[previous - 1]),
                                            (b'(', b'(') | (b'[', b'[')
                                        );
                                    if !target_group_depths.contains(&depth)
                                        || !close_match(open, b)
                                        || (!empty && !target_comma_depths.contains(&depth))
                                    {
                                        return Err(());
                                    }
                                    stack.pop();
                                    self.pos += 1;
                                    target_group_depths.retain(|&d| d != depth);
                                    target_star_depths.retain(|&d| d != depth);
                                    target_comma_depths.retain(|&d| d != depth);
                                    target_unassignable_depths.retain(|&d| d != depth);
                                    target_unassignable_depths.retain(|&d| d != stack.len());
                                    st = St::Have;
                                    last_str = false;
                                    prefix_end = usize::MAX;
                                }
                                _ if ident_char_at(self.s, self.pos, true) => {
                                    let id_start = self.pos;
                                    self.pos = ident_end(self.s, self.pos);
                                    let id = &self.s[id_start..self.pos];
                                    let depth = stack.len();
                                    if id == b"in"
                                        && for_depths.last() == Some(&depth)
                                        && target_comma_depths.contains(&depth)
                                    {
                                        for_depths.pop();
                                        target_group_depths.retain(|&d| d < depth);
                                        target_star_depths.retain(|&d| d < depth);
                                        target_comma_depths.retain(|&d| d < depth);
                                        target_unassignable_depths.retain(|&d| d < depth);
                                        st = St::Need;
                                        need_ctx = NeedCtx::Restricted;
                                    } else {
                                        let direct_receiver = !target_group_depths.contains(&depth)
                                            && (matches!(id, b"True" | b"False" | b"None")
                                                || (string_prefix(id)
                                                    && matches!(self.peek(), Some(b'\'' | b'"'))));
                                        if direct_receiver {
                                            self.pos = id_start;
                                            enter_direct_target_receiver(
                                                depth,
                                                &mut island_stack,
                                                &mut target_star_depths,
                                                &mut target_comma_depths,
                                                &mut target_unassignable_depths,
                                                &mut target_direct_receiver_depths,
                                            );
                                            need_ctx = NeedCtx::Expr;
                                            continue;
                                        }
                                        if reserved_ident(id) {
                                            if !target_group_depths.contains(&depth) {
                                                return Err(());
                                            }
                                            self.pos = id_start;
                                            enter_target_receiver(
                                                depth,
                                                &mut island_stack,
                                                &mut target_group_depths,
                                                &mut target_star_depths,
                                                &mut target_comma_depths,
                                                &mut target_unassignable_depths,
                                                &mut target_receiver_depths,
                                            );
                                            need_ctx = NeedCtx::Expr;
                                            continue;
                                        }
                                        target_unassignable_depths.retain(|&d| d != depth);
                                        st = St::Have;
                                        last_str = false;
                                        prefix_end = usize::MAX;
                                    }
                                }
                                _ => return Err(()),
                            },
                            b'(' | b'[' | b'{' => {
                                stack.push(b);
                                self.pos += 1;
                                await_operand = false;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            // Empty container (`()` `[]` `{}`) or a close after
                            // `,`/`:` inside brackets (`(a,)` `a[1:]`) — never
                            // after an operator (`(a +)` is a SyntaxError).
                            b')' | b']' | b'}' => {
                                let mut p = self.pos;
                                while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r')
                                {
                                    p -= 1;
                                }
                                let trailing_ok = p > 0
                                    && match self.s[p - 1] {
                                        b'(' | b'[' | b'{' | b',' => true,
                                        // `x[1:]` — an open slice end is
                                        // legal only in a subscript; `{k:}`
                                        // is missing its dict value.
                                        b':' => stack.last() == Some(&SUBSCRIPT),
                                        _ => false,
                                    };
                                if !trailing_ok
                                    || (b == b']'
                                        && stack.last() == Some(&SUBSCRIPT)
                                        && self.s[p - 1] == b'[')
                                {
                                    return Err(());
                                }
                                match stack.pop() {
                                    Some(o) if close_match(o, b) => {
                                        // `[x for a,]` — closing the comp
                                        // bracket while its `for` still
                                        // owes `in` is a SyntaxError.
                                        if for_depths.last() == Some(&(stack.len() + 1)) {
                                            return Err(());
                                        }
                                        let closed_depth = stack.len() + 1;
                                        if target_receiver_depths.contains(&closed_depth)
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            island_stack.pop();
                                            target_receiver_depths.retain(|&d| d != closed_depth);
                                            if !target_unassignable_depths.contains(&stack.len()) {
                                                target_unassignable_depths.push(stack.len());
                                            }
                                        } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                            && target_direct_receiver_depths.contains(&stack.len())
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            if o == SUBSCRIPT {
                                                island_stack.pop();
                                                target_direct_receiver_depths
                                                    .retain(|&d| d != stack.len());
                                            }
                                        } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            island_stack.pop();
                                            if o == CALLPAREN {
                                                if !target_unassignable_depths
                                                    .contains(&stack.len())
                                                {
                                                    target_unassignable_depths.push(stack.len());
                                                }
                                            } else {
                                                target_unassignable_depths
                                                    .retain(|&d| d != stack.len());
                                            }
                                        }
                                        self.pos += 1;
                                        st = St::Have;
                                        last_str = false;
                                        prefix_end = usize::MAX;
                                        comp_depths.retain(|&d| d <= stack.len());
                                        for_depths.retain(|&d| d <= stack.len());
                                        starred_depths.retain(|&(d, _)| d <= stack.len());
                                        comma_depths.retain(|&d| d <= stack.len());
                                        dict_key_next.retain(|&d| d <= stack.len());
                                        slice_colons.retain(|(d, _)| *d <= stack.len());
                                        yield_depths.retain(|&d| d <= stack.len());
                                        yield_from_depths.retain(|&d| d <= stack.len());
                                        kwarg_seen_depths.retain(|&d| d <= stack.len());
                                        kw_unpack_depths.retain(|&d| d <= stack.len());
                                        kwarg_elem_depths.retain(|&d| d <= stack.len());
                                        star_elem_depths.retain(|&d| d <= stack.len());
                                    }
                                    _ => return Err(()),
                                }
                            }
                            b'\'' | b'"' => {
                                self.skip_quoted(false, false, false)?;
                                st = St::Have;
                                await_operand = false;
                                last_str = true;
                                prefix_end = usize::MAX;
                            }
                            b'-' | b'+' | b'~' => {
                                self.pos += 1;
                                elem_start = false; // `-*a` is a SyntaxError
                                need_ctx = NeedCtx::Restricted;
                            }
                            // `*a`/`**a` unpacking — legal only at an
                            // element start (`f(*a)`, `[*a]`, `{*a}`,
                            // `(*a,)`), never after an operator (`1 + *a`)
                            // or inside a `:`/`=` continuation.
                            b'*' => {
                                if !elem_start {
                                    return Err(());
                                }
                                let is_kw_unpack = self.s.get(self.pos + 1) == Some(&b'*');
                                let depth = stack.len();
                                match stack.last().copied() {
                                    Some(CALLPAREN) => {
                                        if is_kw_unpack {
                                            kwarg_seen_depths.push(depth);
                                            kw_unpack_depths.push(depth);
                                        } else if kw_unpack_depths.contains(&depth) {
                                            return Err(());
                                        }
                                    }
                                    Some(b'{') if is_kw_unpack => {
                                        *stack.last_mut().unwrap() = DICT;
                                    }
                                    Some(DICT) if is_kw_unpack => {
                                        dict_key_next.retain(|&d| d != depth);
                                    }
                                    Some(DICT) => return Err(()),
                                    Some(SET) if is_kw_unpack => return Err(()),
                                    _ if is_kw_unpack => return Err(()),
                                    _ => {}
                                }
                                if let Some(&top) = stack.last() {
                                    starred_depths.push((stack.len(), top));
                                    star_elem_depths.push(stack.len());
                                }
                                self.pos += 1;
                                if is_kw_unpack {
                                    self.pos += 1; // `**a`
                                }
                                elem_start = false;
                                need_ctx = NeedCtx::Restricted;
                            }
                            b'.' => {
                                // `...` ellipsis or `.5` float — nothing else.
                                if self.s.get(self.pos + 1) == Some(&b'.')
                                    && self.s.get(self.pos + 2) == Some(&b'.')
                                {
                                    self.pos += 3;
                                    st = St::Have;
                                } else if self
                                    .s
                                    .get(self.pos + 1)
                                    .is_some_and(|c| c.is_ascii_digit())
                                {
                                    let num_start = self.pos;
                                    self.pos += 2;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'e' | b'E')
                                    ) {
                                        self.pos += 1;
                                        if matches!(
                                            self.s.get(self.pos),
                                            Some(c) if matches!(c, b'+' | b'-')
                                        ) {
                                            self.pos += 1;
                                        }
                                        let exp_start = self.pos;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                        if self.pos == exp_start {
                                            return Err(());
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'j' | b'J')
                                    ) {
                                        self.pos += 1;
                                    }
                                    if !Self::valid_numeric_underscores(
                                        &self.s[num_start..self.pos],
                                        |c| c.is_ascii_digit(),
                                        false,
                                    ) {
                                        return Err(());
                                    }
                                    st = St::Have;
                                } else {
                                    return Err(());
                                }
                                await_operand = false;
                                last_str = false;
                                prefix_end = usize::MAX;
                            }
                            // `x[:2]` / `x[::]` — `:` where an operand is
                            // expected is slice syntax, only in a subscript.
                            b':' if stack.last() == Some(&SUBSCRIPT) => {
                                let depth = stack.len();
                                match slice_colons.iter_mut().find(|(d, _)| *d == depth) {
                                    Some((_, count)) if *count >= 2 => return Err(()),
                                    Some((_, count)) => *count += 1,
                                    None => slice_colons.push((depth, 1)),
                                }
                                self.pos += 1;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            b',' if stack.last() == Some(&SUBSCRIPT)
                                && slice_colons.iter().any(|(d, _)| *d == stack.len()) =>
                            {
                                slice_colons.retain(|(d, _)| *d != stack.len());
                                self.pos += 1;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ if ident_char_at(self.s, self.pos, true) => {
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                let id = &self.s[id_start..self.pos];
                                let mut next = self.pos;
                                while matches!(self.s.get(next), Some(b' ' | b'\t' | b'\n' | b'\r'))
                                {
                                    next += 1;
                                }
                                let nested_kwarg = elem_start
                                    && stack.last() == Some(&CALLPAREN)
                                    && self.s.get(next) == Some(&b'=')
                                    && self.s.get(next + 1) != Some(&b'=');
                                if !nested_kwarg && id == b"await" {
                                    if await_operand {
                                        return Err(());
                                    }
                                    await_operand = true;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                    continue;
                                }
                                if !nested_kwarg && id == b"yield" {
                                    if await_operand {
                                        return Err(());
                                    }
                                    let mut before = id_start;
                                    while before > 0
                                        && matches!(
                                            self.s[before - 1],
                                            b' ' | b'\t' | b'\n' | b'\r'
                                        )
                                    {
                                        before -= 1;
                                    }
                                    if stack.last() != Some(&b'(')
                                        || before == 0
                                        || self.s[before - 1] != b'('
                                    {
                                        return Err(());
                                    }
                                    let mut next = self.pos;
                                    while matches!(
                                        self.s.get(next),
                                        Some(b' ' | b'\t' | b'\n' | b'\r')
                                    ) {
                                        next += 1;
                                    }
                                    let next_end = ident_end(self.s, next);
                                    let yield_from =
                                        next_end > next && &self.s[next..next_end] == b"from";
                                    yield_depths.push(stack.len());
                                    if yield_from {
                                        self.pos = next_end;
                                        yield_from_depths.push(stack.len());
                                    }
                                    if !yield_from && self.s.get(next) == Some(&b')') {
                                        st = St::Have;
                                    } else {
                                        st = St::Need;
                                        elem_start = !yield_from;
                                        need_ctx = NeedCtx::Expr;
                                    }
                                    last_str = false;
                                    prefix_end = usize::MAX;
                                    continue;
                                }
                                if !nested_kwarg && reject_keyword(id) {
                                    return Err(());
                                }
                                match id {
                                    b"not" if !nested_kwarg => {
                                        if need_ctx == NeedCtx::Restricted {
                                            return Err(());
                                        }
                                        elem_start = false;
                                        need_ctx = NeedCtx::Inversion;
                                    } // unary — still need an operand
                                    b"lambda" if !nested_kwarg => {
                                        if need_ctx != NeedCtx::Expr
                                            || comp_depths.contains(&stack.len())
                                            || pending_ifs.contains(&stack.len())
                                        {
                                            return Err(());
                                        }
                                        st = St::Lambda;
                                        lambda_depth = stack.len();
                                        param_start = true;
                                        param_done = false;
                                        lambda_marker = LambdaMarker::None;
                                        lambda_default_seen = false;
                                        lambda_keyword_only = false;
                                        param_has_default = false;
                                        lambda_positional_seen = false;
                                        lambda_slash_seen = false;
                                        lambda_bare_star_pending = false;
                                    }
                                    _ => {
                                        st = St::Have;
                                        await_operand = false;
                                        last_str = false;
                                        let is_string_prefix = string_prefix(id);
                                        prefix_end = if is_string_prefix {
                                            self.pos
                                        } else {
                                            usize::MAX
                                        };
                                        prefix_f = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'f'));
                                        prefix_b = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'b'));
                                        prefix_r = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'r'));
                                    }
                                }
                            }
                            // Number operand: digits with optional `0x`/`0o`/`0b`
                            // radix, fraction, exponent, `_` separators, `j`. A
                            // letter glued straight on (`5x`) is a SyntaxError,
                            // and `_` placement follows PEP 515 (between digits,
                            // or right after the base prefix).
                            _ if b.is_ascii_digit() => {
                                if b == b'0'
                                    && matches!(
                                        self.s.get(self.pos + 1),
                                        Some(c) if matches!(c, b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
                                    )
                                {
                                    let radix = match self.s[self.pos + 1].to_ascii_lowercase() {
                                        b'x' => 16,
                                        b'o' => 8,
                                        _ => 2,
                                    };
                                    self.pos += 2;
                                    let dstart = self.pos;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| (*c as char).is_digit(radix) || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if self.pos == dstart
                                        || !Self::valid_numeric_underscores(
                                            &self.s[dstart..self.pos],
                                            |c| (c as char).is_digit(radix),
                                            true,
                                        )
                                    {
                                        return Err(());
                                    }
                                } else {
                                    let num_start = self.pos;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if self.s.get(self.pos) == Some(&b'.') {
                                        self.pos += 1;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'e' | b'E')
                                    ) {
                                        self.pos += 1;
                                        if matches!(
                                            self.s.get(self.pos),
                                            Some(c) if matches!(c, b'+' | b'-')
                                        ) {
                                            self.pos += 1;
                                        }
                                        let exp_start = self.pos;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                        // `1e`, `1e+`, `.5e-` — an exponent
                                        // marker with no digits is a SyntaxError.
                                        if self.pos == exp_start {
                                            return Err(());
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'j' | b'J')
                                    ) {
                                        self.pos += 1;
                                    }
                                    if !Self::valid_numeric_underscores(
                                        &self.s[num_start..self.pos],
                                        |c| c.is_ascii_digit(),
                                        false,
                                    ) {
                                        return Err(());
                                    }
                                }
                                // `5x` / `5é` — a digit-glued name is a
                                // SyntaxError.
                                if ident_char_at(self.s, self.pos, false) {
                                    return Err(());
                                }
                                st = St::Have;
                                await_operand = false;
                                last_str = false;
                                prefix_end = usize::MAX;
                            }
                            _ => return Err(()),
                        }
                    }
                    St::Have => match b {
                        _ if target_direct_receiver_depths.contains(&stack.len())
                            && !matches!(b, b'(' | b'[' | b'.' | b'\'' | b'"') =>
                        {
                            return Err(());
                        }
                        b')' | b']' | b'}' => match stack.pop() {
                            Some(o) if close_match(o, b) => {
                                let d = stack.len() + 1; // the popped bracket's inner depth
                                let target_group = target_group_depths.contains(&d);
                                let target_receiver = target_receiver_depths.contains(&d);
                                if target_group
                                    && o == b'('
                                    && target_star_depths.contains(&d)
                                    && !target_comma_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                // `(a if b)` closes with the ternary's
                                // `else` still owed — a SyntaxError. So
                                // does a `for` whose `in` never arrived
                                // (`[x for x]`).
                                if pending_ifs.contains(&d) || for_depths.last() == Some(&d) {
                                    return Err(());
                                }
                                // `(*a)` — a bare starred group is a
                                // SyntaxError; `(*a,)` is a legal tuple.
                                if o == b'('
                                    && starred_depths
                                        .iter()
                                        .any(|&(sd, brk)| sd == d && brk == b'(')
                                    && !yield_depths.contains(&d)
                                    && !comma_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                // `{k:v, k2}` — a dict element whose key
                                // owes `:` before the close.
                                if o == DICT && dict_key_next.contains(&d) {
                                    return Err(());
                                }
                                // `helper(x=1, 2)` — a bare positional
                                // closing a call that saw a kwarg.
                                if o == CALLPAREN
                                    && kwarg_seen_depths.contains(&d)
                                    && !kwarg_elem_depths.contains(&d)
                                    && !star_elem_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                if target_receiver && island_stack.last() == Some(&stack.len()) {
                                    island_stack.pop();
                                    target_receiver_depths.retain(|&x| x != d);
                                    if !target_unassignable_depths.contains(&stack.len()) {
                                        target_unassignable_depths.push(stack.len());
                                    }
                                } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                    && target_direct_receiver_depths.contains(&stack.len())
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    if o == SUBSCRIPT {
                                        island_stack.pop();
                                        target_direct_receiver_depths.retain(|&x| x != stack.len());
                                    }
                                } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    island_stack.pop();
                                    if o == CALLPAREN {
                                        if !target_unassignable_depths.contains(&stack.len()) {
                                            target_unassignable_depths.push(stack.len());
                                        }
                                    } else {
                                        target_unassignable_depths.retain(|&x| x != stack.len());
                                    }
                                }
                                if target_group {
                                    let unassignable = target_unassignable_depths.contains(&d);
                                    target_group_depths.retain(|&x| x != d);
                                    target_star_depths.retain(|&x| x != d);
                                    target_comma_depths.retain(|&x| x != d);
                                    target_unassignable_depths.retain(|&x| x != d);
                                    if unassignable {
                                        if !target_unassignable_depths.contains(&stack.len()) {
                                            target_unassignable_depths.push(stack.len());
                                        }
                                    } else {
                                        target_unassignable_depths.retain(|&x| x != stack.len());
                                    }
                                }
                                self.pos += 1;
                                comp_depths.retain(|&x| x <= stack.len());
                                for_depths.retain(|&x| x <= stack.len());
                                starred_depths.retain(|&(x, _)| x <= stack.len());
                                comma_depths.retain(|&x| x <= stack.len());
                                dict_key_next.retain(|&x| x <= stack.len());
                                slice_colons.retain(|(x, _)| *x <= stack.len());
                                yield_depths.retain(|&x| x <= stack.len());
                                yield_from_depths.retain(|&x| x <= stack.len());
                                kwarg_seen_depths.retain(|&x| x <= stack.len());
                                kw_unpack_depths.retain(|&x| x <= stack.len());
                                kwarg_elem_depths.retain(|&x| x <= stack.len());
                                star_elem_depths.retain(|&x| x <= stack.len());
                            }
                            _ => return Err(()),
                        },
                        // Inside a `for` target only `,` separators, `.`/`[`
                        // postfix chains, and the terminating `in` are
                        // legal — operators, calls, and literals are
                        // SyntaxErrors (`[x for x + y in z]`).
                        _ if strict_target => match b {
                            b',' => {
                                if target_unassignable_depths.contains(&stack.len()) {
                                    if !target_group_depths.contains(&stack.len()) {
                                        return Err(());
                                    }
                                    enter_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_group_depths,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_receiver_depths,
                                    );
                                    continue;
                                }
                                target_comma_depths.push(stack.len());
                                self.pos += 1;
                                st = St::Need;
                            }
                            b'(' => {
                                island_stack.push(stack.len());
                                stack.push(CALLPAREN);
                                self.pos += 1;
                                st = St::Need;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            b'[' => {
                                // `a[i]` — a subscript target; its value is
                                // a full expression island.
                                island_stack.push(stack.len());
                                stack.push(SUBSCRIPT);
                                self.pos += 1;
                                st = St::Need;
                                need_ctx = NeedCtx::Expr;
                            }
                            b'.' => {
                                // `a.b` attribute target.
                                self.pos += 1;
                                let id_start = self.pos;
                                if ident_char_at(self.s, self.pos, true) {
                                    self.pos = ident_end(self.s, self.pos);
                                    if reserved_ident(&self.s[id_start..self.pos]) {
                                        return Err(());
                                    }
                                    target_unassignable_depths.retain(|&d| d != stack.len());
                                } else {
                                    return Err(());
                                }
                            }
                            _ if target_group_depths.contains(&stack.len()) => {
                                enter_target_receiver(
                                    stack.len(),
                                    &mut island_stack,
                                    &mut target_group_depths,
                                    &mut target_star_depths,
                                    &mut target_comma_depths,
                                    &mut target_unassignable_depths,
                                    &mut target_receiver_depths,
                                );
                                continue;
                            }
                            _ if ident_char_at(self.s, self.pos, true) => {
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                // `in` completes the target at the `for`'s
                                // own depth; anything else is a SyntaxError.
                                let depth = stack.len();
                                if &self.s[id_start..self.pos] == b"in"
                                    && for_depths.last() == Some(&depth)
                                {
                                    if target_unassignable_depths.contains(&depth) {
                                        return Err(());
                                    }
                                    for_depths.pop();
                                    target_group_depths.retain(|&d| d < depth);
                                    target_star_depths.retain(|&d| d < depth);
                                    target_comma_depths.retain(|&d| d < depth);
                                    target_unassignable_depths.retain(|&d| d < depth);
                                    st = St::Need;
                                    need_ctx = NeedCtx::Restricted;
                                } else {
                                    return Err(());
                                }
                            }
                            _ => return Err(()),
                        },
                        b'(' | b'[' => {
                            // Postfix call / index — a subscript `[` admits
                            // slice `:` inside; a call `(` admits keyword
                            // `x=1` arguments (`outer(helper(x=1), k=2)`).
                            stack.push(if b == b'[' { SUBSCRIPT } else { CALLPAREN });
                            self.pos += 1;
                            st = St::Need;
                            elem_start = true;
                            need_ctx = NeedCtx::Expr;
                        }
                        b',' if !stack.is_empty() => {
                            if yield_from_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // A comprehension is single-element — `[x for
                            // x in y, 2]` is a SyntaxError.
                            if comp_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // `helper(x=1, 2)` — after a kwarg only `*a`
                            // or another kwarg may follow; a bare
                            // positional is a SyntaxError.
                            if stack.last() == Some(&CALLPAREN)
                                && kwarg_seen_depths.contains(&stack.len())
                                && !kwarg_elem_depths.contains(&stack.len())
                                && !star_elem_depths.contains(&stack.len())
                            {
                                return Err(());
                            }
                            if stack.last() == Some(&SUBSCRIPT) {
                                slice_colons.retain(|(d, _)| *d != stack.len());
                            }
                            if stack.last() == Some(&DICT) && dict_key_next.contains(&stack.len()) {
                                return Err(());
                            }
                            match stack.last() {
                                // `{a,` — a bare element locks the
                                // display to a set.
                                Some(&b'{') => *stack.last_mut().unwrap() = SET,
                                // `{k:v,` — the next operand is a key
                                // that owes a `:`.
                                Some(&DICT) => dict_key_next.push(stack.len()),
                                _ => {}
                            }
                            // The element flags describe the element just
                            // closed — the next one starts fresh.
                            kwarg_elem_depths.retain(|&d| d != stack.len());
                            star_elem_depths.retain(|&d| d != stack.len());
                            comma_depths.push(stack.len());
                            self.pos += 1;
                            st = St::Need;
                            elem_start = true;
                            need_ctx = NeedCtx::Expr;
                        }
                        // `:=` walrus — the target must be a bare NAME
                        // directly after `(`/`,`/`[`/`{` or the expression
                        // start (`x:=1`, `x[a:=1]`, `{k:(v:=1)}`); inside
                        // an undecided `{` it is a bare element, locking
                        // the display to a set (`{x:=1}`). Literals,
                        // attributes, subscripts, calls, groups, and
                        // operator results can't be targets — `1:=2`,
                        // `a.b:=2`, `(x):=2`, `a+b:=2`, `x=y:=2`, and
                        // `{k: v:=1}` are all SyntaxErrors.
                        b':' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            let mut p = self.pos;
                            while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                p -= 1;
                            }
                            let id_end = p;
                            while p > 0 {
                                let c = self.s[p - 1];
                                if c.is_ascii_alphanumeric() || c == b'_' {
                                    p -= 1;
                                } else if c >= 0x80 && (c & 0xC0) == 0x80 {
                                    p -= 1; // UTF-8 continuation byte
                                } else if c >= 0x80 && ident_char_at(self.s, p - 1, false) {
                                    p -= 1; // XID_Continue lead byte
                                } else {
                                    break;
                                }
                            }
                            let mut q = p;
                            while q > 0 && matches!(self.s[q - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                q -= 1;
                            }
                            let bare_name = p < id_end
                                && ident_char_at(self.s, p, true)
                                && !reserved_ident(&self.s[p..id_end])
                                && (q == 0 || matches!(self.s[q - 1], b'(' | b',' | b'[' | b'{'));
                            if !bare_name {
                                return Err(());
                            }
                            if stack.last() == Some(&b'{') {
                                *stack.last_mut().unwrap() = SET;
                            }
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Expr;
                        }
                        b':' if !stack.is_empty() => match stack.last() {
                            // `x[1:` — slice continue; `{k:` — dict
                            // separator locking the display to a dict (a
                            // value must follow). `:` in a set, `()`,
                            // list display, or call is a SyntaxError
                            // (`[1:2]`, `{1, 2:3}`, `f(a:1)`).
                            Some(&SUBSCRIPT) => {
                                let depth = stack.len();
                                match slice_colons.iter_mut().find(|(d, _)| *d == depth) {
                                    Some((_, count)) if *count >= 2 => return Err(()),
                                    Some((_, count)) => *count += 1,
                                    None => slice_colons.push((depth, 1)),
                                }
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            Some(&b'{') => {
                                // `{*a:1}` — a starred element can't be
                                // a dict key.
                                if starred_depths.iter().any(|&(d, _)| d == stack.len()) {
                                    return Err(());
                                }
                                *stack.last_mut().unwrap() = DICT;
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            // `{k:v, k2:` — the pending key's separator.
                            Some(&DICT) if dict_key_next.contains(&stack.len()) => {
                                dict_key_next.retain(|&d| d != stack.len());
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ => return Err(()),
                        },
                        b'.' => match self.s.get(self.pos + 1) {
                            // Attribute access `a.b` — the only legal `.`
                            // after an operand: digit-led numbers already
                            // consumed their fraction/exponent, so `.5`,
                            // `x.`, `.5.5`, `1.5.5` here are SyntaxErrors.
                            Some(_) if ident_char_at(self.s, self.pos + 1, true) => {
                                let id_start = self.pos + 1;
                                self.pos = ident_end(self.s, id_start);
                                if reject_keyword(&self.s[id_start..self.pos]) {
                                    return Err(());
                                }
                                if target_direct_receiver_depths.contains(&stack.len())
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    island_stack.pop();
                                    target_direct_receiver_depths.retain(|&d| d != stack.len());
                                }
                            }
                            _ => return Err(()),
                        },
                        b'\'' | b'"' => {
                            // `rb"x"` (glued prefix → one literal) or
                            // implicit concat `"a" "b"` — nothing else.
                            if !last_str && self.pos != prefix_end {
                                return Err(());
                            }
                            let has_prefix = !last_str && self.pos == prefix_end;
                            self.skip_quoted(
                                has_prefix && prefix_f,
                                has_prefix && prefix_b,
                                has_prefix && prefix_r,
                            )?;
                            last_str = true;
                            prefix_end = usize::MAX;
                        }
                        // Two-character operators are consumed greedily so
                        // `==`/`<=`/`!=`/`**`/`//`/`<<`/`>>` stay one token.
                        b'=' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'!' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        // `=` inside a call paren is keyword-argument
                        // syntax — but only after a bare name (`g(x=1)`
                        // nested in a dropped positional). `g(5=1)`,
                        // `g(a.b=1)`, `g(*a=1)`, `g()=1` are SyntaxErrors;
                        // so is `=` anywhere else (`x[a=1]`, `{a=1}`,
                        // `(a=1)`).
                        b'=' if stack.last() == Some(&CALLPAREN) => {
                            // Walk back over whitespace + the identifier —
                            // a kwarg name is a bare ident directly after
                            // `(` or `,`.
                            let mut p = self.pos;
                            while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                p -= 1;
                            }
                            let id_end = p;
                            while p > 0 {
                                let c = self.s[p - 1];
                                if c.is_ascii_alphanumeric() || c == b'_' {
                                    p -= 1;
                                } else if c >= 0x80 && (c & 0xC0) == 0x80 {
                                    p -= 1; // UTF-8 continuation byte
                                } else if c >= 0x80 && ident_char_at(self.s, p - 1, false) {
                                    p -= 1; // XID_Continue lead byte
                                } else {
                                    break;
                                }
                            }
                            let mut q = p;
                            while q > 0 && matches!(self.s[q - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                q -= 1;
                            }
                            let bare_name = p < id_end
                                && ident_char_at(self.s, p, true)
                                && q > 0
                                && matches!(self.s[q - 1], b'(' | b',');
                            // `f(*a=1)` — a starred element can't take `=`
                            // (element-scoped: `helper(*a, x=1)` is legal).
                            if !bare_name || star_elem_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // A kwarg consumed — later elements at this
                            // depth must be `*a` or kwargs.
                            kwarg_seen_depths.push(stack.len());
                            kwarg_elem_depths.push(stack.len());
                            self.pos += 1;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Expr;
                        }
                        b'<' | b'>' => {
                            self.pos += 1;
                            if matches!(
                                self.s.get(self.pos),
                                Some(c) if *c == b || *c == b'='
                            ) {
                                self.pos += 1;
                            }
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'*' | b'/' => {
                            self.pos += 1;
                            if self.s.get(self.pos) == Some(&b) {
                                self.pos += 1; // `**` `//`
                            }
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'+' | b'-' | b'%' | b'|' | b'&' | b'^' | b'@' => {
                            self.pos += 1;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        _ if ident_char_at(self.s, self.pos, false) => {
                            // Only word operators may follow an operand —
                            // juxtaposed operands (`a b`) are a SyntaxError.
                            let id_start = self.pos;
                            self.pos = ident_end(self.s, self.pos);
                            match &self.s[id_start..self.pos] {
                                b"async" => {
                                    let mut next = self.pos;
                                    while matches!(
                                        self.s.get(next),
                                        Some(b' ' | b'\t' | b'\n' | b'\r')
                                    ) {
                                        next += 1;
                                    }
                                    let next_end = ident_end(self.s, next);
                                    if next_end == next || &self.s[next..next_end] != b"for" {
                                        return Err(());
                                    }
                                    self.pos = next;
                                }
                                b"and" | b"or" | b"is" => {
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Inversion;
                                }
                                b"in" => {
                                    if target_direct_receiver_depths.contains(&stack.len())
                                        && for_depths.last() == Some(&stack.len())
                                    {
                                        return Err(());
                                    }
                                    // Satisfies the innermost `for` at
                                    // this bracket depth; `in` inside
                                    // deeper brackets is a membership op.
                                    if for_depths.last() == Some(&stack.len()) {
                                        for_depths.pop();
                                    }
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                b"if" => {
                                    // Ternary needs `else`; a filter `if`
                                    // inside a comprehension does not.
                                    if !comp_depths.contains(&stack.len()) {
                                        if pending_ifs.contains(&stack.len()) {
                                            return Err(());
                                        }
                                        pending_ifs.push(stack.len());
                                    }
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Inversion;
                                }
                                b"else" => {
                                    // `else` without an open ternary is a
                                    // SyntaxError.
                                    let Some(index) =
                                        pending_ifs.iter().rposition(|&d| d == stack.len())
                                    else {
                                        return Err(());
                                    };
                                    pending_ifs.remove(index);
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Expr;
                                }
                                b"for" => {
                                    let d = stack.len();
                                    // `*` unpacking can't be a comp
                                    // element (`[*a for x in y]`,
                                    // `f(*a for a in b)`); a `for` can't
                                    // live in a subscript (`x[a for a in
                                    // y]`); a comp is single-element
                                    // (`[a, x for x in y]`); and a dict
                                    // key can't be a comp (`{k:v, x
                                    // for}`).
                                    if stack.last() == Some(&SUBSCRIPT)
                                        || yield_depths.contains(&d)
                                        || starred_depths.iter().any(|&(sd, _)| sd == d)
                                        || comma_depths.contains(&d)
                                        || dict_key_next.contains(&d)
                                    {
                                        return Err(());
                                    }
                                    // `for` right after `{` → set comp —
                                    // the element was a bare value, so the
                                    // display can never become a dict.
                                    if stack.last() == Some(&b'{') {
                                        *stack.last_mut().unwrap() = SET;
                                    }
                                    // Comprehension clauses run at the
                                    // bracket depth they appear in and
                                    // each owes an `in` at that depth; a
                                    // top-level `for` is a genexpr, valid
                                    // only as the call's sole argument.
                                    comp_depths.push(d);
                                    for_depths.push(d);
                                    top_genexpr |= stack.is_empty();
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                b"not" => {
                                    after_not_op = true;
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                _ => return Err(()),
                            }
                        }
                        _ => return Err(()),
                    },
                    St::Lambda => match b {
                        b'=' if lambda_marker == LambdaMarker::VarargsDone
                            && stack.len() == lambda_depth =>
                        {
                            return Err(());
                        }
                        _ if lambda_marker == LambdaMarker::KwargsDone
                            && stack.len() == lambda_depth
                            && b != b':'
                            && !(param_done && b == b',') =>
                        {
                            return Err(());
                        }
                        b':' if stack.len() == lambda_depth
                            && (lambda_bare_star_pending
                                || matches!(
                                    lambda_marker,
                                    LambdaMarker::Star | LambdaMarker::DoubleStar
                                )) =>
                        {
                            return Err(());
                        }
                        _ if param_done
                            && stack.len() == lambda_depth
                            && matches!(b, b',' | b':')
                            && lambda_default_seen
                            && !lambda_keyword_only
                            && !param_has_default =>
                        {
                            return Err(());
                        }
                        // After a param name only `,`, `=`, or the ending
                        // `:` is legal — `lambda a b`, `lambda a.b`, and
                        // `lambda a(` are all SyntaxErrors.
                        _ if param_done && stack.len() == lambda_depth => match b {
                            b',' => {
                                param_start = true;
                                param_done = false;
                                if lambda_marker != LambdaMarker::KwargsDone {
                                    lambda_marker = LambdaMarker::None;
                                }
                                self.pos += 1;
                            }
                            b'=' => {
                                self.pos += 1;
                                self.skip_expression_until(false, true)?;
                                lambda_default_seen = true;
                                param_has_default = true;
                            }
                            b':' => {
                                param_done = false;
                                self.pos += 1;
                                st = St::Need;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ => return Err(()),
                        },
                        // The lambda's own `:` ends its parameter list.
                        b':' if stack.len() == lambda_depth => {
                            self.pos += 1;
                            st = St::Need;
                            need_ctx = NeedCtx::Expr;
                        }
                        b':' => self.pos += 1, // inside default-expr brackets
                        // `(a)`-style params are Python 2 — an opener at a
                        // param start is a SyntaxError; inside a default
                        // expression brackets are fine.
                        b'(' | b'[' | b'{' => {
                            if param_start && stack.len() == lambda_depth {
                                return Err(());
                            }
                            stack.push(b);
                            self.pos += 1;
                        }
                        b')' | b']' | b'}' if stack.len() > lambda_depth => match stack.pop() {
                            Some(o) if close_match(o, b) => {
                                self.pos += 1;
                            }
                            _ => return Err(()),
                        },
                        b'\'' | b'"' => {
                            if param_start && stack.len() == lambda_depth {
                                return Err(());
                            }
                            self.skip_quoted(false, false, false)?;
                        }
                        b',' => {
                            if stack.len() == lambda_depth {
                                // `a,,b` — a `,` is only legal right after
                                // a `*`/`/` marker or a name.
                                if param_start
                                    && matches!(
                                        lambda_marker,
                                        LambdaMarker::None | LambdaMarker::DoubleStar
                                    )
                                {
                                    return Err(());
                                }
                                lambda_bare_star_pending |= lambda_marker == LambdaMarker::Star;
                                param_start = true;
                                lambda_marker = LambdaMarker::None;
                            }
                            self.pos += 1;
                        }
                        // `=x` with no name; unary/dot bytes at a param
                        // start — all SyntaxErrors.
                        b'=' | b'-' | b'+' | b'~' | b'.'
                            if param_start && stack.len() == lambda_depth =>
                        {
                            return Err(());
                        }
                        b'=' | b'-' | b'+' | b'~' | b'.' => self.pos += 1,
                        b'*' | b'/' => {
                            if !param_start
                                || stack.len() != lambda_depth
                                || lambda_marker != LambdaMarker::None
                            {
                                return Err(());
                            }
                            if b == b'/' {
                                if !lambda_positional_seen
                                    || lambda_keyword_only
                                    || lambda_slash_seen
                                {
                                    return Err(());
                                }
                                lambda_marker = LambdaMarker::Slash;
                                lambda_slash_seen = true;
                            } else {
                                lambda_marker = if self.s.get(self.pos + 1) == Some(&b'*') {
                                    self.pos += 1;
                                    LambdaMarker::DoubleStar
                                } else {
                                    if lambda_keyword_only {
                                        return Err(());
                                    }
                                    LambdaMarker::Star
                                };
                                lambda_keyword_only = true;
                            }
                            self.pos += 1;
                        }
                        _ if ident_char_at(self.s, self.pos, false) => {
                            if param_start && stack.len() == lambda_depth {
                                if lambda_marker == LambdaMarker::Slash {
                                    return Err(());
                                }
                                // Param names start with a letter/`_` and
                                // can't be keywords — `lambda 1: x` and
                                // `lambda for: x` are both SyntaxErrors.
                                if !ident_char_at(self.s, self.pos, true) {
                                    return Err(());
                                }
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                if reserved_ident(&self.s[id_start..self.pos]) {
                                    return Err(());
                                }
                                lambda_positional_seen |= !lambda_keyword_only;
                                if lambda_marker == LambdaMarker::None {
                                    lambda_bare_star_pending = false;
                                }
                                param_start = false;
                                lambda_marker = match lambda_marker {
                                    LambdaMarker::Star => LambdaMarker::VarargsDone,
                                    LambdaMarker::DoubleStar => LambdaMarker::KwargsDone,
                                    _ => LambdaMarker::None,
                                };
                                param_done = true;
                                param_has_default = false;
                            } else {
                                self.pos += 1;
                            }
                        }
                        _ => return Err(()),
                    },
                },
            }
        }
    }

    /// Parse a string literal: `'…'` `"…"` `'''…'''` `"""…"""` with
    /// optional `r`/`u`/`f`/`R`/`U`/`F` prefix. `f` is accepted only when
    /// the body has no `{`/`}` placeholder (vLLM admits JoinedStr made of
    /// pure constants). Raw control chars inside strings are accepted —
    /// our parser is lenient where `ast.parse` needed the escape rewrite.
    fn parse_string(&mut self) -> Result<String, ()> {
        let mut raw_mode = false;
        let mut f_mode = false;
        // Optional letter prefix — Python admits r/u/f/b alone and the
        // r-pairs rb/br/rf/fr only (`rr`, `uf`, `bu` … are SyntaxErrors).
        // The dispatch guard guarantees a quote within the two-byte
        // lookahead, so the prefix is at most two letters here.
        if matches!(
            self.peek(),
            Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
        ) {
            let pstart = self.pos;
            while matches!(
                self.peek(),
                Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
            ) {
                match self.s[self.pos].to_ascii_lowercase() {
                    b'r' => raw_mode = true,
                    b'f' => f_mode = true,
                    b'b' => return Err(()), // bytes are not JSON-representable
                    _ => {}
                }
                self.pos += 1;
            }
            let prefix = &self.s[pstart..self.pos];
            let valid = match prefix.len() {
                1 => true, // single r/u/f (b already returned above)
                2 => matches!(
                    (
                        prefix[0].to_ascii_lowercase(),
                        prefix[1].to_ascii_lowercase()
                    ),
                    (b'r', b'f') | (b'f', b'r') | (b'r', b'b') | (b'b', b'r')
                ),
                _ => false,
            };
            if !valid {
                return Err(());
            }
        }
        let quote = match self.peek() {
            Some(b'\'' | b'"') => self.s[self.pos],
            _ => return Err(()),
        };
        self.pos += 1;
        // Triple-quoted?
        let triple = self.peek() == Some(quote) && self.s.get(self.pos + 1) == Some(&quote);
        if triple {
            self.pos += 2;
        }
        let mut out = String::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(()); // unterminated
            };
            if b == quote {
                if raw_mode {
                    // Even in a raw string a backslash escapes the quote
                    // for termination (the backslash stays in the value):
                    // `r'foo\'` is unterminated in Python. An odd run of
                    // backslashes right before the quote means it cannot
                    // close the literal. Non-raw mode never reaches this —
                    // its escape arm already consumed `\'`/`\\` as pairs.
                    let mut back = self.pos;
                    while back > 0 && self.s[back - 1] == b'\\' {
                        back -= 1;
                    }
                    if (self.pos - back) % 2 == 1 {
                        out.push(quote as char);
                        self.pos += 1;
                        continue;
                    }
                }
                if triple {
                    if self.s.get(self.pos + 1) == Some(&quote)
                        && self.s.get(self.pos + 2) == Some(&quote)
                    {
                        self.pos += 3;
                        break;
                    }
                    out.push(quote as char);
                    self.pos += 1;
                    continue;
                }
                self.pos += 1;
                break;
            }
            if !triple && (b == b'\n' || b == b'\r') {
                // Single-quoted strings cannot span lines in Python — but
                // raw newlines inside args are the most common model slip
                // (vLLM fixes via escape_ctrl_chars_in_strings). Accept.
                out.push(b as char);
                self.pos += 1;
                continue;
            }
            if !raw_mode && b == b'\\' {
                self.pos += 1;
                let Some(&e) = self.s.get(self.pos) else {
                    return Err(());
                };
                self.pos += 1;
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0C'),
                    b'a' => out.push('\x07'),
                    b'v' => out.push('\x0B'),
                    b'0'..=b'7' => {
                        // Octal escape: up to 3 digits total.
                        let mut val = (e - b'0') as u32;
                        let mut n = 1;
                        while n < 3 {
                            match self.s.get(self.pos) {
                                Some(&d @ b'0'..=b'7') => {
                                    val = val * 8 + (d - b'0') as u32;
                                    self.pos += 1;
                                    n += 1;
                                }
                                _ => break,
                            }
                        }
                        out.push(char::from_u32(val).ok_or(())?);
                    }
                    b'x' => {
                        let h = self.hex_digits(2)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'u' => {
                        let h = self.hex_digits(4)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'U' => {
                        let h = self.hex_digits(8)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'\n' => {} // line continuation
                    b'N' => {
                        // `\N` is a recognized escape introducer: a bare
                        // `\N` is a hard SyntaxError, and every `\N{NAME}`
                        // needs a Unicode-name table we don't carry. Reject
                        // rather than ship the literal escape text as the
                        // argument.
                        return Err(());
                    }
                    _ => {
                        // `\'` `\"` `\\` collapse to the literal char; every
                        // OTHER unrecognized escape keeps the backslash —
                        // Python `'\d'` evaluates to `"\\d"` (backslash
                        // preserved), so `pattern='\d+'` must not lose it.
                        let e = self.s[self.pos - 1];
                        if !matches!(e, b'\'' | b'"' | b'\\') {
                            out.push('\\');
                        }
                        let rest = &self.s[self.pos - 1..];
                        let ch_len = utf8_len(rest[0]);
                        if rest.len() < ch_len {
                            return Err(());
                        }
                        out.push_str(std::str::from_utf8(&rest[..ch_len]).map_err(|_| ())?);
                        self.pos += ch_len - 1;
                    }
                }
                continue;
            }
            if f_mode && (b == b'{' || b == b'}') {
                // `{{`/`}}` are escaped literal braces — a pure-constant
                // f-string like f'{{x}}' evaluates to "{x}" (vLLM accepts;
                // JoinedStr of constants). A single brace is a real
                // placeholder → reject.
                if self.s.get(self.pos + 1) == Some(&b) {
                    out.push(b as char);
                    self.pos += 2;
                    continue;
                }
                return Err(());
            }
            // Copy one UTF-8 char verbatim.
            let rest = &self.s[self.pos..];
            let ch_len = utf8_len(rest[0]);
            if rest.len() < ch_len {
                return Err(());
            }
            out.push_str(std::str::from_utf8(&rest[..ch_len]).map_err(|_| ())?);
            self.pos += ch_len;
        }
        Ok(out)
    }

    fn hex_digits(&mut self, n: usize) -> Result<u32, ()> {
        let mut val = 0u32;
        for _ in 0..n {
            let Some(&d) = self.s.get(self.pos) else {
                return Err(());
            };
            let Some(v) = (d as char).to_digit(16) else {
                return Err(());
            };
            val = val * 16 + v;
            self.pos += 1;
        }
        Ok(val)
    }

    /// `_` separator placement per PEP 515: a separator must sit between
    /// two digits (radix digits for base-prefixed literals), except the
    /// single `_` allowed right after `0x`/`0o`/`0b` (`0x_ff` is legal).
    /// `after_prefix` marks `s[0]` as the position following a base prefix.
    fn valid_numeric_underscores(
        s: &[u8],
        is_digit: impl Fn(u8) -> bool,
        after_prefix: bool,
    ) -> bool {
        for (i, &c) in s.iter().enumerate() {
            if c != b'_' {
                continue;
            }
            let next_ok = s.get(i + 1).is_some_and(|&n| is_digit(n));
            let prev_ok = i > 0 && is_digit(s[i - 1]);
            if !next_ok || !(prev_ok || (after_prefix && i == 0)) {
                return false;
            }
        }
        true
    }

    /// Number: dec/hex/oct/bin int or float, optional leading `+`/`-` is
    /// handled by the caller (unary op). Leading zeros are tolerated
    /// (vLLM `normalize_leading_zero_ints`).
    fn parse_number(&mut self) -> Result<Value, ()> {
        let start = self.pos;
        if self.peek() == Some(b'0')
            && matches!(
                self.s.get(self.pos + 1),
                Some(b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
            )
        {
            let radix = match self.s[self.pos + 1].to_ascii_lowercase() {
                b'x' => 16,
                b'o' => 8,
                _ => 2,
            };
            self.pos += 2;
            let dstart = self.pos;
            while let Some(&d) = self.s.get(self.pos) {
                if (d as char).is_digit(radix) || d == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if !Self::valid_numeric_underscores(
                &self.s[dstart..self.pos],
                |c| (c as char).is_digit(radix),
                true,
            ) {
                return Err(());
            }
            let digits: String = std::str::from_utf8(&self.s[dstart..self.pos])
                .map_err(|_| ())?
                .chars()
                .filter(|&c| c != '_')
                .collect();
            if digits.is_empty() {
                return Err(());
            }
            let v = i128::from_str_radix(&digits, radix).map_err(|_| ())?;
            return i128_to_value(v).ok_or(());
        }
        let mut is_float = false;
        while let Some(&d) = self.s.get(self.pos) {
            match d {
                b'0'..=b'9' | b'_' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    // +/- only legal right after e/E.
                    if d == b'+' || d == b'-' {
                        let prev = self.s.get(self.pos.wrapping_sub(1)).copied();
                        if !matches!(prev, Some(b'e' | b'E')) {
                            break;
                        }
                    }
                    is_float = true;
                    self.pos += 1;
                }
                b'j' | b'J' => return Err(()), // complex: not JSON
                _ => break,
            }
        }
        if !Self::valid_numeric_underscores(&self.s[start..self.pos], |c| c.is_ascii_digit(), false)
        {
            return Err(());
        }
        let text: String = std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| ())?
            .chars()
            .filter(|&c| c != '_')
            .collect();
        // Dispatch guarantees the first char is a digit or `.`, so `text`
        // can only start with those; a `.`-led scan already set `is_float`.
        if text.is_empty() || !text.bytes().any(|b| b.is_ascii_digit()) {
            return Err(());
        }
        if is_float {
            // Tolerate the Python forms Rust f64 rejects: leading-dot
            // `.5` → `0.5`, trailing-dot `1.` → `1.0`.
            let normalized = if let Some(rest) = text.strip_prefix('.') {
                format!("0.{rest}")
            } else if text.ends_with('.') {
                format!("{text}0")
            } else {
                text
            };
            match normalized.parse::<f64>() {
                // Overflow parses to ±inf and Value::from(inf) serializes
                // as null — no exact JSON form → verbatim, like huge ints.
                Ok(f) if f.is_finite() => Ok(Value::from(f)),
                _ => Err(()),
            }
        } else {
            text.parse::<i128>().ok().and_then(i128_to_value).ok_or(())
        }
    }

    /// Parse one literal value: string, number, name constant, or
    /// container. Unary +/- over a numeric operand is folded here
    /// (vLLM `get_parameter_value` UnaryOp branch).
    fn parse_literal(&mut self) -> Result<Value, ()> {
        self.depth += 1;
        if self.depth > MAX_PY_LITERAL_DEPTH {
            self.depth -= 1;
            return Err(());
        }
        let v = self.parse_literal_inner();
        self.depth -= 1;
        v
    }

    fn parse_literal_inner(&mut self) -> Result<Value, ()> {
        self.skip_ws();
        match self.peek() {
            Some(b'-' | b'+') => {
                let neg = self.s[self.pos] == b'-';
                self.pos += 1;
                let v = self.parse_literal()?;
                match v {
                    Value::Number(n) => {
                        if neg {
                            if let Some(i) = n.as_i64() {
                                // `-i64::MIN` overflows i64 (double negation
                                // like `--9223372036854775808`) — the positive
                                // fits u64 exactly.
                                Ok(match i.checked_neg() {
                                    Some(v) => Value::from(v),
                                    None => Value::from(i.unsigned_abs()),
                                })
                            } else if let Some(u) = n.as_u64() {
                                // Negating a u64: exact i64::MIN edge or a
                                // value that fits in i64 — anything larger
                                // has no exact serde_json form → reject.
                                if u == (i64::MAX as u64) + 1 {
                                    return Ok(Value::from(i64::MIN));
                                }
                                let i = i64::try_from(u).map_err(|_| ())?;
                                Ok(Value::from(-i))
                            } else {
                                n.as_f64().map(|f| Value::from(-f)).ok_or(())
                            }
                        } else {
                            Ok(Value::Number(n))
                        }
                    }
                    _ => Err(()), // unary on non-number → reject
                }
            }
            Some(b'\'' | b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
                if matches!(
                    self.s.get(self.pos + 1..self.pos + 3).unwrap_or(&[]),
                    [b'\'' | b'"', ..] | [b'r' | b'R' | b'u' | b'U' | b'f' | b'F', b'\'' | b'"']
                ) =>
            {
                Ok(Value::String(self.parse_string()?))
            }
            Some(b'0'..=b'9') | Some(b'.') => self.parse_number(),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.parse_literal()?);
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.pos += 1;
                            self.skip_ws();
                            if self.peek() == Some(b']') {
                                self.pos += 1;
                                return Ok(Value::Array(items));
                            }
                        }
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(()),
                    }
                }
            }
            Some(b'(') => {
                // Tuple → JSON array. `(x)` is just x; `(x,)` is a tuple.
                self.pos += 1;
                self.skip_ws();
                if self.peek() == Some(b')') {
                    self.pos += 1;
                    return Ok(Value::Array(vec![]));
                }
                let first = self.parse_literal()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        let mut items = vec![first];
                        while self.peek() == Some(b',') {
                            self.pos += 1;
                            self.skip_ws();
                            if self.peek() == Some(b')') {
                                break;
                            }
                            items.push(self.parse_literal()?);
                            self.skip_ws();
                        }
                        self.expect(b')')?;
                        Ok(Value::Array(items))
                    }
                    Some(b')') => {
                        self.pos += 1;
                        Ok(first) // parenthesized value
                    }
                    _ => Err(()),
                }
            }
            Some(b'{') => {
                self.pos += 1;
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(Value::Object(serde_json::Map::new()));
                }
                let first = self.parse_literal()?;
                self.skip_ws();
                if self.peek() == Some(b':') {
                    // dict — keys must be literal (already parsed); Python
                    // allows any hashable; JSON needs strings.
                    self.pos += 1;
                    let mut map = serde_json::Map::new();
                    let key = literal_key_to_string(&first)?;
                    map.insert(key, self.parse_literal()?);
                    loop {
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                                self.skip_ws();
                                if self.peek() == Some(b'}') {
                                    self.pos += 1;
                                    return Ok(Value::Object(map));
                                }
                                let k = self.parse_literal()?;
                                self.skip_ws();
                                self.expect(b':')?;
                                map.insert(literal_key_to_string(&k)?, self.parse_literal()?);
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                return Ok(Value::Object(map));
                            }
                            _ => return Err(()),
                        }
                    }
                } else {
                    // set literal `{v, v}` → array (vLLM maps sets to lists)
                    let mut items = vec![first];
                    loop {
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                                self.skip_ws();
                                if self.peek() == Some(b'}') {
                                    self.pos += 1;
                                    return Ok(Value::Array(items));
                                }
                                items.push(self.parse_literal()?);
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                return Ok(Value::Array(items));
                            }
                            _ => return Err(()),
                        }
                    }
                }
            }
            _ => {
                let name = self.ident()?;
                match name {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Null),
                    _ => Err(()), // bare name → not a literal
                }
            }
        }
    }

    /// One call element: `name(kw=literal, ...)`. Positional arguments are
    /// skipped then dropped (vLLM iterates `call.keywords` only); `**kw`
    /// rejects the call outright (vLLM fails on `arguments[None]`). A
    /// positional AFTER a keyword (`f(x=1, 5)`) mirrors `ast.parse`'s
    /// `SyntaxError`; duplicate keywords keep their final value like vLLM.
    fn parse_call(&mut self) -> Result<(String, Value), ()> {
        let name = self.dotted_name()?;
        self.expect(b'(')?;
        let mut args = serde_json::Map::new();
        let mut seen_kwarg = false;
        self.skip_ws();
        if self.peek() == Some(b')') {
            self.pos += 1;
            return Ok((name, Value::Object(args)));
        }
        loop {
            self.skip_ws();
            if self.peek() == Some(b'*') {
                // `*x` spread: ignored like other positionals (legal even
                // after keywords); `**x` → reject (vLLM's
                // `arguments[None]` failure).
                self.pos += 1;
                if self.peek() == Some(b'*') {
                    return Err(());
                }
                self.skip_expression(true)?;
            } else {
                // `kw=literal` or a positional expression. Try the
                // `ident =` lookahead first (excluding `==`); on no match
                // rewind and skip the positional, which vLLM drops.
                // Reserved-word kwargs (`from=1`) parse natively here —
                // vLLM needs its rename/restore dance, we don't.
                let save = self.pos;
                let kw: Option<String> = match self.ident() {
                    Ok(id) => {
                        let id: String = id.nfkc().collect();
                        self.skip_ws();
                        if self.peek() == Some(b'=') && self.s.get(self.pos + 1) != Some(&b'=') {
                            self.pos += 1;
                            Some(id)
                        } else {
                            self.pos = save;
                            None
                        }
                    }
                    // `ident` consumes leading digits before rejecting a
                    // digit start — REWIND so a numeric positional
                    // (`f(5)`, `f(0x10)`) still scans as an expression.
                    Err(()) => {
                        self.pos = save;
                        None
                    }
                };
                match kw {
                    Some(k) => {
                        seen_kwarg = true;
                        let v = self.parse_literal()?;
                        let _ = args.insert(k, v);
                    }
                    None => {
                        // `f(x=1, 5)` → `SyntaxError: positional argument
                        // follows keyword argument`.
                        if seen_kwarg {
                            return Err(());
                        }
                        self.skip_expression(false)?;
                    }
                }
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                    if self.peek() == Some(b')') {
                        self.pos += 1;
                        return Ok((name, Value::Object(args)));
                    }
                }
                Some(b')') => {
                    self.pos += 1;
                    return Ok((name, Value::Object(args)));
                }
                _ => return Err(()),
            }
        }
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

/// Is the char at `s[p]` identifier syntax? ASCII `[A-Za-z_]` (or
/// `[A-Za-z0-9_]` when `start == false`) plus the Unicode XID_Start /
/// XID_Continue classes CPython uses — `café`/`变量` are names, `😀`
/// is a SyntaxError (`ast.parse` rejects it → vLLM keeps the block
/// verbatim; a bare `>= 0x80` byte gate would promote it).
fn ident_char_at(s: &[u8], p: usize, start: bool) -> bool {
    let Some(&c) = s.get(p) else {
        return false;
    };
    if c < 0x80 {
        return if start {
            c.is_ascii_alphabetic() || c == b'_'
        } else {
            c.is_ascii_alphanumeric() || c == b'_'
        };
    }
    let Some(t) = s
        .get(p..p + utf8_len(c))
        .and_then(|b| std::str::from_utf8(b).ok())
    else {
        return false;
    };
    let Some(ch) = t.chars().next() else {
        return false;
    };
    if start {
        unicode_ident::is_xid_start(ch)
    } else {
        unicode_ident::is_xid_continue(ch)
    }
}

/// End of the identifier starting at `p` — ASCII ident bytes plus
/// XID_Continue chars; everything else stops the scan.
fn ident_end(s: &[u8], mut p: usize) -> usize {
    while let Some(&c) = s.get(p) {
        if c.is_ascii_alphanumeric() || c == b'_' {
            p += 1;
        } else if c >= 0x80 && ident_char_at(s, p, false) {
            p += utf8_len(c);
        } else {
            break;
        }
    }
    p
}

fn i128_to_value(v: i128) -> Option<Value> {
    if let Ok(i) = i64::try_from(v) {
        Some(Value::from(i))
    } else if let Ok(u) = u64::try_from(v) {
        Some(Value::from(u))
    } else {
        // serde_json holds no exact representation past u64::MAX — f64
        // rounding would silently corrupt the argument (IDs, counters,
        // monetary units), so the block stays verbatim like every other
        // non-JSON-representable literal (`b'..'`, `1j`).
        None
    }
}

/// Python dict keys are any hashable; JSON keys are strings — stringify
/// non-string literal keys the way `json.dumps` does (`{True: 1}` →
/// `{"true": 1}`, `{None: 1}` → `{"null": 1}`).
fn literal_key_to_string(v: &Value) -> Result<String, ()> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Null => Ok("null".into()),
        _ => Err(()), // list/dict keys are unhashable in Python too
    }
}

/// Parse the bracket-list body (`f(x=1), g(y='a')` — without the outer
/// `[]`) into `(name, args)` pairs. `None` on any parse failure, a trailing
/// element, or leftover input. A trailing comma (`[f(),]`) is legal Python
/// and accepted.
fn parse_lfm2_call_list(body: &str) -> Option<Vec<(String, Value)>> {
    let mut p = PyLiteralParser::new(body);
    let mut calls = Vec::new();
    p.skip_ws();
    p.peek()?; // `[]` — vLLM requires non-empty elts
    loop {
        let (name, args) = p.parse_call().ok()?;
        calls.push((name, args));
        p.skip_ws();
        match p.peek() {
            Some(b',') => {
                p.pos += 1;
                p.skip_ws();
                if p.peek().is_none() {
                    break; // trailing comma at end of list
                }
            }
            None => break,
            _ => return None,
        }
    }
    (p.eof() && !calls.is_empty()).then_some(calls)
}

/// vLLM `escape_nested_quotes_in_strings` (`tool_parsers/utils.py`): close a
/// broken string literal at the only closing quote that works. Models
/// emitting shell commands nest unescaped same-style quotes inside a string
/// argument (`command='sed -n '1,9p' f.py'`). A string is broken when its
/// first unescaped quote cannot syntactically close it (the next non-space
/// char is none of `,)]}:`); then every plausible closing quote is tried —
/// interior quotes escaped — and the candidate kept only when it is the
/// UNIQUE candidate that re-parses as a call list. Returns `None` on zero
/// or ambiguous candidates.
///
/// vLLM validates each candidate with `ast.parse` (any expression) and only
/// afterwards requires all-`Call` elements — so a second candidate that
/// parses as a non-call expression still counts as ambiguity there. Our
/// validator is `parse_lfm2_call_list` (all-calls up front), which makes
/// the recovery strictly more permissive on that corner: accepted here,
/// declared ambiguous upstream.
fn escape_lfm2_nested_quotes(text: &str) -> Option<String> {
    fn unescaped_quote_positions(text: &str, start: usize, quote: u8) -> Vec<usize> {
        let s = text.as_bytes();
        let mut out = Vec::new();
        let mut j = start;
        while j < s.len() {
            if s[j] == b'\\' {
                j += 2;
                continue;
            }
            if s[j] == quote {
                out.push(j);
            }
            j += 1;
        }
        out
    }
    // vLLM `_QUOTE_FOLLOWERS` — a quote closes the string only when the
    // next non-space byte is a container/argument delimiter.
    fn is_closer(text: &str, pos: usize) -> bool {
        let s = text.as_bytes();
        let mut k = pos + 1;
        while k < s.len() && s[k].is_ascii_whitespace() {
            k += 1;
        }
        k < s.len() && matches!(s[k], b',' | b')' | b']' | b'}' | b':')
    }
    // vLLM `_is_escaped`: escaped iff preceded by an ODD backslash run.
    fn is_escaped(s: &[u8], index: usize) -> bool {
        let mut n = 0usize;
        let mut j = index;
        while j > 0 && s[j - 1] == b'\\' {
            n += 1;
            j -= 1;
        }
        n % 2 == 1
    }

    let s = text.as_bytes();
    let mut index = 0usize;
    while index < s.len() {
        let quote = s[index];
        if quote != b'\'' && quote != b'"' {
            index += 1;
            continue;
        }
        let quotes = unescaped_quote_positions(text, index + 1, quote);
        let Some(&first) = quotes.first() else {
            return None; // unterminated — nothing to requote
        };
        if is_closer(text, first) {
            index = first + 1; // normal string; skip past it
            continue;
        }
        // Broken string: try each plausible close, escaping interior
        // same-quotes; keep the candidate only when exactly one parses.
        let mut winner: Option<String> = None;
        for &close in quotes.iter().filter(|&&j| is_closer(text, j)) {
            let mut candidate = String::with_capacity(text.len() + 8);
            candidate.push_str(&text[..index + 1]); // through opening quote
            for (off, ch) in text[index + 1..close].char_indices() {
                if ch == quote as char && !is_escaped(s, index + 1 + off) {
                    candidate.push('\\');
                }
                candidate.push(ch);
            }
            candidate.push(quote as char);
            candidate.push_str(&text[close + 1..]);
            // The candidate is the full bracketed list; validate by parsing
            // the bracket-free body.
            let valid = {
                let c = candidate.trim();
                c.starts_with('[')
                    && c.ends_with(']')
                    && parse_lfm2_call_list(&c[1..c.len() - 1]).is_some()
            };
            if valid {
                if winner.is_some() {
                    return None; // ambiguous — don't guess
                }
                winner = Some(candidate);
            }
        }
        return winner;
    }
    None
}

/// Parse LFM2 sentinel blocks: every `<|tool_call_start|>` …
/// `<|tool_call_end|>` pair (or end of text when the stream cut mid-call).
/// Returns (cleaned_text, calls). Parse failure returns the ORIGINAL text
/// verbatim with no calls (vLLM `content=model_output` semantics).
fn parse_lfm2_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    // A sentinel inside a `<tool_call>` block is literal argument text
    // (e.g. `{"a": "<|tool_call_start|>…"}`) — not LFM2 markup. Exclude
    // every `<tool_call>` open's span (through its close, or EOF when
    // unclosed) from the sentinel search so a sentinel can never be
    // promoted to a real call out of another family's argument string.
    let protected: Vec<(usize, usize)> = all_positions(text, "<tool_call>")
        .into_iter()
        .map(|open| {
            let end = text[open..]
                .find("</tool_call>")
                .map(|c| open + c + "</tool_call>".len())
                .unwrap_or(text.len());
            (open, end)
        })
        .collect();
    let unprotected = |p: usize| !protected.iter().any(|&(s, e)| p >= s && p < e);
    let next_start = |from: usize| -> Option<usize> {
        let mut search_from = from;
        loop {
            let idx = search_from + text[search_from..].find(LFM2_TOOL_CALL_START)?;
            if unprotected(idx) {
                return Some(idx);
            }
            search_from = idx + LFM2_TOOL_CALL_START.len();
        }
    };

    // Success content strips each sentinel block VERBATIM — the streamed
    // path emits the surrounding fragments byte-for-byte, so a rewritten
    // join (e.g. vLLM's "\n") would make the done-path suffix recovery
    // see the finalized text as entirely unsent and emit it twice.
    let mut calls: Vec<ToolCallResult> = Vec::new();
    let mut cleaned = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(start_idx) = next_start(cursor) {
        cleaned.push_str(&text[cursor..start_idx]);
        let inner_start = start_idx + LFM2_TOOL_CALL_START.len();
        let (inner, block_end) = match text[inner_start..].find(LFM2_TOOL_CALL_END) {
            Some(e) => {
                let end = inner_start + e + LFM2_TOOL_CALL_END.len();
                (&text[inner_start..inner_start + e], end)
            }
            None => (&text[inner_start..], text.len()),
        };
        let tool_text = inner.trim();
        // vLLM `TOOL_CALL_REGEX`: bracketed list only.
        let parsed = if tool_text.starts_with('[') && tool_text.ends_with(']') {
            let inner_body = &tool_text[1..tool_text.len() - 1];
            // Direct parse first; on failure, the nested-quote recovery
            // rewrites `command='sed -n '1,9p' f.py'` shapes and re-parses
            // (vLLM runs `escape_nested_quotes_in_strings` over the
            // ast.parse failure). The recovery sees the BRACKETED text —
            // the trailing `)]` is what makes the outer quote a plausible
            // close.
            parse_lfm2_call_list(inner_body).or_else(|| {
                escape_lfm2_nested_quotes(tool_text).and_then(|fixed| {
                    let f = fixed.trim();
                    if f.starts_with('[') && f.ends_with(']') {
                        parse_lfm2_call_list(&f[1..f.len() - 1])
                    } else {
                        None
                    }
                })
            })
        } else {
            None
        };
        let Some(parsed_calls) = parsed else {
            if calls.is_empty() {
                // The FIRST block failing is vLLM's `content=model_output`:
                // ast.parse on the only block it reads failed, so the whole
                // output stays verbatim with no calls.
                return (text.to_string(), Vec::new());
            }
            // A LATER block failing is out of vLLM's reach — it never
            // parses past the first end sentinel, so the malformed span
            // is echo region, not a call. Keep the calls already promoted
            // and echo-strip from this sentinel on (an orphan END caps
            // the echoed body).
            cleaned.push_str(strip_lfm2_echo(&text[start_idx..]));
            return (cleaned.trim().to_string(), calls);
        };
        let raw = &text[start_idx..block_end];
        calls.extend(
            parsed_calls
                .into_iter()
                .map(|(name, args)| ToolCallResult::ok(name, args, raw.to_string())),
        );
        // The echo region runs from this block's end to the NEXT
        // unprotected start sentinel (or EOF): drop everything through
        // the last orphan `<|tool_call_end|>` inside it, keep the prose.
        let next = next_start(block_end).unwrap_or(text.len());
        cleaned.push_str(strip_lfm2_echo(&text[block_end..next]));
        cursor = next;
    }
    if calls.is_empty() {
        return (text.to_string(), Vec::new());
    }
    cleaned.push_str(&text[cursor..]);
    (cleaned.trim().to_string(), calls)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse tool calls from generated text
///
/// Returns (cleaned_text, tool_calls) where:
/// - `cleaned_text` has all tool-call markup removed
/// - `tool_calls` contains all parsed tool calls with status info
///
/// Supports four formats:
/// - JSON (Qwen3): `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Function (Qwen3.5): `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
/// - XML (legacy): `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
/// - Pythonic (LFM2/LFM2.5): `<|tool_call_start|>[func(arg='v')]<|tool_call_end|>`
///
/// The LFM2 sentinel block is extracted first because its echo suppression
/// owns the tail after the first `<|tool_call_end|>`; a mixed-format output
/// orders LFM2 calls before `<tool_call>` calls.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let (text, mut tool_calls) = parse_lfm2_tool_calls(text);

    let blocks = extract_tag_blocks(&text, "<tool_call>", "</tool_call>");
    for (start, end, inner) in &blocks {
        let raw_content = &text[*start..*end];
        if let Some(result) = classify_and_parse_tool_call(inner, raw_content) {
            tool_calls.push(result);
        }
    }

    let cleaned_text = strip_tag_blocks(&text, "<tool_call>", "</tool_call>");
    (cleaned_text, tool_calls)
}

/// Check if text contains any tool call tags
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>") || text.contains(LFM2_TOOL_CALL_START)
}

/// Parse thinking content from generated text
///
/// Returns (cleaned_text, thinking_content) where:
/// - `cleaned_text` has all `<think>...</think>` tags removed
/// - `thinking_content` is the extracted content from within the tags (None if no tags found)
///
/// If multiple `<think>` blocks exist, they are concatenated with newlines.
///
/// Also handles the case where the chat template already added `<think>\n` as part
/// of the assistant generation prompt — the generated text then starts with thinking
/// content followed by `</think>` but without the opening `<think>` tag. To avoid
/// misinterpreting literal `</think>` in non-thinking output (e.g., the model
/// explaining XML tags), the fallback only applies when `</think>` is followed by
/// a newline or end-of-text — not when it's embedded mid-sentence.
pub fn parse_thinking(text: &str) -> (String, Option<String>) {
    // Check both <think> and <longcat_think> paired blocks.
    for (open, close) in [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ] {
        let blocks = extract_tag_blocks(text, open, close);
        if !blocks.is_empty() {
            let thinking_parts: Vec<&str> = blocks
                .iter()
                .map(|(_, _, inner)| inner.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let thinking = if thinking_parts.is_empty() {
                None
            } else {
                Some(thinking_parts.join("\n\n"))
            };

            let cleaned_text = strip_tag_blocks(text, open, close);
            return (cleaned_text, thinking);
        }
    }

    // Handle missing opening tag (template already added it as prefix).
    // The template adds `<think>\n` (or `<longcat_think>\n`) as the assistant
    // generation prompt, so the model's output starts with thinking + close tag.
    //
    // To distinguish from literal close tags in content, only apply when the
    // close tag is followed by a newline or end-of-text.
    //
    // NOTE: this is the GENERIC parser, shared by `parse_generation_output`
    // (GRPO reward parsing), Qianfan OCR, and chat finalize — it must NOT
    // aggressively reinterpret arbitrary completions, so it stops at the FIRST
    // close tag. The reasoning-suppression scrubber
    // (`strip_reasoning_preserving_tools`) has its own missing-open scanner
    // (`missing_open_close`) that scans past literal closes to the real terminator.
    for close_tag in ["</think>", "</longcat_think>"] {
        if let Some(close_pos) = text.find(close_tag) {
            let after_tag = &text[close_pos + close_tag.len()..];
            if after_tag.is_empty() || after_tag.starts_with('\n') {
                let thinking_content = text[..close_pos].trim();
                let after = after_tag.trim();
                let thinking = if thinking_content.is_empty() {
                    None
                } else {
                    Some(thinking_content.to_string())
                };
                return (after.to_string(), thinking);
            }
        }
    }

    (text.to_string(), None)
}

/// Strip reasoning (`<think>`/`<longcat_think>` blocks, both families) from `text`
/// while preserving tool-call spans that are NOT themselves part of a
/// reasoning block — both `<tool_call>…</tool_call>` and LFM2
/// `<|tool_call_start|>…<|tool_call_end|>` spans.
///
/// Used to scrub reasoning from `raw_text` on the no-`</think>`-token fallback path.
/// Three requirements: reasoning-looking tags *inside* a tool-call argument (a literal
/// `<think>…</think>` or a bare `</think>`) must NOT be treated as reasoning delimiters
/// (else the recovered tool call is mangled); a tool span nested inside a reasoning block
/// must be dropped along with the reasoning (it is suppressed chain-of-thought); and a tool
/// span that STRADDLES a reasoning boundary (opens inside `<think>` but its `</tool_call>`
/// lands after `</think>`, or vice versa) must NOT be allowed to surface — neither leaking
/// the reasoning prefix nor presenting a tool call that began inside reasoning.
///
/// Implementation is purely RANGE-based:
///   1. Tool spans are taken as opaque byte ranges.
///   2. Reasoning ranges are the paired `<think>`/`<longcat_think>` blocks on the ORIGINAL
///      text, MINUS any block wholly contained in a tool span (those are literal argument
///      text). The template "missing-open" case is also handled (a bare close at the
///      injected depth-1 level, newline/EOF-terminated, marks a leading reasoning prefix),
///      mirroring `parse_thinking`. The terminator applied (`applied_missing_open`) PREFERS a
///      top-level close over an in-tool straddle across both families (see that fn).
///   3. The removal set is the reasoning ranges UNION every tool span that overlaps any
///      reasoning range (nested or straddling — entangled with reasoning, so dropped). Tool
///      spans disjoint from all reasoning are PRESERVED verbatim (the `kept` set).
///   4. Emit `text` minus the merged removal ranges.
///   5. SYNTHESIS DEFENSE (RANGE provenance): deleting a reasoning block flanked by tool-tag
///      fragments (e.g. `<tool` <reasoning> `_call>`) would FUSE them into a `<tool_call>` span
///      that never existed as a preserved call — which `parse_tool_calls` would treat as
///      executable. Each `kept` span is mapped to its OUTPUT byte range and only those exact
///      ranges are retained (`keep_only_genuine_tool_spans`). Provenance must be by RANGE, not
///      bytes: a fabricated span can be byte-identical to a genuine call (a substring test, or a
///      per-string budget, would falsely keep the look-alike and drop the real one).
///
/// A single pass strips every paired block plus the LEADING missing-open span. The entry point
/// then iterates the pass to a FIXPOINT so SUCCESSIVE missing-open spans (a second bare close
/// after the first) are all removed — using the scrubber's own `missing_open_close` (which
/// scans past literal closes) instead of the generic `parse_thinking`. Each pass strictly
/// shrinks the text or leaves it unchanged, so it terminates.
///
/// Iteration is gated on whether the CURRENT OUTPUT still has a TOP-LEVEL (definitive)
/// missing-open terminator (`has_top_level_missing_open_terminator`), NOT on mere tool-span
/// presence. The missing-open heuristic models byte 0 as template-injected reasoning; that is
/// true of the original generation and of any remainder that still LEADS with reasoning, but
/// NOT of an output whose applied terminator is an IN-TOOL straddle with no top-level close past
/// it — that is a PRESERVED tool call whose own argument carries `</think>`, and re-running
/// would treat it as a byte-0 straddle and wrongly drop the valid call. So a pass keeps
/// iterating only while the terminator it would apply is top-level (genuine reasoning remains),
/// and halts the moment the applied terminator is in-tool (or none remains). This correctly
/// handles: a reasoning-internal tool call dropped within a pass (its remainder keeps
/// iterating); a second missing-open span that itself contains a reasoning-internal tool call
/// (a top-level terminator survives the first pass, so iteration continues and drops it); a
/// span whose tool call precedes a LATER top-level close of either family (the close extends
/// the reasoning over the call, dropping it — `applied_missing_open` prefers that top-level
/// close over the call's earlier in-tool argument close); and a genuine post-reasoning call
/// whose only following close is its own in-tool argument close (no top-level close past it, so
/// iteration halts and preserves the call). Multiple in-tool straddle candidates within one
/// pass are resolved by last-wins in `missing_open_close`.
pub fn strip_reasoning_preserving_tools(text: &str) -> String {
    let mut current = strip_reasoning_once(text);
    while has_top_level_missing_open_terminator(&current) {
        let next = strip_reasoning_once(&current);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// One pass of the range-based reasoning scrub (see `strip_reasoning_preserving_tools`).
fn strip_reasoning_once(text: &str) -> String {
    let mut tool_ranges: Vec<(usize, usize)> =
        extract_tag_blocks(text, "<tool_call>", "</tool_call>")
            .into_iter()
            .map(|(s, e, _)| (s, e))
            .collect();
    // LFM2's pythonic sentinel blocks get the same protection: a call nested
    // inside reasoning is scrubbed with it; a top-level call is preserved
    // verbatim so `parse_lfm2_tool_calls` still sees it.
    tool_ranges.extend(
        extract_tag_blocks(text, LFM2_TOOL_CALL_START, LFM2_TOOL_CALL_END)
            .into_iter()
            .map(|(s, e, _)| (s, e)),
    );
    // The logic below handles the no-tool case for free (empty `tool_ranges` ⇒ every tag is
    // top-level and no spans are dropped), so there is NO separate `parse_thinking` fast path:
    // the scrubber owns its missing-open scanner (`missing_open_close`) and never delegates to
    // the generic `parse_thinking`, which intentionally keeps weaker (first-close-only)
    // missing-open semantics for `parse_generation_output`/OCR/reward callers.

    // Paired reasoning ranges on the original text, excluding any block that is literal
    // argument text inside a tool span.
    let mut reasoning: Vec<(usize, usize)> = Vec::new();
    for (open, close) in [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ] {
        for (start, end, _inner) in extract_tag_blocks(text, open, close) {
            let in_arg = tool_ranges.iter().any(|(s, e)| start >= *s && end <= *e);
            if !in_arg {
                reasoning.push((start, end));
            }
        }
    }

    // Missing-open template case: the generation begins mid-reasoning (the template injected
    // the opener into the prompt) and emits a bare close. The leading prefix up to the applied
    // terminator (earliest qualifying close across families — see `applied_missing_open` /
    // `missing_open_close`) is reasoning. It composes with the paired set above.
    if let Some((_, close_end, _)) = applied_missing_open(text, &tool_ranges) {
        reasoning.push((0, close_end));
    }

    if reasoning.is_empty() {
        // No reasoning to strip; keep everything (tool spans verbatim).
        return text.to_string();
    }

    // Removal = reasoning ranges ∪ tool spans entangled with reasoning (nested or
    // straddling). A tool span overlapping any reasoning range is part of the suppressed
    // chain-of-thought and must not surface. The remaining (disjoint) tool spans are the
    // PRESERVED set — the only `<tool_call>` spans allowed to appear in the output (see the
    // synthesis defense below).
    let overlaps = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
    let mut removal = reasoning.clone();
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for &(ts, te) in &tool_ranges {
        if reasoning.iter().any(|r| overlaps((ts, te), *r)) {
            removal.push((ts, te));
        } else {
            kept.push((ts, te));
        }
    }
    removal.sort_by_key(|(s, _)| *s);

    // Emit `text` minus the merged removal ranges, recording each copied chunk as
    // (in_start, in_end, out_start) so preserved tool spans can be mapped to their OUTPUT byte
    // ranges (range provenance, below).
    let mut out = String::with_capacity(text.len());
    let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    for (s, e) in removal {
        let s = s.max(cursor);
        if s >= e {
            continue; // already consumed by an earlier overlapping range
        }
        if s > cursor {
            chunks.push((cursor, s, out.len()));
            out.push_str(&text[cursor..s]);
        }
        cursor = e.max(cursor);
    }
    if cursor < text.len() {
        chunks.push((cursor, text.len(), out.len()));
        out.push_str(&text[cursor..]);
    }

    // Synthesis defense (RANGE provenance): a removal seam can fuse tool-tag fragments (e.g.
    // `<tool` <reasoning> `_call>`) into a `<tool_call>` span that never existed as a preserved
    // call — which `parse_tool_calls` would treat as executable. Map each PRESERVED span to its
    // output range (it lies wholly within ONE copied chunk — no removal is ever strictly inside
    // a preserved span — so the map is exact) and keep ONLY those ranges. Provenance is by RANGE,
    // not bytes: a fabricated span can be byte-identical to a genuine call (e.g. a duplicate), so
    // only the output ranges can tell them apart.
    let genuine: Vec<(usize, usize)> = kept
        .iter()
        .filter_map(|&(ts, te)| {
            chunks.iter().find_map(|&(cs, ce, os)| {
                (ts >= cs && te <= ce).then(|| (os + (ts - cs), os + (te - cs)))
            })
        })
        .collect();
    let out = keep_only_genuine_tool_spans(out, genuine);
    out.trim().to_string()
}

/// The missing-open terminator that `strip_reasoning_once` applies, together with whether it is
/// TOP-LEVEL (definitive, outside every tool span) vs an in-tool straddle. `None` if no family
/// has a terminator.
///
/// Selection across families PREFERS a definitive top-level close over a tentative in-tool
/// straddle: the earliest TOP-LEVEL close across both families wins; only if NEITHER family has
/// a top-level close is the (latest) in-tool straddle used. This matters when one family's
/// in-tool straddle is positionally earlier than the other family's top-level close — the
/// reasoning genuinely extends to the top-level close (dropping any tool call that opened
/// before it), so chasing the earlier in-tool straddle would halt early and leak. A straddle
/// is the real terminator only when no top-level close exists at all (a tool call that opened
/// mid-reasoning whose own argument carries the close). When the straddle IS used, the latest
/// one wins so the reasoning range reaches every straddling span (matching `missing_open_close`'s
/// within-family last-wins). This is the single source of truth for both the strip (byte range)
/// and the fixpoint gate (top-level flag), so they never disagree.
fn applied_missing_open(
    text: &str,
    tool_ranges: &[(usize, usize)],
) -> Option<(usize, usize, bool)> {
    let mut top_level: Option<(usize, usize)> = None; // earliest top-level close
    let mut straddle: Option<(usize, usize)> = None; // latest in-tool straddle
    for (open, close) in [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ] {
        if let Some((pos, end)) = missing_open_close(text, open, close, tool_ranges) {
            let is_top = !tool_ranges.iter().any(|(s, e)| pos >= *s && pos < *e);
            if is_top {
                if top_level.is_none_or(|(bp, _)| pos < bp) {
                    top_level = Some((pos, end));
                }
            } else if straddle.is_none_or(|(bp, _)| pos > bp) {
                straddle = Some((pos, end));
            }
        }
    }
    top_level
        .map(|(pos, end)| (pos, end, true))
        .or_else(|| straddle.map(|(pos, end)| (pos, end, false)))
}

/// True iff `text` still leads with a TOP-LEVEL (definitive) missing-open reasoning span — the
/// applied terminator (`applied_missing_open`) exists and is top-level. This is the
/// iterate-again signal for `strip_reasoning_preserving_tools`: a top-level terminator means
/// genuine leading reasoning remains, whereas an in-tool straddle (a preserved call whose
/// argument carries a `</think>`, with no top-level close past it) must NOT drive another pass —
/// re-running would drop the valid call.
fn has_top_level_missing_open_terminator(text: &str) -> bool {
    let mut tool_ranges: Vec<(usize, usize)> =
        extract_tag_blocks(text, "<tool_call>", "</tool_call>")
            .into_iter()
            .map(|(s, e, _)| (s, e))
            .collect();
    // LFM2 sentinel spans count too: a `</think>` inside a preserved LFM2
    // call's string arg is an in-tool straddle, NOT a top-level close —
    // misclassifying it would run a second pass that drops the call.
    tool_ranges.extend(
        extract_tag_blocks(text, LFM2_TOOL_CALL_START, LFM2_TOOL_CALL_END)
            .into_iter()
            .map(|(s, e, _)| (s, e)),
    );
    matches!(applied_missing_open(text, &tool_ranges), Some((_, _, true)))
}

/// Keep only the tool spans in `out` whose byte range is one of `genuine`
/// (the PRESERVED tool spans mapped to output coordinates) — across both
/// families (`<tool_call>…</tool_call>` and the LFM2 sentinel pair). Every
/// other tool span is a removal-seam artifact — fragments fused into a
/// tool span that never existed as a preserved call — and
/// `parse_tool_calls` would treat it as executable, so it is dropped.
///
/// Provenance is by RANGE, not bytes: a fabricated span can be byte-identical to a genuine call
/// (e.g. a duplicate), so a multiset of strings cannot disambiguate them (an earlier fabricated
/// look-alike would consume the genuine call's budget and the real one would be dropped) — but
/// the output ranges can. Dropped one span at a time, re-extracting and shifting `genuine` after
/// each drop so a fusion newly formed at a drop seam is itself caught. Each drop strictly shrinks
/// `out`, so it terminates.
fn keep_only_genuine_tool_spans(mut out: String, mut genuine: Vec<(usize, usize)>) -> String {
    loop {
        // Rescan BOTH families: a removal seam can fuse LFM2 sentinel
        // fragments (`<|tool_call_start` + `|>[f()]<|tool_call_end|>`) into a
        // fabricated block just like `<tool_call>` ones.
        let mut spans = extract_tag_blocks(&out, "<tool_call>", "</tool_call>");
        spans.extend(extract_tag_blocks(
            &out,
            LFM2_TOOL_CALL_START,
            LFM2_TOOL_CALL_END,
        ));
        // A span nested INSIDE a genuine one (`f(x='<tool_call>{}</tool_call>')`)
        // is preserved argument text, not markup — and needs no special case
        // here: the initial `tool_ranges` scan already extracted it and it was
        // disjoint from reasoning, so it sits in `genuine` alongside its
        // container. (No removal seam can form inside a genuine span — kept
        // spans are copied verbatim, so no new nested span can appear either.)
        let suspect = spans.iter().find(|(s, e, _)| !genuine.contains(&(*s, *e)));
        let Some(&(s, e, _)) = suspect else {
            return out; // every surviving tool span maps to a preserved range
        };
        let mut result = String::with_capacity(out.len() - (e - s));
        result.push_str(&out[..s]);
        result.push_str(&out[e..]);
        out = result;
        // Shift preserved ranges that sat after the removed span left by its length. (Tool spans
        // from `extract_tag_blocks` never overlap, so no genuine range straddles the removed one;
        // the `retain` is a defensive no-op for that.)
        let shift = e - s;
        genuine.retain(|&(gs, ge)| gs >= e || ge <= s);
        for r in &mut genuine {
            if r.0 >= e {
                r.0 -= shift;
                r.1 -= shift;
            }
        }
    }
}

/// All byte offsets at which `needle` occurs OUTSIDE every tool span (top-level tokens,
/// not literal text inside a tool-call argument).
fn top_level_positions(text: &str, needle: &str, tool_ranges: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find(needle) {
        let pos = from + rel;
        if !tool_ranges.iter().any(|(s, e)| pos >= *s && pos < *e) {
            out.push(pos);
        }
        from = pos + needle.len();
    }
    out
}

/// All byte offsets at which `needle` occurs in `text` (tool spans included).
fn all_positions(text: &str, needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find(needle) {
        let pos = from + rel;
        out.push(pos);
        from = pos + needle.len();
    }
    out
}

/// `(start, end)` of the reasoning close that terminates a template-injected leading
/// reasoning block of one family (`open`/`close`), or `None` if there is no such block.
///
/// The template injects the opener into the PROMPT, so the generated text begins one level
/// deep — modelled as an implicit depth of 1. Walking the family's open/close tags in
/// position order, a close that brings the depth back to the injected level (1) is a
/// candidate terminator; it actually terminates only when followed by a newline or
/// end-of-text (mirroring `parse_thinking`'s disambiguation of a real template close from a
/// literal close tag in content). A candidate that FAILS the newline gate is a literal close
/// inside reasoning content: the implicit open stays open (depth is kept at 1) and the scan
/// continues to a later newline-terminated close — it does NOT abandon detection, or a
/// reasoning-internal close could mask the real terminator and leak the prefix. A same-family
/// paired block nested in the prefix nets out (its open raises depth, its close lowers it), so
/// the *unmatched* injected close is still found. Depth is therefore provably ≥ 1 throughout
/// (the injected level is never decremented away), so no stray close drives it negative.
///
/// Opens are counted only at TOP LEVEL (outside tool spans): a `<think>` literal inside a
/// tool argument is not a structural nesting opener — counting it would inflate the depth and
/// hide the real terminator (a leak). Closes are scanned EVERYWHERE, tool spans included, but
/// the two kinds rank differently:
///   - A TOP-LEVEL newline-terminated close is a DEFINITIVE terminator (returned immediately).
///   - An IN-TOOL newline-terminated close is only a TENTATIVE straddle candidate: it is the
///     real terminator solely when the tool call straddles the reasoning boundary (opened
///     mid-reasoning) AND no top-level terminator exists. A literal newline-terminated close
///     inside a raw tool parameter would otherwise be chosen falsely, dropping only its span
///     and leaking the trailing reasoning. So an in-tool candidate is recorded but the scan
///     continues; a later top-level close takes precedence, and the tentative is used only if
///     no top-level terminator turns up. When several in-tool candidates exist and no
///     top-level terminator does, the LAST one wins: the reasoning range then reaches the
///     latest straddle, so the overlap-drop removes EVERY straddling span (a first-wins pick
///     would strip only the earliest and leak the later reasoning-started call).
///
/// In every straddle outcome the tool span(s) overlapping the resulting reasoning range are
/// dropped downstream, so the straddling call never surfaces.
fn missing_open_close(
    text: &str,
    open: &str,
    close: &str,
    tool_ranges: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let mut events: Vec<(usize, bool)> = top_level_positions(text, open, tool_ranges)
        .into_iter()
        .map(|p| (p, true))
        .chain(all_positions(text, close).into_iter().map(|p| (p, false)))
        .collect();
    events.sort_by_key(|(p, _)| *p);

    let mut depth = 1i32;
    let mut straddle: Option<(usize, usize)> = None;
    for (pos, is_open) in events {
        if is_open {
            depth += 1;
        } else if depth == 1 {
            // Close at the injected-open level: a terminator candidate iff followed by
            // newline/EOF. A close that fails the gate is a literal close in reasoning
            // content — keep the implicit open (depth stays 1) and keep scanning.
            let end = pos + close.len();
            if text[end..].is_empty() || text[end..].starts_with('\n') {
                if tool_ranges.iter().any(|(s, e)| pos >= *s && pos < *e) {
                    // In-tool close → tentative straddle candidate; prefer a top-level
                    // terminator if one appears later. Keep the LAST candidate (not the
                    // first): if no top-level terminator turns up, the reasoning must reach
                    // the latest straddle so the overlap-drop removes every straddling span;
                    // a first-wins pick would strip only the earliest and leak the later
                    // reasoning-started call. Keep depth at 1 (do not consume the implicit
                    // injected open on a merely-tentative close).
                    straddle = Some((pos, end));
                } else {
                    return Some((pos, end)); // top-level terminator is definitive
                }
            }
        } else {
            // Close of a same-family block nested inside the leading reasoning — nets out.
            depth -= 1;
        }
    }
    straddle
}

/// Check if text contains any thinking tags
pub fn has_thinking(text: &str) -> bool {
    text.contains("<think>") || text.contains("<longcat_think>")
}

/// Result of parsing tool calls from text
#[napi(object)]
pub struct ParseToolCallsResult {
    /// Cleaned text with tool_call tags removed
    pub text: String,
    /// Parsed tool calls
    pub tool_calls: Vec<ToolCallResult>,
}

/// Structured completion information aligned with ChatResult.
/// Contains pre-parsed tool calls, thinking, and clean text.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletionInfo {
    /// Clean text with tool-call markup (`<tool_call>` and the LFM2
    /// `<|tool_call_start|>…<|tool_call_end|>` pair) and <think> tags removed
    pub text: String,
    /// Raw output before tag stripping (for debugging/XML parsing)
    pub raw_text: String,
    /// Parsed tool calls (arguments are already JS objects)
    pub tool_calls: Vec<ToolCallResult>,
    /// Extracted thinking/reasoning from <think> tags (null if none)
    pub thinking: Option<String>,
    /// Number of tokens generated
    pub num_tokens: u32,
    /// Finish reason: "stop" | "length" | "tool_calls"
    pub finish_reason: String,
}

/// Reward function input for a single completion.
/// Provides all context needed to compute a reward score.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewardOutput {
    /// The input prompt text
    pub prompt: String,
    /// Structured completion data aligned with ChatResult
    pub completion: CompletionInfo,
}

/// Parse tool calls from text (NAPI export)
#[napi]
pub fn parse_tool_calls_from_text(text: String) -> ParseToolCallsResult {
    let (cleaned_text, tool_calls) = parse_tool_calls(&text);
    ParseToolCallsResult {
        text: cleaned_text,
        tool_calls,
    }
}

/// Parse both tool calls and thinking from generated text
///
/// Convenience function that extracts both structured components.
/// Returns (cleaned_text, tool_calls, thinking) where cleaned_text has
/// tool-call markup (`<tool_call>` and the LFM2 sentinel pair) and
/// `<think>` tags removed.
pub fn parse_generation_output(text: &str) -> (String, Vec<ToolCallResult>, Option<String>) {
    let (text_without_tools, tool_calls) = parse_tool_calls(text);
    let (cleaned_text, thinking) = parse_thinking(&text_without_tools);
    (cleaned_text, tool_calls, thinking)
}

/// Check if the `</think>` token exists in generated tokens.
pub fn has_think_end_token(generated_tokens: &[u32], think_end_id: Option<u32>) -> bool {
    think_end_id.is_some_and(|id| generated_tokens.contains(&id))
}

/// Count the number of reasoning tokens in generated output.
///
/// When `thinking` is Some (reasoning was detected), scans `generated_tokens`
/// for the `think_end_id` position. Tokens before that position are reasoning
/// tokens (excluding the `</think>` token itself). If `think_end_id` is not
/// in the vocabulary, all generated tokens are counted as reasoning.
/// Returns 0 when thinking is None.
pub fn count_reasoning_tokens(
    thinking: &Option<String>,
    generated_tokens: &[u32],
    think_end_id: Option<u32>,
) -> u32 {
    if thinking.is_none() {
        return 0;
    }
    if let Some(end_id) = think_end_id {
        generated_tokens
            .iter()
            .position(|&t| t == end_id)
            .map(|pos| pos as u32)
            .unwrap_or(generated_tokens.len() as u32)
    } else {
        generated_tokens.len() as u32
    }
}

/// Split generated output using token-level thinking detection.
///
/// When the think-end token was found in generated tokens (`think_end_tag` is Some),
/// splits at the corresponding text boundary. This is the authoritative path that
/// ensures tool parsing isolation: tool calls are only extracted from the content
/// portion after `</think>`, never from reasoning text.
///
/// Supports both `</think>` and `</longcat_think>` variants, and handles old-style
/// templates that emit `<think>` in generated text (stripped as a prefix).
///
/// Falls back to `parse_generation_output` only when `think_end_tag` is None.
pub fn split_at_think_end(
    raw_text: &str,
    think_end_tag: Option<&str>,
) -> (String, Vec<ToolCallResult>, Option<String>) {
    // Token-level split: authoritative when think_end_tag is confirmed.
    // Always takes priority — even when <think> appears in the text (old templates).
    // Tool calls are parsed only from content after the boundary.
    // Uses find (first occurrence): </think> is a special token, so the first
    // text match is the real boundary. Content after the boundary may mention
    // </think> literally; rfind would incorrectly split at that later occurrence.
    if let Some(tag) = think_end_tag
        && let Some(close_pos) = raw_text.find(tag)
    {
        let thinking_text = raw_text[..close_pos].trim();
        // Strip opening think tag from old-style templates that emit it
        // in generated text (newer templates inject it in the prompt).
        let thinking_text = thinking_text
            .strip_prefix("<think>")
            .or_else(|| thinking_text.strip_prefix("<longcat_think>"))
            .unwrap_or(thinking_text)
            .trim();
        let after_tag = &raw_text[close_pos + tag.len()..];
        let response_text = after_tag.trim_start_matches('\n').trim_start();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        let (clean_text, tool_calls) = parse_tool_calls(response_text);
        return (clean_text.trim().to_string(), tool_calls, thinking);
    }
    // No token-level confirmation: fall back to generic text-level parsing.
    // This path is used by callers without token-level info (e.g. build_reward_outputs).
    parse_generation_output(raw_text)
}

/// Build RewardOutput array from generation results.
///
/// Parses tool calls and thinking from completions, creating structured outputs
/// aligned with the ChatResult structure.
#[napi]
pub fn build_reward_outputs(
    prompts: Vec<String>,
    completions: Vec<String>,
    token_counts: Vec<u32>,
    finish_reasons: Vec<String>,
    group_size: u32,
) -> Vec<RewardOutput> {
    let group_size = group_size as usize;
    let mut outputs = Vec::with_capacity(completions.len());

    for (i, completion_text) in completions.iter().enumerate() {
        let prompt_idx = i / group_size;

        let (clean_text, tool_calls, thinking) = parse_generation_output(completion_text);

        let finish_reason = finish_reasons.get(i).cloned().unwrap_or_else(|| {
            if !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }
        });

        let num_tokens = token_counts.get(i).copied().unwrap_or(0);
        let prompt = prompts.get(prompt_idx).cloned().unwrap_or_default();

        outputs.push(RewardOutput {
            prompt,
            completion: CompletionInfo {
                text: clean_text,
                raw_text: completion_text.clone(),
                tool_calls,
                thinking,
                num_tokens,
                finish_reason,
            },
        });
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- strip_reasoning_preserving_tools (raw_text fallback scrubber) ----

    #[test]
    fn test_strip_reasoning_mixed_paired_families() {
        // Both families must be removed (parse_thinking alone stops after the first).
        let out = strip_reasoning_preserving_tools(
            "<think>a</think>mid<longcat_think>secret</longcat_think>answer",
        );
        assert!(
            !out.contains("secret"),
            "longcat reasoning must not leak: {out:?}"
        );
        assert!(
            !out.contains("<think>") && !out.contains("longcat_think"),
            "no reasoning tags: {out:?}"
        );
        assert!(
            out.contains("mid") && out.contains("answer"),
            "content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_preserves_tool_call_with_inner_think() {
        // A literal <think>…</think> INSIDE a tool argument is tool content, not
        // reasoning — the whole <tool_call>…</tool_call> span is copied verbatim.
        let input = r#"<tool_call>{"name":"f","arguments":{"q":"<think>x</think>"}}</tool_call>"#;
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(out, input, "tool-call span must be byte-preserved: {out:?}");
        assert!(out.contains("<tool_call>") && out.contains("</tool_call>"));
    }

    #[test]
    fn test_strip_reasoning_preserves_tool_call_with_bare_close() {
        // A bare </think> inside a tool argument must not trigger the missing-open
        // branch and eat the <tool_call> opener.
        let input =
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"q\":\"</think>\\nfoo\"}}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(out, input, "tool-call span must be byte-preserved: {out:?}");
    }

    #[test]
    fn test_strip_reasoning_before_tool_call() {
        // Reasoning leads, a tool call follows: reasoning gone, tool span intact.
        let input = "<think>reason</think>\n<tool_call>{\"name\":\"f\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("reason") && !out.contains("<think>"),
            "reasoning stripped: {out:?}"
        );
        assert_eq!(
            out, "<tool_call>{\"name\":\"f\"}</tool_call>",
            "tool span preserved: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_between_tool_calls() {
        // Reasoning between two tool calls is stripped; both tool spans survive.
        let input = "<tool_call>{\"name\":\"a\"}</tool_call><think>mid</think><tool_call>{\"name\":\"b\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("mid") && !out.contains("<think>"),
            "reasoning stripped: {out:?}"
        );
        assert!(
            out.contains(r#"{"name":"a"}"#) && out.contains(r#"{"name":"b"}"#),
            "both tools survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_no_tools_no_reasoning_is_verbatim() {
        let input = "just a plain answer with no tags";
        assert_eq!(strip_reasoning_preserving_tools(input), input);
    }

    #[test]
    fn test_strip_reasoning_tool_span_wrapped_by_reasoning() {
        // A tool span NESTED inside a <think> block is part of suppressed reasoning:
        // it must be dropped along with the reasoning, with NO prefix/suffix leak.
        let input = "<think>secret before <tool_call>{\"name\":\"f\"}</tool_call> secret after</think>\nanswer";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret"),
            "no part of the wrapping reasoning may leak: {out:?}"
        );
        assert!(
            !out.contains("<tool_call>") && !out.contains("<think>"),
            "nested tool span + reasoning tags must be gone: {out:?}"
        );
        assert!(out.contains("answer"), "trailing content survives: {out:?}");
    }

    #[test]
    fn test_strip_reasoning_tool_span_wrapped_by_longcat_reasoning() {
        // Same, for the <longcat_think> family.
        let input =
            "<longcat_think>pre <tool_call>{\"name\":\"f\"}</tool_call> post</longcat_think>\ndone";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("pre") && !out.contains("post"),
            "no wrapping longcat reasoning may leak: {out:?}"
        );
        assert!(
            !out.contains("<tool_call>") && !out.contains("longcat_think"),
            "nested tool span + longcat tags must be gone: {out:?}"
        );
        assert!(out.contains("done"), "trailing content survives: {out:?}");
    }

    #[test]
    fn test_strip_reasoning_pua_prose_is_preserved_no_fabrication() {
        // The model emits Private-Use-Area characters in ordinary prose, plus one real
        // tool call, with no reasoning. The range-based scrubber removes nothing and keeps
        // the text verbatim — the PUA prose must NOT be fabricated into a second tool call
        // (a regression that the old PUA-placeholder schemes were prone to).
        let input = "\u{E000}TOOLCALL0\u{E000} look <tool_call>{\"name\":\"real\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out.matches("<tool_call>").count(),
            1,
            "exactly one (real) tool call; the prose literal must not be duplicated: {out:?}"
        );
        assert!(
            out.contains("\u{E000}TOOLCALL0\u{E000} look"),
            "the sentinel-looking prose literal is preserved verbatim: {out:?}"
        );
        assert!(
            out.contains("{\"name\":\"real\"}"),
            "real tool call survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_pua_in_tool_args_round_trips() {
        // Two real tool calls (no reasoning); the first's argument contains Private-Use-Area
        // characters. The range-based scrubber removes nothing and both spans round-trip
        // byte-for-byte — no placeholder substitution that could clobber or mis-restore.
        let input = "<tool_call>{\"name\":\"a\",\"args\":\"\u{E000}TOOLCALL1\u{E000}\"}</tool_call><tool_call>{\"name\":\"b\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(out, input, "both tool spans preserved verbatim: {out:?}");
    }

    #[test]
    fn test_strip_reasoning_deletion_cannot_resurrect_nested_tool_call() {
        // Adversarial (prior Codex No-ship on the multi-char-delimiter scheme): the model
        // surrounds a reasoning-nested tool call with U+E000 fragments so that DELETING the
        // reasoning blocks would concatenate them into a synthesized placeholder, resurrecting
        // the suppressed `secret` call. The range-based scrubber never substitutes a marker,
        // so deletion cannot synthesize one and the nested call stays dropped.
        let e = '\u{E000}';
        let input = format!(
            "{e}<think>x</think>{e}TOOLCALL0{e}<think>y <tool_call>{{\"name\":\"secret\"}}</tool_call> z</think>{e}"
        );
        let out = strip_reasoning_preserving_tools(&input);
        assert!(
            !out.contains("<tool_call>") && !out.contains("secret"),
            "reasoning-nested tool call must NOT be resurrected by deletion: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_straddling_tool_span_is_dropped() {
        // Adversarial (Codex No-ship on the masking scheme): a <tool_call> opens INSIDE a
        // <think> block but its </tool_call> lands after </think>, so first-open→first-close
        // makes the tool span swallow the </think>. Masking then hid the reasoning close and
        // leaked the `<think>secret` prefix + a tool call that began in reasoning. Range-based
        // removal computes the reasoning boundary on the ORIGINAL text, sees the tool span
        // overlaps it, and drops BOTH — no reasoning prefix and no straddling tool call leak.
        let input = "<think>secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret") && !out.contains("<tool_call>") && !out.contains("<think>"),
            "straddling reasoning+tool must be fully dropped: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_close_inside_tool_arg() {
        // Adversarial (Codex No-ship): missing-open reasoning (no `<think>` opener — template
        // injected it) whose FIRST `</think>` is literal text inside a tool argument, with
        // the REAL top-level close later. The scan must skip the in-argument close and find
        // the real one, so the leading reasoning AND the reasoning-internal tool call are
        // dropped — not returned verbatim.
        let input = "secret <tool_call>{\"name\":\"leak\",\"arguments\":{\"q\":\"</think> literal\"}}</tool_call> more secret</think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret") && !out.contains("leak") && !out.contains("<tool_call>"),
            "missing-open reasoning + nested tool call must be dropped: {out:?}"
        );
        assert!(
            out.contains("final"),
            "post-reasoning content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_prefix_plus_later_paired_block() {
        // Adversarial (Codex No-ship): a missing-open `</think>` prefix followed by a later
        // paired block of the OTHER family, with a standalone tool call in the content
        // between them. Missing-open must compose with the paired block (both stripped) while
        // the content tool call is preserved.
        let input = "leading reasoning </think>\n<tool_call>{\"name\":\"f\"}</tool_call> mid <longcat_think>more reasoning</longcat_think> tail";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("leading reasoning") && !out.contains("more reasoning"),
            "both the missing-open prefix and the later paired block must be stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("\"f\"") && out.contains("tail"),
            "the content tool call and trailing content survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_longcat_close_after_think_block() {
        // Adversarial (Codex No-ship): a longcat missing-open prefix that contains a NESTED
        // paired `<think>` block, then ends with a bare `</longcat_think>`. The `<think>`
        // opener is a different family and must NOT gate the `</longcat_think>` missing-open
        // close, else the whole longcat reasoning prefix leaks.
        let input = "secret <think>inner</think> more </longcat_think>\n<tool_call>{\"name\":\"f\"}</tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("inner")
                && !out.contains("more")
                && !out.contains("longcat_think"),
            "the whole longcat missing-open prefix (incl. nested think) must be stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("\"f\"") && out.contains("final"),
            "the content tool call and trailing content survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_think_close_after_longcat_block() {
        // Symmetric to the above: a `<think>` missing-open prefix containing a nested paired
        // `<longcat_think>` block, ended by a bare `</think>`.
        let input = "secret <longcat_think>inner</longcat_think> more </think>\n<tool_call>{\"name\":\"f\"}</tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("inner")
                && !out.contains("more")
                && !out.contains("<think>")
                && !out.contains("</think>"),
            "the whole think missing-open prefix (incl. nested longcat) must be stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("\"f\"") && out.contains("final"),
            "the content tool call and trailing content survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_think_with_nested_same_family_block() {
        // Adversarial (Codex No-ship): a `<think>` missing-open prefix that NESTS a same-family
        // paired `<think>inner</think>` block before the real unmatched bare `</think>`. The
        // earlier inner opener must NOT veto the real terminator — bracket-depth matching nets
        // the inner open/close out and finds the unmatched injected close.
        let input = "secret <think>inner</think> more </think>\n<tool_call>{\"name\":\"f\"}</tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret") && !out.contains("inner") && !out.contains("more"),
            "the whole think missing-open prefix (incl. nested same-family block) must be stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("\"f\"") && out.contains("final"),
            "the content tool call and trailing content survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_longcat_with_nested_same_family_block() {
        // Symmetric: a `<longcat_think>` missing-open prefix nesting a same-family paired block
        // before the unmatched bare `</longcat_think>`.
        let input = "secret <longcat_think>inner</longcat_think> more </longcat_think>\n<tool_call>{\"name\":\"f\"}</tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret") && !out.contains("inner") && !out.contains("more"),
            "the whole longcat missing-open prefix (incl. nested same-family block) must be stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("\"f\"") && out.contains("final"),
            "the content tool call and trailing content survive: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_normal_paired_block_is_not_missing_open() {
        // Guard against regression on the range path (tool span present): a NORMAL paired
        // `<think>r</think>` block (explicit opener, not template-injected) must be handled by
        // paired detection and must NOT trigger missing-open (which would nuke preceding
        // content from byte 0). Leading content before the block survives.
        let input = "answer prefix <think>reasoning</think>\nmore answer <tool_call>{\"name\":\"f\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("reasoning") && !out.contains("<think>"),
            "the paired reasoning block is stripped: {out:?}"
        );
        assert!(
            out.contains("answer prefix")
                && out.contains("more answer")
                && out.contains("<tool_call>"),
            "leading/trailing content and the tool call survive (missing-open did not fire): {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_literal_close_before_real_terminator() {
        // Adversarial (Codex No-ship): a missing-open `<think>` prefix that contains a LITERAL
        // top-level `</think>` NOT followed by newline (reasoning content, e.g. discussing the
        // tag) before the real newline-terminated terminator, with a reasoning-internal tool
        // call between them. The literal close must NOT abandon detection (which would preserve
        // the whole prefix and leak the reasoning-nested tool call) — the scan continues to the
        // real terminator and the entire prefix incl. the nested tool call is stripped.
        let input = "secret </think> literal <tool_call>{\"name\":\"leak\"}</tool_call> more </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("literal")
                && !out.contains("more")
                && !out.contains("leak"),
            "the whole missing-open prefix incl. the reasoning-nested tool call must be stripped: {out:?}"
        );
        assert_eq!(
            out, "final",
            "only post-terminator content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_literal_close_no_tool_call() {
        // Adversarial (Codex No-ship): the SAME literal-close masking, but with NO `<tool_call>`
        // anywhere. The scrubber must NOT delegate the no-tool case to the generic
        // `parse_thinking` (whose first-close-only missing-open would leak the prefix); instead
        // the unified range path owns its `missing_open_close` scanner, which scans past the
        // literal non-newline close to the real terminator even when there is no tool span.
        let input = "secret </think> literal more </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out, "final",
            "no-tool missing-open prefix incl. the literal close must be fully stripped: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_raw_newline_close_in_tool_param() {
        // Adversarial (Codex No-ship): a raw-newline `</think>` inside a tool PARAMETER (XML-
        // style params carry raw text, so a literal newline after a `</think>` is plausible —
        // unlike JSON, which escapes it) BEFORE the real top-level terminator. The in-tool close
        // passes the newline gate but is NOT the boundary; choosing it would drop only the span
        // and leak the trailing reasoning (`more secret`). A top-level terminator must win, so
        // an in-tool newline-close is only a tentative straddle candidate.
        let input = "secret <tool_call><function=leak><parameter=q></think>\nliteral</parameter></function></tool_call> more secret </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("leak")
                && !out.contains("literal")
                && !out.contains("<tool_call>"),
            "the real top-level terminator wins; reasoning + straddling tool call are dropped: {out:?}"
        );
        assert_eq!(
            out, "final",
            "only post-terminator content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_multiple_in_tool_candidates_last_wins() {
        // Adversarial (Codex No-ship): TWO tool calls each carrying an in-tool `</think>\n` and
        // NO top-level terminator. A first-wins pick would stop at the earlier (literal) close,
        // strip only the first span, and leak `more secret` + the later reasoning-started
        // `<tool_call>`. Last-wins extends the reasoning to the latest straddle so the
        // overlap-drop removes BOTH straddling spans and the inter-call reasoning.
        let input = "secret <tool_call><function=a><parameter=q></think>\nliteral</parameter></function></tool_call> more secret <tool_call><function=real></think>\n<parameter=q>1</parameter></function></tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("literal")
                && !out.contains("<tool_call>")
                && !out.contains("function=real"),
            "both straddling tool calls and all reasoning must be dropped: {out:?}"
        );
        assert_eq!(
            out, "final",
            "only post-terminator content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_successive_no_tool_spans() {
        // Adversarial (Codex No-ship): SUCCESSIVE missing-open spans with NO tool call — two
        // bare top-level `</think>\n` closes. A single pass strips only the first span, leaving
        // `more secret </think>\nfinal`; the scrubber iterates to a fixpoint (matching the prior
        // strip_all_reasoning fixpoint) so the second span is stripped too.
        let input = "secret </think>\nmore secret </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out, "final",
            "successive no-tool missing-open spans must all be stripped: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_then_valid_tool_call_with_literal_close() {
        // Adversarial (Codex No-ship on the fixpoint): a LEADING missing-open reasoning span,
        // then a VALID standalone tool call whose argument contains a literal `</think>\n`. Pass
        // 1 correctly strips only the leading span and preserves the tool call. The fixpoint must
        // NOT re-run on this output — a second pass would treat the tool call (now at byte 0) as
        // injected reasoning and drop it. With a tool span present the scrubber runs one pass, so
        // the valid tool call survives.
        let input = "secret </think>\n<tool_call><function=ok><parameter=q></think>\nliteral</parameter></function></tool_call> tail";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret"),
            "the leading missing-open reasoning prefix is stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("function=ok") && out.contains("tail"),
            "the valid post-reasoning tool call and trailing content survive the fixpoint: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_straddling_tool_span_is_dropped() {
        // Adversarial (Codex No-ship): the MISSING-OPEN counterpart of the explicit-opener
        // straddle. A `<tool_call>` opens inside the (template-injected) reasoning prefix and
        // its body contains the real `</think>\n` terminator, with `</tool_call>` landing after
        // it. The terminator is INSIDE a tool span, so a top-level-only close scan misses it and
        // the reasoning-started tool call leaks. Scanning closes everywhere (incl. in-tool) finds
        // the terminator; the straddling span overlaps the reasoning range and is dropped.
        let input = "secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call> final";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret") && !out.contains("leak") && !out.contains("<tool_call>"),
            "missing-open straddling reasoning+tool must be fully dropped: {out:?}"
        );
        assert_eq!(
            out, "final",
            "only post-terminator content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_longcat_literal_close_before_real_terminator() {
        // Symmetric for the longcat family: a literal top-level `</longcat_think>` (no newline)
        // before the real newline-terminated close must not mask the terminator or leak the
        // reasoning-nested tool call.
        let input = "secret </longcat_think> literal <tool_call>{\"name\":\"leak\"}</tool_call> more </longcat_think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("secret")
                && !out.contains("literal")
                && !out.contains("more")
                && !out.contains("leak"),
            "the whole longcat missing-open prefix incl. the reasoning-nested tool call must be stripped: {out:?}"
        );
        assert_eq!(
            out, "final",
            "only post-terminator content survives: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_internal_tool_then_successive_span() {
        // Adversarial (Codex No-ship on the original-text fixpoint guard): the ORIGINAL has a
        // complete `<tool_call>` span, but it is reasoning-INTERNAL (inside the leading
        // missing-open prefix), so pass 1 drops it — leaving a tool-free remainder that STILL
        // begins with a second missing-open span. The fixpoint is gated on whether the output
        // still leads with a TOP-LEVEL missing-open terminator: after pass 1 strips the first
        // span (and its internal tool), the remainder leads with a top-level `</think>\n`, so
        // iteration continues and strips the second span too.
        let input = "secret <tool_call>{\"name\":\"leak\"}</tool_call> </think>\nmore secret </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out, "final",
            "a reasoning-internal tool call must not stop the fixpoint over successive spans: {out:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_successive_spans_each_with_internal_tool() {
        // Adversarial (Codex No-ship on the output-tool-span fixpoint guard): TWO successive
        // missing-open spans, EACH containing a complete reasoning-internal `<tool_call>`. Pass
        // 1 strips the first span + `leak1`, but the remainder `more secret <tool_call>{leak2}…
        // </think>\nfinal` still carries a surviving tool span. Halting on "any output tool
        // span" leaked `more secret` AND surfaced `leak2` as an executable call. Gating on a
        // surviving TOP-LEVEL terminator instead keeps iterating: the remainder leads with a
        // top-level `</think>\n` (the one after leak2's span), so pass 2 strips the second span
        // and drops `leak2`.
        let input = "secret <tool_call>{\"name\":\"leak1\"}</tool_call> </think>\nmore secret <tool_call>{\"name\":\"leak2\"}</tool_call> </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out, "final",
            "both reasoning-internal tool calls and both spans must be stripped: {out:?}"
        );
        let (_clean, calls) = parse_tool_calls(&out);
        assert!(
            calls.is_empty(),
            "no fabricated/leaked tool call must survive to parse_tool_calls: {calls:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_split_opener_does_not_synthesize_tool_call() {
        // Adversarial (Codex No-ship: deletion-synthesis). A reasoning block sits BETWEEN the
        // fragments of a `<tool_call>` opener (`<tool` … `_call>`). The input has NO complete
        // tool span, but naive removal of `<think>secret</think>` would FUSE the fragments into
        // `<tool_call>{…}</tool_call>` — a fabricated executable call. The synthesis defense
        // drops any output tool span absent verbatim from the input, so nothing is surfaced.
        let input = "<tool<think>secret</think>_call>{\"name\":\"leak\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("<tool_call>") && !out.contains("leak"),
            "deletion must not synthesize an executable tool call: {out:?}"
        );
        let (_clean, calls) = parse_tool_calls(&out);
        assert!(
            calls.is_empty(),
            "no synthesized tool call must reach parse_tool_calls: {calls:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_cross_family_later_top_level_drops_call() {
        // Adversarial (Codex No-ship: cross-family ranking). After the leading `</think>\n` is
        // stripped, the remainder LEADS with a tool call whose parameter holds a literal
        // `</longcat_think>\n`, AND there is a LATER top-level `</think>\n` (after `more secret`).
        // The reasoning genuinely extends to that top-level close, so the tool call — which
        // opened before it — is reasoning-internal and must be dropped. Preferring the earlier
        // in-tool longcat straddle over the later top-level `</think>` halted early and leaked
        // both `more secret` and the call. `applied_missing_open` prefers the top-level close, so
        // iteration continues and strips through it.
        //
        // DELIBERATE disposition of the inverse [medium] ("this drops a post-boundary call"):
        // under the successive-missing-open-span model (a second bare close after the first IS a
        // second reasoning span — the same rule that strips `secret </think>\nmore secret </think>\n…`
        // in the no-tool case), a call inside that span is reasoning-internal. Dropping it is the
        // security-conservative, internally-consistent choice; preserving it would re-introduce
        // the leak and would be inconsistent with the no-tool successive-span stripping.
        let input = "r1 </think>\n<tool_call><function=leak><parameter=p></longcat_think>\nliteral</parameter></function></tool_call> more secret </think>\nfinal";
        let out = strip_reasoning_preserving_tools(input);
        assert_eq!(
            out, "final",
            "the reasoning-internal call before a later top-level close must be dropped: {out:?}"
        );
        let (_clean, calls) = parse_tool_calls(&out);
        assert!(
            calls.is_empty(),
            "no leaked/executable tool call survives: {calls:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_missing_open_cross_family_no_later_close_preserves_call() {
        // Counterpart to the above: the genuine post-reasoning call's parameter holds a literal
        // `</longcat_think>\n`, but there is NO top-level close anywhere past the call (trailing
        // ` tail`, no `</think>`). The only candidate terminator in the remainder is the call's
        // own in-tool argument close, so the fixpoint HALTS and the call is preserved.
        let input = "reasoning </think>\n<tool_call><function=ok><parameter=p></longcat_think>\nv</parameter></function></tool_call> tail";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.starts_with("reasoning"),
            "the leading missing-open reasoning prefix is stripped: {out:?}"
        );
        assert!(
            out.contains("<tool_call>") && out.contains("function=ok") && out.contains("tail"),
            "the genuine post-reasoning tool call and trailing content survive: {out:?}"
        );
        let (_clean, calls) = parse_tool_calls(&out);
        assert_eq!(
            calls.len(),
            1,
            "exactly the one genuine tool call is recovered: {calls:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_synthesis_duplicate_bytes_in_deleted_reasoning() {
        // Adversarial (Codex No-ship: substring provenance was insufficient). A reasoning block
        // CONTAINS a real-looking `<tool_call>{leak}</tool_call>` (dropped as reasoning), and a
        // SECOND region fuses `<tool` <reasoning> `_call>` into an identical `<tool_call>{leak}…`
        // span. A substring check would falsely keep the fused span because its bytes also occur
        // inside the deleted reasoning. RANGE provenance keeps only the PRESERVED (kept) tool
        // spans — here none — so the fabricated call is dropped.
        let input = "<think><tool_call>{\"name\":\"leak\"}</tool_call></think><tool<think>secret</think>_call>{\"name\":\"leak\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        assert!(
            !out.contains("<tool_call>") && !out.contains("leak"),
            "a fabricated span matching deleted-reasoning bytes must not survive: {out:?}"
        );
        let (_clean, calls) = parse_tool_calls(&out);
        assert!(
            calls.is_empty(),
            "no synthesized tool call must reach parse_tool_calls: {calls:?}"
        );
    }

    #[test]
    fn test_strip_reasoning_synthesis_duplicate_does_not_steal_genuine_provenance() {
        // Adversarial (Codex No-ship: a per-string budget mis-attributes provenance). A
        // fabricated `<tool_call>{b}</tool_call>` (fused from `<tool` <reasoning> `_call>{b}`)
        // sits BEFORE a genuine `{a}` and a genuine `{b}`. Keyed by byte string, the fabricated
        // `b` would consume the sole `b` budget, the genuine `b` would be dropped as excess, and
        // the executable calls would come out `[b, a]` (fabricated + reordered). RANGE provenance
        // keeps exactly the two PRESERVED spans at their mapped output ranges — the fabricated
        // leading `b` is at no genuine range and is dropped — recovering the true `[a, b]`.
        let input = "<tool<think>secret</think>_call>{\"name\":\"b\"}</tool_call><tool_call>{\"name\":\"a\"}</tool_call><tool_call>{\"name\":\"b\"}</tool_call>";
        let out = strip_reasoning_preserving_tools(input);
        let (_clean, calls) = parse_tool_calls(&out);
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["a", "b"],
            "only the two genuine calls survive, in order; the fused look-alike is dropped: {out:?}"
        );
    }

    // ---- Tag extraction helpers ----

    #[test]
    fn test_extract_tag_blocks_basic() {
        let blocks = extract_tag_blocks("<a>hello</a> world <a>bye</a>", "<a>", "</a>");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].2, "hello");
        assert_eq!(blocks[1].2, "bye");
    }

    #[test]
    fn test_extract_tag_blocks_no_match() {
        let blocks = extract_tag_blocks("no tags here", "<a>", "</a>");
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_strip_tag_blocks() {
        let result = strip_tag_blocks("before <a>inner</a> after", "<a>", "</a>");
        assert_eq!(result, "before  after");
    }

    // ---- JSON format (Qwen3) ----

    #[test]
    fn test_parse_json_tool_call() {
        let (text, calls) = parse_tool_calls(
            r#"I'll help you. <tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#,
        );

        assert_eq!(text, "I'll help you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["location"], "Paris");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let (text, calls) = parse_tool_calls(
            r#"Let me call two tools.
<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call>
<tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>"#,
        );

        assert_eq!(text, "Let me call two tools.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[1].name, "func2");
    }

    #[test]
    fn test_parse_tool_call_no_arguments() {
        let (_, calls) = parse_tool_calls(r#"<tool_call>{"name": "get_time"}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].status, "ok");
        assert!(calls[0].arguments.is_object());
    }

    #[test]
    fn test_parse_invalid_json() {
        let (_, calls) = parse_tool_calls(r#"<tool_call>{not valid json}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "invalid_json");
        assert!(calls[0].error.is_some());
    }

    #[test]
    fn test_parse_no_braces_ignored() {
        let (text, calls) = parse_tool_calls(r#"<tool_call>not valid json</tool_call>"#);

        // The tag is still stripped from text
        assert_eq!(text, "");
        // No recognized format — no tool call detected
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_missing_name() {
        let (_, calls) =
            parse_tool_calls(r#"<tool_call>{"arguments": {"key": "value"}}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "missing_name");
        assert!(calls[0].error.is_some());
    }

    #[test]
    fn test_no_tool_calls() {
        let (text, calls) = parse_tool_calls("This is just regular text without any tool calls.");

        assert_eq!(text, "This is just regular text without any tool calls.");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_ids_unique() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_string_arguments_parsed() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "test", "arguments": "{\"key\": \"value\"}"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["key"], "value");
    }

    #[test]
    fn test_string_arguments_invalid_json() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "test", "arguments": "not valid json"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test");
        assert_eq!(calls[0].status, "parse_error");
        assert!(calls[0].error.is_some());
        assert!(
            calls[0]
                .error
                .as_ref()
                .unwrap()
                .contains("Failed to parse arguments string as JSON")
        );
        assert_eq!(calls[0].arguments, "not valid json");
    }

    #[test]
    fn test_string_arguments_truncated_json() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "search", "arguments": "{\"query\": \"test"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].status, "parse_error");
        assert!(calls[0].error.is_some());
        assert_eq!(calls[0].arguments, r#"{"query": "test"#);
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("<tool_call>...</tool_call>"));
        assert!(!has_tool_calls("no tools here"));
    }

    // ---- Function format (Qwen3.5/Qwen3-Coder) ----

    #[test]
    fn test_parse_function_tool_call_basic() {
        let input = "<tool_call>\n<function=get_current_time>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_current_time");
        assert_eq!(calls[0].status, "ok");
        assert!(calls[0].arguments.is_object());
        assert_eq!(calls[0].arguments.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_function_tool_call_with_params() {
        let input = "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://httpbin.org/json\n</parameter>\n<parameter=method>\nGET\n</parameter>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fetch_url");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["url"], "https://httpbin.org/json");
        assert_eq!(calls[0].arguments["method"], "GET");
    }

    #[test]
    fn test_parse_function_tool_call_multiline_value() {
        let input = "<tool_call>\n<function=multiply>\n<parameter=a>\n12234585\n</parameter>\n<parameter=b>\n48838483920\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "multiply");
        assert_eq!(calls[0].arguments["a"], "12234585");
        assert_eq!(calls[0].arguments["b"], "48838483920");
    }

    #[test]
    fn test_parse_function_tool_call_with_reasoning() {
        let input = "I'll look that up for you.\n\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://example.com\n</parameter>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "I'll look that up for you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fetch_url");
        assert_eq!(calls[0].arguments["url"], "https://example.com");
    }

    #[test]
    fn test_parse_function_tool_call_multiple() {
        let input = "<tool_call>\n<function=func1>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=func2>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[0].arguments["x"], "1");
        assert_eq!(calls[1].name, "func2");
        assert_eq!(calls[1].arguments["y"], "2");
    }

    // ---- XML format (legacy) ----

    #[test]
    fn test_parse_xml_tool_call() {
        let (text, calls) = parse_tool_calls(
            r#"<tool_call><name>search</name><arguments>{"query": "test"}</arguments></tool_call>"#,
        );

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["query"], "test");
    }

    // ---- Thinking parsing ----

    #[test]
    fn test_parse_thinking_basic() {
        let (text, thinking) =
            parse_thinking("<think>\nLet me analyze this problem.\n</think>\n\nThe answer is 42.");

        assert_eq!(text, "The answer is 42.");
        assert_eq!(thinking, Some("Let me analyze this problem.".to_string()));
    }

    #[test]
    fn test_parse_thinking_no_tags() {
        let (text, thinking) = parse_thinking("Just regular text without thinking.");

        assert_eq!(text, "Just regular text without thinking.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_empty_tags() {
        let (text, thinking) = parse_thinking("<think>\n\n</think>\n\nThe response.");

        assert_eq!(text, "The response.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_multiple_blocks() {
        let (text, thinking) = parse_thinking(
            "<think>First thought</think>\nMiddle text\n<think>Second thought</think>\nFinal answer.",
        );

        assert_eq!(text, "Middle text\n\nFinal answer.");
        assert_eq!(
            thinking,
            Some("First thought\n\nSecond thought".to_string())
        );
    }

    #[test]
    fn test_has_thinking() {
        assert!(has_thinking("<think>...</think>"));
        assert!(!has_thinking("no thinking here"));
    }

    // ---- Combined parsing ----

    #[test]
    fn test_parse_generation_output_with_both() {
        let input = r#"<think>Let me think about this...</think>

I'll use a tool.
<tool_call>{"name": "get_time"}</tool_call>

Here's the result."#;

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert!(text.contains("I'll use a tool."));
        assert!(text.contains("Here's the result."));
        assert!(!text.contains("<tool_call>"));
        assert!(!text.contains("<think>"));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_time");
        assert_eq!(thinking, Some("Let me think about this...".to_string()));
    }

    #[test]
    fn test_parse_generation_output_no_special_tags() {
        let input = "Just a plain response without any special tags.";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "Just a plain response without any special tags.");
        assert!(tool_calls.is_empty());
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_generation_output_qwen35_with_thinking() {
        let input = "<think>\nI need to check the time.\n</think>\n\n<tool_call>\n<function=get_current_time>\n</function>\n</tool_call>";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_current_time");
        assert_eq!(thinking, Some("I need to check the time.".to_string()));
    }

    // ---- Thinking: missing opening tag (template prefix) ----

    #[test]
    fn test_parse_thinking_no_opening_tag() {
        // When enable_thinking=true, the chat template adds <think>\n as the
        // assistant prefix. The model's generated text starts after that, so
        // it contains thinking content + </think> but no opening <think>.
        let input = "Let me analyze this problem.\n</think>\n\nThe answer is 42.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "The answer is 42.");
        assert_eq!(thinking, Some("Let me analyze this problem.".to_string()));
    }

    #[test]
    fn test_parse_thinking_literal_close_tag_mid_sentence() {
        // Bare </think> in the middle of a sentence should NOT be treated
        // as a thinking delimiter — it's literal content.
        let input = "Use </think> to close the tag.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "Use </think> to close the tag.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_no_opening_tag_empty_thinking() {
        // Model immediately closes thinking with no content
        let input = "\n</think>\n\nThe response.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "The response.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_generation_output_no_opening_think_with_tools() {
        let input = "I need to check.\n</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_time");
        assert_eq!(thinking, Some("I need to check.".to_string()));
    }

    // ---- JSON sanitizer ----

    #[test]
    fn test_sanitize_json_string_with_raw_newlines() {
        let input = "{\n  \"code\": \"line1\nline2\nline3\"\n}";
        let sanitized = sanitize_json_string(input);

        assert_eq!(sanitized, "{\n  \"code\": \"line1\\nline2\\nline3\"\n}");

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["code"], "line1\nline2\nline3");
    }

    #[test]
    fn test_sanitize_json_string_with_tabs_and_carriage_returns() {
        let input = "{\n  \"text\": \"has\ttab\rand\r\ncrlf\"\n}";
        let sanitized = sanitize_json_string(input);

        assert!(sanitized.contains("\\t"));
        assert!(sanitized.contains("\\r"));

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["text"], "has\ttab\rand\r\ncrlf");
    }

    #[test]
    fn test_sanitize_json_string_with_escaped_quotes() {
        let input = r#"{"text": "he said \"hello\"\nand left"}"#;
        let sanitized = sanitize_json_string(input);

        assert!(sanitized.contains(r#"\"hello\""#));

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["text"], "he said \"hello\"\nand left");
    }

    #[test]
    fn test_sanitize_json_string_with_escaped_backslash() {
        let input = "{\n  \"path\": \"C:\\\\\nD:\\\\\"\n}";
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["path"], "C:\\\nD:\\");
    }

    #[test]
    fn test_sanitize_json_string_multiline_code() {
        let input = r#"{
  "name": "run_js",
  "arguments": {
    "code": "import { foo } from './bar'
export function main() {
  console.log('hello')
}"
  }
}"#;
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["name"], "run_js");
        let code = parsed["arguments"]["code"].as_str().unwrap();
        assert!(code.contains("import { foo }"));
        assert!(code.contains("export function main()"));
        assert!(code.contains("console.log"));
    }

    #[test]
    fn test_sanitize_json_string_preserves_valid_json() {
        let input = r#"{"name": "test", "args": {"key": "value"}}"#;
        let sanitized = sanitize_json_string(input);
        assert_eq!(sanitized, input);
    }

    #[test]
    fn test_sanitize_json_string_nested_objects() {
        let input = "{\n  \"outer\": {\n    \"inner\": \"line1\nline2\"\n  }\n}";
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["outer"]["inner"], "line1\nline2");
    }

    #[test]
    fn test_sanitize_json_tool_call_integration() {
        let input = r#"<tool_call>
{
  "name": "run_js",
  "arguments": {
    "code": "const x = 1
const y = 2
console.log(x + y)"
  }
}
</tool_call>"#;

        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].name, "run_js");
        let code = calls[0].arguments["code"].as_str().unwrap();
        assert!(code.contains("const x = 1"));
        assert!(code.contains("const y = 2"));
        assert!(code.contains("console.log"));
    }

    // ---- Critical tool call code paths ----

    #[test]
    fn test_parse_multiple_json_tool_calls_with_text() {
        // Two JSON-format tool calls (Qwen3 style) with leading text
        let text = r#"Let me check both.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[1].status, "ok");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
        assert_eq!(calls[1].arguments["city"], "Paris");
        assert_eq!(clean.trim(), "Let me check both.");
    }

    #[test]
    fn test_parse_multiple_function_tool_calls_different_names() {
        // Two function-format tool calls (Qwen3.5 style) with different function names
        let text = r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call>
<tool_call>
<function=get_time>
<parameter=timezone>JST</parameter>
</function>
</tool_call>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[1].status, "ok");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
        assert_eq!(calls[1].arguments["timezone"], "JST");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_parse_generation_output_multiple_tools_with_thinking() {
        // Thinking block followed by multiple JSON tool calls
        let text = r#"<think>I need to check both cities.</think>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls, thinking) = parse_generation_output(text);
        assert_eq!(calls.len(), 2);
        assert!(thinking.is_some());
        assert_eq!(thinking.unwrap().trim(), "I need to check both cities.");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_split_at_think_end_with_multiple_tools() {
        // Simulate Qwen3.5 path: thinking prefix (no opening <think> tag) then tool calls.
        // The chat template injects `<think>\n` as a prefix so the generated text
        // starts with thinking content followed by `</think>`.
        let text = r#"I need weather data.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(calls.len(), 2);
        assert!(thinking.is_some());
        assert_eq!(thinking.unwrap().trim(), "I need weather data.");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_parse_unclosed_tool_call() {
        // Truncated by max_tokens — no closing </tool_call> tag
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Tok"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 0); // No complete tool call
        assert_eq!(clean, text); // Text preserved as-is
    }

    #[test]
    fn test_parse_tool_call_with_trailing_hallucination() {
        // Model generates a tool call then hallucinates a response
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
The weather in Tokyo is sunny."#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        // The trailing hallucinated text remains in clean text
        assert!(clean.contains("The weather in Tokyo is sunny."));
    }

    #[test]
    fn test_parse_empty_tool_call() {
        // Empty tool_call block — recognized as a tag pair but no parseable format inside
        let text = "<tool_call></tool_call>";
        let (_, calls) = parse_tool_calls(text);
        // Empty content doesn't start with '{', contain '<function=', or '<name>',
        // so classify_and_parse_tool_call returns None — no tool call produced.
        assert_eq!(calls.len(), 0);
    }

    // ---- split_at_think_end: tool isolation with token-confirmed boundary ----

    #[test]
    fn test_split_at_think_end_old_template_tool_in_reasoning() {
        // Old-style template: explicit <think> + tool_call inside reasoning.
        // Tool call must NOT be extracted — it's inside the reasoning block.
        let text = "<think>Let me call <tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call> to help</think>\nThe answer is 42";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(clean, "The answer is 42");
        assert!(
            tools.is_empty(),
            "tool_call inside reasoning must not be extracted"
        );
        let t = thinking.unwrap();
        assert!(
            t.contains("tool_call"),
            "tool_call text should remain in thinking"
        );
        assert!(
            t.starts_with("Let me call"),
            "<think> prefix should be stripped"
        );
    }

    #[test]
    fn test_split_at_think_end_tool_only_in_content() {
        // Tool call in content portion after </think> — should be extracted.
        let text = "<think>reasoning</think>\n<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call>";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "reasoning");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_split_at_think_end_literal_think_in_reasoning() {
        // Literal <think> inside reasoning text (e.g., model explaining tags).
        // Must not cause mis-split — token boundary is authoritative.
        let text = "The model uses <think> tags for reasoning</think>\ncontent here";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(clean, "content here");
        assert!(tools.is_empty());
        let t = thinking.unwrap();
        assert!(
            t.contains("<think>"),
            "literal <think> preserved in thinking"
        );
    }

    #[test]
    fn test_split_at_think_end_longcat_variant() {
        // longcat_think variant with tool_call inside reasoning.
        let text = "<longcat_think>reasoning <tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call></longcat_think>\nanswer";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</longcat_think>"));
        assert_eq!(clean, "answer");
        assert!(
            tools.is_empty(),
            "tool_call inside longcat reasoning must not be extracted"
        );
        assert!(thinking.unwrap().contains("tool_call"));
    }

    #[test]
    fn test_split_at_think_end_budget_forced_no_newline() {
        // Budget-forced </think> with no newline separator (model continues directly).
        let text = "thinking content</think>immediate content";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "thinking content");
        assert_eq!(clean, "immediate content");
        assert!(tools.is_empty());
    }

    #[test]
    fn test_split_at_think_end_close_tag_in_content() {
        // Content after the real boundary mentions </think> literally.
        // find (first occurrence) splits at the real boundary, not the literal.
        let text = "reasoning here</think>\nThe </think> tag ends reasoning.";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "reasoning here");
        assert!(tools.is_empty());
        assert!(
            clean.contains("</think> tag ends reasoning"),
            "literal </think> in content should be preserved, got: {clean}"
        );
    }

    // ---- Function format JSON-typed parameter values (Qwen3.5+ | tojson) ----

    #[test]
    fn test_parse_function_tool_call_array_parameter() {
        // Array-typed argument: `<parameter=edits>[{...}]</parameter>` must come back
        // as a Value::Array, not a JSON-encoded string.
        let input = "<tool_call>\n<function=edit>\n<parameter=edits>\n[{\"oldText\":\"hello\",\"newText\":\"world\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "edit");
        assert_eq!(calls[0].status, "ok");
        let edits = &calls[0].arguments["edits"];
        assert!(edits.is_array(), "edits must be a JSON array, got: {edits}");
        let arr = edits.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["oldText"], "hello");
        assert_eq!(arr[0]["newText"], "world");
    }

    #[test]
    fn test_parse_function_tool_call_object_parameter() {
        // Object-typed argument: `<parameter=config>{...}</parameter>` must come back
        // as a Value::Object with working nested access.
        let input = "<tool_call>\n<function=configure>\n<parameter=config>\n{\"key\":\"value\",\"nested\":{\"a\":1}}\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        let config = &calls[0].arguments["config"];
        assert!(config.is_object(), "config must be a JSON object");
        assert_eq!(config["key"], "value");
        assert_eq!(config["nested"]["a"], 1);
    }

    #[test]
    fn test_parse_function_tool_call_plain_string_parameter() {
        // Bare string (no `[`/`{` prefix) must remain Value::String.
        let input = "<tool_call>\n<function=search>\n<parameter=query>\nhello world\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let query = &calls[0].arguments["query"];
        assert!(query.is_string());
        assert_eq!(query.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_parse_function_tool_call_bare_number_stays_string() {
        // Conservative choice: bare numeric-looking values stay as Value::String
        // since string/number ambiguity can't be resolved at this layer.
        let input = "<tool_call>\n<function=count>\n<parameter=count>\n42\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let count = &calls[0].arguments["count"];
        assert!(count.is_string(), "bare numbers must remain strings");
        assert_eq!(count.as_str().unwrap(), "42");
    }

    #[test]
    fn test_parse_function_tool_call_invalid_json_array_falls_back_to_string() {
        // Value starts with `[` but isn't valid JSON — must fall back to Value::String
        // preserving the original raw text.
        let input = "<tool_call>\n<function=f>\n<parameter=q>\n[unclosed\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let q = &calls[0].arguments["q"];
        assert!(q.is_string(), "invalid JSON array must fall back to string");
        assert_eq!(q.as_str().unwrap(), "[unclosed");
    }

    #[test]
    fn test_parse_function_tool_call_multiline_json_array() {
        // Array spread across multiple lines must still parse as JSON.
        let input = "<tool_call>\n<function=edit>\n<parameter=edits>\n[\n  {\"oldText\":\"a\",\"newText\":\"b\"},\n  {\"oldText\":\"c\",\"newText\":\"d\"}\n]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let edits = &calls[0].arguments["edits"];
        assert!(edits.is_array());
        let arr = edits.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["oldText"], "a");
        assert_eq!(arr[1]["newText"], "d");
    }

    #[test]
    fn test_parse_function_tool_call_array_two_items_escaped_strings() {
        // Two-element array with JSON-escaped strings parses correctly.
        let input = "<tool_call>\n<function=edit>\n<parameter=edits>\n[{\"oldText\":\"a\",\"newText\":\"b\"},{\"oldText\":\"c\",\"newText\":\"d\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let edits = &calls[0].arguments["edits"];
        let arr = edits.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["oldText"], "a");
        assert_eq!(arr[0]["newText"], "b");
        assert_eq!(arr[1]["oldText"], "c");
        assert_eq!(arr[1]["newText"], "d");
    }

    #[test]
    fn test_parse_function_tool_call_array_with_multiline_string_escape() {
        // JSON-escaped newlines (`\n`) inside string values decode to real newlines
        // after parse.
        let input = "<tool_call>\n<function=edit>\n<parameter=edits>\n[{\"oldText\":\"line1\\nline2\",\"newText\":\"x\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let edits = &calls[0].arguments["edits"];
        let arr = edits.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["oldText"], "line1\nline2");
        assert_eq!(arr[0]["newText"], "x");
    }

    #[test]
    fn test_parse_function_tool_call_mixed_parameter_types() {
        // Real-world shape: array for `edits`, plain string for `path` —
        // both must come back with the correct type.
        let input = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/file.txt\n</parameter>\n<parameter=edits>\n[{\"oldText\":\"hello\",\"newText\":\"world\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        let path = &calls[0].arguments["path"];
        assert!(path.is_string());
        assert_eq!(path.as_str().unwrap(), "/tmp/file.txt");

        let edits = &calls[0].arguments["edits"];
        assert!(edits.is_array());
        let arr = edits.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["oldText"], "hello");
        assert_eq!(arr[0]["newText"], "world");
    }

    #[test]
    fn test_parse_function_tool_call_pi_edit_tool_shape_regression() {
        // Regression guard for the exact vitest-migration bug: pi's `edit` tool
        // requires `edits: array` and `path: string`. Prior to the fix, `edits`
        // arrived as a JSON-encoded string and pi rejected every call.
        let input = "<tool_call>\n<function=edit>\n<parameter=path>\n/repo/src/foo.ts\n</parameter>\n<parameter=edits>\n[{\"oldText\":\"it.skip\",\"newText\":\"it\"},{\"oldText\":\"describe.skip\",\"newText\":\"describe\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "edit");
        assert_eq!(calls[0].status, "ok");
        // path: string
        assert!(calls[0].arguments["path"].is_string());
        assert_eq!(
            calls[0].arguments["path"].as_str().unwrap(),
            "/repo/src/foo.ts"
        );
        // edits: array (this was broken — used to be Value::String)
        assert!(
            calls[0].arguments["edits"].is_array(),
            "edits must validate as array against pi's schema"
        );
        let arr = calls[0].arguments["edits"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["oldText"], "it.skip");
        assert_eq!(arr[1]["newText"], "describe");
    }

    #[test]
    fn test_parse_function_tool_call_preserves_parameter_order() {
        // Cache-reuse regression guard: the model emits parameters in
        // whatever order it learned from the tool schema (usually the
        // `required` order — `[path, edits]` for pi's `edit` tool), and
        // the warm KV cache encodes that exact byte stream. When pi-mono
        // echoes the function_call back on the next turn, the server
        // re-parses the arguments and feeds them through the Qwen3.5
        // template for cache verification. If the parsed `arguments`
        // object re-orders the keys (BTreeMap-style alphabetisation),
        // the echoed `<parameter=…>` blocks come out as
        // `edits, path` instead of `path, edits`, flipping two tokens at
        // the start of the call and zeroing `verify_cache_prefix_direct`.
        //
        // This test pins the `[path, edits]` insertion order that the
        // `preserve_order` serde_json feature enables — without it this
        // assertion fails and turn N+1 cold-prefills the full history.
        // Observed on 2026-04-21 at turn 11 of the vitest-migration
        // session (151 s re-prefill) — see `.logging/requests.ndjson`.
        let input = "<tool_call>\n<function=edit>\n<parameter=path>\n/f.ts\n</parameter>\n<parameter=edits>\n[{\"oldText\":\"a\",\"newText\":\"b\"}]\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);

        let obj = calls[0]
            .arguments
            .as_object()
            .expect("arguments parsed into an object");
        let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec!["path", "edits"],
            "arg-key order must match the `<parameter=…>` emission order; if this fails, serde_json is missing the `preserve_order` feature",
        );

        // Serializing back to JSON must also preserve that order —
        // confirms the stored `ToolCall.arguments` string that pi-mono
        // will echo is byte-parity with the model's original output.
        let serialized = serde_json::to_string(&calls[0].arguments).unwrap();
        let path_idx = serialized.find("\"path\"").expect("path key present");
        let edits_idx = serialized.find("\"edits\"").expect("edits key present");
        assert!(
            path_idx < edits_idx,
            "`path` must appear before `edits` in serialized args; got {serialized}",
        );
    }

    // ----- LFM2 pythonic sentinel format -----

    #[test]
    fn test_lfm2_tool_call_basic() {
        let input = "Let me check.<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
    }

    #[test]
    fn test_lfm2_tool_call_literal_types() {
        let input = "<|tool_call_start|>[wx.forecast(city='Paris', days=-3, opts={\"deep\": [1, 2]}, unit=\"C\", rain=True, snow=false, none=None, tup=(1, 'x'), st={1, 2}, hx=0x1f, fl=1.5e-3)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        let a = &calls[0].arguments;
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["days"], -3);
        assert_eq!(a["opts"], serde_json::json!({"deep": [1, 2]}));
        assert_eq!(a["unit"], "C");
        assert_eq!(a["rain"], true);
        assert_eq!(a["snow"], false);
        assert_eq!(a["none"], Value::Null);
        assert_eq!(a["tup"], serde_json::json!([1, "x"]));
        assert_eq!(a["st"], serde_json::json!([1, 2]));
        assert_eq!(a["hx"], 31);
        assert!((a["fl"].as_f64().unwrap() - 0.0015).abs() < 1e-9);
    }

    #[test]
    fn test_lfm2_tool_call_multiple_calls_one_block() {
        // Parallel calls live inside ONE bracket list (vLLM parity).
        let input = "<|tool_call_start|>[a.f(x=1), b.g(y='z', w=[True])]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a.f"); // dotted name preserved
        assert_eq!(calls[1].name, "b.g");
        assert_eq!(calls[1].arguments["w"], serde_json::json!([true]));
    }

    #[test]
    fn test_lfm2_tool_call_echo_suppression() {
        // LFM2 re-emits the call body after the first end sentinel, capped
        // by a second end sentinel — everything through the last orphan end
        // is dropped, then real post-call prose resumes.
        let input =
            "Check: <|tool_call_start|>[f(x=1)]<|tool_call_end|>[f(x=1)]<|tool_call_end|>Done.";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        // Byte-identical to the streamed fragments ("Check: " + "Done.") —
        // a "\n" join would make the done-path recovery re-emit the text.
        assert_eq!(text, "Check: Done.");
    }

    #[test]
    fn test_lfm2_tool_call_content_matches_streamed_fragments() {
        // Visible text on BOTH sides of a successful call: the cleaned text
        // must equal the bytes the ToolCallTagBuffer streams (original
        // whitespace kept), otherwise the done-path suffix overlap finds no
        // common edge and emits the whole finalized text a second time.
        let input = "before <|tool_call_start|>[f(x=1)]<|tool_call_end|>after";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(text, "before after");
    }

    #[test]
    fn test_lfm2_tool_call_multiple_sentinel_blocks() {
        // Two separate sentinel blocks: both calls survive and the prose
        // between them is kept (a single-block read would swallow the
        // second call AND the middle text into the echo strip).
        let input = "a <|tool_call_start|>[f(x=1)]<|tool_call_end|> mid <|tool_call_start|>[g(y=2)]<|tool_call_end|> b";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[1].name, "g");
        assert_eq!(text, "a  mid  b");
    }

    #[test]
    fn test_lfm2_tool_call_second_block_malformed_keeps_first_call() {
        // A malformed SECOND block never reaches vLLM's parser — it stops
        // at the first end sentinel, so the bad span is echo region. The
        // first block's call survives; the bad block strips like an echo.
        let input = "<|tool_call_start|>[f(x=1)]<|tool_call_end|> mid <|tool_call_start|>[f(@@@)]<|tool_call_end|> tail";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(text, "mid  tail");
        // An unclosed malformed tail has no end sentinel to cap the echo —
        // it stays verbatim as trailing content, calls still kept.
        let input =
            "<|tool_call_start|>[f(x=1)]<|tool_call_end|> mid <|tool_call_start|>[f(@@@)] tail";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(text, "mid <|tool_call_start|>[f(@@@)] tail");
        // The FIRST block failing is still all-or-nothing verbatim —
        // vLLM's `content=model_output`, second blocks are never seen.
        let input = "<|tool_call_start|>[f(@@@)]<|tool_call_end|> <|tool_call_start|>[g(x=1)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_lfm2_tool_call_trailing_operator_positional_rejected() {
        // `1 +` is an incomplete expression: ast.parse raises SyntaxError,
        // so the block must stay verbatim — the surviving `confirmed=True`
        // kwarg must NOT promote an ok call.
        let input = "<|tool_call_start|>[dangerous_action(1 +, confirmed=True)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_lfm2_tool_call_lambda_defaults_rejected() {
        for inner in [
            "dangerous_action(lambda x=+: x, confirmed=True)",
            "f(lambda x=: x, confirmed=True)",
            "f(lambda x=1 +: x, confirmed=True)",
            "f(lambda x=a b: x, confirmed=True)",
            "f(lambda x=a if b: x, confirmed=True)",
            "f(lambda x=a if (b else c): x, confirmed=True)",
            "f(lambda x=helper(a=1, 2): x, confirmed=True)",
            "f((lambda x=+: x), confirmed=True)",
            "f(lambda x=(lambda y=+: y): x, confirmed=True)",
            "f(lambda x=*a: x, confirmed=True)",
            "f(lambda x=a for a in src: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_defaults_dropped() {
        for inner in [
            "f(lambda x=+1: x, confirmed=True)",
            "f(lambda x=1 + 2, y=-3: x + y, confirmed=True)",
            "f(lambda x=a if b else c: x, confirmed=True)",
            "f(lambda x=a if ready(flag) else b: x, confirmed=True)",
            "f(lambda x=(a if ready(flag) else b): x, confirmed=True)",
            "f(lambda x=helper(a=1): x, confirmed=True)",
            "f(lambda x=helper(lambda y=1: y): x, confirmed=True)",
            "f(lambda x=[a for a in src]: x, confirmed=True)",
            "f(lambda x={1: 2}, y=items[1:]: x, confirmed=True)",
            "f((lambda x=+1: x), confirmed=True)",
            "f(lambda x=(lambda y=1: y): x, confirmed=True)",
            "f(lambda x=lambda y=1: y: x, confirmed=True)",
            "f(lambda x=1, /, *, y=2: x + y, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_lambda_default_depth_cap() {
        for depth in [4, MAX_PY_LITERAL_DEPTH as usize + 1] {
            let expr = format!("{}0{}", "lambda x=".repeat(depth), ": x".repeat(depth));
            let input = format!("<|tool_call_start|>[f({expr}, confirmed=True)]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            if depth == 4 {
                assert!(text.is_empty());
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].arguments.to_string(), "{\"confirmed\":true}");
            } else {
                assert_eq!(text, input);
                assert!(calls.is_empty());
            }
        }
    }

    #[test]
    fn test_lfm2_tool_call_nested_keyword_unpacking_rejected() {
        for inner in [
            "dangerous_action(helper(**a, 2), confirmed=True)",
            "f(helper(**a, *b), confirmed=True)",
            "f(helper(**a, x=1, b), confirmed=True)",
            "f(helper(**a, x=1, *b), confirmed=True)",
            "f(helper(**a, 2,), confirmed=True)",
            "f(lambda x=helper(**a, 2): x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_nested_keyword_unpacking_dropped() {
        for inner in [
            "f(helper(**a), confirmed=True)",
            "f(helper(**a,), confirmed=True)",
            "f(helper(**a, x=1), confirmed=True)",
            "f(helper(**a, **b), confirmed=True)",
            "f(helper(*a, **b), confirmed=True)",
            "f(helper(x=1, **a), confirmed=True)",
            "f(helper(**outer(inner=1), x=2), confirmed=True)",
            "f(helper(**a) + other(*b, 2), confirmed=True)",
            "f(helper(**a,) + other(*b, 2), confirmed=True)",
            "f(outer(helper(**a), 2), confirmed=True)",
            "f(lambda x=helper(**a, x=1): x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_lambda_parameter_order_rejected() {
        for inner in [
            "dangerous_action(lambda a=1,b: x, confirmed=True)",
            "f(lambda a=1,b,c=2: x, confirmed=True)",
            "f(lambda a=1,/,b: x, confirmed=True)",
            "f(lambda a=1,b,*args: x, confirmed=True)",
            "f((lambda a=1,b: x), confirmed=True)",
            "f(lambda a=(lambda b=1,c: x): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_parameter_order_dropped() {
        for inner in [
            "f(lambda a=1,b=2: x, confirmed=True)",
            "f(lambda a,b=1: x, confirmed=True)",
            "f(lambda a=1,: x, confirmed=True)",
            "f(lambda a=1,/: x, confirmed=True)",
            "f(lambda a=1,/,b=2: x, confirmed=True)",
            "f(lambda a=1,*,b: x, confirmed=True)",
            "f(lambda a=1,*args,b: x, confirmed=True)",
            "f(lambda a=1,**kwargs: x, confirmed=True)",
            "f(lambda *,a=1,b: x, confirmed=True)",
            "f(lambda a=1,/,*,b: x, confirmed=True)",
            "f((lambda a=1: a) + (lambda b: b), confirmed=True)",
            "f(lambda a=1,b=(lambda c:c): b, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_incomplete_lambda_markers_rejected() {
        for inner in [
            "dangerous_action(lambda *: x, confirmed=True)",
            "f(lambda /: x, confirmed=True)",
            "f(lambda *,: x, confirmed=True)",
            "f(lambda *,**kw: x, confirmed=True)",
            "f(lambda **: x, confirmed=True)",
            "f(lambda **,: x, confirmed=True)",
            "f(lambda /,a: x, confirmed=True)",
            "f(lambda a=1,*: x, confirmed=True)",
            "f((lambda *: x), confirmed=True)",
            "f(lambda a=(lambda /: x): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_markers_dropped() {
        for inner in [
            "f(lambda: x, confirmed=True)",
            "f(lambda *,a: a, confirmed=True)",
            "f(lambda *,a=1: a, confirmed=True)",
            "f(lambda *,a,**kw: a, confirmed=True)",
            "f(lambda *args: args, confirmed=True)",
            "f(lambda **kw: kw, confirmed=True)",
            "f(lambda *args,**kw: args, confirmed=True)",
            "f(lambda a,/: a, confirmed=True)",
            "f(lambda a,/,b: b, confirmed=True)",
            "f(lambda a,/,*,b: b, confirmed=True)",
            "f(lambda a=1,/,*,b=2: b, confirmed=True)",
            "f((lambda *,a: a) + (lambda b,/: b), confirmed=True)",
            "f(lambda a=(lambda *,b: b): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_lambda_kwargs_terminal_rejected() {
        for inner in [
            "dangerous_action(lambda **kw, extra: value, confirmed=True)",
            "f(lambda **kw, extra=1: value, confirmed=True)",
            "f(lambda **kw, *args: value, confirmed=True)",
            "f(lambda **kw, **more: value, confirmed=True)",
            "f(lambda **kw, /: value, confirmed=True)",
            "f(lambda **kw,,: value, confirmed=True)",
            "f(lambda **kw=1: value, confirmed=True)",
            "f((lambda **kw, extra: value), confirmed=True)",
            "f(lambda a=(lambda **kw, extra: value): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_kwargs_terminal_dropped() {
        for inner in [
            "f(lambda **kw: kw, confirmed=True)",
            "f(lambda **kw,: kw, confirmed=True)",
            "f(lambda a,**kw: a, confirmed=True)",
            "f(lambda a=1,**kw: a, confirmed=True)",
            "f(lambda *args,**kw: args, confirmed=True)",
            "f(lambda *,needed,**kw: needed, confirmed=True)",
            "f((lambda **kw: kw) + (lambda extra: extra), confirmed=True)",
            "f(lambda a=(lambda **kw: kw): a, confirmed=True)",
            "f(lambda **kw: lambda extra: extra, confirmed=True)",
            "f(lambda **kw: helper(extra=1), confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_excess_slice_colons_rejected() {
        for inner in [
            "dangerous_action(a[:::], confirmed=True)",
            "f(a[1:2:3:4], confirmed=True)",
            "f(a[:2::3], confirmed=True)",
            "f(a[::, :::], confirmed=True)",
            "f(a[x[:::]], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_slice_colons_dropped() {
        for inner in [
            "f(a[:], confirmed=True)",
            "f(a[::], confirmed=True)",
            "f(a[1:2:3], confirmed=True)",
            "f(a[:2:], confirmed=True)",
            "f(a[::3], confirmed=True)",
            "f(a[1, ::], confirmed=True)",
            "f(a[::, ::], confirmed=True)",
            "f(a[1:2, 3:4:5], confirmed=True)",
            "f(a[x[::], ::], confirmed=True)",
            "f(a[1:,], confirmed=True)",
            "f(a[..., ::], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_dictionary_unpacking_mode_rejected() {
        for inner in [
            "dangerous_action({**a, b}, confirmed=True)",
            "f({**a, *b}, confirmed=True)",
            "f({*a, **b}, confirmed=True)",
            "f({**a for x in y}, confirmed=True)",
            "f({**a, b, c:1}, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_dictionary_unpacking_dropped() {
        for inner in [
            "f({**a}, confirmed=True)",
            "f({**a,}, confirmed=True)",
            "f({**a, b:1}, confirmed=True)",
            "f({a:1, **b}, confirmed=True)",
            "f({**a, **b}, confirmed=True)",
            "f({**outer(x=1), b:2}, confirmed=True)",
            "f([{**a, b:1}], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_lambda_varargs_defaults_rejected() {
        for inner in [
            "dangerous_action(lambda *args=1: x, confirmed=True)",
            "f(lambda *args=(1): x, confirmed=True)",
            "f((lambda *args=1: x), confirmed=True)",
            "f(lambda a=(lambda *args=1: x): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_varargs_dropped() {
        for inner in [
            "f(lambda *args: args, confirmed=True)",
            "f(lambda *args,: args, confirmed=True)",
            "f(lambda *args,b: b, confirmed=True)",
            "f(lambda *args,b=1: b, confirmed=True)",
            "f(lambda *args,**kw: args, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_repeated_lambda_slash_rejected() {
        for inner in [
            "dangerous_action(lambda x,/,y,/: x, confirmed=True)",
            "f(lambda x,/,/: x, confirmed=True)",
            "f((lambda x,/,y,/: x), confirmed=True)",
            "f(lambda a=(lambda x,/,y,/: x): a, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_slash_dropped() {
        for inner in [
            "f(lambda x,/: x, confirmed=True)",
            "f(lambda x,/,y: y, confirmed=True)",
            "f(lambda x,/,*,y: y, confirmed=True)",
            "f((lambda x,/: x) + (lambda y,/: y), confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_nested_ternary_condition_rejected() {
        for inner in [
            "dangerous_action(a if b if c else d else e, confirmed=True)",
            "f(a if b if c else d, confirmed=True)",
            "f(lambda x=a if b if c else d else e: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_nested_ternaries_dropped() {
        for inner in [
            "f(a if b else c if d else e, confirmed=True)",
            "f(a if (b if c else d) else e, confirmed=True)",
            "f((a if b else c) if d else e, confirmed=True)",
            "f(a if helper(b if c else d) else e, confirmed=True)",
            "f([a if b else c for x in y], confirmed=True)",
            "f(lambda x=a if b else c if d else e: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_malformed_positional_fstrings_rejected() {
        for inner in [
            "dangerous_action(f'{', confirmed=True)",
            "f(f'}', confirmed=True)",
            "f(f'{x', confirmed=True)",
            "f(f'\\{', confirmed=True)",
            "f(fr'{', confirmed=True)",
            "f(f'{}', confirmed=True)",
            "f(f'{+}', confirmed=True)",
            "f(f'{a b}', confirmed=True)",
            "f(f'{x!q}', confirmed=True)",
            "f(f'{x:{y:{z:{w}}}}', confirmed=True)",
            "f(f\"{b'\\xGG'}\", confirmed=True)",
            "f(f\"{'\\xGG'}\", confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_positional_fstrings_dropped() {
        for inner in [
            "f(f'plain', confirmed=True)",
            "f(f'{{{{', confirmed=True)",
            "f(f'}}}}', confirmed=True)",
            "f(f'{{{{x}}}}', confirmed=True)",
            "f(rf'{{{{x}}}}', confirmed=True)",
            "f(f'{user}', confirmed=True)",
            "f(f'{a + b}', confirmed=True)",
            "f(f'{(a,b)}', confirmed=True)",
            "f(f'{value!r}', confirmed=True)",
            "f(f'{value:03}', confirmed=True)",
            "f(f'{value:{width}}', confirmed=True)",
            "f(f'{x=}', confirmed=True)",
            "f(f'{user = }', confirmed=True)",
            "f(f'{user = !r}', confirmed=True)",
            "f(f'{user = :03}', confirmed=True)",
            "f(f'{user!r }', confirmed=True)",
            "f(f'{x:{y:{z}}}', confirmed=True)",
            "f(f\"{r'\\xGG'}\", confirmed=True)",
            "f(f\"{br'\\xGG'}\", confirmed=True)",
            "f(f\"{f'{x}'}\", confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_reserved_target_attributes_rejected() {
        for inner in [
            "dangerous_action([x for a.for in source], confirmed=True)",
            "f([x for a.lambda in source], confirmed=True)",
            "f([x for a.None in source], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_target_attributes_dropped() {
        for inner in [
            "f([x for a.match in source], confirmed=True)",
            "f([x for a.café in source], confirmed=True)",
            "f([x for a.b[0] in source], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_unparenthesized_restricted_lambdas_rejected() {
        for inner in [
            "dangerous_action([x for a in source if lambda: value], confirmed=True)",
            "f([x for a in lambda: value], confirmed=True)",
            "f([x for a in source for b in lambda: value], confirmed=True)",
            "f(a if lambda: value else b, confirmed=True)",
            "f(lambda: x if lambda: y else z, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_expression_positions_dropped() {
        for inner in [
            "f([lambda: x for x in source], confirmed=True)",
            "f([x for a in source if (lambda: value)], confirmed=True)",
            "f([x for a in (lambda: value)], confirmed=True)",
            "f([x for a in source for b in (lambda: value)], confirmed=True)",
            "f(a if (lambda: value) else b, confirmed=True)",
            "f([(lambda: x) for x in source], confirmed=True)",
            "f(lambda: x, confirmed=True)",
            "f(helper(lambda: x), confirmed=True)",
            "f({key: lambda: value}, confirmed=True)",
            "f(lambda x=lambda: value: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_not_in_restricted_rhs_rejected() {
        for inner in [
            "dangerous_action(a + not b, confirmed=True)",
            "f(a * not b, confirmed=True)",
            "f(a == not b, confirmed=True)",
            "f(a in not b, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_not_positions_dropped() {
        for inner in [
            "f(not b, confirmed=True)",
            "f(not not b, confirmed=True)",
            "f(a and not b, confirmed=True)",
            "f(a or not b, confirmed=True)",
            "f(a is not b, confirmed=True)",
            "f(a if not b else c, confirmed=True)",
            "f(a if b else not c, confirmed=True)",
            "f({k: not b}, confirmed=True)",
            "f(helper(x=not b), confirmed=True)",
            "f((x := not b), confirmed=True)",
            "f(a[not b:], confirmed=True)",
            "f(lambda x=not b: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_lambda_in_operator_rhs_rejected() {
        for inner in [
            "dangerous_action(a + lambda: b, confirmed=True)",
            "f(a and lambda: b, confirmed=True)",
            "f(a == lambda: b, confirmed=True)",
            "f(a is lambda: b, confirmed=True)",
            "f(a in lambda: b, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_lambda_rhs_positions_dropped() {
        for inner in [
            "f((lambda: b), confirmed=True)",
            "f(a + (lambda: b), confirmed=True)",
            "f(a and (lambda: b), confirmed=True)",
            "f(a if b else lambda: c, confirmed=True)",
            "f({k: lambda: b}, confirmed=True)",
            "f(helper(x=lambda: b), confirmed=True)",
            "f(a[lambda: b], confirmed=True)",
            "f(lambda x=lambda: b: x, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_empty_postfix_subscripts_rejected() {
        for inner in [
            "dangerous_action(x[], confirmed=True)",
            "f(x[ ], confirmed=True)",
            "f(x.y[], confirmed=True)",
            "f(helper(x[]), confirmed=True)",
            "f([x for a[] in source], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_empty_displays_and_subscripts_dropped() {
        for inner in [
            "f([], confirmed=True)",
            "f([ ], confirmed=True)",
            "f(x[:], confirmed=True)",
            "f(x[::], confirmed=True)",
            "f(x[()], confirmed=True)",
            "f(x[[]], confirmed=True)",
            "f(helper()[0], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_invalid_double_star_placements_rejected() {
        for inner in [
            "dangerous_action([**items], confirmed=True)",
            "f((**items,), confirmed=True)",
            "f((**items), confirmed=True)",
            "f(x[**items], confirmed=True)",
            "f({*items, **other}, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_double_star_placements_dropped() {
        for inner in [
            "f(helper(**items), confirmed=True)",
            "f({**items}, confirmed=True)",
            "f({**items, key: 1}, confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_non_ascii_bytes_literals_rejected() {
        for inner in [
            "dangerous_action(b'é', confirmed=True)",
            "f(B'é', confirmed=True)",
            "f(br'é', confirmed=True)",
            "f(rb'abcé', confirmed=True)",
            "f(b'''é''', confirmed=True)",
            "f(b'\\xGG', confirmed=True)",
            "f(b'\\x', confirmed=True)",
            "f(b'\\x0', confirmed=True)",
            "f('\\xGG', confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_bytes_literals_dropped() {
        for inner in [
            "f(b'abc', confirmed=True)",
            "f(b'\\xc3\\xa9', confirmed=True)",
            "f(br'\\xc3', confirmed=True)",
            "f(rb'plain', confirmed=True)",
            "f(br'\\xGG', confirmed=True)",
            "f(b'\\x4f', confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_control_flow_positionals_dropped() {
        for inner in [
            "f(await value, confirmed=True)",
            "f(await helper(), confirmed=True)",
            "f(value + await other, confirmed=True)",
            "f(not await value, confirmed=True)",
            "f(await (lambda: value), confirmed=True)",
            "f((yield), confirmed=True)",
            "f((yield value), confirmed=True)",
            "f((yield *values), confirmed=True)",
            "f((yield lambda: value), confirmed=True)",
            "f((yield from values), confirmed=True)",
            "f([x async for x in values], confirmed=True)",
            "f([await x for x in values], confirmed=True)",
            "f((x async for x in values), confirmed=True)",
            "f([x for x in values async for y in others], confirmed=True)",
            "f([x async for *item in values], confirmed=True)",
            "f([x async for *item, in values], confirmed=True)",
            "f([x async for (*item,) in values], confirmed=True)",
            "f([x async for factory().slot in values], confirmed=True)",
            "f([x async for factory()[index] in values], confirmed=True)",
            "f([x async for (factory()).slot in values], confirmed=True)",
            "f([x async for [factory()].slot in values], confirmed=True)",
            "f([x async for (left + right).slot in values], confirmed=True)",
            "f([x async for (factory(),).slot in values], confirmed=True)",
            "f([x async for [factory(), other].slot in values], confirmed=True)",
            "f([x async for (await value)[index] in values], confirmed=True)",
            "f([x async for (not value).slot in values], confirmed=True)",
            "f([x async for (lambda: value).slot in values], confirmed=True)",
            "f([x async for True.slot in values], confirmed=True)",
            "f([x async for 1[index] in values], confirmed=True)",
            "f([x async for {}.slot in values], confirmed=True)",
            "f([x async for 's'.slot in values], confirmed=True)",
            "f([x async for () in values], confirmed=True)",
            "f([x async for [] in values], confirmed=True)",
            "f([x async for [*item] in values], confirmed=True)",
            "f([x async for *left, *right in values], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_malformed_control_flow_positionals_rejected() {
        for inner in [
            "f(await, confirmed=True)",
            "f(await -value, confirmed=True)",
            "f(await await value, confirmed=True)",
            "f(yield value, confirmed=True)",
            "f(yield from values, confirmed=True)",
            "f((yield from values, other), confirmed=True)",
            "f((yield value for value in values), confirmed=True)",
            "f([x async for (*item) in values], confirmed=True)",
            "f([x for (*item) in values], confirmed=True)",
            "f([x async for factory() in values], confirmed=True)",
            "f([x async for (factory()) in values], confirmed=True)",
            "f([x async for [factory()] in values], confirmed=True)",
            "f([q for cache[[z for factory() in values]] in rows], confirmed=True)",
            "f([q for cache[[z for x + y in values]] in rows], confirmed=True)",
            "f([q for cache[[z for True in values]] in rows], confirmed=True)",
            "f([x for 1 + item.slot in values], confirmed=True)",
            "f([x for True and item[index] in values], confirmed=True)",
            "f(async, confirmed=True)",
            "f([x async x in values], confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_nested_reserved_kwargs_dropped() {
        for inner in [
            "f(helper(await=1), confirmed=True)",
            "f(helper(from=1), confirmed=True)",
            "f(helper(not=1), confirmed=True)",
            "f(helper(lambda=1), confirmed=True)",
            "f(helper(yield=1), confirmed=True)",
            "f(helper(async=1), confirmed=True)",
            "f(helper(True=1), confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_malformed_positionals_rejected() {
        // Each of these is a SyntaxError to ast.parse — verbatim, no call.
        for inner in [
            "f(x and)",   // trailing word operator
            "f(a b)",     // juxtaposed operands
            "f((a +))",   // trailing operator inside a group
            "f((,))",     // empty first element
            "f(5x)",      // letter glued to a number
            "f(a ! b)",   // bare `!` is not a Python operator
            "f(b 'x')",   // spaced string prefix is a name, not a literal
            "f(x not)",   // `not` with no operand
            "f(a > b <)", // trailing operator
            "f(1e+)",     // exponent marker with no digits
            "f(1e)",      // bare `e` after digits
            "f(1e-)",     // signed exponent with no digits
            "f(.5e)",     // same hole on the `.`-led float path
            "f(5.e)",     // `5.` then `e` — juxtaposed name, SyntaxError
            "f(.5.5)",    // second `.` in a `.`-led float
            "f(0xg)",     // no hex digits at all
            "f(0x1g)",    // non-hex letter glued to the digit run
            "f(0b12)",    // digit outside the selected radix
            "f(0o18)",
            "f(a if b, confirmed=True)", // incomplete ternary — `else` owed
            "f(a else b)",               // `else` with no open `if`
            "f((a if b), x=1)",          // bracket closes mid-ternary
            "f([a if b], x=1)",          // `if` outside a comprehension
            "f([a if b for c in d], x=1)", // `for` doesn't rescue the ternary
            "f(x for x in y, k=1)",      // genexpr must be the sole argument
            "f(a not is b, x=1)",        // `not is` isn't an operator
            "f(a not b, x=1)",           // `not` isn't binary — only `in` follows
            "f(a not (b), x=1)",
            "f(a is not in b, x=1)", // `is not` can't chain into `in`
            "f(lambda 1: x, confirmed=True)", // digit-led param name
            "f(lambda 'a': x)",      // string where a param name belongs
            "f(lambda ,: x)",        // `,` before any param
            "f(lambda =x: y)",       // `=` with no name
            "f(lambda (a): x)",      // `(a)` params are Python 2
            "f(lambda a,,b: x)",     // empty param between commas
            "f({1:}, confirmed=True)", // dict entry missing its value
            "f({:}, x=1)",           // `:` where a key belongs
            "f([1:2], x=1)",         // `:` inside a list display
            "f((1:2), x=1)",         // `:` inside parens
            "f([a: b], x=1)",        // `:` inside a list via idents
            "f([x for x], confirmed=True)", // `for` with no `in`
            "f([x for x for y in z], x=1)", // first `for` missing `in`
            "f((x for x), y=1)",     // genexpr missing `in`
            "f(x for x)",            // sole-arg genexpr missing `in`
            "f([x for (a in b)], y=1)", // `in` inside target parens
            "f([x for x + y in z], confirmed=True)", // `x + y` is no target
            "f([x for x and y in z], x=1)", // `and` in a target
            "f([x for 1 in z], x=1)", // literal target
            "f([x for 's' in z], x=1)", // string target
            "f([x for a(b) in z], x=1)", // call target
            "f([x for a.b c in z], x=1)", // juxtaposed names
            "f([x for -a in z], x=1)", // unary target
            "f([x for {a: b} in z], x=1)", // dict target
            "f([x for not a in z], x=1)", // keyword target
            "f([x for a + if b in z], x=1)", // operator then keyword
            "f(x for x + y in z)",   // genexpr target
            "f(lambda for: x, confirmed=True)", // keyword param name
            "f(lambda True: x)",     // constant param name
            "f(lambda None: x)",     // `None` param name
            "f(lambda a b: x)",      // param without `,`
            "f(lambda a.b: x)",      // dotted param name
            "f(lambda a(x): y)",     // call-shaped param
            "f([*items for item in source], confirmed=True)", // starred comp elem
            "f(*a for a in b, x=1)", // starred genexpr elem
            "f((*a), x=1)",          // bare starred group
            "f(x[a for a in y], x=1)", // genexpr in subscript
            "f([x for a[i for j in z] in w], x=1)", // nested subscript genexpr
            "f({1, 2:3}, confirmed=True)", // set element then `:` — mixed literal
            "f({1:2, 3}, confirmed=True)", // dict pair then bare element
            "f({x:=1, a:2}, x=1)",   // walrus locks set — then `:`
            "f(1 + *a, x=1)",        // `*` after an operator
            "f(x[1:*a], x=1)",       // `*` after a slice colon
            "f(helper(5=1), x=1)",   // `=` after a non-name operand
            "f(helper(a.b=1), x=1)", // `=` after an attribute
            "f(helper(*a=1), x=1)",  // `=` after a starred element
            "f(x[a=1], x=1)",        // `=` inside a subscript
            "f({a=1}, x=1)",         // `=` inside a display
            "f([a, x for x in y], x=1)", // comp after an element `,`
            "f({k:v, x for x in y}, x=1)", // comp after a dict `,`
            "f({k:v, k2}, x=1)",     // dict key missing its `:`
            "f(x for x in y, z=1)",  // a genexpr must be the sole arg
            "f(1:=2, confirmed=True)", // literal walrus target
            "f(True:=2, confirmed=True)",
            "f(False:=2, confirmed=True)",
            "f(None:=2, confirmed=True)",
            "f(a.b:=2, x=1)",                    // attribute walrus target
            "f(x[0]:=2, x=1)",                   // subscript walrus target
            "f((x,y):=2, x=1)",                  // tuple walrus target
            "f(g():=2, x=1)",                    // call walrus target
            "f((x):=2, x=1)",                    // parenthesized walrus target
            "f(a+b:=2, x=1)",                    // operator walrus target
            "f(x=y:=2, x=1)",                    // walrus after `=`
            "f({k: v:=1}, x=1)",                 // walrus as a dict value
            "f(not x:=1, x=1)",                  // walrus after `not`
            "f(helper(x=1, 2), confirmed=True)", // positional after kwarg
            "f(helper(x=1, *a, b), x=1)",        // positional after kwarg+star
            "f(helper(*a, x=1, b), x=1)",        // positional after kwarg, later star
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // The exact reviewer case: the surviving kwarg must not promote
        // the call once the positional is proven malformed.
        let input = "<|tool_call_start|>[dangerous_action(1e+, confirmed=True)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_lfm2_tool_call_keyword_call_names_rejected() {
        // A Python keyword can't be the call name — `for(x)` is a
        // SyntaxError and `None(x)`/`True(x)` parse to a `Constant` with
        // no `.func.id` under vLLM's AST extraction. The whole block must
        // stay verbatim; a surviving kwarg must not promote it.
        for inner in [
            "for(confirmed=True)",
            "lambda(x=1)",
            "def()",
            "return()",
            "class()",
            "import()",
            "None()",
            "True(x=1)",
            "False()",
            "a.for(x=1)", // keyword after a dot
            "a.lambda()",
            "a.None(x=1)",
            "not(x=1)", // unary-op keyword can't name a call
            "in(x=1)",
            "😀(confirmed=True)", // emoji is no XID ident — SyntaxError
            "a.😀(x=1)",
            "café.😀(x=1)",
            "5café(x=1)", // digit-led name
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // Soft keywords stay callable; normal names are unaffected.
        let input = "<|tool_call_start|>[match(x=1)]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "match");

        // Unicode XID names promote exactly like ASCII — `café`/`变量`
        // are legal Python identifiers.
        for (inner, want) in [("café(x=1)", "café"), ("变量(x=1)", "变量")] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (_, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, want, "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_malformed_unicode_escapes_rejected() {
        for inner in [
            "f('\\uZZZZ', confirmed=True)",
            "f('\\u12', confirmed=True)",
            "f('\\u', confirmed=True)",
            "f('\\U0001F60', confirmed=True)",
            "f('\\U', confirmed=True)",
            "f('\\U00110000', confirmed=True)",
            "f('\\UFFFFFFFF', confirmed=True)",
            "f('\\N{BULLET', confirmed=True)",
            "f('\\N{}', confirmed=True)",
            "f('\\N{BULLET}', confirmed=True)",
            "f('\\N{NOT A REAL NAME}', confirmed=True)",
            "f('\\N', confirmed=True)",
            "f('C:\\New folder', confirmed=True)",
            "f(\"\\uZZZZ\", confirmed=True)",
            "f(f'\\uZZZZ', confirmed=True)",
            "f(f'\\N', confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        for inner in [
            "f('\\u1234', confirmed=True)",
            "f('\\U0001F600', confirmed=True)",
            "f('\\U0010FFFF', confirmed=True)",
            "f('\\uD800', confirmed=True)",
            "f('\\q', confirmed=True)",
            "f('\\400', confirmed=True)",
            "f(b'\\uZZZZ', confirmed=True)",
            "f(b'\\N', confirmed=True)",
            "f(b'\\N{X}', confirmed=True)",
            "f(r'\\uZZZZ', confirmed=True)",
            "f(r'\\N', confirmed=True)",
            "f(rb'\\uZZZZ', confirmed=True)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(
                calls[0].arguments.to_string(),
                "{\"confirmed\":true}",
                "{inner}"
            );
        }
    }

    #[test]
    fn test_lfm2_tool_call_nfkc_normalizes_identifiers() {
        let input = "<|tool_call_start|>[K(x=1), a.K(y=2), ｆｏｒ(z=3)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "K");
        assert_eq!(calls[1].name, "a.K");
        assert_eq!(calls[2].name, "for");

        let input = "<|tool_call_start|>[f(K=1, K=2)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments.to_string(), "{\"K\":2}");
    }

    #[test]
    fn test_lfm2_tool_call_valid_expressions_still_dropped() {
        // Real Python expressions stay droppable positionals — the call is
        // promoted from its kwargs exactly like before.
        for (inner, want_args) in [
            ("f(a or b, x=1)", "{\"x\":1}"),
            ("f(x if c else y, x=1)", "{\"x\":1}"),
            ("f([i for i in xs], x=1)", "{\"x\":1}"),
            ("f(a not in b, x=1)", "{\"x\":1}"),
            ("f(a is not None, x=1)", "{\"x\":1}"),
            ("f(-x, x=1)", "{\"x\":1}"),
            ("f(a.b[c], x=1)", "{\"x\":1}"),
            ("f(g(x), x=1)", "{\"x\":1}"),
            ("f(\"a\" \"b\", x=1)", "{\"x\":1}"),
            ("f(rb\"x\", x=1)", "{\"x\":1}"),
            ("f(0x1f + 1e5, x=1)", "{\"x\":1}"),
            ("f(x := 1, y=2)", "{\"y\":2}"),
            ("f(a == b, x=1)", "{\"x\":1}"),
            ("f(a <= b, x=1)", "{\"x\":1}"),
            ("f(*args, x=1)", "{\"x\":1}"),
            ("f(lambda v: v * 2, x=1)", "{\"x\":1}"),
            ("f((a, b), x=1)", "{\"x\":1}"),
            ("f(a[1:], x=1)", "{\"x\":1}"),
            ("f({1: 2}, x=1)", "{\"x\":1}"),
            ("f(5. + .5, x=1)", "{\"x\":1}"),
            ("f(.5e3, x=1)", "{\"x\":1}"),
            ("f(5.e3, x=1)", "{\"x\":1}"),
            ("f(1_0e1_0, x=1)", "{\"x\":1}"),
            ("f(.5_0e1, x=1)", "{\"x\":1}"),
            ("f(0x1e5, x=1)", "{\"x\":1}"), // `e` is a hex digit here
            ("f(0o17, x=1)", "{\"x\":1}"),
            ("f(0b101, x=1)", "{\"x\":1}"),
            ("f(a if b else c, x=1)", "{\"x\":1}"), // complete ternary
            ("f(a if b else c if d else e, x=1)", "{\"x\":1}"),
            ("f([x for i in y if z], k=1)", "{\"k\":1}"), // comp filter
            ("f([x for i in y if a if b], k=1)", "{\"k\":1}"), // chained filters
            // Nested comprehension, then an outer-clause filter.
            ("f([x for i in [j for j in y] if q], k=1)", "{\"k\":1}"),
            ("f([a if b else c for i in d], k=1)", "{\"k\":1}"), // ternary elem
            ("f([x for i in (a if b else c)], k=1)", "{\"k\":1}"),
            ("f(x for i in y)", "{}"), // sole genexpr positional
            ("f(x for i in y if i)", "{}"),
            ("f(a not in b, x=1)", "{\"x\":1}"), // `not in` compound
            ("f(a is not b, x=1)", "{\"x\":1}"), // `is not` via unary `not`
            ("f(a not in b and c, x=1)", "{\"x\":1}"),
            ("f(lambda a, b: a + b, x=1)", "{\"x\":1}"),
            ("f(lambda: x, y=1)", "{\"y\":1}"), // zero-param lambda
            ("f(lambda a,: a, x=1)", "{\"x\":1}"), // trailing comma
            ("f(lambda *a, b=1: a, x=1)", "{\"x\":1}"),
            ("f(lambda a, /, b: a, x=1)", "{\"x\":1}"),
            ("f(x[1:2], y=1)", "{\"y\":1}"), // subscript slice
            ("f(x[:], y=1)", "{\"y\":1}"),   // open slice
            ("f(x[::2], y=1)", "{\"y\":1}"),
            ("f(x[a:b:c], y=1)", "{\"y\":1}"),
            ("f(x[1:], y=1)", "{\"y\":1}"),     // trailing-open slice
            ("f(x[y[1:2]], y=1)", "{\"y\":1}"), // nested subscript
            ("f(x[1:2, 3], y=1)", "{\"y\":1}"), // tuple index
            ("f({1:2}, y=1)", "{\"y\":1}"),     // dict literal positional
            ("f({k: v for k in y}, y=1)", "{\"y\":1}"), // dict comprehension
            ("f([x for a in y for b in z], y=1)", "{\"y\":1}"), // chained fors
            ("f([x for a in b in c], y=1)", "{\"y\":1}"), // `in` in iterable
            ("f([x for a in (b in c)], y=1)", "{\"y\":1}"), // paren iterable
            ("f([x for a, b in y], y=1)", "{\"y\":1}"), // tuple target
            ("f([x for (a, b) in y], y=1)", "{\"y\":1}"), // group target
            ("f([x for [a, b] in y], y=1)", "{\"y\":1}"), // list target
            ("f([x for a.b in y], y=1)", "{\"y\":1}"), // attribute target
            ("f([x for a[i + 1] in y], y=1)", "{\"y\":1}"), // subscript target
            ("f([x for a, *b in y], y=1)", "{\"y\":1}"), // starred target
            ("f([x for a[(i for i in z)] in w], y=1)", "{\"y\":1}"), // paren genexpr idx
            ("f([x for a[b in c] in w], y=1)", "{\"y\":1}"), // membership idx
            ("f((x for x in y), y=1)", "{\"y\":1}"), // parenthesized genexpr
            ("f(x[(i for i in y)], y=1)", "{\"y\":1}"), // paren genexpr index
            ("f(outer(helper(x=1), z=2), y=3)", "{\"y\":3}"), // nested kwarg call
            ("f(helper(a, x=1), y=2)", "{\"y\":2}"), // nested pos+kwarg
            ("f(helper(x=1, y=2), z=3)", "{\"z\":3}"), // nested kwarg chain
            ("f(helper(x=1, *a), y=2)", "{\"y\":2}"), // kwarg then star
            ("f(helper(*a, x=1), y=2)", "{\"y\":2}"), // star then kwarg
            ("f(x:=1, y=2)", "{\"y\":2}"),      // walrus positional
            ("f((x:=1), y=2)", "{\"y\":2}"),    // parens walrus
            ("f(x[a:=1], y=2)", "{\"y\":2}"),   // walrus in subscript
            ("f({x:=1}, y=2)", "{\"y\":2}"),    // walrus set element
            ("f({k: (v:=1)}, y=2)", "{\"y\":2}"), // walrus dict value
            ("f({1, 2}, y=1)", "{\"y\":1}"),    // set literal
            ("f({*a}, y=1)", "{\"y\":1}"),      // starred set
            ("f({'a': 1, 'b': 2}, y=1)", "{\"y\":1}"), // dict literal
            ("f({'a': 1,}, y=1)", "{\"y\":1}"), // dict trailing comma
            ("f({}, y=1)", "{\"y\":1}"),        // empty dict
            ("f((*a, b), y=1)", "{\"y\":1}"),   // starred tuple
            ("f([*a], y=1)", "{\"y\":1}"),      // starred list
            ("f(x[*a], y=1)", "{\"y\":1}"),     // starred index
            ("f(lambda match: x, y=1)", "{\"y\":1}"), // soft-keyword param
            ("f(lambda a1: x, y=1)", "{\"y\":1}"), // digit-suffix param
            ("f(lambda a, *b: x, y=1)", "{\"y\":1}"), // starred param
            ("f(lambda x=(1, 2), y='a': x, k=1)", "{\"k\":1}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].name, "f", "{inner}");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_bad_numeric_separators_rejected() {
        // Misplaced `_` is a SyntaxError to ast.parse (PEP 515: separators
        // sit between digits, or directly after a `0x`/`0o`/`0b` prefix).
        // The whole block must stay verbatim — both as a kwarg value and
        // as a skipped positional.
        for inner in [
            "f(count=1__0)",  // consecutive separators
            "f(count=1_)",    // trailing separator
            "f(count=1_.5)",  // separator before `.`
            "f(count=1._5)",  // separator after `.`
            "f(count=1e_5)",  // separator before exponent digits
            "f(count=1_e5)",  // separator after `e`
            "f(count=0x__f)", // consecutive separators after prefix
            "f(count=0x_f_)", // trailing separator in radix digits
            "f(count=0b1_2)", // separator next to a non-radix digit
            "f(count=0x)",    // radix prefix with no digits
            "f(1__0, x=1)",   // malformed positional must not promote kwargs
            "f(1_, x=1)",
            "f(0x_f_, x=1)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
    }

    #[test]
    fn test_lfm2_tool_call_valid_numeric_separators() {
        // PEP 515-legal separators still parse — kwarg values normalize by
        // stripping the `_`s, valid positionals still drop.
        for (inner, want_args) in [
            ("f(count=1_000)", "{\"count\":1000}"),
            ("f(count=1_0.5_0)", "{\"count\":10.5}"),
            ("f(count=1_0e1)", "{\"count\":100.0}"),
            ("f(count=0x_ff)", "{\"count\":255}"),
            ("f(count=0xff_ff)", "{\"count\":65535}"),
            ("f(count=0b1_0)", "{\"count\":2}"),
            ("f(count=0o7_7)", "{\"count\":63}"),
            ("f(1_000, x=1)", "{\"x\":1}"),
            ("f(0x_ff, x=1)", "{\"x\":1}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_string_prefixes() {
        // Only r/u/f/b alone and the r-pairs are real Python prefixes —
        // any other combination makes the glued quote a SyntaxError, so
        // the whole block stays verbatim (kwarg and positional paths).
        for inner in [
            "f(v=rr'x')", // repeated prefix
            "f(v=uu'x')", // repeated prefix
            "f(v=ff'x')", // repeated prefix
            "f(v=uf'x')", // u pairs with nothing
            "f(v=fu'x')",
            "f(v=ru'x')",
            "f(v=ur'x')",
            "f(rr'x', y=1)", // positional path via string_prefix
            "f(uf'x', y=1)",
            "f(u 'x', y=1)", // spaced prefix is a name, not a literal
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // Legal prefixes still parse; b-family stays rejected for
        // JSON-representability (pre-existing behavior, unchanged).
        for (inner, want_args) in [
            ("f(v=r'x')", "{\"v\":\"x\"}"),
            ("f(v=u'x')", "{\"v\":\"x\"}"),
            ("f(v=f'x')", "{\"v\":\"x\"}"),
            ("f(v=rf'x')", "{\"v\":\"x\"}"),
            ("f(v=fr'x')", "{\"v\":\"x\"}"),
            ("f(v=Rf'x')", "{\"v\":\"x\"}"),
            ("f(r'x', y=1)", "{\"y\":1}"),
            ("f(rb'x', y=1)", "{\"y\":1}"), // positional b'..' is valid Python
            ("f(fr'x', y=1)", "{\"y\":1}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_huge_integer_rejected() {
        // Past u64::MAX serde_json holds no exact representation — the old
        // f64 fallback turned 18446744073709551617 into
        // 18446744073709551616 while still returning an ok call. Now the
        // block stays verbatim like every non-JSON-representable literal.
        for inner in [
            "f(id=18446744073709551617)",                    // u64::MAX + 1
            "f(id=-18446744073709551617)",                   // magnitude past u64::MAX
            "f(id=-9223372036854775809)",                    // i64::MIN - 1 (u64 fits, i64 doesn't)
            "f(id=0x1_0000_0000_0000_0000)",                 // 2^64 via radix
            "f(id=340282366920938463463374607431768211455)", // i128::MAX
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // The exact-boundary values still parse — including the double
        // negation that routes -i64::MIN through the u64 edge.
        for (inner, want_args) in [
            (
                "f(id=18446744073709551615)",
                "{\"id\":18446744073709551615}",
            ), // u64::MAX
            (
                "f(id=-9223372036854775808)",
                "{\"id\":-9223372036854775808}",
            ), // i64::MIN
            ("f(id=0xffffffffffffffff)", "{\"id\":18446744073709551615}"),
            (
                "f(id=--9223372036854775808)",
                "{\"id\":9223372036854775808}",
            ), // -(i64::MIN) = 2^63
            ("f(id=--5)", "{\"id\":5}"),
            ("f(id=+-5)", "{\"id\":-5}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_nonfinite_float_rejected() {
        // Overflowing float literals parse to ±inf and serde_json turns
        // Value::from(inf) into Null — an ok call with `count: null`
        // corrupts the argument. Verbatim, like the huge-int path.
        for inner in [
            "f(count=1e999)",  // +inf
            "f(count=-1e999)", // -inf
            "f(count=1e309)",  // past f64::MAX (~1.8e308)
            "f(count=.5e999)",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }
        // Underflow to 0.0 and ordinary finite values still parse.
        for (inner, want_args) in [
            ("f(count=1e-999)", "{\"count\":0.0}"),
            ("f(count=1e308)", "{\"count\":1e+308}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_echo_no_second_end() {
        // Echo body without a closing sentinel still holds (rfind finds
        // nothing to strip through) — trailing starts with '[' and stays
        // part of content per vLLM `_strip_echo` on the non-streaming path:
        // with no orphan end there is nothing to strip, so it IS content.
        let input = "<|tool_call_start|>[f(x=1)]<|tool_call_end|>[f(x=1)]";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(text, "[f(x=1)]");
    }

    #[test]
    fn test_lfm2_tool_call_reserved_kwarg_and_positional() {
        // `from` is a Python keyword — vLLM renames it to parse; we accept
        // it natively.
        let input = "<|tool_call_start|>[mem.get(from=1)]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, serde_json::json!({"from": 1}));

        // A positional AFTER a keyword is a Python SyntaxError — vLLM's
        // ast.parse rejects, so the whole block stays raw with no calls.
        let input = "<|tool_call_start|>[mem.get(from=1, 'dropped')]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_lfm2_tool_call_failures_keep_raw_text() {
        // vLLM: parse failure → tools_called=False, content=model_output
        // verbatim (sentinels included). Each of these rejects the whole
        // block: bare name, constant element, `**` spread, f-string
        // placeholder, empty list, missing bracket.
        for input in [
            "<|tool_call_start|>[foo]<|tool_call_end|>",
            "<|tool_call_start|>[f(), 42]<|tool_call_end|>",
            "<|tool_call_start|>[f(**cfg)]<|tool_call_end|>",
            "<|tool_call_start|>[f(x=f'{y}')]<|tool_call_end|>",
            "<|tool_call_start|>[]<|tool_call_end|>",
            "<|tool_call_start|>f(x=1)<|tool_call_end|>",
            "<|tool_call_start|>[f(x=undefined)]<|tool_call_end|>",
        ] {
            let (text, calls) = parse_tool_calls(input);
            assert_eq!(text, input, "failure must keep raw text: {input}");
            assert!(calls.is_empty(), "failure must yield no calls: {input}");
        }
    }

    #[test]
    fn test_lfm2_tool_call_unclosed_start() {
        // Stream ended mid-call (max_tokens): vLLM treats text after the
        // start sentinel as the call body; parse still attempted.
        let input = "Intro.<|tool_call_start|>[f(x=1)]";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(text, "Intro.");
    }

    #[test]
    fn test_lfm2_tool_call_string_escapes_and_newline() {
        let input = "<|tool_call_start|>[run(cmd='a\\nb', raw=r'c\\d', lit='line1\nline2')]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        let a = &calls[0].arguments;
        assert_eq!(a["cmd"], "a\nb");
        assert_eq!(a["raw"], "c\\d");
        assert_eq!(a["lit"], "line1\nline2"); // raw newline tolerated
    }

    /// A backslash escapes the closing quote even in RAW strings (the
    /// backslash stays in the value), so `r'foo\'` is unterminated in
    /// Python — the block must stay verbatim, not promote `foo\`.
    #[test]
    fn test_lfm2_tool_call_raw_string_escaped_quote() {
        // Odd backslash run before the quote → unterminated → reject.
        for inner in [
            "f(value=r'foo\\')",
            "f(value=r'foo\\\\\\')",
            "f(value=rf'foo\\')",
            "f(r'foo\\', x=1)", // positional — same unterminated literal
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // An even run ends the literal normally; the escaped quote inside
        // keeps both bytes (raw strings retain the backslash).
        for (inner, want) in [
            ("f(value=r'foo\\\\')", "foo\\\\"),
            ("f(value=r'foo\\'')", "foo\\'"),
            ("f(value='foo\\\\')", "foo\\"),
            ("f(value='foo\\\'')", "foo'"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments["value"], want, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    /// `\N` is a recognized escape introducer: a bare `\N` is a hard
    /// SyntaxError in Python, and every `\N{NAME}` needs a Unicode-name
    /// table we don't carry — the block stays verbatim rather than
    /// shipping the literal escape text as the argument.
    #[test]
    fn test_lfm2_tool_call_named_unicode_rejected() {
        for inner in [
            "f(value='\\N{SNOWMAN}')",
            "f(value='x\\N{BAD}y')",
            "f(value=\"\\N{X}\")",
            "f(value='\\N')",
            "f(value='C:\\New folder')",
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(text, input, "{inner} must stay verbatim");
            assert!(calls.is_empty(), "{inner} must not promote a call");
        }

        // Unrecognized escapes (`\d`) keep the backslash like Python, and
        // raw/bytes literals have no `\N` escape at all — Python only warns
        // there, so those positional literals stay valid.
        for (inner, want_args) in [
            ("f(value='\\d')", "{\"value\":\"\\\\d\"}"),
            ("f(value=r'\\N')", "{\"value\":\"\\\\N\"}"),
            ("f(b'\\N', x=1)", "{\"x\":1}"),
            ("f(r'\\N', x=1)", "{\"x\":1}"),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert_eq!(calls.len(), 1, "{inner} must produce one call");
            assert_eq!(calls[0].arguments.to_string(), want_args, "{inner}");
            assert_eq!(text, "", "{inner}");
        }
    }

    /// Trailing/leading-dot floats (`1.`, `.5`) keep their value — a
    /// botched normalization must not turn `1.` into `10`.
    #[test]
    fn test_lfm2_tool_call_dot_floats() {
        let input = "<|tool_call_start|>[f(a=1., b=.5, c=2.5)]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["a"], 1.0);
        assert_eq!(calls[0].arguments["b"], 0.5);
        assert_eq!(calls[0].arguments["c"], 2.5);
    }

    /// Positional EXPRESSIONS (`f(x)`, `f(a+b)`, `f(len(a))`, `f(*args)`)
    /// parse fine under ast.parse — vLLM accepts the call and keeps only
    /// keywords. They must be skipped, not reject the call.
    #[test]
    fn test_lfm2_tool_call_expression_positionals_dropped() {
        let input = "<|tool_call_start|>[f(x, len(a), *args, k=1), g(a+b, 'lit')]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, serde_json::json!({"k": 1}));
        assert_eq!(calls[1].name, "g");
        assert_eq!(calls[1].arguments, serde_json::json!({}));
    }

    /// `ast.parse` preserves duplicate keywords and vLLM's argument map
    /// keeps the final value.
    #[test]
    fn test_lfm2_tool_call_duplicate_kwarg_last_wins() {
        for (inner, expected) in [
            ("f(x=1, x=2)", serde_json::json!({"x": 2})),
            ("f(x=1, y=2, x=3)", serde_json::json!({"x": 3, "y": 2})),
            ("f(é=1, é=2)", serde_json::json!({"é": 2})),
        ] {
            let input = format!("<|tool_call_start|>[{inner}]<|tool_call_end|>");
            let (text, calls) = parse_tool_calls(&input);
            assert!(text.is_empty(), "{inner}");
            assert_eq!(calls.len(), 1, "{inner}");
            assert_eq!(calls[0].arguments, expected, "{inner}");
        }
    }

    /// An LFM2 sentinel inside a `<tool_call>` argument is literal text,
    /// not a real call — it must not be promoted.
    #[test]
    fn test_lfm2_sentinel_inside_tool_call_arg_not_promoted() {
        let input = "<tool_call>{\"name\":\"g\",\"arguments\":{\"a\":\"<|tool_call_start|>[f()]<|tool_call_end|>\"}}</tool_call>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "g");
        assert_eq!(
            calls[0].arguments["a"],
            "<|tool_call_start|>[f()]<|tool_call_end|>"
        );
        assert_eq!(text, "");
    }

    /// Numeric positionals (`f(5)`, `f(0x10)`) are dropped like any other
    /// positional — they must not kill the whole call.
    #[test]
    fn test_lfm2_tool_call_numeric_positional_dropped() {
        let input = "<|tool_call_start|>[f(5, x=1), g(0x10)]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments["x"], 1);
        assert_eq!(calls[0].arguments.as_object().unwrap().len(), 1);
        assert_eq!(calls[1].name, "g");
    }

    /// A trailing comma in the call list (`[f(),]`) is legal Python.
    #[test]
    fn test_lfm2_tool_call_trailing_comma() {
        let input = "<|tool_call_start|>[f(x=1),]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["x"], 1);
    }

    /// Unknown escapes keep the backslash (Python `'\d'` → `"\\d"`), while
    /// `\'`/`\"`/`\\` collapse to the literal char.
    #[test]
    fn test_lfm2_tool_call_unknown_escape_keeps_backslash() {
        let input = r"<|tool_call_start|>[f(pattern='\d+', q='it\'s', bs='a\\b')]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "\\d+");
        assert_eq!(calls[0].arguments["q"], "it's");
        assert_eq!(calls[0].arguments["bs"], "a\\b");
    }

    /// Dict keys stringify like `json.dumps` (`{True:1}` → `{"true":1}`).
    #[test]
    fn test_lfm2_tool_call_nonstring_dict_keys() {
        let input = "<|tool_call_start|>[f(m={True:1, None:2, 3:4})]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        let m = &calls[0].arguments["m"];
        assert_eq!(m["true"], 1);
        assert_eq!(m["null"], 2);
        assert_eq!(m["3"], 4);
    }

    /// `{{`/`}}` in an f-string are literal braces — a pure-constant
    /// f-string parses; a real `{placeholder}` still rejects the block.
    #[test]
    fn test_lfm2_tool_call_fstring_escaped_braces() {
        let input = r"<|tool_call_start|>[f(s=f'{{x}}')]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["s"], "{x}");

        let bad = r"<|tool_call_start|>[f(s=f'{x}')]<|tool_call_end|>";
        let (text, calls) = parse_tool_calls(bad);
        assert!(calls.is_empty());
        assert_eq!(text, bad);
    }

    /// Pathological nesting must fail closed (raw text back), never
    /// overflow the stack.
    #[test]
    fn test_lfm2_tool_call_depth_cap_fails_closed() {
        let deep = "[".repeat(200) + &"]".repeat(200);
        let input = format!("<|tool_call_start|>[f(x={deep})]<|tool_call_end|>");
        let (text, calls) = parse_tool_calls(&input);
        assert!(calls.is_empty());
        assert_eq!(text, input);
    }

    /// vLLM `escape_nested_quotes_in_strings`: `command='sed -n '1,9p'
    /// f.py'` recovers when exactly one closing quote parses.
    #[test]
    fn test_lfm2_tool_call_nested_quote_recovery() {
        let input = "<|tool_call_start|>[run(command='sed -n '1,9p' f.py')]<|tool_call_end|>";
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], "sed -n '1,9p' f.py");
    }

    /// A `</think>` inside a preserved LFM2 call's string arg must not be
    /// misread as a top-level missing-open terminator (would run a second
    /// scrub pass that drops the call).
    #[test]
    fn test_strip_reasoning_lfm2_call_with_inner_think_close() {
        let input = "R</think>\n<|tool_call_start|>[f(x='</think>\ny')]<|tool_call_end|>";
        let out = strip_reasoning_preserving_tools(input);
        let (_, calls) = parse_tool_calls(&out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["x"], "</think>\ny");
    }

    /// The synthesis defense applies to LFM2 sentinels too: a removal seam
    /// must not fuse sentinel fragments into a fabricated call.
    #[test]
    fn test_strip_reasoning_cannot_fuse_lfm2_sentinel_fragments() {
        // `<|tool_call_start` before reasoning + `|>[f()]<|tool_call_end|>`
        // after — deleting the reasoning would fuse a fake call without the
        // LFM2-aware rescan.
        let input = "<|tool_call_start<think>secret</think>|>[f()]<|tool_call_end|>";
        let out = strip_reasoning_preserving_tools(input);
        let (_, calls) = parse_tool_calls(&out);
        assert!(
            calls.is_empty(),
            "fused sentinel must not yield a call: {out}"
        );
    }
}
