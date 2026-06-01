//! Tool call parsing utilities
//!
//! Extracts structured tool calls from model-generated text.
//! Supports JSON format (Qwen3), function/parameter format (Qwen3.5), and XML format (legacy).
//!
//! Uses simple string-based parsers instead of regex for clarity and debuggability.

use napi_derive::napi;
use serde_json::Value;
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
    /// Raw content from <tool_call> tag (preserved for debugging/persistence)
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
            error: Some(format!("Tool call missing name: {}", &raw_content)),
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
// Public API
// ---------------------------------------------------------------------------

/// Parse tool calls from generated text
///
/// Returns (cleaned_text, tool_calls) where:
/// - `cleaned_text` has all `<tool_call>...</tool_call>` tags removed
/// - `tool_calls` contains all parsed tool calls with status info
///
/// Supports three formats:
/// - JSON (Qwen3): `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Function (Qwen3.5): `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
/// - XML (legacy): `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let blocks = extract_tag_blocks(text, "<tool_call>", "</tool_call>");

    let mut tool_calls = Vec::new();
    for (start, end, inner) in &blocks {
        let raw_content = &text[*start..*end];
        if let Some(result) = classify_and_parse_tool_call(inner, raw_content) {
            tool_calls.push(result);
        }
    }

    let cleaned_text = strip_tag_blocks(text, "<tool_call>", "</tool_call>");
    (cleaned_text, tool_calls)
}

/// Check if text contains any tool call tags
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>")
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
/// while preserving `<tool_call>…</tool_call>` spans that are NOT themselves part of a
/// reasoning block.
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
    let tool_ranges: Vec<(usize, usize)> = extract_tag_blocks(text, "<tool_call>", "</tool_call>")
        .into_iter()
        .map(|(s, e, _)| (s, e))
        .collect();
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
    let tool_ranges: Vec<(usize, usize)> = extract_tag_blocks(text, "<tool_call>", "</tool_call>")
        .into_iter()
        .map(|(s, e, _)| (s, e))
        .collect();
    matches!(applied_missing_open(text, &tool_ranges), Some((_, _, true)))
}

/// Keep only the `<tool_call>…</tool_call>` spans in `out` whose byte range is one of `genuine`
/// (the PRESERVED tool spans mapped to output coordinates). Every other tool span is a
/// removal-seam artifact — fragments fused into a `<tool_call>` span that never existed as a
/// preserved call — and `parse_tool_calls` would treat it as executable, so it is dropped.
///
/// Provenance is by RANGE, not bytes: a fabricated span can be byte-identical to a genuine call
/// (e.g. a duplicate), so a multiset of strings cannot disambiguate them (an earlier fabricated
/// look-alike would consume the genuine call's budget and the real one would be dropped) — but
/// the output ranges can. Dropped one span at a time, re-extracting and shifting `genuine` after
/// each drop so a fusion newly formed at a drop seam is itself caught. Each drop strictly shrinks
/// `out`, so it terminates.
fn keep_only_genuine_tool_spans(mut out: String, mut genuine: Vec<(usize, usize)>) -> String {
    loop {
        let spans = extract_tag_blocks(&out, "<tool_call>", "</tool_call>");
        let Some(&(s, e, _)) = spans.iter().find(|(s, e, _)| !genuine.contains(&(*s, *e))) else {
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
    /// Clean text with <tool_call> and <think> tags removed
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
/// both `<tool_call>` and `<think>` tags removed.
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
}
