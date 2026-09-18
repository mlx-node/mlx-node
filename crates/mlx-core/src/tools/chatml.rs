use serde_json::Value;

use super::{ToolCallResult, extract_tag_blocks};

// ---------------------------------------------------------------------------
// JSON sanitizer (for LLM-generated JSON with raw control characters)
// ---------------------------------------------------------------------------

/// Sanitize JSON string by escaping raw control characters inside string values.
///
/// LLMs often generate JSON with raw newlines inside strings for readability.
/// This function escapes control characters (`\u0000-\u001F`) found inside
/// quoted string values so that standard JSON parsers can handle them.
pub(super) fn sanitize_json_string(input: &str) -> String {
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
pub(crate) fn parse_json_tool_call(json_str: &str, raw_content: &str) -> ToolCallResult {
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
pub(super) fn classify_and_parse_tool_call(
    inner: &str,
    raw_content: &str,
) -> Option<ToolCallResult> {
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
