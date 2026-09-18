//! K2-Horizon finalize: the [`crate::engine::finalize::finalize_chat_result`]
//! pipeline against K2's own markup — `<ifm|tool_call>` blocks and the three
//! effort-selected `<ifm|think*>` reasoning tag families.
//!
//! K2's chat template emits tool calls inside a `<ifm|tool_calls>` wrapper as
//! `<ifm|tool_call>` blocks in one of three inner formats (`tool_call_format`):
//!   * `xml` (default) — `NAME\n<ifm|arg_key>K</ifm|arg_key>\n<ifm|arg_value>V</ifm|arg_value>…`
//!   * `xml_typed`    — same plus `<ifm|arg_type>T</ifm|arg_type>` per argument
//!   * `json`         — `{"name": …, "arguments": {…}}`
//!
//! The scrub/split/parse machinery in [`crate::tools`] is tag-parameterized
//! ([`tools::MarkupSpec`]); this module supplies K2's spec, its inner-block
//! classifier, and the `finalize_turn` entry the `ChatBackend` impl calls.

use napi::bindgen_prelude::*;
use serde_json::Value;

use crate::engine::backend::FinalizeArgs;
use crate::engine::finalize::finalize_chat_result_with;
use crate::engine::types::ChatResult;
use crate::tools::{self, MarkupSpec, ToolCallResult};

use super::model::K2_REASONING_PAIRS;

/// K2's markup spec: `<ifm|tool_call>` delimiters + the three reasoning
/// families. NOTE: `<ifm|tool_calls>` (the plural wrapper) never matches the
/// singular open tag — `starts_with`/`find` require the `>` terminator.
const K2_MARKUP: MarkupSpec = MarkupSpec {
    reasoning: K2_REASONING_PAIRS,
    tool_open: "<ifm|tool_call>",
    tool_close: "</ifm|tool_call>",
};

/// K2 reasoning open tags (for the strip-open-prefix arms that mirror the
/// ChatML `<think>`/`<longcat_think>` prefix handling).
const K2_OPEN_PREFIXES: &[&str] = &["<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"];

/// Coerce one `xml`/`xml_typed` argument value.
///
/// The template instructs "string and scalar parameters as plain text;
/// array and object parameters as JSON literals". With an `arg_type`
/// (`xml_typed`), non-string scalar types coerce through JSON
/// (`number`/`integer`/`boolean`/`object`/`array`/`null`). Without one
/// (plain `xml`), only values that parse as a JSON object or array are
/// treated as structured — everything else stays a string, so `"3"` or
/// `"true"` written as plain text never silently becomes a number.
fn coerce_arg_value(value: &str, arg_type: Option<&str>) -> Value {
    let trimmed = value.trim();
    match arg_type.map(str::trim) {
        Some("object" | "array" | "number" | "integer" | "boolean" | "null") => {
            serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(value.to_string()))
        }
        Some(_) => Value::String(value.to_string()),
        None => match serde_json::from_str::<Value>(trimmed) {
            Ok(v @ (Value::Object(_) | Value::Array(_))) => v,
            _ => Value::String(value.to_string()),
        },
    }
}

/// Classify one `<ifm|tool_call>` inner block.
///
/// `json` format → `{"name":…, "arguments":{…}}` through the shared JSON
/// parser. `xml`/`xml_typed` → first non-empty line is the function name,
/// then `arg_key`/`arg_value` (and `arg_type`) tag pairs zip into the
/// arguments object. Unrecognized shapes return `None` so the span still
/// gets stripped from the clean text but surfaces no executable call.
fn classify_k2_tool_call(inner: &str, raw_content: &str) -> Option<ToolCallResult> {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('{') {
        return Some(tools::parse_json_tool_call(trimmed, raw_content));
    }

    // xml / xml_typed: name = first non-empty line that is not itself a
    // tag (a leading `<ifm|arg_key>` means the name line is missing).
    let name = trimmed
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if name.is_empty() || name.starts_with('<') {
        return Some(ToolCallResult::missing_name(raw_content.to_string()));
    }

    let keys = tools::extract_tag_blocks(inner, "<ifm|arg_key>", "</ifm|arg_key>");
    let values = tools::extract_tag_blocks(inner, "<ifm|arg_value>", "</ifm|arg_value>");
    let types = tools::extract_tag_blocks(inner, "<ifm|arg_type>", "</ifm|arg_type>");

    if keys.is_empty() {
        // Bare `<ifm|tool_call>name</ifm|tool_call>` — a call with no args.
        return Some(ToolCallResult::ok(
            name.to_string(),
            Value::Object(serde_json::Map::new()),
            raw_content.to_string(),
        ));
    }
    if keys.len() != values.len() {
        return Some(ToolCallResult::parse_error(
            name.to_string(),
            trimmed.to_string(),
            format!(
                "K2 tool call arg_key/arg_value count mismatch ({} keys, {} values)",
                keys.len(),
                values.len()
            ),
            raw_content.to_string(),
        ));
    }

    let mut args = serde_json::Map::new();
    for (i, (_, _, key)) in keys.iter().enumerate() {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let arg_type = types.get(i).map(|(_, _, t)| *t);
        args.insert(key.to_string(), coerce_arg_value(values[i].2, arg_type));
    }
    Some(ToolCallResult::ok(
        name.to_string(),
        Value::Object(args),
        raw_content.to_string(),
    ))
}

/// `parse_tool_calls` against K2's `<ifm|tool_call>` markup. K2 wraps the
/// call list in `<ifm|tool_calls>…</ifm|tool_calls>` — the shared parser
/// strips the inner blocks only, so the wrapper literals come off here
/// (an empty wrapper left behind would otherwise leak into clean text).
pub(crate) fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let (cleaned, calls) = tools::parse_tool_calls_with(text, K2_MARKUP, classify_k2_tool_call);
    let cleaned = cleaned
        .replace("<ifm|tool_calls>", "")
        .replace("</ifm|tool_calls>", "");
    (cleaned.trim().to_string(), calls)
}

/// K2 port of `tools::parse_thinking`: paired `<ifm|think*>` blocks first
/// (concatenated with newlines), then the missing-open fallback — the
/// template injects the opener into the prompt, so generated text starts
/// inside reasoning and emits a bare close.
///
/// Unlike the ChatML `</think>` fallback, NO next-char disambiguation:
/// `</ifm|think*>` is a dedicated added token the model emits only as
/// markup — and K2 routinely appends content directly after it
/// (`...</ifm|think>391`), where a newline-or-end requirement would
/// leave the answer stranded inside `reasoning_content`.
fn parse_k2_thinking(text: &str) -> (String, Option<String>) {
    for (open, close) in K2_REASONING_PAIRS {
        let blocks = tools::extract_tag_blocks(text, open, close);
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
            let cleaned_text = tools::strip_tag_blocks(text, open, close);
            return (cleaned_text, thinking);
        }
    }

    // Missing-open fallback: first close of ANY K2 family, regardless of
    // what follows it. Earliest close position wins across families (the
    // generation lives inside exactly one open block).
    let mut best: Option<(usize, &str)> = None;
    for close_tag in K2_REASONING_PAIRS.iter().map(|(_, c)| *c) {
        if let Some(pos) = text.find(close_tag)
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, close_tag));
        }
    }
    if let Some((close_pos, close_tag)) = best {
        let after_tag = &text[close_pos + close_tag.len()..];
        let thinking_content = text[..close_pos].trim();
        let thinking = if thinking_content.is_empty() {
            None
        } else {
            Some(thinking_content.to_string())
        };
        return (after_tag.trim().to_string(), thinking);
    }
    (text.to_string(), None)
}

/// K2 port of `tools::parse_generation_output` — the text-level fallback
/// when no think-end token is available.
fn parse_k2_generation_output(text: &str) -> (String, Vec<ToolCallResult>, Option<String>) {
    let (text_without_tools, tool_calls) = parse_tool_calls(text);
    let (cleaned_text, thinking) = parse_k2_thinking(&text_without_tools);
    (cleaned_text, tool_calls, thinking)
}

/// The close tag the generation actually emitted: earliest text
/// occurrence among K2's three `</ifm|think*>` members. The caller has
/// already confirmed a close TOKEN in `generated_tokens`; the tags are
/// dedicated added tokens, so any occurrence in the decoded text is that
/// token. `think_end_str` (the armed tag) wins ties.
fn k2_emitted_close_tag<'a>(text: &str, think_end_str: Option<&'a str>) -> Option<&'a str> {
    let mut best: Option<(usize, &'a str)> =
        think_end_str.and_then(|tag| text.find(tag).map(|pos| (pos, tag)));
    for close_tag in K2_REASONING_PAIRS.iter().map(|(_, c)| *c) {
        if let Some(pos) = text.find(close_tag)
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, close_tag));
        }
    }
    best.map(|(_, tag)| tag)
}

/// Any armed-or-alternate close token present in the generation —
/// the K2-aware form of `tools::has_think_end_token`.
fn has_k2_think_end(generated_tokens: &[u32], ids: &[u32]) -> bool {
    ids.iter().any(|id| generated_tokens.contains(id))
}

/// K2 port of `engine::finalize::parse_thinking_and_tools` — same four-way
/// branch, K2 tags.
fn parse_thinking_and_tools(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    think_end_extra_ids: &[u32],
    include_reasoning: bool,
) -> (String, Vec<ToolCallResult>, Option<String>) {
    let all_end_ids: Vec<u32> = think_end_id
        .into_iter()
        .chain(think_end_extra_ids.iter().copied())
        .collect();
    let (clean_text, tool_calls, thinking) = if !thinking_enabled {
        let (clean, calls) = parse_tool_calls(text);
        (clean, calls, None)
    } else if has_k2_think_end(generated_tokens, &all_end_ids) {
        // Token-level split at whichever family member was emitted.
        let emitted = k2_emitted_close_tag(text, think_end_str);
        tools::split_at_think_end_with(
            text,
            emitted.or(think_end_str),
            K2_OPEN_PREFIXES,
            parse_tool_calls,
            parse_k2_generation_output,
        )
    } else if think_end_id.is_some() {
        // Truncated generation: entire output is reasoning.
        let thinking_text = text.trim();
        let thinking_text = K2_OPEN_PREFIXES
            .iter()
            .find_map(|p| thinking_text.strip_prefix(p))
            .unwrap_or(thinking_text)
            .trim();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        (String::new(), vec![], thinking)
    } else {
        // No think_end_id in vocab: text-level scrub preserving tool spans.
        let content = tools::strip_reasoning_preserving_tools_with(text, K2_MARKUP);
        let (clean, calls) = parse_tool_calls(&content);
        let (text_without_tools, _) = parse_tool_calls(text);
        let thinking = parse_k2_thinking(&text_without_tools).1;
        (clean.trim().to_string(), calls, thinking)
    };

    let thinking = if include_reasoning { thinking } else { None };
    (clean_text, tool_calls, thinking)
}

/// K2 port of `engine::finalize::raw_text_with_reasoning_suppressed`.
fn raw_text_with_reasoning_suppressed(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    think_end_extra_ids: &[u32],
    include_reasoning: bool,
) -> String {
    if include_reasoning || !thinking_enabled {
        return text.to_string();
    }
    let all_end_ids: Vec<u32> = think_end_id
        .into_iter()
        .chain(think_end_extra_ids.iter().copied())
        .collect();
    if has_k2_think_end(generated_tokens, &all_end_ids) {
        if let Some(tag) = k2_emitted_close_tag(text, think_end_str)
            && let Some(close_pos) = text.find(tag)
        {
            return text[close_pos + tag.len()..].to_string();
        }
    } else if think_end_id.is_some() {
        return String::new();
    }
    tools::strip_reasoning_preserving_tools_with(text, K2_MARKUP)
}

/// K2 `finalize_turn` — `finalize_chat_result` against K2 markup.
///
/// Decode skips special tokens (`<|ifm|im_end|>` never surfaces); the
/// `<ifm|*>` reasoning/tool tags are NON-special added tokens, so they
/// decode as literal text and this function sees the full markup — exact
/// parity with ChatML's non-special `</think>` handling.
pub(crate) fn finalize_k2_chat_result(args: FinalizeArgs<'_>) -> Result<ChatResult> {
    finalize_chat_result_with(
        args,
        |text, args| {
            let (clean_text, tool_calls, thinking) = parse_thinking_and_tools(
                text,
                args.generated_tokens,
                args.thinking_enabled,
                args.think_end_id,
                args.think_end_str,
                args.think_end_extra_ids,
                args.include_reasoning,
            );
            // K2's template RAISES on any assistant history message without a
            // thinking field (`reasoning_content`/`think`/…), so a suppressed or
            // absent reasoning body must still round-trip as a DEFINED field —
            // `Some("")` renders `<ifm|think>\n</ifm|think>` on replay where
            // `None` would make the next continuation's render raise.
            (clean_text, tool_calls, Some(thinking.unwrap_or_default()))
        },
        |text, args| {
            raw_text_with_reasoning_suppressed(
                text,
                args.generated_tokens,
                args.thinking_enabled,
                args.think_end_id,
                args.think_end_str,
                args.think_end_extra_ids,
                false,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finalizer_tokenizer() -> crate::tokenizer::Qwen3Tokenizer {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mlx-k2-finalize-tokenizer-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "secret": 0,
                    "<unk>": 1,
                    "answer": 2,
                    "</ifm|think>": 3,
                    "</ifm|think_fast>": 4,
                    "</ifm|think_faster>": 5
                },
                "unk_token": "<unk>"
            }
        });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let tokenizer = crate::tokenizer::Qwen3Tokenizer::from_file(&path).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        tokenizer
    }

    #[test]
    fn finalize_k2_chat_result_preserves_close_families_and_hidden_replay() {
        let tokenizer = finalizer_tokenizer();
        let closes = [
            (3, "</ifm|think>"),
            (4, "</ifm|think_fast>"),
            (5, "</ifm|think_faster>"),
        ];
        for (close_id, close_tag) in closes {
            let result = finalize_k2_chat_result(FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &[0, close_id, 2],
                finish_reason: "length".to_string(),
                think_end_id: Some(close_id),
                think_end_str: Some(close_tag),
                think_end_extra_ids: &[],
                performance: None,
                include_reasoning: true,
                thinking_enabled: true,
                prompt_tokens: 7,
                reasoning_tokens: 1,
            })
            .unwrap();
            assert_eq!(result.text, "answer");
            assert_eq!(result.thinking.as_deref(), Some("secret"));
            assert_eq!(result.num_tokens, 3);
            assert_eq!(result.prompt_tokens, 7);
            assert_eq!(result.reasoning_tokens, 1);
            assert_eq!(result.cached_tokens, 0);
        }

        let alternate = finalize_k2_chat_result(FinalizeArgs {
            tokenizer: &tokenizer,
            generated_tokens: &[0, 4, 2],
            finish_reason: "length".to_string(),
            think_end_id: Some(3),
            think_end_str: Some("</ifm|think>"),
            think_end_extra_ids: &[4, 5],
            performance: None,
            include_reasoning: true,
            thinking_enabled: true,
            prompt_tokens: 3,
            reasoning_tokens: 1,
        })
        .unwrap();
        assert_eq!(alternate.text, "answer");
        assert_eq!(alternate.thinking.as_deref(), Some("secret"));

        let hidden = finalize_k2_chat_result(FinalizeArgs {
            tokenizer: &tokenizer,
            generated_tokens: &[0, 3, 2],
            finish_reason: "length".to_string(),
            think_end_id: Some(3),
            think_end_str: Some("</ifm|think>"),
            think_end_extra_ids: &[4, 5],
            performance: None,
            include_reasoning: false,
            thinking_enabled: true,
            prompt_tokens: 3,
            reasoning_tokens: 1,
        })
        .unwrap();
        assert_eq!(hidden.thinking.as_deref(), Some(""));
        assert_eq!(hidden.raw_text.trim(), "answer");
        assert_eq!(
            hidden.public_raw_text.as_deref().map(str::trim),
            Some("answer")
        );
        assert!(!hidden.raw_text.contains("secret"));

        let truncated = finalize_k2_chat_result(FinalizeArgs {
            tokenizer: &tokenizer,
            generated_tokens: &[0],
            finish_reason: "length".to_string(),
            think_end_id: Some(3),
            think_end_str: Some("</ifm|think>"),
            think_end_extra_ids: &[4, 5],
            performance: None,
            include_reasoning: false,
            thinking_enabled: true,
            prompt_tokens: 3,
            reasoning_tokens: 1,
        })
        .unwrap();
        assert_eq!(truncated.thinking.as_deref(), Some(""));
        assert_eq!(truncated.raw_text, "");
        assert_eq!(truncated.public_raw_text.as_deref(), Some(""));
    }

    /// The template's default `xml` format: name line, then
    /// arg_key/arg_value pairs. Scalars stay strings (the `xml` format
    /// carries no types), JSON objects/arrays parse structured.
    #[test]
    fn test_parse_xml_tool_call() {
        let text = "<ifm|tool_calls>\n<ifm|tool_call>get_weather\n\
                    <ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n\
                    <ifm|arg_key>opts</ifm|arg_key>\n<ifm|arg_value>{\"units\": \"c\"}</ifm|arg_value>\n\
                    </ifm|tool_call>\n</ifm|tool_calls>";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["city"], Value::from("Paris"));
        assert_eq!(calls[0].arguments["opts"]["units"], Value::from("c"));
    }

    /// `xml_typed` coerces non-string scalars through `arg_type`.
    #[test]
    fn test_parse_xml_typed_tool_call() {
        let text = "<ifm|tool_call>add\n\
                    <ifm|arg_key>a</ifm|arg_key><ifm|arg_type>number</ifm|arg_type>\n\
                    <ifm|arg_value>3</ifm|arg_value>\n\
                    <ifm|arg_key>b</ifm|arg_key><ifm|arg_type>number</ifm|arg_type>\n\
                    <ifm|arg_value>4.5</ifm|arg_value>\n\
                    </ifm|tool_call>";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["a"], Value::from(3));
        assert_eq!(calls[0].arguments["b"], Value::from(4.5));
    }

    /// `json` inner format goes through the shared JSON classifier.
    #[test]
    fn test_parse_json_tool_call() {
        let text = "<ifm|tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}</ifm|tool_call>";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["x"], Value::from(1));
    }

    /// Bare name with no args parses as an empty-arguments call.
    #[test]
    fn test_parse_bare_tool_call() {
        let (_clean, calls) = parse_tool_calls("<ifm|tool_call>ping</ifm|tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].status, "ok");
        assert!(calls[0].arguments.is_object());
    }

    /// Mismatched arg_key/arg_value counts surface a parse_error, never
    /// a silently truncated argument list.
    #[test]
    fn test_parse_tool_call_arg_mismatch() {
        let text = "<ifm|tool_call>f\n<ifm|arg_key>a</ifm|arg_key>\n</ifm|tool_call>";
        let (_clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "parse_error");
    }

    /// Each of the three reasoning families round-trips: open+close
    /// strips cleanly and yields the thinking body.
    #[test]
    fn test_parse_k2_thinking_all_families() {
        for (open, close) in K2_REASONING_PAIRS {
            let text = format!("{open}\nplan carefully{close}\nThe answer.");
            let (clean, thinking) = parse_k2_thinking(&text);
            assert_eq!(clean, "The answer.");
            assert_eq!(thinking.as_deref(), Some("plan carefully"));
        }
    }

    /// The template injects the open tag into the prompt, so generated
    /// text starts INSIDE reasoning: a bare close splits it — including
    /// content glued directly after the tag (K2 emits `</ifm|think>391`
    /// with no separator).
    #[test]
    fn test_parse_k2_thinking_missing_open() {
        let (clean, thinking) = parse_k2_thinking("step one\n</ifm|think>\nfinal");
        assert_eq!(clean, "final");
        assert_eq!(thinking.as_deref(), Some("step one"));

        let (clean, thinking) = parse_k2_thinking("work</ifm|think>391");
        assert_eq!(clean, "391");
        assert_eq!(thinking.as_deref(), Some("work"));
    }

    /// Reasoning must never swallow a tool call that follows it.
    #[test]
    fn test_tool_call_after_thinking() {
        let text = "hmm</ifm|think>\n<ifm|tool_call>f</ifm|tool_call>";
        let (_t, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }
}
