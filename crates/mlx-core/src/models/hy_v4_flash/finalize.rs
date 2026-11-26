//! Hy4-flash turn finalizer: the reasoning splits at the emitted think-close
//! id, then the `<tool_calls:X>` block is parsed from the decoded content.
//!
//! The grammar is the checkpoint's chat template (assistant branch):
//! `{content}<tool_calls:X><tool_call:X>NAME<arg_key:X>k</arg_key:X><arg_value:X>v</arg_value:X>…</tool_call:X>…</tool_calls:X>`,
//! a string value written raw and any other value as `tojson`. The leniency
//! follows vLLM's `hy_v4_tool_parser.py` (Apache-2.0): whitespace between
//! tags is skipped, the name is trimmed, text after the block is dropped.
//! Values stay text (an object or array literal parses as JSON) and the
//! request's tool schema types them at the NAPI boundary (`tools::coerce`).
//!
//! The think, tool-call and argument tags are not special tokens, so they
//! decode as text. Their `:X` suffix differs between the GGUF (`:6124c78e`)
//! and HF (`:opensource`) vocabularies, so every tag is read by id.

use std::collections::HashMap;

use napi::{Error, Result};
use serde_json::{Map, Value};

use crate::engine::backend::FinalizeArgs;
use crate::engine::finalize::finalize_chat_result_with;
use crate::engine::types::ChatResult;
use crate::tokenizer::Qwen3Tokenizer;
use crate::tools::ToolCallResult;

const THINK_IDS: (u32, u32) = (120029, 120030);
const TOOL_CALL_IDS: (u32, u32) = (120033, 120034);
const TOOL_CALLS_IDS: (u32, u32) = (120035, 120036);
const ARG_KEY_IDS: (u32, u32) = (120040, 120041);
const ARG_VALUE_IDS: (u32, u32) = (120042, 120043);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Pair {
    open: String,
    close: String,
}

/// The Hy4 reasoning and tool-call tags of one vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4Tags {
    think: Pair,
    tool_calls: Pair,
    tool_call: Pair,
    arg_key: Pair,
    arg_value: Pair,
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 tags: {msg}"))
}

fn is_ws(c: char) -> bool {
    c.is_ascii_whitespace()
}

fn decode(tok: &Qwen3Tokenizer, ids: &[u32]) -> String {
    tok.decode_sync(ids, true).unwrap_or_else(|e| {
        tracing::warn!("hy_v4_flash: decode generated ids: {e}");
        String::new()
    })
}

impl Hy4Tags {
    pub fn from_tokenizer(tok: &Qwen3Tokenizer) -> Result<Self> {
        Self::from_vocab(|id| tok.id_to_token(id))
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        let pair = |name: &str| Pair {
            open: format!("<{name}:test>"),
            close: format!("</{name}:test>"),
        };
        Self {
            think: pair("think"),
            tool_calls: pair("tool_calls"),
            tool_call: pair("tool_call"),
            arg_key: pair("arg_key"),
            arg_value: pair("arg_value"),
        }
    }

    /// Every tag must be `<name:X>` / `</name:X>` at its id, with the suffix
    /// `:X` of `<think:X>` (no `<` or `>` in it).
    fn from_vocab(token: impl Fn(u32) -> Option<String>) -> Result<Self> {
        let think_open = token(THINK_IDS.0)
            .ok_or_else(|| bad(format!("no token {} in the vocabulary", THINK_IDS.0)))?;
        let suffix = think_open
            .strip_prefix("<think")
            .and_then(|rest| rest.strip_suffix('>'))
            .filter(|suffix| !suffix.contains(['<', '>']))
            .ok_or_else(|| bad(format!("token {} is {think_open:?}", THINK_IDS.0)))?;
        let pair = |name: &str, (open_id, close_id): (u32, u32)| -> Result<Pair> {
            let pair = Pair {
                open: format!("<{name}{suffix}>"),
                close: format!("</{name}{suffix}>"),
            };
            for (id, want) in [(open_id, &pair.open), (close_id, &pair.close)] {
                let got = token(id);
                if got.as_ref() != Some(want) {
                    return Err(bad(format!("token {id} is {got:?}, expected {want:?}")));
                }
            }
            Ok(pair)
        };
        Ok(Self {
            think: pair("think", THINK_IDS)?,
            tool_calls: pair("tool_calls", TOOL_CALLS_IDS)?,
            tool_call: pair("tool_call", TOOL_CALL_IDS)?,
            arg_key: pair("arg_key", ARG_KEY_IDS)?,
            arg_value: pair("arg_value", ARG_VALUE_IDS)?,
        })
    }

    /// The id that ends a generated reasoning.
    pub fn think_close_id(&self) -> u32 {
        THINK_IDS.1
    }

    /// `</think:X>`.
    pub fn think_close(&self) -> &str {
        &self.think.close
    }

    /// `(text, tool_calls, thinking)` of one generated turn. With thinking on,
    /// the ids before the first `think_end_id` are the reasoning; no close
    /// before the end makes the whole turn reasoning.
    pub fn parse_turn(
        &self,
        tok: &Qwen3Tokenizer,
        ids: &[u32],
        thinking_enabled: bool,
        think_end_id: Option<u32>,
        include_reasoning: bool,
    ) -> (String, Vec<ToolCallResult>, Option<String>) {
        let close = think_end_id.and_then(|t| ids.iter().position(|&x| x == t));
        let (reasoning, content) = match close {
            Some(k) if thinking_enabled => (Some(&ids[..k]), &ids[k + 1..]),
            None if thinking_enabled && think_end_id.is_some() => (Some(ids), &ids[..0]),
            _ => (None, ids),
        };
        let thinking = reasoning
            .map(|r| {
                let text = decode(tok, r);
                self.strip_think_open(&text)
                    .trim_end_matches(is_ws)
                    .to_string()
            })
            .filter(|t| !t.is_empty())
            .filter(|_| include_reasoning);
        if reasoning.is_some() && close.is_none() {
            return (String::new(), Vec::new(), thinking);
        }
        let content = decode(tok, content);
        let content = if reasoning.is_some() {
            content.trim_start_matches(is_ws)
        } else {
            self.strip_empty_think(&content)
        };
        let (text, calls) = self.parse_content(content);
        (text, calls, thinking)
    }

    /// The generated text after the reasoning close (`raw_text` without the
    /// reasoning), cut at the close id; `text` is the whole turn decoded.
    pub fn public_raw(
        &self,
        tok: &Qwen3Tokenizer,
        ids: &[u32],
        thinking_enabled: bool,
        think_end_id: Option<u32>,
        text: &str,
    ) -> String {
        if !thinking_enabled {
            return text.to_string();
        }
        match think_end_id.and_then(|t| ids.iter().position(|&x| x == t)) {
            Some(k) => decode(tok, &ids[k + 1..]),
            None if think_end_id.is_some() => String::new(),
            None => text.to_string(),
        }
    }

    /// A reasoning without a re-emitted `<think:X>` (the prompt opens it).
    fn strip_think_open<'a>(&self, text: &'a str) -> &'a str {
        text.trim_start_matches(is_ws)
            .strip_prefix(&self.think.open)
            .unwrap_or(text)
    }

    /// Content without a leading empty `<think:X></think:X>` (the `no_think`
    /// prompt's own pair, when the model writes it again).
    fn strip_empty_think<'a>(&self, text: &'a str) -> &'a str {
        text.trim_start_matches(is_ws)
            .strip_prefix(&self.think.open)
            .and_then(|r| r.trim_start_matches(is_ws).strip_prefix(&self.think.close))
            .map_or(text, |r| r.trim_start_matches(is_ws))
    }

    /// Content before the first `<tool_calls:X>` and the block's calls. A
    /// block without its close (length stop) stays in the text; malformed
    /// markup inside a closed block is one `parse_error` call.
    fn parse_content(&self, text: &str) -> (String, Vec<ToolCallResult>) {
        let open = &self.tool_calls.open;
        let Some(at) = text.find(open.as_str()) else {
            return (text.to_string(), Vec::new());
        };
        let pre = text[..at].trim_end_matches(is_ws).to_string();
        let body = &text[at + open.len()..];
        match self.parse_calls(body) {
            Ok(calls) => (pre, calls),
            Err(e) => match body.find(self.tool_calls.close.as_str()) {
                Some(len) => {
                    let raw =
                        text[at..at + open.len() + len + self.tool_calls.close.len()].to_string();
                    let call = ToolCallResult::parse_error(
                        String::new(),
                        raw.clone(),
                        format!("Hy4 tool calls: {e}"),
                        raw,
                    );
                    (pre, vec![call])
                }
                None => (text.to_string(), Vec::new()),
            },
        }
    }

    fn parse_calls(&self, body: &str) -> std::result::Result<Vec<ToolCallResult>, String> {
        let (call_open, call_close) = (self.tool_call.open.as_str(), self.tool_call.close.as_str());
        let (key_open, key_close) = (self.arg_key.open.as_str(), self.arg_key.close.as_str());
        let (value_open, value_close) =
            (self.arg_value.open.as_str(), self.arg_value.close.as_str());
        let mut calls = Vec::new();
        let mut rest = body.trim_start_matches(is_ws);
        loop {
            if rest.starts_with(self.tool_calls.close.as_str()) {
                return Ok(calls);
            }
            let start = rest;
            let r = rest.strip_prefix(call_open).ok_or_else(|| {
                format!(
                    "expected a tool call or the block close at {:?}",
                    head(rest)
                )
            })?;
            let name_end = [r.find(key_open), r.find(call_close)]
                .into_iter()
                .flatten()
                .min()
                .ok_or("unterminated tool call")?;
            let name = r[..name_end].trim();
            let mut r = &r[name_end..];
            let mut args = Map::new();
            let mut raw_args = HashMap::new();
            let mut fail: Option<String> = None;
            loop {
                r = r.trim_start_matches(is_ws);
                if let Some(after) = r.strip_prefix(call_close) {
                    r = after;
                    break;
                }
                let k = r
                    .strip_prefix(key_open)
                    .ok_or_else(|| format!("expected an argument at {:?}", head(r)))?;
                let (key, k) = k
                    .split_once(key_close)
                    .ok_or("argument key is not closed")?;
                let v = k
                    .trim_start_matches(is_ws)
                    .strip_prefix(value_open)
                    .ok_or_else(|| format!("argument {key:?} has no value"))?;
                let (value, v) = v
                    .split_once(value_close)
                    .ok_or_else(|| format!("argument {key:?}: value is not closed"))?;
                r = v;
                let key = key.trim();
                if key.is_empty() {
                    fail.get_or_insert_with(|| "empty argument key".to_string());
                    continue;
                }
                if args.contains_key(key) {
                    fail.get_or_insert_with(|| format!("duplicate argument {key:?}"));
                    continue;
                }
                args.insert(key.to_string(), arg_value(value));
                raw_args.insert(key.to_string(), value.to_string());
            }
            let raw = start[..start.len() - r.len()].to_string();
            calls.push(match fail {
                _ if name.is_empty() => ToolCallResult::missing_name(raw),
                Some(e) => ToolCallResult::parse_error(name.to_string(), raw.clone(), e, raw),
                None => ToolCallResult::ok(name.to_string(), Value::Object(args), raw)
                    .with_raw_argument_text(raw_args),
            });
            rest = r.trim_start_matches(is_ws);
        }
    }
}

/// An object or array literal as JSON; anything else as the text written.
fn arg_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.starts_with(['{', '['])
        && let Ok(v @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str(trimmed)
    {
        return v;
    }
    Value::String(raw.to_string())
}

fn head(s: &str) -> &str {
    let end = s.char_indices().nth(40).map_or(s.len(), |(i, _)| i);
    &s[..end]
}

pub(crate) fn finalize(args: FinalizeArgs<'_>, tags: &Hy4Tags) -> Result<ChatResult> {
    finalize_chat_result_with(
        args,
        |_, a| {
            tags.parse_turn(
                a.tokenizer,
                a.generated_tokens,
                a.thinking_enabled,
                a.think_end_id,
                a.include_reasoning,
            )
        },
        |text, a| {
            tags.public_raw(
                a.tokenizer,
                a.generated_tokens,
                a.thinking_enabled,
                a.think_end_id,
                text,
            )
        },
    )
}

#[cfg(test)]
mod tests;
