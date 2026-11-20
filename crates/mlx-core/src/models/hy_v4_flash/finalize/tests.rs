//! Completions are id sequences: eos and the Hy4 tags are the vocabulary's
//! own ids (120025, 120029-120043), the text between them one plain piece
//! each. Expected values come from the checkpoint's chat template renders
//! (`tokenizer/fixtures/hf/hy-v4-flash-09a8ba8fb355`).

use std::collections::HashMap;

use serde_json::{Value, json};

use super::{Hy4Tags, finalize};
use crate::engine::backend::FinalizeArgs;
use crate::engine::penalties::ReasoningOutcome;
use crate::engine::types::{ChatConfig, ChatResult};
use crate::tokenizer::{FunctionDefinition, FunctionParameters, Qwen3Tokenizer, ToolDefinition};
use crate::tools::{ToolArgSchemas, ToolCallResult};

const FINAL_RENDER: &str = include_str!(
    "../../../tokenizer/fixtures/hf/hy-v4-flash-09a8ba8fb355/tools_call_result_final_no_agp.txt"
);
const FINAL_CASE: &str = include_str!(
    "../../../tokenizer/fixtures/hf/hy-v4-flash-09a8ba8fb355/tools_call_result_final_no_agp.json"
);

const S: &str = ":6124c78e";

fn open(name: &str) -> String {
    format!("<{name}{S}>")
}

fn close(name: &str) -> String {
    format!("</{name}{S}>")
}

fn eos() -> String {
    format!("<｜hy_end{S}｜>")
}

/// `(id, token, special)`: GGUF flags, the think / tool-call / argument tags
/// are USER_DEFINED (not special).
fn markers() -> Vec<(u32, String, bool)> {
    let mut m = vec![(120025, eos(), true)];
    for (name, id, special) in [
        ("think", 120029, false),
        ("tool_call", 120033, false),
        ("tool_calls", 120035, false),
        ("tool_response", 120037, true),
        ("arg_key", 120040, false),
        ("arg_value", 120042, false),
    ] {
        m.push((id, open(name), special));
        m.push((id + 1, close(name), special));
    }
    m
}

/// `text` cut before and after every marker.
fn pieces(text: &str) -> Vec<String> {
    let markers = markers();
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let next = markers
            .iter()
            .filter_map(|(_, m, _)| rest.find(m.as_str()).map(|i| (i, m.len())))
            .min();
        let cut = match next {
            Some((0, len)) => len,
            Some((i, _)) => i,
            None => rest.len(),
        };
        out.push(rest[..cut].to_string());
        rest = &rest[cut..];
    }
    out
}

/// A WordLevel tokenizer whose pieces decode verbatim (`Fuse`), with the
/// markers at their checkpoint ids.
fn tokenizer(pieces: &[String]) -> Qwen3Tokenizer {
    let markers = markers();
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".into(), json!(0));
    for p in pieces {
        if !vocab.contains_key(p) && !markers.iter().any(|(_, m, _)| m == p) {
            let id = vocab.len();
            vocab.insert(p.clone(), json!(id));
        }
    }
    for (id, m, _) in &markers {
        vocab.insert(m.clone(), json!(id));
    }
    let added: Vec<Value> = markers
        .iter()
        .map(|(id, m, special)| {
            json!({"id": id, "content": m, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": special})
        })
        .collect();
    let spec = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": null, "post_processor": null,
        "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&spec).unwrap()).unwrap();
    Qwen3Tokenizer::from_tokenizer(tok, None).unwrap()
}

fn ids(tok: &Qwen3Tokenizer, pieces: &[String]) -> Vec<u32> {
    pieces
        .iter()
        .map(|p| tok.token_to_id(p.clone()).unwrap())
        .collect()
}

fn case_of(pieces: &[String]) -> (Qwen3Tokenizer, Vec<u32>) {
    let tok = tokenizer(pieces);
    let ids = ids(&tok, pieces);
    (tok, ids)
}

fn case(completion: &str) -> (Qwen3Tokenizer, Vec<u32>) {
    case_of(&pieces(completion))
}

fn tags(tok: &Qwen3Tokenizer) -> Hy4Tags {
    let tags = Hy4Tags::from_tokenizer(tok).unwrap();
    assert_eq!(tok.think_end_id(), Some(tags.think_close_id()));
    assert_eq!(tok.think_end_str(), Some(tags.think_close()));
    tags
}

fn parse(
    tok: &Qwen3Tokenizer,
    ids: &[u32],
    thinking: bool,
) -> (String, Vec<ToolCallResult>, Option<String>) {
    let tags = tags(tok);
    tags.parse_turn(tok, ids, thinking, Some(tags.think_close_id()), true)
}

fn run(completion: &str, thinking: bool) -> (String, Vec<ToolCallResult>, Option<String>) {
    let (tok, ids) = case(completion);
    parse(&tok, &ids, thinking)
}

fn finalize_turn(
    tok: &Qwen3Tokenizer,
    ids: &[u32],
    finish_reason: &str,
    include_reasoning: bool,
) -> ChatResult {
    let tags = tags(tok);
    finalize(
        FinalizeArgs {
            tokenizer: tok,
            generated_tokens: ids,
            finish_reason: finish_reason.to_string(),
            think_end_id: Some(tags.think_close_id()),
            think_end_str: Some(tags.think_close()),
            think_end_extra_ids: &[],
            performance: None,
            include_reasoning,
            thinking_enabled: true,
            prompt_tokens: 9,
            reasoning: ReasoningOutcome::counted(1),
        },
        &tags,
    )
    .unwrap()
}

/// `<tool_call:X>NAME(<arg_key:X>k</arg_key:X><arg_value:X>v</arg_value:X>)*</tool_call:X>`.
fn call(name: &str, args: &[(&str, &str)]) -> String {
    let mut s = format!("{}{name}", open("tool_call"));
    for (k, v) in args {
        s += &format!(
            "{}{k}{}{}{v}{}",
            open("arg_key"),
            close("arg_key"),
            open("arg_value"),
            close("arg_value")
        );
    }
    s + &close("tool_call")
}

fn block(calls: &[String]) -> String {
    format!(
        "{}{}{}",
        open("tool_calls"),
        calls.concat(),
        close("tool_calls")
    )
}

fn schemas(tools: &[Value]) -> std::sync::Arc<ToolArgSchemas> {
    let tools = tools
        .iter()
        .map(|t| {
            let f = &t["function"];
            ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: f["name"].as_str().unwrap().to_string(),
                    description: None,
                    parameters: Some(FunctionParameters {
                        r#type: "object".to_string(),
                        properties: Some(f["parameters"]["properties"].to_string()),
                        required: None,
                    }),
                },
            }
        })
        .collect();
    ToolArgSchemas::from_config(&ChatConfig {
        tools: Some(tools),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn tags_are_read_by_id_with_the_vocabulary_suffix() {
    let (tok, _) = case("");
    let tags = tags(&tok);
    assert_eq!(tags.think_close_id(), 120030);
    assert_eq!(tags.think_close(), close("think"));

    let hf: HashMap<u32, String> = markers()
        .into_iter()
        .map(|(id, m, _)| (id, m.replace(S, ":opensource")))
        .collect();
    let tags = Hy4Tags::from_vocab(|id| hf.get(&id).cloned()).unwrap();
    assert_eq!(tags.think_close(), "</think:opensource>");

    let mut mixed = hf.clone();
    mixed.insert(120041, "</arg_key:6124c78e>".to_string());
    let err = Hy4Tags::from_vocab(|id| mixed.get(&id).cloned()).unwrap_err();
    assert!(err.reason.contains("120041"), "{}", err.reason);

    let mut missing = hf;
    missing.remove(&120029);
    assert!(Hy4Tags::from_vocab(|id| missing.get(&id).cloned()).is_err());
}

#[test]
fn thinking_then_content() {
    let c = format!(
        "Let me think.\n{}\n\nThe answer is 4.{}",
        close("think"),
        eos()
    );
    let (text, calls, thinking) = run(&c, true);
    assert_eq!(thinking.as_deref(), Some("Let me think."));
    assert_eq!(text, "The answer is 4.");
    assert!(calls.is_empty());

    let (tok, ids) = case(&c);
    let tags = tags(&tok);
    let hidden = tags.parse_turn(&tok, &ids, true, Some(120030), false);
    assert_eq!((hidden.0.as_str(), hidden.2), ("The answer is 4.", None));

    let reopened = format!("{}plan{}ok", open("think"), close("think"));
    let (text, _, thinking) = run(&reopened, true);
    assert_eq!((text.as_str(), thinking.as_deref()), ("ok", Some("plan")));
}

#[test]
fn no_close_makes_the_turn_reasoning() {
    let (text, calls, thinking) = run("still thinking ", true);
    assert_eq!(
        (text.as_str(), calls.len(), thinking.as_deref()),
        ("", 0, Some("still thinking"))
    );
}

#[test]
fn no_think_and_the_empty_pair() {
    let pair = format!("{}{}Hi there.{}", open("think"), close("think"), eos());
    for c in [pair.as_str(), "Hi there."] {
        let (text, calls, thinking) = run(c, false);
        assert_eq!((text.as_str(), thinking), ("Hi there.", None), "{c:?}");
        assert!(calls.is_empty());
    }
    let empty = format!("{}\n\nHi", close("think"));
    let (text, _, thinking) = run(&empty, true);
    assert_eq!((text.as_str(), thinking), ("Hi", None));
}

#[test]
fn one_call() {
    let tc = call("get_weather", &[("city", "北京"), ("days", "3")]);
    let c = format!(
        "r{}Checking.{}{}",
        close("think"),
        block(std::slice::from_ref(&tc)),
        eos()
    );
    let (text, calls, thinking) = run(&c, true);
    assert_eq!(
        (text.as_str(), thinking.as_deref()),
        ("Checking.", Some("r"))
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(
        (calls[0].name.as_str(), calls[0].status.as_str()),
        ("get_weather", "ok")
    );
    assert_eq!(calls[0].arguments, json!({"city": "北京", "days": "3"}));
    assert_eq!(calls[0].raw_content, tc);
    let raw = calls[0].raw_argument_text.as_ref().unwrap();
    assert_eq!(raw.get("days").map(String::as_str), Some("3"));

    let (tok, ids) = case(&c);
    let result = finalize_turn(&tok, &ids, "stop", true);
    assert_eq!(result.finish_reason, "tool_calls");
    assert_eq!(result.text, "Checking.");
    assert_eq!(result.thinking.as_deref(), Some("r"));
    assert_eq!(result.tool_calls.len(), 1);

    let hidden = finalize_turn(&tok, &ids, "stop", false);
    assert_eq!(hidden.thinking, None);
    assert_eq!(hidden.raw_text, format!("Checking.{}", block(&[tc])));
    assert_eq!(
        hidden.public_raw_text.as_deref(),
        Some(hidden.raw_text.as_str())
    );
}

#[test]
fn two_calls_with_whitespace_between_tags() {
    let c = format!(
        "{}\n{}\n  {}\n{}\n{}",
        close("think"),
        open("tool_calls"),
        call("ping", &[]),
        call("echo", &[("s", " a <b> \"c\"\nd ")]),
        close("tool_calls")
    );
    let (text, calls, _) = run(&c, true);
    assert_eq!(text, "");
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0].name.as_str(), &calls[0].arguments),
        ("ping", &json!({}))
    );
    assert_eq!(calls[1].name, "echo");
    assert_eq!(calls[1].arguments, json!({"s": " a <b> \"c\"\nd "}));
    assert!(calls.iter().all(|c| c.status == "ok"));
}

#[test]
fn json_values_and_schema_coercion() {
    let c = format!(
        "{}{}",
        close("think"),
        block(&[call(
            "f",
            &[
                ("o", r#"{"lang": ["en", "zh"]}"#),
                ("a", "[1, 2]"),
                ("bad", "{bad"),
                ("q", "\"quoted\""),
                ("n", "42"),
                ("b", "true"),
            ],
        )])
    );
    let (_, mut calls, _) = run(&c, true);
    assert_eq!(
        calls[0].arguments,
        json!({"o": {"lang": ["en", "zh"]}, "a": [1, 2], "bad": "{bad", "q": "\"quoted\"",
               "n": "42", "b": "true"})
    );
    let tools = [
        json!({"function": {"name": "f", "parameters": {"properties": {
            "n": {"type": "integer"}, "b": {"type": "boolean"}, "q": {"type": "string"},
            "a": {"type": "string"}
        }}}}),
    ];
    schemas(&tools).coerce_calls(&mut calls);
    assert_eq!(
        calls[0].arguments,
        json!({"o": {"lang": ["en", "zh"]}, "a": "[1, 2]", "bad": "{bad", "q": "\"quoted\"",
               "n": 42, "b": true})
    );
}

#[test]
fn a_truncated_call_stays_text() {
    let content = format!(
        "Sure.{}{}f{}a{}{}par",
        open("tool_calls"),
        open("tool_call"),
        open("arg_key"),
        close("arg_key"),
        open("arg_value")
    );
    let c = format!("r{}{content}", close("think"));
    let (text, calls, _) = run(&c, true);
    assert_eq!(text, content);
    assert!(calls.is_empty());

    let open_block = format!("{}{}{}", close("think"), open("tool_calls"), call("f", &[]));
    let (text, calls, _) = run(&open_block, true);
    assert_eq!(text, format!("{}{}", open("tool_calls"), call("f", &[])));
    assert!(calls.is_empty());

    let (tok, ids) = case(&c);
    let result = finalize_turn(&tok, &ids, "length", true);
    assert_eq!(result.finish_reason, "length");
    assert_eq!(result.text, content);
}

#[test]
fn malformed_markup_in_a_closed_block() {
    let junk = block(&[format!("junk{}", call("f", &[]))]);
    let (text, calls, _) = run(&format!("{}A.{junk}", close("think")), true);
    assert_eq!(text, "A.");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        (calls[0].name.as_str(), calls[0].status.as_str()),
        ("", "parse_error")
    );
    assert_eq!(calls[0].raw_content, junk);

    let c = format!(
        "{}{}after",
        close("think"),
        block(&[
            call("", &[("a", "1")]),
            call("dup", &[("a", "1"), ("a", "2")]),
            call("good", &[("a", "1")]),
        ])
    );
    let (text, calls, _) = run(&c, true);
    assert_eq!(text, "");
    let status: Vec<(&str, &str)> = calls
        .iter()
        .map(|c| (c.name.as_str(), c.status.as_str()))
        .collect();
    assert_eq!(
        status,
        [("", "missing_name"), ("dup", "parse_error"), ("good", "ok")]
    );
}

#[test]
fn a_tool_tag_split_across_tokens_still_parses() {
    let p = |s: &str| s.to_string();
    let parts = vec![
        p("r"),
        close("think"),
        open("tool_calls"),
        open("tool_call"),
        p("f"),
        open("arg_key"),
        p("a"),
        close("arg_key"),
        p("<arg_value"),
        format!("{S}>"),
        p("1"),
        close("arg_value"),
        close("tool_call"),
        p("</tool_calls"),
        format!("{S}>"),
    ];
    let (tok, ids) = case_of(&parts);
    assert!(!ids.contains(&120042) && !ids.contains(&120036));
    let (text, calls, thinking) = parse(&tok, &ids, true);
    assert_eq!((text.as_str(), thinking.as_deref()), ("", Some("r")));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, json!({"a": "1"}));

    // the reasoning ends only at the emitted close id
    let spelled = vec![p("r"), p("</think"), format!("{S}>"), p("x")];
    let (tok, ids) = case_of(&spelled);
    let (text, _, thinking) = parse(&tok, &ids, true);
    assert_eq!(text, "");
    assert_eq!(thinking, Some(format!("r</think{S}>x")));
}

/// Each assistant turn of the template render, generated back as ids, gives
/// the message it was rendered from.
#[test]
fn template_renders_round_trip() {
    let case_json: Value = serde_json::from_str(FINAL_CASE).unwrap();
    let schemas = schemas(case_json["tools"].as_array().unwrap());
    let messages: Vec<&Value> = case_json["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect();
    let head = format!(
        "<｜hy_start{S}｜>assistant<｜hy_middle{S}｜>{}",
        open("think")
    );
    let turns: Vec<&str> = FINAL_RENDER
        .split(head.as_str())
        .skip(1)
        .map(|t| &t[..t.find(eos().as_str()).unwrap() + eos().len()])
        .collect();
    assert_eq!(turns.len(), messages.len());
    assert_eq!(turns.len(), 2);
    for (turn, message) in turns.iter().zip(messages) {
        let (text, mut calls, thinking) = run(turn, true);
        schemas.coerce_calls(&mut calls);
        assert_eq!(
            thinking.as_deref(),
            message["reasoning_content"].as_str(),
            "{turn}"
        );
        assert_eq!(text, message["content"].as_str().unwrap(), "{turn}");
        let want: Vec<(&str, &Value)> = message["tool_calls"]
            .as_array()
            .map(|c| {
                c.iter()
                    .map(|c| {
                        (
                            c["function"]["name"].as_str().unwrap(),
                            &c["function"]["arguments"],
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let got: Vec<(&str, &Value)> = calls
            .iter()
            .map(|c| (c.name.as_str(), &c.arguments))
            .collect();
        assert_eq!(got, want, "{turn}");
        assert!(calls.iter().all(|c| c.status == "ok"));
    }
}
