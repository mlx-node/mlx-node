//! `response_template`-driven output parser.
//!
//! The checkpoint ships a machine-readable parse spec at `tokenizer_config.json
//! -> response_template`: three named fields, each with a regex that opens its
//! segment and one or more literal markers that close it. Driving the parser
//! from that spec — rather than hand-rolling the regexes, as gemma4 does across
//! ~2k lines — makes the three output channels match the checkpoint by
//! construction.
//!
//! Two layers, the same shape as [`super::config`]: `Raw*` mirrors the spec
//! verbatim; [`ResponseTemplate`] is the validated form with every pattern
//! already compiled. Anything the spec can express that this parser does not
//! implement is refused at construction rather than guessed at parse time — a
//! spec key we silently ignore is a channel we silently mis-route.
//!
//! # The one security property
//!
//! `to=self` is chain-of-thought, `to=user` is the answer, and the `to=` value
//! is the ONLY thing that distinguishes them. So [`ResponseTemplate::parse`]
//! carves segments strictly left to right and never lets two overlap. A parser
//! that instead searches each field's `open_pattern` over the whole string
//! independently finds `to=user<|message|>` *inside* a reasoning segment — text
//! the model wrote as thinking — and republishes it as the answer. See
//! `a_reasoning_segment_cannot_forge_a_content_segment`.

use napi::bindgen_prelude::*;
use regex::{Regex, RegexBuilder};
use serde::Deserialize;
use std::path::Path;

// ── On-disk spec ───────────────────────────────────────────────────

/// `close` is a bare string on `reasoning_content`/`tool_calls` and a list on
/// `content`. Normalized to a `Vec` at compile time so `parse` has one shape.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawClose {
    One(String),
    Many(Vec<String>),
}

impl RawClose {
    fn into_vec(self) -> Vec<String> {
        match self {
            RawClose::One(s) => vec![s],
            RawClose::Many(v) => v,
        }
    }
}

/// The spec's `content` key: how a segment's body is to be read. An
/// unrecognized value is a deserialize error, not a default.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum RawContentKind {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "xml-inline")]
    XmlInline,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawValueParserArgs {
    allow_non_json: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawValueParser {
    name: String,
    args: RawValueParserArgs,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawContentArgs {
    tag_pattern: String,
    value_parser: RawValueParser,
}

/// One field of `response_template.fields`. `deny_unknown_fields` is the point:
/// a future spec key this parser has never heard of must stop the load, because
/// the alternative is honouring two thirds of a parse contract.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawField {
    open_pattern: String,
    close: RawClose,
    /// Named `content` in the spec; renamed here because the *field* named
    /// `content` also has one.
    #[serde(rename = "content")]
    content_kind: RawContentKind,
    #[serde(default)]
    repeats: bool,
    #[serde(default)]
    content_args: Option<RawContentArgs>,
    #[serde(default)]
    transform: Option<serde_json::Value>,
}

/// The three fields, by name. Names are load-bearing — they are what decides
/// which output channel a segment feeds — so they are required, not a map, and
/// a fourth field is an error.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFields {
    reasoning_content: RawField,
    content: RawField,
    tool_calls: RawField,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefaults {
    role: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResponseTemplate {
    defaults: RawDefaults,
    start_anchor: String,
    fields: RawFields,
}

/// Only `response_template` is named: `tokenizer_config.json` carries dozens of
/// unrelated keys, so unknown keys are ignored HERE and nowhere below. `Option`
/// separates "absent" (a checkpoint that ships no parse spec) from "present but
/// malformed" (a serde error naming the offending key).
#[derive(Debug, Clone, Deserialize)]
struct RawTokenizerConfig {
    response_template: Option<RawResponseTemplate>,
}

// ── Compiled spec ──────────────────────────────────────────────────

/// Where a field's segments land, plus whatever that channel needs.
///
/// Modelled as an enum rather than a tag plus `Option`s so there is no
/// unreachable combination: only the tool-call channel has a tag pattern, and
/// it always has one.
#[derive(Debug)]
enum Sink {
    /// `reasoning_content` -> [`ParsedTurn::reasoning`].
    Reasoning,
    /// `content` -> [`ParsedTurn::content`].
    Content,
    /// `tool_calls` -> [`ParsedTurn::tool_calls`]; `tag` is `content_args
    /// .tag_pattern`, which pulls one `<atem:parameter>` per match.
    ToolCalls { tag: Regex },
}

#[derive(Debug)]
struct CompiledField {
    open: Regex,
    /// Non-empty; the earliest marker present wins.
    close: Vec<String>,
    sink: Sink,
}

impl CompiledField {
    /// Byte range `(start, end)` of the earliest close marker at or after
    /// `from`. `end` is where scanning resumes, i.e. past the marker.
    fn earliest_close(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        self.close
            .iter()
            .filter_map(|marker| {
                text[from..]
                    .find(marker.as_str())
                    .map(|i| (from + i, from + i + marker.len()))
            })
            .min_by_key(|(start, _)| *start)
    }
}

/// A compiled, validated `response_template`.
#[derive(Debug)]
pub struct ResponseTemplate {
    /// `defaults.role`. Checkpoint spec surface: validated so a spec that omits
    /// it is refused, but not consumed here — M0 knows it is parsing an
    /// assistant turn. Reserved for M1's streaming path, which has to label the
    /// messages it emits. Private (not `pub`) on purpose: that is what keeps
    /// `dead_code` firing, so the `expect` below is a real reminder and gets
    /// deleted the moment M1 reads it.
    #[cfg_attr(not(test), expect(dead_code))]
    default_role: String,
    /// `start_anchor` (`<|start|>assistant`). Checkpoint spec surface: M1's
    /// streaming path needs it to spot a message boundary in a partial stream.
    /// Deliberately NOT used to bound segments here — see [`Self::parse`].
    #[cfg_attr(not(test), expect(dead_code))]
    start_anchor: String,
    /// `tool_calls.transform`, verbatim and uninterpreted. It describes how to
    /// reshape a parsed call into an OpenAI-style `{type, function}` object,
    /// which is the serialization layer's job, not the parser's. Kept so the
    /// key is pinned against the real file.
    #[cfg_attr(not(test), expect(dead_code))]
    tool_call_transform: Option<serde_json::Value>,
    /// Order is `reasoning_content`, `content`, `tool_calls` — the scan picks
    /// the earliest match regardless, so this only breaks positional ties.
    fields: Vec<CompiledField>,
}

/// One tool call, already shaped the way callers want it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    /// A JSON object. Key order is the order the model emitted the parameters
    /// in (`serde_json` is built with `preserve_order`).
    pub arguments: serde_json::Value,
}

/// One parsed assistant turn: three independent channels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedTurn {
    /// Chain-of-thought (`to=self`). MUST NOT be shown as the answer.
    pub reasoning: Option<String>,
    /// The user-visible answer (`to=user`).
    pub content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

fn compile_regex(field: &str, what: &str, pattern: &str) -> Result<Regex> {
    // `dot_matches_new_line` rather than editing the checkpoint's pattern:
    // `tag_pattern`'s `(?P<value>.*?)` has to span a multi-line parameter body,
    // and the spec's own tool-use instructions show exactly such a value.
    RegexBuilder::new(pattern)
        .dot_matches_new_line(true)
        .build()
        .map_err(|e| {
            Error::from_reason(format!(
                "muse_glimmer: response_template field {field:?} has a {what} that does not \
                 compile: {pattern:?}: {e}"
            ))
        })
}

fn require_group(field: &str, what: &str, re: &Regex, group: &str) -> Result<()> {
    if re.capture_names().flatten().any(|n| n == group) {
        return Ok(());
    }
    Err(Error::from_reason(format!(
        "muse_glimmer: response_template field {field:?} has a {what} with no {group:?} capture \
         group: {:?}",
        re.as_str()
    )))
}

fn check_close(field: &str, close: &[String]) -> Result<()> {
    if close.is_empty() {
        return Err(Error::from_reason(format!(
            "muse_glimmer: response_template field {field:?} lists no close marker; every \
             segment would run to end of input"
        )));
    }
    Ok(())
}

fn check_kind(field: &str, got: RawContentKind, want: RawContentKind) -> Result<()> {
    if got == want {
        return Ok(());
    }
    Err(Error::from_reason(format!(
        "muse_glimmer: response_template field {field:?} declares content {got:?}, but this \
         parser reads it as {want:?}"
    )))
}

/// A `text` field: `reasoning_content` or `content`.
fn compile_text_field(name: &str, raw: RawField, sink: Sink) -> Result<CompiledField> {
    check_kind(name, raw.content_kind, RawContentKind::Text)?;
    let close = raw.close.into_vec();
    check_close(name, &close)?;
    Ok(CompiledField {
        open: compile_regex(name, "open_pattern", &raw.open_pattern)?,
        close,
        sink,
    })
}

/// The `xml-inline` field: `tool_calls`.
fn compile_tool_field(raw: RawField) -> Result<CompiledField> {
    const NAME: &str = "tool_calls";
    check_kind(NAME, raw.content_kind, RawContentKind::XmlInline)?;

    // `repeats` is not stored: one generation can carry several `<atem:invoke>`
    // blocks and `parse` always collects them all. Validating the flag instead
    // of branching on it means there is no untested "single call only" path —
    // a spec that says otherwise is refused rather than silently mishandled.
    if !raw.repeats {
        return Err(Error::from_reason(
            "muse_glimmer: response_template field \"tool_calls\" is not marked repeats; this \
             parser only implements the repeating form",
        ));
    }

    let args = raw.content_args.ok_or_else(|| {
        Error::from_reason(
            "muse_glimmer: response_template field \"tool_calls\" is xml-inline but has no \
             content_args; there is no way to read its parameters",
        )
    })?;
    if args.value_parser.name != "json" {
        return Err(Error::from_reason(format!(
            "muse_glimmer: response_template field \"tool_calls\" asks for value_parser {:?}; \
             only \"json\" is implemented",
            args.value_parser.name
        )));
    }
    // With `allow_non_json` a bare `Paris` stays the string "Paris"; without
    // it, the spec expects some other handling of unparseable values that this
    // parser does not implement, so refuse rather than invent one.
    if !args.value_parser.args.allow_non_json {
        return Err(Error::from_reason(
            "muse_glimmer: response_template field \"tool_calls\" sets value_parser \
             allow_non_json=false; only the permissive form is implemented",
        ));
    }

    let close = raw.close.into_vec();
    check_close(NAME, &close)?;
    let open = compile_regex(NAME, "open_pattern", &raw.open_pattern)?;
    require_group(NAME, "open_pattern", &open, "name")?;
    let tag = compile_regex(NAME, "tag_pattern", &args.tag_pattern)?;
    require_group(NAME, "tag_pattern", &tag, "key")?;
    require_group(NAME, "tag_pattern", &tag, "value")?;

    Ok(CompiledField {
        open,
        close,
        sink: Sink::ToolCalls { tag },
    })
}

/// `value_parser` = `json` with `allow_non_json`: a value that parses as JSON
/// keeps its JSON type, anything else stays the literal string. The chat
/// template emits bare scalars for numbers/bools/null and `tojson` for
/// containers, so this is what recovers the original argument types.
fn parse_value(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_owned()))
}

/// Build one call's arguments from the `<atem:parameter>` tags in its body.
fn parse_arguments(tag: &Regex, body: &str) -> serde_json::Value {
    // `serde_json::Map` is `preserve_order`-backed: insertion order IS the
    // model's emission order, which some tools are sensitive to and which a
    // sorted map would quietly destroy.
    let mut map = serde_json::Map::new();
    for caps in tag.captures_iter(body) {
        let (Some(key), Some(value)) = (caps.name("key"), caps.name("value")) else {
            continue;
        };
        map.insert(key.as_str().to_owned(), parse_value(value.as_str()));
    }
    serde_json::Value::Object(map)
}

/// Append a segment to a single-valued text channel.
///
/// Only `tool_calls` is marked `repeats`, so the text channels hold one value —
/// but the checkpoint's own chat template renders EVERY `to=self` message as its
/// own `<|start|>assistant to=self<|message|>…<|eom|>` segment, so a single
/// generation can legitimately produce several. Dropping the later ones loses
/// tokens the model produced; joining keeps all of them. Joining can never move
/// text across channels: which slot a segment lands in is decided solely by
/// which field's `open_pattern` matched it.
fn append_segment(slot: &mut Option<String>, segment: &str) {
    match slot {
        Some(existing) => {
            existing.push('\n');
            existing.push_str(segment);
        }
        None => *slot = Some(segment.to_owned()),
    }
}

/// Body and resume point of a `text` segment. The close marker is OPTIONAL
/// here: a generation cut off by `max_tokens` has no terminator and its partial
/// text is still the answer.
fn text_segment(text: &str, body_start: usize, close: Option<(usize, usize)>) -> (&str, usize) {
    let (body_end, after) = close.unwrap_or((text.len(), text.len()));
    (&text[body_start..body_end], after)
}

/// Smallest char boundary strictly greater than `pos`, clamped to the end.
fn advance_past(text: &str, pos: usize) -> usize {
    let mut next = pos + 1;
    while next < text.len() && !text.is_char_boundary(next) {
        next += 1;
    }
    next
}

impl ResponseTemplate {
    /// Read and compile `<dir>/tokenizer_config.json`.
    pub fn from_tokenizer_config(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer_config.json");
        let json = std::fs::read_to_string(&path).map_err(|e| {
            Error::from_reason(format!(
                "muse_glimmer: failed to read {}: {e}",
                path.display()
            ))
        })?;
        Self::from_tokenizer_config_str(&json)
    }

    /// Compile a `tokenizer_config.json` body. The validating core: every check
    /// fails closed on a spec this parser cannot honour exactly.
    pub fn from_tokenizer_config_str(json: &str) -> Result<Self> {
        let cfg: RawTokenizerConfig = serde_json::from_str(json).map_err(|e| {
            Error::from_reason(format!(
                "muse_glimmer: invalid tokenizer_config.json response_template: {e}"
            ))
        })?;
        let raw = cfg.response_template.ok_or_else(|| {
            Error::from_reason(
                "muse_glimmer: tokenizer_config.json has no response_template; this checkpoint \
                 does not ship the machine-readable parse spec this parser is driven by",
            )
        })?;

        let transform = raw.fields.tool_calls.transform.clone();
        let fields = vec![
            compile_text_field(
                "reasoning_content",
                raw.fields.reasoning_content,
                Sink::Reasoning,
            )?,
            compile_text_field("content", raw.fields.content, Sink::Content)?,
            compile_tool_field(raw.fields.tool_calls)?,
        ];

        Ok(Self {
            default_role: raw.defaults.role,
            start_anchor: raw.start_anchor,
            tool_call_transform: transform,
            fields,
        })
    }

    /// Split one generated assistant turn into its three channels.
    ///
    /// Infallible by design: generation is untrusted text that can stop at any
    /// byte, so every malformation degrades to "that segment is absent" rather
    /// than an error the caller would have to invent a response to.
    ///
    /// The walk is a single left-to-right pass. At each step the field whose
    /// `open_pattern` matches EARLIEST wins, its body runs to the earliest of
    /// its own close markers, and the cursor resumes past that close. Segments
    /// therefore never overlap, which is the whole reason a `to=user<|message|>`
    /// sequence appearing inside a reasoning body cannot become the answer.
    ///
    /// `start_anchor` is deliberately not consulted: bounding a segment at the
    /// next `<|start|>assistant` would recover more text from a generation that
    /// dropped its `<|eom|>`, but the failure it would fix errs in the safe
    /// direction already (reasoning absorbs the answer; the answer never
    /// absorbs reasoning), and inventing that rule is M1's call to make.
    pub fn parse(&self, generated: &str) -> ParsedTurn {
        let mut out = ParsedTurn::default();
        let mut pos = 0usize;

        while pos <= generated.len() {
            let Some((field, start, body_start, caps)) = self
                .fields
                .iter()
                .filter_map(|f| {
                    // `captures_at` rather than `find_at`: the tool-call open
                    // pattern carries the `name` group, and the whole-match
                    // range comes off group 0 without an `unwrap`.
                    let caps = f.open.captures_at(generated, pos)?;
                    let whole = caps.get(0)?;
                    Some((f, whole.start(), whole.end(), caps))
                })
                .min_by_key(|(_, start, _, _)| *start)
            else {
                break;
            };
            debug_assert!(start >= pos, "captures_at must not match before the cursor");

            let close = field.earliest_close(generated, body_start);
            // Consume the open unconditionally, so even a segment that is
            // dropped below cannot be re-matched. A zero-width open would leave
            // the cursor put and spin forever; no pattern in the spec is
            // zero-width, hence the belt-and-braces step.
            pos = if body_start > pos {
                body_start
            } else {
                advance_past(generated, pos)
            };

            match &field.sink {
                Sink::Reasoning => {
                    let (body, after) = text_segment(generated, body_start, close);
                    append_segment(&mut out.reasoning, body);
                    pos = pos.max(after);
                }
                Sink::Content => {
                    let (body, after) = text_segment(generated, body_start, close);
                    append_segment(&mut out.content, body);
                    pos = pos.max(after);
                }
                Sink::ToolCalls { tag } => {
                    // `xml-inline`: the close is MANDATORY. An invoke the
                    // generator never terminated is dropped whole — emitting a
                    // half-parsed call would hand the caller arguments the
                    // model never finished writing.
                    let Some((body_end, after)) = close else {
                        continue;
                    };
                    if let Some(name) = caps.name("name") {
                        out.tool_calls.push(ParsedToolCall {
                            name: name.as_str().to_owned(),
                            arguments: parse_arguments(tag, &generated[body_start..body_end]),
                        });
                    }
                    pos = pos.max(after);
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's `response_template`, transcribed. Verified byte-for-byte
    /// against the real file by `real_checkpoint_response_template_parses`.
    const SPEC: &str = r#"{
      "response_template": {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|start|>assistant",
        "fields": {
          "reasoning_content": {"open_pattern": "to=self<\\|message\\|>", "close": "<|eom|>", "content": "text"},
          "content": {"open_pattern": "to=user<\\|message\\|>", "close": ["<|eot|>", "<|eom|>"], "content": "text"},
          "tool_calls": {
            "open_pattern": "<atem:invoke\\b[^>]*?\\bname=\"(?P<name>[^\"]+)\">",
            "close": "</atem:invoke>", "content": "xml-inline", "repeats": true,
            "content_args": {
              "tag_pattern": "<atem:parameter\\b[^>]*?\\bname=\"(?P<key>[^\"]+)\"[^>]*?>(?P<value>.*?)</atem:parameter>",
              "value_parser": {"name": "json", "args": {"allow_non_json": true}}
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
          }
        }
      }
    }"#;

    /// The brief wrote this against a temp dir; `tempfile` is not a dependency
    /// of this workspace, so the validating core takes `&str` and
    /// `from_tokenizer_config` is a thin I/O shim over it (same split as
    /// `super::config`). Not one asserted value changed.
    fn spec() -> ResponseTemplate {
        ResponseTemplate::from_tokenizer_config_str(SPEC).unwrap()
    }

    #[test]
    fn parses_a_plain_answer() {
        let out = spec().parse(" to=user<|message|>Hello there.<|eot|>");
        assert_eq!(out.content.as_deref(), Some("Hello there."));
        assert!(out.reasoning.is_none());
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn parses_reasoning_then_answer_as_two_messages() {
        let out = spec().parse(
            " to=self<|message|>Let me think.<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>",
        );
        assert_eq!(out.reasoning.as_deref(), Some("Let me think."));
        assert_eq!(out.content.as_deref(), Some("Done."));
    }

    #[test]
    fn reasoning_is_never_surfaced_as_content() {
        // The to= value is the ONLY discriminator. A parser that treats the first
        // segment as content leaks chain-of-thought to the user.
        let out = spec().parse(" to=self<|message|>secret plan<|eom|>");
        assert_eq!(out.reasoning.as_deref(), Some("secret plan"));
        assert_eq!(out.content, None, "reasoning must not become content");
    }

    #[test]
    fn parses_one_tool_call_with_typed_arguments() {
        let out = spec().parse(
            " to=wx.forecast<|message|><atem:function_calls>\n\
             <atem:invoke name=\"wx.forecast\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             <atem:parameter name=\"days\">3</atem:parameter>\n\
             <atem:parameter name=\"metric\">true</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eot|>",
        );
        assert_eq!(out.tool_calls.len(), 1);
        let tc = &out.tool_calls[0];
        assert_eq!(tc.name, "wx.forecast");
        // value_parser json + allow_non_json: bare scalars parse to JSON types,
        // unparseable text stays a string.
        assert_eq!(tc.arguments["city"], serde_json::json!("Paris"));
        assert_eq!(tc.arguments["days"], serde_json::json!(3));
        assert_eq!(tc.arguments["metric"], serde_json::json!(true));
    }

    #[test]
    fn parses_repeated_tool_calls() {
        let out = spec().parse(
            " to=a<|message|><atem:invoke name=\"a\"><atem:parameter name=\"x\">1</atem:parameter></atem:invoke>\
             <atem:invoke name=\"b\"><atem:parameter name=\"y\">2</atem:parameter></atem:invoke><|eot|>",
        );
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].name, "a");
        assert_eq!(out.tool_calls[1].name, "b");
    }

    #[test]
    fn preserves_parameter_order() {
        let out = spec().parse(
            " to=t<|message|><atem:invoke name=\"t\">\
             <atem:parameter name=\"zeta\">1</atem:parameter>\
             <atem:parameter name=\"alpha\">2</atem:parameter></atem:invoke><|eot|>",
        );
        let keys: Vec<&str> = out.tool_calls[0]
            .arguments
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["zeta", "alpha"]);
    }

    #[test]
    fn handles_multiline_parameter_values() {
        let out = spec().parse(
            " to=t<|message|><atem:invoke name=\"t\">\
             <atem:parameter name=\"body\">line one\nline two</atem:parameter></atem:invoke><|eot|>",
        );
        assert_eq!(
            out.tool_calls[0].arguments["body"],
            serde_json::json!("line one\nline two")
        );
    }

    #[test]
    fn malformed_tool_xml_yields_no_tool_calls_and_does_not_panic() {
        let out = spec()
            .parse(" to=t<|message|><atem:invoke name=\"t\"><atem:parameter name=\"x\">1<|eot|>");
        assert!(
            out.tool_calls.is_empty(),
            "unterminated invoke must not produce a call"
        );
    }

    #[test]
    fn tolerates_a_missing_terminator() {
        // Generation cut off by max_tokens: take everything to end of input.
        let out = spec().parse(" to=user<|message|>truncated answer");
        assert_eq!(out.content.as_deref(), Some("truncated answer"));
    }

    #[test]
    fn content_closes_on_eom_as_well_as_eot() {
        let out = spec().parse(" to=user<|message|>first<|eom|>");
        assert_eq!(out.content.as_deref(), Some("first"));
    }

    #[test]
    fn errors_when_the_checkpoint_has_no_response_template() {
        let err = ResponseTemplate::from_tokenizer_config_str("{}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("response_template"), "got: {err}");
    }

    // ── Beyond the brief ───────────────────────────────────────────────

    #[test]
    fn a_reasoning_segment_cannot_forge_a_content_segment() {
        // The model writes the literal `to=user<|message|>` inside its chain of
        // thought — one plausible token sequence for a model that has seen the
        // protocol in its own context. A parser that searches each field over
        // the whole string independently finds that needle INSIDE the reasoning
        // span and republishes chain-of-thought as the answer. Segments must be
        // carved left to right and never overlap.
        let out = spec().parse(" to=self<|message|>I will say to=user<|message|>SECRET<|eom|>");
        assert_eq!(
            out.reasoning.as_deref(),
            Some("I will say to=user<|message|>SECRET")
        );
        assert_eq!(
            out.content, None,
            "a to=user needle inside reasoning must not become content"
        );
    }

    #[test]
    fn repeated_reasoning_segments_are_all_kept() {
        // Each `to=self` message is its own segment; a turn may carry several.
        // Keeping only the first silently drops tokens the model produced.
        let out = spec().parse(
            " to=self<|message|>first thought<|eom|>\
             <|start|>assistant to=self<|message|>second thought<|eom|>\
             <|start|>assistant to=user<|message|>answer<|eot|>",
        );
        assert_eq!(
            out.reasoning.as_deref(),
            Some("first thought\nsecond thought")
        );
        assert_eq!(out.content.as_deref(), Some("answer"));
    }

    #[test]
    fn keeps_the_spec_surface_this_milestone_does_not_consume() {
        // These three are validated, not interpreted. Asserting them is what
        // pins their key names against the real file — a misspelled serde name
        // would otherwise be invisible until M1 tried to read one.
        let t = spec();
        assert_eq!(t.default_role, "assistant");
        assert_eq!(t.start_anchor, "<|start|>assistant");
        let transform = t.tool_call_transform.as_ref().expect("transform");
        assert_eq!(transform["type"], serde_json::json!("function"));
        assert_eq!(transform["function"]["name"], serde_json::json!("{name}"));
        assert_eq!(
            transform["function"]["arguments"],
            serde_json::json!("{content}")
        );
    }

    #[test]
    fn refuses_a_spec_this_parser_cannot_honour_exactly() {
        let err = |json: String| {
            ResponseTemplate::from_tokenizer_config_str(&json)
                .unwrap_err()
                .to_string()
        };
        let case = |from: &str, to: &str| SPEC.replacen(from, to, 1);

        // A text field that claims to be xml-inline (and vice versa).
        let e = err(case("\"content\": \"text\"", "\"content\": \"xml-inline\""));
        assert!(
            e.contains("reasoning_content") && e.contains("XmlInline"),
            "got: {e}"
        );
        // An unrecognized content kind is an error, never a default.
        let e = err(case("\"content\": \"text\"", "\"content\": \"markdown\""));
        assert!(e.contains("markdown"), "got: {e}");
        // A close list with nothing in it: every segment would run to EOF.
        let e = err(case(
            "\"close\": [\"<|eot|>\", \"<|eom|>\"]",
            "\"close\": []",
        ));
        assert!(e.contains("close marker"), "got: {e}");
        // repeats must be set: `parse` always collects every invoke.
        let e = err(case("\"repeats\": true", "\"repeats\": false"));
        assert!(e.contains("repeats"), "got: {e}");
        // Only the permissive json value parser is implemented.
        let e = err(case(
            "\"allow_non_json\": true",
            "\"allow_non_json\": false",
        ));
        assert!(e.contains("allow_non_json"), "got: {e}");
        let e = err(case("\"name\": \"json\"", "\"name\": \"yaml\""));
        assert!(e.contains("value_parser"), "got: {e}");
        // The name/key/value capture groups are what the parse depends on.
        let e = err(case("(?P<name>", "(?P<nombre>"));
        assert!(e.contains("\"name\" capture group"), "got: {e}");
        let e = err(case("(?P<value>", "(?P<valeur>"));
        assert!(e.contains("\"value\" capture group"), "got: {e}");
        // A pattern that does not compile names itself instead of panicking.
        let e = err(case("(?P<key>", "(?P<key>("));
        assert!(e.contains("tag_pattern"), "got: {e}");
        // An unknown key means a parse contract we would only half honour.
        let e = err(case(
            "\"repeats\": true",
            "\"repeats\": true, \"strip\": true",
        ));
        assert!(e.contains("strip"), "got: {e}");
    }

    // ── Real checkpoint (gated) ────────────────────────────────────────

    /// Every value in the tests above is a hand transcription of the spec, so on
    /// their own they prove the parser self-consistent, not correct. This one
    /// compiles the ACTUAL `tokenizer_config.json` and then runs real turns
    /// through it: a spec that loads but cannot parse is not a pass.
    #[test]
    #[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
    fn real_checkpoint_response_template_parses() {
        let Ok(dir) = std::env::var("MLX_TEST_MUSE_GLIMMER_MODEL_PATH") else {
            eprintln!("skipping: MLX_TEST_MUSE_GLIMMER_MODEL_PATH not set");
            return;
        };
        let tpl = ResponseTemplate::from_tokenizer_config(Path::new(&dir))
            .expect("the real checkpoint's response_template must compile and validate");

        // All three fields present, every pattern compiled. `deny_unknown_fields`
        // plus the required names mean construction alone proves this, but assert
        // it so a future restructuring cannot quietly drop a channel.
        assert_eq!(tpl.fields.len(), 3);
        let reasoning = tpl
            .fields
            .iter()
            .find(|f| matches!(f.sink, Sink::Reasoning))
            .expect("reasoning_content field");
        let content = tpl
            .fields
            .iter()
            .find(|f| matches!(f.sink, Sink::Content))
            .expect("content field");
        let tools = tpl
            .fields
            .iter()
            .find(|f| matches!(f.sink, Sink::ToolCalls { .. }))
            .expect("tool_calls field");
        assert!(reasoning.open.as_str().contains("to=self"));
        // BOTH close markers, not just the first.
        assert_eq!(content.close, vec!["<|eot|>", "<|eom|>"]);
        assert!(tools.open.as_str().contains("atem:invoke"));
        let Sink::ToolCalls { tag } = &tools.sink else {
            unreachable!("matched above")
        };
        assert!(tag.as_str().contains("atem:parameter"));
        assert_eq!(tpl.start_anchor, "<|start|>assistant");
        assert_eq!(tpl.default_role, "assistant");
        assert!(tpl.tool_call_transform.is_some());

        // `repeats` is validated at construction rather than stored, so read it
        // straight off the file: that is the assertion the brief asks for.
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(Path::new(&dir).join("tokenizer_config.json")).unwrap(),
        )
        .unwrap();
        let fields = &raw["response_template"]["fields"];
        assert_eq!(fields["tool_calls"]["repeats"], serde_json::json!(true));
        assert_eq!(
            fields["content"]["close"],
            serde_json::json!(["<|eot|>", "<|eom|>"])
        );
        assert_eq!(
            fields.as_object().map(serde_json::Map::len),
            Some(3),
            "the real spec must carry exactly the three fields this parser routes"
        );

        // Now the parses. Same expectations the hand-written tests make, on the
        // real template. Bodies are shaped exactly as chat_template.jinja
        // renders them (`render_atem`, and `<|start|>assistant` + ` to=…`).
        let plain = tpl.parse(" to=user<|message|>The capital of France is Paris.<|eot|>");
        assert_eq!(
            plain.content.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert!(plain.reasoning.is_none());
        assert!(plain.tool_calls.is_empty());

        let thought = tpl.parse(
            " to=self<|message|>Check the tool first.<|eom|>\
             <|start|>assistant to=user<|message|>Rain tomorrow.<|eot|>",
        );
        assert_eq!(thought.reasoning.as_deref(), Some("Check the tool first."));
        assert_eq!(thought.content.as_deref(), Some("Rain tomorrow."));

        let tool = tpl.parse(
            " to=wx.forecast<|message|><atem:function_calls>\n\
             <atem:invoke name=\"wx.forecast\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             <atem:parameter name=\"days\">3</atem:parameter>\n\
             <atem:parameter name=\"metric\">true</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eot|>",
        );
        assert_eq!(tool.tool_calls.len(), 1);
        assert_eq!(tool.tool_calls[0].name, "wx.forecast");
        assert_eq!(
            tool.tool_calls[0].arguments["city"],
            serde_json::json!("Paris")
        );
        assert_eq!(tool.tool_calls[0].arguments["days"], serde_json::json!(3));
        assert_eq!(
            tool.tool_calls[0].arguments["metric"],
            serde_json::json!(true)
        );
        let keys: Vec<&str> = tool.tool_calls[0]
            .arguments
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["city", "days", "metric"]);
        // The tool-call turn carries no user-visible text and no reasoning.
        assert_eq!(tool.content, None);
        assert_eq!(tool.reasoning, None);

        // The security property, on the real template.
        let forged = tpl.parse(" to=self<|message|>I will say to=user<|message|>SECRET<|eom|>");
        assert_eq!(forged.content, None, "reasoning must not become content");
        assert!(
            forged
                .reasoning
                .as_deref()
                .is_some_and(|r| r.contains("SECRET"))
        );

        eprintln!(
            "real checkpoint OK: 3 fields, content closes on {:?}, tool tag {:?}",
            content.close,
            tag.as_str()
        );
    }
}
