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
//! the model wrote as thinking — and republishes it as the answer.
//!
//! Non-overlap alone is not enough, because a segment whose own close marker
//! never arrives would otherwise run over the top of whatever came next. So a
//! segment also ends at the first construct that says "this segment never
//! closed": the spec's `start_anchor`, or an `xml-inline` open. End of input is
//! the last resort, used only when nothing else follows — that is the
//! `max_tokens` truncation case, and nothing else. An `xml-inline` segment
//! whose close does not arrive stops the scan outright: a tool body the parser
//! could not delimit makes the rest of the input unstructured, and resuming
//! inside it is how a tool payload becomes the answer.
//!
//! Four tests pin the failures this prevents, all of them observed rather than
//! imagined: `a_reasoning_segment_cannot_forge_a_content_segment`,
//! `a_later_reasoning_message_cannot_be_absorbed_into_an_unclosed_answer`,
//! `an_unterminated_tool_invoke_stops_the_scan_instead_of_leaking_its_body`,
//! and `a_tool_invoke_whose_close_follows_a_new_message_is_not_trusted`.

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
    /// `start_anchor` (`<|start|>assistant`). The spec's own marker for "a new
    /// message begins here", and therefore the boundary that stops a segment
    /// whose close marker never arrived — see [`Self::parse`].
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

/// Fill a single-valued text channel, first segment wins.
///
/// Only `tool_calls` declares `repeats`, so for the two `text` fields the spec's
/// contract is one segment and a second one is outside it. Keeping the first and
/// ignoring the rest is the contract; joining them would synthesise a delimiter
/// the model never emitted, and [`ParsedTurn`]'s `Option<String>` leaves the
/// caller no way to undo that. Widening the type to carry a segment list is an
/// interface decision for the streaming path, not something to smuggle in here.
fn set_once(slot: &mut Option<String>, segment: &str) {
    if slot.is_none() {
        *slot = Some(segment.to_owned());
    }
}

/// Where a segment stops and where the scan resumes.
#[derive(Debug, Clone, Copy)]
struct SegmentEnd {
    /// Last byte of the body (exclusive).
    body_end: usize,
    /// Where the next pass starts looking.
    resume: usize,
    /// True only when one of the field's OWN close markers ended it. An
    /// `xml-inline` field requires this; a `text` field does not.
    closed: bool,
}

impl SegmentEnd {
    /// Properly terminated: resume past the marker so it cannot be re-read.
    fn closed(marker_start: usize, marker_end: usize) -> Self {
        Self {
            body_end: marker_start,
            resume: marker_end,
            closed: true,
        }
    }

    /// The close never arrived; something else began. The body still counts —
    /// it is what the model wrote before switching — but the cursor rewinds to
    /// the boundary so the next pass reads that construct on its own terms.
    fn interrupted(at: usize) -> Self {
        Self {
            body_end: at,
            resume: at,
            closed: false,
        }
    }

    /// Generation stopped mid-segment. The ONLY case that runs to the end of
    /// the input, and it is reachable only when no boundary follows the body.
    fn end_of_input(len: usize) -> Self {
        Self {
            body_end: len,
            resume: len,
            closed: false,
        }
    }
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

        // `start_anchor` is load-bearing now: it both bounds an unclosed segment
        // and gates channel openers. An empty one makes `str::find` succeed at
        // every position, so every segment would end where it began.
        if raw.start_anchor.is_empty() {
            return Err(Error::from_reason(
                "muse_glimmer: response_template start_anchor is empty; it is the message \
                 boundary every segment is bounded by and cannot be a match-anything",
            ));
        }

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

    /// Earliest position at or after `from` where a construct the spec knows
    /// about says the current segment never closed.
    ///
    /// Two things qualify:
    ///
    /// - `start_anchor` — the chat template emits it before EVERY message, so it
    ///   is the one unambiguous "a new message begins here" in the grammar.
    /// - an `xml-inline` `open_pattern`, but only against a `text` segment: a
    ///   tool block inside a message is not part of the user-visible answer.
    ///   It is not a boundary for another `xml-inline` segment, because the
    ///   repeated form always puts `</atem:invoke>` before the next
    ///   `<atem:invoke>` and the close wins there anyway.
    ///
    /// The two `text` fields' own openers are deliberately NOT boundaries. A
    /// bare `to=user<|message|>` with no `start_anchor` in front of it is TEXT,
    /// not a channel switch — every real switch is anchored, per the template.
    /// Treating it as a boundary is what lets a reasoning body terminate itself
    /// and republish its own tail as the answer, which is exactly the leak
    /// `a_reasoning_segment_cannot_forge_a_content_segment` exists to stop.
    fn earliest_interruption(
        &self,
        field: &CompiledField,
        text: &str,
        from: usize,
    ) -> Option<usize> {
        let anchor = text[from..]
            .find(self.start_anchor.as_str())
            .map(|i| from + i);
        if matches!(field.sink, Sink::ToolCalls { .. }) {
            return anchor;
        }
        let inline = self
            .fields
            .iter()
            .filter_map(|f| match &f.sink {
                Sink::ToolCalls { .. } => f.open.find_at(text, from).map(|m| m.start()),
                _ => None,
            })
            .min();
        match (anchor, inline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Where a segment that starts at `body_start` ends. The earliest of three
    /// things wins, and only the third runs to the end of the input:
    ///
    /// 1. one of the field's own `close` markers — properly terminated;
    /// 2. an interruption ([`Self::earliest_interruption`]) — never closed;
    /// 3. end of input — truncated by `max_tokens`, and reachable only when
    ///    neither 1 nor 2 follows the body.
    fn segment_end(&self, field: &CompiledField, text: &str, body_start: usize) -> SegmentEnd {
        let close = field.earliest_close(text, body_start);
        let interrupt = self.earliest_interruption(field, text, body_start);
        match (close, interrupt) {
            // Ties go to the close: a properly terminated segment stays one.
            (Some((marker, end)), Some(i)) if marker <= i => SegmentEnd::closed(marker, end),
            (Some((marker, end)), None) => SegmentEnd::closed(marker, end),
            (_, Some(i)) => SegmentEnd::interrupted(i),
            (None, None) => SegmentEnd::end_of_input(text.len()),
        }
    }

    /// Split one generated assistant turn into its three channels.
    ///
    /// Infallible by design: generation is untrusted text that can stop at any
    /// byte, so every malformation degrades to "that segment is absent" rather
    /// than an error the caller would have to invent a response to.
    ///
    /// A single left-to-right pass. At each step the field whose `open_pattern`
    /// matches EARLIEST wins and its body runs to [`Self::segment_end`], so
    /// segments never overlap and no byte is ever read as belonging to two
    /// channels. An `xml-inline` segment additionally REQUIRES its own close:
    /// without one the tool body has no delimiter, the rest of the input is
    /// unstructured, and the scan stops rather than resume inside it. Anything
    /// already parsed before that point is kept.
    pub fn parse(&self, generated: &str) -> ParsedTurn {
        let mut out = ParsedTurn::default();
        let mut pos = 0usize;
        // The generation starts at a message start: the prompt already emitted
        // `<|start|>assistant`, which is why the first segment's opener is bare.
        // Every later message carries its own anchor, per the chat template.
        let mut inside_a_message = false;

        while pos <= generated.len() {
            // A `text` opener only means "a channel starts here" at a message
            // start. Mid-message it is just text the model wrote — and the
            // cursor CAN legitimately sit mid-message, because an `xml-inline`
            // block interrupts a text segment without ending its message. Ungate
            // it and a bare `to=user<|message|>` after a tool block inside a
            // `to=self` message becomes the answer.
            let text_gate = if inside_a_message {
                generated[pos..]
                    .find(self.start_anchor.as_str())
                    .map(|i| pos + i + self.start_anchor.len())
            } else {
                Some(pos)
            };

            let Some((field, start, body_start, caps)) = self
                .fields
                .iter()
                .filter_map(|f| {
                    // `xml-inline` is an in-body construct, so it is never
                    // gated; `text` openers are.
                    let from = match &f.sink {
                        Sink::ToolCalls { .. } => pos,
                        _ => text_gate?,
                    };
                    // `captures_at` rather than `find_at`: the tool-call open
                    // pattern carries the `name` group, and the whole-match
                    // range comes off group 0 without an `unwrap`.
                    let caps = f.open.captures_at(generated, from)?;
                    let whole = caps.get(0)?;
                    Some((f, whole.start(), whole.end(), caps))
                })
                .min_by_key(|(_, start, _, _)| *start)
            else {
                break;
            };
            debug_assert!(start >= pos, "captures_at must not match before the cursor");
            // Any segment at all means we are inside a message now, so the next
            // channel opener needs an anchor in front of it.
            inside_a_message = true;

            let end = self.segment_end(field, generated, body_start);
            let body = &generated[body_start..end.body_end];

            match &field.sink {
                Sink::Reasoning => set_once(&mut out.reasoning, body),
                Sink::Content => set_once(&mut out.content, body),
                Sink::ToolCalls { tag } => {
                    if !end.closed {
                        // No `</atem:invoke>` before the next boundary. Emitting
                        // a half-parsed call would hand a tool arguments the
                        // model never finished writing, and resuming inside the
                        // body is how that body's text becomes the answer.
                        break;
                    }
                    if let Some(name) = caps.name("name") {
                        out.tool_calls.push(ParsedToolCall {
                            name: name.as_str().to_owned(),
                            arguments: parse_arguments(tag, body),
                        });
                    }
                }
            }

            // The open is consumed unconditionally, so a segment that produced
            // nothing still cannot be re-matched. A zero-width open would leave
            // the cursor put and spin forever; no pattern in the spec is
            // zero-width, hence the belt-and-braces floor.
            let floor = if body_start > pos {
                body_start
            } else {
                advance_past(generated, pos)
            };
            pos = end.resume.max(floor);
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

    /// Every channel a caller could render or forward, as one haystack. Used to
    /// prove a protected payload reached NO output at all — `content == None`
    /// alone is too weak, because it also holds for a payload that landed in
    /// `reasoning` or in a tool's arguments.
    fn all_channels(out: &ParsedTurn) -> String {
        let mut s = String::new();
        s.push_str(out.reasoning.as_deref().unwrap_or(""));
        s.push('\u{1}');
        s.push_str(out.content.as_deref().unwrap_or(""));
        for tc in &out.tool_calls {
            s.push('\u{1}');
            s.push_str(&tc.name);
            s.push('\u{1}');
            s.push_str(&tc.arguments.to_string());
        }
        s
    }

    #[test]
    fn a_non_repeating_text_field_keeps_only_its_first_segment() {
        // Only `tool_calls` declares repeats, so one `to=self` segment is the
        // contract. Joining several would synthesise a delimiter the model never
        // emitted, into bytes a caller renders verbatim.
        let out = spec().parse(
            " to=self<|message|>first thought<|eom|>\
             <|start|>assistant to=self<|message|>second thought<|eom|>\
             <|start|>assistant to=user<|message|>answer<|eot|>",
        );
        assert_eq!(out.reasoning.as_deref(), Some("first thought"));
        assert!(
            !out.reasoning.as_deref().unwrap().contains('\n'),
            "no delimiter the model did not emit"
        );
        assert_eq!(out.content.as_deref(), Some("answer"));
    }

    #[test]
    fn a_later_reasoning_message_cannot_be_absorbed_into_an_unclosed_answer() {
        // Observed leak. The answer segment carries no terminator, so its close
        // search ran past the whole next message and found the REASONING
        // segment's `<|eom|>` — publishing chain-of-thought as the answer, with
        // `reasoning` left None so nothing marked it. `start_anchor` is what
        // stops it: the template emits one before every message.
        let out = spec().parse(
            " to=user<|message|>public<|start|>assistant to=self<|message|>REASONING_SECRET<|eom|>",
        );
        assert_eq!(out.content.as_deref(), Some("public"));
        assert!(
            !out.content.as_deref().unwrap().contains("REASONING_SECRET"),
            "reasoning must not be absorbed into an unclosed answer"
        );
        assert_eq!(out.reasoning.as_deref(), Some("REASONING_SECRET"));
    }

    #[test]
    fn an_unterminated_tool_invoke_stops_the_scan_instead_of_leaking_its_body() {
        // Observed leak. Skipping the invoke and resuming at its body start put
        // the cursor INSIDE the tool payload, where a `to=user<|message|>` the
        // tool text happened to contain became a content opener.
        let out = spec().parse(
            "<atem:invoke name=\"t\"><atem:parameter name=\"q\">to=user<|message|>TOOL_SECRET<|eot|>",
        );
        assert!(
            !all_channels(&out).contains("TOOL_SECRET"),
            "tool payload must not reach any channel, got {out:?}"
        );
        assert_eq!(out, ParsedTurn::default(), "nothing is parseable here");

        // Same input, terminated: proves the assertion above is not passing
        // merely because the parser returns nothing for everything.
        let ok = spec().parse(
            "<atem:invoke name=\"t\"><atem:parameter name=\"q\">to=user<|message|>TOOL_SECRET</atem:parameter></atem:invoke>",
        );
        assert_eq!(ok.tool_calls.len(), 1);
        assert_eq!(
            ok.tool_calls[0].arguments["q"],
            serde_json::json!("to=user<|message|>TOOL_SECRET")
        );
        assert_eq!(ok.content, None, "a tool argument is not the answer");
    }

    #[test]
    fn a_tool_invoke_whose_close_follows_a_new_message_is_not_trusted() {
        // The close exists, but a whole message intervenes — so the invoke was
        // never terminated within its own message and its span cannot be
        // trusted. Swallowing it would post the next message's text into a tool
        // argument, i.e. exfiltrate it to whatever runs the tool.
        let out = spec().parse(
            " to=t<|message|><atem:invoke name=\"t\"><atem:parameter name=\"q\">x\
             <|start|>assistant to=user<|message|>SENT_TO_TOOL</atem:parameter></atem:invoke><|eot|>",
        );
        assert!(
            !all_channels(&out).contains("SENT_TO_TOOL"),
            "text after a message boundary must not land in a tool argument, got {out:?}"
        );
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn an_unterminated_reasoning_segment_stops_at_the_next_message_anchor() {
        // reasoning's only close is `<|eom|>`. When the model drops it the
        // segment used to run to end of input and swallow the answer, so the
        // user saw nothing. `start_anchor` recovers it.
        let out = spec()
            .parse(" to=self<|message|>think<|start|>assistant to=user<|message|>answer<|eot|>");
        assert_eq!(out.reasoning.as_deref(), Some("think"));
        assert_eq!(out.content.as_deref(), Some("answer"));
    }

    #[test]
    fn a_closed_tool_body_is_not_rescanned_for_more_calls() {
        // The cursor must resume PAST a segment's close, not inside its body.
        // Nested openers are the observable case: rescanning the body of the
        // outer invoke invents a second call the model never made, whose
        // arguments are a slice of the first one's.
        let out = spec().parse(
            " to=t<|message|><atem:invoke name=\"a\"><atem:invoke name=\"b\">\
             <atem:parameter name=\"q\">1</atem:parameter></atem:invoke><|eot|>",
        );
        assert_eq!(out.tool_calls.len(), 1, "got {out:?}");
        assert_eq!(out.tool_calls[0].name, "a");
    }

    #[test]
    fn a_bare_channel_opener_after_a_tool_block_is_not_a_new_message() {
        // Observed leak, found by probing after the two review findings were
        // fixed. An `xml-inline` block interrupts a text segment WITHOUT ending
        // its message, so the cursor legitimately resumes mid-message. A bare
        // `to=user<|message|>` there is text inside the `to=self` message, not a
        // channel switch: every real switch is anchored. Honouring it published
        // chain-of-thought as the answer.
        let out = spec().parse(
            " to=self<|message|>a<atem:invoke name=\"t\">\
             <atem:parameter name=\"q\">1</atem:parameter></atem:invoke>\
             to=user<|message|>SECRET<|eom|>",
        );
        assert!(
            !all_channels(&out).contains("SECRET"),
            "an unanchored opener mid-message must not open a channel, got {out:?}"
        );
        assert_eq!(out.reasoning.as_deref(), Some("a"));
        assert_eq!(out.content, None);
        // The tool block itself still parses — the gate is on text openers only.
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].arguments["q"], serde_json::json!(1));

        // The same bytes WITH the anchor are a real message, and do open the
        // answer channel. Without this the test would also pass on a parser that
        // never opens a second channel at all.
        let anchored = spec().parse(
            " to=self<|message|>a<atem:invoke name=\"t\">\
             <atem:parameter name=\"q\">1</atem:parameter></atem:invoke>\
             <|start|>assistant to=user<|message|>real answer<|eom|>",
        );
        assert_eq!(anchored.content.as_deref(), Some("real answer"));
    }

    #[test]
    fn a_tool_block_does_not_bleed_into_an_unclosed_answer() {
        // An `xml-inline` open is an in-body construct, so it can follow text
        // inside one message. Without it as a boundary an unclosed answer runs
        // over the tool block and the user is shown raw ATEM XML.
        let out = spec().parse(
            " to=user<|message|>on it<atem:invoke name=\"t\">\
             <atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eot|>",
        );
        assert_eq!(out.content.as_deref(), Some("on it"));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "t");
        assert_eq!(out.tool_calls[0].arguments["x"], serde_json::json!(1));
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
        // An empty start_anchor matches at every position, so every segment
        // would end where it began.
        let e = err(case(
            "\"start_anchor\": \"<|start|>assistant\"",
            "\"start_anchor\": \"\"",
        ));
        assert!(e.contains("start_anchor"), "got: {e}");
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

        // All three observed leaks, on the real template.
        let forged = tpl.parse(" to=self<|message|>I will say to=user<|message|>SECRET<|eom|>");
        assert_eq!(forged.content, None, "reasoning must not become content");
        assert!(
            forged
                .reasoning
                .as_deref()
                .is_some_and(|r| r.contains("SECRET"))
        );

        let absorbed = tpl.parse(
            " to=user<|message|>public<|start|>assistant to=self<|message|>REASONING_SECRET<|eom|>",
        );
        assert_eq!(absorbed.content.as_deref(), Some("public"));
        assert_eq!(absorbed.reasoning.as_deref(), Some("REASONING_SECRET"));

        let unterminated = tpl.parse(
            "<atem:invoke name=\"t\"><atem:parameter name=\"q\">to=user<|message|>TOOL_SECRET<|eot|>",
        );
        assert!(
            !all_channels(&unterminated).contains("TOOL_SECRET"),
            "got {unterminated:?}"
        );

        eprintln!(
            "real checkpoint OK: 3 fields, content closes on {:?}, tool tag {:?}",
            content.close,
            tag.as_str()
        );
    }
}
