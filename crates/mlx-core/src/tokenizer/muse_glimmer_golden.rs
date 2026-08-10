//! Golden rendered-prompt strings for Muse-Glimmer's ATEM/Onyx chat template.
//!
//! This is the M0 gate: the only test that drives all five tokenizer fixes
//! (`.items()` on maps, the family-aware marker sanitizer, its pinned marker
//! list, `current_date`/`reasoning_strength` context keys, and `tojson`'s
//! Python `json.dumps` separators) through the checkpoint's *own*
//! `chat_template.jinja` and pins the bytes that come out.
//!
//! Needs only `tokenizer.json` + `chat_template.jinja` from the checkpoint —
//! no weights, no GPU, no Metal.
//!
//! ## Why this lives in the library rather than in `tests/`
//!
//! The render entry point ([`Qwen3Tokenizer::render_chat_template_sync_with_content_order`])
//! is private and [`RenderContextOptions`] is `pub(crate)`; an integration test
//! is an external consumer and can see neither. Rather than widen either — a
//! test-only `pub` on a shipping library, and a `pub fn` taking a `pub(crate)`
//! type trips `private_interfaces` under `-D warnings` — this is a `#[cfg(test)]`
//! child module of `tokenizer`, which reaches its ancestor's private items
//! directly. Net new public API: zero. The EOS assertion additionally reads
//! `crate::engine::persistence`, which is `pub(crate)` and likewise invisible
//! from `tests/`.
//!
//! ## Every test is `#[ignore]`d on purpose
//!
//! CI has no checkpoint. A silent in-test `return` there would report a pass
//! that proved nothing, which for the M0 gate is worse than a failure, so the
//! gate is `#[ignore]` + an explicit opt-in run instead. Locally:
//!
//! ```text
//! MLX_TEST_MUSE_GLIMMER_MODEL_PATH=/path/to/muse-glimmer-30b \
//!   cargo test -p mlx-core --lib -- muse_glimmer_golden --ignored --nocapture
//! ```
//!
//! ## READ THIS FIRST: the gate currently FAILS, and that is the point
//!
//! These goldens pin what HuggingFace's renderer produces from this template —
//! the definition of a correct prompt. Ours does not produce it yet. Two
//! production defects remain, neither of which any of the five landed fixes
//! addresses, and both were found by writing this gate. Do **not** "repair"
//! these tests by relaxing them to match current output; that would enshrine
//! the bugs.
//!
//! 1. **The template does not parse at all.** miniJinja parses a call keyword
//!    argument's value with `parse_expr_noif` (`compiler/parser.rs:672`), so a
//!    conditional expression there is a hard *parse* error. The template's
//!    tool-name fallback has exactly one:
//!    `{%- set rns = namespace(name=tcid if tcid else '') -%}`. Every
//!    Muse-Glimmer render therefore fails with
//!    `Template parse error: syntax error: unexpected identifier, expected ','`
//!    — even a bare "hi", because parsing precedes rendering. Parenthesising
//!    the value (`name=(tcid if tcid else '')`) parses, because parentheses
//!    re-enter full `parse_expr` (`parser.rs:806`); a probe confirmed that one
//!    change alone makes the entire template parse and render.
//!
//! 2. **`.get(key)` returns `Undefined` where Python returns `None`.** With a
//!    one-argument `.get`, the map branch of the unknown-method callback
//!    defaults to `Value::UNDEFINED`, but `dict.get` yields `None`. The
//!    template's assistant branch is gated on exactly that difference:
//!
//!    ```jinja
//!    {%- set end_turn = message.get('end_turn') -%}
//!    {%- if end_turn is none -%}
//!      {%- set end_turn = not (recipient and recipient != 'user') -%}
//!    {%- endif -%}
//!    ```
//!
//!    `undefined is none` is false, so the fallback never runs, `end_turn`
//!    stays falsy, and every plain assistant turn terminates `<|eom|>`
//!    ("channel continues") instead of `<|eot|>` ("turn ends"). Under HF the
//!    fallback runs and computes `not ('user' and false)` = `true` = `<|eot|>`.
//!    A probe returning `Value::from(())` instead flips all of these goldens
//!    green.
//!
//! With both probes applied every test here passes, so these expectations are
//! verified reachable, not aspirational.

use super::*;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Pinned so the golden strings are stable. The template prints this verbatim
/// into the system prefix, and the production path pins it per session for the
/// same reason — see [`RenderContextOptions::current_date`].
const PINNED_DATE: &str = "2026-08-10";

/// Locate the checkpoint. The env var wins (Tasks 6/7 use the same name); the
/// relative path is the fallback for a plain checkout, and resolves to nothing
/// inside a git worktree, which has no `.cache`.
fn checkpoint_dir() -> Option<PathBuf> {
    let candidate = match std::env::var("MLX_TEST_MUSE_GLIMMER_MODEL_PATH") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.cache/models/muse-glimmer-30b")
            .canonicalize()
            .ok()?,
    };
    candidate
        .join("chat_template.jinja")
        .exists()
        .then_some(candidate)
}

/// Load the real tokenizer once per process: `tokenizer.json` is 28 MB, and
/// every test here needs the same one.
fn checkpoint_tokenizer() -> Option<&'static Qwen3Tokenizer> {
    static TOKENIZER: OnceLock<Option<Qwen3Tokenizer>> = OnceLock::new();
    TOKENIZER
        .get_or_init(|| {
            let dir = checkpoint_dir()?;
            Some(
                Qwen3Tokenizer::from_file(&dir.join("tokenizer.json"))
                    .expect("the real checkpoint's tokenizer.json must load"),
            )
        })
        .as_ref()
}

macro_rules! checkpoint {
    () => {
        match checkpoint_tokenizer() {
            Some(tokenizer) => tokenizer,
            None => {
                eprintln!(
                    "skipping: no Muse-Glimmer checkpoint (set MLX_TEST_MUSE_GLIMMER_MODEL_PATH)"
                );
                return;
            }
        }
    };
}

/// A [`ChatMessage`] with every optional field cleared, so each fixture below
/// sets only the two or three that matter to it.
fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

fn pinned_ctx() -> RenderContextOptions {
    RenderContextOptions {
        current_date: Some(PINNED_DATE.to_string()),
        reasoning_strength: None,
    }
}

fn render_with(
    tokenizer: &Qwen3Tokenizer,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    render_ctx: RenderContextOptions,
) -> String {
    tokenizer
        .render_chat_template_sync_with_content_order(
            messages,
            Some(true),
            tools,
            None,
            MultimodalContentOrder::TextThenMedia,
            None,
            render_ctx,
        )
        .expect("the checkpoint's own chat template must render")
}

fn render(
    tokenizer: &Qwen3Tokenizer,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
) -> String {
    render_with(tokenizer, messages, tools, pinned_ctx())
}

/// `properties` and `required` are the JSON-string / `Vec` shapes our
/// `#[napi(object)]` types actually carry; the render path re-parses
/// `properties` into a real object so `fn.parameters | tojson` renders nested
/// JSON — which is what makes the separators prompt-visible.
///
/// The description is deliberately set: `fn.description | tojson` renders the
/// literal `null` when it is `None`.
fn wx_forecast_tool() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "wx.forecast".to_string(),
            description: Some("Get a forecast.".to_string()),
            parameters: Some(FunctionParameters {
                r#type: "object".to_string(),
                properties: Some(r#"{"city": {"type": "string"}}"#.to_string()),
                required: Some(vec!["city".to_string()]),
            }),
        },
    }
}

// ── System prefix ──────────────────────────────────────────────────────────

/// The whole no-system prompt, not fragments of it: this string is the
/// reference every later change is diffed against.
const GOLDEN_NO_SYSTEM: &str = r##"<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.
Knowledge cutoff: 2026-01-04.
Current date: 2026-08-10.

Reasoning strength: high.

# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>hi<|eot|><|start|>assistant"##;

#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn default_system_preamble_is_injected_when_no_system_message_is_present() {
    let tokenizer = checkpoint!();
    let out = render(tokenizer, &[msg("user", "hi")], None);
    assert_eq!(out, GOLDEN_NO_SYSTEM);
}

#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn generation_prompt_is_bare_with_no_message_marker() {
    let tokenizer = checkpoint!();
    let out = render(tokenizer, &[msg("user", "hi")], None);
    assert!(
        out.ends_with("<|start|>assistant"),
        "got tail: {}",
        &out[out.len().saturating_sub(40)..]
    );
    assert!(
        !out.ends_with("<|start|>assistant<|message|>"),
        "generation prompt must not carry <|message|>: {out}"
    );
    assert!(
        !out.ends_with("<|start|>assistant "),
        "generation prompt must not carry a trailing space: {out}"
    );
}

#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn a_caller_system_message_replaces_the_default_preamble() {
    let tokenizer = checkpoint!();
    let out = render(
        tokenizer,
        &[msg("system", "Be terse."), msg("user", "hi")],
        None,
    );
    assert!(
        out.contains("<|start|>system<|message|>Be terse.\n\nReasoning strength: high."),
        "got: {out}"
    );
    assert!(
        !out.contains("You are a helpful AI assistant."),
        "default preamble must be suppressed: {out}"
    );
    assert!(
        !out.contains("Knowledge cutoff"),
        "default preamble must be suppressed: {out}"
    );
    assert!(
        !out.contains("Current date"),
        "the default preamble owns the date line; a caller system message drops it: {out}"
    );
}

/// `current_date` is the whole reason [`RenderContextOptions`] exists: the
/// template's other branch is `strftime_now`, a global `transformers` registers
/// and miniJinja does not, so leaving it unset silently deletes the line.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn an_unset_current_date_deletes_the_date_line_entirely() {
    let tokenizer = checkpoint!();
    let out = render_with(
        tokenizer,
        &[msg("user", "hi")],
        None,
        RenderContextOptions::default(),
    );
    assert!(
        out.contains("\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength: high."),
        "the cutoff line must butt straight up against the reasoning line: {out}"
    );
    assert!(
        !out.contains("Current date"),
        "with current_date unset the line must vanish, not render hollow: {out}"
    );
}

/// The second key Task 4 added. Nothing else covers it: the template
/// substitutes `high` when it is undefined *or* empty, so `None` and `high` are
/// byte-identical and only a non-default value can prove the key is wired.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn a_pinned_reasoning_strength_replaces_the_templates_high_default() {
    let tokenizer = checkpoint!();
    let out = render_with(
        tokenizer,
        &[msg("user", "hi")],
        None,
        RenderContextOptions {
            current_date: Some(PINNED_DATE.to_string()),
            reasoning_strength: Some("low".to_string()),
        },
    );
    assert!(
        out.contains("\n\nReasoning strength: low."),
        "reasoning_strength not forwarded: {out}"
    );
    assert!(
        !out.contains("Reasoning strength: high."),
        "the template default must not survive an explicit value: {out}"
    );
}

// ── Assistant turns ────────────────────────────────────────────────────────

/// The brief expected `<|eom|>` then `<|eot|>` here. Nothing our `ChatMessage`
/// can carry produces that, so this pins what the template really renders.
///
/// The non-tool-call assistant branch never consults the loop-level
/// `end_token`; it computes its own:
///
/// ```jinja
/// {%- set recipient = message.get('recipient') or 'user' -%}
/// {%- set end_turn = message.get('end_turn') -%}
/// {%- if end_turn is none -%}
///   {%- set end_turn = not (recipient and recipient != 'user') -%}
/// {%- endif -%}
/// ```
///
/// `ChatMessage` has neither `recipient` nor `end_turn`. `recipient` therefore
/// defaults to `'user'` — and since that is truthy, every plain assistant turn
/// renders ` to=user`. `end_turn` resolves through the fallback to
/// `not ('user' and false)` = `true`, i.e. `<|eot|>` for *both* messages, which
/// is why `<|eom|>` must appear nowhere in this render.
///
/// This is one of the two tests blocked on defect 2 in the module docs: our
/// `.get` hands back `Undefined`, `undefined is none` is false, the fallback is
/// skipped, and both turns come out `<|eom|>`.
///
/// GAP, recorded deliberately and *separate* from that defect: a genuinely
/// multi-part assistant turn (`<|eom|>`-joined) and any non-`user` recipient
/// cannot be replayed through our types at all, because the fields do not
/// exist. Adding them is out of scope — `ChatMessage` is `#[napi(object)]`, so
/// it is also the TypeScript surface.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn consecutive_assistant_messages_both_terminate_with_eot_because_recipient_defaults_to_user() {
    let tokenizer = checkpoint!();
    let out = render(
        tokenizer,
        &[
            msg("user", "hi"),
            msg("assistant", "part one"),
            msg("assistant", "part two"),
        ],
        None,
    );
    assert!(
        out.contains(
            "<|start|>assistant to=user<|message|>part one<|eot|>\
             <|start|>assistant to=user<|message|>part two<|eot|>"
        ),
        "got: {out}"
    );
    assert!(
        !out.contains("<|eom|>"),
        "no field we can set makes the template emit <|eom|> on a plain assistant turn: {out}"
    );
}

/// The `<|eom|>` after the reasoning is a hard literal in the template, so it
/// is right either way; the `<|eot|>` on the answer is computed, so this test is
/// also blocked on defect 2.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn reasoning_renders_on_the_self_channel_before_the_answer() {
    let tokenizer = checkpoint!();
    let mut assistant = msg("assistant", "answer");
    assistant.reasoning_content = Some("thinking out loud".to_string());
    let out = render(tokenizer, &[msg("user", "hi"), assistant], None);
    assert!(
        out.contains(
            "<|start|>assistant to=self<|message|>thinking out loud<|eom|>\
             <|start|>assistant to=user<|message|>answer<|eot|>"
        ),
        "got: {out}"
    );
}

// ── THE M0 GATE ────────────────────────────────────────────────────────────

/// The whole tool-round-trip prompt. Along with [`GOLDEN_NO_SYSTEM`] this is
/// the reference for every later change, so it is pinned entire rather than
/// probed with `contains`.
const GOLDEN_TOOL_ROUND_TRIP: &str = r##"<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.
Knowledge cutoff: 2026-01-04.
Current date: 2026-08-10.

Reasoning strength: high.

In this environment you have access to a set of tools you can use to answer the user's question.

You can invoke a function by writing a "<atem:function_calls>" block like the following:
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
// Tool metadata
{"name": "wx", "description": ""}
// Function schemas
{"name": "wx.forecast", "description": "Get a forecast.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}

Here's an example of how to call a function in the tool set:
(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)

to=example_tool_name.example_function_name

<atem:function_calls>
<atem:invoke name="example_tool_name.example_function_name">
<atem:parameter name="example_parameter_1">value_1</atem:parameter>
<atem:parameter name="example_parameter_2">This is the value for the second parameter
that can span
"multiple" lines
</atem:parameter>
</atem:invoke>
</atem:function_calls>

# Valid recipients: "self", "wx.*", "user".<|eot|><|start|>user<|message|>weather in Paris?<|eot|><|start|>assistant to=wx.forecast<|message|><atem:function_calls>
<atem:invoke name="wx.forecast">
<atem:parameter name="city">Paris</atem:parameter>
<atem:parameter name="days">3</atem:parameter>
</atem:invoke>
</atem:function_calls><|eot|><|start|>tool wx.forecast<|message|><tool_output name="wx.forecast">
18C, clear
</tool_output><|eot|><|start|>assistant to=user<|message|>18C and clear.<|eot|><|start|>assistant"##;

/// A tool round trip re-renders an assistant message that *carries*
/// `tool_calls`, which is the only path reaching the template's
/// `args.items()`. The tool-definitions block uses no `.items()`, so a test
/// that merely declares a tool renders fine while turn 2 of every real tool
/// loop hard-fails. Hence: call, result, answer — all three.
///
/// Verified: deleting the `items` arm from the unknown-method callback fails
/// *only* this test out of the eleven here, including the sibling that declares
/// the very same tool — `unknown method: map has no method named items`.
///
/// The ATEM block's own `<|eot|>` comes from the loop-level `end_token` and is
/// correct today; only the final answer's terminator is blocked on defect 2.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn full_tool_round_trip_renders() {
    let tokenizer = checkpoint!();

    let mut call = msg("assistant", "");
    call.tool_calls = Some(vec![ToolCall {
        // The template resolves the tool-result name by matching
        // `message.tool_call_id` against a prior `tool_calls[].id` — there is
        // no `name` field on a tool message. Drop this `id` and the rendered
        // name goes empty.
        id: Some("call_1".to_string()),
        name: "wx.forecast".to_string(),
        // A JSON *object* string: `render_atem` raises unless the parsed
        // arguments are a mapping.
        arguments: r#"{"city": "Paris", "days": 3}"#.to_string(),
    }]);

    let mut result = msg("tool", "18C, clear");
    result.tool_call_id = Some("call_1".to_string());

    let out = render(
        tokenizer,
        &[
            msg("user", "weather in Paris?"),
            call,
            result,
            msg("assistant", "18C and clear."),
        ],
        Some(&[wx_forecast_tool()]),
    );
    assert_eq!(out, GOLDEN_TOOL_ROUND_TRIP);
}

/// Split out from the round trip so a separator regression names itself.
/// `fn.parameters | tojson` is the only nested JSON in the prompt, so it is
/// where Task 5's `json.dumps` separators are visible.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn tool_schemas_carry_python_json_dumps_separators() {
    let tokenizer = checkpoint!();
    let out = render(
        tokenizer,
        &[msg("user", "weather in Paris?")],
        Some(&[wx_forecast_tool()]),
    );
    assert!(
        out.contains(
            r#"{"name": "wx.forecast", "description": "Get a forecast.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}"#
        ),
        "tool schema separators wrong: {out}"
    );
    assert!(
        !out.contains(r#"{"type":"object""#),
        "serde_json's compact separators must not reach the prompt: {out}"
    );
    // Declaring a tool widens the recipient list; `render_system_meta` derives
    // the namespace from the part of the name before the first `.`.
    assert!(
        out.contains(r#"# Valid recipients: "self", "wx.*", "user"."#),
        "got: {out}"
    );
}

// ── Media ──────────────────────────────────────────────────────────────────

/// `<|patch|>` is what this template emits for an image part. `<|image|>`
/// (200090) is in the vocabulary but is a decoy the template never reaches.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn image_part_emits_a_single_patch_placeholder() {
    let tokenizer = checkpoint!();
    let mut user = msg("user", "what is this?");
    // A `user` message with a non-empty `images` is what turns `content` into
    // the parts array the template's `render_content` loops over.
    user.images = Some(vec![Uint8Array::new(vec![0; 4])]);
    let out = render(tokenizer, &[user], None);
    assert!(
        out.contains("<|start|>user<|message|>what is this?<|patch|><|eot|>"),
        "got: {out}"
    );
    assert_eq!(
        out.matches("<|patch|>").count(),
        1,
        "one image must yield exactly one placeholder: {out}"
    );
    assert!(
        !out.contains("<|image|>"),
        "<|image|> (200090) is a decoy and must never be emitted: {out}"
    );
}

// ── Stop tokens ────────────────────────────────────────────────────────────

/// The real `generation_config.json`, not a synthetic fixture: this pins the
/// checkpoint's actual `[200001, 200008]` against the parser an earlier task
/// already shape-tested, so the two are complementary.
#[test]
#[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
fn eos_ids_resolve_to_both_stops() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipping: no Muse-Glimmer checkpoint (set MLX_TEST_MUSE_GLIMMER_MODEL_PATH)");
        return;
    };
    let ids = crate::engine::persistence::parse_generation_defaults(&dir).eos_token_ids;
    assert_eq!(
        ids,
        vec![200001, 200008],
        "both <|end_of_text|> and <|eot|> must be stops, in file order"
    );
    assert!(
        !ids.contains(&200007),
        "<|eom|> must not be a stop — it ends a channel, not a turn: {ids:?}"
    );
}
