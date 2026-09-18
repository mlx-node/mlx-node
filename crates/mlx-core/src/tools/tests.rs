use super::chatml::sanitize_json_string;
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
    let input = "<tool_call>{\"name\":\"f\",\"arguments\":{\"q\":\"</think>\\nfoo\"}}</tool_call>";
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
    let input =
        "<think>secret before <tool_call>{\"name\":\"f\"}</tool_call> secret after</think>\nanswer";
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
    let input =
        "secret <think>inner</think> more </think>\n<tool_call>{\"name\":\"f\"}</tool_call> final";
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
        out.contains("answer prefix") && out.contains("more answer") && out.contains("<tool_call>"),
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
    let input =
        "secret </think> literal <tool_call>{\"name\":\"leak\"}</tool_call> more </think>\nfinal";
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
    let input =
        "secret <tool_call>{\"name\":\"leak\"}</tool_call> </think>\nmore secret </think>\nfinal";
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
    let (_, calls) = parse_tool_calls(r#"<tool_call>{"arguments": {"key": "value"}}</tool_call>"#);

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
    let input =
        "I need to check.\n</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";

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
    let input = "Check: <|tool_call_start|>[f(x=1)]<|tool_call_end|>[f(x=1)]<|tool_call_end|>Done.";
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
    let input = "<|tool_call_start|>[f(x=1)]<|tool_call_end|> mid <|tool_call_start|>[f(@@@)] tail";
    let (text, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(text, "mid <|tool_call_start|>[f(@@@)] tail");
    // The FIRST block failing is still all-or-nothing verbatim —
    // vLLM's `content=model_output`, second blocks are never seen.
    let input =
        "<|tool_call_start|>[f(@@@)]<|tool_call_end|> <|tool_call_start|>[g(x=1)]<|tool_call_end|>";
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
        "f(a if b, confirmed=True)",   // incomplete ternary — `else` owed
        "f(a else b)",                 // `else` with no open `if`
        "f((a if b), x=1)",            // bracket closes mid-ternary
        "f([a if b], x=1)",            // `if` outside a comprehension
        "f([a if b for c in d], x=1)", // `for` doesn't rescue the ternary
        "f(x for x in y, k=1)",        // genexpr must be the sole argument
        "f(a not is b, x=1)",          // `not is` isn't an operator
        "f(a not b, x=1)",             // `not` isn't binary — only `in` follows
        "f(a not (b), x=1)",
        "f(a is not in b, x=1)",          // `is not` can't chain into `in`
        "f(lambda 1: x, confirmed=True)", // digit-led param name
        "f(lambda 'a': x)",               // string where a param name belongs
        "f(lambda ,: x)",                 // `,` before any param
        "f(lambda =x: y)",                // `=` with no name
        "f(lambda (a): x)",               // `(a)` params are Python 2
        "f(lambda a,,b: x)",              // empty param between commas
        "f({1:}, confirmed=True)",        // dict entry missing its value
        "f({:}, x=1)",                    // `:` where a key belongs
        "f([1:2], x=1)",                  // `:` inside a list display
        "f((1:2), x=1)",                  // `:` inside parens
        "f([a: b], x=1)",                 // `:` inside a list via idents
        "f([x for x], confirmed=True)",   // `for` with no `in`
        "f([x for x for y in z], x=1)",   // first `for` missing `in`
        "f((x for x), y=1)",              // genexpr missing `in`
        "f(x for x)",                     // sole-arg genexpr missing `in`
        "f([x for (a in b)], y=1)",       // `in` inside target parens
        "f([x for x + y in z], confirmed=True)", // `x + y` is no target
        "f([x for x and y in z], x=1)",   // `and` in a target
        "f([x for 1 in z], x=1)",         // literal target
        "f([x for 's' in z], x=1)",       // string target
        "f([x for a(b) in z], x=1)",      // call target
        "f([x for a.b c in z], x=1)",     // juxtaposed names
        "f([x for -a in z], x=1)",        // unary target
        "f([x for {a: b} in z], x=1)",    // dict target
        "f([x for not a in z], x=1)",     // keyword target
        "f([x for a + if b in z], x=1)",  // operator then keyword
        "f(x for x + y in z)",            // genexpr target
        "f(lambda for: x, confirmed=True)", // keyword param name
        "f(lambda True: x)",              // constant param name
        "f(lambda None: x)",              // `None` param name
        "f(lambda a b: x)",               // param without `,`
        "f(lambda a.b: x)",               // dotted param name
        "f(lambda a(x): y)",              // call-shaped param
        "f([*items for item in source], confirmed=True)", // starred comp elem
        "f(*a for a in b, x=1)",          // starred genexpr elem
        "f((*a), x=1)",                   // bare starred group
        "f(x[a for a in y], x=1)",        // genexpr in subscript
        "f([x for a[i for j in z] in w], x=1)", // nested subscript genexpr
        "f({1, 2:3}, confirmed=True)",    // set element then `:` — mixed literal
        "f({1:2, 3}, confirmed=True)",    // dict pair then bare element
        "f({x:=1, a:2}, x=1)",            // walrus locks set — then `:`
        "f(1 + *a, x=1)",                 // `*` after an operator
        "f(x[1:*a], x=1)",                // `*` after a slice colon
        "f(helper(5=1), x=1)",            // `=` after a non-name operand
        "f(helper(a.b=1), x=1)",          // `=` after an attribute
        "f(helper(*a=1), x=1)",           // `=` after a starred element
        "f(x[a=1], x=1)",                 // `=` inside a subscript
        "f({a=1}, x=1)",                  // `=` inside a display
        "f([a, x for x in y], x=1)",      // comp after an element `,`
        "f({k:v, x for x in y}, x=1)",    // comp after a dict `,`
        "f({k:v, k2}, x=1)",              // dict key missing its `:`
        "f(x for x in y, z=1)",           // a genexpr must be the sole arg
        "f(1:=2, confirmed=True)",        // literal walrus target
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
        ("f(lambda: x, y=1)", "{\"y\":1}"),    // zero-param lambda
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
    let input =
        "<|tool_call_start|>[run(cmd='a\\nb', raw=r'c\\d', lit='line1\nline2')]<|tool_call_end|>";
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
