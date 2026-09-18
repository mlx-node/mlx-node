use super::lfm2::{LFM2_TOOL_CALL_END, LFM2_TOOL_CALL_START};
use super::markup::all_positions;
use super::{
    CHATML_MARKUP, MarkupSpec, ToolCallResult, extract_tag_blocks, parse_tool_calls,
    strip_tag_blocks,
};

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
    strip_reasoning_preserving_tools_with(text, CHATML_MARKUP)
}

/// [`strip_reasoning_preserving_tools`] parameterized by the family's
/// markup spec — same fixpoint scrub, family's own reasoning/tool tags.
pub(crate) fn strip_reasoning_preserving_tools_with(text: &str, spec: MarkupSpec) -> String {
    let mut current = strip_reasoning_once(text, spec);
    while has_top_level_missing_open_terminator(&current, spec) {
        let next = strip_reasoning_once(&current, spec);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// One pass of the range-based reasoning scrub (see `strip_reasoning_preserving_tools`).
fn strip_reasoning_once(text: &str, spec: MarkupSpec) -> String {
    let mut tool_ranges: Vec<(usize, usize)> =
        extract_tag_blocks(text, spec.tool_open, spec.tool_close)
            .into_iter()
            .map(|(s, e, _)| (s, e))
            .collect();
    // LFM2's pythonic sentinel blocks get the same protection: a call nested
    // inside reasoning is scrubbed with it; a top-level call is preserved
    // verbatim so `parse_lfm2_tool_calls` still sees it.
    if spec.tool_open == CHATML_MARKUP.tool_open && spec.tool_close == CHATML_MARKUP.tool_close {
        tool_ranges.extend(
            extract_tag_blocks(text, LFM2_TOOL_CALL_START, LFM2_TOOL_CALL_END)
                .into_iter()
                .map(|(s, e, _)| (s, e)),
        );
    }
    // The logic below handles the no-tool case for free (empty `tool_ranges` ⇒ every tag is
    // top-level and no spans are dropped), so there is NO separate `parse_thinking` fast path:
    // the scrubber owns its missing-open scanner (`missing_open_close`) and never delegates to
    // the generic `parse_thinking`, which intentionally keeps weaker (first-close-only)
    // missing-open semantics for `parse_generation_output`/OCR/reward callers.

    // Paired reasoning ranges on the original text, excluding any block that is literal
    // argument text inside a tool span.
    let mut reasoning: Vec<(usize, usize)> = Vec::new();
    for (open, close) in spec.reasoning {
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
    if let Some((_, close_end, _)) = applied_missing_open(text, &tool_ranges, spec) {
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
    let out = keep_only_genuine_tool_spans(out, genuine, spec);
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
    spec: MarkupSpec,
) -> Option<(usize, usize, bool)> {
    let mut top_level: Option<(usize, usize)> = None; // earliest top-level close
    let mut straddle: Option<(usize, usize)> = None; // latest in-tool straddle
    for (open, close) in spec.reasoning {
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
fn has_top_level_missing_open_terminator(text: &str, spec: MarkupSpec) -> bool {
    let mut tool_ranges: Vec<(usize, usize)> =
        extract_tag_blocks(text, spec.tool_open, spec.tool_close)
            .into_iter()
            .map(|(s, e, _)| (s, e))
            .collect();
    // LFM2 sentinel spans count too: a `</think>` inside a preserved LFM2
    // call's string arg is an in-tool straddle, NOT a top-level close —
    // misclassifying it would run a second pass that drops the call.
    if spec.tool_open == CHATML_MARKUP.tool_open && spec.tool_close == CHATML_MARKUP.tool_close {
        tool_ranges.extend(
            extract_tag_blocks(text, LFM2_TOOL_CALL_START, LFM2_TOOL_CALL_END)
                .into_iter()
                .map(|(s, e, _)| (s, e)),
        );
    }
    matches!(
        applied_missing_open(text, &tool_ranges, spec),
        Some((_, _, true))
    )
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
fn keep_only_genuine_tool_spans(
    mut out: String,
    mut genuine: Vec<(usize, usize)>,
    spec: MarkupSpec,
) -> String {
    loop {
        // Rescan BOTH families: a removal seam can fuse LFM2 sentinel
        // fragments (`<|tool_call_start` + `|>[f()]<|tool_call_end|>`) into a
        // fabricated block just like `<tool_call>` ones.
        let mut spans = extract_tag_blocks(&out, spec.tool_open, spec.tool_close);
        if spec.tool_open == CHATML_MARKUP.tool_open && spec.tool_close == CHATML_MARKUP.tool_close
        {
            spans.extend(extract_tag_blocks(
                &out,
                LFM2_TOOL_CALL_START,
                LFM2_TOOL_CALL_END,
            ));
        }
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
    split_at_think_end_with(
        raw_text,
        think_end_tag,
        &["<think>", "<longcat_think>"],
        parse_tool_calls,
        parse_generation_output,
    )
}

/// [`split_at_think_end`] parameterized by the family's open-tag prefixes
/// and tool/generation parsers. K2-Horizon passes its `<ifm|think*>`
/// opens and `<ifm|tool_call>` parser; ChatML callers keep the default.
pub(crate) fn split_at_think_end_with(
    raw_text: &str,
    think_end_tag: Option<&str>,
    open_prefixes: &[&str],
    parse_tools: fn(&str) -> (String, Vec<ToolCallResult>),
    parse_fallback: fn(&str) -> (String, Vec<ToolCallResult>, Option<String>),
) -> (String, Vec<ToolCallResult>, Option<String>) {
    // Token-level split: authoritative when think_end_tag is confirmed.
    // Always takes priority — even when an open tag appears in the text
    // (old templates). Tool calls are parsed only from content after the
    // boundary. Uses find (first occurrence): the close tag is a single
    // emitted token, so the first text match is the real boundary.
    if let Some(tag) = think_end_tag
        && let Some(close_pos) = raw_text.find(tag)
    {
        let thinking_text = raw_text[..close_pos].trim();
        // Strip an opening reasoning tag from old-style templates that
        // emit it in generated text (newer templates inject it in the
        // prompt).
        let thinking_text = open_prefixes
            .iter()
            .find_map(|p| thinking_text.strip_prefix(p))
            .unwrap_or(thinking_text)
            .trim();
        let after_tag = &raw_text[close_pos + tag.len()..];
        let response_text = after_tag.trim_start_matches('\n').trim_start();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        let (clean_text, tool_calls) = parse_tools(response_text);
        return (clean_text.trim().to_string(), tool_calls, thinking);
    }
    // No token-level confirmation: fall back to generic text-level parsing.
    // This path is used by callers without token-level info (e.g. build_reward_outputs).
    parse_fallback(raw_text)
}
