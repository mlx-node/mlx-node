// ---------------------------------------------------------------------------
// Tag extraction helpers — replace all regex with simple string scanning
// ---------------------------------------------------------------------------

/// Tag set a family's reasoning/tool markup uses.
///
/// ChatML parsers use `<think>` / `<longcat_think>` and `<tool_call>`.
/// K2-Horizon supplies `<ifm|think*>` / `<ifm|tool_call>` through the
/// spec-parameterized entry points; LFM2 sentinel parsing stays separate.
#[derive(Clone, Copy)]
pub(crate) struct MarkupSpec {
    /// Reasoning `(open, close)` pairs, scanned in order.
    pub reasoning: &'static [(&'static str, &'static str)],
    /// Tool-call block delimiters.
    pub tool_open: &'static str,
    pub tool_close: &'static str,
}

/// The default ChatML markup spec.
pub(crate) const CHATML_MARKUP: MarkupSpec = MarkupSpec {
    reasoning: &[
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ],
    tool_open: "<tool_call>",
    tool_close: "</tool_call>",
};

/// Extract all blocks between `<open_tag>` and `</close_tag>`.
/// Returns Vec of (start_of_open_tag, end_of_close_tag, inner_content).
pub(crate) fn extract_tag_blocks<'a>(
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
pub(crate) fn strip_tag_blocks(text: &str, open_tag: &str, close_tag: &str) -> String {
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

/// All byte offsets at which `needle` occurs in `text` (tool spans included).
pub(super) fn all_positions(text: &str, needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find(needle) {
        let pos = from + rel;
        out.push(pos);
        from = pos + needle.len();
    }
    out
}
