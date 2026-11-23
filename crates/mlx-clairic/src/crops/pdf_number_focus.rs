//! Focused view selection and eligibility for page-scoped PDF source-only number fields.
//!
//! Exact port of upstream `clairic_runtime/pdf_number_focus.py`.

use std::{collections::HashSet, sync::LazyLock};

use image::RgbImage;
use regex::Regex;
use serde_json::{Map, Value};

use super::{DEFAULT_MAX_PIXELS, focus, is_truthy, state_only_spec};

/// Stable view cache identifier matching `pdf_number_focus.VIEW_ID`.
pub const VIEW_ID: &str = "pdf-source-only-scalars-v1";

static PAGE_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:page\s+(?:[1-4]|one|two|three|four)|(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+page)\b",
    )
    .ok()
});
static EXCLUDED_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:not|ignore|exclude|excluding|except|without|rather than)\b[^.;:\n]*$")
        .ok()
});

/// Return `true` if `instructions` positively scope the read to a specific page (1..4).
///
/// Exact port of `pdf_number_focus.py::_page_scoped`.
pub fn is_page_scoped(instructions: &str) -> bool {
    let Some(page_re) = PAGE_SCOPE_RE.as_ref() else {
        return false;
    };
    page_re.find_iter(instructions).any(|m| {
        !EXCLUDED_SCOPE_RE
            .as_ref()
            .is_some_and(|excl| excl.is_match(&instructions[..m.start()]))
    })
}

/// Select active fields eligible for PDF source-only number focusing.
///
/// Exact port of the selection predicate in `pdf_number_focus.py::prepare`.
pub fn selected_fields(fields: &Map<String, Value>, active_tasks: &HashSet<String>) -> Vec<String> {
    let mut selected = Vec::new();
    for (name, spec) in fields {
        if !active_tasks.contains(name) {
            continue;
        }
        let is_number = spec.get("type").and_then(Value::as_str) == Some("number");
        let is_source_only = spec.get("source_only").and_then(Value::as_bool) == Some(true);
        let has_knowledge = is_truthy(spec.get("knowledge"));
        let instructions = match spec.get("instructions") {
            None => "",
            Some(Value::String(s)) => s.as_str(),
            Some(_) => continue,
        };
        if is_number
            && is_source_only
            && !has_knowledge
            && !state_only_spec(spec)
            && is_page_scoped(instructions)
        {
            selected.push(name.clone());
        }
    }
    selected
}

/// Return `true` if at least one rendered PDF page can be cropped & upscaled by [`focus::focus_document`].
pub fn has_focusable_pdf_page(pdf_pages: &[RgbImage]) -> bool {
    pdf_pages
        .iter()
        .any(|page| focus::focus_document(page, DEFAULT_MAX_PIXELS, 2.0).is_some())
}

/// Evaluate complete eligibility for the PDF number focus pass, returning the selected
/// field names (or empty when disabled, no PDF documents, or no focusable PDF pages).
pub fn eligible_fields(
    enabled: bool,
    has_documents: bool,
    fields: &Map<String, Value>,
    active_tasks: &HashSet<String>,
    pdf_pages: &[RgbImage],
) -> Vec<String> {
    if !enabled || !has_documents || pdf_pages.is_empty() || !has_focusable_pdf_page(pdf_pages) {
        return Vec::new();
    }
    selected_fields(fields, active_tasks)
}

/// Build focused document views for each PDF page (applying `focus_document` up to
/// `max_scale` / `max_pixels`, falling back to the original page when unchanged).
pub fn focused_document_views(
    pages: &[RgbImage],
    max_pixels: usize,
    max_scale: f64,
) -> Vec<RgbImage> {
    pages
        .iter()
        .map(|page| focus::focus_document_or_clone(page, max_pixels, max_scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use serde_json::json;

    #[test]
    fn detects_page_scoped_instructions_and_exclusions() {
        assert!(is_page_scoped("Read the subtotal from page 2."));
        assert!(is_page_scoped("Take the amount printed on the first page."));
        assert!(!is_page_scoped("Read the total, ignore page 2."));
        assert!(!is_page_scoped("Read the grand total."));
    }

    #[test]
    fn selects_only_eligible_source_only_pdf_numbers() {
        let schema: Map<String, Value> = serde_json::from_value(json!({
            "p2_total": {
                "type": "number",
                "source_only": true,
                "instructions": "Extract subtotal from page 2."
            },
            "state_only_num": {
                "type": "number",
                "source_only": true,
                "instructions": "From state text only, read page 2 amount."
            },
            "non_source": {
                "type": "number",
                "instructions": "Extract subtotal from page 2."
            },
            "text_field": {
                "type": "text",
                "source_only": true,
                "instructions": "Extract label from page 1."
            }
        }))
        .unwrap();
        let active: HashSet<String> = schema.keys().cloned().collect();

        // Sparse white-margin PDF page is focusable
        let mut page = RgbImage::from_pixel(1000, 1000, Rgb([255, 255, 255]));
        for y in 400..500 {
            for x in 400..550 {
                page.put_pixel(x, y, Rgb([10, 10, 10]));
            }
        }
        let selected = eligible_fields(true, true, &schema, &active, &[page]);
        assert_eq!(selected, vec!["p2_total".to_string()]);
    }
}
