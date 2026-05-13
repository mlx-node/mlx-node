//! BIOES span extraction for the privacy-filter token classifier.
//!
//! Walks a Viterbi-decoded BIOES tag sequence and emits coherent
//! [`Entity`] spans. Multi-token entities are `B-X ... E-X` and
//! singletons are `S-X`. Each emitted entity is annotated with the
//! mean of the per-token probabilities over its constituent tokens;
//! spans whose mean score falls below the threshold are filtered out.
//!
//! ## Offsets are byte offsets
//!
//! Hugging Face tokenizers (the ones used to produce the input
//! `offsets` slice) report **byte** offsets into the original
//! UTF-8 source string. This implementation therefore slices the
//! source via `&source_text[start..end]` directly without any
//! `char_indices` conversion. Callers passing non-byte offsets must
//! convert ahead of time.

/// A coherent span extracted from a BIOES tag sequence.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Start byte offset into the source text (inclusive).
    pub start: usize,
    /// End byte offset into the source text (exclusive).
    pub end: usize,
    /// Privacy class without the BIOES prefix, e.g. `"private_email"`.
    pub label: String,
    /// Mean of per-token probabilities over the span tokens.
    pub score: f32,
    /// Owned copy of `source_text[start..end]`.
    pub text: String,
}

/// Internal book-keeping for an in-progress span.
struct OpenSpan {
    class: String,
    start: usize,
    probs: Vec<f32>,
}

/// Parse a BIOES tag string into `(prefix, class)`.
///
/// Returns `None` for `"O"` and any malformed tag (no `-` separator).
fn parse_tag(tag: &str) -> Option<(char, &str)> {
    if tag == "O" {
        return None;
    }
    let (prefix, class) = tag.split_once('-')?;
    let prefix_char = prefix.chars().next()?;
    if prefix.len() != 1 || !matches!(prefix_char, 'B' | 'I' | 'E' | 'S') {
        return None;
    }
    Some((prefix_char, class))
}

/// Emit an [`Entity`] from an [`OpenSpan`] if its mean score clears
/// the threshold and its byte slice is valid.
fn emit(open: OpenSpan, end: usize, source_text: &str, threshold: f32, out: &mut Vec<Entity>) {
    if open.probs.is_empty() {
        return;
    }
    let mean = open.probs.iter().sum::<f32>() / open.probs.len() as f32;
    if mean < threshold {
        return;
    }
    let Some(text) = source_text.get(open.start..end) else {
        return;
    };
    out.push(Entity {
        start: open.start,
        end,
        label: open.class,
        score: mean,
        text: text.to_string(),
    });
}

/// Walk a BIOES tag sequence, extracting coherent spans.
///
/// Tags follow strict BIOES legality (assumed already enforced by the
/// Viterbi decoder); this function does NOT re-validate. If it
/// encounters a malformed sequence (e.g. an `I-X` without a preceding
/// `B-X`, or a `B-X` followed by `O`), it silently drops the offending
/// fragment and continues.
///
/// `offsets` are interpreted as **byte** offsets into `source_text`
/// (Hugging Face tokenizer convention).
pub fn extract_spans(
    tags: &[usize],
    id2label: &[String],
    per_token_probs: &[f32],
    offsets: &[(usize, usize)],
    source_text: &str,
    threshold: f32,
) -> Vec<Entity> {
    let n = tags.len();
    assert_eq!(
        per_token_probs.len(),
        n,
        "per_token_probs length must match tags"
    );
    assert_eq!(offsets.len(), n, "offsets length must match tags");

    let mut out: Vec<Entity> = Vec::new();
    let mut open: Option<OpenSpan> = None;

    for i in 0..n {
        let tag_id = tags[i];
        let tag = id2label.get(tag_id).map(String::as_str).unwrap_or("O");
        let prob = per_token_probs[i];
        let (tok_start, tok_end) = offsets[i];

        match parse_tag(tag) {
            None => {
                // `O` or malformed → close any open span (without emitting).
                open = None;
            }
            Some(('B', class)) => {
                // Close any open span (malformed: no E before new B).
                open = Some(OpenSpan {
                    class: class.to_string(),
                    start: tok_start,
                    probs: vec![prob],
                });
            }
            Some(('I', class)) => {
                if let Some(mut span) = open.take()
                    && span.class == class
                {
                    span.probs.push(prob);
                    open = Some(span);
                }
                // else: class mismatch (drop open span) or I-X without
                // B-X (skip token). Both fall through silently.
            }
            Some(('E', class)) => {
                if let Some(mut span) = open.take()
                    && span.class == class
                {
                    span.probs.push(prob);
                    emit(span, tok_end, source_text, threshold, &mut out);
                }
                // else: class mismatch or E-X without B-X — drop silently.
            }
            Some(('S', class)) => {
                // Close any open span (malformed if present).
                open = None;
                let singleton = OpenSpan {
                    class: class.to_string(),
                    start: tok_start,
                    probs: vec![prob],
                };
                emit(singleton, tok_end, source_text, threshold, &mut out);
            }
            Some(_) => {
                // Unreachable: parse_tag only returns B/I/E/S.
                open = None;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical 33-label fixture matching the Viterbi tests.
    fn id2label_fixture() -> Vec<String> {
        [
            "O",
            "B-account_number",
            "I-account_number",
            "E-account_number",
            "S-account_number",
            "B-private_address",
            "I-private_address",
            "E-private_address",
            "S-private_address",
            "B-private_date",
            "I-private_date",
            "E-private_date",
            "S-private_date",
            "B-private_email",
            "I-private_email",
            "E-private_email",
            "S-private_email",
            "B-private_person",
            "I-private_person",
            "E-private_person",
            "S-private_person",
            "B-private_phone",
            "I-private_phone",
            "E-private_phone",
            "S-private_phone",
            "B-private_url",
            "I-private_url",
            "E-private_url",
            "S-private_url",
            "B-secret",
            "I-secret",
            "E-secret",
            "S-secret",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn lid(tag: &str, id2label: &[String]) -> usize {
        id2label.iter().position(|t| t == tag).unwrap()
    }

    #[test]
    fn extracts_b_i_e_span_with_mean_score() {
        let id2label = id2label_fixture();
        // Source crafted so the offsets in the spec line up: "Alice" at
        // bytes 7..12 and "Smith" at bytes 13..18.
        let source = "Hi I m Alice Smith ok";
        let tags = vec![
            lid("O", &id2label),                // "Hi"    0..2
            lid("O", &id2label),                // "I"     3..4
            lid("O", &id2label),                // "m"     5..6
            lid("B-private_person", &id2label), // "Alice" 7..12
            lid("E-private_person", &id2label), // "Smith" 13..18
            lid("O", &id2label),                // "ok"    19..21
        ];
        let probs = vec![0.99, 0.99, 0.99, 0.95, 0.97, 0.99];
        let offsets = vec![(0, 2), (3, 4), (5, 6), (7, 12), (13, 18), (19, 21)];
        let entities = extract_spans(&tags, &id2label, &probs, &offsets, source, 0.5);
        assert_eq!(entities.len(), 1);
        let e = &entities[0];
        assert_eq!(e.label, "private_person");
        assert_eq!(e.start, 7);
        assert_eq!(e.end, 18);
        // mean of (0.95, 0.97) = 0.96
        assert!((e.score - 0.96).abs() < 1e-5);
        assert_eq!(e.text, "Alice Smith");
    }

    #[test]
    fn extracts_s_singleton_span() {
        let id2label = id2label_fixture();
        // "call 555-1234" — phone at bytes (5, 13).
        let source = "call 555-1234";
        let tags = vec![
            lid("O", &id2label),               // "call"      0..4
            lid("S-private_phone", &id2label), // "555-1234"  5..13
        ];
        let probs = vec![0.99, 0.88];
        let offsets = vec![(0, 4), (5, 13)];
        let entities = extract_spans(&tags, &id2label, &probs, &offsets, source, 0.5);
        assert_eq!(entities.len(), 1);
        let e = &entities[0];
        assert_eq!(e.label, "private_phone");
        assert_eq!(e.start, 5);
        assert_eq!(e.end, 13);
        assert!((e.score - 0.88).abs() < 1e-5);
        assert_eq!(e.text, "555-1234");
    }

    #[test]
    fn excludes_below_threshold() {
        let id2label = id2label_fixture();
        let source = "Hi Alice Smith";
        let tags = vec![
            lid("O", &id2label),                // "Hi"    0..2
            lid("B-private_person", &id2label), // "Alice" 3..8
            lid("E-private_person", &id2label), // "Smith" 9..14
        ];
        // mean = 0.4 — below threshold 0.5
        let probs = vec![0.99, 0.35, 0.45];
        let offsets = vec![(0, 2), (3, 8), (9, 14)];
        let entities = extract_spans(&tags, &id2label, &probs, &offsets, source, 0.5);
        assert!(
            entities.is_empty(),
            "expected no entities, got {:?}",
            entities
        );
    }

    #[test]
    fn extracts_multiple_spans() {
        let id2label = id2label_fixture();
        // "Alice Smith called 555-1234"
        //  A l i c e   S m i t h   c a l l e d   5 5 5 - 1 2 3 4
        //  0 1 2 3 4 5 6 7 8 9 ...
        let source = "Alice Smith called 555-1234";
        let tags = vec![
            lid("B-private_person", &id2label), // "Alice"   0..5
            lid("E-private_person", &id2label), // "Smith"   6..11
            lid("O", &id2label),                // "called"  12..18
            lid("S-private_phone", &id2label),  // "555-1234" 19..27
        ];
        let probs = vec![0.9, 0.92, 0.99, 0.85];
        let offsets = vec![(0, 5), (6, 11), (12, 18), (19, 27)];
        let entities = extract_spans(&tags, &id2label, &probs, &offsets, source, 0.5);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].label, "private_person");
        assert_eq!(entities[0].start, 0);
        assert_eq!(entities[0].end, 11);
        assert_eq!(entities[0].text, "Alice Smith");
        assert_eq!(entities[1].label, "private_phone");
        assert_eq!(entities[1].start, 19);
        assert_eq!(entities[1].end, 27);
        assert_eq!(entities[1].text, "555-1234");
    }

    #[test]
    fn handles_b_then_o_malformed_gracefully() {
        let id2label = id2label_fixture();
        let source = "Hi Alice ok";
        let tags = vec![
            lid("O", &id2label),                // "Hi"    0..2
            lid("B-private_person", &id2label), // "Alice" 3..8 (B without E)
            lid("O", &id2label),                // "ok"    9..11
        ];
        let probs = vec![0.99, 0.95, 0.99];
        let offsets = vec![(0, 2), (3, 8), (9, 11)];
        let entities = extract_spans(&tags, &id2label, &probs, &offsets, source, 0.5);
        assert!(
            entities.is_empty(),
            "expected no entities, got {:?}",
            entities
        );
    }
}
