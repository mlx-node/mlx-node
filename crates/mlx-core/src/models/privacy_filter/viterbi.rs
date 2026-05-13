//! BIOES Viterbi decoder for the privacy-filter token classifier.
//!
//! Pure-Rust implementation operating on emission scores produced by the
//! classifier head. The transition matrix encodes BIOES legality:
//!
//! - `O   → O | B-* | S-*`
//! - `B-X → I-X | E-X`           (same class only)
//! - `I-X → I-X | E-X`           (same class only)
//! - `E-X → O | B-* | S-*`
//! - `S-X → O | B-* | S-*`
//!
//! Six configurable scalar biases are added to the corresponding legal
//! transitions for runtime precision/recall tuning. Illegal transitions
//! remain `f32::NEG_INFINITY`.

use serde::Deserialize;

/// Calibration biases applied to legal BIOES transitions.
///
/// Mirrors the `operating_points.<name>.biases` block in
/// `viterbi_calibration.json`. All fields default to `0.0`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Calibration {
    /// Bias for `O → O`.
    #[serde(default)]
    pub transition_bias_background_stay: f32,
    /// Bias for `O → {B-*, S-*}`.
    #[serde(default)]
    pub transition_bias_background_to_start: f32,
    /// Bias for `{E-*, S-*} → O`.
    #[serde(default)]
    pub transition_bias_end_to_background: f32,
    /// Bias for `{E-*, S-*} → {B-*, S-*}`.
    #[serde(default)]
    pub transition_bias_end_to_start: f32,
    /// Bias for `{B-X, I-X} → I-X` (same-class continue).
    #[serde(default)]
    pub transition_bias_inside_to_continue: f32,
    /// Bias for `{B-X, I-X} → E-X` (same-class end).
    #[serde(default)]
    pub transition_bias_inside_to_end: f32,
}

/// Parsed BIOES tag: prefix plus optional class name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prefix {
    O,
    B,
    I,
    E,
    S,
}

#[derive(Debug, Clone, Copy)]
struct ParsedTag<'a> {
    prefix: Prefix,
    /// `None` for `O`, `Some("private_email")` for `B-private_email`, etc.
    class: Option<&'a str>,
}

fn parse_tag(tag: &str) -> ParsedTag<'_> {
    if tag == "O" {
        return ParsedTag {
            prefix: Prefix::O,
            class: None,
        };
    }
    // BIOES tags are `<prefix>-<class>`; the class may itself contain `-` or
    // `_` (e.g. `private_email`), so we only split on the first `-`.
    let mut iter = tag.splitn(2, '-');
    let head = iter.next().unwrap_or("");
    let class = iter.next();
    let prefix = match head {
        "B" => Prefix::B,
        "I" => Prefix::I,
        "E" => Prefix::E,
        "S" => Prefix::S,
        other => panic!("invalid BIOES tag prefix: {other:?} (tag: {tag:?})"),
    };
    let class = class.unwrap_or_else(|| panic!("BIOES tag missing class: {tag:?}"));
    ParsedTag {
        prefix,
        class: Some(class),
    }
}

/// Classifies a transition and returns `Some(bias)` if legal, `None` otherwise.
fn classify_transition(prev: ParsedTag<'_>, next: ParsedTag<'_>, cal: &Calibration) -> Option<f32> {
    use Prefix::*;
    match (prev.prefix, next.prefix) {
        // O → O
        (O, O) => Some(cal.transition_bias_background_stay),
        // O → {B, S}
        (O, B) | (O, S) => Some(cal.transition_bias_background_to_start),
        // {E, S} → O
        (E, O) | (S, O) => Some(cal.transition_bias_end_to_background),
        // {E, S} → {B, S}
        (E, B) | (E, S) | (S, B) | (S, S) => Some(cal.transition_bias_end_to_start),
        // {B, I} → I (same class only)
        (B, I) | (I, I) if prev.class == next.class => Some(cal.transition_bias_inside_to_continue),
        // {B, I} → E (same class only)
        (B, E) | (I, E) if prev.class == next.class => Some(cal.transition_bias_inside_to_end),
        // Everything else is illegal.
        _ => None,
    }
}

/// Builds the `[num_labels, num_labels]` transition matrix.
///
/// Legal transitions receive the calibration bias for their category. Illegal
/// transitions are filled with `f32::NEG_INFINITY`.
pub fn build_transition_matrix(id2label: &[String], cal: &Calibration) -> Vec<Vec<f32>> {
    let parsed: Vec<ParsedTag<'_>> = id2label.iter().map(|t| parse_tag(t.as_str())).collect();
    let n = id2label.len();
    let mut t = vec![vec![f32::NEG_INFINITY; n]; n];
    for i in 0..n {
        for j in 0..n {
            if let Some(bias) = classify_transition(parsed[i], parsed[j], cal) {
                t[i][j] = bias;
            }
        }
    }
    t
}

/// Returns the label id for the given tag string. Panics if the tag is missing.
pub fn label_id(id2label: &[String], tag: &str) -> usize {
    id2label
        .iter()
        .position(|t| t == tag)
        .unwrap_or_else(|| panic!("tag not found in id2label: {tag:?}"))
}

/// Standard Viterbi decoder.
///
/// - `emit` is shape `[T, num_labels]`.
/// - `transitions` is shape `[num_labels, num_labels]`, with
///   `transitions[i][j]` the score for moving from state `i` to state `j`.
///
/// The initial step uses a virtual-O predecessor:
/// `dp[0][j] = emit[0][j] + transitions[O][j]`.
///
/// Returns the most-likely tag sequence as label indices. Returns an empty
/// vector for empty input.
pub fn viterbi_decode(emit: &[Vec<f32>], transitions: &[Vec<f32>]) -> Vec<usize> {
    let t_len = emit.len();
    if t_len == 0 {
        return Vec::new();
    }
    let n = emit[0].len();
    assert!(
        transitions.len() == n && transitions.iter().all(|row| row.len() == n),
        "transitions must be [num_labels, num_labels]"
    );

    // The virtual-O predecessor uses row 0 of `transitions` (O is label 0).
    const O_ID: usize = 0;

    let mut dp = vec![vec![f32::NEG_INFINITY; n]; t_len];
    // `back[t][j]` is `Some(i)` when reachable, else `None`.
    let mut back: Vec<Vec<Option<usize>>> = vec![vec![None; n]; t_len];

    // Initial step.
    for j in 0..n {
        let score = transitions[O_ID][j] + emit[0][j];
        dp[0][j] = score;
        // No predecessor for the first step.
    }

    // Recurrence.
    for t in 1..t_len {
        for j in 0..n {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev: Option<usize> = None;
            for i in 0..n {
                let prev = dp[t - 1][i];
                if !prev.is_finite() {
                    continue;
                }
                let trans = transitions[i][j];
                if !trans.is_finite() {
                    continue;
                }
                let score = prev + trans;
                if score > best_score {
                    best_score = score;
                    best_prev = Some(i);
                }
            }
            if best_prev.is_some() {
                dp[t][j] = best_score + emit[t][j];
                back[t][j] = best_prev;
            }
            // Else unreachable: leave dp[t][j] = -inf and back[t][j] = None.
        }
    }

    // Backtrace from argmax_j dp[T-1][j].
    let mut last = 0usize;
    let mut last_score = f32::NEG_INFINITY;
    for (j, &score) in dp[t_len - 1].iter().enumerate() {
        if score > last_score {
            last_score = score;
            last = j;
        }
    }

    let mut path = vec![0usize; t_len];
    path[t_len - 1] = last;
    for t in (1..t_len).rev() {
        // If `back[t][path[t]]` is None the path is unreachable; this should
        // not happen for the argmax tail when at least one initial state is
        // finite, but fall back to O to keep the function total.
        path[t - 1] = back[t][path[t]].unwrap_or(O_ID);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn transition_legality_o_to_b_allowed() {
        let id2label = id2label_fixture();
        let t = build_transition_matrix(&id2label, &Calibration::default());
        let o = label_id(&id2label, "O");
        let b_email = label_id(&id2label, "B-private_email");
        assert!(
            t[o][b_email].is_finite(),
            "O → B-private_email should be finite"
        );
    }

    #[test]
    fn transition_legality_o_to_i_forbidden() {
        let id2label = id2label_fixture();
        let t = build_transition_matrix(&id2label, &Calibration::default());
        let o = label_id(&id2label, "O");
        let i_email = label_id(&id2label, "I-private_email");
        assert_eq!(
            t[o][i_email],
            f32::NEG_INFINITY,
            "O → I-private_email should be -inf"
        );
    }

    #[test]
    fn transition_legality_b_to_i_same_class_allowed_other_class_forbidden() {
        let id2label = id2label_fixture();
        let t = build_transition_matrix(&id2label, &Calibration::default());
        let b_email = label_id(&id2label, "B-private_email");
        let i_email = label_id(&id2label, "I-private_email");
        let i_phone = label_id(&id2label, "I-private_phone");
        assert!(
            t[b_email][i_email].is_finite(),
            "B-private_email → I-private_email should be finite"
        );
        assert_eq!(
            t[b_email][i_phone],
            f32::NEG_INFINITY,
            "B-private_email → I-private_phone should be -inf (cross-class)"
        );
    }

    #[test]
    fn calibration_bias_applied_to_inside_continue() {
        let cal = Calibration {
            transition_bias_inside_to_continue: 1.5,
            ..Calibration::default()
        };
        let t = build_transition_matrix(&id2label_fixture(), &cal);
        let b_email = label_id(&id2label_fixture(), "B-private_email");
        let i_email = label_id(&id2label_fixture(), "I-private_email");
        assert!((t[b_email][i_email] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn viterbi_decodes_known_trace() {
        let id2label = id2label_fixture();
        let n = id2label.len();
        let t = build_transition_matrix(&id2label, &Calibration::default());

        let o = label_id(&id2label, "O");
        let b_email = label_id(&id2label, "B-private_email");
        let i_email = label_id(&id2label, "I-private_email");
        let e_email = label_id(&id2label, "E-private_email");

        let target = [o, b_email, i_email, e_email, o];
        let mut emit: Vec<Vec<f32>> = vec![vec![0.0; n]; target.len()];
        for (t_idx, &lab) in target.iter().enumerate() {
            emit[t_idx][lab] = 10.0;
        }

        let path = viterbi_decode(&emit, &t);
        assert_eq!(path, target.to_vec());
    }
}
