use super::*;

const BOS: u32 = 1;
const USER: u32 = 100;
const TEXT: u32 = 200;
const IMG: u32 = IMAGE_TOKEN_ID as u32;

#[test]
fn expands_single_placeholder_per_image_inline() {
    // Template emitted: BOS, USER, <|image_pad|>, TEXT
    // Expected: BOS, USER, <|image_pad|>×5, TEXT  (vision wrapper stays
    // INSIDE the user turn instead of getting spliced after BOS).
    let tokens = vec![BOS, USER, IMG, TEXT];
    let out = inject_image_placeholders(&tokens, &[5]).unwrap();
    assert_eq!(out, vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT]);
}

#[test]
fn expands_distinct_grid_counts_for_multiple_images_in_order() {
    // Two images with different grid sizes — each placeholder must be
    // replaced by its own image's count, not the other way around.
    let tokens = vec![BOS, IMG, TEXT, IMG];
    let out = inject_image_placeholders(&tokens, &[2, 3]).unwrap();
    assert_eq!(out, vec![BOS, IMG, IMG, TEXT, IMG, IMG, IMG]);
}

#[test]
fn empty_counts_is_passthrough() {
    let tokens = vec![BOS, USER, TEXT];
    let out = inject_image_placeholders(&tokens, &[]).unwrap();
    assert_eq!(out, tokens);
}

#[test]
fn rejects_template_that_emits_no_image_markers() {
    let tokens = vec![BOS, USER, TEXT];
    let error = inject_image_placeholders(&tokens, &[3]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("model chat template emitted no image placeholder tokens"),
        "unexpected error: {error}",
    );
    assert!(expanded_image_prompt_len(&tokens, &[3]).is_err());
}

#[test]
fn fully_expanded_input_passes_through_unchanged() {
    // Case 2: template already emitted the full 5-token run. `existing`
    // (5) != `per_image.len()` (1) so the "one-per-image" branch
    // doesn't fire; total (5) matches, so the input is preserved.
    let tokens = vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT];
    let out = inject_image_placeholders(&tokens, &[5]).unwrap();
    assert_eq!(out, tokens);
}

#[test]
fn preserves_relative_position_of_surrounding_tokens() {
    // Regression guard: every non-IMG token must survive in its
    // original relative order.
    let tokens = vec![BOS, USER, 10, 11, IMG, 12, 13];
    let out = inject_image_placeholders(&tokens, &[4]).unwrap();
    assert_eq!(out, vec![BOS, USER, 10, 11, IMG, IMG, IMG, IMG, 12, 13]);
}

#[test]
fn rejects_mismatched_image_marker_count() {
    let tokens = vec![BOS, IMG, IMG, TEXT];
    let error = inject_image_placeholders(&tokens, &[3]).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("emitted 2 image placeholder token(s)"),
        "unexpected error: {message}",
    );
    assert!(
        message.contains("expected 1 unexpanded marker(s) or 3 already-expanded marker(s)"),
        "unexpected error: {message}",
    );
    assert!(expanded_image_prompt_len(&tokens, &[3]).is_err());
}

#[test]
fn non_allocating_length_plan_matches_placeholder_injection() {
    let cases = [
        (vec![BOS, USER, IMG, TEXT], vec![5]),
        (vec![BOS, IMG, TEXT, IMG], vec![2, 3]),
        (vec![BOS, USER, TEXT], Vec::new()),
        (vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT], vec![5]),
    ];

    for (tokens, counts) in cases {
        let planned = expanded_image_prompt_len(&tokens, &counts).expect("plan prompt length");
        let expanded =
            inject_image_placeholders(&tokens, &counts).expect("expand image placeholders");
        assert_eq!(
            planned,
            expanded.len(),
            "tokens={tokens:?}, counts={counts:?}"
        );
    }
}
