//! `get_rope_index` builds M-RoPE position IDs for VLM prefill. The
//! pre-fix implementation collapsed every `IMAGE_TOKEN_ID` in the
//! prompt to a single contiguous span from `positions[0]` to
//! `positions[last]`; a multi-turn history with two image-bearing
//! user turns joined by an assistant reply silently skipped every
//! interior text token, leaving `all_position_ids` shorter than
//! `seq_len` and crashing the downstream reshape with a cryptic
//! "length mismatch". These tests pin per-run indexing against that
//! regression.
use super::*;
use crate::array::MxArray;
use std::sync::Mutex;
use std::sync::OnceLock;
const IMG: i32 = IMAGE_TOKEN_ID;
const TEXT_A: i32 = 100;
const TEXT_B: i32 = 200;

/// MLX's MPS backend is not re-entrant — every test that touches an
/// `MxArray` must hold this mutex so only one such test runs at a
/// time across the test binary.
fn mlx_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Encode a flat `Vec<i32>` token stream as a `[1, seq_len]`
/// `MxArray` and a `[num_images, 3]` grid array (or `None`) to
/// feed `get_rope_index`.
fn mk_inputs(tokens: &[i32], grids: &[(i64, i64, i64)]) -> (MxArray, Option<MxArray>) {
    let seq_len = tokens.len() as i64;
    let input_ids = MxArray::from_int32(tokens, &[1, seq_len]).unwrap();
    let grid = if grids.is_empty() {
        None
    } else {
        let flat: Vec<i32> = grids
            .iter()
            .flat_map(|(t, h, w)| [*t as i32, *h as i32, *w as i32])
            .collect();
        Some(MxArray::from_int32(&flat, &[grids.len() as i64, 3]).unwrap())
    };
    (input_ids, grid)
}

fn extract_positions(pos: &MxArray) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    // pos shape [3, 1, seq_len] — flatten to Vec<i32> and split.
    pos.eval();
    let flat = pos.to_int32().unwrap();
    let n = flat.len() / 3;
    (
        flat[0..n].to_vec(),
        flat[n..2 * n].to_vec(),
        flat[2 * n..3 * n].to_vec(),
    )
}

#[test]
fn pure_text_prompt_gets_sequential_positions() {
    let _g = mlx_lock().lock().unwrap();
    let tokens = vec![TEXT_A, TEXT_B, TEXT_A, TEXT_B];
    let (ids, grid) = mk_inputs(&tokens, &[]);
    let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, h, w) = extract_positions(&pos);
    assert_eq!(t, vec![0, 1, 2, 3]);
    assert_eq!(h, vec![0, 1, 2, 3]);
    assert_eq!(w, vec![0, 1, 2, 3]);
    assert_eq!(rope_deltas, 0);
}

#[test]
fn single_image_run_preserves_baseline_shape() {
    let _g = mlx_lock().lock().unwrap();
    // 2 text + (grid 2x2x2=8 tokens after spatial_merge=2, so t=2,h=4,w=4, split=2
    //   → llm grid 2×2×2 = 8 image tokens) + 2 text
    let tokens: Vec<i32> = [TEXT_A, TEXT_B]
        .iter()
        .chain(std::iter::repeat_n(&IMG, 8))
        .chain([TEXT_A, TEXT_B].iter())
        .copied()
        .collect();
    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
    let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, h, w) = extract_positions(&pos);
    // Leading text
    assert_eq!(&t[..2], &[0, 1]);
    assert_eq!(&h[..2], &[0, 1]);
    // Image span starts at 2; with llm_grid_t=2 h=2 w=2 → max_axis=1
    // so current_pos advances to 2 + 1 + 1 = 4 after the image.
    // Trailing text: 4, 5
    assert_eq!(&t[10..], &[4, 5]);
    assert_eq!(&h[10..], &[4, 5]);
    assert_eq!(&w[10..], &[4, 5]);

    // The image run compresses 8 placeholder tokens into 4 distinct
    // temporal positions, so the running M-RoPE counter lags the physical
    // sequence length: `rope_deltas = max_position + 1 - seq_len` MUST be
    // negative. This is the per-session delta the paged decode/warm-
    // continuation path adds to the physical KV slot to recover the
    // compressed rotation position; previously it was dropped, leaving
    // image-turn decode rotating ~|delta| positions too far ahead.
    let max_position = *t.iter().max().unwrap() as i64; // temporal axis (axis 0)
    let seq_len = tokens.len() as i64;
    assert_eq!(rope_deltas, max_position + 1 - seq_len);
    assert!(
        rope_deltas < 0,
        "image prefill must compress positions (rope_deltas={rope_deltas})"
    );
    // 2 text + 8 image (compressed to positions 2..=3) + 2 text → max
    // temporal position 5 over 12 tokens → delta = 5 + 1 - 12 = -6.
    assert_eq!(rope_deltas, -6);
}

#[test]
fn image_final_prompt_delta_uses_global_max_axis() {
    // An image-FINAL prompt (no trailing text) exposes which axis feeds the
    // decode delta: the spatial (h, w) axes outrun the temporal one, so the
    // global max M-RoPE position lives on a spatial axis. The delta must use
    // that global max (mlx-vlm `llm_positions.max()`), NOT the temporal axis
    // alone — otherwise the first generated token rotates at a position
    // INSIDE the image's spatial range instead of at global_max + 1.
    let _g = mlx_lock().lock().unwrap();
    // 1 text + (grid 1x4x4, spatial_merge=2 → llm grid t=1,h=2,w=2 = 4 image
    // tokens) and NOTHING after the image.
    let tokens: Vec<i32> = std::iter::once(TEXT_A)
        .chain(std::iter::repeat_n(IMG, 4))
        .collect();
    let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4)]);
    let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, h, w) = extract_positions(&pos);

    let t_max = *t.iter().max().unwrap() as i64;
    let global_max = *t.iter().chain(&h).chain(&w).max().unwrap() as i64;
    let seq_len = tokens.len() as i64;

    // The spatial axes must outrun the temporal one here, else the test
    // would not distinguish the global-max fix from the axis-0 regression.
    assert!(
        global_max > t_max,
        "test grid is not asymmetric: global_max={global_max} t_max={t_max}"
    );
    // The delta references the GLOBAL max, not the temporal-axis max.
    assert_eq!(rope_deltas, global_max + 1 - seq_len);
    assert_ne!(
        rope_deltas,
        t_max + 1 - seq_len,
        "delta must not use the temporal axis alone (axis-0 regression)"
    );
}

#[test]
fn two_image_runs_separated_by_text_emits_every_position() {
    // Two image runs separated by interior text must emit a position for
    // EVERY token; a dropped interior-text position makes the downstream
    // reshape in get_rope_index fail with a length mismatch.
    let _g = mlx_lock().lock().unwrap();
    let mut tokens: Vec<i32> = Vec::new();
    tokens.push(TEXT_A); // position 0
    tokens.extend(std::iter::repeat_n(IMG, 8)); // 1 image → llm 2×2×2=8
    tokens.push(TEXT_A); // interior text between images
    tokens.push(TEXT_B);
    tokens.extend(std::iter::repeat_n(IMG, 8)); // 2nd image → same grid
    tokens.push(TEXT_A); // trailing text

    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
    let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, _h, _w) = extract_positions(&pos);

    // seq_len == tokens.len() — every token must have a position;
    // dropping the interior text entries fails the reshape at the end
    // of get_rope_index.
    assert_eq!(
        t.len(),
        tokens.len(),
        "position count must equal token count"
    );

    // Leading text at pos 0
    assert_eq!(t[0], 0);
    // Image 1 at base=1, max_axis=1 → current_pos after = 3
    // Interior text: 3, 4
    assert_eq!(t[9], 3);
    assert_eq!(t[10], 4);
    // Image 2 at base=5, max_axis=1 → current_pos after = 7
    // Trailing text: 7
    assert_eq!(*t.last().unwrap(), 7);
}

#[test]
fn leading_image_run_no_text_prefix() {
    let _g = mlx_lock().lock().unwrap();
    let tokens: Vec<i32> = std::iter::repeat_n(IMG, 8)
        .chain([TEXT_A, TEXT_B].iter().copied())
        .collect();
    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
    let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, _, _) = extract_positions(&pos);
    assert_eq!(t.len(), tokens.len());
    // Image at base=0, max_axis=1 → current_pos=2 after, trailing text 2, 3
    assert_eq!(&t[8..], &[2, 3]);
}

#[test]
fn trailing_image_run_no_text_suffix() {
    let _g = mlx_lock().lock().unwrap();
    let tokens: Vec<i32> = [TEXT_A, TEXT_B]
        .iter()
        .copied()
        .chain(std::iter::repeat_n(IMG, 8))
        .collect();
    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
    let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
    let (t, _, _) = extract_positions(&pos);
    assert_eq!(t.len(), tokens.len());
    assert_eq!(&t[..2], &[0, 1]);
}

#[test]
fn run_count_must_match_image_count() {
    // 2 image runs in the prompt but only 1 grid supplied — ambiguous
    // pairing; reject.
    let _g = mlx_lock().lock().unwrap();
    let mut tokens = vec![TEXT_A];
    tokens.extend(std::iter::repeat_n(IMG, 4));
    tokens.push(TEXT_A);
    tokens.extend(std::iter::repeat_n(IMG, 4));
    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
    let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
        Ok(_) => panic!("expected get_rope_index to error"),
        Err(e) => e,
    };
    assert!(
        err.reason.contains("Image run layout mismatch"),
        "got: {}",
        err.reason
    );
}

#[test]
fn per_run_length_must_match_its_grid_count() {
    // 2 runs, 2 grids — but run 0 has too few tokens for grid 0.
    let _g = mlx_lock().lock().unwrap();
    let mut tokens = vec![TEXT_A];
    tokens.extend(std::iter::repeat_n(IMG, 4)); // should be 8 for (2,4,4)
    tokens.push(TEXT_A);
    tokens.extend(std::iter::repeat_n(IMG, 12)); // compensates the total, but per-run wrong
    let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
    let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
        Ok(_) => panic!("expected get_rope_index to error"),
        Err(e) => e,
    };
    assert!(err.reason.contains("Image run 0"), "got: {}", err.reason);
}

#[test]
fn multi_image_already_expanded_single_contiguous_run_is_accepted() {
    // A checkpoint template may emit one fully expanded contiguous
    // run whose length is `sum(per_image_counts)`. The path
    // synthesises per-image sub-run offsets from the shared span and
    // emits correct M-RoPE positions for each image.
    let _g = mlx_lock().lock().unwrap();
    // Two 1×2×2 grids → 4 image tokens each, 8 total.
    let mut tokens = vec![TEXT_A];
    tokens.extend(std::iter::repeat_n(IMG, 8));
    tokens.push(TEXT_B);
    let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 4, 4)]);
    let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
        .expect("already-expanded single-run layout for two images must be accepted");
    let (t, _, _) = extract_positions(&pos);
    assert_eq!(t.len(), tokens.len(), "every token must have a position");
    // Leading text at 0.
    assert_eq!(t[0], 0);
    // First image base = 1, llm grid 1×2×2, max_axis=1 → current_pos
    // after = 3. Next image base = 3, max_axis=1 → current_pos
    // after = 5. Trailing TEXT_B at 5.
    assert_eq!(*t.last().unwrap(), 5);
}

#[test]
fn multi_image_already_expanded_distinct_grids_preserve_per_image_offsets() {
    // Same already-expanded shape but with different grid sizes per
    // image. The synthesised sub-run offsets must distribute the
    // shared span correctly.
    let _g = mlx_lock().lock().unwrap();
    // image 0: 1×2×2 → 4 tokens. image 1: 1×4×4 → 16 tokens. Total 20.
    let mut tokens = vec![TEXT_A];
    tokens.extend(std::iter::repeat_n(IMG, 20));
    tokens.push(TEXT_B);
    let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 8, 8)]);
    let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
        .expect("already-expanded layout with distinct per-image grids must succeed");
    let (t, _, _) = extract_positions(&pos);
    assert_eq!(t.len(), tokens.len());
    assert_eq!(t[0], 0);
    // image 0 base=1, max_axis = max(0,1,1) = 1 → current_pos = 3
    // image 1 base=3, max_axis = max(0,3,3) = 3 → current_pos = 7
    // Trailing TEXT_B at 7.
    assert_eq!(*t.last().unwrap(), 7);
}
