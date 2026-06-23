//! Blockwise bidirectional attention overlay for unified Gemma 4 vision
//! prefill.
//!
//! Ports mlx-vlm `gemma4/language.py::_block_sequence_ids_for_mask` and
//! `_apply_blockwise_bidirectional_overlay`. During a unified-vision prefill the
//! training-time invariant `use_bidirectional_attention == "vision"` makes every
//! contiguous image-token run attend to itself bidirectionally, on top of the
//! base causal/sliding mask. Text positions stay causal.
//!
//! Masks here follow MLX boolean semantics (`true = keep`), matching
//! [`create_sliding_mask`] and `crate::array::mask::create_causal_mask`. The
//! overlay is `base_mask | same_block`, exactly as the reference.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

/// Whether the unified-vision bidirectional overlay should be applied for a
/// prefill.
///
/// Ports the mlx-vlm gate (`gemma4/language.py::_make_masks`): the vision block
/// overlay is enabled only for a unified checkpoint trained with
/// `use_bidirectional_attention == "vision"`, on a real prefill (`seq_len > 1`),
/// when image tokens are present AND no audio tokens are present. Mixed
/// image+audio prompts stay PURELY causal — audio spans are sequential and the
/// vision block overlay would otherwise dominate. With audio never in the stream
/// today this only narrows on `has_audio`; it is wired now for correctness.
pub fn vision_overlay_active(
    is_unified: bool,
    use_bidirectional_vision: bool,
    has_image: bool,
    has_audio: bool,
    seq_len: usize,
) -> bool {
    is_unified && use_bidirectional_vision && has_image && !has_audio && seq_len > 1
}

/// Per-position image-token indicator for the expanded prompt: `1` where the
/// token equals `image_token_id`, else `0`. Shaped `[1, seq_len]` int32.
///
/// This is the `mm_token_type_ids` the overlay consumes. Video (2) and audio
/// (3) are not produced in the image-only unified path; only image (1) is set.
pub fn build_image_token_type_ids(tokens: &[u32], image_token_id: u32) -> Result<MxArray> {
    let ids: Vec<i32> = tokens
        .iter()
        .map(|&t| i32::from(t == image_token_id))
        .collect();
    MxArray::from_int32(&ids, &[1, ids.len() as i64])
}

/// `_block_sequence_ids_for_mask`: assign each contiguous image run a 0-based
/// group id; non-image positions are `-1`.
///
/// `mm_token_type_ids` is `[1, seq_len]` int32 (1 = image). Returns `[1, seq_len]`
/// int32 block ids.
pub fn block_sequence_ids(mm_token_type_ids: &MxArray) -> Result<MxArray> {
    let mm = mm_token_type_ids.astype(DType::Int32)?;
    // is_vision = (type == 1) | (type == 2). Image-only path uses 1; keep 2 for
    // faithfulness even though the builder never emits it in Phase 2b.
    let one = MxArray::scalar_int(1)?;
    let two = MxArray::scalar_int(2)?;
    let is_one = mm.equal(&one)?;
    let is_two = mm.equal(&two)?;
    let is_vision = is_one.logical_or(&is_two)?;

    let seq_len = mm.shape_at(1)?;
    // prev = is_vision shifted right by 1 (prepend false).
    let prev = if seq_len <= 1 {
        // Nothing to shift; every "previous" is false → starts == is_vision.
        let zeros = MxArray::from_int32(&vec![0; seq_len as usize], &[1, seq_len])?;
        zeros.equal(&one)? // all false, bool dtype
    } else {
        let head_false = MxArray::from_int32(&[0], &[1, 1])?.equal(&one)?; // [1,1] false
        let body = is_vision.slice_axis(1, 0, seq_len - 1)?;
        MxArray::concatenate_many(vec![&head_false, &body], Some(1))?
    };

    // starts = is_vision & !prev
    let not_prev = prev.logical_not()?;
    let starts = is_vision.logical_and(&not_prev)?;

    // group_ids = cumsum(starts as int, axis=1) - 1
    let starts_int = starts.astype(DType::Int32)?;
    let cumsum = starts_int.cumsum(1)?;
    let one_i = MxArray::scalar_int(1)?;
    let group_ids = cumsum.sub(&one_i)?;

    // where(is_vision, group_ids, -1)
    let neg_one = MxArray::scalar_int(-1)?;
    let neg_one_b = neg_one.broadcast_to(&group_ids.shape()?)?;
    is_vision.where_(&group_ids, &neg_one_b)
}

/// `_apply_blockwise_bidirectional_overlay`: force `true` (keep) wherever the
/// query and key sit in the SAME image run, on top of `base_mask`.
///
/// `base_mask` is a boolean keep-mask broadcastable to `[1, 1, L, L]` (true =
/// allowed), `mm_token_type_ids` is `[1, L]`. Returns `base_mask | same_block`.
/// When the type-id length does not match the base mask's key dimension the
/// base mask is returned unchanged (mirrors the reference guard).
pub fn apply_bidirectional_vision_overlay(
    base_mask: &MxArray,
    mm_token_type_ids: &MxArray,
) -> Result<MxArray> {
    let key_len = base_mask.shape_at(base_mask.ndim()? - 1)?;
    if mm_token_type_ids.shape_at(1)? != key_len {
        return Ok(base_mask.clone());
    }

    let block_ids = block_sequence_ids(mm_token_type_ids)?; // [1, L]
    // q_blocks = [1, L, 1], k_blocks = [1, 1, L]
    let q_blocks = block_ids.expand_dims(-1)?;
    let k_blocks = block_ids.expand_dims(-2)?;

    let neg_one = MxArray::scalar_int(-1)?;
    let q_is_vision = q_blocks.not_equal(&neg_one)?;
    let same_idx = q_blocks.equal(&k_blocks)?;
    let same_block = q_is_vision.logical_and(&same_idx)?; // [1, L, L]

    // base_mask is [..., 1, 1, L, L]; same_block needs a head axis → [1, 1, L, L].
    let same_block = same_block.expand_dims(1)?;
    base_mask.logical_or(&same_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_2d_block_ids(arr: &MxArray) -> Vec<i32> {
        let n = arr.shape_at(1).unwrap();
        let a = arr.astype(DType::Int32).unwrap();
        a.eval();
        (0..n)
            .map(|i| a.item_at_int32(i as usize).unwrap())
            .collect()
    }

    /// Read a `[1,1,L,L]` boolean mask into a flat `Vec<bool>` of length L*L
    /// (row-major over q then k).
    fn read_mask(arr: &MxArray, l: i64) -> Vec<bool> {
        let a = arr.astype(DType::Int32).unwrap();
        a.eval();
        (0..(l * l))
            .map(|i| a.item_at_int32(i as usize).unwrap() != 0)
            .collect()
    }

    fn keep(mask: &[bool], l: i64, q: i64, k: i64) -> bool {
        mask[(q * l + k) as usize]
    }

    /// Boolean causal keep-mask `[1,1,L,L]` (true on/below diagonal).
    fn causal_keep_mask(l: i64) -> MxArray {
        let mut data = Vec::with_capacity((l * l) as usize);
        for q in 0..l {
            for k in 0..l {
                data.push(i32::from(k <= q));
            }
        }
        let m = MxArray::from_int32(&data, &[l, l]).unwrap();
        let m = m.astype(DType::Int32).unwrap();
        // != 0 → bool
        let zero = MxArray::scalar_int(0).unwrap();
        m.not_equal(&zero).unwrap().reshape(&[1, 1, l, l]).unwrap()
    }

    #[test]
    fn block_sequence_ids_single_block_numbering() {
        // [0,0,1,1,1,0,0]: one image block at idx 2..4.
        let mm = MxArray::from_int32(&[0, 0, 1, 1, 1, 0, 0], &[1, 7]).unwrap();
        let block = block_sequence_ids(&mm).unwrap();
        assert_eq!(read_2d_block_ids(&block), vec![-1, -1, 0, 0, 0, -1, -1]);
    }

    #[test]
    fn block_sequence_ids_two_blocks_numbering() {
        // Two separate image runs → group ids 0 and 1.
        let mm = MxArray::from_int32(&[0, 1, 1, 0, 1, 1, 1, 0], &[1, 8]).unwrap();
        let block = block_sequence_ids(&mm).unwrap();
        assert_eq!(read_2d_block_ids(&block), vec![-1, 0, 0, -1, 1, 1, 1, -1]);
    }

    #[test]
    fn overlay_makes_image_block_bidirectional_and_keeps_text_causal() {
        let l = 7;
        let mm = MxArray::from_int32(&[0, 0, 1, 1, 1, 0, 0], &[1, l]).unwrap();
        let base = causal_keep_mask(l);
        let out = apply_bidirectional_vision_overlay(&base, &mm).unwrap();
        let mask = read_mask(&out, l);

        // (a) within the image block [2..4], keep in BOTH directions.
        for q in 2..=4 {
            for k in 2..=4 {
                assert!(keep(&mask, l, q, k), "image block ({q},{k}) must be kept");
            }
        }
        // Specifically the upper-triangle image entry 2→4 (future key) is now kept.
        assert!(
            keep(&mask, l, 2, 4),
            "image query 2 must attend forward to key 4"
        );

        // (b) text positions stay causal: text query 0 cannot see future text key 1.
        assert!(
            !keep(&mask, l, 0, 1),
            "text query 0 must NOT see future text key 1"
        );
        assert!(
            keep(&mask, l, 6, 0),
            "text query 6 sees past text key 0 (causal)"
        );

        // (c) an image query cannot attend to a FUTURE text key (block -1).
        assert!(
            !keep(&mask, l, 4, 5),
            "image query 4 must NOT attend future text key 5"
        );
        // (and image query CAN attend a past text key, causal-preserved)
        assert!(keep(&mask, l, 2, 0), "image query 2 sees past text key 0");
    }

    #[test]
    fn overlay_does_not_cross_attend_two_blocks() {
        // [0,1,1,0,1,1,0]: block0 idx1..2, block1 idx4..5.
        let l = 7;
        let mm = MxArray::from_int32(&[0, 1, 1, 0, 1, 1, 0], &[1, l]).unwrap();
        let base = causal_keep_mask(l);
        let out = apply_bidirectional_vision_overlay(&base, &mm).unwrap();
        let mask = read_mask(&out, l);

        // Within block1 (4,5): bidirectional (5→4 already causal; 4→5 newly kept).
        assert!(keep(&mask, l, 4, 5), "block1 internal forward kept");
        // Block1 query must NOT cross-attend to block0 future... block0 is in the
        // past for block1, so causal already keeps 4→1; the cross-block guard is
        // about block0 query attending to FUTURE block1 keys.
        assert!(
            !keep(&mask, l, 1, 4),
            "block0 query 1 must NOT attend future block1 key 4"
        );
        assert!(
            !keep(&mask, l, 2, 5),
            "block0 query 2 must NOT attend future block1 key 5"
        );
    }

    #[test]
    fn overlay_noop_on_length_mismatch() {
        let l = 5;
        let mm = MxArray::from_int32(&[0, 1, 1, 0], &[1, 4]).unwrap(); // wrong length
        let base = causal_keep_mask(l);
        let out = apply_bidirectional_vision_overlay(&base, &mm).unwrap();
        let mask = read_mask(&out, l);
        let base_flat = read_mask(&base, l);
        assert_eq!(mask, base_flat, "length mismatch must be a no-op");
    }

    #[test]
    fn overlay_active_image_only_unified_prefill() {
        // Unified + bidir-vision + image present + no audio + seq>1 → active.
        assert!(vision_overlay_active(true, true, true, false, 7));
    }

    #[test]
    fn overlay_disabled_when_audio_present() {
        // Same image-bearing unified prefill, but with audio tokens also present
        // → overlay OFF (mixed image+audio is causal, audio wins).
        assert!(!vision_overlay_active(true, true, true, true, 7));
    }

    #[test]
    fn overlay_disabled_for_non_image_or_non_unified_or_decode() {
        assert!(
            !vision_overlay_active(true, true, false, false, 7),
            "no image"
        );
        assert!(
            !vision_overlay_active(false, true, true, false, 7),
            "not unified"
        );
        assert!(
            !vision_overlay_active(true, false, true, false, 7),
            "not bidir-vision"
        );
        assert!(
            !vision_overlay_active(true, true, true, false, 1),
            "decode step"
        );
    }

    /// An image block that WOULD be bidirectional under image-only becomes
    /// causal when audio tokens are also present, because the gate returns false
    /// so no overlay type-ids are built (the prefill keeps the pure causal mask).
    #[test]
    fn overlay_gate_keeps_image_block_causal_when_audio_present() {
        let l = 7;
        // Image-only: gate active → overlay applied → image block bidirectional.
        assert!(vision_overlay_active(true, true, true, false, l as usize));
        let mm = MxArray::from_int32(&[0, 0, 1, 1, 1, 0, 0], &[1, l]).unwrap();
        let base = causal_keep_mask(l);
        let with_overlay = apply_bidirectional_vision_overlay(&base, &mm).unwrap();
        assert!(
            keep(&read_mask(&with_overlay, l), l, 2, 4),
            "image-only: query 2 attends forward to key 4"
        );

        // Image+audio: gate inactive → no overlay built → base causal mask wins,
        // so the same image-block forward edge is NOT kept.
        assert!(!vision_overlay_active(true, true, true, true, l as usize));
        let base_flat = read_mask(&base, l);
        assert!(
            !keep(&base_flat, l, 2, 4),
            "image+audio: query 2 must NOT attend forward to key 4 (causal)"
        );
    }

    #[test]
    fn build_image_token_type_ids_marks_image_positions() {
        let tokens: Vec<u32> = vec![10, 258880, 258880, 11, 258880];
        let mm = build_image_token_type_ids(&tokens, 258880).unwrap();
        assert_eq!(read_2d_block_ids(&mm), vec![0, 1, 1, 0, 1]);
    }
}
