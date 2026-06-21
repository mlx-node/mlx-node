//! Model-neutral vision merge contract.
//!
//! The qwen3.5 dense and MoE families share one image->text merge pipeline:
//! the vision encoder produces image features, those features are scattered
//! into the text token embeddings at the image-placeholder positions, and
//! M-RoPE position ids are computed for the mixed image+text sequence. The
//! product of that pipeline is [`VisionMerge`] — the embeddings + positions
//! that feed the language-model paged prefill.

use crate::array::MxArray;

/// The output of the vision encode + image-feature merge + M-RoPE position
/// computation: everything the language-model paged prefill needs for an
/// image-bearing turn.
pub(crate) struct VisionMerge {
    /// Token embeddings with image features scattered into the image
    /// placeholder slots. Shape `[1, seq_len, hidden]`, text-embedding dtype.
    pub inputs_embeds: MxArray,
    /// M-RoPE position ids for the mixed image+text sequence. Shape
    /// `[3, 1, seq_len]` (the temporal/height/width axes), i32.
    pub position_ids: MxArray,
}
