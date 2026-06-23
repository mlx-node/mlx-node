//! Encoder-free audio feature extraction for unified Gemma 4.
//!
//! Ports the raw-window path of mlx-vlm
//! `processing_gemma4_unified.py::Gemma4UnifiedAudioFeatureExtractor`. The
//! unified audio model has NO mel/FFT/spectrogram front-end: a decoded mono
//! float32 PCM waveform (already 16 kHz) is zero-padded to a multiple of
//! `audio_samples_per_token` (640) and reshaped into `[n_frames, 640]` raw
//! windows. One frame = one audio token = 40 ms @ 16 kHz. No scaling, no
//! normalization, no preemphasis — samples pass through untouched.

use napi::bindgen_prelude::*;

use crate::array::MxArray;

/// Default raw samples per audio token (640 = 40 ms @ 16 kHz).
pub const DEFAULT_AUDIO_SAMPLES_PER_TOKEN: usize = 640;

/// Build the encoder-free audio frame tensor from a mono float32 PCM waveform.
///
/// Steps (mlx-vlm `_extract_waveform_features`):
/// 1. zero-pad right to a multiple of `samples_per_token`,
/// 2. reshape to `[n_frames, samples_per_token]`,
/// 3. NO scaling/normalization — samples pass through as f32.
///
/// `n_frames = ceil(pcm.len() / samples_per_token)`. An empty waveform yields a
/// `[0, samples_per_token]` tensor (no frames). Returns the `[n_frames, S]` f32
/// `MxArray`.
pub fn frames_from_pcm(pcm: &[f32], samples_per_token: usize) -> Result<MxArray> {
    if samples_per_token == 0 {
        return Err(Error::from_reason(
            "frames_from_pcm: samples_per_token must be > 0",
        ));
    }

    let n = pcm.len();
    let pad = (samples_per_token - (n % samples_per_token)) % samples_per_token;
    let n_frames = (n + pad) / samples_per_token;

    // Pad-right with zeros to a whole number of frames. A zero-length waveform
    // produces zero frames (no padding needed).
    let mut data = Vec::with_capacity(n + pad);
    data.extend_from_slice(pcm);
    data.resize(n + pad, 0.0);

    MxArray::from_float32(&data, &[n_frames as i64, samples_per_token as i64])
}

/// Expand each audio placeholder token into the full audio span the merge
/// expects: `boa + audio_token × n_frames + eoa`.
///
/// Mirrors `expand_image_tokens` (BOI + N×image + EOI). `n_frames_per_audio`
/// supplies the frame count for each placeholder in order; its length must match
/// the number of `audio_token_id` occurrences in `tokens`. Non-placeholder
/// tokens pass through unchanged.
pub fn expand_audio_tokens(
    tokens: &[u32],
    n_frames_per_audio: &[usize],
    audio_token_id: u32,
    boa_token_id: u32,
    eoa_token_id: u32,
) -> Result<Vec<u32>> {
    let placeholder_count = tokens.iter().filter(|&&t| t == audio_token_id).count();
    if placeholder_count != n_frames_per_audio.len() {
        return Err(Error::from_reason(format!(
            "expand_audio_tokens: {} audio placeholder(s) but {} frame count(s) supplied",
            placeholder_count,
            n_frames_per_audio.len()
        )));
    }

    let total_frames: usize = n_frames_per_audio.iter().sum();
    let mut result = Vec::with_capacity(tokens.len() + total_frames + 2 * placeholder_count);
    let mut audio_idx = 0usize;
    for &t in tokens {
        if t == audio_token_id {
            let n_frames = n_frames_per_audio[audio_idx];
            result.push(boa_token_id);
            for _ in 0..n_frames {
                result.push(audio_token_id);
            }
            result.push(eoa_token_id);
            audio_idx += 1;
        } else {
            result.push(t);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn dims(arr: &MxArray) -> Vec<i64> {
        let nd = arr.ndim().unwrap();
        (0..nd).map(|i| arr.shape_at(i).unwrap()).collect()
    }

    fn read_frames(arr: &MxArray, n_frames: i64, s: i64) -> Vec<f32> {
        let a = arr.astype(DType::Float32).unwrap();
        a.eval();
        (0..(n_frames * s))
            .map(|i| a.item_at_float32(i as usize).unwrap())
            .collect()
    }

    #[test]
    fn frames_exact_multiple_one_frame() {
        let pcm: Vec<f32> = (0..640).map(|i| i as f32).collect();
        let frames = frames_from_pcm(&pcm, 640).unwrap();
        assert_eq!(dims(&frames), vec![1, 640]);
        let flat = read_frames(&frames, 1, 640);
        // No scaling: sample x reads back as x.
        assert_eq!(flat[0], 0.0);
        assert_eq!(flat[639], 639.0);
    }

    #[test]
    fn frames_pads_partial_tail_with_zeros() {
        // N=641 → pad to 1280 → 2 frames; tail (positions 641..1279) zero-filled.
        let pcm: Vec<f32> = (0..641).map(|i| (i + 1) as f32).collect();
        let frames = frames_from_pcm(&pcm, 640).unwrap();
        assert_eq!(dims(&frames), vec![2, 640]);
        let flat = read_frames(&frames, 2, 640);
        // Real samples pass through unscaled.
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[640], 641.0); // first sample of row 2 = the 641st sample
        // Everything after the real samples is zero-padding.
        for &v in &flat[641..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn frames_two_full_frames() {
        let pcm: Vec<f32> = vec![0.5; 1280];
        let frames = frames_from_pcm(&pcm, 640).unwrap();
        assert_eq!(dims(&frames), vec![2, 640]);
        let flat = read_frames(&frames, 2, 640);
        assert!(flat.iter().all(|&v| v == 0.5), "no scaling applied");
    }

    #[test]
    fn frames_empty_waveform_zero_frames() {
        let frames = frames_from_pcm(&[], 640).unwrap();
        assert_eq!(dims(&frames), vec![0, 640]);
    }

    #[test]
    fn expand_audio_one_placeholder() {
        // [..., AUDIO, ...] with n=3 → [..., BOA, A, A, A, EOA, ...].
        let tokens: Vec<u32> = vec![10, 258881, 11];
        let out = expand_audio_tokens(&tokens, &[3], 258881, 256000, 258883).unwrap();
        assert_eq!(out, vec![10, 256000, 258881, 258881, 258881, 258883, 11]);
        // Exactly n audio tokens between the markers.
        let audio_count = out.iter().filter(|&&t| t == 258881).count();
        assert_eq!(audio_count, 3);
    }

    #[test]
    fn expand_audio_two_placeholders() {
        let tokens: Vec<u32> = vec![258881, 7, 258881];
        let out = expand_audio_tokens(&tokens, &[1, 2], 258881, 256000, 258883).unwrap();
        assert_eq!(
            out,
            vec![256000, 258881, 258883, 7, 256000, 258881, 258881, 258883]
        );
    }

    #[test]
    fn expand_audio_count_mismatch_errors() {
        let tokens: Vec<u32> = vec![258881];
        // 1 placeholder but 2 frame counts → error.
        assert!(expand_audio_tokens(&tokens, &[1, 2], 258881, 256000, 258883).is_err());
    }

    #[test]
    fn expand_audio_no_placeholder_passthrough() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let out = expand_audio_tokens(&tokens, &[], 258881, 256000, 258883).unwrap();
        assert_eq!(out, tokens);
    }

    #[test]
    fn expand_audio_zero_frames_placeholder() {
        // Empty audio (0 frames) expands to BOA+EOA with NO audio-token positions.
        // Paired with `frames_empty_waveform_zero_frames` (features [0,640]), this
        // gives mask_count == feature_count == 0, which `build_gemma4_audio_embeds`
        // short-circuits before the (modulo-zero) masked_scatter.
        let tokens: Vec<u32> = vec![10, 258881, 11];
        let out = expand_audio_tokens(&tokens, &[0], 258881, 256000, 258883).unwrap();
        assert_eq!(out, vec![10, 256000, 258883, 11]);
        assert_eq!(
            out.iter().filter(|&&t| t == 258881).count(),
            0,
            "zero-frame audio yields no audio-token positions"
        );
    }
}
