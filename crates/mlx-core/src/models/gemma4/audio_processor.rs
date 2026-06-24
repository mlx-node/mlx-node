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

/// Sample rate the unified Gemma 4 audio front-end requires (16 kHz mono).
pub const REQUIRED_AUDIO_SAMPLE_RATE: u32 = 16_000;

/// Decode a RIFF/WAVE byte stream into a mono float32 PCM waveform at 16 kHz.
///
/// The unified Gemma 4 audio feature extractor consumes a **mono float32
/// 16 kHz** waveform; it does NOT decode containers or resample. Mirroring the
/// IMAGE path (encoded bytes in, Rust decodes), file decode lives here. This is
/// a minimal hand-rolled RIFF/WAVE parser — no new crate, hermetic build.
///
/// Supported `fmt ` formats:
///   * PCM16 (`audioFormat == 1`, `bitsPerSample == 16`): `i16 → f32 / 32768.0`.
///   * IEEE float32 (`audioFormat == 3`, `bitsPerSample == 32`): little-endian
///     f32 passthrough.
///
/// Multi-channel input is downmixed to mono by averaging the channels per frame.
/// The sample rate MUST be exactly 16 kHz — any other rate is rejected with a
/// clear error (resampling is deliberately deferred to a higher layer). Unknown
/// chunks between `fmt ` and `data` are skipped. Truncated or non-RIFF bytes are
/// rejected.
pub fn decode_wav_to_pcm(bytes: &[u8]) -> Result<Vec<f32>> {
    // RIFF header: "RIFF" <u32 size> "WAVE".
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::from_reason(
            "decode_wav_to_pcm: not a RIFF/WAVE file (missing RIFF/WAVE header)",
        ));
    }

    let mut audio_format: Option<u16> = None;
    let mut num_channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut data: Option<&[u8]> = None;

    // Walk chunks starting after the 12-byte RIFF/WAVE header. Each chunk is
    // a 4-byte id + 4-byte little-endian size + payload, with the payload
    // padded to an even byte boundary.
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.checked_add(chunk_size).ok_or_else(|| {
            Error::from_reason("decode_wav_to_pcm: chunk size overflow (corrupt WAV)")
        })?;
        if body_end > bytes.len() {
            return Err(Error::from_reason(
                "decode_wav_to_pcm: truncated chunk (declared size exceeds file)",
            ));
        }
        let body = &bytes[body_start..body_end];

        if chunk_id == b"fmt " {
            if body.len() < 16 {
                return Err(Error::from_reason(
                    "decode_wav_to_pcm: fmt chunk too short (need >= 16 bytes)",
                ));
            }
            audio_format = Some(u16::from_le_bytes([body[0], body[1]]));
            num_channels = Some(u16::from_le_bytes([body[2], body[3]]));
            sample_rate = Some(u32::from_le_bytes([body[4], body[5], body[6], body[7]]));
            bits_per_sample = Some(u16::from_le_bytes([body[14], body[15]]));
        } else if chunk_id == b"data" {
            data = Some(body);
        }

        // Advance past payload + the pad byte for odd-length payloads.
        pos = body_end + (chunk_size & 1);
    }

    let audio_format =
        audio_format.ok_or_else(|| Error::from_reason("decode_wav_to_pcm: missing fmt chunk"))?;
    let num_channels = num_channels.unwrap_or(0);
    let sample_rate = sample_rate.unwrap_or(0);
    let bits_per_sample = bits_per_sample.unwrap_or(0);
    let data = data.ok_or_else(|| Error::from_reason("decode_wav_to_pcm: missing data chunk"))?;

    if num_channels == 0 {
        return Err(Error::from_reason(
            "decode_wav_to_pcm: invalid channel count (0)",
        ));
    }
    if sample_rate != REQUIRED_AUDIO_SAMPLE_RATE {
        return Err(Error::from_reason(format!(
            "gemma4 audio expects 16 kHz mono; got {sample_rate} Hz (resampling is not supported)"
        )));
    }

    let channels = num_channels as usize;

    // Decode interleaved samples into a flat f32 buffer, then downmix.
    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        // PCM16.
        (1, 16) => {
            if !data.len().is_multiple_of(2) {
                return Err(Error::from_reason(
                    "decode_wav_to_pcm: PCM16 data length is not a multiple of 2 bytes",
                ));
            }
            data.chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect()
        }
        // IEEE float32.
        (3, 32) => {
            if !data.len().is_multiple_of(4) {
                return Err(Error::from_reason(
                    "decode_wav_to_pcm: float32 data length is not a multiple of 4 bytes",
                ));
            }
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        (fmt, bits) => {
            return Err(Error::from_reason(format!(
                "decode_wav_to_pcm: unsupported WAV format (audioFormat={fmt}, bitsPerSample={bits}); \
                 only PCM16 (1/16) and IEEE float32 (3/32) are supported"
            )));
        }
    };

    if !interleaved.len().is_multiple_of(channels) {
        return Err(Error::from_reason(
            "decode_wav_to_pcm: sample count is not a multiple of the channel count",
        ));
    }

    if channels == 1 {
        return Ok(interleaved);
    }

    // Downmix to mono: mean of the per-frame channel samples.
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * channels;
        let sum: f32 = interleaved[base..base + channels].iter().sum();
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

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
/// supplies the frame count for each placeholder in order. Non-placeholder
/// tokens pass through unchanged.
///
/// When the prompt carries no `audio_token_id` placeholder but clips are
/// supplied (the template-less manual prompt formatter emits none), each clip's
/// span is inserted after BOS, in clip order. An `Err` is returned only for a
/// genuine placeholder/clip-count mismatch (some placeholders present, but their
/// count differs from `n_frames_per_audio.len()`).
pub fn expand_audio_tokens(
    tokens: &[u32],
    n_frames_per_audio: &[usize],
    audio_token_id: u32,
    boa_token_id: u32,
    eoa_token_id: u32,
) -> Result<Vec<u32>> {
    let placeholder_count = tokens.iter().filter(|&&t| t == audio_token_id).count();

    // The template-less prompt formatter emits no <|audio|> placeholder, so when
    // clips are supplied without one, insert each clip's span after BOS (mirrors
    // expand_image_tokens). Each clip becomes boa + audio_token × n_frames + eoa,
    // in clip order.
    if placeholder_count == 0 && !n_frames_per_audio.is_empty() {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let total_frames: usize = n_frames_per_audio.iter().sum();
        let mut result =
            Vec::with_capacity(tokens.len() + total_frames + 2 * n_frames_per_audio.len());
        result.push(tokens[0]); // BOS
        for &n_frames in n_frames_per_audio {
            result.push(boa_token_id);
            for _ in 0..n_frames {
                result.push(audio_token_id);
            }
            result.push(eoa_token_id);
        }
        result.extend_from_slice(&tokens[1..]);
        return Ok(result);
    }

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
    fn expand_audio_no_placeholder_inserts_spans_after_bos() {
        // Single clip: no <|audio|> placeholder but a 3-frame clip is supplied,
        // so the span is inserted after BOS (tokens[0] = 2 here).
        let tokens: Vec<u32> = vec![2, 10, 11];
        let out = expand_audio_tokens(&tokens, &[3], 258881, 256000, 258883).unwrap();
        assert_eq!(out, vec![2, 256000, 258881, 258881, 258881, 258883, 10, 11]);
        assert_eq!(out[0], 2, "BOS preserved at position 0");
        assert_eq!(
            out.iter().filter(|&&t| t == 258881).count(),
            3,
            "audio-token count equals total frames"
        );

        // Two clips: spans inserted after BOS in clip order (1 frame, then 2).
        let tokens: Vec<u32> = vec![2, 9];
        let out = expand_audio_tokens(&tokens, &[1, 2], 258881, 256000, 258883).unwrap();
        assert_eq!(
            out,
            vec![2, 256000, 258881, 258883, 256000, 258881, 258881, 258883, 9]
        );
        assert_eq!(out[0], 2, "BOS preserved at position 0");
        assert_eq!(
            out.iter().filter(|&&t| t == 258881).count(),
            3,
            "audio-token count equals total frames"
        );
    }

    /// Build a minimal RIFF/WAVE byte stream for testing.
    fn build_wav(
        audio_format: u16,
        num_channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let block_align = num_channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&audio_format.to_le_bytes());
        out.extend_from_slice(&num_channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits_per_sample.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn decode_wav_pcm16_mono_roundtrip() {
        // i16 samples → f32 / 32768.0.
        let samples: [i16; 4] = [0, 16384, -32768, 32767];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(1, 1, 16_000, 16, &data);
        let pcm = decode_wav_to_pcm(&wav).unwrap();
        assert_eq!(pcm.len(), 4);
        assert_eq!(pcm[0], 0.0);
        assert_eq!(pcm[1], 0.5);
        assert_eq!(pcm[2], -1.0);
        assert!((pcm[3] - 32767.0 / 32768.0).abs() < 1e-9);
    }

    #[test]
    fn decode_wav_stereo_downmix_to_mono() {
        // 2 frames, 2 channels: frame0 = (0, 32768→0.5*2?), use simple values.
        // frame0 L=1.0(32767≈), R=0.0 → mean ≈ 0.5; frame1 L=-1.0, R=1.0 → mean ≈ 0.
        let interleaved: [i16; 4] = [16384, 0, -32768, 16384];
        let mut data = Vec::new();
        for s in interleaved {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(1, 2, 16_000, 16, &data);
        let pcm = decode_wav_to_pcm(&wav).unwrap();
        assert_eq!(pcm.len(), 2, "2 stereo frames downmix to 2 mono samples");
        // frame0: (0.5 + 0.0)/2 = 0.25
        assert!((pcm[0] - 0.25).abs() < 1e-6);
        // frame1: (-1.0 + 0.5)/2 = -0.25
        assert!((pcm[1] - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn decode_wav_float32_mono() {
        let samples: [f32; 3] = [0.0, 0.5, -0.75];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(3, 1, 16_000, 32, &data);
        let pcm = decode_wav_to_pcm(&wav).unwrap();
        assert_eq!(pcm, vec![0.0, 0.5, -0.75]);
    }

    #[test]
    fn decode_wav_rejects_non_16khz() {
        let data: [u8; 4] = [0, 0, 0, 0];
        let wav = build_wav(1, 1, 44_100, 16, &data);
        let err = decode_wav_to_pcm(&wav).unwrap_err();
        assert!(
            err.reason.contains("16 kHz"),
            "error should mention 16 kHz: {}",
            err.reason
        );
    }

    #[test]
    fn decode_wav_rejects_truncated_and_garbage() {
        // Not RIFF.
        assert!(decode_wav_to_pcm(b"not a wav file at all").is_err());
        // Too short.
        assert!(decode_wav_to_pcm(&[0u8; 4]).is_err());
        // RIFF/WAVE header but a data chunk claiming more bytes than present.
        let mut wav = build_wav(1, 1, 16_000, 16, &[0u8; 8]);
        // Corrupt the data chunk size to claim a huge length.
        let len = wav.len();
        wav[len - 8 - 4..len - 8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(decode_wav_to_pcm(&wav).is_err());
    }

    #[test]
    fn decode_wav_skips_unknown_chunks() {
        // Insert a "LIST" chunk between fmt and data; it must be skipped.
        let samples: [i16; 2] = [16384, -16384];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let mut wav = build_wav(1, 1, 16_000, 16, &data);
        // Rebuild with an extra LIST chunk after fmt (offset 36 = after fmt).
        let mut with_list = Vec::new();
        with_list.extend_from_slice(&wav[0..36]); // RIFF/WAVE + fmt
        let list_body = b"INFOtest";
        with_list.extend_from_slice(b"LIST");
        with_list.extend_from_slice(&(list_body.len() as u32).to_le_bytes());
        with_list.extend_from_slice(list_body);
        with_list.extend_from_slice(&wav[36..]); // data chunk
        // Fix RIFF size.
        let riff_size = (with_list.len() - 8) as u32;
        with_list[4..8].copy_from_slice(&riff_size.to_le_bytes());
        wav = with_list;
        let pcm = decode_wav_to_pcm(&wav).unwrap();
        assert_eq!(pcm.len(), 2);
        assert!((pcm[0] - 0.5).abs() < 1e-6);
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
