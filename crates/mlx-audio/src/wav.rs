//! RIFF/WAVE container parsing, sample decoding, PCM16 encoding, and header building.
//!
//! File/socket ownership stays with the caller; frame-level calls expect complete
//! sample frames while [`decode_wav_mono_f32`] walks a full RIFF/WAVE byte stream.

use crate::error::AudioError;

/// Parsed WAV `fmt ` chunk metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WavFormat {
    pub encoding: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub block_align: u32,
}

fn validate(format: &WavFormat) -> Result<(), AudioError> {
    if format.sample_rate == 0
        || format.channels == 0
        || format.channels > 32
        || !matches!(
            (format.encoding, format.bits_per_sample),
            (1, 8 | 16 | 24 | 32) | (3, 32 | 64)
        )
        || format.block_align != format.channels * (format.bits_per_sample / 8)
    {
        return Err(AudioError::InvalidWavFormat);
    }
    Ok(())
}

/// Parse a WAV `fmt ` chunk body (PCM 8/16/24/32, IEEE float 32/64, and
/// `WAVE_FORMAT_EXTENSIBLE`).
pub fn parse_wav_format(bytes: &[u8]) -> Result<WavFormat, AudioError> {
    if bytes.len() < 16 {
        return Err(AudioError::InvalidWavFormat);
    }
    let u16_at = |i| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u32;
    let u32_at = |i| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let mut encoding = u16_at(0);
    if encoding == 0xfffe {
        if bytes.len() < 40
            || u16_at(16) < 22
            || bytes[28..40] != [0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]
        {
            return Err(AudioError::InvalidWavFormat);
        }
        encoding = u32_at(24);
        let valid_bits = u16_at(18);
        let container_bits = u16_at(14);
        if valid_bits == 0
            || valid_bits > container_bits
            || (encoding != 1 && valid_bits != container_bits)
        {
            return Err(AudioError::InvalidWavFormat);
        }
        // Extensible PCM may have lower precision than its container. Its
        // valid bits are left-aligned, so container-width decoding preserves
        // the amplitude without any additional shift or rescaling.
    }
    let format = WavFormat {
        encoding,
        sample_rate: u32_at(4),
        channels: u16_at(2),
        bits_per_sample: u16_at(14),
        block_align: u16_at(12),
    };
    validate(&format)?;
    if u32_at(8) as u64 != format.sample_rate as u64 * format.block_align as u64 {
        return Err(AudioError::InvalidWavFormat);
    }
    Ok(format)
}

/// Decode complete WAV sample frames into `f32` samples (optionally downmixed
/// to mono by per-frame channel mean).
pub fn decode_wav_frames(
    bytes: &[u8],
    format: &WavFormat,
    mono: bool,
) -> Result<Vec<f32>, AudioError> {
    validate(format)?;
    if !bytes.len().is_multiple_of(format.block_align as usize) {
        return Err(AudioError::IncompleteWavFrame);
    }
    let width = (format.bits_per_sample / 8) as usize;
    let channels = format.channels as usize;
    let mut output = Vec::with_capacity(
        bytes.len()
            / if mono {
                format.block_align as usize
            } else {
                width
            },
    );
    let downmix = mono && channels > 1;
    for frame in bytes.chunks_exact(format.block_align as usize) {
        let mut sum = 0.0f64;
        for sample in frame.chunks_exact(width) {
            let value = match (format.encoding, width) {
                (3, 4) => f32::from_le_bytes(
                    sample
                        .try_into()
                        .map_err(|_| AudioError::InvalidWavFormat)?,
                ),
                (3, 8) => f64::from_le_bytes(
                    sample
                        .try_into()
                        .map_err(|_| AudioError::InvalidWavFormat)?,
                ) as f32,
                (1, 1) => (sample[0] as f32 - 128.0) / 128.0,
                (1, 2) => {
                    i16::from_le_bytes(
                        sample
                            .try_into()
                            .map_err(|_| AudioError::InvalidWavFormat)?,
                    ) as f32
                        / 32768.0
                }
                (1, 3) => {
                    ((sample[0] as i32 | (sample[1] as i32) << 8 | (sample[2] as i32) << 16) << 8
                        >> 8) as f32
                        / 8388608.0
                }
                (1, 4) => {
                    i32::from_le_bytes(
                        sample
                            .try_into()
                            .map_err(|_| AudioError::InvalidWavFormat)?,
                    ) as f32
                        / 2147483648.0
                }
                _ => return Err(AudioError::InvalidWavFormat),
            };
            if !value.is_finite() {
                return Err(AudioError::NonFiniteWavSample);
            }
            if downmix {
                sum += value as f64;
            } else {
                output.push(value);
            }
        }
        if downmix {
            output.push((sum / channels as f64) as f32);
        }
    }
    Ok(output)
}

/// Walk a complete RIFF/WAVE container and decode its `data` chunk into mono
/// `f32` PCM along with the declared sample rate in Hz.
///
/// Unknown chunks (such as `LIST`, `JUNK`, or `fact`) are skipped, including
/// the 1-byte alignment pad after odd-length chunk payloads. Supports every
/// encoding accepted by [`parse_wav_format`] and [`decode_wav_frames`]
/// (8/16/24/32-bit PCM, 32/64-bit IEEE float, and `WAVE_FORMAT_EXTENSIBLE`).
pub fn decode_wav_mono_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32), AudioError> {
    // RIFF header: "RIFF" <u32 size> "WAVE".
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::InvalidWavContainer(
            "not a RIFF/WAVE file (missing RIFF/WAVE header)".into(),
        ));
    }

    let mut fmt_body: Option<&[u8]> = None;
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
            AudioError::InvalidWavContainer("chunk size overflow (corrupt WAV)".into())
        })?;
        if body_end > bytes.len() {
            return Err(AudioError::InvalidWavContainer(
                "truncated chunk (declared size exceeds file)".into(),
            ));
        }
        let body = &bytes[body_start..body_end];

        if chunk_id == b"fmt " {
            if body.len() < 16 {
                return Err(AudioError::InvalidWavContainer(
                    "fmt chunk too short (need >= 16 bytes)".into(),
                ));
            }
            fmt_body = Some(body);
        } else if chunk_id == b"data" {
            data = Some(body);
        }

        // Advance past payload + the pad byte for odd-length payloads.
        pos = body_end.saturating_add(chunk_size & 1);
    }

    let fmt_body =
        fmt_body.ok_or_else(|| AudioError::InvalidWavContainer("missing fmt chunk".into()))?;
    let data = data.ok_or_else(|| AudioError::InvalidWavContainer("missing data chunk".into()))?;

    let num_channels = u16::from_le_bytes([fmt_body[2], fmt_body[3]]);
    if num_channels == 0 {
        return Err(AudioError::InvalidWavContainer(
            "invalid channel count (0)".into(),
        ));
    }

    let format = parse_wav_format(fmt_body)?;
    let pcm = decode_wav_frames(data, &format, true)?;
    Ok((pcm, format.sample_rate))
}

/// Encode normalized `f32` PCM samples into little-endian signed 16-bit PCM.
pub fn encode_pcm16(samples: &[f32]) -> Result<Vec<u8>, AudioError> {
    let mut output = Vec::with_capacity(samples.len() * 2);
    for value in samples {
        if !value.is_finite() {
            return Err(AudioError::NonFinitePcmSample);
        }
        // Match JavaScript Math.round, including negative half-way values.
        let value = ((*value as f64 * 32768.0 + 0.5).floor()).clamp(-32768.0, 32767.0) as i16;
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output)
}

/// Build a 44-byte canonical PCM16 RIFF/WAVE header.
pub fn wav_header(sample_rate: u32, channels: u32, data_bytes: u32) -> Result<Vec<u8>, AudioError> {
    if sample_rate == 0
        || channels == 0
        || channels > 32
        || data_bytes > u32::MAX - 36
        || !(data_bytes as u64).is_multiple_of(channels as u64 * 2)
        || sample_rate as u64 * channels as u64 * 2 > u32::MAX as u64
    {
        return Err(AudioError::InvalidWavFormat);
    }
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(data_bytes + 36).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * channels * 2).to_le_bytes());
    out.extend_from_slice(&(channels as u16 * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extensible_format(encoding: u32, bits: u16, valid_bits: u16) -> Vec<u8> {
        let mut bytes = vec![0; 40];
        bytes[0..2].copy_from_slice(&0xfffeu16.to_le_bytes());
        bytes[2..4].copy_from_slice(&2u16.to_le_bytes());
        bytes[4..8].copy_from_slice(&48000u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&(48000 * bits as u32 / 8 * 2).to_le_bytes());
        bytes[12..14].copy_from_slice(&(bits / 8 * 2).to_le_bytes());
        bytes[14..16].copy_from_slice(&bits.to_le_bytes());
        bytes[16..18].copy_from_slice(&22u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&valid_bits.to_le_bytes());
        bytes[24..28].copy_from_slice(&encoding.to_le_bytes());
        bytes[28..40].copy_from_slice(&[0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
        bytes
    }

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
    fn extensible_pcm_precision_preserves_amplitude() {
        for (bits, valid_bits) in [(24, 20), (32, 24)] {
            let format = parse_wav_format(&extensible_format(1, bits, valid_bits)).unwrap();
            let samples: Vec<u8> = [0.5, -0.25, -1.0, 0.25]
                .into_iter()
                .flat_map(|v| {
                    let sample = (v * (1u64 << (bits - 1)) as f64) as i32;
                    sample.to_le_bytes()[..bits as usize / 8].to_vec()
                })
                .collect();
            assert_eq!(
                decode_wav_frames(&samples, &format, false).unwrap(),
                vec![0.5, -0.25, -1.0, 0.25]
            );
            assert_eq!(
                decode_wav_frames(&samples, &format, true).unwrap(),
                vec![0.125, -0.375]
            );
        }
    }

    #[test]
    fn extensible_precision_rejects_invalid_formats() {
        for (encoding, bits, valid_bits) in [(1, 32, 0), (1, 24, 25), (3, 32, 24), (3, 64, 32)] {
            assert!(parse_wav_format(&extensible_format(encoding, bits, valid_bits)).is_err());
        }
        for bits in [32, 64] {
            assert!(parse_wav_format(&extensible_format(3, bits, bits)).is_ok());
        }
    }

    #[test]
    fn pcm_formats_and_downmix() {
        for rate in [16000, 22050, 24000, 44100, 48000, 96000] {
            let f = WavFormat {
                encoding: 1,
                sample_rate: rate,
                channels: 2,
                bits_per_sample: 16,
                block_align: 4,
            };
            assert_eq!(
                decode_wav_frames(&[0, 128, 0, 64], &f, true).unwrap(),
                vec![-0.25]
            );
            assert!(decode_wav_frames(&[0], &f, true).is_err());
        }
        let f = WavFormat {
            encoding: 3,
            sample_rate: 48000,
            channels: 1,
            bits_per_sample: 32,
            block_align: 4,
        };
        assert!(decode_wav_frames(&f32::NAN.to_le_bytes(), &f, false).is_err());
    }

    #[test]
    fn pcm16_wav_roundtrip_preserves_clipping_and_rounding() {
        let samples = vec![-2.0, -1.0, -0.5, -1.0 / 65536.0, 0.0, 0.5, 1.0, 2.0];
        let pcm = encode_pcm16(&samples).unwrap();
        let header = wav_header(44100, 2, pcm.len() as u32).unwrap();
        assert_eq!(header.len(), 44);
        assert_eq!(&header[..4], b"RIFF");
        assert_eq!(&header[8..16], b"WAVEfmt ");
        assert_eq!(&header[36..40], b"data");
        assert_eq!(&header[4..8], &(36 + pcm.len() as u32).to_le_bytes());
        assert_eq!(&header[40..44], &(pcm.len() as u32).to_le_bytes());
        let format = parse_wav_format(&header[20..36]).unwrap();
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 2);
        let decoded = decode_wav_frames(&pcm, &format, false).unwrap();
        assert_eq!(
            decoded.as_slice(),
            &[
                -1.0,
                -1.0,
                -0.5,
                0.0,
                0.0,
                0.5,
                32767.0 / 32768.0,
                32767.0 / 32768.0
            ]
        );
    }

    #[test]
    fn public_codecs_reject_invalid_samples_and_dimensions() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(encode_pcm16(&[value]).is_err());
        }
        for (rate, channels, data_bytes) in [
            (0, 1, 0),
            (24000, 0, 0),
            (24000, 33, 0),
            (24000, 2, 2),
            (24000, 1, u32::MAX),
            (u32::MAX, 1, 0),
        ] {
            assert!(wav_header(rate, channels, data_bytes).is_err());
        }
        let format = WavFormat {
            encoding: 1,
            sample_rate: 24000,
            channels: 2,
            bits_per_sample: 16,
            block_align: 4,
        };
        assert!(decode_wav_frames(&[0; 3], &format, true).is_err());
    }

    #[test]
    fn public_decoder_supports_float_widths_and_downmix() {
        for bits in [32, 64] {
            let format = WavFormat {
                encoding: 3,
                sample_rate: 48000,
                channels: 2,
                bits_per_sample: bits,
                block_align: bits / 8 * 2,
            };
            let bytes: Vec<u8> = [-0.5, 0.25]
                .into_iter()
                .flat_map(|value: f64| {
                    if bits == 32 {
                        (value as f32).to_le_bytes().to_vec()
                    } else {
                        value.to_le_bytes().to_vec()
                    }
                })
                .collect();
            let decoded = decode_wav_frames(&bytes, &format, false).unwrap();
            assert_eq!(decoded.as_slice(), &[-0.5, 0.25]);
            let mono = decode_wav_frames(&bytes, &format, true).unwrap();
            assert_eq!(mono.as_slice(), &[-0.125]);
        }
    }

    #[test]
    fn decode_wav_mono_f32_handles_pcm16_stereo_float_and_unknown_chunks() {
        // Stereo PCM16 downmix.
        let interleaved: [i16; 4] = [16384, 0, -32768, 16384];
        let data: Vec<u8> = interleaved
            .into_iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let wav = build_wav(1, 2, 16_000, 16, &data);
        let (pcm, rate) = decode_wav_mono_f32(&wav).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(pcm, vec![0.25, -0.25]);

        // IEEE float32 mono with odd-length JUNK chunk before data.
        let float_data: Vec<u8> = [0.0f32, 0.5, -0.75]
            .into_iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let base = build_wav(3, 1, 24_000, 32, &float_data);
        let mut with_odd_junk = Vec::new();
        with_odd_junk.extend_from_slice(&base[..36]);
        with_odd_junk.extend_from_slice(b"JUNK");
        with_odd_junk.extend_from_slice(&3u32.to_le_bytes());
        with_odd_junk.extend_from_slice(b"abc\0"); // 3 bytes + 1 pad byte
        with_odd_junk.extend_from_slice(&base[36..]);
        let riff_size = (with_odd_junk.len() - 8) as u32;
        with_odd_junk[4..8].copy_from_slice(&riff_size.to_le_bytes());
        let (pcm, rate) = decode_wav_mono_f32(&with_odd_junk).unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(pcm, vec![0.0, 0.5, -0.75]);
    }

    #[test]
    fn decode_wav_mono_f32_rejects_malformed_containers() {
        assert!(decode_wav_mono_f32(b"not a wav file").is_err());
        assert!(decode_wav_mono_f32(&[0u8; 4]).is_err());
        // Missing fmt chunk.
        let mut no_fmt = Vec::new();
        no_fmt.extend_from_slice(b"RIFF");
        no_fmt.extend_from_slice(&12u32.to_le_bytes());
        no_fmt.extend_from_slice(b"WAVEdata");
        no_fmt.extend_from_slice(&0u32.to_le_bytes());
        assert!(decode_wav_mono_f32(&no_fmt).is_err());
        // Truncated data chunk.
        let mut wav = build_wav(1, 1, 16_000, 16, &[0u8; 8]);
        let len = wav.len();
        wav[len - 12..len - 8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(decode_wav_mono_f32(&wav).is_err());
    }
}
