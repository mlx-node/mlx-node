//! Numerical parity tests against pinned checkpoints and external fixtures.
//! They are `#[ignore]`d by default. Run them with
//! `TTS_TEST_MODEL=<checkpoint dir> TTS_TEST_GOLDENS=<fixture dir> cargo test -p mlx-core --lib -- --ignored qwen3_tts`;
//! fixture contents are described in docs/tts.md.
use super::{
    codec::CodecDecoder, config::read_json, encoder::CodecEncoder, tokenizer::TextTokenizer,
    weights::Weights,
};
use crate::{array::MxArray, engine::persistence::load_all_safetensors};
use std::path::Path;

fn difference(actual: &MxArray, expected: &MxArray) -> (f32, f32) {
    let a = actual.to_float32().unwrap();
    let b = expected.to_float32().unwrap();
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let rms = (a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2) as f64)
        .sum::<f64>()
        / a.len() as f64)
        .sqrt() as f32;
    (max, rms)
}

#[test]
#[ignore = "Requires a Qwen3-TTS checkpoint in TTS_TEST_MODEL"]
fn qwen3_tts_predictor_cache_capacity_parity() {
    use super::{config::ModelConfig, transformer::Decoder};
    use crate::transformer::KVCache;

    let root = std::env::var("TTS_TEST_MODEL").unwrap();
    let root = Path::new(&root);
    let config: ModelConfig = read_json(&root.join("config.json")).unwrap();
    let weights = Weights::load(root).unwrap();
    let groups = config.talker_config.num_code_groups;
    let cfg = &config.talker_config.code_predictor_config.transformer;
    let predictor =
        Decoder::load(&weights, "talker.code_predictor.model", cfg, true, false).unwrap();
    let mut standard = predictor.state();
    let mut compact = predictor.state();
    for state in &mut compact {
        state.kv = KVCache::with_growth_step(groups).unwrap();
    }
    // Deterministic teacher inputs isolate allocation layout from sampling.
    // Repeated frames exercise buffer reuse, with different inputs after reset.
    for frame in 0..3 {
        for state in standard.iter_mut().chain(&mut compact) {
            state.kv.reset_keep_capacity();
            state.position = 0;
        }
        for code in 0..groups - 1 {
            let rows = if code == 0 { 2 } else { 1 };
            let values: Vec<f32> = (0..rows * cfg.hidden_size)
                .map(|i| ((i + code as i64 * 17 + frame * 31) % 127) as f32 / 127. - 0.5)
                .collect();
            let input = MxArray::from_float32(&values, &[1, rows, cfg.hidden_size]).unwrap();
            let head = weights
                .linear(&format!("talker.code_predictor.lm_head.{code}"))
                .unwrap();
            let logits = |state: &mut [_]| {
                let hidden = predictor.forward(&input, state).unwrap();
                head.forward(&hidden.slice_axis(1, rows - 1, rows).unwrap())
                    .unwrap()
            };
            let expected = logits(&mut standard);
            let actual = logits(&mut compact);
            let (max, rms) = difference(&actual, &expected);
            assert_eq!(
                (max, rms),
                (0., 0.),
                "frame {frame}, code {code}: allocation changed logits"
            );
            assert!(
                actual
                    .to_float32()
                    .unwrap()
                    .iter()
                    .zip(expected.to_float32().unwrap().iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "frame {frame}, code {code}: logit bits differ"
            );
        }
        for state in &compact {
            assert_eq!(state.kv.get_offset(), groups as i32);
            assert_eq!(state.kv.capacity().unwrap(), groups as i64);
        }
    }
}
#[test]
#[ignore = "Requires pinned Qwen3-TTS weights and external development fixtures; see docs/tts.md"]
fn qwen3_tts_reference_parity() {
    let root = std::env::var("TTS_TEST_MODEL").expect("TTS_TEST_MODEL");
    let root = Path::new(&root);
    let goldens = std::env::var("TTS_TEST_GOLDENS").expect("TTS_TEST_GOLDENS");
    let goldens = Path::new(&goldens);
    let expected: std::collections::HashMap<String, Vec<i32>> =
        read_json(&goldens.join("tokenizer.json")).unwrap();
    let tokenizer = TextTokenizer::load(root).unwrap();
    for (text, ids) in expected {
        assert_eq!(tokenizer.encode(&text).unwrap(), ids, "{text}");
    }
    let fixtures = load_all_safetensors(goldens, false).unwrap();
    let weights = Weights::load(&root.join("speech_tokenizer")).unwrap();
    let config: super::config::CodecConfig =
        read_json(&root.join("speech_tokenizer/config.json")).unwrap();
    let encoder = CodecEncoder::load(&weights, &config).unwrap();
    // The encoder's own entry point must enforce its context bound, even when
    // called without NativeModel's pre-resampling validation.
    let mut bounded_config = config.clone();
    let encoder_config = bounded_config.encoder_config.as_mut().unwrap();
    let stride: usize = encoder_config["upsampling_ratios"]
        .as_array()
        .unwrap()
        .iter()
        .map(|ratio| ratio.as_u64().unwrap() as usize)
        .product();
    encoder_config["max_position_embeddings"] = serde_json::json!(1);
    let bounded = CodecEncoder::load(&weights, &bounded_config).unwrap();
    assert!(
        bounded
            .encode(&vec![0.; stride + 1])
            .err()
            .unwrap()
            .reason
            .contains("context capacity")
    );
    let codec = CodecDecoder::load(&weights, config).unwrap();
    let codes = &fixtures["codes"];
    let full = codec.decode(codes).unwrap();
    full.eval();
    let (max, rms) = difference(&full, &fixtures["decoded"]);
    eprintln!("codec reference max={max} rms={rms}");
    assert!(
        max < 0.005 && rms < 0.001,
        "decoder differs from pinned reference"
    );
    for chunks in [vec![1; 13], vec![2, 3, 1, 4, 3], vec![13]] {
        let mut state = codec.state();
        let mut offset = 0;
        let mut audio = Vec::new();
        for n in chunks {
            let x = codec
                .step(
                    &codes.slice_axis(1, offset, offset + n).unwrap(),
                    &mut state,
                )
                .unwrap();
            x.eval();
            audio.push(x);
            offset += n;
        }
        let joined = MxArray::concatenate_many(audio.iter().collect(), Some(0)).unwrap();
        let (max, rms) = difference(&joined, &full);
        eprintln!("codec chunks max={max} rms={rms}");
        assert!(
            max < 0.005 && rms < 0.001,
            "chunking changed codec waveform"
        );
    }
    let wave = fixtures["wave"].to_float32().unwrap();
    let encoded = encoder.encode(&wave).unwrap();
    let (max, _) = difference(&encoded, &fixtures["encoded"]);
    eprintln!("encoder token max difference={max}");
    assert_eq!(max, 0.);
    let long_wave = fixtures["long_wave"].to_float32().unwrap();
    let long_codes = encoder.encode(&long_wave).unwrap();
    let (max, _) = difference(&long_codes, &fixtures["long_encoded"]);
    eprintln!("long encoder token max difference={max}");
    assert_eq!(max, 0.);

    // Cross the configured attention window with legal zero-valued codes.
    // Chunk boundaries may fall on either side of cache eviction.
    let window = codec
        .config
        .decoder_config
        .transformer
        .sliding_window
        .unwrap();
    let length = window + 17;
    let groups = codec.config.decoder_config.num_quantizers;
    let values: Vec<i32> = (0..length * groups).map(|i| (i % 127) as i32).collect();
    let codes = MxArray::from_int32(&values, &[1, length as i64, groups as i64]).unwrap();
    let full = codec.decode(&codes).unwrap();
    full.eval();
    for chunks in [vec![1; length], vec![window - 1, 2, 16]] {
        let mut state = codec.state();
        let mut at = 0;
        let mut parts = vec![];
        for n in chunks {
            let out = codec
                .step(
                    &codes.slice_axis(1, at as i64, (at + n) as i64).unwrap(),
                    &mut state,
                )
                .unwrap();
            out.eval();
            parts.push(out);
            at += n;
        }
        let joined = MxArray::concatenate_many(parts.iter().collect(), Some(0)).unwrap();
        assert_eq!(
            joined.size().unwrap(),
            length as u64 * codec.config.decode_upsample_rate as u64
        );
        let (max, rms) = difference(&joined, &full);
        eprintln!("codec across window max={max} rms={rms}");
        assert!(max < 0.005 && rms < 0.001);
    }

    // Reusable voice prefixes must match whole-reference decoding even after
    // attention eviction, and one continuation must not modify another's state.
    let prefix_len = window + 1;
    let prefix_codes = codes.slice_axis(1, 0, prefix_len as i64).unwrap();
    let continuation = codes
        .slice_axis(1, prefix_len as i64, length as i64)
        .unwrap();
    assert!(
        codec
            .prepare_prefix(&prefix_codes, &std::sync::atomic::AtomicBool::new(true))
            .is_err()
    );
    let prepared = codec
        .prepare_prefix(&prefix_codes, &std::sync::atomic::AtomicBool::new(false))
        .unwrap();
    let expected = full
        .slice_axis(
            0,
            (prefix_len * codec.config.decode_upsample_rate) as i64,
            (length * codec.config.decode_upsample_rate) as i64,
        )
        .unwrap();
    for _ in 0..2 {
        let mut state = prepared.fork().unwrap();
        let actual = codec.step(&continuation, &mut state).unwrap();
        let (max, rms) = difference(&actual, &expected);
        eprintln!("prepared codec prefix max={max} rms={rms}");
        assert!(max < 0.005 && rms < 0.001);
        // Mutate/evict the fork before checking a second fork of the snapshot.
        codec.step(&prefix_codes, &mut state).unwrap().eval();
    }
}

#[test]
#[ignore = "Requires Base speech tokenizer weights and official CPU encoder fixtures"]
fn qwen3_tts_encoder_reference_parity() {
    let root = std::env::var("TTS_TEST_MODEL").expect("TTS_TEST_MODEL");
    let root = Path::new(&root).join("speech_tokenizer");
    let goldens = std::env::var("TTS_TEST_GOLDENS").expect("TTS_TEST_GOLDENS");
    let fixtures = load_all_safetensors(Path::new(&goldens), false).unwrap();
    let weights = Weights::load(&root).unwrap();
    let config = read_json(&root.join("config.json")).unwrap();
    let encoder = CodecEncoder::load(&weights, &config).unwrap();
    for (audio, expected) in [("wave", "encoded"), ("long_wave", "long_encoded")] {
        let wave = fixtures[audio].to_float32().unwrap();
        let codes = encoder.encode(&wave).unwrap();
        assert_eq!(
            codes.shape().unwrap().as_ref(),
            fixtures[expected].shape().unwrap().as_ref(),
            "{expected} shape differs from official encoder"
        );
        let (max, _) = difference(&codes, &fixtures[expected]);
        eprintln!("official {expected} token max difference={max}");
        assert_eq!(max, 0., "{expected} differs from official encoder");
    }
}

#[test]
#[ignore = "Requires pinned Base checkpoint and real ICL prompt development goldens"]
fn qwen3_tts_icl_prompt_parity() {
    use super::model::{NativeModel, Options};
    let root = std::env::var("TTS_TEST_MODEL").unwrap();
    let fixtures = load_all_safetensors(
        Path::new(&std::env::var("TTS_TEST_GOLDENS").unwrap()),
        false,
    )
    .unwrap();
    let (mut model, _) = NativeModel::load(Path::new(&root)).unwrap();
    // Cancellation is checked before validating or computing reference inputs.
    let error = model
        .prepare_voice(
            vec![],
            0,
            String::new(),
            &std::sync::atomic::AtomicBool::new(true),
        )
        .unwrap_err();
    assert!(error.reason.contains("cancelled"));
    let id = model
        .prepare_voice(
            fixtures["wave"].to_float32().unwrap().to_vec(),
            24000,
            "你好，这是参考声音。".into(),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
    let options = Options {
        prepared_voice_id: Some(id.clone()),
        language: Some("chinese".into()),
        ..Default::default()
    };
    let (prompt, trailing, pad) = model.prepare("你好，这是新的句子。", &options).unwrap();
    for (value, name) in [
        (&prompt, "icl_prompt"),
        (&trailing, "icl_trailing"),
        (&pad, "icl_pad"),
    ] {
        let (max, rms) = difference(value, &fixtures[name]);
        eprintln!("{name} max={max} rms={rms}");
        // Checkpoint bf16 projections can differ by one rounding unit.
        assert!(max < 0.05 && rms < 0.01);
    }
    model.release_voice(&id);
    model.release_voice(&id);
    assert!(model.prepare("声音已经释放。", &options).is_err());
}

#[test]
#[ignore = "Requires pinned Qwen3-TTS Base weights and float32 development goldens"]
fn qwen3_tts_teacher_parity() {
    use super::{
        config::ModelConfig,
        speaker::{SpeakerEncoder, speaker_mel},
        transformer::Decoder,
    };
    use crate::{array::DType, nn::Embedding};
    let root = std::env::var("TTS_TEST_MODEL").unwrap();
    let root = Path::new(&root);
    let goldens = std::env::var("TTS_TEST_GOLDENS").unwrap();
    let goldens = Path::new(&goldens);
    let fixtures = load_all_safetensors(goldens, false).unwrap();
    let config: ModelConfig = read_json(&root.join("config.json")).unwrap();
    let mut weights = Weights::load(root).unwrap();
    for weight in weights.tensors.values_mut() {
        *weight = weight.astype(DType::Float32).unwrap();
    }
    let wave = fixtures["wave"].to_float32().unwrap();
    let mel = speaker_mel(&wave, &super::config::SpeakerConfig::default()).unwrap();
    let (max, rms) = difference(&mel, &fixtures["mel"]);
    eprintln!("speaker mel max={max} rms={rms}");
    // A pure sinusoid places most bins near the log floor, amplifying f32
    // FFT roundoff between rustfft and MLX. Also bound downstream embeddings.
    assert!(max < 0.005 && rms < 0.0005);
    let speaker = SpeakerEncoder::load(&weights, config.speaker_encoder_config.unwrap()).unwrap();
    let embedding = speaker.encode(&wave).unwrap();
    let (max, rms) = difference(&embedding, &fixtures["speaker"]);
    eprintln!("speaker embedding max={max} rms={rms}");
    assert!(max < 0.002 && rms < 0.0005);
    let talker = Decoder::load(
        &weights,
        "talker.model",
        &config.talker_config.transformer,
        true,
        false,
    )
    .unwrap();
    let hidden = talker
        .forward(&fixtures["talker_input"], &mut talker.state())
        .unwrap();
    let logits = weights
        .linear("talker.codec_head")
        .unwrap()
        .forward(&hidden)
        .unwrap();
    let (max, rms) = difference(&logits, &fixtures["talker_logits"]);
    eprintln!("talker logits max={max} rms={rms}");
    assert!(max < 0.003 && rms < 0.001);
    let predictor = Decoder::load(
        &weights,
        "talker.code_predictor.model",
        &config.talker_config.code_predictor_config.transformer,
        true,
        false,
    )
    .unwrap();
    let mut state = predictor.state();
    for i in 0..config.talker_config.num_code_groups - 1 {
        let input = if i == 0 {
            fixtures["code_input"].clone()
        } else {
            Embedding::from_weight(
                &weights
                    .get(&format!(
                        "talker.code_predictor.model.codec_embedding.{}.weight",
                        i - 1
                    ))
                    .unwrap(),
            )
            .unwrap()
            .forward(&MxArray::from_int32(&[i as i32], &[1, 1]).unwrap())
            .unwrap()
        };
        let hidden = predictor.forward(&input, &mut state).unwrap();
        let n = hidden.shape().unwrap()[1];
        let logits = weights
            .linear(&format!("talker.code_predictor.lm_head.{i}"))
            .unwrap()
            .forward(&hidden.slice_axis(1, n - 1, n).unwrap())
            .unwrap();
        let (max, rms) = difference(
            &logits,
            &fixtures["code_logits"]
                .slice_axis(1, i as i64, i as i64 + 1)
                .unwrap(),
        );
        eprintln!("predictor {i} logits max={max} rms={rms}");
        assert!(max < 0.003 && rms < 0.001);
    }
}
