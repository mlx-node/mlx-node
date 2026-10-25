//! Deterministic GGUF fixtures shared by the GGUF import tests, plus the
//! capture of the converter's output for each import fixture into
//! `testdata/gguf_import_golden/`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::gguf::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_3, GgufMetaValue, GgufTensorType,
    GgufValueType,
};

pub(crate) fn muse_native_fixture(source: &Path) -> PathBuf {
    use crate::models::muse_glimmer::config::fixtures::{config_json, text_config_json};
    fs::create_dir_all(source).unwrap();
    let text = text_config_json(2)
        .replace("6656", "256")
        .replace("19968", "512")
        .replace("\"num_attention_heads\": 32", "\"num_attention_heads\": 2");
    fs::write(source.join("config.json"), config_json(&text)).unwrap();
    fs::write(source.join("tokenizer.json"), "{}").unwrap();
    let input = source.join("muse-Q4_K_M.gguf");
    fs::write(
        &input,
        build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("muse-glimmer".into()),
            )],
            &[
                (
                    "blk.0.ffn_up.weight",
                    &[256, 2],
                    GgufTensorType::Q4K,
                    &[0; 288],
                ),
                ("blk.0.attn_norm.weight", &[2], GgufTensorType::F32, &[0; 8]),
            ],
        ),
    )
    .unwrap();
    input
}

pub(crate) fn muse_native_draft_fixture(path: &Path) {
    let mut metadata = complete_muse_glimmer_dflash_metadata();
    for (key, value) in [
        ("dflash.block_count", 1),
        ("dflash.block_size", 2),
        ("dflash.embedding_length", 256),
        ("dflash.feed_forward_length", 512),
        ("dflash.attention.head_count", 2),
        ("dflash.attention.head_count_kv", 1),
        ("dflash.attention.key_length", 128),
    ] {
        metadata.insert(key.into(), GgufMetaValue::Uint32(value));
    }
    metadata.insert(
        "dflash.target_layers".into(),
        GgufMetaValue::ArrayI32(vec![1]),
    );
    let metadata: Vec<_> = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.clone()))
        .collect();
    let descriptors: [(&str, &[u64]); 14] = [
        ("fc.weight", &[256, 256]),
        ("enc.output_norm.weight", &[256]),
        ("output_norm.weight", &[256]),
        ("blk.0.attn_norm.weight", &[256]),
        ("blk.0.ffn_norm.weight", &[256]),
        ("blk.0.attn_q_norm.weight", &[128]),
        ("blk.0.attn_k_norm.weight", &[128]),
        ("blk.0.attn_q.weight", &[256, 256]),
        ("blk.0.attn_k.weight", &[256, 128]),
        ("blk.0.attn_v.weight", &[256, 128]),
        ("blk.0.attn_output.weight", &[256, 256]),
        ("blk.0.ffn_gate.weight", &[256, 512]),
        ("blk.0.ffn_up.weight", &[256, 512]),
        ("blk.0.ffn_down.weight", &[512, 256]),
    ];
    let data: Vec<_> = descriptors
        .iter()
        .map(|(_, dims)| {
            let elements = dims.iter().product::<u64>() as usize;
            vec![
                0_u8;
                if dims.len() == 2 {
                    elements / 256 * 144
                } else {
                    elements * 4
                }
            ]
        })
        .collect();
    let tensors: Vec<_> = descriptors
        .iter()
        .zip(&data)
        .map(|((name, dims), data)| {
            (
                *name,
                *dims,
                if dims.len() == 2 {
                    GgufTensorType::Q4K
                } else {
                    GgufTensorType::F32
                },
                data.as_slice(),
            )
        })
        .collect();
    fs::write(path, build_minimal_gguf(&metadata, &tensors)).unwrap();
}

pub(crate) fn gemma_native_fixture(source: &Path) -> PathBuf {
    fs::create_dir_all(source).unwrap();
    fs::write(source.join("config.json"), r#"{"model_type":"gemma4_unified","text_config":{"num_hidden_layers":0},"vision_config":{}}"#).unwrap();
    fs::write(source.join("tokenizer.json"), "{}").unwrap();
    let input = source.join("gemma-with-a-long-enough-name-to-exercise-cache-key-length.gguf");
    fs::write(
        &input,
        build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".into()),
            )],
            &[
                (
                    "token_embd.weight",
                    &[256, 2],
                    GgufTensorType::Q6K,
                    &[0; 420],
                ),
                ("output.weight", &[256, 2], GgufTensorType::Q4_0, &[0; 288]),
            ],
        ),
    )
    .unwrap();
    fs::write(
        source.join("mmproj-gemma.gguf"),
        build_minimal_gguf(
            &[
                ("general.architecture", GgufMetaValue::String("clip".into())),
                (
                    "clip.vision.projector_type",
                    GgufMetaValue::String("gemma4uv".into()),
                ),
            ],
            &[(
                "mm.input_projection.weight",
                &[2, 2],
                GgufTensorType::BF16,
                &[0; 8],
            )],
        ),
    )
    .unwrap();
    input
}

/// A minimal `gemma4v` SigLIP tower mmproj written to `path`: the two
/// layout-relevant tensors (4-D conv patch kernel, 3-D position table),
/// one encoder block weight, the standardize pair, and the multimodal
/// projection. Conversion does not demand the full 356-tensor inventory,
/// so this exercises every code path the real `mmproj-BF16.gguf` drives.
pub(crate) fn gemma4v_mmproj_fixture(path: &Path) {
    fs::write(
        path,
        build_minimal_gguf(
            &[
                ("general.architecture", GgufMetaValue::String("clip".into())),
                (
                    "clip.vision.projector_type",
                    GgufMetaValue::String("gemma4v".into()),
                ),
            ],
            &[
                // ggml dims: [patch_w, patch_h, channels, hidden] → MLX
                // [1, 3, 2, 2] → input_proj [1, 12] after the fixup.
                (
                    "v.patch_embd.weight",
                    &[2, 2, 3, 1],
                    GgufTensorType::F32,
                    &[0; 48],
                ),
                // ggml dims: [hidden, positions, 2] → MLX [2, 2, 1].
                (
                    "v.position_embd.weight",
                    &[1, 2, 2],
                    GgufTensorType::F32,
                    &[0; 16],
                ),
                (
                    "v.blk.0.attn_q.weight",
                    &[2, 2],
                    GgufTensorType::BF16,
                    &[0; 8],
                ),
                ("v.std_bias", &[1], GgufTensorType::F32, &[0; 4]),
                ("v.std_scale", &[1], GgufTensorType::F32, &[0; 4]),
                (
                    "mm.input_projection.weight",
                    &[2, 2],
                    GgufTensorType::BF16,
                    &[0; 8],
                ),
            ],
        ),
    )
    .unwrap();
}

pub(crate) fn build_minimal_gguf(
    metadata: &[(&str, GgufMetaValue)],
    tensors: &[(&str, &[u64], GgufTensorType, &[u8])],
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version
    buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
    // Tensor count
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    // Metadata count
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Write metadata KV pairs
    for (key, value) in metadata {
        // Key string: len(u64) + bytes
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());

        match value {
            GgufMetaValue::Uint32(v) => {
                buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::String(s) => {
                buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            GgufMetaValue::Float32(v) => {
                buf.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::Int32(v) => {
                buf.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::Bool(v) => {
                buf.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
                buf.push(u8::from(*v));
            }
            GgufMetaValue::ArrayString(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    buf.extend_from_slice(v.as_bytes());
                }
            }
            GgufMetaValue::ArrayI32(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
                buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            GgufMetaValue::ArrayU32(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
                buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            GgufMetaValue::ArrayU64(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
                buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            GgufMetaValue::ArrayI64(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::Int64 as u32).to_le_bytes());
                buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            _ => panic!("Unsupported metadata type in test helper"),
        }
    }

    // Compute tensor data: collect offsets
    let alignment: u64 = GGUF_DEFAULT_ALIGNMENT;
    let mut tensor_data_parts: Vec<&[u8]> = Vec::new();
    let mut tensor_offsets: Vec<u64> = Vec::new();
    let mut current_offset: u64 = 0;

    for (_, _, _, data) in tensors {
        // Align offset
        let remainder = current_offset % alignment;
        if remainder != 0 {
            current_offset += alignment - remainder;
        }
        tensor_offsets.push(current_offset);
        tensor_data_parts.push(data);
        current_offset += data.len() as u64;
    }

    // Write tensor info descriptors
    for (i, (name, dims, ttype, _)) in tensors.iter().enumerate() {
        // Name string
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        // n_dims
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        // dims
        for d in *dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        // type
        buf.extend_from_slice(&(*ttype as u32).to_le_bytes());
        // offset
        buf.extend_from_slice(&tensor_offsets[i].to_le_bytes());
    }

    // Pad to alignment for data section
    let remainder = buf.len() as u64 % alignment;
    if remainder != 0 {
        let padding = alignment - remainder;
        buf.extend(vec![0u8; padding as usize]);
    }

    // Write tensor data
    let data_start = buf.len();
    for (i, data) in tensor_data_parts.iter().enumerate() {
        let target_offset = data_start + tensor_offsets[i] as usize;
        while buf.len() < target_offset {
            buf.push(0);
        }
        buf.extend_from_slice(data);
    }

    buf
}

/// Header of a structurally complete standalone `qwen35moe` checkpoint:
/// every key `validate_qwen35_standalone_geometry` and the MoE loader
/// require, with dims small enough to hand-write tensor payloads.
pub(crate) fn qwen35_moe_standalone_metadata() -> Vec<(&'static str, GgufMetaValue)> {
    vec![
        (
            "general.architecture",
            GgufMetaValue::String("qwen35moe".to_string()),
        ),
        ("qwen35moe.embedding_length", GgufMetaValue::Uint32(4)),
        ("qwen35moe.block_count", GgufMetaValue::Uint32(2)),
        ("qwen35moe.attention.head_count", GgufMetaValue::Uint32(2)),
        (
            "qwen35moe.attention.head_count_kv",
            GgufMetaValue::Uint32(1),
        ),
        ("qwen35moe.attention.key_length", GgufMetaValue::Uint32(2)),
        ("qwen35moe.rope.dimension_count", GgufMetaValue::Uint32(2)),
        ("qwen35moe.expert_count", GgufMetaValue::Uint32(4)),
        ("qwen35moe.expert_used_count", GgufMetaValue::Uint32(2)),
        (
            "qwen35moe.expert_feed_forward_length",
            GgufMetaValue::Uint32(8),
        ),
        (
            "qwen35moe.expert_shared_feed_forward_length",
            GgufMetaValue::Uint32(8),
        ),
        ("qwen35moe.ssm.state_size", GgufMetaValue::Uint32(2)),
        ("qwen35moe.ssm.inner_size", GgufMetaValue::Uint32(4)),
        ("qwen35moe.ssm.time_step_rank", GgufMetaValue::Uint32(2)),
        ("qwen35moe.ssm.group_count", GgufMetaValue::Uint32(1)),
        ("qwen35moe.ssm.conv_kernel", GgufMetaValue::Uint32(4)),
        (
            "qwen35moe.full_attention_interval",
            GgufMetaValue::Uint32(2),
        ),
    ]
}

pub(crate) fn build_qwen35_moe_fixture(
    tensors: &[(&str, &[u64], GgufTensorType, Vec<u8>)],
) -> Vec<u8> {
    let metadata = qwen35_moe_standalone_metadata();
    let descriptors: Vec<(&str, &[u64], GgufTensorType, &[u8])> = tensors
        .iter()
        .map(|(name, dims, ty, data)| (*name, *dims, *ty, data.as_slice()))
        .collect();
    build_minimal_gguf(&metadata, &descriptors)
}

pub(crate) fn muse_glimmer_dflash_metadata() -> HashMap<String, GgufMetaValue> {
    HashMap::from([(
        "general.architecture".to_string(),
        GgufMetaValue::String("dflash".to_string()),
    )])
}

pub(crate) fn complete_muse_glimmer_dflash_metadata() -> HashMap<String, GgufMetaValue> {
    let mut metadata = muse_glimmer_dflash_metadata();
    for (key, value) in [
        ("dflash.block_count", 5),
        ("dflash.block_size", 16),
        ("dflash.embedding_length", 6656),
        ("dflash.feed_forward_length", 1536),
        ("dflash.attention.head_count", 8),
        ("dflash.attention.head_count_kv", 2),
        ("dflash.attention.key_length", 64),
        ("dflash.attention.sliding_window", 2048),
        ("dflash.context_length", 131_072),
        ("tokenizer.ggml.mask_token_id", 201_818),
    ] {
        metadata.insert(key.to_string(), GgufMetaValue::Uint32(value));
    }
    metadata.insert(
        "dflash.target_layers".to_string(),
        GgufMetaValue::ArrayU32(vec![1, 8, 16]),
    );
    metadata.insert(
        "dflash.attention.layer_norm_rms_epsilon".to_string(),
        GgufMetaValue::Float32(1e-6),
    );
    metadata.insert(
        "dflash.rope.freq_base".to_string(),
        GgufMetaValue::Float32(10_000.0),
    );
    metadata
}

/// Families of the import fixtures; each maps to one native prepare entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImportFixtureFamily {
    Qwen35,
    Qwen35Moe,
    Gemma4,
    MuseGlimmer,
}

pub(crate) struct ImportFixture {
    pub name: &'static str,
    pub family: ImportFixtureFamily,
    pub main: PathBuf,
    pub companion: Option<PathBuf>,
}

pub(crate) const IMPORT_FIXTURE_NAMES: [&str; 5] = [
    "qwen35-dense-untied",
    "qwen35-dense-tied",
    "qwen35moe-3d-experts",
    "gemma4-mmproj",
    "muse-draft",
];

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

pub(crate) fn fixture_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..len).map(|_| lcg_next(&mut state) as u8).collect()
}

fn fixture_f32(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed ^ 0x51_7cc1_b727_220a;
    (0..count)
        .map(|_| lo + (hi - lo) * ((lcg_next(&mut state) % 65_521) as f32 / 65_520.0))
        .collect()
}

fn f16_bits(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_bits().to_le_bytes()
}

/// Random block payload for `ty` covering `elements` values, with every f16
/// scale field overwritten by a small finite value so dequantized values stay
/// finite on every path.
pub(crate) fn fixture_payload(ty: GgufTensorType, elements: usize, seed: u64) -> Vec<u8> {
    let (block_elems, block_bytes, scale_offsets): (usize, usize, &[usize]) = match ty {
        GgufTensorType::F32 => {
            return fixture_f32(seed, elements, -1.0, 1.0)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
        }
        GgufTensorType::BF16 => {
            return fixture_f32(seed, elements, -1.0, 1.0)
                .into_iter()
                .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                .collect();
        }
        GgufTensorType::Q4K => (256, 144, &[0, 2]),
        GgufTensorType::Q5K => (256, 176, &[0, 2]),
        GgufTensorType::Q6K => (256, 210, &[208]),
        GgufTensorType::Q8_0 => (32, 34, &[0]),
        GgufTensorType::Q4_0 => (32, 18, &[0]),
        other => panic!("no fixture payload for {other:?}"),
    };
    assert_eq!(elements % block_elems, 0, "{ty:?} needs whole blocks");
    let blocks = elements / block_elems;
    let mut data = fixture_bytes(seed, blocks * block_bytes);
    let scales = fixture_f32(
        seed.wrapping_add(1),
        blocks * scale_offsets.len(),
        0.002,
        0.03,
    );
    for block in 0..blocks {
        for (j, offset) in scale_offsets.iter().enumerate() {
            let at = block * block_bytes + offset;
            data[at..at + 2].copy_from_slice(&f16_bits(scales[block * scale_offsets.len() + j]));
        }
    }
    data
}

type FixtureTensor = (&'static str, Vec<u64>, GgufTensorType);

fn write_fixture_gguf(
    path: &Path,
    metadata: &[(&str, GgufMetaValue)],
    tensors: &[FixtureTensor],
    seed: u64,
) {
    let data: Vec<Vec<u8>> = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, dims, ty))| {
            let elements = dims.iter().product::<u64>() as usize;
            if name.ends_with("ssm_a") {
                return fixture_f32(seed + i as u64, elements, -2.0, -0.5)
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect();
            }
            fixture_payload(*ty, elements, seed + i as u64)
        })
        .collect();
    let descriptors: Vec<(&str, &[u64], GgufTensorType, &[u8])> = tensors
        .iter()
        .zip(&data)
        .map(|((name, dims, ty), bytes)| (*name, dims.as_slice(), *ty, bytes.as_slice()))
        .collect();
    fs::write(path, build_minimal_gguf(metadata, &descriptors)).unwrap();
}

fn qwen35_dense_import_metadata() -> Vec<(&'static str, GgufMetaValue)> {
    vec![
        (
            "general.architecture",
            GgufMetaValue::String("qwen35".into()),
        ),
        ("qwen35.embedding_length", GgufMetaValue::Uint32(256)),
        ("qwen35.block_count", GgufMetaValue::Uint32(2)),
        ("qwen35.attention.head_count", GgufMetaValue::Uint32(2)),
        ("qwen35.attention.head_count_kv", GgufMetaValue::Uint32(1)),
        ("qwen35.attention.key_length", GgufMetaValue::Uint32(128)),
        ("qwen35.rope.dimension_count", GgufMetaValue::Uint32(64)),
        ("qwen35.ssm.state_size", GgufMetaValue::Uint32(128)),
        ("qwen35.ssm.inner_size", GgufMetaValue::Uint32(256)),
        ("qwen35.ssm.time_step_rank", GgufMetaValue::Uint32(2)),
        ("qwen35.ssm.group_count", GgufMetaValue::Uint32(1)),
        ("qwen35.ssm.conv_kernel", GgufMetaValue::Uint32(4)),
        ("qwen35.full_attention_interval", GgufMetaValue::Uint32(2)),
    ]
}

fn qwen35_dense_import_tensors() -> Vec<FixtureTensor> {
    use GgufTensorType::*;
    vec![
        ("token_embd.weight", vec![256, 64], Q8_0),
        ("output_norm.weight", vec![256], F32),
        ("blk.0.attn_norm.weight", vec![256], F32),
        ("blk.0.attn_qkv.weight", vec![256, 128], Q4K),
        ("blk.0.attn_gate.weight", vec![256, 64], Q5K),
        ("blk.0.ssm_a", vec![2], F32),
        ("blk.0.ssm_conv1d.weight", vec![4, 128], F32),
        ("blk.0.ssm_out.weight", vec![256, 256], Q6K),
        ("blk.0.post_attention_norm.weight", vec![256], F32),
        ("blk.0.ffn_gate.weight", vec![256, 32], Q4K),
        ("blk.0.ffn_up.weight", vec![256, 64], Q4K),
        ("blk.0.ffn_down.weight", vec![512, 256], Q5K),
        ("blk.1.attn_norm.weight", vec![256], F32),
        ("blk.1.attn_q.weight", vec![256, 512], Q4K),
        ("blk.1.attn_k.weight", vec![256, 128], Q8_0),
        ("blk.1.attn_v.weight", vec![256, 128], Q6K),
        ("blk.1.attn_q_norm.weight", vec![128], F32),
        ("blk.1.attn_k_norm.weight", vec![128], F32),
        ("blk.1.attn_output.weight", vec![256, 256], Q4K),
        ("blk.1.ffn_up.weight", vec![256, 64], BF16),
        ("output.weight", vec![256, 64], Q6K),
    ]
}

fn qwen35moe_import_tensors() -> Vec<FixtureTensor> {
    use GgufTensorType::*;
    vec![
        ("token_embd.weight", vec![256, 64], Q8_0),
        ("output_norm.weight", vec![4], BF16),
        ("output.weight", vec![256, 64], Q4K),
        ("blk.0.ffn_gate_inp.weight", vec![4, 4], F32),
        ("blk.0.ffn_gate_inp_shexp.weight", vec![4], F32),
        ("blk.0.ffn_gate_exps.weight", vec![256, 64, 2], Q4K),
        ("blk.0.ffn_up_exps.weight", vec![256, 64, 2], Q5K),
        ("blk.0.ffn_down_exps.weight", vec![256, 64, 2], Q6K),
        ("blk.0.ffn_gate_shexp.weight", vec![256, 64], Q4K),
        ("blk.0.ffn_up_shexp.weight", vec![256, 32], Q5K),
        ("blk.0.ffn_down_shexp.weight", vec![256, 64], BF16),
    ]
}

/// Writes import fixture `name` under `dir` (created) and returns its paths.
pub(crate) fn write_import_fixture(name: &str, dir: &Path) -> ImportFixture {
    use GgufTensorType::*;
    fs::create_dir_all(dir).unwrap();
    match name {
        "qwen35-dense-untied" | "qwen35-dense-tied" => {
            let tied = name.ends_with("tied") && !name.ends_with("untied");
            if tied {
                fs::write(
                    dir.join("config.json"),
                    serde_json::to_string_pretty(&serde_json::json!({
                        "model_type": "qwen3_5",
                        "hidden_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 128,
                        "linear_num_value_heads": 2,
                        "linear_num_key_heads": 1,
                        "linear_key_head_dim": 128,
                        "linear_value_head_dim": 128,
                        "linear_conv_kernel_dim": 4,
                        "full_attention_interval": 2,
                        "partial_rotary_factor": 0.5,
                        "layer_types": ["linear_attention", "full_attention"],
                        "qwen35_gguf_gdn_layout": "tiled",
                        "tie_word_embeddings": true
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
            let main = dir.join(format!("{name}.gguf"));
            write_fixture_gguf(
                &main,
                &qwen35_dense_import_metadata(),
                &qwen35_dense_import_tensors(),
                if tied { 200 } else { 100 },
            );
            ImportFixture {
                name: if tied {
                    "qwen35-dense-tied"
                } else {
                    "qwen35-dense-untied"
                },
                family: ImportFixtureFamily::Qwen35,
                main,
                companion: None,
            }
        }
        "qwen35moe-3d-experts" => {
            let main = dir.join("qwen35moe-3d-experts.gguf");
            write_fixture_gguf(
                &main,
                &qwen35_moe_standalone_metadata(),
                &qwen35moe_import_tensors(),
                300,
            );
            ImportFixture {
                name: "qwen35moe-3d-experts",
                family: ImportFixtureFamily::Qwen35Moe,
                main,
                companion: None,
            }
        }
        "gemma4-mmproj" => {
            fs::write(
                dir.join("config.json"),
                r#"{"model_type":"gemma4_unified","text_config":{"num_hidden_layers":0},"vision_config":{}}"#,
            )
            .unwrap();
            fs::write(dir.join("tokenizer.json"), "{}").unwrap();
            let main = dir.join("gemma4-mmproj.gguf");
            write_fixture_gguf(
                &main,
                &[(
                    "general.architecture",
                    GgufMetaValue::String("gemma4".into()),
                )],
                &[
                    ("token_embd.weight", vec![256, 64], Q6K),
                    ("output.weight", vec![256, 64], Q4_0),
                    ("blk.0.attn_norm.weight", vec![256], F32),
                    ("blk.0.ffn_up.weight", vec![256, 64], Q4K),
                    ("blk.0.ffn_down.weight", vec![256, 64], Q5K),
                    ("blk.0.attn_q.weight", vec![256, 32], Q4K),
                ],
                400,
            );
            let companion = dir.join("mmproj-gemma.gguf");
            write_fixture_gguf(
                &companion,
                &[
                    ("general.architecture", GgufMetaValue::String("clip".into())),
                    (
                        "clip.vision.projector_type",
                        GgufMetaValue::String("gemma4uv".into()),
                    ),
                ],
                &[("mm.input_projection.weight", vec![4, 4], BF16)],
                450,
            );
            ImportFixture {
                name: "gemma4-mmproj",
                family: ImportFixtureFamily::Gemma4,
                main,
                companion: Some(companion),
            }
        }
        "muse-draft" => {
            use crate::models::muse_glimmer::config::fixtures::{config_json, text_config_json};
            let text = text_config_json(2)
                .replace("6656", "256")
                .replace("19968", "512")
                .replace("\"num_attention_heads\": 32", "\"num_attention_heads\": 2");
            fs::write(dir.join("config.json"), config_json(&text)).unwrap();
            fs::write(dir.join("tokenizer.json"), "{}").unwrap();
            let main = dir.join("muse-Q4_K_M.gguf");
            write_fixture_gguf(
                &main,
                &[(
                    "general.architecture",
                    GgufMetaValue::String("muse-glimmer".into()),
                )],
                &[
                    ("blk.0.ffn_up.weight", vec![256, 64], Q4K),
                    ("blk.0.ffn_down.weight", vec![512, 256], Q5K),
                    ("blk.0.attn_norm.weight", vec![256], F32),
                ],
                500,
            );
            let companion = dir.join("dflash-muse-Q4_K_M.gguf");
            let mut metadata = complete_muse_glimmer_dflash_metadata();
            for (key, value) in [
                ("dflash.block_count", 1),
                ("dflash.block_size", 2),
                ("dflash.embedding_length", 256),
                ("dflash.feed_forward_length", 512),
                ("dflash.attention.head_count", 2),
                ("dflash.attention.head_count_kv", 1),
                ("dflash.attention.key_length", 128),
            ] {
                metadata.insert(key.into(), GgufMetaValue::Uint32(value));
            }
            metadata.insert(
                "dflash.target_layers".into(),
                GgufMetaValue::ArrayI32(vec![1]),
            );
            let mut metadata: Vec<_> = metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone()))
                .collect();
            metadata.sort_by(|a, b| a.0.cmp(b.0));
            let tensors: Vec<FixtureTensor> = [
                ("fc.weight", vec![256, 256]),
                ("enc.output_norm.weight", vec![256]),
                ("output_norm.weight", vec![256]),
                ("blk.0.attn_norm.weight", vec![256]),
                ("blk.0.ffn_norm.weight", vec![256]),
                ("blk.0.attn_q_norm.weight", vec![128]),
                ("blk.0.attn_k_norm.weight", vec![128]),
                ("blk.0.attn_q.weight", vec![256, 256]),
                ("blk.0.attn_k.weight", vec![256, 128]),
                ("blk.0.attn_v.weight", vec![256, 128]),
                ("blk.0.attn_output.weight", vec![256, 256]),
                ("blk.0.ffn_gate.weight", vec![256, 512]),
                ("blk.0.ffn_up.weight", vec![256, 512]),
                ("blk.0.ffn_down.weight", vec![512, 256]),
            ]
            .into_iter()
            .map(|(name, dims)| {
                let ty = if dims.len() == 2 { Q4K } else { F32 };
                (name, dims, ty)
            })
            .collect();
            write_fixture_gguf(&companion, &metadata, &tensors, 550);
            ImportFixture {
                name: "muse-draft",
                family: ImportFixtureFamily::MuseGlimmer,
                main,
                companion: Some(companion),
            }
        }
        other => panic!("unknown import fixture '{other}'"),
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// `{name: {dtype, shape, sha256}}` of a safetensors file, the same JSON the
/// real-file goldens use (dtype spelled as safetensors spells it).
pub(crate) fn safetensors_manifest(path: &Path) -> serde_json::Value {
    let bytes = fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let base = 8 + header_len;
    let mut out = BTreeMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let offsets = info["data_offsets"].as_array().unwrap();
        let start = base + offsets[0].as_u64().unwrap() as usize;
        let end = base + offsets[1].as_u64().unwrap() as usize;
        out.insert(
            name,
            serde_json::json!({
                "dtype": info["dtype"],
                "shape": info["shape"],
                "sha256": sha256_hex(&bytes[start..end]),
            }),
        );
    }
    serde_json::to_value(out).unwrap()
}

/// Config-normalized prefixes whose quantization entry carries `"layout": "t64"`.
pub(crate) fn t64_prefixes(config: &serde_json::Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for block in ["quantization", "quantization_config"] {
        if let Some(entries) = config.get(block).and_then(serde_json::Value::as_object) {
            for (key, value) in entries {
                if value.get("layout").and_then(serde_json::Value::as_str) == Some("t64") {
                    out.insert(key.clone());
                }
            }
        }
    }
    out
}

fn write_json(path: &Path, value: &serde_json::Value) {
    let mut text = serde_json::to_string_pretty(value).unwrap();
    text.push('\n');
    fs::write(path, text).unwrap();
}

/// Convert a fixture through the retained `mlx convert` oracle, without the
/// native asset cache or a lease.
pub(crate) async fn convert_import_fixture(fixture: &ImportFixture, output_dir: &Path) -> PathBuf {
    use super::gguf::{GgufConversionOptions, convert_gguf_to_safetensors};

    let parent = fixture.main.parent().unwrap();
    let options = |input: &Path,
                   config_source_dir: Option<String>,
                   output_filename: Option<&str>,
                   native_qwen35_layout: Option<bool>| {
        GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: output_dir.to_string_lossy().into_owned(),
            config_source_dir,
            dtype: Some("bfloat16".to_string()),
            verbose: None,
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: output_filename.map(str::to_string),
            vlm_key_prefix: None,
            quant_mxfp: None,
            import_k_quants: Some(true),
            native_qwen35_layout,
        }
    };
    let config_source_dir = parent
        .join("config.json")
        .is_file()
        .then(|| parent.to_string_lossy().into_owned());
    convert_gguf_to_safetensors(options(
        &fixture.main,
        config_source_dir,
        None,
        Some(matches!(
            fixture.family,
            ImportFixtureFamily::Qwen35 | ImportFixtureFamily::Qwen35Moe
        )),
    ))
    .await
    .unwrap_or_else(|error| panic!("convert {}: {}", fixture.name, error.reason));

    if let Some(companion) = &fixture.companion {
        let filename = match fixture.family {
            ImportFixtureFamily::MuseGlimmer => "draft.safetensors",
            ImportFixtureFamily::Qwen35
            | ImportFixtureFamily::Qwen35Moe
            | ImportFixtureFamily::Gemma4 => "vision.safetensors",
        };
        convert_gguf_to_safetensors(options(
            companion,
            Some(parent.to_string_lossy().into_owned()),
            Some(filename),
            None,
        ))
        .await
        .unwrap_or_else(|error| panic!("convert {} companion: {}", fixture.name, error.reason));
    }
    output_dir.to_path_buf()
}

/// With `MLX_CAPTURE_FIXTURE_GOLDEN=<dir>`, converts every import fixture
/// through the explicit converter oracle and writes, per fixture: every
/// non-weight file of the entry, one
/// `<stem>.manifest.json` per safetensors file, `t64.json`, and
/// `fixture.json` (sha256 of the fixture GGUFs, to catch builder drift).
#[tokio::test]
#[ignore = "writes goldens; set MLX_CAPTURE_FIXTURE_GOLDEN"]
async fn capture_base_fixture_goldens() {
    let Some(out_root) = std::env::var_os("MLX_CAPTURE_FIXTURE_GOLDEN").map(PathBuf::from) else {
        panic!("set MLX_CAPTURE_FIXTURE_GOLDEN=<golden dir>");
    };
    let work = std::env::temp_dir().join(format!("mlx-fixture-golden-{}", uuid::Uuid::new_v4()));
    for name in IMPORT_FIXTURE_NAMES {
        let fixture = write_import_fixture(name, &work.join("src").join(name));
        let entry = convert_import_fixture(&fixture, &work.join("cache").join(name)).await;
        let out = out_root.join(name);
        if out.exists() {
            fs::remove_dir_all(&out).unwrap();
        }
        fs::create_dir_all(&out).unwrap();
        let mut files: Vec<_> = fs::read_dir(&entry)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for file in &files {
            let file_name = file.file_name().unwrap().to_string_lossy().into_owned();
            if file_name.starts_with('.') {
                continue;
            }
            if let Some(stem) = file_name.strip_suffix(".safetensors") {
                write_json(
                    &out.join(format!("{stem}.manifest.json")),
                    &safetensors_manifest(file),
                );
            } else {
                fs::copy(file, out.join(&file_name)).unwrap();
            }
        }
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(entry.join("config.json")).unwrap()).unwrap();
        write_json(
            &out.join("t64.json"),
            &serde_json::to_value(t64_prefixes(&config)).unwrap(),
        );
        let mut sources = BTreeMap::new();
        sources.insert(
            "main".to_string(),
            serde_json::json!(sha256_hex(&fs::read(&fixture.main).unwrap())),
        );
        sources.insert(
            "companion".to_string(),
            fixture
                .companion
                .as_ref()
                .map(|p| serde_json::json!(sha256_hex(&fs::read(p).unwrap())))
                .unwrap_or(serde_json::Value::Null),
        );
        write_json(
            &out.join("fixture.json"),
            &serde_json::to_value(sources).unwrap(),
        );
        println!("captured {name}: {} files", files.len());
    }
    fs::remove_dir_all(&work).ok();
}
