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
    let described: Vec<(&str, &[u64], GgufTensorType, u64)> = tensors
        .iter()
        .map(|(name, dims, ty, data)| (*name, *dims, *ty, data.len() as u64))
        .collect();
    let (mut buf, tensor_offsets) = gguf_header(metadata, &described);
    let data_start = buf.len();
    for (i, (_, _, _, data)) in tensors.iter().enumerate() {
        let target_offset = data_start + tensor_offsets[i] as usize;
        while buf.len() < target_offset {
            buf.push(0);
        }
        buf.extend_from_slice(data);
    }
    buf
}

/// A GGUF file's bytes before its data section (padded to the alignment),
/// and each tensor's offset in that section, for tensors of the given byte
/// lengths.
fn gguf_header(
    metadata: &[(&str, GgufMetaValue)],
    tensors: &[(&str, &[u64], GgufTensorType, u64)],
) -> (Vec<u8>, Vec<u64>) {
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
            GgufMetaValue::Uint16(v) => {
                buf.extend_from_slice(&(GgufValueType::Uint16 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::Int16(v) => {
                buf.extend_from_slice(&(GgufValueType::Int16 as u32).to_le_bytes());
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
            GgufMetaValue::Uint64(v) => {
                buf.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::Int64(v) => {
                buf.extend_from_slice(&(GgufValueType::Int64 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::Uint8(v) => {
                buf.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
                buf.push(*v);
            }
            GgufMetaValue::Float64(v) => {
                buf.extend_from_slice(&(GgufValueType::Float64 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetaValue::ArrayF32(values) => {
                buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                buf.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
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
    let mut tensor_offsets: Vec<u64> = Vec::new();
    let mut current_offset: u64 = 0;

    for (_, _, _, len) in tensors {
        // Align offset
        let remainder = current_offset % alignment;
        if remainder != 0 {
            current_offset += alignment - remainder;
        }
        tensor_offsets.push(current_offset);
        current_offset += len;
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

    (buf, tensor_offsets)
}

/// A GGUF read back whole: metadata sorted by key, tensors in file order with
/// their payload bytes.
pub(crate) struct GgufParts {
    pub metadata: Vec<(String, GgufMetaValue)>,
    pub tensors: Vec<(String, Vec<u64>, GgufTensorType, Vec<u8>)>,
}

impl GgufParts {
    pub(crate) fn read(path: &Path) -> Self {
        let header = super::gguf::parse_gguf(path).unwrap();
        let bytes = fs::read(path).unwrap();
        let mut metadata: Vec<(String, GgufMetaValue)> = header
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        metadata.sort_by(|a, b| a.0.cmp(&b.0));
        let tensors = header
            .tensors
            .iter()
            .map(|t| {
                let start = (header.data_offset + t.offset) as usize;
                let end = start + t.data_size().unwrap() as usize;
                (
                    t.name.clone(),
                    t.dims.clone(),
                    t.tensor_type,
                    bytes[start..end].to_vec(),
                )
            })
            .collect();
        Self { metadata, tensors }
    }

    pub(crate) fn set(&mut self, key: &str, value: GgufMetaValue) {
        self.metadata.retain(|(k, _)| k != key);
        self.metadata.push((key.to_string(), value));
        self.metadata.sort_by(|a, b| a.0.cmp(&b.0));
    }

    /// Adds a tensor of zero bytes.
    pub(crate) fn add_tensor(&mut self, name: &str, dims: &[u64], ty: GgufTensorType) {
        let info = super::gguf::GgufTensorInfo {
            name: name.to_string(),
            n_dims: dims.len() as u32,
            dims: dims.to_vec(),
            tensor_type: ty,
            offset: 0,
        };
        let len = info.data_size().unwrap() as usize;
        self.tensors
            .push((name.to_string(), dims.to_vec(), ty, vec![0; len]));
    }

    pub(crate) fn bytes(&self) -> Vec<u8> {
        let metadata: Vec<(&str, GgufMetaValue)> = self
            .metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let tensors: Vec<(&str, &[u64], GgufTensorType, &[u8])> = self
            .tensors
            .iter()
            .map(|(n, d, t, b)| (n.as_str(), d.as_slice(), *t, b.as_slice()))
            .collect();
        build_minimal_gguf(&metadata, &tensors)
    }

    pub(crate) fn write(&self, path: &Path) {
        fs::write(path, self.bytes()).unwrap();
    }
}

/// `src` re-split as llama-gguf-split writes it into `{prefix}-0000K-of-0000N.gguf`
/// under `dir`: the tensors in file order cut into `parts` runs, part 1 keeps
/// the metadata, every part `split.no` / `split.count` u16 and
/// `split.tensors.count` i32.
pub(crate) fn split_import_fixture(
    src: &Path,
    dir: &Path,
    prefix: &str,
    parts: usize,
) -> Vec<PathBuf> {
    let whole = GgufParts::read(src);
    let total = whole.tensors.len();
    assert!(
        parts >= 2 && parts <= total,
        "{parts} parts of {total} tensors"
    );
    fs::create_dir_all(dir).unwrap();
    let per = total.div_ceil(parts);
    (0..parts)
        .map(|part| {
            let mut metadata = if part == 0 {
                whole.metadata.clone()
            } else {
                Vec::new()
            };
            metadata.push(("split.no".into(), GgufMetaValue::Uint16(part as u16)));
            metadata.push(("split.count".into(), GgufMetaValue::Uint16(parts as u16)));
            metadata.push((
                "split.tensors.count".into(),
                GgufMetaValue::Int32(total as i32),
            ));
            let end = ((part + 1) * per).min(total);
            let piece = GgufParts {
                metadata,
                tensors: whole.tensors[(part * per).min(end)..end].to_vec(),
            };
            let path = dir.join(format!("{prefix}-{:05}-of-{parts:05}.gguf", part + 1));
            piece.write(&path);
            path
        })
        .collect()
}

fn split_tensor_payloads(tensors: &[(&str, &[u64])]) -> Vec<Vec<u8>> {
    tensors
        .iter()
        .enumerate()
        .map(|(seed, (_, dims))| {
            let elements = dims.iter().product::<u64>() as usize;
            fixture_payload(GgufTensorType::F32, elements, seed as u64)
        })
        .collect()
}

/// A split GGUF as llama-gguf-split writes it: `{prefix}-0000K-of-0000N.gguf`,
/// one F32 tensor per part, `metadata` in part 1 only, split.no / split.count
/// u16 in every part, and split.tensors.count i32 in every part when
/// `tensors_count`.
pub(crate) fn write_split_gguf(
    dir: &Path,
    prefix: &str,
    metadata: &[(&str, GgufMetaValue)],
    tensors: &[(&str, &[u64])],
    tensors_count: bool,
) -> Vec<PathBuf> {
    let count = tensors.len();
    let payloads = split_tensor_payloads(tensors);
    fs::create_dir_all(dir).unwrap();
    (0..count)
        .map(|part| {
            let mut keys: Vec<(&str, GgufMetaValue)> = if part == 0 {
                metadata.to_vec()
            } else {
                Vec::new()
            };
            keys.push(("split.no", GgufMetaValue::Uint16(part as u16)));
            keys.push(("split.count", GgufMetaValue::Uint16(count as u16)));
            if tensors_count {
                keys.push(("split.tensors.count", GgufMetaValue::Int32(count as i32)));
            }
            let (name, dims) = tensors[part];
            let path = dir.join(format!("{prefix}-{:05}-of-{count:05}.gguf", part + 1));
            fs::write(
                &path,
                build_minimal_gguf(&keys, &[(name, dims, GgufTensorType::F32, &payloads[part])]),
            )
            .unwrap();
            path
        })
        .collect()
}

/// The one file llama-gguf-split --merge makes of that split: every tensor,
/// part 1's keys, split.count 0.
pub(crate) fn write_merged_gguf(
    path: &Path,
    metadata: &[(&str, GgufMetaValue)],
    tensors: &[(&str, &[u64])],
) {
    let payloads = split_tensor_payloads(tensors);
    let mut keys = metadata.to_vec();
    keys.push(("split.no", GgufMetaValue::Uint16(0)));
    keys.push(("split.count", GgufMetaValue::Uint16(0)));
    keys.push((
        "split.tensors.count",
        GgufMetaValue::Int32(tensors.len() as i32),
    ));
    let described: Vec<(&str, &[u64], GgufTensorType, &[u8])> = tensors
        .iter()
        .zip(&payloads)
        .map(|((name, dims), payload)| (*name, *dims, GgufTensorType::F32, payload.as_slice()))
        .collect();
    fs::write(path, build_minimal_gguf(&keys, &described)).unwrap();
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
        other => block_layout(other).unwrap_or_else(|| panic!("no fixture payload for {other:?}")),
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

/// A block type's values per block, bytes per block and the offsets of its
/// f16 scale fields.
fn block_layout(ty: GgufTensorType) -> Option<(usize, usize, &'static [usize])> {
    Some(match ty {
        GgufTensorType::Q2K => (256, 84, &[80, 82]),
        GgufTensorType::Q3K => (256, 110, &[108]),
        GgufTensorType::Q4K => (256, 144, &[0, 2]),
        GgufTensorType::Q5K => (256, 176, &[0, 2]),
        GgufTensorType::Q6K => (256, 210, &[208]),
        GgufTensorType::Q8_0 => (32, 34, &[0]),
        GgufTensorType::Q4_0 => (32, 18, &[0]),
        GgufTensorType::IQ4NL => (32, 18, &[0]),
        GgufTensorType::IQ4XS => (256, 136, &[0]),
        GgufTensorType::IQ3S => (256, 110, &[0]),
        _ => return None,
    })
}

/// Bytes [`fixture_payload`] writes for `elements` values of `ty`.
fn payload_len(ty: GgufTensorType, elements: u64) -> u64 {
    match ty {
        GgufTensorType::F32 => elements * 4,
        GgufTensorType::BF16 => elements * 2,
        other => {
            let (block_elems, block_bytes, _) =
                block_layout(other).unwrap_or_else(|| panic!("no fixture payload for {other:?}"));
            elements / block_elems as u64 * block_bytes as u64
        }
    }
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

/// The checked-in converter output per import fixture, pinned to the live
/// converter by `converter_output_equals_base_golden`.
pub(crate) fn import_golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/testdata/gguf_import_golden")
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
    if !crate::test_support::in_own_process(
        "utils::gguf_test_fixtures::capture_base_fixture_goldens",
    ) {
        return;
    }
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

/// Invented GGUF GPT-2 vocabulary in the Qwen layout: the 256 byte-level
/// symbols as NORMAL rows, then CONTROL `<|endoftext|>` / `<|im_start|>` /
/// `<|im_end|>`, USER_DEFINED `<tool_call>` / `</tool_call>` / `<think>` /
/// `</think>`, and one UNUSED row. `overrides` replace a row's type by token.
pub(crate) fn qwen_tag_tokenizer_metadata(
    overrides: &[(&str, i32)],
) -> HashMap<String, GgufMetaValue> {
    let mut alphabet: Vec<char> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut tokens: Vec<String> = alphabet.iter().map(char::to_string).collect();
    let mut types = vec![1; tokens.len()];
    for (token, token_type) in [
        ("<|endoftext|>", 3),
        ("<|im_start|>", 3),
        ("<|im_end|>", 3),
        ("<tool_call>", 4),
        ("</tool_call>", 4),
        ("<think>", 4),
        ("</think>", 4),
        ("<|unused_0|>", 5),
    ] {
        tokens.push(token.to_string());
        types.push(
            overrides
                .iter()
                .find(|(name, _)| *name == token)
                .map_or(token_type, |(_, kind)| *kind),
        );
    }
    HashMap::from([
        (
            "tokenizer.ggml.model".to_string(),
            GgufMetaValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            GgufMetaValue::String("qwen35".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufMetaValue::ArrayString(tokens),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            GgufMetaValue::ArrayI32(types),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            GgufMetaValue::ArrayString(Vec::new()),
        ),
    ])
}

/// The dimensions of a synthetic Qwen3.5 checkpoint; the role-to-type class
/// tables are fixed per family, the sizes are the checkpoint's or small ones.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35Geometry {
    pub hidden: u64,
    pub vocab: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub key: u64,
    pub rope: u64,
    pub ssm_inner: u64,
    pub ssm_rank: u64,
    pub ssm_groups: u64,
    pub ssm_state: u64,
    /// Dense FFN width; 0 for MoE.
    pub ffn: u64,
    pub experts: u64,
    pub experts_used: u64,
    pub expert_ffn: u64,
    /// Decoder blocks before the MTP block.
    pub blocks: u64,
}

impl Qwen35Geometry {
    /// Ornith-1.5-35B Q4_K_M, with 4 decoder blocks.
    pub(crate) const ORNITH_35B: Self = Self {
        hidden: 2048,
        vocab: 248_320,
        heads: 16,
        kv_heads: 2,
        key: 256,
        rope: 64,
        ssm_inner: 4096,
        ssm_rank: 32,
        ssm_groups: 16,
        ssm_state: 128,
        ffn: 0,
        experts: 256,
        experts_used: 8,
        expert_ffn: 512,
        blocks: 4,
    };

    /// Qwen3.8-27B UD-Q4_K_XL, with 12 decoder blocks: enough for every
    /// (role, type) class of its header once.
    pub(crate) const QWEN38_27B: Self = Self {
        hidden: 5120,
        vocab: 248_320,
        heads: 24,
        kv_heads: 4,
        key: 256,
        rope: 64,
        ssm_inner: 6144,
        ssm_rank: 48,
        ssm_groups: 16,
        ssm_state: 128,
        ffn: 17_408,
        experts: 0,
        experts_used: 0,
        expert_ffn: 0,
        blocks: 12,
    };

    /// The same class tables at dimensions small enough for a default test.
    pub(crate) const fn small(self) -> Self {
        let moe = self.experts > 0;
        Self {
            hidden: 256,
            vocab: 512,
            heads: 4,
            kv_heads: 1,
            key: 128,
            rope: 64,
            ssm_inner: 256,
            ssm_rank: 2,
            ssm_groups: 2,
            ssm_state: 128,
            ffn: if moe { 0 } else { 512 },
            experts: if moe { 8 } else { 0 },
            experts_used: if moe { 2 } else { 0 },
            expert_ffn: if moe { 256 } else { 0 },
            blocks: self.blocks,
        }
    }

    fn moe(&self) -> bool {
        self.experts > 0
    }

    fn arch(&self) -> &'static str {
        if self.moe() { "qwen35moe" } else { "qwen35" }
    }

    fn attention_block(&self, block: u64) -> bool {
        block == self.blocks || (block + 1).is_multiple_of(4)
    }
}

/// Per-role quant types, by the role's block index (cycled): the classes of
/// the real checkpoint's header.
fn qwen35_classes(moe: bool, role: &str) -> &'static [GgufTensorType] {
    use GgufTensorType::*;
    if moe {
        return match role {
            "attn_qkv" | "ffn_down_exps" | "ffn_down_shexp" | "attn_v" => &[Q6K, Q4K],
            "token_embd" | "attn_gate" | "ssm_out" | "ssm_alpha" | "ssm_beta" | "ffn_gate_exps"
            | "ffn_up_exps" | "ffn_gate_shexp" | "ffn_up_shexp" | "attn_q" | "attn_k"
            | "attn_output" | "nextn.eh_proj" => &[Q4K],
            "output" => &[Q6K],
            other => panic!("no class for {other}"),
        };
    }
    match role {
        "attn_qkv" => &[Q5K, Q4K, Q6K, IQ4NL, IQ4XS],
        "attn_gate" => &[Q5K, Q4K, IQ4XS, Q6K],
        "ssm_alpha" | "ssm_beta" => &[Q8_0],
        "ssm_out" => &[Q8_0, Q6K, Q5K, Q4K],
        "ffn_down" => &[IQ4XS, IQ4NL, Q5K, Q4K, IQ3S, Q6K],
        "ffn_gate" => &[IQ4XS, Q4K, Q3K, IQ4NL, Q5K, Q6K],
        "ffn_up" => &[Q3K, IQ4XS, Q5K, Q4K, IQ4NL, Q6K],
        "attn_q" => &[Q5K, Q4K, IQ4XS, Q6K],
        "attn_k" => &[Q4K, Q5K, Q6K, Q8_0],
        "attn_v" => &[Q5K, Q8_0, Q6K],
        "attn_output" => &[Q6K, Q5K, IQ4XS],
        "token_embd" => &[Q4K],
        "output" | "nextn.eh_proj" => &[Q6K],
        other => panic!("no class for {other}"),
    }
}

/// A synthetic checkpoint's header and tensors at `geometry`.
pub(crate) fn qwen35_geometry_checkpoint(
    geometry: &Qwen35Geometry,
) -> (
    Vec<(String, GgufMetaValue)>,
    Vec<(String, Vec<u64>, GgufTensorType)>,
) {
    let g = geometry;
    let arch = g.arch();
    let mut metadata: Vec<(String, GgufMetaValue)> = vec![(
        "general.architecture".into(),
        GgufMetaValue::String(arch.into()),
    )];
    let mut key = |suffix: &str, value: u64| {
        metadata.push((
            format!("{arch}.{suffix}"),
            GgufMetaValue::Uint32(value as u32),
        ));
    };
    key("block_count", g.blocks + 1);
    key("context_length", 262_144);
    key("embedding_length", g.hidden);
    key("vocab_size", g.vocab);
    key("attention.head_count", g.heads);
    key("attention.head_count_kv", g.kv_heads);
    key("attention.key_length", g.key);
    key("attention.value_length", g.key);
    key("rope.dimension_count", g.rope);
    key("nextn_predict_layers", 1);
    key("ssm.conv_kernel", 4);
    key("ssm.state_size", g.ssm_state);
    key("ssm.group_count", g.ssm_groups);
    key("ssm.time_step_rank", g.ssm_rank);
    key("ssm.inner_size", g.ssm_inner);
    key("full_attention_interval", 4);
    if g.moe() {
        key("expert_count", g.experts);
        key("expert_used_count", g.experts_used);
        key("expert_feed_forward_length", g.expert_ffn);
        key("expert_shared_feed_forward_length", g.expert_ffn);
    } else {
        key("feed_forward_length", g.ffn);
    }
    metadata.push((
        format!("{arch}.rope.freq_base"),
        GgufMetaValue::Float32(10_000_000.0),
    ));
    metadata.push((
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        GgufMetaValue::Float32(1e-6),
    ));
    let mut tokenizer: Vec<(String, GgufMetaValue)> =
        qwen_tag_tokenizer_metadata(&[]).into_iter().collect();
    tokenizer.sort_by(|a, b| a.0.cmp(&b.0));
    metadata.extend(tokenizer);
    for (name, id) in [
        ("tokenizer.ggml.bos_token_id", 256),
        ("tokenizer.ggml.padding_token_id", 256),
        ("tokenizer.ggml.eos_token_id", 258),
    ] {
        metadata.push((name.into(), GgufMetaValue::Uint32(id)));
    }
    metadata.push((
        "tokenizer.chat_template".into(),
        GgufMetaValue::String(
            "{% for message in messages %}<|im_start|>{{ message['role'] }}\n\
             {{ message['content'] }}<|im_end|>\n{% endfor %}\
             {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
                .into(),
        ),
    ));

    let qkv = 2 * g.ssm_groups * g.ssm_state + g.ssm_inner;
    let mut tensors = Vec::new();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut quant = |tensors: &mut Vec<(String, Vec<u64>, GgufTensorType)>,
                     name: String,
                     role: &'static str,
                     dims: Vec<u64>| {
        let classes = qwen35_classes(g.moe(), role);
        let index = counts.entry(role).or_default();
        tensors.push((name, dims, classes[*index % classes.len()]));
        *index += 1;
    };
    let float =
        |tensors: &mut Vec<(String, Vec<u64>, GgufTensorType)>, name: String, dims: Vec<u64>| {
            tensors.push((name, dims, GgufTensorType::F32));
        };
    quant(
        &mut tensors,
        "token_embd.weight".into(),
        "token_embd",
        vec![g.hidden, g.vocab],
    );
    float(&mut tensors, "output_norm.weight".into(), vec![g.hidden]);
    quant(
        &mut tensors,
        "output.weight".into(),
        "output",
        vec![g.hidden, g.vocab],
    );
    for block in 0..=g.blocks {
        let b = |suffix: &str| format!("blk.{block}.{suffix}");
        float(&mut tensors, b("attn_norm.weight"), vec![g.hidden]);
        float(
            &mut tensors,
            b("post_attention_norm.weight"),
            vec![g.hidden],
        );
        if g.attention_block(block) {
            quant(
                &mut tensors,
                b("attn_q.weight"),
                "attn_q",
                vec![g.hidden, 2 * g.heads * g.key],
            );
            quant(
                &mut tensors,
                b("attn_k.weight"),
                "attn_k",
                vec![g.hidden, g.kv_heads * g.key],
            );
            quant(
                &mut tensors,
                b("attn_v.weight"),
                "attn_v",
                vec![g.hidden, g.kv_heads * g.key],
            );
            float(&mut tensors, b("attn_q_norm.weight"), vec![g.key]);
            float(&mut tensors, b("attn_k_norm.weight"), vec![g.key]);
            quant(
                &mut tensors,
                b("attn_output.weight"),
                "attn_output",
                vec![g.heads * g.key, g.hidden],
            );
        } else {
            quant(
                &mut tensors,
                b("attn_qkv.weight"),
                "attn_qkv",
                vec![g.hidden, qkv],
            );
            quant(
                &mut tensors,
                b("attn_gate.weight"),
                "attn_gate",
                vec![g.hidden, g.ssm_inner],
            );
            float(&mut tensors, b("ssm_a"), vec![g.ssm_rank]);
            quant(
                &mut tensors,
                b("ssm_alpha.weight"),
                "ssm_alpha",
                vec![g.hidden, g.ssm_rank],
            );
            quant(
                &mut tensors,
                b("ssm_beta.weight"),
                "ssm_beta",
                vec![g.hidden, g.ssm_rank],
            );
            float(&mut tensors, b("ssm_conv1d.weight"), vec![4, qkv]);
            float(&mut tensors, b("ssm_dt.bias"), vec![g.ssm_rank]);
            float(&mut tensors, b("ssm_norm.weight"), vec![g.ssm_state]);
            quant(
                &mut tensors,
                b("ssm_out.weight"),
                "ssm_out",
                vec![g.ssm_inner, g.hidden],
            );
        }
        if g.moe() {
            float(
                &mut tensors,
                b("ffn_gate_inp.weight"),
                vec![g.hidden, g.experts],
            );
            float(&mut tensors, b("ffn_gate_inp_shexp.weight"), vec![g.hidden]);
            for (suffix, role) in [
                ("ffn_gate_exps.weight", "ffn_gate_exps"),
                ("ffn_up_exps.weight", "ffn_up_exps"),
            ] {
                quant(
                    &mut tensors,
                    b(suffix),
                    role,
                    vec![g.hidden, g.expert_ffn, g.experts],
                );
            }
            quant(
                &mut tensors,
                b("ffn_down_exps.weight"),
                "ffn_down_exps",
                vec![g.expert_ffn, g.hidden, g.experts],
            );
            for (suffix, role) in [
                ("ffn_gate_shexp.weight", "ffn_gate_shexp"),
                ("ffn_up_shexp.weight", "ffn_up_shexp"),
            ] {
                quant(&mut tensors, b(suffix), role, vec![g.hidden, g.expert_ffn]);
            }
            quant(
                &mut tensors,
                b("ffn_down_shexp.weight"),
                "ffn_down_shexp",
                vec![g.expert_ffn, g.hidden],
            );
        } else {
            quant(
                &mut tensors,
                b("ffn_gate.weight"),
                "ffn_gate",
                vec![g.hidden, g.ffn],
            );
            quant(
                &mut tensors,
                b("ffn_up.weight"),
                "ffn_up",
                vec![g.hidden, g.ffn],
            );
            quant(
                &mut tensors,
                b("ffn_down.weight"),
                "ffn_down",
                vec![g.ffn, g.hidden],
            );
        }
        if block == g.blocks {
            quant(
                &mut tensors,
                b("nextn.eh_proj.weight"),
                "nextn.eh_proj",
                vec![2 * g.hidden, g.hidden],
            );
            for suffix in [
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
            ] {
                float(&mut tensors, b(suffix), vec![g.hidden]);
            }
        }
    }
    (metadata, tensors)
}

/// Writes a GGUF of `tensors` with [`fixture_payload`] values one tensor
/// at a time; `ssm_a` rows are negative as a real decay is.
pub(crate) fn write_streamed_gguf(
    path: &Path,
    metadata: &[(String, GgufMetaValue)],
    tensors: &[(String, Vec<u64>, GgufTensorType)],
    seed: u64,
) -> std::io::Result<u64> {
    use std::io::Write;
    let metadata: Vec<(&str, GgufMetaValue)> = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.clone()))
        .collect();
    let described: Vec<(&str, &[u64], GgufTensorType, u64)> = tensors
        .iter()
        .map(|(name, dims, ty)| {
            let elements = dims.iter().product::<u64>();
            (
                name.as_str(),
                dims.as_slice(),
                *ty,
                payload_len(*ty, elements),
            )
        })
        .collect();
    let (header, offsets) = gguf_header(&metadata, &described);
    let mut file = std::io::BufWriter::with_capacity(8 << 20, fs::File::create(path)?);
    file.write_all(&header)?;
    let mut written = 0u64;
    for (i, (name, dims, ty)) in tensors.iter().enumerate() {
        let pad = offsets[i] - written;
        file.write_all(&vec![0u8; pad as usize])?;
        let elements = dims.iter().product::<u64>() as usize;
        let data = if name.ends_with("ssm_a") {
            fixture_f32(seed + i as u64, elements, -2.0, -0.5)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect()
        } else {
            fixture_payload(*ty, elements, seed + i as u64)
        };
        file.write_all(&data)?;
        written = offsets[i] + data.len() as u64;
    }
    file.flush()?;
    Ok(header.len() as u64 + written)
}
