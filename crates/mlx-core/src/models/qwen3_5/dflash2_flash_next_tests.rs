//! The Flash Next draft (`incoai/Qwen3.8-Flash-Next-DFlash2`, revision
//! `a22a1744`): its config and safetensors header, and a tiny checkpoint of
//! the same layout.
use super::*;

/// The validated config of the DFlash2 checkpoint directory `path` whose
/// safetensors header names exactly the tensors `load_dflash2` takes, at
/// their shapes, all BF16. Reads the two headers only.
pub(crate) fn read_dflash2_header(path: &Path) -> Result<DFlash2Config> {
    let config = read_dflash2_config(path)?;
    let header = crate::utils::safetensors::SafeTensorsFile::load(resolve_safetensors(path)?)?;
    let expected = expected_tensor_shapes(&config);
    let mut problems = Vec::new();
    for (name, shape) in &expected {
        match header.tensors.get(name) {
            None => problems.push(format!("missing {name}")),
            Some(info) => {
                let found = info.shape.iter().map(|&d| d as i64).collect::<Vec<_>>();
                if &found != shape {
                    problems.push(format!("{name} has shape {found:?}, expected {shape:?}"));
                }
                if !matches!(info.dtype, crate::utils::safetensors::SafeTensorDType::BF16) {
                    problems.push(format!("{name} is {:?}, expected BF16", info.dtype));
                }
            }
        }
    }
    let mut unexpected = header
        .tensors
        .keys()
        .filter(|name| !expected.iter().any(|(want, _)| want == *name))
        .cloned()
        .collect::<Vec<_>>();
    unexpected.sort();
    problems.extend(
        unexpected
            .into_iter()
            .map(|name| format!("unexpected {name}")),
    );
    if !problems.is_empty() {
        return Err(Error::from_reason(format!(
            "DFlash2 checkpoint header mismatch: {}",
            problems.join("; ")
        )));
    }
    Ok(config)
}

/// Qwen3.8-Flash-Next as its GGUF header gives it: `qwen4exp.block_count`
/// 48, `qwen4exp.embedding_length` 2560, 248,320 tokens.
const FLASH_NEXT: DFlash2TargetShape = DFlash2TargetShape {
    hidden_size: 2560,
    vocab_size: 248_320,
    num_layers: 48,
};

/// The draft's `config.json` keys, as the release writes them, at the tiny
/// geometry (hidden 256, 2 layers, vocab 32, 4 target layers).
const TINY_CONFIG: &str = r#"{
  "architectures": ["DFlash2DraftModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 30,
    "selector_rank": 8,
    "selector_top_k": 4,
    "target_layer_ids": [1, 3]
  },
  "dtype": "bfloat16",
  "eos_token_id": 2,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 256,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "is_causal": false,
  "layer_types": ["sliding_attention", "sliding_attention"],
  "max_position_embeddings": 262144,
  "max_window_layers": 2,
  "model_type": "qwen3",
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "num_key_value_heads": 1,
  "num_target_layers": 4,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
  "sliding_window": 16,
  "tie_word_embeddings": false,
  "transformers_version": "5.12.1",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 32
}"#;

/// The tensors of the release's safetensors header (81 at 5 layers), named
/// as it names them, at the tiny geometry: hidden `h`, intermediate `i`,
/// `q` query heads and `kv` key/value heads of `d`, `taps` target layers,
/// conv groups of 16, selector rank `r` over `vocab`.
fn tiny_layout() -> Vec<(String, Vec<i64>)> {
    let (h, i, q, kv, d, taps, r, vocab) = (256, 256, 4, 1, 64, 2, 8, 32);
    let conv = 2 * 2 * (h / 16);
    let mut tensors = vec![
        (
            "candidate_selector.hidden_projection.weight".into(),
            vec![r, h],
        ),
        (
            "candidate_selector.predecessor_codebook".into(),
            vec![vocab, r],
        ),
        (
            "candidate_selector.successor_codebook".into(),
            vec![vocab, r],
        ),
        ("fc.weight".into(), vec![h, taps * h]),
        ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]),
    ];
    for layer in 0..2 {
        let at = |suffix: &str| format!("layers.{layer}.{suffix}");
        tensors.extend([
            (at("attention_conv.base_kernel"), vec![2, 2, h]),
            (at("attention_conv.kernel_projection.weight"), vec![conv, h]),
            (at("input_layernorm.weight"), vec![h]),
            (at("mlp.down_proj.weight"), vec![h, i]),
            (at("mlp.gate_proj.weight"), vec![i, h]),
            (at("mlp.up_proj.weight"), vec![i, h]),
            (at("mlp_conv.base_kernel"), vec![2, 2, h]),
            (at("mlp_conv.kernel_projection.weight"), vec![conv, h]),
            (at("post_attention_layernorm.weight"), vec![h]),
            (at("self_attn.k_norm.weight"), vec![d]),
            (at("self_attn.k_proj.weight"), vec![kv * d, h]),
            (at("self_attn.o_proj.weight"), vec![h, q * d]),
            (at("self_attn.q_norm.weight"), vec![d]),
            (at("self_attn.q_proj.weight"), vec![q * d, h]),
            (at("self_attn.v_proj.weight"), vec![kv * d, h]),
        ]);
    }
    tensors
}

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A checkpoint directory of `tensors`, each BF16 random unless `f32`
/// names it.
fn write_checkpoint(tensors: &[(String, Vec<i64>)], f32: Option<&str>) -> Dir {
    let dir = Dir(std::env::temp_dir().join(format!("mlx-fn-draft-{}", uuid::Uuid::new_v4())));
    std::fs::create_dir(&dir.0).unwrap();
    std::fs::write(dir.0.join("config.json"), TINY_CONFIG).unwrap();
    let mut arrays = HashMap::new();
    for (name, shape) in tensors {
        let dtype = if Some(name.as_str()) == f32 {
            DType::Float32
        } else {
            DType::BFloat16
        };
        let array = MxArray::random_normal(shape, 0.0, 0.02, Some(dtype)).unwrap();
        array.eval();
        arrays.insert(name.clone(), array);
    }
    crate::utils::safetensors::save_safetensors(dir.0.join("model.safetensors"), &mut arrays, None)
        .unwrap();
    dir
}

/// A directory of the release's layout passes the header check and the
/// loader; its config is the release's rule set at the tiny geometry. A
/// header that drops, retypes, reshapes or adds a tensor is refused by
/// name before any load.
#[test]
fn a_draft_of_the_flash_next_layout_is_accepted_and_loads() {
    let layout = tiny_layout();
    let dir = write_checkpoint(&layout, None);
    let config = read_dflash2_header(&dir.0).unwrap();
    assert_eq!(config.block_size, DFLASH2_DRAFT_PROPOSALS);
    assert_eq!(config.block_size, 7);
    assert_eq!(config.mask_token_id, 30);
    assert_eq!(config.target_layers, vec![1, 3]);
    assert_eq!(config.target_num_layers, 4);
    assert_eq!(
        (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size
        ),
        (256, 256, 32)
    );
    assert_eq!(
        (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim
        ),
        (2, 4, 1, 64)
    );
    assert_eq!(config.sliding_window, 16);
    assert_eq!((config.conv_group_size, config.conv_kernel_size), (16, 2));
    assert_eq!((config.selector_rank, config.selector_top_k), (8, 4));
    assert_eq!(config.rope_theta, 10_000.0);

    let target = DFlash2TargetShape {
        hidden_size: 256,
        vocab_size: 32,
        num_layers: 4,
    };
    if unsafe { sys::mlx_metal_is_available() } {
        let (model, bytes) = load_dflash2(&dir.0).unwrap();
        assert!(bytes > 0);
        assert_eq!(format!("{:?}", model.config), format!("{config:?}"));
        model.validate_target(target).unwrap();
        config.validate_target(target).unwrap();
        for wrong in [
            DFlash2TargetShape {
                num_layers: 3,
                ..target
            },
            DFlash2TargetShape {
                num_layers: 5,
                ..target
            },
            DFlash2TargetShape {
                hidden_size: 128,
                ..target
            },
            DFlash2TargetShape {
                vocab_size: 64,
                ..target
            },
        ] {
            assert!(
                model.validate_target(wrong).is_err(),
                "a {wrong:?} target is refused"
            );
        }
    }

    let refused = |tensors: &[(String, Vec<i64>)], f32: Option<&str>, says: &str| {
        let dir = write_checkpoint(tensors, f32);
        let error = read_dflash2_header(&dir.0).unwrap_err().to_string();
        assert!(error.contains(says), "{says}: {error}");
    };
    let name = "layers.1.mlp_conv.kernel_projection.weight";
    let without: Vec<_> = layout.iter().filter(|(n, _)| n != name).cloned().collect();
    refused(&without, None, &format!("missing {name}"));
    refused(&layout, Some(name), &format!("{name} is F32"));
    let reshaped: Vec<_> = layout
        .iter()
        .map(|(n, shape)| {
            let shape = if n == name {
                vec![shape[0] / 2, shape[1]]
            } else {
                shape.clone()
            };
            (n.clone(), shape)
        })
        .collect();
    refused(&reshaped, None, &format!("{name} has shape"));
    let mut extra = layout.clone();
    extra.push(("embed_tokens.weight".into(), vec![32, 256]));
    refused(&extra, None, "unexpected embed_tokens.weight");
}

/// The released file itself (CPU: the two headers only). Set
/// `MLX_TEST_FLASH_NEXT_DFLASH2_PATH` to its directory.
#[test]
#[ignore = "reads the external Flash Next DFlash2 checkpoint's headers"]
fn the_flash_next_draft_header_is_accepted() {
    let path = PathBuf::from(
        std::env::var("MLX_TEST_FLASH_NEXT_DFLASH2_PATH")
            .expect("set MLX_TEST_FLASH_NEXT_DFLASH2_PATH"),
    );
    let config = read_dflash2_header(&path).unwrap();
    assert_eq!(config.block_size, 7);
    assert_eq!(config.target_layers, vec![5, 15, 25, 35, 45]);
    assert_eq!(config.target_num_layers, 48);
    assert_eq!(config.mask_token_id, 248_077);
    assert_eq!(
        (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size
        ),
        (2560, 7680, 248_320)
    );
    assert_eq!(
        (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim
        ),
        (5, 32, 8, 128)
    );
    assert_eq!(config.sliding_window, 2048);
    assert_eq!((config.conv_group_size, config.conv_kernel_size), (16, 2));
    assert_eq!((config.selector_rank, config.selector_top_k), (256, 16));
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.kv_ring_bytes(), 42_106_880);

    let header =
        crate::utils::safetensors::SafeTensorsFile::load(path.join("model.safetensors")).unwrap();
    let params: usize = header
        .tensors
        .values()
        .map(|info| info.shape.iter().product::<usize>())
        .sum();
    assert_eq!((header.tensors.len(), params), (81, 603_065_600));
    assert_eq!(dflash2_file_bytes(&path).unwrap(), 1_206_140_008);
    let names: Vec<&String> = header.tensors.keys().collect();
    for missing in ["embed_tokens.weight", "lm_head.weight"] {
        assert!(!names.iter().any(|n| n.contains(missing)), "{missing}");
    }

    config.validate_target(FLASH_NEXT).unwrap();
    let qwen38_27b = DFlash2TargetShape {
        hidden_size: 5120,
        vocab_size: 248_320,
        num_layers: 64,
    };
    assert!(config.validate_target(qwen38_27b).is_err());
}
