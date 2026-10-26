// Included in gguf::tests so fixtures exercise the real converter/cache path.
use crate::utils::gguf_test_fixtures::qwen35_moe_standalone_metadata;

fn source_operand_fixture(root: &Path) -> PathBuf {
    source_operand_fixture_with_inline_mtp(root, false)
}
fn source_operand_fixture_with_inline_mtp(root: &Path, inline_mtp: bool) -> PathBuf {
    let mut metadata = qwen35_moe_standalone_metadata();
    for (key, value) in [
        ("qwen35moe.embedding_length", 2048),
        ("qwen35moe.block_count", if inline_mtp { 41 } else { 40 }),
        ("qwen35moe.attention.head_count", 16),
        ("qwen35moe.attention.head_count_kv", 2),
        ("qwen35moe.attention.key_length", 256),
        ("qwen35moe.full_attention_interval", 4),
        ("qwen35moe.expert_count", 256),
        ("qwen35moe.expert_used_count", 8),
        ("qwen35moe.expert_feed_forward_length", 512),
        ("qwen35moe.expert_shared_feed_forward_length", 512),
    ] {
        metadata
            .iter_mut()
            .find(|(name, _)| *name == key)
            .unwrap()
            .1 = GgufMetaValue::Uint32(value);
    }
    let mut owned = source_math::expected_tensors()
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let data = (0..name.width)
                .flat_map(|n| (1.0001 + ((index * 7 + n) % 97) as f32 * 0.00011).to_le_bytes())
                .collect::<Vec<_>>();
            (
                name.original,
                vec![name.width as u64],
                GgufTensorType::F32,
                data,
            )
        })
        .collect::<Vec<_>>();
    if inline_mtp {
        // The full structural nextn witness uses the real H2048 dimensions.
        // Its ~16MiB projection is bounded and excluded from the 101 operands.
        owned.push((
            "blk.40.nextn.eh_proj.weight".into(),
            vec![4096, 2048],
            GgufTensorType::BF16,
            vec![0; 4096 * 2048 * 2],
        ));
        for suffix in [
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
            "attn_norm.weight",
            "post_attention_norm.weight",
        ] {
            owned.push((
                format!("blk.40.{suffix}"),
                vec![2048],
                GgufTensorType::F32,
                (0..2048).flat_map(|_| 1.00013f32.to_le_bytes()).collect(),
            ));
        }
    }
    let tensors = owned
        .iter()
        .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
        .collect::<Vec<_>>();
    let path = root.join("source-norms.gguf");
    fs::write(&path, build_minimal_gguf(&metadata, &tensors)).unwrap();
    path
}
fn source_operand_options(input: &Path, output: &Path) -> GgufConversionOptions {
    GgufConversionOptions {
        input_path: input.to_string_lossy().into_owned(),
        output_dir: output.to_string_lossy().into_owned(),
        config_source_dir: None,
        dtype: Some("bfloat16".into()),
        verbose: Some(false),
        quantize: Some(false),
        quant_bits: None,
        quant_group_size: None,
        quant_mode: None,
        quant_recipe: None,
        imatrix_path: None,
        output_filename: None,
        vlm_key_prefix: None,
        quant_mxfp: None,
        import_k_quants: Some(true),
        native_qwen35_layout: Some(true),
    }
}
fn source_operand_identity(input: &Path) -> source_math::ImportIdentity {
    let input = input.canonicalize().unwrap();
    let metadata = fs::metadata(&input).unwrap();
    source_math::ImportIdentity {
        source_stat_path_sha256: source_file_identity_digest(&input, &metadata),
        source_bytes: metadata.len(),
        assets_sha256: qwen35_native_asset_digest(input.parent().unwrap()).unwrap(),
    }
}
#[tokio::test]
#[ignore = "bounded native CPU converter; run in a fresh process"]
async fn original_f32_preservation_keeps_main_safetensors_bytes_exact() {
    let temp = GemmaNativeTestDir::new();
    let input = source_operand_fixture(temp.path());
    let native = temp.path().join("native");
    let preserved = temp.path().join("preserved");
    convert_gguf_to_safetensors(source_operand_options(&input, &native))
        .await
        .unwrap();
    let identity = source_operand_identity(&input);
    convert_gguf_to_safetensors_inner(
        source_operand_options(&input, &preserved),
        Some(identity.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        fs::read(native.join("model.safetensors")).unwrap(),
        fs::read(preserved.join("model.safetensors")).unwrap()
    );
    assert_eq!(
        fs::read(native.join("config.json")).unwrap(),
        fs::read(preserved.join("config.json")).unwrap()
    );
    assert!(!native.join(".mlx-source-math").exists());
    source_math::validate(&preserved, &identity, None).unwrap();
    let header = parse_gguf(&input).unwrap();
    let source_bytes = fs::read(&input).unwrap();
    let payload = crate::utils::safetensors::load_safetensors_lazy(
        preserved.join(".mlx-source-math/weights.safetensors"),
    )
    .unwrap();
    assert_eq!(payload.len(), 101);
    for name in source_math::expected_tensors() {
        let descriptor = header
            .tensors
            .iter()
            .find(|t| t.name == name.original)
            .unwrap();
        let start = (header.data_offset + descriptor.offset) as usize;
        let actual = payload[&name.mapped]
            .to_float32()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            &source_bytes[start..start + name.width * 4],
            "{}: original F32 must be preserved without +1",
            name.mapped
        );
    }
    assert_eq!(
        fs::read_dir(&preserved)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .count(),
        1
    );
}
#[test]
fn source_operand_layout_change_is_moe_only() {
    assert_eq!(NativeGgufFamily::Qwen35.layout(), "tiled");
    assert_eq!(NativeGgufFamily::Qwen35Moe.layout(), source_math::LAYOUT);
    assert_eq!(NativeGgufFamily::Clef.layout(), "clef-v1");
    assert_eq!(NativeGgufFamily::Gemma4.layout(), "gemma4-text-dtype-v2");
    assert_eq!(
        NativeGgufFamily::MuseGlimmer.layout(),
        "muse-glimmer-packed-v1"
    );
    let old = source_math::sha256(b"assets:tiled-moe-v1:none");
    let new = source_math::sha256(format!("assets:{}:none", source_math::LAYOUT).as_bytes());
    assert_ne!(old, new);
}

#[tokio::test]
#[ignore = "bounded 40+1 inline-MTP CPU converter; run in a fresh process"]
async fn source_operand_import_excludes_inline_mtp_from_main_geometry() {
    let temp = GemmaNativeTestDir::new();
    let input = source_operand_fixture_with_inline_mtp(temp.path(), true);
    let output = temp.path().join("preserved");
    let parsed = parse_gguf(&input).unwrap();
    assert_eq!(
        extract_config(&parsed.metadata)["num_hidden_layers"],
        serde_json::json!(41)
    );
    assert_eq!(qwen35_inline_mtp_index(&parsed).unwrap(), Some(40));
    let identity = source_operand_identity(&input);
    convert_gguf_to_safetensors_inner(
        source_operand_options(&input, &output),
        Some(identity.clone()),
    )
    .await
    .unwrap();
    source_math::validate(&output, &identity, None).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join(".mlx-source-math/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["geometry"]["layers"], serde_json::json!(40));
    assert_eq!(
        manifest["capability"]["status"],
        serde_json::json!("available")
    );
    let tensors = manifest["capability"]["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 101);
    assert!(tensors.iter().all(|tensor| {
        !tensor["original_name"]
            .as_str()
            .unwrap()
            .starts_with("blk.40.")
    }));
    assert!(
        tensors
            .iter()
            .any(|tensor| tensor["original_name"] == "blk.0.post_attention_norm.weight")
    );
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["num_hidden_layers"], serde_json::json!(40));
    assert_eq!(config["mtp_num_hidden_layers"], serde_json::json!(1));
}
