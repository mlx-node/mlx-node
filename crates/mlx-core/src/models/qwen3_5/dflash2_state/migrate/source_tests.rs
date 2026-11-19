use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheKey, ColdCacheLayout, ColdGroup, ColdLayerBlock, ColdSourceReader,
    ColdStateLabel, ColdStateLayout, ColdStateObject, Landed, MakeRoom,
};

use super::*;
use crate::array::DType;
use crate::array::memory::WeightsResident;
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
use crate::disk_tier::{CacheDisk, DiskTierOptions};
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner;
use crate::models::qwen3_5::dflash2_state::disk::{
    DraftLoadGuard, DraftTier, lineage_keys, open_state_disk,
};
use crate::models::qwen3_5::dflash2_state::store::Lineage;
use crate::models::qwen3_5::gdn_sidecar;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5::model::dflash2_park_tests::{install_pool, passing_probe};
use crate::transformer::Activations;

const SEED: u64 = 0x5713_1616;

/// A fresh directory for one test, removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-source-identity-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(path.join("tmp")).expect("temp root");
        Self(path)
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The config bytes a load of `config` hashes into its tier's identity.
fn config_json(config: &Qwen3_5Config) -> Vec<u8> {
    let geometry = gdn_sidecar::geometry(config, DType::BFloat16).expect("a GDN geometry");
    crate::models::qwen3_5::model::cold_tier_config_json(
        config,
        &config.scaling(),
        &geometry.fingerprint_component(),
    )
    .expect("config bytes")
}

/// The draft load whose pool `inner` built: `inner`'s config with its
/// pool's page, hashed as a load with no explicit page hashes it.
fn draft_load_config(inner: &Qwen35Inner) -> Vec<u8> {
    let mut hashed = inner.config.clone();
    hashed.paged_block_size = None;
    config_json(&hashed)
}

fn int8_inner(block: Option<u32>, activations: Activations) -> Qwen35Inner {
    let mut inner = tiny_paged_int8_dflash_inner(SEED);
    inner.config.paged_block_size = block;
    install_pool(&mut inner, 8, 8, passing_probe()).expect("pool");
    inner
        .paged_adapter
        .as_mut()
        .expect("pool")
        .set_activations(activations);
    inner
}

/// One block of the pool's layout under `fingerprint`, keyed as the cold
/// walk keys block 0 of `tokens`.
fn first_block(
    inner: &Qwen35Inner,
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    tokens: &[u32],
) -> ColdCacheBlock {
    let pool = inner.paged_adapter.as_ref().expect("pool").layer_kv_pool();
    let (key_bytes, value_bytes) = pool.block_blob_bytes().expect("blob bytes");
    let layout = ColdCacheLayout {
        block_size: pool.block_size(),
        num_layers: pool.num_layers() as u32,
        num_kv_heads: pool.config().num_kv_heads,
        head_size: pool.config().head_size,
        cache_dtype: format!("{:?}", pool.cache_dtype()),
        key_bytes_per_layer: key_bytes as usize,
        value_bytes_per_layer: value_bytes as usize,
    };
    let block_size = pool.block_size() as usize;
    let key = lineage_keys(
        ColdGroup::Kv,
        fingerprint,
        Lineage::text(tokens, 0),
        pool.block_size(),
    )[0];
    ColdCacheBlock {
        key,
        fingerprint,
        tokens: tokens[..block_size].to_vec(),
        layers: (0..layout.num_layers)
            .map(|layer| ColdLayerBlock {
                keys: vec![layer as u8 ^ 0x5a; layout.key_bytes_per_layer],
                values: vec![layer as u8 ^ 0xa5; layout.value_bytes_per_layer],
            })
            .collect(),
        layout,
    }
}

fn state_object(
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    tokens: &[u32],
    block_size: u32,
) -> ColdStateObject {
    let key: ColdCacheKey = *lineage_keys(
        ColdGroup::DFlash2State,
        fingerprint,
        Lineage::text(tokens, 0),
        block_size,
    )
    .last()
    .expect("a whole block");
    ColdStateObject {
        key,
        fingerprint,
        layout: ColdStateLayout {
            boundary_tokens: tokens.len() as u32,
            spans: vec![96, 64],
            geometry: "test geometry".to_string(),
        },
        label: ColdStateLabel {
            tokens: tokens.to_vec(),
            extra_keys: Vec::new(),
            salt: 0,
            last_used: 1,
            checkpoint: false,
        },
        payload: vec![7u8; 160],
    }
}

struct Files {
    target: PathBuf,
    draft: PathBuf,
}

fn model_files(root: &Path) -> Files {
    let target = root.join("target");
    let draft = root.join("draft");
    std::fs::create_dir_all(&target).expect("target dir");
    std::fs::create_dir_all(&draft).expect("draft dir");
    std::fs::write(target.join("model.safetensors"), b"target").expect("target shard");
    std::fs::write(draft.join("model.safetensors"), b"draft").expect("draft shard");
    std::fs::write(draft.join("config.json"), b"{}").expect("draft config");
    Files { target, draft }
}

/// The source of a draft load whose pool has 32-token pages is the store
/// the same model wrote at block 16 with identity `Int8`: the directory on
/// disk, its state object under the state fingerprint and its KV block
/// under the KV fingerprint. The computed identity of the writer's own
/// pool equals the tier the product opened; the 32-token load's own
/// identity is another namespace.
/// Mutation config-some-32 (the 32-token load hashes its config with an
/// explicit page) → red.
#[test]
fn the_source_is_the_store_the_block_16_load_wrote() {
    if crate::test_support::paged_prefill_unsupported(
        "the_source_is_the_store_the_block_16_load_wrote",
    ) {
        return;
    }
    source_of_a_block_16_store("block16", Activations::Bf16, Activations::Auto);
}

/// The same with the activations a dense draft load takes when its options
/// name none, for the writer and for the 32-token load.
#[test]
fn the_source_is_the_store_a_default_dense_load_wrote() {
    if crate::test_support::paged_prefill_unsupported(
        "the_source_is_the_store_a_default_dense_load_wrote",
    ) {
        return;
    }
    let unset = crate::models::qwen3_5::load_plan::default_activations(false, true);
    assert_eq!(unset, Activations::Auto, "the dense draft load's default");
    source_of_a_block_16_store("default", unset, unset);
}

fn source_of_a_block_16_store(tag: &str, writer: Activations, reader: Activations) {
    let root = Root::new(tag);
    let files = model_files(&root.0);
    let options = DiskTierOptions {
        cache_disk: CacheDisk::Bytes(1 << 30),
        persistent: true,
        cache_dir: Some(root.0.join("cache")),
        temp_root: Some(root.0.join("tmp")),
    };
    let weights = WeightsResident::for_test(1, 4096);
    let target = CheckpointFiles::Dir(&files.target);
    let mut guard = CheckpointLoadGuard::before_read(target, true);
    guard.record_read();
    let draft_files = DraftLoadGuard::before_read(&files.draft, true);

    let writer = int8_inner(None, writer);
    let writer_adapter = writer.paged_adapter.as_ref().expect("pool");
    assert_eq!(writer_adapter.block_size(), SOURCE_BLOCK);
    assert_eq!(writer_adapter.cache_identity(), SOURCE_CACHE_IDENTITY);
    let writer_load = TierLoad {
        family: "qwen3_5",
        config_json: config_json(&writer.config),
        dense: &writer.config,
        activation: DType::BFloat16,
        draft: Some(DraftTier {
            model: writer.dflash2.as_ref().expect("a draft"),
            files: &draft_files,
        }),
        adapter: writer_adapter,
        files: target,
        weights: &weights,
        guard: &guard,
    };
    let (_, disk) = open_state_disk(&writer_load, &options, writer.model_id).expect("a tier");
    let tier = std::sync::Arc::clone(&disk.tier);
    assert!(tier.persistent);
    let own = tier_identity(&writer_load, SOURCE_BLOCK, SOURCE_CACHE_IDENTITY).expect("own");
    assert_eq!(
        (own.namespace.as_str(), own.state, own.kv),
        (
            tier.namespace.as_str(),
            tier.state_fingerprint,
            tier.kv_fingerprint
        ),
        "the computed identity of the writer's pool is the tier the product opened"
    );

    let tokens: Vec<u32> = (1..=32).collect();
    let block = first_block(&writer, tier.kv_fingerprint, &tokens);
    let (block_key, block_layout) = (block.key, block.layout.clone());
    assert!(tier.manager.enqueue(block).expect("a block job"));
    let state = state_object(tier.state_fingerprint, &tokens, SOURCE_BLOCK);
    let state_key = state.key;
    let (tx, rx) = mpsc::channel();
    tier.manager
        .enqueue_state(state, MakeRoom::Evict, tx)
        .expect("a state job");
    assert!(matches!(
        rx.recv_timeout(Duration::from_secs(10)).expect("landed"),
        Landed::State { ok: true, .. }
    ));
    assert!(tier.close(Instant::now() + Duration::from_secs(10)));
    let written = tier.namespace.clone();
    drop((disk, tier));

    let paged = int8_inner(Some(32), reader);
    let paged_adapter = paged.paged_adapter.as_ref().expect("pool");
    assert_eq!(paged_adapter.block_size(), 32);
    let paged_load = TierLoad {
        family: "qwen3_5",
        config_json: draft_load_config(&paged),
        dense: &paged.config,
        activation: DType::BFloat16,
        draft: Some(DraftTier {
            model: paged.dflash2.as_ref().expect("a draft"),
            files: &draft_files,
        }),
        adapter: paged_adapter,
        files: target,
        weights: &weights,
        guard: &guard,
    };
    let source = source_identity(&paged_load).expect("a source identity");
    assert_eq!(
        source.namespace, written,
        "the computed source is the namespace the block-16 load wrote"
    );
    let dir = root.0.join("cache").join(&source.namespace);
    assert!(dir.is_dir(), "the source is a directory on disk");
    let reader = ColdSourceReader::open(&dir).expect("a reader");
    assert!(
        reader.state_label(state_key, source.state).is_some(),
        "the state object is keyed by the source's state fingerprint"
    );
    assert!(
        reader
            .read_block(block_key, source.kv, &block_layout)
            .is_some(),
        "the block is keyed by the source's KV fingerprint"
    );
    let paged_own = tier_identity(&paged_load, 32, &paged_adapter.cache_identity())
        .expect("the 32-token load's own identity");
    assert_ne!(paged_own.namespace, written);
}
