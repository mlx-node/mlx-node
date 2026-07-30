//! CPU-only persistence coverage for Gemma4's image-aware cold-cache identity.
//!
//! The real-weight process-restart gate lives in
//! `gemma4_image_cold_tier_process_restart.rs`. This test pins the lower-level
//! persisted-object contract without Metal or model weights: the exact
//! per-block image `extra_keys` used by the paged adapter feed both the KV and
//! sliding-window key chains, and both object types survive a manager reopen.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mlx_core::transformer::paged_kv_cache_adapter::compute_per_block_image_extra_keys;
use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdLayerBlock, ColdSidecar, ColdSidecarLayout,
};

const BLOCK_SIZE: usize = 4;
const NUM_BLOCKS: usize = 3;
const CACHE_SALT: u64 = 0xA4B4_1A6E;
const GIB: u64 = 1024 * 1024 * 1024;

static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

struct TestRoot(PathBuf);

impl TestRoot {
    fn new() -> Self {
        let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "mlx-gemma4-image-cold-reopen-{}-{sequence}",
            std::process::id()
        )))
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
struct ImageAwareChains {
    kv: Vec<ColdCacheKey>,
    sliding: Vec<ColdCacheKey>,
}

fn fingerprint() -> ColdCacheFingerprint {
    ColdCacheFingerprint::from_components([
        b"gemma4-image-restart-fixture".as_slice(),
        b"paged-block-size=4".as_slice(),
    ])
}

fn tokens() -> Vec<u32> {
    (101..101 + (BLOCK_SIZE * NUM_BLOCKS) as u32).collect()
}

fn image_positions(first: u64, second: u64) -> Vec<(u32, u64)> {
    // Two logical images occupy equal-size spans in separate blocks. Swapping
    // `first` and `second` therefore models reordered image inputs without
    // changing token ids, positions, expansion lengths, or cache geometry.
    [(1u32, first), (2, first), (5, second), (6, second)].to_vec()
}

fn chains(
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
    image_positions: &[(u32, u64)],
) -> ImageAwareChains {
    let per_block =
        compute_per_block_image_extra_keys(image_positions, NUM_BLOCKS, BLOCK_SIZE as u32);
    assert_eq!(per_block.len(), NUM_BLOCKS);

    let derive = |group| {
        let mut parent = None;
        (0..NUM_BLOCKS)
            .map(|index| {
                let start = index * BLOCK_SIZE;
                let key = ColdCacheKey::chain(
                    group,
                    fingerprint,
                    parent,
                    &tokens[start..start + BLOCK_SIZE],
                    &per_block[index],
                    CACHE_SALT,
                    index,
                );
                parent = Some(key);
                key
            })
            .collect()
    };

    ImageAwareChains {
        kv: derive(ColdGroup::Kv),
        sliding: derive(ColdGroup::SlidingWindow),
    }
}

fn block(key: ColdCacheKey, fingerprint: ColdCacheFingerprint, tokens: &[u32]) -> ColdCacheBlock {
    ColdCacheBlock {
        key,
        fingerprint,
        tokens: tokens.to_vec(),
        layout: ColdCacheLayout {
            block_size: BLOCK_SIZE as u32,
            num_layers: 2,
            num_kv_heads: 1,
            head_size: 2,
            cache_dtype: "BFloat16".to_string(),
            key_bytes_per_layer: BLOCK_SIZE,
            value_bytes_per_layer: BLOCK_SIZE,
        },
        layers: vec![
            ColdLayerBlock {
                keys: vec![1, 2, 3, 4],
                values: vec![5, 6, 7, 8],
            },
            ColdLayerBlock {
                keys: vec![9, 10, 11, 12],
                values: vec![13, 14, 15, 16],
            },
        ],
    }
}

fn sliding_layout() -> ColdSidecarLayout {
    ColdSidecarLayout {
        group: ColdGroup::SlidingWindow,
        boundary_tokens: (BLOCK_SIZE * NUM_BLOCKS) as u32,
        num_layers: 2,
        tensors_per_layer: 1,
        dtype: "BFloat16".to_string(),
        dims: vec![(BLOCK_SIZE * NUM_BLOCKS) as u32, 1],
        bytes_per_tensor: BLOCK_SIZE * NUM_BLOCKS * 2,
    }
}

fn sliding_sidecar(
    key: ColdCacheKey,
    fingerprint: ColdCacheFingerprint,
    layout: &ColdSidecarLayout,
) -> ColdSidecar {
    ColdSidecar {
        key,
        fingerprint,
        layout: layout.clone(),
        tensors: vec![
            (0..layout.bytes_per_tensor as u8).collect(),
            (32..32 + layout.bytes_per_tensor as u8).collect(),
        ],
    }
}

#[test]
fn image_aware_kv_and_sliding_sidecar_survive_reopen_and_fail_closed() {
    const IMAGE_A: u64 = 0xAAAA_AAAA_AAAA_AAAA;
    const IMAGE_B: u64 = 0xBBBB_BBBB_BBBB_BBBB;
    const IMAGE_C: u64 = 0xCCCC_CCCC_CCCC_CCCC;

    let root = TestRoot::new();
    let fingerprint = fingerprint();
    let tokens = tokens();
    let captured = chains(fingerprint, &tokens, &image_positions(IMAGE_A, IMAGE_B));
    let same = chains(fingerprint, &tokens, &image_positions(IMAGE_A, IMAGE_B));
    let changed = chains(fingerprint, &tokens, &image_positions(IMAGE_A, IMAGE_C));
    let reordered = chains(fingerprint, &tokens, &image_positions(IMAGE_B, IMAGE_A));

    assert_eq!(same.kv, captured.kv);
    assert_eq!(same.sliding, captured.sliding);
    // The unchanged first image still matches block zero. The changed second
    // image breaks its block and every descendant in both parent chains.
    assert_eq!(changed.kv[0], captured.kv[0]);
    assert_ne!(changed.kv[1], captured.kv[1]);
    assert_ne!(changed.kv[2], captured.kv[2]);
    assert_eq!(changed.sliding[0], captured.sliding[0]);
    assert_ne!(changed.sliding[1], captured.sliding[1]);
    assert_ne!(changed.sliding[2], captured.sliding[2]);
    // Reordering changes the first image-bearing block, so no later boundary
    // in either chain may alias the captured order.
    assert!(
        reordered
            .kv
            .iter()
            .zip(&captured.kv)
            .all(|(left, right)| left != right)
    );
    assert!(
        reordered
            .sliding
            .iter()
            .zip(&captured.sliding)
            .all(|(left, right)| left != right)
    );

    let layout = sliding_layout();
    let sidecar_key = *captured.sliding.last().expect("three sliding keys");
    let expected_sidecar = sliding_sidecar(sidecar_key, fingerprint, &layout);

    {
        let manager = ColdCacheManager::open_at(root.0.clone(), GIB, 0, NUM_BLOCKS + 2).unwrap();
        for (index, key) in captured.kv.iter().copied().enumerate() {
            let start = index * BLOCK_SIZE;
            assert!(
                manager
                    .enqueue(block(key, fingerprint, &tokens[start..start + BLOCK_SIZE]))
                    .unwrap()
            );
        }
        assert!(manager.enqueue_sidecar(expected_sidecar.clone()).unwrap());
        assert!(manager.drain(Duration::from_secs(5)));
    }

    // A new manager must rebuild its index from disk. Recomputing the same
    // image-aware identities then reaches all KV blocks and the exact sliding
    // boundary, while changed/reordered identities remain absent.
    let reopened = ColdCacheManager::open_at(root.0.clone(), GIB, 0, NUM_BLOCKS + 2).unwrap();
    for (index, key) in same.kv.iter().copied().enumerate() {
        let start = index * BLOCK_SIZE;
        assert_eq!(
            reopened.load(key, fingerprint),
            Some(block(key, fingerprint, &tokens[start..start + BLOCK_SIZE]))
        );
    }
    assert_eq!(
        reopened.load_sidecar(sidecar_key, fingerprint, &layout),
        Some(expected_sidecar)
    );

    assert!(!reopened.contains(&changed.kv[1]));
    assert!(!reopened.contains(&changed.kv[2]));
    assert!(!reopened.contains_in(
        changed.sliding.last().expect("changed sliding key"),
        ColdGroup::SlidingWindow
    ));
    assert!(reordered.kv.iter().all(|key| !reopened.contains(key)));
    assert!(
        reordered
            .sliding
            .iter()
            .all(|key| !reopened.contains_in(key, ColdGroup::SlidingWindow))
    );
    assert_eq!(
        reopened.load(changed.kv[1], fingerprint),
        None,
        "changed image identity must miss from its first changed block"
    );
    assert_eq!(
        reopened.load(*reordered.kv.last().unwrap(), fingerprint),
        None,
        "reordered images must miss the captured KV chain"
    );
    assert_eq!(
        reopened.load_sidecar(*changed.sliding.last().unwrap(), fingerprint, &layout),
        None,
        "changed image identity must miss the sliding sidecar"
    );
    assert_eq!(
        reopened.load_sidecar(*reordered.sliding.last().unwrap(), fingerprint, &layout),
        None,
        "reordered images must miss the sliding sidecar"
    );

    // Corruption is a safe miss: the decoder must not return partial rotating
    // state, must count the corruption, and must prune the bad object.
    let sidecar_path = root.0.join(format!(
        "{}.{}.safetensors",
        sidecar_key.to_hex(),
        ColdGroup::SlidingWindow.label()
    ));
    let bytes = fs::read(&sidecar_path).expect("persisted sliding sidecar");
    fs::write(&sidecar_path, &bytes[..bytes.len() / 2]).expect("truncate sidecar");
    let stats_before = reopened.stats();
    assert_eq!(
        reopened.load_sidecar(sidecar_key, fingerprint, &layout),
        None
    );
    let stats_after = reopened.stats();
    assert_eq!(stats_after.corruptions, stats_before.corruptions + 1);
    assert_eq!(stats_after.misses, stats_before.misses + 1);
    assert!(!sidecar_path.exists(), "corrupt sidecar must be pruned");
    assert!(!reopened.contains_in(&sidecar_key, ColdGroup::SlidingWindow));
}
