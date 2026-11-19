use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{
    ColdCacheFingerprint, ColdCacheManager, ColdSourceReader, Landed, LayerKVPool, MakeRoom,
    PageLayout, PagedAttentionConfig, PhysicalBlock,
};

use super::*;
use crate::engine::build_paged_extra_keys;

const HEADS: usize = 4;
const DIM: usize = 256;
const LAYERS: usize = 16;
const INT8_CODE: u8 = 3;
const SALT: u64 = 0x5a17;

/// A fresh directory for one test, removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-migrate-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).expect("test root");
        Self(path)
    }

    fn dir(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fingerprints() -> Fingerprints {
    let of = |text: &[u8]| ColdCacheFingerprint::from_components([text]);
    Fingerprints {
        source_kv: of(b"source kv"),
        kv: of(b"page kv"),
        state: of(b"page state"),
    }
}

fn source_state_fingerprint() -> ColdCacheFingerprint {
    ColdCacheFingerprint::from_components([&b"source state"[..]])
}

fn text(tokens: &[u32]) -> Lineage<'_> {
    Lineage::text(tokens, SALT)
}

fn label(tokens: Vec<u32>, extra_keys: Vec<Vec<u64>>) -> ColdStateLabel {
    ColdStateLabel {
        tokens,
        extra_keys,
        salt: SALT,
        last_used: 41,
        checkpoint: false,
    }
}

fn canonical(mut keys: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    while keys.last().is_some_and(Vec::is_empty) {
        keys.pop();
    }
    keys
}

struct Owned(*mut mlx_sys::mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

fn int8_pool(block: u32, blocks: u32, layout: PageLayout) -> LayerKVPool {
    let config = PagedAttentionConfig {
        block_size: block,
        gpu_memory_mb: 256,
        head_size: DIM as u32,
        num_kv_heads: HEADS as u32,
        num_layers: LAYERS as u32,
        max_seq_len: Some(blocks * block),
        max_batch_size: Some(1),
    };
    LayerKVPool::new_with_layout(config, blocks, blocks, MetalDtype::Int8, layout)
        .expect("an int8 pool")
}

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// `[tokens, HEADS, DIM]` bf16 rows of `layer`, several magnitudes, one
/// row all zero.
fn rows(tokens: usize, layer: usize, seed: u64) -> Vec<u16> {
    let mut state = seed ^ (layer as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut out = vec![0u16; tokens * HEADS * DIM];
    for token in 0..tokens {
        for head in 0..HEADS {
            if (token, head) == (2, 1) {
                continue;
            }
            let magnitude = [0.01f32, 1.0, 3.7, 40.0, 250.0][(token * HEADS + head) % 5];
            for d in 0..DIM {
                out[(token * HEADS + head) * DIM + d] = bf16(next() * magnitude);
            }
        }
    }
    out
}

/// The product int8 store writes `tokens` rows into slots `0..tokens` of
/// every layer of `pool`.
fn store(pool: &LayerKVPool, tokens: usize) {
    let block = pool.config().block_size as i32;
    let slots: Vec<i64> = (0..tokens as i64).collect();
    let shape = [tokens as i64, HEADS as i64, DIM as i64];
    for layer in 0..LAYERS {
        let keys = rows(tokens, layer, 0x6b65_7973);
        let values = rows(tokens, layer, 0x7661_6c73);
        let owned = |raw: *mut mlx_sys::mlx_array| {
            assert!(!raw.is_null());
            Owned(raw)
        };
        let new_k = owned(unsafe {
            mlx_sys::mlx_array_from_bfloat16(keys.as_ptr(), shape.as_ptr(), shape.len())
        });
        let new_v = owned(unsafe {
            mlx_sys::mlx_array_from_bfloat16(values.as_ptr(), shape.as_ptr(), shape.len())
        });
        let slot_shape = [slots.len() as i64];
        let slot_map =
            owned(unsafe { mlx_sys::mlx_array_from_int64(slots.as_ptr(), slot_shape.as_ptr(), 1) });
        let layer = layer as u32;
        let key = owned(pool.key_cache_array_raw(layer).expect("K"));
        let value = owned(pool.value_cache_array_raw(layer).expect("V"));
        let key_scales = owned(pool.key_scale_array_raw(layer).expect("K scales"));
        let value_scales = owned(pool.value_scale_array_raw(layer).expect("V scales"));
        let mut out = [std::ptr::null_mut::<mlx_sys::mlx_array>(); 4];
        let [k_out, v_out, ks_out, vs_out] = &mut out;
        let ok = unsafe {
            mlx_sys::mlx_paged_kv_write_forward(
                key.0,
                value.0,
                new_k.0,
                new_v.0,
                slot_map.0,
                key_scales.0,
                value_scales.0,
                block,
                HEADS as i32,
                DIM as i32,
                INT8_CODE,
                false,
                k_out,
                v_out,
                ks_out,
                vs_out,
            )
        };
        assert!(ok, "the int8 store took the write");
        assert!(unsafe { mlx_sys::mlx_eval(out.as_mut_ptr(), 4) });
        drop(out.map(Owned));
    }
}

fn manager(dir: &Path) -> ColdCacheManager {
    ColdCacheManager::open_at(dir.to_path_buf(), 1 << 30, 0, 8).expect("a cache dir")
}

/// The product capture of `pool`'s block `block` into `manager`.
fn capture(
    manager: &ColdCacheManager,
    pool: &LayerKVPool,
    block: u32,
    key: ColdCacheKey,
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
) {
    let block = Arc::new(PhysicalBlock::new(block));
    assert_eq!(
        manager.capture_and_enqueue_before(
            pool,
            &block,
            key,
            fingerprint,
            tokens,
            std::time::Instant::now() + Duration::from_secs(10),
        ),
        Ok(true)
    );
    assert!(manager.drain(Duration::from_secs(30)));
}

fn write_state(manager: &ColdCacheManager, object: ColdStateObject) {
    let (done, landed) = mpsc::channel();
    assert!(manager.enqueue_state(object, MakeRoom::Evict, done).is_ok());
    match landed.recv_timeout(Duration::from_secs(30)) {
        Ok(Landed::State { ok, .. }) => assert!(ok, "the state landed"),
        other => panic!("no state landing: {other:?}"),
    }
}

fn source_layout() -> ColdCacheLayout {
    let side = HEADS * DIM * 16 + HEADS * 16 * 4;
    ColdCacheLayout {
        block_size: 16,
        num_layers: LAYERS as u32,
        num_kv_heads: HEADS as u32,
        head_size: DIM as u32,
        cache_dtype: "Int8".to_string(),
        key_bytes_per_layer: side,
        value_bytes_per_layer: side,
    }
}

/// One odd chat of 48 rows stored three ways: by the product at block 16
/// (the source store), by the product on the new 32-token page, and the
/// stored rows of both pools.
struct Fixture {
    root: Root,
    tokens: Vec<u32>,
    label: ColdStateLabel,
    plan: StatePlan,
    /// The new page's pool, rows 0..48 stored.
    page_pool: LayerKVPool,
    source: ColdCacheManager,
    product: ColdCacheManager,
}

const ROWS: usize = 48;

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = Root::new(tag);
        let fingerprints = fingerprints();
        let tokens: Vec<u32> = (0..ROWS as u32).map(|token| 1000 + token * 7).collect();
        let label = label(tokens.clone(), Vec::new());
        let plan = plan_state(&label, &fingerprints).expect("a plan");

        let source_pool = int8_pool(16, 4, PageLayout::default());
        store(&source_pool, ROWS);
        let source = manager(&root.dir("source"));
        let source_keys = lineage_keys(
            ColdGroup::Kv,
            fingerprints.source_kv,
            text(&tokens),
            SOURCE_BLOCK,
        );
        assert_eq!(source_keys.len(), 3);
        for (block, key) in source_keys.iter().enumerate() {
            capture(
                &source,
                &source_pool,
                block as u32,
                *key,
                fingerprints.source_kv,
                &tokens[block * 16..(block + 1) * 16],
            );
        }

        let page_pool = int8_pool(
            32,
            2,
            PageLayout {
                keys_token_major: true,
                ..PageLayout::default()
            },
        );
        store(&page_pool, ROWS);
        let product = manager(&root.dir("product"));
        capture(
            &product,
            &page_pool,
            0,
            plan.pages[0].key,
            fingerprints.kv,
            &tokens[..32],
        );
        Self {
            root,
            tokens,
            label,
            plan,
            page_pool,
            source,
            product,
        }
    }

    fn reader(&self) -> ColdSourceReader {
        ColdSourceReader::open(self.source.root()).expect("the source reads")
    }
}

/// A safetensors file: its header as JSON and the bytes after it.
fn split_file(path: &Path) -> (serde_json::Value, Vec<u8>) {
    let bytes = std::fs::read(path).expect("an object file");
    let (prefix, rest) = bytes.split_at(8);
    let len = u64::from_le_bytes(prefix.try_into().expect("8 bytes")) as usize;
    let (header, data) = rest.split_at(len);
    (
        serde_json::from_slice(header).expect("a JSON header"),
        data.to_vec(),
    )
}

fn object_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("list")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

/// Two block-16 objects the product stored convert into the file the
/// product writes for the same rows on the new page: same name and length,
/// the same header (the encoder writes its metadata map in hash order, so
/// the header is compared as JSON) and byte-equal tensor data; it decodes
/// to the same block.
/// Mutation halves-swapped (the page takes the second half first) → red.
#[test]
fn a_converted_page_is_the_page_the_new_store_writes() {
    let fixture = Fixture::new("page");
    let fingerprints = fingerprints();
    let reader = fixture.reader();
    let page = &fixture.plan.pages[0];
    let first = reader
        .read_block(page.sources[0], fingerprints.source_kv, &source_layout())
        .expect("first half");
    let second = reader
        .read_block(page.sources[1], fingerprints.source_kv, &source_layout())
        .expect("second half");
    let converted = convert_page(page, &fixture.tokens, &first, &second, &fingerprints)
        .expect("a converted page");
    let out = manager(&fixture.root.dir("converted"));
    assert_eq!(out.enqueue(converted), Ok(true));
    assert!(out.drain(Duration::from_secs(30)));

    let name = format!("{}.safetensors", page.key.to_hex());
    assert_eq!(object_names(out.root()), vec![name.clone()]);
    let product_file = fixture.product.root().join(&name);
    let converted_file = out.root().join(&name);
    assert_eq!(
        std::fs::metadata(&converted_file).expect("converted").len(),
        std::fs::metadata(&product_file).expect("product").len(),
        "file length"
    );
    let (product_header, product_data) = split_file(&product_file);
    let (converted_header, converted_data) = split_file(&converted_file);
    assert!(
        converted_data == product_data,
        "the converted page's bytes are the new store's"
    );
    assert_eq!(converted_header, product_header, "header");
    assert_eq!(
        out.load(page.key, fingerprints.kv),
        fixture.product.load(page.key, fingerprints.kv),
        "decoded block"
    );
}

/// An odd state (3 source blocks) converts to one page, one 16-row
/// fragment and its state object. The fragment holds the new store's first
/// 16 rows of a page, K token-major; its key and the state's are chained on
/// the last page key over the 16 tail tokens; the state keeps its layout,
/// label and payload.
/// Mutation fragment-skipped (the plan has no fragment) → red.
#[test]
fn an_odd_state_converts_to_pages_a_fragment_and_a_state() {
    let fixture = Fixture::new("odd");
    let fingerprints = fingerprints();
    let plan = &fixture.plan;
    let tokens = &fixture.tokens;
    let page_keys = lineage_keys(ColdGroup::Kv, fingerprints.kv, text(tokens), 32);
    assert_eq!(
        plan.pages.iter().map(|page| page.key).collect::<Vec<_>>(),
        page_keys
    );
    assert!(
        plan.fragment.is_some(),
        "an odd state plans its 16-row fragment"
    );
    let fragment = plan.fragment.as_ref().expect("a fragment");
    let source_keys = lineage_keys(ColdGroup::Kv, fingerprints.source_kv, text(tokens), 16);
    assert_eq!(fragment.source, source_keys[2]);
    assert_eq!(fragment.row, 32);
    assert_eq!(
        fragment.key,
        ColdCacheKey::chain(
            ColdGroup::Kv,
            fingerprints.kv,
            Some(page_keys[0]),
            &tokens[32..],
            &[],
            SALT,
            1,
        )
    );
    let state_pages = lineage_keys(
        ColdGroup::DFlash2State,
        fingerprints.state,
        text(tokens),
        32,
    );
    let state_key = ColdCacheKey::chain(
        ColdGroup::DFlash2State,
        fingerprints.state,
        Some(state_pages[0]),
        &tokens[32..],
        &[],
        SALT,
        1,
    );
    assert_eq!(plan.state_key, state_key);

    let layout = ColdStateLayout {
        boundary_tokens: ROWS as u32,
        spans: vec![96, 64],
        geometry: "gdn test\n".to_string(),
    };
    let payload: Vec<u8> = (0..160).map(|byte| (byte * 3) as u8).collect();
    let source_state_key = lineage_keys(
        ColdGroup::DFlash2State,
        source_state_fingerprint(),
        text(tokens),
        16,
    )
    .last()
    .copied()
    .expect("a source state key");
    write_state(
        &fixture.source,
        ColdStateObject {
            key: source_state_key,
            fingerprint: source_state_fingerprint(),
            layout: layout.clone(),
            label: fixture.label.clone(),
            payload: payload.clone(),
        },
    );

    let reader = fixture.reader();
    let out = manager(&fixture.root.dir("converted"));
    for page in &plan.pages {
        let halves = page
            .sources
            .map(|key| reader.read_block(key, fingerprints.source_kv, &source_layout()));
        let [Some(first), Some(second)] = halves else {
            panic!("both halves read");
        };
        let block = convert_page(page, tokens, &first, &second, &fingerprints).expect("page");
        assert_eq!(out.enqueue(block), Ok(true));
    }
    let tail = reader
        .read_block(fragment.source, fingerprints.source_kv, &source_layout())
        .expect("tail block");
    let converted = convert_fragment(fragment, tokens, &tail, &fingerprints).expect("fragment");
    assert_eq!(out.enqueue(converted), Ok(true));
    assert!(out.drain(Duration::from_secs(30)));

    let stored = out
        .load(fragment.key, fingerprints.kv)
        .expect("the fragment reads back");
    assert_eq!(stored.layout, source_layout());
    assert_eq!(stored.tokens, tokens[32..]);
    let product = fixture
        .page_pool
        .read_block_all_layers(1)
        .expect("the new store's second page");
    let plane = HEADS * 32 * DIM;
    for (layer, (keys, values)) in product.iter().enumerate() {
        let mut want_keys = Vec::new();
        let mut want_key_scales = Vec::new();
        let mut want_values = Vec::new();
        let mut want_value_scales = Vec::new();
        for head in 0..HEADS {
            want_keys.extend_from_slice(&keys[head * 32 * DIM..][..16 * DIM]);
            want_key_scales.extend_from_slice(&keys[plane + head * 32 * 4..][..16 * 4]);
            for d in 0..DIM {
                want_values.extend_from_slice(&values[(head * DIM + d) * 32..][..16]);
            }
            want_value_scales.extend_from_slice(&values[plane + head * 32 * 4..][..16 * 4]);
        }
        want_keys.extend(want_key_scales);
        want_values.extend(want_value_scales);
        assert!(
            stored.layers[layer].keys == want_keys && stored.layers[layer].values == want_values,
            "layer {layer}: the fragment is the new store's first 16 rows"
        );
    }

    let mut read = Vec::new();
    let (source_layout_read, source_label) = reader
        .read_state(source_state_key, source_state_fingerprint(), &mut read)
        .expect("the source state reads");
    let object = convert_state(plan, source_layout_read, source_label, read, &fingerprints)
        .expect("a converted state");
    write_state(&out, object);
    let (stored_layout, stored_label) = out
        .state_label(state_key, fingerprints.state)
        .expect("the state reads back");
    assert_eq!(stored_layout, layout);
    assert_eq!(stored_label, fixture.label);
    let mut stored_payload = vec![0; payload.len()];
    assert!(
        out.load_state(
            state_key,
            fingerprints.state,
            &layout,
            &[],
            &mut stored_payload
        )
        .is_some()
    );
    assert_eq!(stored_payload, payload);
    let mut names = vec![
        format!("{}.safetensors", page_keys[0].to_hex()),
        format!("{}.safetensors", fragment.key.to_hex()),
        format!("{}.dflash2_state.safetensors", state_key.to_hex()),
    ];
    names.sort();
    assert_eq!(object_names(out.root()), names);
}

/// The image keys of a converted state are the keys a lineage takes
/// at 32-token pages (`build_paged_extra_keys(total, 32, positions)`):
/// runs across a half boundary, a second-half image and one in the tail
/// half; page and fragment keys hash them.
/// Mutation second-half-not-rebased → red.
#[test]
fn image_keys_are_the_keys_of_32_token_pages() {
    let rows = 80;
    let mut positions: Vec<(u32, u64)> = Vec::new();
    for (token, image) in [
        (14u32, 0xa0u64),
        (15, 0xa0),
        (16, 0xa0),
        (19, 0xa0),
        (40, 0xb0),
        (52, 0xc0),
        (53, 0xc0),
        (70, 0xd0),
    ] {
        for word in 0..4 {
            positions.push((token, image + word));
        }
    }
    let source = canonical(build_paged_extra_keys(rows, 16, &positions));
    let want = canonical(build_paged_extra_keys(rows, 32, &positions));
    assert_eq!(source.len(), 5);
    assert_eq!(want.len(), 3);
    let tokens: Vec<u32> = (0..rows as u32).collect();
    let fingerprints = fingerprints();
    let plan = plan_state(&label(tokens.clone(), source), &fingerprints).expect("a plan");
    assert_eq!(plan.extra_keys, want, "page image keys");
    let lineage = Lineage {
        tokens: &tokens,
        extra_keys: &want,
        salt: SALT,
    };
    let pages = lineage_keys(ColdGroup::Kv, fingerprints.kv, lineage, 32);
    assert_eq!(
        plan.pages.iter().map(|page| page.key).collect::<Vec<_>>(),
        pages
    );
    assert_eq!(
        plan.fragment.as_ref().map(|fragment| fragment.key),
        Some(ColdCacheKey::chain(
            ColdGroup::Kv,
            fingerprints.kv,
            pages.last().copied(),
            &tokens[64..],
            &want[2],
            SALT,
            2,
        ))
    );
    let plain = plan_state(&label(tokens.clone(), Vec::new()), &fingerprints).expect("a plan");
    assert!(plain.extra_keys.is_empty());
    assert_ne!(plain.pages[1].key, plan.pages[1].key);
}

/// A newer chat B that extends an older chat A past A's odd last
/// block holds that block as the first half of one of its pages; the tail
/// set still holds it (and the tail of a one-block chat), never an even
/// chat's last block.
/// Mutation tail-skips-covered (a chat another chat extends has no tail) →
/// red.
#[test]
fn the_tail_set_holds_a_tail_a_newer_chat_extends() {
    let fingerprints = fingerprints();
    let tokens: Vec<u32> = (0..96).map(|token| 5000 + token).collect();
    let older = label(tokens[..48].to_vec(), Vec::new());
    let newer = label(tokens.clone(), Vec::new());
    let even = label((0..64).map(|token| 9000 + token).collect(), Vec::new());
    let single = label((0..16).map(|token| 7000 + token).collect(), Vec::new());
    let older_keys = lineage_keys(
        ColdGroup::Kv,
        fingerprints.source_kv,
        text(&older.tokens),
        16,
    );
    let single_keys = lineage_keys(
        ColdGroup::Kv,
        fingerprints.source_kv,
        text(&single.tokens),
        16,
    );
    let newer_plan = plan_state(&newer, &fingerprints).expect("a plan");
    assert_eq!(newer_plan.pages[1].sources[0], older_keys[2]);
    assert!(newer_plan.fragment.is_none());

    let tails = tail_keys([&newer, &older, &even, &single], fingerprints.source_kv);
    assert!(
        tails.contains(&older_keys[2]),
        "the tail set holds the older chat's tail, which the newer chat extends"
    );
    assert_eq!(tails, HashSet::from([older_keys[2], single_keys[0]]));
}

/// What the converter refuses: rows that are not whole source blocks,
/// image keys out of form, a bf16 block, a half of another lineage.
#[test]
fn the_converter_refuses_what_it_cannot_re_pack() {
    let fingerprints = fingerprints();
    let tokens: Vec<u32> = (0..64).collect();
    assert!(plan_state(&label(tokens[..40].to_vec(), Vec::new()), &fingerprints).is_none());
    assert!(plan_state(&label(tokens.clone(), vec![vec![1, 2, 3]]), &fingerprints).is_none());
    assert!(
        plan_state(
            &label(tokens.clone(), vec![vec![1, 2, 7, 16]]),
            &fingerprints
        )
        .is_none()
    );
    assert!(
        plan_state(
            &label(tokens.clone(), vec![vec![1, 2, 7, 3], vec![9, 2, 7, 3]]),
            &fingerprints
        )
        .is_none()
    );

    let plan = plan_state(&label(tokens.clone(), Vec::new()), &fingerprints).expect("a plan");
    let page = &plan.pages[0];
    let layout = source_layout();
    let half = |key: ColdCacheKey, rows: &[u32], layout: &ColdCacheLayout| ColdCacheBlock {
        key,
        fingerprint: fingerprints.source_kv,
        tokens: rows.to_vec(),
        layout: layout.clone(),
        layers: (0..layout.num_layers)
            .map(|_| ColdLayerBlock {
                keys: vec![1; layout.key_bytes_per_layer],
                values: vec![2; layout.value_bytes_per_layer],
            })
            .collect(),
    };
    let first = half(page.sources[0], &tokens[..16], &layout);
    let second = half(page.sources[1], &tokens[16..32], &layout);
    assert!(convert_page(page, &tokens, &first, &second, &fingerprints).is_some());
    assert!(convert_page(page, &tokens, &second, &first, &fingerprints).is_none());
    let other_rows = half(page.sources[1], &tokens[32..48], &layout);
    assert!(convert_page(page, &tokens, &first, &other_rows, &fingerprints).is_none());
    let bf16 = ColdCacheLayout {
        cache_dtype: "BFloat16".to_string(),
        ..layout.clone()
    };
    let first_bf16 = half(page.sources[0], &tokens[..16], &bf16);
    let second_bf16 = half(page.sources[1], &tokens[16..32], &bf16);
    assert!(convert_page(page, &tokens, &first_bf16, &second_bf16, &fingerprints).is_none());
}
