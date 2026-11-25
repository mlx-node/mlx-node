//! The stored states' disk tier: identity text, chain/segment keys, block
//! codec, turn keys, lookup bounds, pool blobs, and disk round-trips.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheManager, ColdGroup,
    ColdLayerBlock, ColdObject, hash_tokens,
};

use super::cache::WINDOW_SLACK;
use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{IndexerGeometry, PagedIndexerState, TokTail};
use super::ladder::Kind;
use super::owner::{CellLayer, OwnerCell};
use super::paged::PageGeometry;
use super::persist::{
    Blocks, Disk, FLAT_RUN, Identity, MINIMUM_ROWS, QUEUE_BYTES, chain, disk_plan, fingerprint,
    identity_text, segment_key, segments_at,
};
use super::persist_codec::{
    StateShape, block_layout, decode_state, encode_state_into, join_block, split_block,
};
use super::persist_writer::Telemetry;
use super::prefill::BLOCK_ROWS;
use super::prefix_state::{
    Key, PrefixState, StateLayer, States, WRITE_TAG, any_written, is_written, prompt_keys, resumes,
    turn_keys, written_key,
};
use super::sched::lookup_bounds;
use super::test_support::{Temp, arr, tiny_cfg};
use crate::array::{DType, MxArray};
use crate::disk_tier::{CacheDisk, DiskTierOptions, PersistentIdentity, open_disk_tier};
use crate::transformer::paged_kv_cache_adapter::SeqId;

const PAGE: PageGeometry = PageGeometry {
    layers: 12,
    half: 288,
    index_rows: 8,
    index_dim: 128,
};

fn tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| 11 + (i % 251) as u32).collect()
}

fn bare(row: usize) -> PrefixState {
    PrefixState {
        row,
        layers: Vec::new(),
        draft: None,
    }
}

struct TierRig {
    disk: Disk,
    manager: Arc<ColdCacheManager>,
    fp: ColdCacheFingerprint,
    telem: Arc<Telemetry>,
}

impl TierRig {
    fn open(root: &Temp, persist: Option<(&str, &[u8])>, shape: &StateShape) -> Self {
        let tier = match persist {
            None => {
                let opts = DiskTierOptions {
                    temp_root: Some(root.path().to_path_buf()),
                    ..DiskTierOptions::from_load(Some(&(1u64 << 30).to_string()), false, None)
                        .unwrap()
                };
                open_disk_tier(&opts, || None, shape.span_bytes()).unwrap()
            }
            Some((ns, seed)) => {
                let tmp = root.join("tmp");
                std::fs::create_dir_all(&tmp).unwrap();
                let opts = DiskTierOptions {
                    cache_disk: CacheDisk::Bytes(2 << 30),
                    persistent: true,
                    cache_dir: Some(root.join("cache")),
                    temp_root: Some(tmp),
                };
                let fp = ColdCacheFingerprint::from_components([seed]);
                let id = PersistentIdentity {
                    namespace: ns.into(),
                    state: fp,
                    kv: fp,
                };
                let t = open_disk_tier(&opts, || Some(id), shape.span_bytes()).unwrap();
                assert!(t.persistent);
                t
            }
        };
        let (manager, fp) = (Arc::clone(&tier.manager), tier.state_fingerprint);
        let telem = Arc::new(Telemetry::default());
        let disk = Disk::open(tier, Arc::clone(&telem), shape.clone(), PAGE).unwrap();
        Self {
            disk,
            manager,
            fp,
            telem,
        }
    }

    fn offer_block(&self, key: ColdCacheKey, tok: &[u32], layers: Vec<ColdLayerBlock>) {
        let blk = ColdCacheBlock {
            key,
            fingerprint: self.fp,
            tokens: tok.to_vec(),
            layout: block_layout(&PAGE),
            layers,
        };
        let ddl = Instant::now() + Duration::from_secs(5);
        assert!(
            self.manager
                .offer(ColdObject::Block(blk), ddl)
                .unwrap()
                .is_none()
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn publish(
        &mut self,
        states: &mut States<PrefixState>,
        seq: SeqId,
        tok: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        state: PrefixState,
        kind: Kind,
    ) -> u64 {
        let row = tok.len();
        let hot = vec![0u64; row / BLOCK_ROWS];
        self.disk.start(seq, tok, keys, &hot, salt);
        let k = Key {
            tokens: tok,
            keys,
            salt,
        };
        let id = states.publish(k, state, kind).unwrap().unwrap();
        self.disk
            .published(states.store(), seq, row, None, Some(id));
        id
    }

    fn flush(&mut self, states: &mut States<PrefixState>, secs: u64) -> bool {
        let ddl = Instant::now() + Duration::from_secs(secs);
        self.disk
            .flush_points(states.store_mut(), &mut NumberedBlocks, ddl)
    }
}

fn make_cell_state(
    cfg: &HyV4FlashConfig,
    tok: &[u32],
    chunks: &[usize],
    patterned: bool,
) -> (OwnerCell, PrefixState) {
    let width = cfg.mla.kv_row();
    let il = cfg
        .trunk()
        .iter()
        .position(|l| l.attn == Attn::Indexed)
        .unwrap();
    let idx_g = IndexerGeometry::of(cfg, il).unwrap();
    let mut cell = OwnerCell::new(cfg, None).unwrap();
    let mut pos = 0usize;
    for &chunk in chunks {
        let end = pos + chunk;
        let kv = if patterned {
            let bits: Vec<u16> = (pos..end)
                .flat_map(|p| {
                    (0..width).map(move |c| 0x3C00 | (((p * 29 + c * 7) & 0x03FF) as u16))
                })
                .collect();
            MxArray::from_float16(&bits, &[chunk as i64, width as i64]).unwrap()
        } else {
            MxArray::zeros(&[chunk as i64, width as i64], Some(DType::Float16)).unwrap()
        };
        for part in cell.layers_mut() {
            match part {
                CellLayer::Window(ring) => {
                    ring.append(&kv, pos).unwrap();
                }
                CellLayer::Index(ix) => {
                    let held = (idx_g.kpool - 1 + WINDOW_SLACK).min(end);
                    let r = if patterned {
                        let v: Vec<f32> = (0..held * idx_g.head_dim)
                            .map(|i| (end + i) as f32 * 0.25)
                            .collect();
                        arr(&v, &[held, idx_g.head_dim])
                    } else {
                        MxArray::zeros(&[held as i64, idx_g.head_dim as i64], Some(DType::Float32))
                            .unwrap()
                    };
                    let tail = TokTail::of_rows(&idx_g, Some(r), end).unwrap();
                    *ix = PagedIndexerState::at(ix.layer(), tail);
                }
            }
        }
        cell.advance(&tok[pos..end]).unwrap();
        pos = end;
    }
    assert_eq!(pos, tok.len());
    let state = cell.state().unwrap().unwrap();
    (cell, state)
}

#[test]
fn identity_text_is_pinned() {
    let shards = vec!["aa".into(), "bb".into()];
    let id = Identity {
        shards: &shards,
        config: "cc".into(),
        page: PAGE,
        window: 512,
        mtp: true,
        gpu: 17,
    };
    let on = identity_text(&id);
    assert_eq!(
        on,
        "hy4-prefix-v1\n\
         shard aa\n\
         shard bb\n\
         config cc\n\
         page block=32 layers=12 heads=1 half=288 dtype=Float16 token-major index=8x128 index-dtype=Float16\n\
         window 512 slack 3\n\
         mtp on\n\
         gpu 17\n"
    );
    let off = identity_text(&Identity { mtp: false, ..id });
    assert!(off.contains("mtp off\n"));
    let ((a, fa), (b, fb)) = (fingerprint(&on).unwrap(), fingerprint(&off).unwrap());
    assert!(a.len() == 32 && a != b && fa != fb);
}

#[test]
fn chain_keys_follow_tokens_keys_and_salt() {
    let fp = ColdCacheFingerprint::from_components([b"hy4".as_slice()]);
    let t = tokens(4 * BLOCK_ROWS + 5);
    let keys = vec![vec![1u64], vec![2], vec![3], vec![4]];
    let a = chain(ColdGroup::Kv, fp, &t, &keys, 9, 4);
    assert_eq!(a.len(), 4);
    assert_eq!(a, chain(ColdGroup::Kv, fp, &t, &keys, 9, 4));
    assert_eq!(chain(ColdGroup::Kv, fp, &t, &keys, 9, 2), a[..2]);
    let mut other = keys.clone();
    other[2] = vec![7];
    let b = chain(ColdGroup::Kv, fp, &t, &other, 9, 4);
    assert!(a[..2] == b[..2] && a[2] != b[2] && a[3] != b[3]);
    assert_ne!(a[0], chain(ColdGroup::Kv, fp, &t, &keys, 10, 1)[0]);
    assert_ne!(a[0], chain(ColdGroup::HybridDraft, fp, &t, &keys, 9, 1)[0]);
}

#[test]
fn segment_keys_need_the_next_token() {
    let fp = ColdCacheFingerprint::from_components([b"hy4".as_slice()]);
    let t = tokens(3 * BLOCK_ROWS);
    let draft = chain(ColdGroup::HybridDraft, fp, &t, &[], 0, 3);
    let (k0, k1) = (
        segment_key(fp, &draft, &t, 0).unwrap(),
        segment_key(fp, &draft, &t, 1).unwrap(),
    );
    assert!(k0 != k1 && segment_key(fp, &draft, &t, 2).is_none());
    let mut u = t.clone();
    u[BLOCK_ROWS] += 1;
    assert_ne!(Some(k0), segment_key(fp, &draft, &u, 0));
    assert_eq!(
        [0, BLOCK_ROWS, 2 * BLOCK_ROWS, 2 * BLOCK_ROWS + 1].map(segments_at),
        [0, 0, 1, 2]
    );
}

#[test]
fn split_and_join_round_trip_one_block() {
    let half = PAGE.half as usize;
    let latent: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| i as u16).collect();
    let pooled: Vec<u16> = (0..(PAGE.index_rows * PAGE.index_dim) as usize)
        .map(|i| 40_000 + i as u16)
        .collect();
    let layer = split_block(&latent, &pooled, half);
    let l = block_layout(&PAGE);
    assert_eq!(
        (layer.keys.len(), layer.values.len()),
        (l.key_bytes_per_layer, l.value_bytes_per_layer)
    );
    let u16_at = |b: &[u8], i: usize| u16::from_ne_bytes([b[i], b[i + 1]]);
    assert_eq!(u16_at(&layer.keys, 0), latent[0]);
    assert_eq!(u16_at(&layer.keys, 2 * half), latent[2 * half]);
    assert_eq!(u16_at(&layer.values, 0), latent[half]);
    assert_eq!(join_block(&layer, &PAGE).unwrap(), (latent, pooled));
}

#[test]
fn block_layout_matches_the_page() {
    let l = block_layout(&PAGE);
    assert_eq!(
        (l.block_size, l.num_layers, l.head_size),
        (BLOCK_ROWS as u32, 12, 288)
    );
    assert_eq!(l.cache_dtype, "Float16");
    assert_eq!(l.key_bytes_per_layer, BLOCK_ROWS * 288 * 2);
    assert_eq!(l.value_bytes_per_layer, BLOCK_ROWS * 288 * 2 + 8 * 128 * 2);
}

#[test]
fn prompt_and_turn_keys() {
    let t = tokens(6 * BLOCK_ROWS + 3);
    let pass: Vec<Vec<u64>> = (0..4).map(|i| vec![100 + i]).collect();
    assert_eq!(prompt_keys(&t, 0, &[], &pass, false, 5), pass);
    let line = vec![vec![100], vec![101]];
    assert_eq!(prompt_keys(&t, 2 * BLOCK_ROWS, &line, &pass, true, 5), pass);
    let warm = prompt_keys(&t, 2 * BLOCK_ROWS + 7, &line, &pass, false, 5);
    assert!(warm.len() == 6 && warm[..2] == line[..] && warm[2..].iter().all(|k| is_written(k)));
    assert_eq!(warm[2], written_key(&t, 2, 5));
    assert_ne!(written_key(&t, 2, 5), written_key(&t, 2, 6));
    let end = turn_keys(&t, &pass, 6 * BLOCK_ROWS, 5);
    assert!(end.len() == 6 && end[..4] == pass[..] && end[4] == written_key(&t, 4, 5));
    assert!(written_key(&t, 4, 5)[0] == WRITE_TAG && any_written(&end) && !any_written(&pass));
    let h = hash_tokens(&t[4 * BLOCK_ROWS..5 * BLOCK_ROWS], 5, &[4]);
    assert_eq!(end[4], vec![WRITE_TAG, h]);
}

#[test]
fn write_path_states_resume_their_lineage() {
    let t = tokens(4 * BLOCK_ROWS + 9);
    let pass: Vec<Vec<u64>> = (0..4).map(|i| vec![100 + i]).collect();
    let mut s = States::<PrefixState>::new();
    let k_p = Key {
        tokens: &t[..2 * BLOCK_ROWS],
        keys: &pass[..2],
        salt: 3,
    };
    let id = s
        .publish(k_p, bare(2 * BLOCK_ROWS), Kind::Prompt)
        .unwrap()
        .unwrap();
    let written = turn_keys(&t, &pass[..1], 3 * BLOCK_ROWS, 8);
    let k_t = Key {
        tokens: &t[..3 * BLOCK_ROWS],
        keys: &written,
        salt: 3,
    };
    let tid = s
        .publish(k_t, bare(3 * BLOCK_ROWS), Kind::Turn)
        .unwrap()
        .unwrap();
    let prompt = Key {
        tokens: &t,
        keys: &pass,
        salt: 3,
    };
    let e = s.store().get(id).unwrap();
    let other: Vec<Vec<u64>> = (0..4).map(|i| vec![200 + i]).collect();
    assert!(
        resumes(e, prompt)
            && !resumes(
                e,
                Key {
                    keys: &other,
                    ..prompt
                }
            )
    );
    assert!(!resumes(e, Key { salt: 4, ..prompt }) && resumes(s.store().get(tid).unwrap(), prompt));
    assert_eq!(s.lookup(prompt, &|_| true), Some((tid, 3 * BLOCK_ROWS)));
    assert_eq!(s.lookup(prompt, &|_| false), None);
    assert_eq!(s.keys_of(tid).unwrap(), written);
    assert!(s.disk_only_above(prompt, 0).is_empty());
}

#[test]
fn a_kv_only_prefix_is_never_admitted_without_a_state() {
    assert_eq!(lookup_bounds(None, 100), (true, 99));
    assert_eq!(lookup_bounds(Some(64), 100), (false, 64));
}

#[test]
fn disk_plan_follows_the_options() {
    let (tier, on) = disk_plan("Hy4", Some("0"), None, None, None).unwrap();
    assert!(!on && tier.cache_disk == CacheDisk::Off);
    let (tier, on) = disk_plan("Hy4", None, None, None, Some("1")).unwrap();
    assert!(on && tier.persistent);
    let (tier, _) = disk_plan("Hy4", None, Some(false), None, Some("1")).unwrap();
    assert!(!tier.persistent);
    assert!(disk_plan("Hy4", None, None, Some("/x".into()), None).is_err());
}

#[test]
fn a_block_round_trips_through_a_temporary_tier() {
    let root = Temp::new("persist");
    let shape = StateShape::of(&tiny_cfg(), false).unwrap();
    let rig = TierRig::open(&root, None, &shape);
    let half = PAGE.half as usize;
    let t = tokens(MINIMUM_ROWS);
    let key = chain(ColdGroup::Kv, rig.fp, &t, &[], 0, 1)[0];
    let layers: Vec<_> = (0..PAGE.layers as usize)
        .map(|l| {
            let latent: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| (i + l) as u16).collect();
            let pooled = vec![l as u16; (PAGE.index_rows * PAGE.index_dim) as usize];
            split_block(&latent, &pooled, half)
        })
        .collect();
    rig.offer_block(key, &t[..BLOCK_ROWS], layers.clone());
    assert!(rig.manager.drain(Duration::from_secs(5)) && rig.manager.contains(&key));
    let back = rig.manager.load(key, rig.fp).unwrap();
    assert_eq!(back.layers.len(), layers.len());
    for (a, b) in back.layers.iter().zip(&layers) {
        assert_eq!((&a.keys, &a.values), (&b.keys, &b.values));
    }
}

#[test]
fn block_runs_read_a_chain_back_in_runs() {
    let root = Temp::new("persist");
    let shape = StateShape::of(&tiny_cfg(), false).unwrap();
    let rig = TierRig::open(&root, None, &shape);
    let (half, pooled_len) = (
        PAGE.half as usize,
        (PAGE.index_rows * PAGE.index_dim) as usize,
    );
    let n = FLAT_RUN + 1;
    let t = tokens(n * BLOCK_ROWS);
    let keys = chain(ColdGroup::Kv, rig.fp, &t, &[], 0, n);
    let bit = |b: usize, l: usize, i: usize| 0x3C00 | ((b * 131 + l * 17 + i) % 1024) as u16;
    for (b, &key) in keys.iter().enumerate() {
        let layers = (0..PAGE.layers as usize)
            .map(|l| {
                let lat: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| bit(b, l, i)).collect();
                let pool: Vec<u16> = (0..pooled_len).map(|i| bit(b, l, i + 7)).collect();
                split_block(&lat, &pool, half)
            })
            .collect();
        rig.offer_block(key, &t[b * BLOCK_ROWS..(b + 1) * BLOCK_ROWS], layers);
    }
    assert!(rig.manager.drain(Duration::from_secs(5)));
    let mut runs = rig.disk.block_runs(&t, &[], 0, n * BLOCK_ROWS).unwrap();
    let read: Vec<_> = runs.by_ref().collect();
    assert!(!runs.failed() && read.len() == 2);
    let mut first = 0;
    for run in &read {
        assert_eq!(run.len(), PAGE.layers as usize);
        let blocks = run[0].0.shape_at(0).unwrap() as usize / BLOCK_ROWS;
        for (l, (rows, pooled)) in run.iter().enumerate() {
            assert_eq!(rows.shape_at(1).unwrap(), 2 * half as i64);
            let (rows, pooled) = (
                rows.to_uint16_native().unwrap(),
                pooled.to_uint16_native().unwrap(),
            );
            for b in 0..blocks {
                let row_len = BLOCK_ROWS * 2 * half;
                let want_r: Vec<u16> = (0..row_len).map(|i| bit(first + b, l, i)).collect();
                assert_eq!(
                    rows[b * row_len..(b + 1) * row_len],
                    want_r[..],
                    "b={} l={l}",
                    first + b
                );
                let want_p: Vec<u16> = (0..pooled_len).map(|i| bit(first + b, l, i + 7)).collect();
                assert_eq!(
                    pooled[b * pooled_len..(b + 1) * pooled_len],
                    want_p[..],
                    "b={} l={l}",
                    first + b
                );
            }
        }
        first += blocks;
    }
    assert_eq!(first, n);
    assert!(rig.manager.remove(&keys[FLAT_RUN]));
    let mut runs = rig.disk.block_runs(&t, &[], 0, n * BLOCK_ROWS).unwrap();
    assert_eq!(runs.by_ref().count(), 1);
    assert!(runs.failed());
}

/// The pool's page of a block holds the bytes `split_block` makes of the
/// same latent rows and pooled keys. Needs Metal (skips without it).
#[cfg(target_os = "macos")]
#[test]
fn pool_blobs_are_split_block_of_the_rows() {
    use mlx_paged_attn::metal::MetalDtype;
    use mlx_paged_attn::{LayerKVPool, PageLayout, PagedAttentionConfig};

    let (half, dk, per) = (
        PAGE.half as usize,
        PAGE.index_dim as usize,
        PAGE.index_rows as usize,
    );
    let n = 2 * BLOCK_ROWS;
    let config = PagedAttentionConfig {
        block_size: BLOCK_ROWS as u32,
        gpu_memory_mb: 256,
        head_size: PAGE.half,
        num_kv_heads: 1,
        num_layers: 1,
        max_seq_len: Some(1024),
        max_batch_size: Some(4),
    };
    let layout = PageLayout {
        values_token_major: true,
        keys_token_major: false,
        index_rows: PAGE.index_rows,
        index_dimension: PAGE.index_dim,
    };
    let Ok(pool) = LayerKVPool::new_with_index_dtype(
        config,
        4,
        4,
        MetalDtype::Float16,
        layout,
        MetalDtype::Float16,
    ) else {
        return;
    };
    let bit = |t: usize, c: usize| 0x3C00 | ((t * 37 + c * 11) % 1024) as u16;
    let latent: Vec<u16> = (0..n)
        .flat_map(|t| (0..2 * half).map(move |c| bit(t, c)))
        .collect();
    let groups = n / (BLOCK_ROWS / per);
    let pooled: Vec<u16> = (0..groups * dk)
        .map(|i| 0x3800 | ((i * 5) % 1024) as u16)
        .collect();
    let plane = |lo: usize| {
        let bits: Vec<u16> = (0..n)
            .flat_map(|t| (lo..lo + half).map(move |c| bit(t, c)))
            .collect();
        MxArray::from_float16(&bits, &[n as i64, 1, half as i64]).unwrap()
    };
    let (keys, values) = (plane(0), plane(half));
    let table = [2usize, 1];
    let slots: Vec<i64> = (0..n)
        .map(|t| (table[t / BLOCK_ROWS] * BLOCK_ROWS + t % BLOCK_ROWS) as i64)
        .collect();
    let slots = MxArray::from_int64(&slots, &[n as i64]).unwrap();
    let k_pool = MxArray::from_handle(pool.key_cache_array_raw(0).unwrap(), "k pool").unwrap();
    let v_pool = MxArray::from_handle(pool.value_cache_array_raw(0).unwrap(), "v pool").unwrap();
    let unit = arr(&[1.0], &[1]);
    let mut out = [std::ptr::null_mut(); 4];
    let [ko, vo, kso, vso] = &mut out;
    // SAFETY: live MLX array handles; outputs written on success and owned here.
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            k_pool.as_raw_ptr(),
            v_pool.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            slots.as_raw_ptr(),
            unit.as_raw_ptr(),
            unit.as_raw_ptr(),
            BLOCK_ROWS as i32,
            1,
            half as i32,
            0,
            true,
            ko,
            vo,
            kso,
            vso,
        )
    };
    assert!(ok);
    let written = out.map(|raw| MxArray::from_handle(raw, "store output").unwrap());
    let islots: Vec<i32> = (0..groups)
        .map(|g| (table[g / per] * per + g % per) as i32)
        .collect();
    let islots = MxArray::from_int32(&islots, &[groups as i64]).unwrap();
    let rows = MxArray::from_float16(&pooled, &[groups as i64, dk as i64]).unwrap();
    let index = MxArray::from_handle(pool.index_array_raw(0).unwrap(), "index plane").unwrap();
    let mut stored = std::ptr::null_mut();
    // SAFETY: live MLX array handles; `stored` written on success and owned here.
    let ok = unsafe {
        mlx_sys::mlx_paged_index_store(
            index.as_raw_ptr(),
            rows.as_raw_ptr(),
            islots.as_raw_ptr(),
            &mut stored,
        )
    };
    assert!(ok);
    let stored = MxArray::from_handle(stored, "index store").unwrap();
    let mut live: Vec<&MxArray> = written.iter().collect();
    live.push(&stored);
    MxArray::eval_arrays_with_context(&live, "hy4 pool blob test").unwrap();
    crate::array::memory::synchronize();
    let row = 2 * half;
    for (b, &block) in table.iter().enumerate() {
        let blobs = pool.read_block_all_layers(block as u32).unwrap();
        let want = split_block(
            &latent[b * BLOCK_ROWS * row..(b + 1) * BLOCK_ROWS * row],
            &pooled[b * per * dk..(b + 1) * per * dk],
            half,
        );
        assert_eq!(
            (blobs.len(), &blobs[0].0, &blobs[0].1),
            (1, &want.keys, &want.values)
        );
    }
}

struct NumberedBlocks;

impl NumberedBlocks {
    fn layers(i: usize) -> Vec<ColdLayerBlock> {
        let half = PAGE.half as usize;
        (0..PAGE.layers as usize)
            .map(|l| {
                let lat: Vec<u16> = (0..BLOCK_ROWS * 2 * half)
                    .map(|r| ((i * 31 + l * 7 + r) % 4096) as u16)
                    .collect();
                let pool = vec![(i + l) as u16; (PAGE.index_rows * PAGE.index_dim) as usize];
                split_block(&lat, &pool, half)
            })
            .collect()
    }
}

impl Blocks for NumberedBlocks {
    fn begin(&mut self, _: &[u32]) -> bool {
        true
    }
    fn block(&mut self, i: usize, _: u64) -> Option<Vec<ColdLayerBlock>> {
        Some(Self::layers(i))
    }
}

/// Clean stop on a persistent tier copies chains past one host queue window.
#[test]
fn a_clean_stop_copies_every_block_past_the_queue_window() {
    let root = Temp::new("persist");
    let shape = StateShape::of(&tiny_cfg(), false).unwrap();
    let mut rig = TierRig::open(
        &root,
        Some((
            "0123456789abcdef0123456789abcdef",
            b"hy4-flush-window".as_slice(),
        )),
        &shape,
    );
    let block: u64 = NumberedBlocks::layers(0)
        .iter()
        .map(|l| (l.keys.len() + l.values.len()) as u64)
        .sum();
    let n = (QUEUE_BYTES / block) as usize + 3;
    let (t, keys, hot) = (tokens(n * BLOCK_ROWS), vec![Vec::new(); n], vec![0u64; n]);
    let mut states = States::<PrefixState>::with_cap(4);
    rig.disk.start(1, &t, &keys, &hot, 0);
    rig.disk
        .published(states.store_mut(), 1, n * BLOCK_ROWS, None, None);
    let flushed = rig.flush(&mut states, 120);
    let cold = chain(ColdGroup::Kv, rig.fp, &t, &keys, 0, n);
    let missing = cold.iter().filter(|k| !rig.manager.contains(k)).count();
    assert!(
        missing == 0 && flushed,
        "{missing} of {n} blocks missing, flushed={flushed}"
    );
}

/// Circular `WindowRing` wrapped across many multiples of `cap` round-trips via `HY4S`.
#[test]
fn hy4s_state_round_trips_wrapped_circular_window_ring() {
    let cfg = tiny_cfg();
    let shape = StateShape::of(&cfg, false).unwrap();
    let total = 2 * BLOCK_ROWS;
    let tok: Vec<u32> = (0..total as u32).map(|i| 7 + i).collect();
    let (cell, state) = make_cell_state(&cfg, &tok, &[19, 1, 3, 2, 1, 6, 32], true);

    let mut buf = vec![0u8; shape.span_bytes() as usize];
    assert!(encode_state_into(&state, &mut buf).unwrap() && &buf[..4] == b"HY4S");

    let decoded = decode_state(&buf, &shape, Vec::new()).unwrap();
    assert_eq!(decoded.row, total);
    let restored = OwnerCell::from_state(&cfg, &decoded, &tok, None).unwrap();

    for (l, (orig, back)) in cell.layers().iter().zip(restored.layers()).enumerate() {
        if let (CellLayer::Window(a), CellLayer::Window(b)) = (orig, back) {
            assert_eq!((a.len(), a.kept()), (b.len(), b.kept()), "layer {l}");
            let a_bits = a
                .snapshot()
                .unwrap()
                .rows()
                .unwrap()
                .to_uint16_native()
                .unwrap()
                .to_vec();
            let b_bits = b
                .snapshot()
                .unwrap()
                .rows()
                .unwrap()
                .to_uint16_native()
                .unwrap()
                .to_vec();
            assert_eq!(a_bits, b_bits, "layer {l} window bits");
            let StateLayer::Window(sw) = &decoded.layers[l] else {
                panic!("expected Window at {l}");
            };
            assert_eq!(
                sw.rows().unwrap().to_uint16_native().unwrap().to_vec(),
                a_bits
            );
        }
    }
}

/// Fast (< `DELAY`) decode turns must not supersede the prompt-end replay state,
/// while linear `Turn -> Turn` superseding continues to work.
#[test]
fn fast_decode_turn_does_not_supersede_prompt_end_state() {
    let root = Temp::new("persist");
    let cfg = tiny_cfg();
    let shape = StateShape::of(&cfg, false).unwrap();
    let pid = Some((
        "fedcba9876543210fedcba9876543210",
        b"hy4-fast-turn-restore".as_slice(),
    ));

    let (prompt_row, prompt_len) = (MINIMUM_ROWS, MINIMUM_ROWS + 33);
    let (turn1_row, turn2_row) = (prompt_row + 256, prompt_row + 512);
    let full = tokens(turn2_row + 16);
    let pass: Vec<Vec<u64>> = (0..prompt_row / BLOCK_ROWS)
        .map(|b| vec![0x5000_0000 + b as u64])
        .collect();
    let t1_keys = turn_keys(&full, &pass, turn1_row, 111);
    let t2_keys = turn_keys(&full, &t1_keys, turn2_row, 222);
    let mk = |row: usize| make_cell_state(&cfg, &full[..row], &[row], false).1;

    // Session 1: publish prompt-end state at 2048, then fast Turn 1 at 2304, then flush.
    {
        let mut rig = TierRig::open(&root, pid, &shape);
        let mut st = States::<PrefixState>::with_cap(8);
        let p_id = rig.publish(
            &mut st,
            1,
            &full[..prompt_row],
            &pass,
            0,
            mk(prompt_row),
            Kind::Prompt,
        );
        let t1_id = rig.publish(
            &mut st,
            1,
            &full[..turn1_row],
            &t1_keys,
            0,
            mk(turn1_row),
            Kind::Turn,
        );
        assert!(
            st.store().superseded(p_id),
            "raw store.superseded(p_id) must be true"
        );
        assert!(rig.flush(&mut st, 30));
        assert_eq!(
            rig.telem.unneeded.load(Ordering::Relaxed),
            0,
            "prompt-end must not be superseded"
        );
        for (id, label) in [(p_id, "prompt-end"), (t1_id, "turn")] {
            assert!(
                st.store()
                    .get(id)
                    .and_then(|e| e.disk())
                    .is_some_and(|d| d.landed),
                "{label}"
            );
        }
    }

    // Session 2: adopt from disk, verify repeated prompt finds `prompt_row`, and Turn1->Turn2 supersedes.
    {
        let mut rig = TierRig::open(&root, pid, &shape);
        let mut st = States::<PrefixState>::with_cap(8);
        rig.disk.adopt(st.store_mut());

        let pk = Key {
            tokens: &full[..prompt_len],
            keys: &pass,
            salt: 0,
        };
        let found: Vec<usize> = st
            .store()
            .entries()
            .filter(|e| resumes(e, pk))
            .map(|e| e.depth())
            .collect();
        assert_eq!(
            found,
            vec![prompt_row],
            "restarted store must find prompt-end state"
        );

        rig.publish(
            &mut st,
            2,
            &full[..turn1_row],
            &t1_keys,
            7,
            mk(turn1_row),
            Kind::Turn,
        );
        rig.publish(
            &mut st,
            2,
            &full[..turn2_row],
            &t2_keys,
            7,
            mk(turn2_row),
            Kind::Turn,
        );
        assert!(rig.flush(&mut st, 30));
        assert_eq!(
            rig.telem.unneeded.load(Ordering::Relaxed),
            1,
            "linear Turn 1 should supersede"
        );
    }
}

/// Multi-turn cold `Kind::Prompt` states (`!any_written`) stay resumable across
/// Turn 1 -> Turn 2 -> Turn 3 because `Grid::ends` preserves earlier assistant
/// opening boundaries so `pass_keys` is prefix-monotonic.
#[test]
fn cold_prompt_state_resumes_across_multi_turn_prompts() {
    use super::prefill::Grid;
    use super::sched::pass_keys;

    const OPEN: [u32; 3] = [120_000, 611, 10_372];
    const GEN: [u32; 5] = [120_000, 611, 10_372, 120_001, 120_029];
    const CHUNK: usize = 4096;
    const SALT: u64 = 19;

    let g = Grid::new(CHUNK, OPEN.to_vec()).unwrap();
    let mut st = States::<PrefixState>::with_cap(8);

    // Turn 1: 5000 tokens ending in GEN (opening at 4995 -> replay_row 4992).
    let mut p1: Vec<u32> = (0..5000).map(|i| 17 + (i % 97) as u32).collect();
    p1[5000 - GEN.len()..].copy_from_slice(&GEN);
    let r1 = g.replay_row(&p1);
    assert_eq!(r1, 4992);
    let ends1 = g.ends(0, &p1);
    let pass1 = pass_keys(&p1, &ends1, CHUNK);
    assert!(!any_written(&pass1));
    let id1 = st
        .publish(
            Key {
                tokens: &p1[..r1],
                keys: &pass1[..r1 / BLOCK_ROWS],
                salt: SALT,
            },
            bare(r1),
            Kind::Prompt,
        )
        .unwrap()
        .unwrap();

    // Turn 2: extends Turn 1 (mutating post-OPEN token 4998 to simulate think
    // stripping) + assistant reply + user msg + GEN -> len 6500, r2 = 6464.
    let mut p2 = p1.clone();
    p2[4998] = 999;
    p2.extend((0..1500).map(|i| 31 + (i % 89) as u32));
    let len2 = p2.len();
    p2[len2 - GEN.len()..].copy_from_slice(&GEN);
    let r2 = g.replay_row(&p2);
    assert_eq!(r2, 6464);
    let ends2 = g.ends(0, &p2);
    let pass2 = pass_keys(&p2, &ends2, CHUNK);
    assert!(
        ends2.contains(&r1),
        "Turn 2 grid ends must contain Turn 1 replay_row"
    );
    assert_eq!(&pass2[..r1 / BLOCK_ROWS], &pass1[..r1 / BLOCK_ROWS]);

    let pk2 = Key {
        tokens: &p2,
        keys: &pass2,
        salt: SALT,
    };
    assert!(
        resumes(st.store().get(id1).unwrap(), pk2),
        "cold Turn 1 Kind::Prompt (!any_written) must resume on Turn 2"
    );
    assert_eq!(st.lookup(pk2, &|_| true), Some((id1, r1)));

    // Resuming Turn 2 from id1 at r1 (a cold end!) keeps Turn 2's Prompt cold too.
    let line1 = st.keys_of(id1).unwrap();
    let keys2 = prompt_keys(&p2, r1, &line1, &pass2, ends2.contains(&r1), 42);
    assert!(
        !any_written(&keys2),
        "resuming at cold boundary must remain cold"
    );
    let id2 = st
        .publish(
            Key {
                tokens: &p2[..r2],
                keys: &keys2[..r2 / BLOCK_ROWS],
                salt: SALT,
            },
            bare(r2),
            Kind::Prompt,
        )
        .unwrap()
        .unwrap();

    // Turn 3: crosses 8192 checkpoint -> len 11_000, r3 = 10_976.
    let mut p3 = p2.clone();
    p3.extend((0..4500).map(|i| 53 + (i % 79) as u32));
    let len3 = p3.len();
    p3[len3 - GEN.len()..].copy_from_slice(&GEN);
    let ends3 = g.ends(0, &p3);
    let pass3 = pass_keys(&p3, &ends3, CHUNK);
    assert!(ends3.contains(&r1) && ends3.contains(&r2));
    let pk3 = Key {
        tokens: &p3,
        keys: &pass3,
        salt: SALT,
    };
    assert!(resumes(st.store().get(id1).unwrap(), pk3));
    assert!(resumes(st.store().get(id2).unwrap(), pk3));
    assert_eq!(st.lookup(pk3, &|_| true), Some((id2, r2)));
}

/// Warm `Kind::Prompt` states (`any_written == true`, e.g. Turn 2+ resumed from
/// a prior `Kind::Turn`) and `Checkpoint -> Prompt` upgrades must NOT be
/// superseded by a fast `< DELAY` `Kind::Turn`, while linear `Turn -> Turn`
/// still supersedes.
#[test]
fn warm_prompt_and_upgraded_checkpoint_are_not_superseded_by_fast_turn() {
    let root = Temp::new("persist-warm-prompt");
    let cfg = tiny_cfg();
    let shape = StateShape::of(&cfg, false).unwrap();
    let pid = Some((
        "00112233445566778899aabbccddeeff",
        b"hy4-warm-prompt-supersede".as_slice(),
    ));

    let prompt_row = MINIMUM_ROWS + 256;
    let turn_row = prompt_row + 256;
    let turn2_row = turn_row + 256;
    let full = tokens(turn2_row + 32);

    // Warm keys: Turn 2 resumed from a Turn 1 write-path state, so block 0..
    // already carries WRITE_TAG -> `any_written(&warm_prompt_keys)` is TRUE.
    let pass: Vec<Vec<u64>> = (0..MINIMUM_ROWS / BLOCK_ROWS)
        .map(|b| vec![0x6000_0000 + b as u64])
        .collect();
    let prev_turn_keys = turn_keys(&full, &pass[..1], MINIMUM_ROWS, 333);
    let warm_prompt_keys = prompt_keys(
        &full[..prompt_row],
        MINIMUM_ROWS,
        &prev_turn_keys,
        &[],
        false,
        444,
    );
    assert!(
        any_written(&warm_prompt_keys),
        "warm Turn 2 prompt keys must carry write-path keys"
    );
    let fast_turn_keys = turn_keys(&full, &warm_prompt_keys, turn_row, 555);
    let turn2_keys = turn_keys(&full, &fast_turn_keys, turn2_row, 666);
    let mk = |row: usize| make_cell_state(&cfg, &full[..row], &[row], false).1;

    let mut rig = TierRig::open(&root, pid, &shape);
    let mut st = States::<PrefixState>::with_cap(8);

    // 1. Warm `Kind::Prompt` (`any_written == true`) followed immediately (< DELAY)
    //    by `Kind::Turn` on the same lineage: Prompt must NOT be superseded.
    let p_id = rig.publish(
        &mut st,
        1,
        &full[..prompt_row],
        &warm_prompt_keys,
        0,
        mk(prompt_row),
        Kind::Prompt,
    );
    let t_id = rig.publish(
        &mut st,
        1,
        &full[..turn_row],
        &fast_turn_keys,
        0,
        mk(turn_row),
        Kind::Turn,
    );
    assert!(
        st.store().superseded(p_id),
        "raw store sees Turn descendant"
    );
    assert!(rig.flush(&mut st, 30));
    assert_eq!(
        rig.telem.unneeded.load(Ordering::Relaxed),
        0,
        "warm Kind::Prompt must not be superseded by fast Kind::Turn"
    );
    for (id, label) in [(p_id, "warm prompt"), (t_id, "fast turn")] {
        assert!(
            st.store()
                .get(id)
                .and_then(|e| e.disk())
                .is_some_and(|d| d.landed),
            "{label} must land on disk"
        );
    }

    // 2. Checkpoint at `prompt_row` upgraded to `Kind::Prompt` at the exact same
    //    block (and never downgraded by a subsequent Checkpoint/Turn at the same
    //    block) must ALSO be protected from superseding by a deeper `Kind::Turn`.
    let ck_id = st
        .publish(
            Key {
                tokens: &full[..prompt_row],
                keys: &warm_prompt_keys,
                salt: 9,
            },
            mk(prompt_row),
            Kind::Checkpoint,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        st.store().get(ck_id).unwrap().ram().unwrap().kind,
        Kind::Checkpoint
    );
    let upgraded_id = rig.publish(
        &mut st,
        2,
        &full[..prompt_row],
        &warm_prompt_keys,
        9,
        mk(prompt_row),
        Kind::Prompt,
    );
    assert_eq!(upgraded_id, ck_id, "reuses existing entry on same block");
    assert_eq!(
        st.store().get(ck_id).unwrap().ram().unwrap().kind,
        Kind::Prompt,
        "Checkpoint must upgrade to Kind::Prompt"
    );
    assert!(!st.store().get(ck_id).unwrap().checkpoint());
    // Publishing Checkpoint or Turn on the same block must NEVER downgrade Prompt.
    for downgrade in [Kind::Checkpoint, Kind::Turn] {
        let same = st
            .publish(
                Key {
                    tokens: &full[..prompt_row],
                    keys: &warm_prompt_keys,
                    salt: 9,
                },
                mk(prompt_row),
                downgrade,
            )
            .unwrap()
            .unwrap();
        assert_eq!(same, ck_id);
        assert_eq!(
            st.store().get(ck_id).unwrap().ram().unwrap().kind,
            Kind::Prompt
        );
    }
    let t9_id = rig.publish(
        &mut st,
        2,
        &full[..turn_row],
        &fast_turn_keys,
        9,
        mk(turn_row),
        Kind::Turn,
    );
    assert!(rig.flush(&mut st, 30));
    assert_eq!(
        rig.telem.unneeded.load(Ordering::Relaxed),
        0,
        "Checkpoint upgraded to Prompt must not be superseded"
    );
    assert!(
        st.store()
            .get(ck_id)
            .and_then(|e| e.disk())
            .is_some_and(|d| d.landed)
    );
    assert!(
        st.store()
            .get(t9_id)
            .and_then(|e| e.disk())
            .is_some_and(|d| d.landed)
    );

    // 3. Linear Turn 1 -> Turn 2 on a fresh salt still supersedes Turn 1.
    let before_unneeded = rig.telem.unneeded.load(Ordering::Relaxed);
    rig.publish(
        &mut st,
        3,
        &full[..turn_row],
        &fast_turn_keys,
        11,
        mk(turn_row),
        Kind::Turn,
    );
    rig.publish(
        &mut st,
        3,
        &full[..turn2_row],
        &turn2_keys,
        11,
        mk(turn2_row),
        Kind::Turn,
    );
    assert!(rig.flush(&mut st, 30));
    assert_eq!(
        rig.telem.unneeded.load(Ordering::Relaxed) - before_unneeded,
        1,
        "linear Turn -> Turn must still supersede"
    );
}
