//! The stored states' disk tier: the identity text, chain and segment keys,
//! a block's bytes from either source, the record codec, the turn keys,
//! the lookup bounds and a block's trip through a temporary tier. CPU only
//! unless marked.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mlx_paged_attn::{ColdCacheBlock, ColdCacheFingerprint, ColdGroup, ColdObject, hash_tokens};

use super::ladder::Kind;
use super::paged::PageGeometry;
use super::persist::{
    Disk, FLAT_RUN, Identity, MINIMUM_ROWS, chain, disk_plan, fingerprint, identity_text,
    segment_key, segments_at,
};
use super::persist_codec::{StateShape, block_layout, join_block, split_block};
use super::persist_writer::Telemetry;
use super::prefill::BLOCK_ROWS;
use super::prefix_state::{
    Key, PrefixState, States, WRITE_TAG, any_written, is_written, prompt_keys, resumes, turn_keys,
    written_key,
};
use super::sched::lookup_bounds;
use crate::disk_tier::{CacheDisk, DiskTierOptions, open_disk_tier};

const PAGE: PageGeometry = PageGeometry {
    layers: 12,
    half: 288,
    index_rows: 8,
    index_dim: 128,
};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("hy4-persist-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

/// Review-3 D4: the identity text binds the f16 index plane, the MTP
/// switch and the model; a change to any line is a new namespace.
#[test]
fn identity_text_is_pinned() {
    let shards = vec!["aa".to_string(), "bb".to_string()];
    let id = Identity {
        shards: &shards,
        config: "cc".to_string(),
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
    let (a, fa) = fingerprint(&on).unwrap();
    let (b, fb) = fingerprint(&off).unwrap();
    assert_eq!(a.len(), 32);
    assert_ne!(a, b);
    assert_ne!(fa, fb);
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
    assert_eq!(a[..2], b[..2]);
    assert_ne!(a[2], b[2]);
    assert_ne!(a[3], b[3]);
    assert_ne!(a[0], chain(ColdGroup::Kv, fp, &t, &keys, 10, 1)[0]);
    assert_ne!(a[0], chain(ColdGroup::HybridDraft, fp, &t, &keys, 9, 1)[0]);
}

/// Segment `k` hangs off the draft chain below `32(k + 1)` and the token
/// at that row; a lineage without that token has no key for it.
#[test]
fn segment_keys_need_the_next_token() {
    let fp = ColdCacheFingerprint::from_components([b"hy4".as_slice()]);
    let t = tokens(3 * BLOCK_ROWS);
    let draft = chain(ColdGroup::HybridDraft, fp, &t, &[], 0, 3);
    let k0 = segment_key(fp, &draft, &t, 0).unwrap();
    let k1 = segment_key(fp, &draft, &t, 1).unwrap();
    assert_ne!(k0, k1);
    assert!(segment_key(fp, &draft, &t, 2).is_none());
    let mut u = t.clone();
    u[BLOCK_ROWS] += 1;
    assert_ne!(Some(k0), segment_key(fp, &draft, &u, 0));
    assert_eq!(segments_at(0), 0);
    assert_eq!(segments_at(BLOCK_ROWS), 0);
    assert_eq!(segments_at(2 * BLOCK_ROWS), 1);
    assert_eq!(segments_at(2 * BLOCK_ROWS + 1), 2);
}

/// The pool's two 288-column planes: plane `h` of a row is columns
/// `[288h, 288h + 288)` of its latent row, the index plane rides with the
/// values; join gives the token-major rows back.
#[test]
fn split_and_join_round_trip_one_block() {
    let half = PAGE.half as usize;
    let latent: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| i as u16).collect();
    let pooled: Vec<u16> = (0..(PAGE.index_rows * PAGE.index_dim) as usize)
        .map(|i| 40_000 + i as u16)
        .collect();
    let layer = split_block(&latent, &pooled, half);
    let layout = block_layout(&PAGE);
    assert_eq!(layer.keys.len(), layout.key_bytes_per_layer);
    assert_eq!(layer.values.len(), layout.value_bytes_per_layer);
    let first_key = u16::from_ne_bytes([layer.keys[0], layer.keys[1]]);
    let row1_key = u16::from_ne_bytes([layer.keys[2 * half], layer.keys[2 * half + 1]]);
    let first_value = u16::from_ne_bytes([layer.values[0], layer.values[1]]);
    assert_eq!(first_key, latent[0]);
    assert_eq!(row1_key, latent[2 * half]);
    assert_eq!(first_value, latent[half]);
    let (rows, keys) = join_block(&layer, &PAGE).unwrap();
    assert_eq!(rows, latent);
    assert_eq!(keys, pooled);
}

#[test]
fn block_layout_matches_the_page() {
    let l = block_layout(&PAGE);
    assert_eq!(l.block_size, BLOCK_ROWS as u32);
    assert_eq!(l.num_layers, 12);
    assert_eq!(l.head_size, 288);
    assert_eq!(l.cache_dtype, "Float16");
    assert_eq!(l.key_bytes_per_layer, BLOCK_ROWS * 288 * 2);
    assert_eq!(l.value_bytes_per_layer, BLOCK_ROWS * 288 * 2 + 8 * 128 * 2);
}

/// A cold prefill keeps its pass keys; a prefill from a row no cold
/// prefill cuts at writes write-path keys from the block holding it, and
/// a turn's end covers its decode rows with them.
#[test]
fn prompt_and_turn_keys() {
    let t = tokens(6 * BLOCK_ROWS + 3);
    let pass: Vec<Vec<u64>> = (0..4).map(|i| vec![100 + i]).collect();
    assert_eq!(prompt_keys(&t, 0, &[], &pass, false, 5), pass);
    let line = vec![vec![100], vec![101]];
    let cold = prompt_keys(&t, 2 * BLOCK_ROWS, &line, &pass, true, 5);
    assert_eq!(cold, pass);
    let warm = prompt_keys(&t, 2 * BLOCK_ROWS + 7, &line, &pass, false, 5);
    assert_eq!(warm.len(), 6);
    assert_eq!(warm[..2], line[..]);
    assert!(warm[2..].iter().all(|k| is_written(k)));
    assert_eq!(warm[2], written_key(&t, 2, 5));
    assert_ne!(written_key(&t, 2, 5), written_key(&t, 2, 6));
    let end = turn_keys(&t, &pass, 6 * BLOCK_ROWS, 5);
    assert_eq!(end.len(), 6);
    assert_eq!(end[..4], pass[..]);
    assert_eq!(end[4], written_key(&t, 4, 5));
    assert_eq!(written_key(&t, 4, 5)[0], WRITE_TAG);
    assert!(any_written(&end));
    assert!(!any_written(&pass));
    let h = hash_tokens(&t[4 * BLOCK_ROWS..5 * BLOCK_ROWS], 5, &[4]);
    assert_eq!(end[4], vec![WRITE_TAG, h]);
}

/// A state under write-path keys resumes the lineage that wrote it; one
/// under pass keys only a prompt with those keys.
#[test]
fn write_path_states_resume_their_lineage() {
    let t = tokens(4 * BLOCK_ROWS + 9);
    let pass: Vec<Vec<u64>> = (0..4).map(|i| vec![100 + i]).collect();
    let mut s = States::<PrefixState>::new();
    let id = s
        .publish(
            Key {
                tokens: &t[..2 * BLOCK_ROWS],
                keys: &pass[..2],
                salt: 3,
            },
            bare(2 * BLOCK_ROWS),
            Kind::Prompt,
        )
        .unwrap()
        .unwrap();
    let written = turn_keys(&t, &pass[..1], 3 * BLOCK_ROWS, 8);
    let tid = s
        .publish(
            Key {
                tokens: &t[..3 * BLOCK_ROWS],
                keys: &written,
                salt: 3,
            },
            bare(3 * BLOCK_ROWS),
            Kind::Turn,
        )
        .unwrap()
        .unwrap();
    let prompt = Key {
        tokens: &t,
        keys: &pass,
        salt: 3,
    };
    let e = s.store().get(id).unwrap();
    assert!(resumes(e, prompt));
    let other: Vec<Vec<u64>> = (0..4).map(|i| vec![200 + i]).collect();
    assert!(!resumes(
        e,
        Key {
            keys: &other,
            ..prompt
        }
    ));
    assert!(!resumes(e, Key { salt: 4, ..prompt }));
    assert!(resumes(s.store().get(tid).unwrap(), prompt));
    assert_eq!(s.lookup(prompt, &|_| true), Some((tid, 3 * BLOCK_ROWS)));
    assert_eq!(s.lookup(prompt, &|_| false), None);
    assert_eq!(s.keys_of(tid).unwrap(), written);
    assert!(s.disk_only_above(prompt, 0).is_empty());
}

/// No stored state, no pool lookup: a KV-only cold prefix has nothing to
/// resume the out-of-pool state with; with one, never past its row.
#[test]
fn a_kv_only_prefix_is_never_admitted_without_a_state() {
    assert_eq!(lookup_bounds(None, 100), (true, 99));
    assert_eq!(lookup_bounds(Some(64), 100), (false, 64));
}

#[test]
fn disk_plan_follows_the_options() {
    let (tier, on) = disk_plan("Hy4", Some("0"), None, None, None).unwrap();
    assert!(!on);
    assert_eq!(tier.cache_disk, CacheDisk::Off);
    let (tier, on) = disk_plan("Hy4", None, None, None, Some("1")).unwrap();
    assert!(on);
    assert!(tier.persistent);
    let (tier, _) = disk_plan("Hy4", None, Some(false), None, Some("1")).unwrap();
    assert!(!tier.persistent);
    assert!(disk_plan("Hy4", None, None, Some("/x".to_string()), None).is_err());
}

/// A block copied out through the tier comes back with the same bytes,
/// under the chain key of its lineage.
#[test]
fn a_block_round_trips_through_a_temporary_tier() {
    let root = Temp::new();
    let options = DiskTierOptions {
        temp_root: Some(root.0.clone()),
        ..DiskTierOptions::from_load(Some(&(1u64 << 30).to_string()), false, None).unwrap()
    };
    let shape = StateShape::of(&super::tests_model::tiny_cfg(), false).unwrap();
    let tier = open_disk_tier(&options, || None, shape.span_bytes()).unwrap();
    let manager = Arc::clone(&tier.manager);
    let fp = tier.state_fingerprint;
    let disk = Disk::open(tier, Arc::new(Telemetry::default()), shape, PAGE).unwrap();
    let half = PAGE.half as usize;
    let t = tokens(MINIMUM_ROWS);
    let key = chain(ColdGroup::Kv, fp, &t, &[], 0, 1)[0];
    let layers: Vec<_> = (0..PAGE.layers as usize)
        .map(|l| {
            let latent: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| (i + l) as u16).collect();
            let pooled: Vec<u16> = vec![l as u16; (PAGE.index_rows * PAGE.index_dim) as usize];
            split_block(&latent, &pooled, half)
        })
        .collect();
    let block = ColdCacheBlock {
        key,
        fingerprint: fp,
        tokens: t[..BLOCK_ROWS].to_vec(),
        layout: block_layout(&PAGE),
        layers: layers.clone(),
    };
    let left = manager
        .offer(
            ColdObject::Block(block),
            Instant::now() + Duration::from_secs(5),
        )
        .unwrap();
    assert!(left.is_none());
    assert!(manager.drain(Duration::from_secs(5)));
    assert!(manager.contains(&key));
    let back = manager.load(key, fp).unwrap();
    assert_eq!(back.layers.len(), layers.len());
    for (a, b) in back.layers.iter().zip(&layers) {
        assert_eq!(a.keys, b.keys);
        assert_eq!(a.values, b.values);
    }
    drop(disk);
}

/// A flat restore reads a chain back a run at a time, rows and pooled keys
/// in chain order, and a block missing from the tier ends the runs as a
/// failure. Needs Metal for the arrays.
#[test]
fn block_runs_read_a_chain_back_in_runs() {
    let root = Temp::new();
    let options = DiskTierOptions {
        temp_root: Some(root.0.clone()),
        ..DiskTierOptions::from_load(Some(&(1u64 << 30).to_string()), false, None).unwrap()
    };
    let shape = StateShape::of(&super::tests_model::tiny_cfg(), false).unwrap();
    let tier = open_disk_tier(&options, || None, shape.span_bytes()).unwrap();
    let manager = Arc::clone(&tier.manager);
    let fp = tier.state_fingerprint;
    let disk = Disk::open(tier, Arc::new(Telemetry::default()), shape, PAGE).unwrap();
    let half = PAGE.half as usize;
    let pooled_len = (PAGE.index_rows * PAGE.index_dim) as usize;
    let n = FLAT_RUN + 1;
    let t = tokens(n * BLOCK_ROWS);
    let keys = chain(ColdGroup::Kv, fp, &t, &[], 0, n);
    let bit = |b: usize, l: usize, i: usize| 0x3C00 | ((b * 131 + l * 17 + i) % 1024) as u16;
    for (b, &key) in keys.iter().enumerate() {
        let layers = (0..PAGE.layers as usize)
            .map(|l| {
                let latent: Vec<u16> = (0..BLOCK_ROWS * 2 * half).map(|i| bit(b, l, i)).collect();
                let pooled: Vec<u16> = (0..pooled_len).map(|i| bit(b, l, i + 7)).collect();
                split_block(&latent, &pooled, half)
            })
            .collect();
        let block = ColdCacheBlock {
            key,
            fingerprint: fp,
            tokens: t[b * BLOCK_ROWS..(b + 1) * BLOCK_ROWS].to_vec(),
            layout: block_layout(&PAGE),
            layers,
        };
        let left = manager
            .offer(
                ColdObject::Block(block),
                Instant::now() + Duration::from_secs(5),
            )
            .unwrap();
        assert!(left.is_none());
    }
    assert!(manager.drain(Duration::from_secs(5)));
    let mut runs = disk.block_runs(&t, &[], 0, n * BLOCK_ROWS).unwrap();
    let read: Vec<_> = runs.by_ref().collect();
    assert!(!runs.failed());
    assert_eq!(read.len(), 2);
    let mut first = 0;
    for run in &read {
        assert_eq!(run.len(), PAGE.layers as usize);
        let blocks = run[0].0.shape_at(0).unwrap() as usize / BLOCK_ROWS;
        for (l, (rows, pooled)) in run.iter().enumerate() {
            assert_eq!(rows.shape_at(1).unwrap(), 2 * half as i64);
            let rows = rows.to_uint16_native().unwrap();
            let pooled = pooled.to_uint16_native().unwrap();
            for b in 0..blocks {
                let at = b * BLOCK_ROWS * 2 * half;
                let want: Vec<u16> = (0..BLOCK_ROWS * 2 * half)
                    .map(|i| bit(first + b, l, i))
                    .collect();
                assert_eq!(
                    rows[at..at + want.len()],
                    want[..],
                    "block {} layer {l}",
                    first + b
                );
                let at = b * pooled_len;
                let want: Vec<u16> = (0..pooled_len).map(|i| bit(first + b, l, i + 7)).collect();
                assert_eq!(
                    pooled[at..at + pooled_len],
                    want[..],
                    "block {} layer {l}",
                    first + b
                );
            }
        }
        first += blocks;
    }
    assert_eq!(first, n);
    assert!(manager.remove(&keys[FLAT_RUN]));
    let mut runs = disk.block_runs(&t, &[], 0, n * BLOCK_ROWS).unwrap();
    assert_eq!(runs.by_ref().count(), 1);
    assert!(runs.failed());
    drop(disk);
}

/// The pool's page of a block holds the bytes `split_block` makes of the
/// same latent rows and pooled keys: the flat source and the pool source
/// write one object. Needs Metal (skips without it).
#[test]
fn pool_blobs_are_split_block_of_the_rows() {
    use mlx_paged_attn::metal::MetalDtype;
    use mlx_paged_attn::{LayerKVPool, PageLayout, PagedAttentionConfig};

    use crate::array::MxArray;

    const BLOCKS: u32 = 4;
    let half = PAGE.half as usize;
    let dk = PAGE.index_dim as usize;
    let per = PAGE.index_rows as usize;
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
        index_rows: PAGE.index_rows,
        index_dimension: PAGE.index_dim,
    };
    let pool = match LayerKVPool::new_with_index_dtype(
        config,
        BLOCKS,
        BLOCKS,
        MetalDtype::Float16,
        layout,
        MetalDtype::Float16,
    ) {
        Ok(pool) => pool,
        Err(error) => {
            eprintln!("skipping the Hy4 pool blob test: {error}");
            return;
        }
    };
    let bit = |t: usize, c: usize| 0x3C00 | ((t * 37 + c * 11) % 1024) as u16;
    let latent: Vec<u16> = (0..n)
        .flat_map(|t| (0..2 * half).map(move |c| bit(t, c)))
        .collect();
    let groups = n / (BLOCK_ROWS / per);
    let pooled: Vec<u16> = (0..groups * dk)
        .map(|i| 0x3800 | ((i * 5) % 1024) as u16)
        .collect();
    let plane = |lo: usize| -> MxArray {
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
    let unit = MxArray::from_float32(&[1.0], &[1]).unwrap();
    let mut out = [std::ptr::null_mut(); 4];
    let [ko, vo, kso, vso] = &mut out;
    // SAFETY: every input is a live MLX array handle; each output is written
    // only on success and is then owned here.
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
    let index_slots: Vec<i32> = (0..groups)
        .map(|g| (table[g / per] * per + g % per) as i32)
        .collect();
    let index_slots = MxArray::from_int32(&index_slots, &[groups as i64]).unwrap();
    let rows = MxArray::from_float16(&pooled, &[groups as i64, dk as i64]).unwrap();
    let index = MxArray::from_handle(pool.index_array_raw(0).unwrap(), "index plane").unwrap();
    let mut stored = std::ptr::null_mut();
    // SAFETY: every input is a live MLX array handle; `stored` is written
    // only on success and is then owned here.
    let ok = unsafe {
        mlx_sys::mlx_paged_index_store(
            index.as_raw_ptr(),
            rows.as_raw_ptr(),
            index_slots.as_raw_ptr(),
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
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].0, want.keys, "block {b}: K plane");
        assert_eq!(
            blobs[0].1, want.values,
            "block {b}: V plane and index plane"
        );
    }
}
