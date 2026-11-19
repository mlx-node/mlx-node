use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use super::*;
use crate::array::alloc::SendPtr;
use crate::models::hy_v4_flash::scan::Scan;
use crate::moe_stream::io::{AlignedBuf, ShardSet};
use crate::utils::gguf_gpu::RepackDst;
use crate::utils::gguf_kquant::{KQuantArrays, KQuantRepacker, KQuantScales, repack_kquant};

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn moe(n_experts: usize, ff: usize, latent: usize) -> MoeConfig {
    MoeConfig {
        n_experts,
        n_used: 2,
        n_shared: 1,
        ff,
        latent,
        route_scale: 1.0,
        norm_topk: true,
    }
}

fn real_moe() -> MoeConfig {
    moe(512, 1536, 1536)
}

fn located(name: &str, dims: [usize; 3], ty: GgufTensorType, at: u64) -> Located {
    Located {
        shard: 0,
        data_offset: at,
        info: GgufTensorInfo {
            name: name.to_string(),
            n_dims: 3,
            dims: dims.iter().map(|&d| d as u64).collect(),
            tensor_type: ty,
            offset: 0,
        },
    }
}

/// Layer `il` of `class` with the loader dims (gate/up `{latent, ff, E}`,
/// down `{ff, latent, E}`), tensors back to back from file offset `at`.
fn layer(m: &MoeConfig, il: usize, class: ExpertClass, at: u64) -> (ExpertLayer, u64) {
    let types = class_types(class).unwrap();
    let geo = geometry(m, class).unwrap();
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let mut off = at;
    let mut locs = Vec::new();
    for (p, name) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .enumerate()
    {
        locs.push(located(
            &format!("blk.{il}.{name}.weight"),
            dims[p],
            types[p],
            off,
        ));
        off += geo[p].file_bytes() * m.n_experts as u64;
    }
    let [gate, up, down]: [Located; 3] = locs.try_into().unwrap();
    (
        ExpertLayer {
            class,
            gate,
            up,
            down,
        },
        off,
    )
}

/// Host memory of one projection's bank.
struct Host {
    geo: ProjGeo,
    w: Vec<u32>,
    s: Vec<u8>,
    b: Vec<u16>,
}

impl Host {
    fn new(geo: ProjGeo, slots: usize) -> Self {
        Self {
            geo,
            w: vec![0; slots * geo.rows * geo.weight_cols()],
            s: vec![0; slots * geo.rows * geo.scales_cols()],
            b: vec![0; slots * geo.rows * geo.biases_cols()],
        }
    }

    fn dst(&mut self) -> SlotDst {
        let ptrs = DstPtrs {
            weight: SendPtr(self.w.as_mut_ptr().cast()),
            scales: Some(SendPtr(self.s.as_mut_ptr())),
            biases: Some(SendPtr(self.b.as_mut_ptr().cast())),
            rows: self.s.len() / self.geo.scales_cols(),
            k: self.geo.k,
            layout: OutLayout::RowMajor,
        };
        // SAFETY: the Vecs hold every slot's rows and outlive each use of
        // the destination in these tests.
        unsafe { SlotDst::new(ptrs, self.geo.format, self.geo.rows) }.unwrap()
    }

    fn slot(&self, i: usize) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
        let g = self.geo;
        let r = |cols: usize| i * g.rows * cols..(i + 1) * g.rows * cols;
        (
            self.w[r(g.weight_cols())].to_vec(),
            self.s[r(g.scales_cols())].to_vec(),
            self.b[r(g.biases_cols())].to_vec(),
        )
    }
}

fn scale_bytes(s: &KQuantScales) -> Vec<u8> {
    match s {
        KQuantScales::Unsigned(v) => v.clone(),
        KQuantScales::Signed(v) => v.iter().map(|&x| x as u8).collect(),
    }
}

/// Expert `e` of a whole-tensor repack.
fn expert_of(a: &KQuantArrays, g: ProjGeo, e: usize) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
    let r = |cols: usize| e * g.rows * cols..(e + 1) * g.rows * cols;
    (
        a.weight[r(g.weight_cols())].to_vec(),
        scale_bytes(&a.scales)[r(g.scales_cols())].to_vec(),
        a.biases[r(g.biases_cols())].to_vec(),
    )
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("hy4-bank-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn file(&self, bytes: &[u8]) -> PathBuf {
        let p = self.0.join("data.bin");
        std::fs::write(&p, bytes).unwrap();
        p
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pool_over(path: &Path, threads: usize) -> IoPool {
    let shards = Arc::new(ShardSet::open(&[path]).unwrap());
    IoPool::new(shards, threads).unwrap()
}

#[test]
fn class_layout_follows_the_fixture() {
    let f = fixture("expert-classes.json");
    let (a, b) = (ids(&f["q6k_down_layers"]), ids(&f["q4k_down_layers"]));
    assert_eq!((a.len(), b.len()), (19, 22));
    let mut classes = vec![None; 42];
    for &il in &a {
        classes[il] = Some(ExpertClass::Q6kDown);
    }
    for &il in &b {
        classes[il] = Some(ExpertClass::Q4kDown);
    }
    let l = ClassLayout::new(&classes).unwrap();
    assert_eq!(l.layers, [a.clone(), b.clone()]);
    assert_eq!(l.of_layer[0], None);
    for (ci, list) in [a, b].iter().enumerate() {
        for (local, &il) in list.iter().enumerate() {
            assert_eq!(l.of_layer[il], Some((ci, local)), "layer {il}");
        }
    }
    assert!(ClassLayout::new(&[Some(ExpertClass::Bf16)]).is_err());
}

#[test]
fn slot_and_file_bytes_match_the_extents_fixture() {
    let f = fixture("expert-extents.json");
    let m = real_moe();
    for (key, class) in [("A", ExpertClass::Q6kDown), ("B", ExpertClass::Q4kDown)] {
        let s = &f["slots"][key];
        assert_eq!(
            slot_bytes(&m, class).unwrap(),
            s["mlx_slot_bytes"].as_u64().unwrap(),
            "{key} slot"
        );
        assert_eq!(
            file_bytes(&m, class).unwrap(),
            s["file_bytes_per_expert"].as_u64().unwrap(),
            "{key} file"
        );
        assert_eq!(
            slot_bytes(&m, class).unwrap() * 512,
            s["staging_layer_mlx_bytes"].as_u64().unwrap()
        );
    }
    for l in f["layers"].as_array().unwrap() {
        let class = match l["class"].as_str().unwrap() {
            "A" => ExpertClass::Q6kDown,
            "B" => ExpertClass::Q4kDown,
            _ => continue,
        };
        let geo = geometry(&m, class).unwrap();
        for (p, key) in ["gate", "up", "down"].iter().enumerate() {
            let t = &l[*key];
            assert_eq!(geo[p].file_bytes(), t["per_expert_bytes"].as_u64().unwrap());
            assert_eq!(
                geo[p].slot_bytes(),
                t["per_expert_mlx_bytes"].as_u64().unwrap()
            );
        }
    }
}

/// latent != ff: a gate or down tensor with the other projection's dims is
/// refused, as is a wrong type or class.
#[test]
fn expert_orientation_is_checked_per_projection() {
    let m = moe(4, 512, 256);
    let class = ExpertClass::Q6kDown;
    let geo = geometry(&m, class).unwrap();
    assert_eq!(
        geo.map(|g| (g.rows, g.k)),
        [(512, 256), (512, 256), (256, 512)]
    );
    let (good, _) = layer(&m, 1, class, 0);
    check_layer(1, &good, class, &geo, 4).unwrap();
    let mut swapped = good.clone();
    swapped.gate.info.dims = vec![512, 256, 4];
    assert!(check_layer(1, &swapped, class, &geo, 4).is_err());
    let mut swapped = good.clone();
    swapped.down.info.dims = vec![256, 512, 4];
    assert!(check_layer(1, &swapped, class, &geo, 4).is_err());
    let mut retyped = good.clone();
    retyped.down.info.tensor_type = GgufTensorType::Q4K;
    assert!(check_layer(1, &retyped, class, &geo, 4).is_err());
    assert!(check_layer(1, &good, ExpertClass::Q4kDown, &geo, 4).is_err());
    assert!(check_layer(1, &good, class, &geo, 5).is_err());
    assert!(geometry(&m, ExpertClass::Bf16).is_err());
    assert!(geometry(&moe(4, 512, 100), class).is_err());
}

#[test]
fn decode_misses_take_victims_then_hit() {
    let mut d = Directory::new(1, 2, 8, 2).unwrap();
    d.check_step(1, &[3, 5], &[]).unwrap();
    let p = d.plan(1, &[3, 5], &[], true);
    assert_eq!(p.victims.len(), 2);
    assert!(p.staging.is_empty() && p.copies.is_empty());
    assert!(p.slots.iter().all(|&s| s >= d.base()));
    d.untag(1, &p);
    d.publish(1, &p);
    d.complete(p.epoch);
    for (&e, &s) in [3u32, 5].iter().zip(&p.slots) {
        assert!(d.holds(1, e, s));
        assert!(!d.holds(0, e, s));
    }
    let q = d.plan(1, &[3, 5], &[], true);
    assert_eq!((q.hits, q.reads()), (2, 0));
    assert_eq!(q.slots, p.slots);
    d.cache().check().unwrap();
}

#[test]
fn prefill_misses_stage_and_keep_their_tags() {
    let mut d = Directory::new(1, 3, 8, 0).unwrap();
    let p = d.plan(0, &[1, 4, 7], &[], false);
    assert_eq!(p.staging, vec![1, 4, 7]);
    assert_eq!(p.slots, vec![1, 4, 7]);
    d.untag(0, &p);
    d.publish(0, &p);
    assert_eq!(d.plan(0, &[1, 4], &[], false).reads(), 0);
    // the next layer shares the one staging layer: only its own tag counts
    let q = d.plan(1, &[4], &[], false);
    assert_eq!(q.staging, vec![4]);
    d.untag(1, &q);
    d.publish(1, &q);
    assert!(!d.holds(0, 4, 4));
    assert!(d.holds(1, 4, 4));
    assert!(d.holds(0, 1, 1));
}

#[test]
fn staging_halves_alternate_by_layer() {
    let mut d = Directory::new(2, 4, 8, 0).unwrap();
    let p0 = d.plan(0, &[2], &[], false);
    let p1 = d.plan(1, &[2], &[], false);
    let p2 = d.plan(2, &[2], &[], false);
    assert_eq!((p0.slots[0], p1.slots[0], p2.slots[0]), (2, 10, 2));
    assert_eq!(d.total_slots(), 16);
}

#[test]
fn prefill_seed_copies_a_valid_staging_slot() {
    let mut d = Directory::new(1, 1, 8, 1).unwrap();
    let p = d.plan(0, &[1, 2], &[], false);
    assert_eq!(p.victims, vec![(1, 0)]);
    assert_eq!(p.staging, vec![2]);
    d.untag(0, &p);
    d.publish(0, &p);
    d.complete(p.epoch);
    d.new_turn();
    let q = d.plan(0, &[2], &[], false);
    assert_eq!(q.copies, vec![(2, 0)]);
    assert_eq!(q.reads(), 0);
    d.commit_copies(0, &q);
    assert!(d.holds(0, 2, d.base()));
    assert_eq!(d.cache().lookup(0, 1), None);
    d.cache().check().unwrap();
}

#[test]
fn abandon_frees_victims() {
    let mut d = Directory::new(1, 1, 8, 3).unwrap();
    let p = d.plan(0, &[0, 6], &[], true);
    d.untag(0, &p);
    d.abandon(&p);
    assert_eq!(d.cache().occupied(), 0);
    d.cache().check().unwrap();
    d.complete_all();
    let q = d.plan(0, &[0, 6], &[], true);
    assert_eq!(q.victims.len(), 2);
}

#[test]
fn promote_moves_decode_staging_copies() {
    let mut d = Directory::new(1, 4, 8, 2).unwrap();
    let p = d.plan(0, &[5], &[], false);
    assert_eq!(p.staging, vec![5]);
    d.untag(0, &p);
    d.publish(0, &p);
    let q = d.plan(0, &[5], &[], true);
    assert_eq!((q.reads(), q.slots.clone()), (0, vec![5]));
    d.complete_all();
    let moved = d.promote();
    assert_eq!(moved.len(), 1);
    assert_eq!(moved[0].0, 5);
    assert!(d.holds(0, 5, moved[0].1));
    d.cache().check().unwrap();
}

#[test]
fn step_checks_refuse_bad_input() {
    let d = Directory::new(1, 2, 8, 0).unwrap();
    assert!(d.check_step(2, &[0], &[]).is_err());
    assert!(d.check_step(0, &[8], &[]).is_err());
    assert!(d.check_step(0, &[3, 3], &[]).is_err());
    assert!(d.check_step(0, &[3, 1], &[]).is_err());
    assert!(d.check_step(0, &[1, 3], &[1]).is_err());
    d.check_step(0, &[1, 3], &[2, 1]).unwrap();
    assert!(Directory::new(3, 2, 8, 0).is_err());
    assert!(Directory::new(1, 2, 0, 0).is_err());
}

/// Two classes plan with class-local layer ids in their own directories.
#[test]
fn two_classes_keep_separate_directories() {
    let f = fixture("expert-classes.json");
    let mut classes = vec![None; 42];
    for &il in &ids(&f["q6k_down_layers"]) {
        classes[il] = Some(ExpertClass::Q6kDown);
    }
    for &il in &ids(&f["q4k_down_layers"]) {
        classes[il] = Some(ExpertClass::Q4kDown);
    }
    let l = ClassLayout::new(&classes).unwrap();
    let mut dirs = [
        Directory::new(1, l.layers[0].len(), 512, 64).unwrap(),
        Directory::new(1, l.layers[1].len(), 512, 32).unwrap(),
    ];
    assert_eq!(l.of_layer.len(), 42);
    for (il, at) in l.of_layer.iter().enumerate().skip(1) {
        let (ci, local) = at.unwrap();
        let experts = [il as u32, 100 + il as u32, 511];
        dirs[ci].check_step(local, &experts, &[]).unwrap();
        let p = dirs[ci].plan(local, &experts, &[], true);
        dirs[ci].untag(local, &p);
        dirs[ci].publish(local, &p);
        dirs[ci].complete(p.epoch);
        for (&e, &s) in experts.iter().zip(&p.slots) {
            assert!(dirs[ci].holds(local, e, s), "layer {il} expert {e}");
        }
    }
    assert_eq!(dirs[0].cache().occupied(), 19 * 3);
    assert_eq!(dirs[1].cache().occupied(), 32);
    for d in &dirs {
        d.cache().check().unwrap();
    }
}

/// `Read::repack` hands each extent's bytes (unaligned offsets, the last one
/// ending at the file end) to its callback, on the reader threads.
#[test]
fn repack_sink_reads_unaligned_extents() {
    let dir = Temp::new();
    let data = random_bytes(3 * 16384 + 777, 3);
    let path = dir.file(&data);
    let pool = pool_over(&path, 3);
    let spans: [(u64, u64); 4] = [
        (0, 5),
        (16380, 9000),
        (2 * 16384 + 1, 16384),
        (data.len() as u64 - 4000, 4000),
    ];
    let got: Arc<Mutex<Vec<Option<Vec<u8>>>>> = Arc::new(Mutex::new(vec![None; spans.len()]));
    let reads = spans
        .iter()
        .enumerate()
        .map(|(i, &(offset, len))| {
            let got = Arc::clone(&got);
            Read::repack(
                Extent {
                    shard: 0,
                    offset,
                    len,
                },
                Arc::new(move |b: &[u8]| -> Result<()> {
                    lock(&got)[i] = Some(b.to_vec());
                    Ok(())
                }),
            )
        })
        .collect();
    // SAFETY: the callbacks write only `got`, under its lock.
    let io = unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(io.reads, spans.len());
    assert_eq!(io.bytes, spans.iter().map(|s| s.1).sum::<u64>());
    for (i, &(offset, len)) in spans.iter().enumerate() {
        let (o, l) = (offset as usize, len as usize);
        assert_eq!(
            lock(&got)[i].as_deref(),
            Some(&data[o..o + l]),
            "extent {i}"
        );
    }
}

/// A callback error or panic fails the ticket; a short read never calls the
/// callback; an extent outside the file is refused at submit.
#[test]
fn repack_sink_failures_fail_the_ticket() {
    let dir = Temp::new();
    let data = random_bytes(4 * 16384, 5);
    let path = dir.file(&data);
    let pool = pool_over(&path, 2);
    let ext = |offset: u64, len: u64| Extent {
        shard: 0,
        offset,
        len,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let counting = |c: &Arc<AtomicUsize>| -> RepackFnArc {
        let c = Arc::clone(c);
        Arc::new(move |_: &[u8]| -> Result<()> {
            c.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    };
    let fail: RepackFnArc =
        Arc::new(|_: &[u8]| -> Result<()> { Err(Error::from_reason("repack refused")) });
    let reads = vec![
        Read::repack(ext(0, 100), fail),
        Read::repack(ext(100, 100), counting(&calls)),
    ];
    // SAFETY: the callbacks write nothing but an atomic.
    let err = unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(err.reason.contains("repack refused"), "{}", err.reason);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let boom: RepackFnArc = Arc::new(|_: &[u8]| -> Result<()> { panic!("repack panics") });
    // SAFETY: as above.
    let err = unsafe { pool.submit(vec![Read::repack(ext(0, 10), boom)], Priority::Low) }
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(err.reason.contains("panicked"), "{}", err.reason);
    // SAFETY: as above.
    let refused = unsafe {
        pool.submit(
            vec![Read::repack(
                ext(data.len() as u64 - 10, 11),
                counting(&calls),
            )],
            Priority::High,
        )
    };
    assert!(refused.is_err());
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(2 * 16384)
        .unwrap();
    let before = calls.load(Ordering::Relaxed);
    // SAFETY: as above.
    let err = unsafe {
        pool.submit(
            vec![Read::repack(ext(2 * 16384 - 50, 200), counting(&calls))],
            Priority::High,
        )
    }
    .unwrap()
    .wait()
    .unwrap_err();
    assert!(err.reason.contains("short read"), "{}", err.reason);
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

type RepackFnArc = crate::moe_stream::io::RepackFn;

/// Slot writes (direct and through the IO pool) == the same experts of a
/// whole-tensor `KQuantRepacker`, Q4_K and Q6_K, into slots in another order.
#[test]
fn slot_repack_equals_whole_tensor_repacker() {
    let (n, rows, k) = (5usize, 64usize, 512usize);
    for (i, ty) in [GgufTensorType::Q4K, GgufTensorType::Q6K]
        .into_iter()
        .enumerate()
    {
        let format = ty.k_quant_format().unwrap();
        let geo = ProjGeo { format, rows, k };
        let per = geo.file_bytes() as usize;
        let at = 1000 + 7 * i;
        let mut file = random_bytes(at, 1);
        let payload = random_bytes(n * per, 11 + i as u64);
        file.extend_from_slice(&payload);
        let whole = repack_kquant(format, &payload, n * rows, k).unwrap();
        let loc = Arc::new(located(
            "blk.1.ffn_down_exps.weight",
            [k, rows, n],
            ty,
            at as u64,
        ));
        let picks = [4u32, 0, 2];
        let mut direct = Host::new(geo, picks.len());
        let d = direct.dst();
        for (slot, &e) in picks.iter().enumerate() {
            let e = e as usize;
            d.write(&loc.info, slot, &payload[e * per..(e + 1) * per])
                .unwrap();
        }
        assert!(d.write(&loc.info, 3, &payload[..per]).is_err());
        assert!(d.write(&loc.info, 0, &payload[..per - 1]).is_err());
        let dir = Temp::new();
        let path = dir.file(&file);
        let pool = pool_over(&path, 3);
        let mut pooled = Host::new(geo, picks.len());
        let p = pooled.dst();
        let reads = picks
            .iter()
            .enumerate()
            .map(|(slot, &e)| p.read(&loc, e, slot).unwrap())
            .collect();
        // SAFETY: distinct slots of `pooled`, alive until the wait returns.
        unsafe { pool.submit(reads, Priority::High) }
            .unwrap()
            .wait()
            .unwrap();
        assert!(p.read(&loc, 5, 0).is_err());
        assert!(p.read(&loc, 0, 3).is_err());
        for (slot, &e) in picks.iter().enumerate() {
            let want = expert_of(&whole, geo, e as usize);
            assert_eq!(direct.slot(slot), want, "{ty:?} direct slot {slot}");
            assert_eq!(pooled.slot(slot), want, "{ty:?} pooled slot {slot}");
        }
    }
}

/// A file of 4 layers (0 dense, 1 and 3 class A, 2 class B), 6 experts of
/// 256 x 256, every tensor at an unaligned offset.
fn bank_fixture() -> (Temp, PathBuf, MoeConfig, Vec<Option<ExpertLayer>>, Vec<u8>) {
    let m = moe(6, 256, 256);
    let mut experts = vec![None];
    let mut at = 1000u64;
    for (il, class) in [
        (1, ExpertClass::Q6kDown),
        (2, ExpertClass::Q4kDown),
        (3, ExpertClass::Q6kDown),
    ] {
        let (l, end) = layer(&m, il, class, at);
        experts.push(Some(l));
        at = end + 3;
    }
    let file = random_bytes(at as usize, 9);
    let dir = Temp::new();
    let path = dir.file(&file);
    (dir, path, m, experts, file)
}

fn bank_slot(p: &ProjArrays, geo: ProjGeo, slot: u32) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
    let s = slot as usize;
    let r = |cols: usize| s * geo.rows * cols..(s + 1) * geo.rows * cols;
    let w = p.w.to_uint32().unwrap().to_vec();
    let sc = if geo.format.scales_are_signed() {
        p.s.to_int8()
            .unwrap()
            .into_iter()
            .map(|x| x as u8)
            .collect()
    } else {
        p.s.to_uint8().unwrap()
    };
    let b = p.b.to_uint16_native().unwrap();
    (
        w[r(geo.weight_cols())].to_vec(),
        sc[r(geo.scales_cols())].to_vec(),
        b[r(geo.biases_cols())].to_vec(),
    )
}

fn check_prepared(file: &[u8], m: &MoeConfig, l: &ExpertLayer, p: &Prepared) {
    let geo = geometry(m, l.class).unwrap();
    for (pi, loc) in [&l.gate, &l.up, &l.down].iter().enumerate() {
        let g = geo[pi];
        let per = g.file_bytes() as usize;
        for (&e, &slot) in p.experts.iter().zip(&p.slots) {
            let at = loc.data_offset as usize + e as usize * per;
            let want = repack_kquant(g.format, &file[at..at + per], g.rows, g.k).unwrap();
            assert_eq!(
                bank_slot(&p.arrays[pi], g, slot),
                expert_of(&want, g, 0),
                "layer {} proj {pi} expert {e} slot {slot}",
                p.layer
            );
        }
    }
}

/// Metal banks: decode victims, hits, an eviction, a prefill seed and
/// staging, turn end / begin, and a rebuild refused while an array holds
/// the old bank.
#[test]
fn bank_serves_both_classes_across_turns() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    assert_eq!(bank.class_of(0), None);
    assert_eq!(bank.class_of(2), Some(ExpertClass::Q4kDown));
    assert_eq!(bank.cache_slots(), [2, 1]);
    let l1 = experts[1].as_ref().unwrap();
    let p = bank.prepare(1, &[0, 3], &[], true).unwrap();
    assert_eq!(p.class, ExpertClass::Q6kDown);
    bank.assert_slots(&p);
    check_prepared(&file, &m, l1, &p);
    bank.done(&p);
    assert_eq!(bank.stats()[0].slots, 2);
    let h = bank.prepare(1, &[0, 3], &[], true).unwrap();
    assert_eq!(h.slots, p.slots);
    assert_eq!(bank.stats()[0].hits, 2);
    bank.done(&h);
    let e = bank.prepare(3, &[5], &[], true).unwrap();
    bank.assert_slots(&e);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &e);
    bank.done(&e);
    let b = bank.prepare(2, &[1, 2, 5], &[1, 4, 1], false).unwrap();
    assert_eq!(b.class, ExpertClass::Q4kDown);
    bank.assert_slots(&b);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &b);
    bank.done(&b);
    assert_eq!(bank.stats()[1].slots, 3);
    assert!(bank.prepare(0, &[1], &[], true).is_err());
    assert!(bank.prepare(1, &[3, 1], &[], true).is_err());
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());
    bank.begin_turn().unwrap();
    let again = bank.prepare(1, &[3], &[], true).unwrap();
    bank.assert_slots(&again);
    check_prepared(&file, &m, l1, &again);
    bank.done(&again);
    bank.end_turn().unwrap();
    assert!(bank.rebuild([0, 0], 2).is_err());
    drop((p, h, e, b, again));
    bank.rebuild([1, 0], 2).unwrap();
    assert_eq!(bank.staging_layers(), 2);
    assert_eq!(bank.cache_slots(), [1, 0]);
    let r = bank.prepare(3, &[0, 5], &[], false).unwrap();
    bank.assert_slots(&r);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &r);
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

/// Streams a whole expert tensor in 32-expert reads through `f(row0,
/// rows, bytes)`.
fn stream_tensor(
    shards: &ShardSet,
    loc: &Located,
    g: ProjGeo,
    n: usize,
    mut f: impl FnMut(usize, usize, &[u8]),
) {
    let per = g.file_bytes();
    let mut buf = AlignedBuf::new();
    for e0 in (0..n).step_by(32) {
        let k = 32.min(n - e0);
        let ext = Extent {
            shard: loc.shard,
            offset: loc.abs_offset().unwrap() + e0 as u64 * per,
            len: k as u64 * per,
        };
        f(
            e0 * g.rows,
            k * g.rows,
            shards.read_aligned(ext, &mut buf).unwrap(),
        );
    }
}

/// G-9: real Q4_K and Q6_K experts read through `Read::repack` into slots
/// == the same experts of a whole-tensor `KQuantRepacker` (CPU only).
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_expert_repack_equals_whole_tensor_repacker() {
    let scan = real_scan();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let pool = IoPool::new(Arc::clone(&shards), 8).unwrap();
    let n = scan.cfg.moe.n_experts;
    for (il, pi) in [(1usize, 0usize), (1, 2), (5, 2)] {
        let l = scan.experts[il].as_ref().unwrap();
        let loc = Arc::new([&l.gate, &l.up, &l.down][pi].clone());
        let g = geometry(&scan.cfg.moe, l.class).unwrap()[pi];
        let mut rp = KQuantRepacker::new(g.format, g.k, n * g.rows).unwrap();
        stream_tensor(&shards, &loc, g, n, |_, rows, b| {
            rp.push_rows(b, rows).unwrap()
        });
        let whole = rp.finish();
        let picks = [0u32, 1, 255, 511];
        let mut host = Host::new(g, picks.len());
        let dst = host.dst();
        let reads = picks
            .iter()
            .enumerate()
            .map(|(i, &e)| dst.read(&loc, e, picks.len() - 1 - i).unwrap())
            .collect();
        // SAFETY: distinct slots of `host`, alive until the wait returns.
        unsafe { pool.submit(reads, Priority::High) }
            .unwrap()
            .wait()
            .unwrap();
        for (i, &e) in picks.iter().enumerate() {
            assert_eq!(
                host.slot(picks.len() - 1 - i),
                expert_of(&whole, g, e as usize),
                "layer {il} proj {pi} expert {e}"
            );
        }
    }
}

/// Bank-slot `gather_qmm` == `gather_qmm` on the whole tensor repacked by
/// the loader (`RepackDst::alloc` + `cpu_kquant_band_into`), f32 input,
/// bitwise.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_slot_gather_equals_whole_tensor_gather() {
    let scan = real_scan();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let m = &scan.cfg.moe;
    let pool = IoPool::new(Arc::clone(&shards), 8).unwrap();
    let bank = KqBank::new(pool, m, scan.trunk_experts(), [0, 0], 1).unwrap();
    let experts = [0u32, 7, 511];
    for il in [1usize, 5] {
        let l = scan.experts[il].as_ref().unwrap();
        let p = bank.prepare(il, &experts, &[], false).unwrap();
        bank.assert_slots(&p);
        let geo = geometry(m, l.class).unwrap();
        for (pi, loc) in [&l.gate, &l.up, &l.down].into_iter().enumerate() {
            let g = geo[pi];
            let (dst, ptrs) = RepackDst::alloc(&loc.info, OutLayout::RowMajor).unwrap();
            stream_tensor(&shards, loc, g, m.n_experts, |row0, rows, b| {
                cpu_kquant_band_into(&loc.info, b, row0, rows, &ptrs).unwrap()
            });
            let whole = ProjArrays {
                w: dst.weight,
                s: dst.scales.unwrap(),
                b: dst.biases.unwrap(),
                format: g.format,
            };
            let x: Vec<f32> = random_bytes(3 * g.k, 17 + pi as u64)
                .into_iter()
                .map(|v| (f32::from(v) - 127.5) / 128.0)
                .collect();
            let x = MxArray::from_float32(&x, &[3, 1, 1, g.k as i64]).unwrap();
            let row = |v: &[u32]| -> Vec<u32> { (0..3).flat_map(|_| v.iter().copied()).collect() };
            let by_slot = MxArray::from_uint32(&row(&p.slots[..]), &[3, 3]).unwrap();
            let by_id = MxArray::from_uint32(&row(&experts[..]), &[3, 3]).unwrap();
            let got = p.arrays[pi].gather(&x, &by_slot, false).unwrap();
            let want = whole.gather(&x, &by_id, false).unwrap();
            let (got, want) = (
                got.to_float32().unwrap().to_vec(),
                want.to_float32().unwrap().to_vec(),
            );
            assert_eq!(got.len(), 3 * 3 * g.rows);
            assert!(got.iter().all(|v| v.is_finite()), "layer {il} proj {pi}");
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "layer {il} proj {pi}"
            );
        }
        bank.done(&p);
    }
    bank.end_turn().unwrap();
}
