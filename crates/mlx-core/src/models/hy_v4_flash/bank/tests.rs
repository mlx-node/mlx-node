use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use super::*;
use crate::array::alloc::SendPtr;
use crate::models::hy_v4_flash::scan::Scan;
use crate::moe_stream::io::{AlignedBuf, ShardSet};
use crate::utils::gguf_cpu_repack::cpu_kquant_band_into;
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

/// A cache slot whose step has not settled is never a victim: the next
/// layer's misses go to staging until that step settles.
#[test]
fn an_unsettled_slot_is_never_a_victim() {
    let mut d = Directory::new(1, 2, 8, 2).unwrap();
    let p = d.plan(0, &[1, 2], &[], true);
    assert_eq!(p.victims.len(), 2);
    d.untag(0, &p);
    d.publish(0, &p);
    let q = d.plan(1, &[3, 4], &[], true);
    assert!(q.victims.is_empty() && q.copies.is_empty());
    assert_eq!(q.staging, vec![3, 4]);
    assert!(!q.drain);
    d.untag(1, &q);
    d.publish(1, &q);
    d.complete(p.epoch);
    let r = d.plan(1, &[5, 6], &[], true);
    assert_eq!(r.victims.len(), 2);
    let mut got = r.slots.clone();
    let mut was = p.slots.clone();
    got.sort_unstable();
    was.sort_unstable();
    assert_eq!(got, was, "the settled step's slots, and only those");
    d.untag(1, &r);
    d.publish(1, &r);
    // r is unsettled: a miss of layer 0 finds no victim
    let s = d.plan(0, &[7], &[], true);
    assert!(s.victims.is_empty());
    assert_eq!(s.staging, vec![7]);
    d.cache().check().unwrap();
}

/// A staging slot whose last step has not settled is never refilled
/// without a drain: the plan says so, and says nothing once it settled.
#[test]
fn a_staging_refill_of_an_unsettled_slot_drains() {
    let mut d = Directory::new(1, 2, 8, 0).unwrap();
    let p = d.plan(0, &[2, 5], &[], false);
    assert_eq!(p.staging, vec![2, 5]);
    assert!(!p.drain, "nothing used the slots yet");
    d.untag(0, &p);
    d.publish(0, &p);
    // a hit on a valid staging copy reads nothing and drains nothing
    let h = d.plan(0, &[5], &[], false);
    assert_eq!((h.reads(), h.drain), (0, false));
    let q = d.plan(1, &[2], &[], false);
    assert_eq!(q.staging, vec![2]);
    assert!(q.drain, "slot 2 is still used by an unsettled step");
    d.complete(q.epoch - 1);
    d.untag(1, &q);
    d.publish(1, &q);
    let r = d.plan(0, &[2], &[], false);
    assert!(r.drain, "q itself is unsettled");
    d.complete(r.epoch - 1);
    let s = d.plan(0, &[3], &[], false);
    assert!(!s.drain, "slot 3 was never used");
    d.complete_all();
    let t = d.plan(1, &[5], &[], false);
    assert_eq!(t.staging, vec![5]);
    assert!(!t.drain, "every use of slot 5 settled");
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

/// Slot writes (direct and through the IO pool in row bands: one band, the
/// default bands, uneven bands, one row per band) == the same experts of a
/// whole-tensor `KQuantRepacker`, Q4_K and Q6_K, into slots in another order.
#[test]
fn slot_repack_equals_whole_tensor_repacker() {
    let (n, rows, k) = (5usize, 67usize, 512usize);
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
        let row = format.row_bytes(k);
        assert!(
            d.write_rows(&loc.info, 0, rows - 1, 2, &payload[..2 * row])
                .is_err()
        );
        assert!(d.write_rows(&loc.info, 0, 0, 0, &[]).is_err());
        let dir = Temp::new();
        let path = dir.file(&file);
        let pool = pool_over(&path, 3);
        let row = row as u64;
        for (band, want_bands) in [
            (per as u64, 1),
            (BAND_BYTES, 1),
            (7 * row + 5, 10),
            (row, rows),
            (1, rows),
        ] {
            let mut pooled = Host::new(geo, picks.len());
            let p = pooled.dst();
            let reads: Vec<Read> = picks
                .iter()
                .enumerate()
                .flat_map(|(slot, &e)| p.reads_banded(&loc, e, slot, band).unwrap())
                .collect();
            assert_eq!(reads.len(), picks.len() * want_bands, "band {band}");
            let covered: u64 = reads.iter().map(|r| r.extent.len).sum();
            assert_eq!(covered, (picks.len() * per) as u64);
            // SAFETY: distinct rows of distinct slots of `pooled`, alive until
            // the wait returns.
            unsafe { pool.submit(reads, Priority::High) }
                .unwrap()
                .wait()
                .unwrap();
            for (slot, &e) in picks.iter().enumerate() {
                let want = expert_of(&whole, geo, e as usize);
                assert_eq!(direct.slot(slot), want, "{ty:?} direct slot {slot}");
                assert_eq!(pooled.slot(slot), want, "{ty:?} band {band} slot {slot}");
            }
            assert!(p.reads_banded(&loc, 5, 0, band).is_err());
            assert!(p.reads_banded(&loc, 0, 3, band).is_err());
        }
    }
}

#[test]
fn band_rows_aligns_q4k_1536_to_16k_pages_and_matches_repack() {
    let q4k_row = KQuantFormat::Q4K.row_bytes(1536);
    assert_eq!(q4k_row, 864);
    let step = band_rows(1536, q4k_row, BAND_BYTES);
    assert_eq!(step, 512);
    assert_eq!((step * q4k_row) % 16384, 0);

    // Every 32-byte-aligned GGUF offset yields exact 16 KiB internal boundaries
    // covering 0..1536 with zero gaps or overlaps.
    for m in 0..512u64 {
        let off = m * 32;
        let ranges = band_ranges(off, 1536, q4k_row, BAND_BYTES);
        assert!(!ranges.is_empty());
        let mut cursor = 0usize;
        for (i, &(r0, len)) in ranges.iter().enumerate() {
            assert_eq!(r0, cursor, "gap or overlap at m={m} band {i}");
            assert!(len > 0, "empty band at m={m} band {i}");
            cursor += len;
            if i + 1 < ranges.len() {
                assert_eq!(
                    (off + cursor as u64 * q4k_row as u64) % 16384,
                    0,
                    "internal cut not 16 KiB aligned at m={m} band {i}"
                );
            }
        }
        assert_eq!(cursor, 1536, "incomplete coverage at m={m}");
    }

    // Q6_K K=1536 also covers 0..1536 contiguously with zero gaps or overlaps.
    let q6k_row = KQuantFormat::Q6K.row_bytes(1536);
    assert_eq!(q6k_row, 1260);
    let q6_ranges = band_ranges(14_336, 1536, q6k_row, BAND_BYTES);
    let mut cursor = 0usize;
    for &(r0, len) in &q6_ranges {
        assert_eq!(r0, cursor);
        assert!(len > 0);
        cursor += len;
    }
    assert_eq!(cursor, 1536);

    // Full 1536x1536 Q4_K read at real-fixture offset mod 16384 = 14336 matches
    // whole-tensor repack bit-for-bit, and sort_reads orders by (shard, offset).
    let geo = ProjGeo {
        format: KQuantFormat::Q4K,
        rows: 1536,
        k: 1536,
    };
    let per = geo.file_bytes() as usize;
    let at = 14_336usize;
    let mut file = random_bytes(at, 41);
    let payload = random_bytes(2 * per, 42);
    file.extend_from_slice(&payload);
    let whole = repack_kquant(KQuantFormat::Q4K, &payload, 2 * 1536, 1536).unwrap();
    let loc = Arc::new(located(
        "blk.1.ffn_gate_exps.weight",
        [1536, 1536, 2],
        GgufTensorType::Q4K,
        at as u64,
    ));
    let dir = Temp::new();
    let path = dir.file(&file);
    let pool = pool_over(&path, 4);
    let mut host = Host::new(geo, 2);
    let dst = host.dst();
    let mut reads = dst.reads(&loc, 1, 0).unwrap();
    reads.extend(dst.reads(&loc, 0, 1).unwrap());
    sort_reads(&mut reads);
    assert!(reads.windows(2).all(
        |w| (w[0].extent.shard, w[0].extent.offset) <= (w[1].extent.shard, w[1].extent.offset)
    ));
    // SAFETY: `host` outlives the ticket and bands write disjoint rows/slots.
    unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(host.slot(0), expert_of(&whole, geo, 1));
    assert_eq!(host.slot(1), expert_of(&whole, geo, 0));
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
    bank.settle_all();
    assert_eq!(bank.stats()[0].slots, 2);
    let h = bank.prepare(1, &[0, 3], &[], true).unwrap();
    assert_eq!(h.slots, p.slots);
    assert_eq!(bank.stats()[0].hits, 2);
    bank.settle_all();
    let e = bank.prepare(3, &[5], &[], true).unwrap();
    bank.assert_slots(&e);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &e);
    bank.settle_all();
    let b = bank.prepare(2, &[1, 2, 5], &[1, 4, 1], false).unwrap();
    assert_eq!(b.class, ExpertClass::Q4kDown);
    bank.assert_slots(&b);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &b);
    bank.settle_all();
    assert_eq!(bank.stats()[1].slots, 3);
    assert!(bank.prepare(0, &[1], &[], true).is_err());
    assert!(bank.prepare(1, &[3, 1], &[], true).is_err());
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());
    bank.begin_turn().unwrap();
    let again = bank.prepare(1, &[3], &[], true).unwrap();
    bank.assert_slots(&again);
    check_prepared(&file, &m, l1, &again);
    bank.settle_all();
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

/// Metal banks: a staging refill drains the GPU only while a step that
/// used the slot is unsettled, the drained step reads its own bytes, and
/// settling lets the next refill go without a drain.
#[test]
fn bank_drains_only_for_an_unsettled_staging_slot() {
    use crate::array::memory::TEST_SYNCHRONIZE_CALLS;
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [0, 0], 1).unwrap();
    let syncs = || TEST_SYNCHRONIZE_CALLS.with(std::cell::Cell::get);
    let at = syncs();
    let a = bank.prepare(1, &[0, 3], &[], false).unwrap();
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &a);
    assert_eq!(syncs(), at, "a first fill drains nothing");
    let issued = bank.issued();
    let b = bank.prepare(3, &[0], &[], false).unwrap();
    assert_eq!(syncs(), at + 1, "slot 0 was used by the unsettled step");
    assert_eq!(bank.stats()[0].drains, 1);
    bank.assert_slots(&b);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &b);
    bank.settle(issued);
    let c = bank.prepare(1, &[3], &[], false).unwrap();
    assert_eq!(c.slots, a.slots[1..]);
    assert_eq!(syncs(), at + 1, "a valid staging copy reads nothing");
    bank.settle_all();
    let d = bank.prepare(1, &[0], &[], false).unwrap();
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &d);
    assert_eq!(syncs(), at + 1, "every use of slot 0 settled");
    assert_eq!(bank.stats()[0].drains, 1);
    bank.end_turn().unwrap();
}

#[test]
fn safe_point_syncs_only_when_staged_copies_are_unsettled() {
    use crate::array::memory::TEST_SYNCHRONIZE_CALLS;
    let (_dir, path, m, experts, _file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    let syncs = || TEST_SYNCHRONIZE_CALLS.with(std::cell::Cell::get);

    // Nothing staged yet.
    let at = syncs();
    assert_eq!(bank.safe_point(false).unwrap(), 0);
    assert_eq!(syncs(), at, "empty safe_point must not synchronize");

    // Unsettled decode miss into cache victim (nothing staged).
    let _v = bank.prepare(1, &[0], &[], true).unwrap();
    let at = syncs();
    assert_eq!(bank.safe_point(false).unwrap(), 0);
    assert_eq!(
        syncs(),
        at,
        "unsettled non-staged step must not synchronize"
    );
    bank.settle_all();

    // Prefill class 0 (2 layers, 2 cache slots -> quota 1/layer): expert 1 seeds cache,
    // expert 2 lands in staging. Decode then serves expert 2 from staging.
    let _p1 = bank.prepare(1, &[1, 2], &[2, 1], false).unwrap();
    bank.settle_all();
    let _d1 = bank.prepare(1, &[2], &[], true).unwrap();
    bank.settle_all();
    let at = syncs();
    assert_eq!(bank.safe_point(false).unwrap(), 1);
    assert_eq!(syncs(), at, "settled staged copy must not synchronize");

    // Quota spent for layer 1 this turn: prefill stages expert 4; decode serves it
    // from staging without settling, so safe_point must synchronize before promote.
    let _p2 = bank.prepare(1, &[4], &[], false).unwrap();
    bank.settle_all();
    let _d2 = bank.prepare(1, &[4], &[], true).unwrap();
    let at = syncs();
    assert_eq!(bank.safe_point(false).unwrap(), 1);
    assert_eq!(syncs(), at + 1, "unsettled staged copy must synchronize");

    // Subsequent safe_point with seed_reset still skips sync when nothing is staged.
    let at = syncs();
    assert_eq!(bank.safe_point(true).unwrap(), 0);
    assert_eq!(syncs(), at);

    bank.end_turn().unwrap();
}

#[test]
fn lfu_directory_keeps_frequent_expert_over_recent_one_hit() {
    let mut d = Directory::new_with_lfu(1, 1, 8, 2, 100.0).unwrap();
    let p0 = d.plan(0, &[0], &[], true);
    d.untag(0, &p0);
    d.publish(0, &p0);
    d.complete(p0.epoch);
    for _ in 0..5 {
        let h = d.plan(0, &[0], &[], true);
        assert_eq!(h.hits, 1);
        d.complete(h.epoch);
    }
    // More recently inserted one-hit expert.
    let p1 = d.plan(0, &[1], &[], true);
    d.untag(0, &p1);
    d.publish(0, &p1);
    d.complete(p1.epoch);

    let p2 = d.plan(0, &[2], &[], true);
    d.untag(0, &p2);
    d.publish(0, &p2);
    d.complete(p2.epoch);

    assert!(
        d.cache().lookup(0, 0).is_some(),
        "repeatedly hit expert should stay cached"
    );
    assert!(
        d.cache().lookup(0, 1).is_none(),
        "recent one-hit expert should be evicted first"
    );
    assert!(d.cache().lookup(0, 2).is_some());
    d.cache().check().unwrap();
}

#[test]
fn default_bank_uses_lru_policy() {
    if use_lfu_policy() {
        return;
    }
    let (_dir, path, m, experts, _file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 0], 1).unwrap();
    let _ = bank.prepare(1, &[0], &[], true).unwrap();
    bank.settle_all();
    for _ in 0..5 {
        let _ = bank.prepare(1, &[0], &[], true).unwrap();
        bank.settle_all();
    }
    // Insert a one-hit expert more recently than expert 0.
    let _ = bank.prepare(1, &[1], &[], true).unwrap();
    bank.settle_all();
    // Third distinct expert evicts LRU (expert 0), keeping recent expert 1.
    let _ = bank.prepare(1, &[2], &[], true).unwrap();
    bank.settle_all();
    let g = lock(&bank.inner);
    let cache = g.classes[0].dir.cache();
    assert!(
        cache.lookup(0, 0).is_none(),
        "default LRU policy must evict least-recently-used expert 0"
    );
    assert!(
        cache.lookup(0, 1).is_some(),
        "default LRU policy must keep more recently used expert 1"
    );
    assert!(cache.lookup(0, 2).is_some());
    drop(g);
    bank.end_turn().unwrap();
}

#[test]
fn sorted_slot_bounds_counts_exact_bm32_tiles() {
    assert_eq!(sorted_slot_bounds(&[]), (0, 0));
    assert_eq!(sorted_slot_bounds(&[7]), (1, 8));
    let mut s = vec![3u32; 32];
    s.extend(std::iter::repeat_n(10u32, 33));
    s.push(200);
    // 32 -> 1 tile, 33 -> 2 tiles, 1 -> 1 tile => 4 tiles; max_slot + 1 = 201.
    assert_eq!(sorted_slot_bounds(&s), (4, 201));
}

#[test]
fn split_step_publish_abandon_and_drop_on_error() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [4, 2], 1).unwrap();
    assert!(bank.split());
    assert!(bank.prefetch());

    // Pre-fill expert 1 into cache.
    let warm = bank.prepare(1, &[1], &[], true).unwrap();
    bank.assert_slots(&warm);
    bank.settle_all();

    // Mixed hit (1) + miss (3) in prepare_split.
    let (p, pending) = bank.prepare_split(1, &[1, 3], &[], true).unwrap();
    assert!(!p.is_reading(1));
    assert!(p.is_reading(3));
    assert_eq!(p.reading, vec![3]);
    bank.assert_ready_slots(&p);
    let pending = pending.expect("miss expert 3 must return Pending");
    bank.note_split(1, true);
    pending.finish().unwrap();
    bank.assert_slots(&p);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &p);
    bank.settle_all();
    assert_eq!(bank.stats()[0].split_steps, 1);
    assert_eq!(bank.stats()[0].split_mixed, 1);
    assert!(bank.stats()[0].wait_ns > 0);
    bank.test_check_caches();

    // Dropping Pending without finish waits for reads and releases victims.
    let occ_before = lock(&bank.inner).classes[0].dir.cache().occupied();
    let (p_drop, pending_drop) = bank.prepare_split(1, &[4, 5], &[], true).unwrap();
    assert_eq!(p_drop.reading, vec![4, 5]);
    assert!(pending_drop.is_some());
    drop(pending_drop);
    assert_eq!(bank.reads_in_flight(), 0);
    assert_eq!(
        lock(&bank.inner).classes[0].dir.cache().occupied(),
        occ_before
    );
    bank.test_check_caches();
    bank.settle_all();

    // Truncate file to trigger a real read failure on Pending::finish.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(1024)
        .unwrap();
    let (_p_err, pending_err) = bank.prepare_split(1, &[2, 4], &[], true).unwrap();
    assert!(pending_err.unwrap().finish().is_err());
    assert_eq!(
        lock(&bank.inner).classes[0].dir.cache().occupied(),
        occ_before
    );
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn two_class_interleaved_read_ahead() {
    let (_dir, path, m, experts, file) = bank_fixture();
    // 1 staging layer: prefetch is a no-op.
    let single = KqBank::new(pool_over(&path, 4), &m, &experts, [0, 0], 1).unwrap();
    assert_eq!(single.prefetch_layer(1, &[0, 2]).unwrap(), 0);
    single.end_turn().unwrap();

    // 2 staging layers: class 0 has global layers 1 (local 0) and 3 (local 1),
    // class 1 has global layer 2 (local 0).
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [0, 0], 2).unwrap();
    bank.set_prefetch(false);
    assert_eq!(bank.prefetch_layer(1, &[0, 2]).unwrap(), 0);
    bank.set_prefetch(true);

    // Queue prefetch for layer 1 (class 0 local 0) and layer 2 (class 1 local 0).
    assert_eq!(bank.prefetch_layer(1, &[0, 2, 5]).unwrap(), 3);
    assert_eq!(bank.prefetch_layer(2, &[1, 4]).unwrap(), 2);

    // Layer 1 uses [0, 2] (5 unused); settles class 0 local 0 while keeping class 1 local 0 alive.
    let p1 = bank.prepare(1, &[0, 2], &[1, 1], false).unwrap();
    bank.assert_slots(&p1);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &p1);
    bank.end_layer(1);
    bank.settle_all();
    assert_eq!(bank.stats()[0].prefetch_slots, 3);
    assert_eq!(bank.stats()[0].prefetch_unused, 1);

    // Queue prefetch for layer 3 (class 0 local 1) into the other staging half of class 0.
    assert_eq!(bank.prefetch_layer(3, &[0, 3]).unwrap(), 2);

    // Layer 2 uses [1, 4] (all prefetched); must not disturb class 0's in-flight prefetch.
    let slots_before_l2 = bank.stats()[1].slots;
    let p2 = bank.prepare(2, &[1, 4], &[1, 1], false).unwrap();
    bank.assert_slots(&p2);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &p2);
    bank.end_layer(2);
    bank.settle_all();
    // Layer 2's 2 experts came from prefetch (no duplicate reads).
    assert_eq!(bank.stats()[1].slots - slots_before_l2, 2);
    assert_eq!(bank.stats()[1].prefetch_slots, 2);
    assert_eq!(bank.stats()[1].prefetch_unused, 0);

    // Layer 3 uses [0, 3, 4]: 0 and 3 were prefetched, only 4 reads now.
    let p3 = bank.prepare(3, &[0, 3, 4], &[1, 1, 1], false).unwrap();
    bank.assert_slots(&p3);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &p3);
    bank.end_layer(3);
    bank.settle_all();
    assert_eq!(bank.stats()[0].prefetch_slots, 5);
    assert_eq!(bank.stats()[0].prefetch_unused, 1);
    assert_eq!(bank.stats()[0].slots, 6);
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn decode_read_ahead_harvests_hits_and_counts_unused() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [4, 2], 2).unwrap();
    bank.set_decode_prefetch(false);
    assert!(!bank.decode_prefetch());
    bank.set_decode_prefetch(true);
    assert!(bank.decode_prefetch());

    // Direct and single-staging banks must not prefetch.
    let single = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 2], 1).unwrap();
    assert_eq!(single.prefetch_layer(1, &[0, 1]).unwrap(), 0);
    single.end_turn().unwrap();

    // Layer 1 (class 0 local 0): submit Priority::High decode misses and immediately
    // queue Layer 2 (class 1 local 0) at Priority::Low before waiting on Layer 1.
    let (p1, pending1) = bank.prepare_split(1, &[0, 1], &[1, 1], true).unwrap();
    assert_eq!(bank.prefetch_layer(2, &[1, 3, 5]).unwrap(), 3);
    pending1.expect("layer 1 misses").finish().unwrap();
    bank.assert_slots(&p1);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &p1);
    bank.end_layer(1);
    bank.settle_all();
    assert_eq!(bank.stats()[1].prefetch_slots, 3);

    // Layer 2 (class 1 local 0) needs [1, 3, 4]: 1 and 3 hit prefetched staging slots
    // (not in `reading`), 4 falls through to Priority::High, 5 counts as unused.
    let (p2, pending2) = bank.prepare_split(2, &[1, 3, 4], &[1, 1, 1], true).unwrap();
    assert_eq!(p2.reading, vec![4]);
    bank.assert_ready_slots(&p2);
    assert_eq!(bank.prefetch_layer(3, &[0, 2]).unwrap(), 2);
    pending2.expect("mispredicted expert 4").finish().unwrap();
    bank.assert_slots(&p2);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &p2);
    bank.end_layer(2);
    bank.settle_all();
    assert_eq!(bank.stats()[1].prefetch_slots, 3);
    assert_eq!(bank.stats()[1].prefetch_unused, 1);
    assert_eq!(bank.stats()[1].slots, 4);

    // Layer 3 (class 0 local 1) needs [0, 2]: 100% prefetched hits -> zero High reads.
    let (p3, pending3) = bank.prepare_split(3, &[0, 2], &[1, 1], true).unwrap();
    assert!(pending3.is_none());
    assert!(p3.reading.is_empty());
    bank.assert_slots(&p3);
    check_prepared(&file, &m, experts[3].as_ref().unwrap(), &p3);
    bank.end_layer(3);
    bank.settle_all();
    assert_eq!(bank.stats()[0].prefetch_slots, 2);
    assert_eq!(bank.stats()[0].prefetch_unused, 0);
    assert_eq!(bank.stats()[0].drains, 0);
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn prefill_subgroups_last_use_order_and_drop_on_error() {
    let (_dir, path, m, experts, file) = bank_fixture();
    // Class 0 has 2 layers (1, 3), 4 cache slots -> quota 2/layer.
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [4, 0], 1).unwrap();

    // Warm expert 1 in cache so group_0 is non-empty.
    let warm = bank.prepare(1, &[1], &[], true).unwrap();
    bank.assert_slots(&warm);
    bank.settle_all();

    let used = [0u32, 1, 2, 3, 4, 5];
    let uses = [1u32, 1, 5, 4, 2, 3];
    let last_use = [5u32, 1, 3, 0, 4, 2];
    let (p, group_0, mut sg) = bank
        .plan_prefill_subgroups(1, &used, &uses, &last_use, 2)
        .unwrap();
    assert_eq!(group_0, vec![1]);
    bank.assert_ready_slots(&p);

    let mut got_order = Vec::new();
    let mut batches = Vec::new();
    while let Some(batch) = sg.next_subgroup().unwrap() {
        bank.assert_experts_slots(&p, &batch);
        batches.push(batch.clone());
        got_order.extend(batch);
    }
    assert_eq!(got_order, vec![5, 3, 0, 4, 2]);
    assert_eq!(batches, vec![vec![5, 3], vec![0, 4], vec![2]]);
    bank.assert_slots(&p);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &p);
    bank.settle_all();
    bank.test_check_caches();

    // Early drop on layer 3 (quota 2 victims across subgroups): dropping before
    // draining must wait for active IO and release every uncommitted victim.
    let occ_before = lock(&bank.inner).classes[0].dir.cache().occupied();
    {
        let (_p2, g0, _sg_drop) = bank
            .plan_prefill_subgroups(
                3,
                &[0, 1, 2, 3, 4, 5],
                &[6, 5, 4, 3, 2, 1],
                &[5, 4, 3, 2, 1, 0],
                2,
            )
            .unwrap();
        assert!(g0.is_empty());
    }
    assert_eq!(bank.reads_in_flight(), 0);
    assert_eq!(
        lock(&bank.inner).classes[0].dir.cache().occupied(),
        occ_before
    );
    bank.test_check_caches();
    bank.settle_all();

    // Truncated file triggers an IO error inside next_subgroup and releases all victims.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(1024)
        .unwrap();
    let (_p3, _g0, mut sg_err) = bank
        .plan_prefill_subgroups(3, &[0, 1, 2, 3], &[4, 3, 2, 1], &[3, 2, 1, 0], 2)
        .unwrap();
    assert!(sg_err.next_subgroup().is_err());
    drop(sg_err);
    assert_eq!(
        lock(&bank.inner).classes[0].dir.cache().occupied(),
        occ_before
    );
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn direct_bank_background_fill_matches_streamed_prepare_bit_for_bit() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let streamed = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    let direct = KqBank::new_direct(pool_over(&path, 4), &m, &experts).unwrap();
    assert!(!streamed.is_direct());
    assert!(direct.is_direct());
    assert_eq!(direct.staging_layers(), 0);
    assert_eq!(direct.cache_slots(), [2 * m.n_experts, m.n_experts]);
    let [sa, sb] = direct.slot_bytes();
    assert_eq!(
        direct.nbytes(),
        (2 * m.n_experts as u64) * sa + (m.n_experts as u64) * sb
    );
    assert!(streamed.direct_layer(1).unwrap().is_none());
    assert!(direct.direct_layer(0).is_err());
    assert!(direct.rebuild([0, 0], 1).is_err());

    let all_experts: Vec<u32> = (0..m.n_experts as u32).collect();
    for il in [1usize, 2, 3] {
        let l = experts[il].as_ref().unwrap();
        let geo = geometry(&m, l.class).unwrap();
        let dir_arrays = direct.direct_layer(il).unwrap().expect("direct arrays");
        let sp = streamed.prepare(il, &all_experts, &[], false).unwrap();
        for pi in 0..3 {
            for (&e, &s) in sp.experts.iter().zip(&sp.slots) {
                assert_eq!(
                    bank_slot(&dir_arrays[pi], geo[pi], e),
                    bank_slot(&sp.arrays[pi], geo[pi], s),
                    "layer {il} proj {pi} expert {e}"
                );
            }
        }
        streamed.settle_all();
    }
    direct.wait_direct_fill().unwrap();
    assert!(direct.direct_fill_complete());

    // prepare / prepare_split / plan_prefill_subgroups work with identity mapping and 0 misses.
    let (ps, pending) = direct
        .prepare_split(1, &[0, 3, 5], &[1, 2, 1], true)
        .unwrap();
    assert!(pending.is_none());
    assert!(ps.reading.is_empty());
    assert_eq!(ps.slots, vec![0, 3, 5]);
    direct.assert_slots(&ps);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &ps);

    let (pg, g0, mut sg) = direct
        .plan_prefill_subgroups(2, &[1, 2, 4], &[1, 1, 1], &[4, 2, 1], 2)
        .unwrap();
    assert_eq!(g0, vec![1, 2, 4]);
    assert_eq!(pg.slots, vec![1, 2, 4]);
    assert!(sg.next_subgroup().unwrap().is_none());
    direct.assert_slots(&pg);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &pg);

    let st = direct.stats();
    assert!(st[0].hits > 0 && st[0].hits == st[0].uses);
    assert!(st[1].hits > 0 && st[1].hits == st[1].uses);
    assert_eq!(st[0].slots, 0);
    assert_eq!(st[1].slots, 0);

    // Direct banks remain NonVolatile across end_turn / begin_turn.
    assert_eq!(direct.safe_point(true).unwrap(), 0);
    direct.end_turn().unwrap();
    assert!(!direct.in_turn());
    let res = direct.residency().unwrap();
    assert!(res.resident > 0);
    assert_eq!(res.volatile, 0, "direct banks must never flip to Volatile");
    assert!(!direct.begin_turn().unwrap());
    assert!(direct.in_turn());
    streamed.end_turn().unwrap();
}

#[test]
fn direct_bank_drop_mid_fill_and_io_error_propagate_cleanly() {
    let (_dir, path, m, experts, _file) = bank_fixture();
    // Slow reads so Drop cancels and joins the background fill thread mid-fill.
    let shards = Arc::new(ShardSet::open(&[path.as_path()]).unwrap());
    shards.delay_ms.store(15, Ordering::Relaxed);
    let pool = IoPool::new(Arc::clone(&shards), 2).unwrap();
    let bank = KqBank::new_direct(pool, &m, &experts).unwrap();
    assert!(!bank.direct_fill_complete());
    drop(bank);

    // Truncated file fails background fill and wakes waiters with an error.
    let shards = Arc::new(ShardSet::open(&[path.as_path()]).unwrap());
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(1024)
        .unwrap();
    let pool = IoPool::new(shards, 2).unwrap();
    let bank = KqBank::new_direct(pool, &m, &experts).unwrap();
    assert!(bank.direct_layer(1).is_err());
    assert!(bank.wait_direct_fill().is_err());
}

#[test]
fn direct_bank_late_layer_read_error_propagates_through_wait_direct_fill() {
    let (_dir, path, m, experts, _file) = bank_fixture();
    // Truncate only inside the last MoE layer (layer 3) so earlier layers fill cleanly.
    let layer3_start = experts[3].as_ref().unwrap().gate.data_offset;
    assert!(layer3_start > 0);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(layer3_start + 64)
        .unwrap();
    let shards = Arc::new(ShardSet::open(&[path.as_path()]).unwrap());
    let pool = IoPool::new(shards, 2).unwrap();
    let bank = KqBank::new_direct(pool, &m, &experts).unwrap();
    assert!(bank.direct_layer(1).unwrap().is_some());
    assert!(bank.direct_layer(2).unwrap().is_some());
    assert!(bank.wait_direct_fill().is_err());
    assert!(bank.direct_layer(3).is_err());
}

#[test]
fn partial_bank_promotes_to_direct_when_idle_and_serves_identical_bytes() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    let reference = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    assert!(!bank.is_direct());

    // Serve a streamed turn first.
    let warm = bank.prepare(1, &[0, 3], &[], true).unwrap();
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &warm);
    bank.settle_all();

    // Promotion while in_turn == true must be refused without altering state.
    assert!(bank.promote_to_direct().is_err());
    assert!(!bank.is_direct());
    assert!(bank.in_turn());

    // Release Prepared handle before idle promotion so old banks can free cleanly.
    drop(warm);
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());

    bank.promote_to_direct().unwrap();
    assert!(bank.is_direct());
    assert!(!bank.in_turn());
    assert_eq!(bank.staging_layers(), 0);
    assert_eq!(bank.cache_slots(), [2 * m.n_experts, m.n_experts]);
    bank.wait_direct_fill().unwrap();
    assert!(bank.direct_fill_complete());
    // Promoting an already-direct bank is an idempotent no-op.
    bank.promote_to_direct().unwrap();

    // Every layer and expert matches the streamed reference bit-for-bit.
    let all_experts: Vec<u32> = (0..m.n_experts as u32).collect();
    for il in [1usize, 2, 3] {
        let l = experts[il].as_ref().unwrap();
        let geo = geometry(&m, l.class).unwrap();
        let dir_arrays = bank.direct_layer(il).unwrap().expect("direct arrays");
        let sp = reference.prepare(il, &all_experts, &[], false).unwrap();
        for pi in 0..3 {
            for (&e, &s) in sp.experts.iter().zip(&sp.slots) {
                assert_eq!(
                    bank_slot(&dir_arrays[pi], geo[pi], e),
                    bank_slot(&sp.arrays[pi], geo[pi], s),
                    "promoted layer {il} proj {pi} expert {e}"
                );
            }
        }
        reference.settle_all();
    }

    // Next turn works in direct mode and never flips volatile at turn end.
    assert!(!bank.begin_turn().unwrap());
    assert!(bank.in_turn());
    let p = bank.prepare(2, &[1, 4, 5], &[], true).unwrap();
    assert_eq!(p.slots, vec![1, 4, 5]);
    bank.assert_slots(&p);
    check_prepared(&file, &m, experts[2].as_ref().unwrap(), &p);
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());
    let res = bank.residency().unwrap();
    assert!(res.resident > 0);
    assert_eq!(res.volatile, 0);
    reference.end_turn().unwrap();
}

#[test]
fn promote_to_direct_restores_partial_bank_on_alloc_failure() {
    let (_dir, path, m, experts, file) = bank_fixture();
    let bank = KqBank::new(pool_over(&path, 4), &m, &experts, [2, 1], 1).unwrap();
    let old_nbytes = bank.nbytes();
    let old_slots = bank.cache_slots();
    bank.end_turn().unwrap();

    // Inject allocation failure on the second direct MoE layer.
    TEST_FAIL_DIRECT_ALLOC_AT.with(|c| c.set(Some(1)));
    let err = bank.promote_to_direct();
    TEST_FAIL_DIRECT_ALLOC_AT.with(|c| c.set(None));
    assert!(err.is_err());
    assert!(!bank.is_direct());
    assert!(!bank.in_turn());
    assert_eq!(bank.cache_slots(), old_slots);
    assert_eq!(bank.nbytes(), old_nbytes);

    // Restored partial bank serves bit-exact experts on the next turn.
    bank.begin_turn().unwrap();
    let p = bank.prepare(1, &[2, 5], &[], true).unwrap();
    bank.assert_slots(&p);
    check_prepared(&file, &m, experts[1].as_ref().unwrap(), &p);
    bank.settle_all();
    bank.end_turn().unwrap();
}

/// Aligned version of `bank_fixture` (32-byte aligned offsets like GGUF).
fn aligned_bank_fixture() -> (Temp, PathBuf, MoeConfig, Vec<Option<ExpertLayer>>, Vec<u8>) {
    let m = moe(6, 256, 256);
    let mut experts = vec![None];
    let mut at = 1024u64;
    for (il, class) in [
        (1, ExpertClass::Q6kDown),
        (2, ExpertClass::Q4kDown),
        (3, ExpertClass::Q6kDown),
    ] {
        let (l, end) = layer(&m, il, class, at);
        experts.push(Some(l));
        at = end.div_ceil(32) * 32;
    }
    let file = random_bytes(at as usize, 19);
    let dir = Temp::new();
    let path = dir.file(&file);
    (dir, path, m, experts, file)
}

#[test]
fn mmap_bank_maps_views_reports_zero_anon_and_toggles_in_turn() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let (_dir, path, m, experts, file) = aligned_bank_fixture();
    let expected_mapped: u64 = experts
        .iter()
        .flatten()
        .map(|l| file_bytes(&m, l.class).unwrap() * m.n_experts as u64)
        .sum();
    let bank = KqBank::new_mmap(
        pool_over(&path, 2),
        &m,
        &experts,
        std::slice::from_ref(&path),
        file.len() as u64,
    )
    .unwrap();

    assert!(bank.is_direct());
    assert!(bank.is_mmap());
    assert_eq!(bank.nbytes(), 0);
    assert_eq!(bank.resident_bytes().unwrap(), 0);
    assert_eq!(
        bank.residency().unwrap(),
        Residency {
            resident: 0,
            volatile: 0,
        }
    );
    assert_eq!(bank.mapped_bytes(), expected_mapped);
    assert_eq!(bank.staging_layers(), 0);
    assert_eq!(bank.cache_slots(), [2 * m.n_experts, m.n_experts]);
    assert!(bank.direct_fill_complete());
    bank.wait_direct_fill().unwrap();
    assert!(bank.direct_layer_arrays(0).is_err());
    assert!(bank.direct_layer(1).is_err());
    assert!(bank.rebuild([0, 0], 1).is_err());
    assert!(bank.prepare(1, &[0], &[], true).is_err());
    bank.promote_to_direct().unwrap();
    assert!(bank.is_mmap());

    assert!(!bank.in_turn());
    assert!(!bank.begin_turn().unwrap());
    assert!(bank.in_turn());
    bank.set_in_turn(false);
    assert!(!bank.in_turn());
    bank.set_in_turn(true);
    assert!(bank.in_turn());
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());

    for il in [1usize, 2, 3] {
        let l = experts[il].as_ref().unwrap();
        let geo = geometry(&m, l.class).unwrap();
        let Some(DirectLayerArrays::Mmap(projs)) = bank.direct_layer_arrays(il).unwrap() else {
            panic!("expected Mmap layer arrays for layer {il}");
        };
        for (p, g) in projs.iter().zip(&geo) {
            assert_eq!((p.format, p.rows, p.k), (g.format, g.rows, g.k));
            assert_eq!(p.buf.dtype().unwrap(), DType::Uint8);
            assert_eq!(p.buf.ndim().unwrap(), 2);
        }
    }
    bank.record_direct_use(2, 3);
    let st = bank.stats();
    assert_eq!(st[0].uses, 2 * m.n_used);
    assert_eq!(st[0].hits, st[0].uses);
    assert_eq!(st[1].uses, m.n_used + 3);
    assert_eq!(st[1].hits, st[1].uses);
    assert_eq!(st[0].slots, 0);
    assert_eq!(st[1].slots, 0);
    drop(bank);
}

#[test]
fn mmap_bank_rejects_misaligned_or_truncated_shard_cleanly() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let (_dir, path, m, mut experts, _file) = aligned_bank_fixture();
    // Odd offset violates 4-byte Q4_K alignment.
    experts[1].as_mut().unwrap().gate.data_offset += 1;
    let err = KqBank::new_mmap(
        pool_over(&path, 2),
        &m,
        &experts,
        std::slice::from_ref(&path),
        0,
    );
    assert!(err.is_err());

    // Truncated file fails cleanly without leaking mappings.
    let (_dir2, path2, m2, experts2, _file2) = aligned_bank_fixture();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path2)
        .unwrap()
        .set_len(2048)
        .unwrap();
    let err = KqBank::new_mmap(
        pool_over(&path2, 2),
        &m2,
        &experts2,
        std::slice::from_ref(&path2),
        0,
    );
    assert!(err.is_err());
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
            .flat_map(|(i, &e)| dst.reads(&loc, e, picks.len() - 1 - i).unwrap())
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
            let got = p.arrays[pi].gather(&x, &by_slot).unwrap();
            let want = whole.gather(&x, &by_id).unwrap();
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
        bank.settle_all();
    }
    bank.end_turn().unwrap();
}
