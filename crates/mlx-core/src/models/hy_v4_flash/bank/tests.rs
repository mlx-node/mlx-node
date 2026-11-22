use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::array::alloc::SendPtr;
use crate::models::hy_v4_flash::test_support::{
    Rng, Temp, fixture, j_u64, j_usizes, kquant_layer, located, read_f32, real_scan,
};
use crate::moe_stream::io::{AlignedBuf, ShardSet};
use crate::utils::gguf_cpu_repack::cpu_kquant_band_into;
use crate::utils::gguf_gpu::RepackDst;
use crate::utils::gguf_kquant::{KQuantArrays, KQuantRepacker, KQuantScales, repack_kquant};

type RepackFnArc = crate::moe_stream::io::RepackFn;

const CLASSES_4: [Option<ExpertClass>; 4] = [
    None,
    Some(ExpertClass::Q6kDown),
    Some(ExpertClass::Q4kDown),
    Some(ExpertClass::Q6kDown),
];

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

fn pool_over(path: &Path, threads: usize) -> IoPool {
    IoPool::new(Arc::new(ShardSet::open(&[path]).unwrap()), threads).unwrap()
}

fn truncate_file(path: &Path, len: u64) {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(len)
        .unwrap();
}

fn syncs() -> usize {
    crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(std::cell::Cell::get) as usize
}

fn occupied(bank: &KqBank, ci: usize) -> usize {
    lock(&bank.inner).classes[ci].dir.cache().occupied()
}

fn dir_commit(d: &mut Directory, layer: usize, p: &Plan, settle: bool) {
    d.untag(layer, p);
    d.publish(layer, p);
    if settle {
        d.complete(p.epoch);
    }
}

fn dir_step(d: &mut Directory, layer: usize, exps: &[u32], dec: bool, settle: bool) -> Plan {
    let p = d.plan(layer, exps, &[], dec);
    dir_commit(d, layer, &p, settle);
    p
}

fn layer_meta(m: &MoeConfig, il: usize, class: ExpertClass, mut off: u64) -> (ExpertLayer, u64) {
    let (types, geo) = (class_types(class).unwrap(), geometry(m, class).unwrap());
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let locs: Vec<Located> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .enumerate()
        .map(|(p, name)| {
            let loc = located(&format!("blk.{il}.{name}.weight"), dims[p], types[p], off);
            off += geo[p].file_bytes() * m.n_experts as u64;
            loc
        })
        .collect();
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
        // SAFETY: `self` owns the Vecs and outlives every submit using this SlotDst.
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

fn expert_of(a: &KQuantArrays, g: ProjGeo, e: usize) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
    let r = |cols: usize| e * g.rows * cols..(e + 1) * g.rows * cols;
    (
        a.weight[r(g.weight_cols())].to_vec(),
        scale_bytes(&a.scales)[r(g.scales_cols())].to_vec(),
        a.biases[r(g.biases_cols())].to_vec(),
    )
}

/// Synthetic 4-layer file (0 dense, 1 & 3 class A, 2 class B), 6 experts of 256×256.
struct BankFixture {
    _dir: Temp,
    path: PathBuf,
    moe: MoeConfig,
    experts: Vec<Option<ExpertLayer>>,
    file: Vec<u8>,
}

impl BankFixture {
    fn new(align: usize, seed: u64) -> Self {
        let m = moe(6, 256, 256);
        let mut rng = Rng::new(seed);
        let mut file = rng.bytes(if align > 1 { 1024 } else { 1000 });
        let experts = CLASSES_4
            .iter()
            .enumerate()
            .map(|(il, c)| c.map(|c| kquant_layer(&mut rng, &m, il, c, align, &mut file)))
            .collect();
        let dir = Temp::new("bank");
        let path = dir.write("data.bin", &file);
        Self {
            _dir: dir,
            path,
            moe: m,
            experts,
            file,
        }
    }

    fn unaligned() -> Self {
        Self::new(0, 9)
    }
    fn aligned() -> Self {
        Self::new(32, 19)
    }

    fn with_dense(vocab: i64, dim: i64, seed: u64) -> (Self, MmapDenseSpec, u64) {
        let m = moe(6, 256, 256);
        let mut rng = Rng::new(seed);
        let mut file = rng.bytes(1024);
        let experts = CLASSES_4
            .iter()
            .enumerate()
            .map(|(il, c)| c.map(|c| kquant_layer(&mut rng, &m, il, c, 32, &mut file)))
            .collect();
        let mut add = |name: &str, ty: GgufTensorType| {
            file.resize(file.len() + (32 - file.len() % 32) % 32, 0);
            let off = file.len() as u64;
            let info = GgufTensorInfo {
                name: name.into(),
                n_dims: 2,
                dims: vec![dim as u64, vocab as u64],
                tensor_type: ty,
                offset: 0,
            };
            let bytes = info.data_size().unwrap();
            file.extend(rng.bytes(bytes as usize));
            (
                Located {
                    shard: 0,
                    data_offset: off,
                    info,
                },
                bytes,
            )
        };
        let (emb_loc, emb_b) = add("token_embd.weight", GgufTensorType::Q8_0);
        let (out_loc, out_b) = add("output.weight", GgufTensorType::BF16);
        let dir = Temp::new("bank-dense");
        let path = dir.write("data.bin", &file);
        let spec = MmapDenseSpec {
            token_embd: Some((emb_loc, vocab, dim)),
            output: Some((out_loc, vocab, dim)),
        };
        (
            Self {
                _dir: dir,
                path,
                moe: m,
                experts,
                file,
            },
            spec,
            emb_b + out_b,
        )
    }

    fn pool(&self, threads: usize) -> IoPool {
        pool_over(&self.path, threads)
    }
    fn streamed(&self, cache: [usize; 2], staging: usize) -> KqBank {
        KqBank::new(self.pool(4), &self.moe, &self.experts, cache, staging).unwrap()
    }
    fn direct(&self) -> KqBank {
        KqBank::new_direct(self.pool(4), &self.moe, &self.experts).unwrap()
    }
    fn mmap(&self, warm: u64, force_prewarm: bool) -> Result<KqBank> {
        let ps = std::slice::from_ref(&self.path);
        KqBank::new_mmap(
            self.pool(2),
            &self.moe,
            &self.experts,
            ps,
            warm,
            force_prewarm,
        )
    }
    fn expected_mapped(&self) -> u64 {
        self.experts
            .iter()
            .flatten()
            .map(|l| file_bytes(&self.moe, l.class).unwrap() * self.moe.n_experts as u64)
            .sum()
    }
    fn check(&self, il: usize, p: &Prepared) {
        check_prepared(&self.file, &self.moe, self.experts[il].as_ref().unwrap(), p);
    }
    fn prep(&self, bank: &KqBank, il: usize, exps: &[u32], uses: &[u32], dec: bool) -> Prepared {
        let p = bank.prepare(il, exps, uses, dec).unwrap();
        bank.assert_slots(&p);
        self.check(il, &p);
        p
    }
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
        let (g, per) = (geo[pi], geo[pi].file_bytes() as usize);
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

fn assert_direct_matches_streamed(
    direct: &KqBank,
    streamed: &KqBank,
    fx: &BankFixture,
    label: &str,
) {
    let all: Vec<u32> = (0..fx.moe.n_experts as u32).collect();
    for il in [1usize, 2, 3] {
        let geo = geometry(&fx.moe, fx.experts[il].as_ref().unwrap().class).unwrap();
        let da = direct.direct_layer(il).unwrap().expect("direct arrays");
        let sp = streamed.prepare(il, &all, &[], false).unwrap();
        for pi in 0..3 {
            for (&e, &s) in sp.experts.iter().zip(&sp.slots) {
                assert_eq!(
                    bank_slot(&da[pi], geo[pi], e),
                    bank_slot(&sp.arrays[pi], geo[pi], s),
                    "{label} layer {il} proj {pi} expert {e}"
                );
            }
        }
        streamed.settle_all();
    }
}

#[test]
fn class_layout_follows_the_fixture() {
    let f = fixture("expert-classes.json");
    let (a, b) = (
        j_usizes(&f["q6k_down_layers"]),
        j_usizes(&f["q4k_down_layers"]),
    );
    assert_eq!((a.len(), b.len()), (19, 22));
    let mut classes = vec![None; 42];
    for &il in &a {
        classes[il] = Some(ExpertClass::Q6kDown);
    }
    for &il in &b {
        classes[il] = Some(ExpertClass::Q4kDown);
    }
    let l = ClassLayout::new(&classes).unwrap();
    assert_eq!((&l.layers, l.of_layer[0]), (&[a.clone(), b.clone()], None));
    for (ci, list) in [a, b].iter().enumerate() {
        for (local, &il) in list.iter().enumerate() {
            assert_eq!(l.of_layer[il], Some((ci, local)), "layer {il}");
        }
    }
    assert!(ClassLayout::new(&[Some(ExpertClass::Bf16)]).is_err());
}

#[test]
fn slot_and_file_bytes_match_the_extents_fixture() {
    let (f, m) = (fixture("expert-extents.json"), moe(512, 1536, 1536));
    for (key, class) in [("A", ExpertClass::Q6kDown), ("B", ExpertClass::Q4kDown)] {
        let s = &f["slots"][key];
        let (sb, fb) = (
            slot_bytes(&m, class).unwrap(),
            file_bytes(&m, class).unwrap(),
        );
        assert_eq!(sb, j_u64(&s["mlx_slot_bytes"]), "{key} slot");
        assert_eq!(fb, j_u64(&s["file_bytes_per_expert"]), "{key} file");
        assert_eq!(sb * 512, j_u64(&s["staging_layer_mlx_bytes"]));
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
            assert_eq!(geo[p].file_bytes(), j_u64(&t["per_expert_bytes"]));
            assert_eq!(geo[p].slot_bytes(), j_u64(&t["per_expert_mlx_bytes"]));
        }
    }
}

#[test]
fn expert_orientation_is_checked_per_projection() {
    let (m, class) = (moe(4, 512, 256), ExpertClass::Q6kDown);
    let geo = geometry(&m, class).unwrap();
    assert_eq!(
        geo.map(|g| (g.rows, g.k)),
        [(512, 256), (512, 256), (256, 512)]
    );
    let (good, _) = layer_meta(&m, 1, class, 0);
    check_layer(1, &good, class, &geo, 4).unwrap();
    let mut bad_gate = good.clone();
    bad_gate.gate.info.dims = vec![512, 256, 4];
    let mut bad_down = good.clone();
    bad_down.down.info.dims = vec![256, 512, 4];
    let mut bad_ty = good.clone();
    bad_ty.down.info.tensor_type = GgufTensorType::Q4K;
    for b in [&bad_gate, &bad_down, &bad_ty] {
        assert!(check_layer(1, b, class, &geo, 4).is_err());
    }
    assert!(check_layer(1, &good, ExpertClass::Q4kDown, &geo, 4).is_err());
    assert!(check_layer(1, &good, class, &geo, 5).is_err());
    assert!(geometry(&m, ExpertClass::Bf16).is_err());
    assert!(geometry(&moe(4, 512, 100), class).is_err());
}

#[test]
fn decode_misses_take_victims_then_hit() {
    let mut d = Directory::new(1, 2, 8, 2).unwrap();
    d.check_step(1, &[3, 5], &[]).unwrap();
    let p = dir_step(&mut d, 1, &[3, 5], true, true);
    assert!(p.victims.len() == 2 && p.staging.is_empty() && p.copies.is_empty());
    assert!(p.slots.iter().all(|&s| s >= d.base()));
    for (&e, &s) in [3u32, 5].iter().zip(&p.slots) {
        assert!(d.holds(1, e, s) && !d.holds(0, e, s));
    }
    let q = d.plan(1, &[3, 5], &[], true);
    assert_eq!((q.hits, q.reads(), q.slots), (2, 0, p.slots));
    d.cache().check().unwrap();
}

#[test]
fn prefill_misses_stage_and_keep_their_tags() {
    let mut d = Directory::new(1, 3, 8, 0).unwrap();
    let p = dir_step(&mut d, 0, &[1, 4, 7], false, false);
    assert_eq!((p.staging, p.slots), (vec![1, 4, 7], vec![1, 4, 7]));
    assert_eq!(d.plan(0, &[1, 4], &[], false).reads(), 0);
    let q = dir_step(&mut d, 1, &[4], false, false);
    assert!(q.staging == vec![4] && !d.holds(0, 4, 4) && d.holds(1, 4, 4) && d.holds(0, 1, 1));
}

#[test]
fn staging_halves_alternate_by_layer() {
    let mut d = Directory::new(2, 4, 8, 0).unwrap();
    let slots = [0, 1, 2].map(|il| d.plan(il, &[2], &[], false).slots[0]);
    assert_eq!((slots, d.total_slots()), ([2, 10, 2], 16));
}

#[test]
fn prefill_seed_copies_a_valid_staging_slot() {
    let mut d = Directory::new(1, 1, 8, 1).unwrap();
    let p = dir_step(&mut d, 0, &[1, 2], false, true);
    assert_eq!((p.victims, p.staging), (vec![(1, 0)], vec![2]));
    d.new_turn();
    let q = d.plan(0, &[2], &[], false);
    assert_eq!((q.copies.clone(), q.reads()), (vec![(2, 0)], 0));
    d.commit_copies(0, &q);
    assert!(d.holds(0, 2, d.base()) && d.cache().lookup(0, 1).is_none());
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
    assert_eq!(d.plan(0, &[0, 6], &[], true).victims.len(), 2);
}

#[test]
fn promote_moves_decode_staging_copies() {
    let mut d = Directory::new(1, 4, 8, 2).unwrap();
    assert_eq!(dir_step(&mut d, 0, &[5], false, false).staging, vec![5]);
    let q = d.plan(0, &[5], &[], true);
    assert_eq!((q.reads(), q.slots), (0, vec![5]));
    d.complete_all();
    let moved = d.promote();
    assert!(moved.len() == 1 && moved[0].0 == 5 && d.holds(0, 5, moved[0].1));
    d.cache().check().unwrap();
}

#[test]
fn an_unsettled_slot_is_never_a_victim() {
    let mut d = Directory::new(1, 2, 8, 2).unwrap();
    let p = dir_step(&mut d, 0, &[1, 2], true, false);
    assert_eq!(p.victims.len(), 2);
    let q = dir_step(&mut d, 1, &[3, 4], true, false);
    assert!(q.victims.is_empty() && q.copies.is_empty() && !q.drain && q.staging == vec![3, 4]);
    d.complete(p.epoch);
    let r = dir_step(&mut d, 1, &[5, 6], true, false);
    assert_eq!(r.victims.len(), 2);
    let (mut got, mut was) = (r.slots, p.slots);
    got.sort_unstable();
    was.sort_unstable();
    assert_eq!(got, was, "only settled step's slots");
    let s = d.plan(0, &[7], &[], true);
    assert!(s.victims.is_empty() && s.staging == vec![7]);
    d.cache().check().unwrap();
}

#[test]
fn a_staging_refill_of_an_unsettled_slot_drains() {
    let mut d = Directory::new(1, 2, 8, 0).unwrap();
    let p = dir_step(&mut d, 0, &[2, 5], false, false);
    assert_eq!((p.staging, p.drain), (vec![2, 5], false));
    let h = d.plan(0, &[5], &[], false);
    assert_eq!((h.reads(), h.drain), (0, false));
    let q = d.plan(1, &[2], &[], false);
    assert_eq!((q.staging.clone(), q.drain), (vec![2], true));
    d.complete(q.epoch - 1);
    dir_commit(&mut d, 1, &q, false);
    let r = d.plan(0, &[2], &[], false);
    assert!(r.drain);
    d.complete(r.epoch - 1);
    assert!(!d.plan(0, &[3], &[], false).drain);
    d.complete_all();
    let t = d.plan(1, &[5], &[], false);
    assert_eq!((t.staging, t.drain), (vec![5], false));
}

#[test]
fn step_checks_refuse_bad_input() {
    let d = Directory::new(1, 2, 8, 0).unwrap();
    for (layer, exps, uses) in [
        (2usize, &[0u32][..], &[][..]),
        (0, &[8][..], &[][..]),
        (0, &[3, 3][..], &[][..]),
        (0, &[3, 1][..], &[][..]),
        (0, &[1, 3][..], &[1][..]),
    ] {
        assert!(d.check_step(layer, exps, uses).is_err());
    }
    d.check_step(0, &[1, 3], &[2, 1]).unwrap();
    assert!(Directory::new(3, 2, 8, 0).is_err() && Directory::new(1, 2, 0, 0).is_err());
}

#[test]
fn two_classes_keep_separate_directories() {
    let f = fixture("expert-classes.json");
    let mut classes = vec![None; 42];
    for &il in &j_usizes(&f["q6k_down_layers"]) {
        classes[il] = Some(ExpertClass::Q6kDown);
    }
    for &il in &j_usizes(&f["q4k_down_layers"]) {
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
        let p = dir_step(&mut dirs[ci], local, &experts, true, true);
        for (&e, &s) in experts.iter().zip(&p.slots) {
            assert!(dirs[ci].holds(local, e, s), "layer {il} expert {e}");
        }
    }
    assert_eq!(
        (dirs[0].cache().occupied(), dirs[1].cache().occupied()),
        (19 * 3, 32)
    );
    dirs.iter().for_each(|d| d.cache().check().unwrap());
}

#[test]
fn repack_sink_reads_unaligned_extents() {
    let dir = Temp::new("bank");
    let data = Rng::new(3).bytes(3 * 16384 + 777);
    let pool = pool_over(&dir.write("data.bin", &data), 3);
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
                Arc::new(move |b: &[u8]| {
                    lock(&got)[i] = Some(b.to_vec());
                    Ok(())
                }),
            )
        })
        .collect();
    // SAFETY: callbacks write only `got` under its lock.
    let io = unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        (io.reads, io.bytes),
        (spans.len(), spans.iter().map(|s| s.1).sum())
    );
    for (i, &(off, len)) in spans.iter().enumerate() {
        assert_eq!(
            lock(&got)[i].as_deref(),
            Some(&data[off as usize..(off + len) as usize])
        );
    }
}

#[test]
fn repack_sink_failures_fail_the_ticket() {
    let dir = Temp::new("bank");
    let data = Rng::new(5).bytes(4 * 16384);
    let path = dir.write("data.bin", &data);
    let pool = pool_over(&path, 2);
    let ext = |offset, len| Extent {
        shard: 0,
        offset,
        len,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let counting = |c: &Arc<AtomicUsize>| -> RepackFnArc {
        let c = Arc::clone(c);
        Arc::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    };
    let fail: RepackFnArc = Arc::new(|_| Err(Error::from_reason("repack refused")));
    let reads = vec![
        Read::repack(ext(0, 100), fail),
        Read::repack(ext(100, 100), counting(&calls)),
    ];
    // SAFETY: callbacks touch only atomics.
    let err = unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(err.reason.contains("repack refused") && calls.load(Ordering::Relaxed) == 1);

    let boom: RepackFnArc = Arc::new(|_| panic!("repack panics"));
    // SAFETY: as above.
    let err = unsafe { pool.submit(vec![Read::repack(ext(0, 10), boom)], Priority::Low) }
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(err.reason.contains("panicked"), "{}", err.reason);

    let oob = vec![Read::repack(
        ext(data.len() as u64 - 10, 11),
        counting(&calls),
    )];
    // SAFETY: as above.
    assert!(unsafe { pool.submit(oob, Priority::High) }.is_err());

    truncate_file(&path, 2 * 16384);
    let before = calls.load(Ordering::Relaxed);
    let short = vec![Read::repack(ext(2 * 16384 - 50, 200), counting(&calls))];
    // SAFETY: as above.
    let err = unsafe { pool.submit(short, Priority::High) }
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(err.reason.contains("short read") && calls.load(Ordering::Relaxed) == before);
}

#[test]
fn slot_repack_equals_whole_tensor_repacker() {
    let (n, rows, k) = (5usize, 67usize, 512usize);
    for (i, ty) in [GgufTensorType::Q4K, GgufTensorType::Q6K]
        .into_iter()
        .enumerate()
    {
        let format = ty.k_quant_format().unwrap();
        let geo = ProjGeo { format, rows, k };
        let (per, at) = (geo.file_bytes() as usize, 1000 + 7 * i);
        let mut file = Rng::new(1).bytes(at);
        let payload = Rng::new(11 + i as u64).bytes(n * per);
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
            d.write(
                &loc.info,
                slot,
                &payload[e as usize * per..(e as usize + 1) * per],
            )
            .unwrap();
        }
        let row = format.row_bytes(k);
        assert!(d.write(&loc.info, 3, &payload[..per]).is_err());
        assert!(d.write(&loc.info, 0, &payload[..per - 1]).is_err());
        assert!(
            d.write_rows(&loc.info, 0, rows - 1, 2, &payload[..2 * row])
                .is_err()
        );
        assert!(d.write_rows(&loc.info, 0, 0, 0, &[]).is_err());

        let dir = Temp::new("bank");
        let pool = pool_over(&dir.write("data.bin", &file), 3);
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
            assert_eq!(
                reads.iter().map(|r| r.extent.len).sum::<u64>(),
                (picks.len() * per) as u64
            );
            // SAFETY: distinct rows/slots of `pooled`, alive until wait returns.
            unsafe { pool.submit(reads, Priority::High) }
                .unwrap()
                .wait()
                .unwrap();
            for (slot, &e) in picks.iter().enumerate() {
                let want = expert_of(&whole, geo, e as usize);
                assert_eq!((direct.slot(slot), pooled.slot(slot)), (want.clone(), want));
            }
            assert!(
                p.reads_banded(&loc, 5, 0, band).is_err()
                    && p.reads_banded(&loc, 0, 3, band).is_err()
            );
        }
    }
}

#[test]
fn band_rows_aligns_q4k_1536_to_16k_pages_and_matches_repack() {
    let q4k_row = KQuantFormat::Q4K.row_bytes(1536);
    assert_eq!(q4k_row, 864);
    let step = band_rows(1536, q4k_row, BAND_BYTES);
    assert_eq!((step, (step * q4k_row) % 16384), (512, 0));

    for m in 0..512u64 {
        let off = m * 32;
        let ranges = band_ranges(off, 1536, q4k_row, BAND_BYTES);
        assert!(!ranges.is_empty());
        let mut cur = 0usize;
        for (i, &(r0, len)) in ranges.iter().enumerate() {
            assert!(r0 == cur && len > 0, "m={m} band {i}");
            cur += len;
            if i + 1 < ranges.len() {
                assert_eq!(
                    (off + cur as u64 * q4k_row as u64) % 16384,
                    0,
                    "m={m} band {i}"
                );
            }
        }
        assert_eq!(cur, 1536);
    }

    let q6k_row = KQuantFormat::Q6K.row_bytes(1536);
    assert_eq!(q6k_row, 1260);
    let mut cur = 0usize;
    for (r0, len) in band_ranges(14_336, 1536, q6k_row, BAND_BYTES) {
        assert!(r0 == cur && len > 0);
        cur += len;
    }
    assert_eq!(cur, 1536);

    let geo = ProjGeo {
        format: KQuantFormat::Q4K,
        rows: 1536,
        k: 1536,
    };
    let (per, at) = (geo.file_bytes() as usize, 14_336usize);
    let mut file = Rng::new(41).bytes(at);
    let payload = Rng::new(42).bytes(2 * per);
    file.extend_from_slice(&payload);
    let whole = repack_kquant(KQuantFormat::Q4K, &payload, 2 * 1536, 1536).unwrap();
    let loc = Arc::new(located(
        "blk.1.ffn_gate_exps.weight",
        [1536, 1536, 2],
        GgufTensorType::Q4K,
        at as u64,
    ));
    let dir = Temp::new("bank");
    let pool = pool_over(&dir.write("data.bin", &file), 4);
    let mut host = Host::new(geo, 2);
    let dst = host.dst();
    let mut reads = dst.reads(&loc, 1, 0).unwrap();
    reads.extend(dst.reads(&loc, 0, 1).unwrap());
    sort_reads(&mut reads);
    assert!(reads.windows(2).all(
        |w| (w[0].extent.shard, w[0].extent.offset) <= (w[1].extent.shard, w[1].extent.offset)
    ));
    // SAFETY: `host` outlives ticket; disjoint slots.
    unsafe { pool.submit(reads, Priority::High) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        (host.slot(0), host.slot(1)),
        (expert_of(&whole, geo, 1), expert_of(&whole, geo, 0))
    );
}

#[test]
fn bank_serves_both_classes_across_turns() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([2, 1], 1);
    assert_eq!(
        (bank.class_of(0), bank.class_of(2)),
        (None, Some(ExpertClass::Q4kDown))
    );
    assert_eq!(bank.cache_slots(), [2, 1]);

    let p = fx.prep(&bank, 1, &[0, 3], &[], true);
    assert_eq!(p.class, ExpertClass::Q6kDown);
    bank.settle_all();
    assert_eq!(bank.stats()[0].slots, 2);

    let h = bank.prepare(1, &[0, 3], &[], true).unwrap();
    assert_eq!(
        (h.slots.clone(), bank.stats()[0].hits),
        (p.slots.clone(), 2)
    );
    bank.settle_all();

    let e = fx.prep(&bank, 3, &[5], &[], true);
    bank.settle_all();

    let b = fx.prep(&bank, 2, &[1, 2, 5], &[1, 4, 1], false);
    assert_eq!(b.class, ExpertClass::Q4kDown);
    bank.settle_all();
    assert_eq!(bank.stats()[1].slots, 3);
    assert!(
        bank.prepare(0, &[1], &[], true).is_err() && bank.prepare(1, &[3, 1], &[], true).is_err()
    );

    bank.end_turn().unwrap();
    assert!(!bank.in_turn());
    bank.begin_turn().unwrap();
    let again = fx.prep(&bank, 1, &[3], &[], true);
    bank.settle_all();
    bank.end_turn().unwrap();

    assert!(bank.rebuild([0, 0], 2).is_err());
    drop((p, h, e, b, again));
    bank.rebuild([1, 0], 2).unwrap();
    assert_eq!((bank.staging_layers(), bank.cache_slots()), (2, [1, 0]));
    let _r = fx.prep(&bank, 3, &[0, 5], &[], false);
}

#[test]
fn bank_drains_only_for_an_unsettled_staging_slot() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([0, 0], 1);
    let at = syncs();
    let a = fx.prep(&bank, 1, &[0, 3], &[], false);
    assert_eq!(syncs(), at);

    let issued = bank.issued();
    let _b = fx.prep(&bank, 3, &[0], &[], false);
    assert_eq!((syncs(), bank.stats()[0].drains), (at + 1, 1));

    bank.settle(issued);
    let c = bank.prepare(1, &[3], &[], false).unwrap();
    assert_eq!((c.slots, syncs()), (a.slots[1..].to_vec(), at + 1));

    bank.settle_all();
    let _d = fx.prep(&bank, 1, &[0], &[], false);
    assert_eq!((syncs(), bank.stats()[0].drains), (at + 1, 1));
    bank.end_turn().unwrap();
}

#[test]
fn safe_point_syncs_only_when_staged_copies_are_unsettled() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([2, 1], 1);

    let at = syncs();
    assert_eq!((bank.safe_point(false).unwrap(), syncs()), (0, at));

    let _v = bank.prepare(1, &[0], &[], true).unwrap();
    let at = syncs();
    assert_eq!((bank.safe_point(false).unwrap(), syncs()), (0, at));
    bank.settle_all();

    let _p1 = bank.prepare(1, &[1, 2], &[2, 1], false).unwrap();
    bank.settle_all();
    let _d1 = bank.prepare(1, &[2], &[], true).unwrap();
    bank.settle_all();
    let at = syncs();
    assert_eq!((bank.safe_point(false).unwrap(), syncs()), (1, at));

    let _p2 = bank.prepare(1, &[4], &[], false).unwrap();
    bank.settle_all();
    let _d2 = bank.prepare(1, &[4], &[], true).unwrap();
    let at = syncs();
    assert_eq!((bank.safe_point(false).unwrap(), syncs()), (1, at + 1));

    let at = syncs();
    assert_eq!((bank.safe_point(true).unwrap(), syncs()), (0, at));
    bank.end_turn().unwrap();
}

#[test]
fn lfu_directory_keeps_frequent_expert_over_recent_one_hit() {
    let mut d = Directory::new_with_lfu(1, 1, 8, 2, 100.0).unwrap();
    dir_step(&mut d, 0, &[0], true, true);
    for _ in 0..5 {
        let h = d.plan(0, &[0], &[], true);
        assert_eq!(h.hits, 1);
        d.complete(h.epoch);
    }
    for e in [1u32, 2] {
        dir_step(&mut d, 0, &[e], true, true);
    }
    assert!(d.cache().lookup(0, 0).is_some() && d.cache().lookup(0, 1).is_none());
    assert!(d.cache().lookup(0, 2).is_some());
    d.cache().check().unwrap();
}

#[test]
fn default_bank_uses_lru_policy() {
    if use_lfu_policy() {
        return;
    }
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([2, 0], 1);
    for e in [0, 0, 0, 0, 0, 0, 1, 2] {
        let _ = bank.prepare(1, &[e], &[], true).unwrap();
        bank.settle_all();
    }
    let g = lock(&bank.inner);
    let c = g.classes[0].dir.cache();
    assert!(c.lookup(0, 0).is_none() && c.lookup(0, 1).is_some() && c.lookup(0, 2).is_some());
    drop(g);
    bank.end_turn().unwrap();
}

#[test]
fn sorted_slot_bounds_counts_exact_bm32_tiles() {
    assert_eq!(
        (sorted_slot_bounds(&[]), sorted_slot_bounds(&[7])),
        ((0, 0), (1, 8))
    );
    let mut s = vec![3u32; 32];
    s.extend(std::iter::repeat_n(10u32, 33));
    s.push(200);
    assert_eq!(sorted_slot_bounds(&s), (4, 201));
}

#[test]
fn split_step_publish_abandon_and_drop_on_error() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([4, 2], 1);
    assert!(bank.split() && bank.prefetch());

    let warm = bank.prepare(1, &[1], &[], true).unwrap();
    bank.assert_slots(&warm);
    bank.settle_all();

    let (p, pending) = bank.prepare_split(1, &[1, 3], &[], true).unwrap();
    assert!(!p.is_reading(1) && p.is_reading(3) && p.reading == vec![3]);
    bank.assert_ready_slots(&p);
    bank.note_split(1, true);
    pending.expect("miss 3").finish().unwrap();
    bank.assert_slots(&p);
    fx.check(1, &p);
    bank.settle_all();
    let st = bank.stats()[0];
    assert!(st.split_steps == 1 && st.split_mixed == 1 && st.wait_ns > 0);
    bank.test_check_caches();

    // Early drop of Pending waits for IO and releases victims.
    let occ = occupied(&bank, 0);
    let (p_drop, pending_drop) = bank.prepare_split(1, &[4, 5], &[], true).unwrap();
    assert!(p_drop.reading == vec![4, 5] && pending_drop.is_some());
    drop(pending_drop);
    assert_eq!((bank.reads_in_flight(), occupied(&bank, 0)), (0, occ));
    bank.test_check_caches();
    bank.settle_all();

    // Truncated file triggers IO error on Pending::finish and releases victims.
    truncate_file(&fx.path, 1024);
    let (_p_err, pending_err) = bank.prepare_split(1, &[2, 4], &[], true).unwrap();
    assert!(pending_err.unwrap().finish().is_err() && occupied(&bank, 0) == occ);
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn two_class_interleaved_read_ahead() {
    let fx = BankFixture::unaligned();
    let single = fx.streamed([0, 0], 1);
    assert_eq!(single.prefetch_layer(1, &[0, 2]).unwrap(), 0);
    single.end_turn().unwrap();

    let bank = fx.streamed([0, 0], 2);
    bank.set_prefetch(false);
    assert_eq!(bank.prefetch_layer(1, &[0, 2]).unwrap(), 0);
    bank.set_prefetch(true);

    assert_eq!(bank.prefetch_layer(1, &[0, 2, 5]).unwrap(), 3);
    assert_eq!(bank.prefetch_layer(2, &[1, 4]).unwrap(), 2);

    let _p1 = fx.prep(&bank, 1, &[0, 2], &[1, 1], false);
    bank.end_layer(1);
    bank.settle_all();
    assert_eq!(
        (
            bank.stats()[0].prefetch_slots,
            bank.stats()[0].prefetch_unused
        ),
        (3, 1)
    );

    assert_eq!(bank.prefetch_layer(3, &[0, 3]).unwrap(), 2);

    let s1 = bank.stats()[1].slots;
    let _p2 = fx.prep(&bank, 2, &[1, 4], &[1, 1], false);
    bank.end_layer(2);
    bank.settle_all();
    let st1 = bank.stats()[1];
    assert_eq!(
        (st1.slots - s1, st1.prefetch_slots, st1.prefetch_unused),
        (2, 2, 0)
    );

    let _p3 = fx.prep(&bank, 3, &[0, 3, 4], &[1, 1, 1], false);
    bank.end_layer(3);
    bank.settle_all();
    let st0 = bank.stats()[0];
    assert_eq!(
        (st0.prefetch_slots, st0.prefetch_unused, st0.slots),
        (5, 1, 6)
    );
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn decode_read_ahead_harvests_hits_and_counts_unused() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([4, 2], 2);
    bank.set_decode_prefetch(false);
    assert!(!bank.decode_prefetch());
    bank.set_decode_prefetch(true);
    assert!(bank.decode_prefetch());

    let single = fx.streamed([2, 2], 1);
    assert_eq!(single.prefetch_layer(1, &[0, 1]).unwrap(), 0);
    single.end_turn().unwrap();

    let (p1, pend1) = bank.prepare_split(1, &[0, 1], &[1, 1], true).unwrap();
    assert_eq!(bank.prefetch_layer(2, &[1, 3, 5]).unwrap(), 3);
    pend1.expect("l1 misses").finish().unwrap();
    bank.assert_slots(&p1);
    fx.check(1, &p1);
    bank.end_layer(1);
    bank.settle_all();
    assert_eq!(bank.stats()[1].prefetch_slots, 3);

    let (p2, pend2) = bank.prepare_split(2, &[1, 3, 4], &[1, 1, 1], true).unwrap();
    assert_eq!(p2.reading, vec![4]);
    bank.assert_ready_slots(&p2);
    assert_eq!(bank.prefetch_layer(3, &[0, 2]).unwrap(), 2);
    pend2.expect("miss 4").finish().unwrap();
    bank.assert_slots(&p2);
    fx.check(2, &p2);
    bank.end_layer(2);
    bank.settle_all();
    let st1 = bank.stats()[1];
    assert_eq!(
        (st1.prefetch_slots, st1.prefetch_unused, st1.slots),
        (3, 1, 4)
    );

    let (p3, pend3) = bank.prepare_split(3, &[0, 2], &[1, 1], true).unwrap();
    assert!(pend3.is_none() && p3.reading.is_empty());
    bank.assert_slots(&p3);
    fx.check(3, &p3);
    bank.end_layer(3);
    bank.settle_all();
    let st0 = bank.stats()[0];
    assert_eq!(
        (st0.prefetch_slots, st0.prefetch_unused, st0.drains),
        (2, 0, 0)
    );
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn prefill_subgroups_last_use_order_and_drop_on_error() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([4, 0], 1);
    let warm = bank.prepare(1, &[1], &[], true).unwrap();
    bank.assert_slots(&warm);
    bank.settle_all();

    let used = [0u32, 1, 2, 3, 4, 5];
    let (p, g0, mut sg) = bank
        .plan_prefill_subgroups(1, &used, &[1, 1, 5, 4, 2, 3], &[5, 1, 3, 0, 4, 2], 2)
        .unwrap();
    assert_eq!(g0, vec![1]);
    bank.assert_ready_slots(&p);
    let (mut order, mut batches) = (Vec::new(), Vec::new());
    while let Some(batch) = sg.next_subgroup().unwrap() {
        bank.assert_experts_slots(&p, &batch);
        batches.push(batch.clone());
        order.extend(batch);
    }
    assert_eq!(
        (order, batches),
        (vec![5, 3, 0, 4, 2], vec![vec![5, 3], vec![0, 4], vec![2]])
    );
    bank.assert_slots(&p);
    fx.check(1, &p);
    bank.settle_all();
    bank.test_check_caches();

    // Early drop releases uncommitted victims and waits for IO.
    let occ = occupied(&bank, 0);
    {
        let (_, g0, _sg) = bank
            .plan_prefill_subgroups(3, &used, &[6, 5, 4, 3, 2, 1], &[5, 4, 3, 2, 1, 0], 2)
            .unwrap();
        assert!(g0.is_empty());
    }
    assert_eq!((bank.reads_in_flight(), occupied(&bank, 0)), (0, occ));
    bank.test_check_caches();
    bank.settle_all();

    // Truncated file triggers IO error inside next_subgroup and releases all victims.
    truncate_file(&fx.path, 1024);
    let (_, _, mut sg_err) = bank
        .plan_prefill_subgroups(3, &[0, 1, 2, 3], &[4, 3, 2, 1], &[3, 2, 1, 0], 2)
        .unwrap();
    assert!(sg_err.next_subgroup().is_err());
    drop(sg_err);
    assert_eq!(occupied(&bank, 0), occ);
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn direct_bank_background_fill_matches_streamed_prepare_bit_for_bit() {
    let fx = BankFixture::unaligned();
    let (streamed, direct) = (fx.streamed([2, 1], 1), fx.direct());
    let ne = fx.moe.n_experts;
    assert!(!streamed.is_direct() && direct.is_direct() && direct.staging_layers() == 0);
    assert_eq!(direct.cache_slots(), [2 * ne, ne]);
    let [sa, sb] = direct.slot_bytes();
    assert_eq!(direct.nbytes(), (2 * ne as u64) * sa + (ne as u64) * sb);
    assert!(streamed.direct_layer(1).unwrap().is_none());
    assert!(direct.direct_layer(0).is_err() && direct.rebuild([0, 0], 1).is_err());

    assert_direct_matches_streamed(&direct, &streamed, &fx, "direct");
    direct.wait_direct_fill().unwrap();
    assert!(direct.direct_fill_complete());

    let (ps, pend) = direct
        .prepare_split(1, &[0, 3, 5], &[1, 2, 1], true)
        .unwrap();
    assert!(pend.is_none() && ps.reading.is_empty() && ps.slots == vec![0, 3, 5]);
    direct.assert_slots(&ps);
    fx.check(1, &ps);

    let (pg, g0, mut sg) = direct
        .plan_prefill_subgroups(2, &[1, 2, 4], &[1, 1, 1], &[4, 2, 1], 2)
        .unwrap();
    assert!(
        g0 == vec![1, 2, 4] && pg.slots == vec![1, 2, 4] && sg.next_subgroup().unwrap().is_none()
    );
    direct.assert_slots(&pg);
    fx.check(2, &pg);

    assert!(
        direct
            .stats()
            .iter()
            .all(|s| s.hits > 0 && s.hits == s.uses && s.slots == 0)
    );
    assert_eq!(direct.safe_point(true).unwrap(), 0);
    direct.end_turn().unwrap();
    let res = direct.residency().unwrap();
    assert!(!direct.in_turn() && res.resident > 0 && res.volatile == 0);
    assert!(!direct.begin_turn().unwrap() && direct.in_turn());
    streamed.end_turn().unwrap();
}

#[test]
fn direct_bank_drop_mid_fill_and_io_error_propagate_cleanly() {
    let fx = BankFixture::unaligned();
    let shards = Arc::new(ShardSet::open(&[fx.path.as_path()]).unwrap());
    shards.delay_ms.store(15, Ordering::Relaxed);
    let bank = KqBank::new_direct(IoPool::new(shards, 2).unwrap(), &fx.moe, &fx.experts).unwrap();
    assert!(!bank.direct_fill_complete());
    drop(bank);

    let shards = Arc::new(ShardSet::open(&[fx.path.as_path()]).unwrap());
    truncate_file(&fx.path, 1024);
    let bank = KqBank::new_direct(IoPool::new(shards, 2).unwrap(), &fx.moe, &fx.experts).unwrap();
    assert!(bank.direct_layer(1).is_err() && bank.wait_direct_fill().is_err());
}

#[test]
fn direct_bank_late_layer_read_error_propagates_through_wait_direct_fill() {
    let fx = BankFixture::unaligned();
    let layer3_start = fx.experts[3].as_ref().unwrap().gate.data_offset;
    assert!(layer3_start > 0);
    truncate_file(&fx.path, layer3_start + 64);
    let bank = KqBank::new_direct(fx.pool(2), &fx.moe, &fx.experts).unwrap();
    assert!(bank.direct_layer(1).unwrap().is_some() && bank.direct_layer(2).unwrap().is_some());
    assert!(bank.wait_direct_fill().is_err() && bank.direct_layer(3).is_err());
}

#[test]
fn partial_bank_promotes_to_direct_when_idle_and_serves_identical_bytes() {
    let fx = BankFixture::unaligned();
    let (bank, reference) = (fx.streamed([2, 1], 1), fx.streamed([2, 1], 1));
    assert!(!bank.is_direct());

    let warm = fx.prep(&bank, 1, &[0, 3], &[], true);
    bank.settle_all();
    assert!(bank.promote_to_direct().is_err() && !bank.is_direct() && bank.in_turn());

    drop(warm);
    bank.end_turn().unwrap();
    bank.promote_to_direct().unwrap();
    let ne = fx.moe.n_experts;
    assert!(bank.is_direct() && !bank.in_turn() && bank.staging_layers() == 0);
    assert_eq!(bank.cache_slots(), [2 * ne, ne]);
    bank.wait_direct_fill().unwrap();
    assert!(bank.direct_fill_complete());
    bank.promote_to_direct().unwrap();

    assert_direct_matches_streamed(&bank, &reference, &fx, "promoted");

    assert!(!bank.begin_turn().unwrap() && bank.in_turn());
    let p = fx.prep(&bank, 2, &[1, 4, 5], &[], true);
    assert_eq!(p.slots, vec![1, 4, 5]);
    bank.end_turn().unwrap();
    let res = bank.residency().unwrap();
    assert!(!bank.in_turn() && res.resident > 0 && res.volatile == 0);
    reference.end_turn().unwrap();
}

#[test]
fn promote_to_direct_restores_partial_bank_on_alloc_failure() {
    let fx = BankFixture::unaligned();
    let bank = fx.streamed([2, 1], 1);
    let (old_nbytes, old_slots) = (bank.nbytes(), bank.cache_slots());
    bank.end_turn().unwrap();

    TEST_FAIL_DIRECT_ALLOC_AT.with(|c| c.set(Some(1)));
    let err = bank.promote_to_direct();
    TEST_FAIL_DIRECT_ALLOC_AT.with(|c| c.set(None));
    assert!(err.is_err() && !bank.is_direct() && !bank.in_turn());
    assert_eq!((bank.cache_slots(), bank.nbytes()), (old_slots, old_nbytes));

    bank.begin_turn().unwrap();
    let _p = fx.prep(&bank, 1, &[2, 5], &[], true);
    bank.settle_all();
    bank.end_turn().unwrap();
}

#[test]
fn mmap_bank_maps_views_reports_zero_anon_and_toggles_in_turn() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    for (raw, want) in [
        (None, Some(Duration::from_millis(DEFAULT_RSET_KEEPALIVE_MS))),
        (Some("0"), None),
        (Some("500"), Some(Duration::from_millis(500))),
        (
            Some("bad"),
            Some(Duration::from_millis(DEFAULT_RSET_KEEPALIVE_MS)),
        ),
    ] {
        assert_eq!(parse_rset_keepalive_ms(raw), want, "{raw:?}");
    }

    let fx = BankFixture::aligned();
    let bank = fx.mmap(fx.file.len() as u64, false).unwrap();
    assert!(bank.is_direct() && bank.is_mmap() && bank.direct_fill_complete());
    assert_eq!(bank.has_mmap_prewarm(), !bank.has_residency_set());
    assert_eq!(
        bank.has_rset_keepalive(),
        bank.has_residency_set() && rset_keepalive_interval().is_some()
    );
    bank.keep_residency_alive();
    assert!(!bank.take_prefill_ran());
    bank.mark_prefill_wave();
    assert!(bank.take_prefill_ran() && !bank.take_prefill_ran());
    assert_eq!((bank.nbytes(), bank.resident_bytes().unwrap()), (0, 0));
    assert_eq!(bank.residency().unwrap(), Residency::default());
    assert_eq!(
        (bank.mapped_bytes(), bank.staging_layers()),
        (fx.expected_mapped(), 0)
    );
    assert_eq!(bank.cache_slots(), [2 * fx.moe.n_experts, fx.moe.n_experts]);
    bank.wait_direct_fill().unwrap();
    assert!(bank.direct_layer_arrays(0).is_err() && bank.direct_layer(1).is_err());
    assert!(bank.rebuild([0, 0], 1).is_err() && bank.prepare(1, &[0], &[], true).is_err());
    bank.promote_to_direct().unwrap();
    assert!(bank.is_mmap());

    assert!(!bank.in_turn() && !bank.begin_turn().unwrap() && bank.in_turn());
    bank.set_in_turn(false);
    assert!(!bank.in_turn());
    bank.set_in_turn(true);
    assert!(bank.in_turn());
    bank.end_turn().unwrap();
    assert!(!bank.in_turn());

    for il in [1usize, 2, 3] {
        let geo = geometry(&fx.moe, fx.experts[il].as_ref().unwrap().class).unwrap();
        let Some(DirectLayerArrays::Mmap(projs)) = bank.direct_layer_arrays(il).unwrap() else {
            panic!("expected Mmap layer arrays for layer {il}");
        };
        for (p, g) in projs.iter().zip(&geo) {
            assert_eq!((p.format, p.rows, p.k), (g.format, g.rows, g.k));
            assert_eq!(
                (p.buf.dtype().unwrap(), p.buf.ndim().unwrap()),
                (DType::Uint8, 2)
            );
        }
    }
    bank.record_direct_use(2, 3);
    let st = bank.stats();
    assert_eq!(
        (st[0].uses, st[0].hits, st[0].slots),
        (2 * fx.moe.n_used, 2 * fx.moe.n_used, 0)
    );
    assert_eq!(
        (st[1].uses, st[1].hits, st[1].slots),
        (fx.moe.n_used + 3, fx.moe.n_used + 3, 0)
    );
    drop(bank);
}

#[test]
fn mmap_bank_rejects_misaligned_or_truncated_shard_cleanly() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let mut fx = BankFixture::aligned();
    fx.experts[1].as_mut().unwrap().gate.data_offset += 1;
    assert!(fx.mmap(0, false).is_err());

    let fx2 = BankFixture::aligned();
    truncate_file(&fx2.path, 2048);
    assert!(fx2.mmap(0, false).is_err());

    let fx3 = BankFixture::aligned();
    let forced = fx3.mmap(fx3.file.len() as u64, true).unwrap();
    assert!(forced.has_mmap_prewarm());
    drop(forced);
}

#[test]
fn rset_keepalive_idle_grace_and_mmap_dense_residency() {
    for (in_turn, ended_ms, now_ms, want) in [
        (false, RSET_NO_TURN_ENDED, 0u64, false),
        (false, RSET_NO_TURN_ENDED, 60_000, false),
        (true, RSET_NO_TURN_ENDED, 60_000, true),
        (true, 10_000, 90_000, true),
        (false, 10_000, 10_000, true),
        (false, 10_000, 25_000, true),
        (false, 10_000, 25_001, false),
        (false, 10_000, 70_000, false),
    ] {
        assert_eq!(
            should_ping_rset(in_turn, ended_ms, now_ms, RSET_IDLE_GRACE_MS),
            want
        );
    }

    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let (fx, dense_spec, dense_bytes) = BankFixture::with_dense(16, 64, 37);
    let ps = std::slice::from_ref(&fx.path);
    let (bank, dense) =
        KqBank::new_mmap_with_dense(fx.pool(2), &fx.moe, &fx.experts, ps, 0, false, &dense_spec)
            .unwrap();

    assert_eq!((bank.extra_mmap_bufs(), bank.nbytes()), (2, 0));
    assert_eq!(bank.mapped_bytes(), fx.expected_mapped() + dense_bytes);
    assert_eq!(bank.last_turn_end_ms(), Some(RSET_NO_TURN_ENDED));
    assert!(matches!(dense.token_embd, Some(Dense::MmapQ8_0 { .. })));
    assert!(matches!(dense.output, Some(Dense::MmapBf16 { .. })));
    assert_eq!(dense.token_embd.as_ref().unwrap().nbytes(), 0);
    assert_eq!(dense.output.as_ref().unwrap().nbytes(), 0);

    bank.begin_turn().unwrap();
    assert!(bank.in_turn() && bank.last_turn_end_ms() == Some(RSET_NO_TURN_ENDED));
    bank.end_turn().unwrap();
    assert!(!bank.in_turn() && bank.last_turn_end_ms() != Some(RSET_NO_TURN_ENDED));

    // Drop dense views BEFORE bank (matching Store's drop order).
    drop(dense);
    drop(bank);
}

fn stream_tensor(
    shards: &ShardSet,
    loc: &Located,
    g: ProjGeo,
    n: usize,
    mut f: impl FnMut(usize, usize, &[u8]),
) {
    let (per, mut buf) = (g.file_bytes(), AlignedBuf::new());
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

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_expert_repack_equals_whole_tensor_repacker() {
    let scan = real_scan();
    assert_eq!(scan.paths.len(), 2);
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
        // SAFETY: distinct slots of `host`, alive until wait returns.
        unsafe { pool.submit(reads, Priority::High) }
            .unwrap()
            .wait()
            .unwrap();
        for (i, &e) in picks.iter().enumerate() {
            assert_eq!(
                host.slot(picks.len() - 1 - i),
                expert_of(&whole, g, e as usize),
                "l{il} p{pi} e{e}"
            );
        }
    }
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_slot_gather_equals_whole_tensor_gather() {
    let scan = real_scan();
    assert_eq!(scan.paths.len(), 2);
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
            let x: Vec<f32> = Rng::new(17 + pi as u64)
                .bytes(3 * g.k)
                .into_iter()
                .map(|v| (f32::from(v) - 127.5) / 128.0)
                .collect();
            let x = MxArray::from_float32(&x, &[3, 1, 1, g.k as i64]).unwrap();
            let row = |v: &[u32]| (0..3).flat_map(|_| v.iter().copied()).collect::<Vec<_>>();
            let by_slot = MxArray::from_uint32(&row(&p.slots), &[3, 3]).unwrap();
            let by_id = MxArray::from_uint32(&row(&experts), &[3, 3]).unwrap();
            let got = read_f32(&p.arrays[pi].gather(&x, &by_slot).unwrap());
            let want = read_f32(&whole.gather(&x, &by_id).unwrap());
            assert!(got.len() == 3 * 3 * g.rows && got.iter().all(|v| v.is_finite()));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "layer {il} proj {pi}");
        }
        bank.settle_all();
    }
    bank.end_turn().unwrap();
}
