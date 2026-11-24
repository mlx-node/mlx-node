use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::models::hy_v4_flash::config::MoeConfig;
use crate::models::hy_v4_flash::scan::{ExpertClass, ExpertLayer};
use crate::models::hy_v4_flash::test_support::{
    Rng, Temp, fixture, j_u64, j_usize, j_usizes, kquant_layer, real_scan, tiny_cfg,
};
use crate::moe_stream::budget::{Bound, GROW_MIN, KEEP_SLACK, Terms};

const G: u64 = 1 << 30;
const STORE_CLASSES: [ExpertClass; 3] = [
    ExpertClass::Q6kDown,
    ExpertClass::Q4kDown,
    ExpertClass::Q4kDown,
];

fn host_mem(physical: u64, wired: u64) -> HostMemory {
    HostMemory {
        physical,
        wired,
        ..Default::default()
    }
}

fn host(inputs: &Value, others: u64) -> (HostMemory, u64) {
    (
        host_mem(j_u64(&inputs["physical"]), others),
        j_u64(&inputs["ws"]),
    )
}

fn base(inputs: &Value, variant: &Value) -> ModelSize {
    ModelSize {
        dense: j_u64(&variant["dense"]),
        slot: j_u64(&inputs["slot"]),
        n_experts: j_u64(&inputs["n_experts"]),
        ctx: 0,
    }
}

fn real_sizes() -> Sizes {
    Sizes {
        dense_trunk: 8_003_184_948,
        dense_mtp: 2_189_929_600,
        classes: Classes {
            layers: [19, 22],
            slot: [4_663_296, 4_091_904],
        },
        n_experts: 512,
    }
}

fn store_moe() -> MoeConfig {
    MoeConfig {
        n_experts: 4,
        n_used: 2,
        n_shared: 1,
        ff: 256,
        latent: 512,
        route_scale: 1.0,
        norm_topk: true,
    }
}

struct TestEnvGuard;
impl TestEnvGuard {
    fn new() -> Self {
        Self
    }
    fn host(&self, h: HostMemory, ws: u64) {
        TEST_HOST.with(|c| c.set(Some((h, ws))));
    }
    fn pool(&self, bytes: u64) {
        TEST_POOL.with(|c| c.set(Some(bytes)));
    }
    fn before_flip(&self, bytes: u64) {
        TEST_BEFORE_FLIP.with(|c| c.set(Some(bytes)));
    }
}
impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        TEST_HOST.with(|c| c.set(None));
        TEST_POOL.with(|c| c.set(None));
        TEST_BEFORE_FLIP.with(|c| c.set(None));
    }
}

/// Fixture rig backed by `admission-addons.json` (`mtp_on`).
struct AdmissionRig {
    p: Profile,
    base: ModelSize,
    max_ctx: usize,
    others: u64,
    h0: HostMemory,
    ws: u64,
    load: Pick,
    m: ModelSize,
}

impl AdmissionRig {
    fn mtp_on(others_gib: u64) -> Self {
        let f = fixture("admission-addons.json");
        let (inputs, v) = (&f["inputs"], &f["variants"]["mtp_on"]);
        let (p, b, others) = (
            profile(j_u64(&v["kv_per_ctx"])),
            base(inputs, v),
            others_gib * G,
        );
        let (h0, ws) = host(inputs, others);
        let max_ctx = j_usize(&inputs["max_context"]);
        let load = pick(
            &p,
            &b,
            max_ctx,
            &Limits::default(),
            &h0,
            &Own::default(),
            ws,
        )
        .unwrap();
        let m = ModelSize {
            ctx: load.ctx as u64,
            ..b
        };
        Self {
            p,
            base: b,
            max_ctx,
            others,
            h0,
            ws,
            load,
            m,
        }
    }
    fn admit(&self, h: &HostMemory, own: &Own) -> Result<Admission> {
        admit_with(
            &self.p,
            &self.m,
            self.load.chunk as u64,
            &Limits::default(),
            h,
            own,
            self.ws,
        )
    }
    fn readmit(&self, h: &HostMemory, own: &Own) -> Turn {
        let a = &self.load.admission;
        readmit_with(
            &self.p,
            a,
            &self.m,
            self.load.chunk as u64,
            &Limits::default(),
            h,
            own,
            self.ws,
        )
        .unwrap()
    }
    fn readmit_settled(&self, h: HostMemory, resident: u64, volatile: u64, own: &Own) -> Turn {
        self.readmit(&settled(h, Residency { resident, volatile }, true), own)
    }
}

/// Synthetic 3-layer K-quant file (1 class A + 2 class B, 32-byte aligned).
struct BankFiles {
    moe: MoeConfig,
    experts: Vec<Option<ExpertLayer>>,
    _tmp: Temp,
    path: PathBuf,
}

impl BankFiles {
    fn new(seed: u64) -> Self {
        let (moe, mut rng) = (store_moe(), Rng::new(seed));
        let mut file = rng.bytes(1024);
        let experts = STORE_CLASSES
            .iter()
            .enumerate()
            .map(|(il, &c)| Some(kquant_layer(&mut rng, &moe, il, c, 32, &mut file)))
            .collect();
        let tmp = Temp::new("store");
        let path = tmp.write("experts.bin", &file);
        Self {
            moe,
            experts,
            _tmp: tmp,
            path,
        }
    }
    fn pool(&self, threads: usize) -> IoPool {
        IoPool::new(
            Arc::new(ShardSet::open(&[self.path.as_path()]).unwrap()),
            threads,
        )
        .unwrap()
    }
    fn streamed(&self, cache: [usize; 2], staging: usize) -> KqBank {
        KqBank::new(self.pool(2), &self.moe, &self.experts, cache, staging).unwrap()
    }
    fn direct(&self) -> KqBank {
        let b = KqBank::new_direct(self.pool(2), &self.moe, &self.experts).unwrap();
        b.wait_direct_fill().unwrap();
        b
    }
    fn mmap(&self) -> KqBank {
        let ps = std::slice::from_ref(&self.path);
        KqBank::new_mmap(self.pool(2), &self.moe, &self.experts, ps, 0, false).unwrap()
    }
    fn classes(&self, bank: &KqBank) -> Classes {
        Classes {
            layers: [1, 2],
            slot: bank.slot_bytes(),
        }
    }
}

fn live_lease_rig(seed: u64) -> (BankFiles, KqBank, Lease, Classes, Admission, u64, u64, u64) {
    let s = 64 << 20;
    let classes = Classes {
        layers: [1, 2],
        slot: [s, s],
    };
    let (p, limits, ws, others) = (profile(100), Limits::default(), 96 * G, 10 * G);
    let size = ModelSize {
        dense: 0,
        slot: s,
        n_experts: 8,
        ctx: 1024,
    };
    let physical = limits.reserve + others + Terms::with(&p, &size, ws, 1024).fixed(1) + 64 * s;
    let loaded = admit_with(
        &p,
        &size,
        1024,
        &limits,
        &host_mem(physical, others),
        &Own::default(),
        ws,
    )
    .unwrap();
    assert_eq!(loaded.bound, Bound::Live);
    let files = BankFiles::new(seed);
    let bank = files.streamed(
        classes.split(loaded.cache_bytes()),
        loaded.staging_layers as usize,
    );
    let lease = Lease::new(p, size, 1024, classes, Some(limits), Some(loaded), &bank);
    (files, bank, lease, classes, loaded, physical, ws, others)
}

#[allow(clippy::too_many_arguments)]
fn synth_adm(
    physical: u64,
    others: u64,
    reserve: u64,
    live: u64,
    ws: u64,
    terms: Terms,
    staging_layers: u64,
    fixed: u64,
    slot: u64,
    cache_slots: u64,
) -> Admission {
    Admission {
        physical,
        host_used: others,
        own: 0,
        others,
        reserve,
        live,
        ws,
        terms,
        staging_layers,
        fixed,
        slot,
        cache_slots,
        bound: Bound::Live,
    }
}

/// Collapses 10–17 arg calls to `direct_gate`, `pick_direct`, `select_expert_mode*`, etc.
#[derive(Clone)]
struct ModeRig {
    sizes: Sizes,
    p: Profile,
    ctx: usize,
    ws: u64,
    dense_full: u64,
    dense_mmap: u64,
    direct_bytes: u64,
    mapped_bytes: u64,
    mmap_chunk_max: usize,
    metal: bool,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
    cache_cap: Option<u64>,
}

impl ModeRig {
    fn real() -> Self {
        let sizes = real_sizes();
        let dense_full = sizes.dense_trunk + sizes.dense_mtp;
        Self {
            direct_bytes: sizes.direct_expert_bytes(),
            sizes,
            p: profile(15_812),
            ctx: 262_144,
            ws: 96 * G,
            dense_full,
            dense_mmap: dense_full,
            mapped_bytes: 89_500_000_000,
            mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
            metal: true,
            reserve_override: None,
            max_memory: None,
            cache_cap: None,
        }
    }
    fn with_savings(mut self, saved: u64) -> Self {
        self.dense_mmap = self.dense_full - saved;
        self.mapped_bytes = 89_500_000_000 + saved;
        self
    }
    fn gate(&self, mode: DirectMode, h: &HostMemory, kv: u64, sc: u64) -> DirectDecision {
        direct_gate(
            mode,
            h,
            &Own::default(),
            self.dense_full,
            self.direct_bytes,
            kv,
            sc,
            DIRECT_MARGIN,
            self.max_memory,
            self.cache_cap,
        )
    }
    fn pick_direct(&self, mode: DirectMode, h: &HostMemory) -> (usize, DirectDecision) {
        pick_direct(
            mode,
            &self.p,
            self.ctx,
            h,
            &Own::default(),
            self.dense_full,
            self.direct_bytes,
            self.reserve_override,
            self.max_memory,
            self.cache_cap,
        )
    }
    fn select(
        &self,
        mmap: MmapMode,
        direct: DirectMode,
        h: &HostMemory,
    ) -> Result<ExpertResidency> {
        select_expert_mode(
            mmap,
            direct,
            self.metal,
            &self.p,
            self.ctx,
            h,
            &Own::default(),
            self.dense_full,
            self.direct_bytes,
            self.mapped_bytes,
            self.reserve_override,
            self.max_memory,
            self.cache_cap,
        )
    }
    fn select_for(
        &self,
        mmap: MmapMode,
        direct: DirectMode,
        h: &HostMemory,
    ) -> Result<ExpertResidency> {
        select_expert_mode_for(
            mmap,
            direct,
            self.metal,
            &self.p,
            self.ctx,
            h,
            &Own::default(),
            self.dense_full,
            self.dense_mmap,
            self.direct_bytes,
            self.mapped_bytes,
            self.mmap_chunk_max,
            self.reserve_override,
            self.max_memory,
            self.cache_cap,
        )
    }
    fn retry(
        &self,
        mmap: MmapMode,
        direct: DirectMode,
        read_host: impl FnMut() -> Result<(HostMemory, u64)>,
        pool_bytes: impl FnMut() -> u64,
        clear_cache: impl FnMut(),
    ) -> Result<(HostMemory, u64, ExpertResidency)> {
        select_expert_mode_with_retry(
            mmap,
            direct,
            self.metal,
            &self.p,
            self.ctx,
            &Own::default(),
            self.dense_full,
            self.dense_mmap,
            self.direct_bytes,
            self.mapped_bytes,
            self.mmap_chunk_max,
            self.reserve_override,
            self.max_memory,
            self.cache_cap,
            read_host,
            pool_bytes,
            clear_cache,
        )
    }
    fn warm(&self, mode: MmapMode, h: &HostMemory) -> u64 {
        let kv = direct_kv_allowance(&self.p, self.ctx);
        mmap_warm_budget(
            mode,
            h,
            self.dense_full,
            kv,
            DIRECT_MARGIN,
            self.mapped_bytes,
            self.max_memory,
        )
    }
}

/// Shared harness for idle-promotion tests.
struct PromoRig {
    files: BankFiles,
    bank: KqBank,
    reference: KqBank,
    classes: Classes,
    p: Profile,
    ctx: usize,
    dense: u64,
    physical: u64,
    ws: u64,
}

impl PromoRig {
    fn new(physical: u64, ws: u64) -> Self {
        let files = BankFiles::new(7);
        let (bank, reference) = (files.streamed([2, 2], 1), files.streamed([0, 0], 1));
        let classes = files.classes(&bank);
        Self {
            files,
            bank,
            reference,
            classes,
            p: profile(15_812),
            ctx: 262_144,
            dense: 10 * G,
            physical,
            ws,
        }
    }
    fn promo(&self, mode: DirectMode, mmap: MmapMode) -> DirectPromotion {
        DirectPromotion {
            mode,
            mmap,
            metal_available: true,
            physical: self.physical,
            reserve_override: None,
            n_experts: self.files.moe.n_experts,
        }
    }
    fn direct_bytes(&self) -> u64 {
        self.promo(DirectMode::Auto, MmapMode::Off)
            .direct_expert_bytes(&self.classes)
    }
    fn need_2k(&self) -> (u64, u64, u64) {
        let (kv, sc) = (
            direct_kv_allowance(&self.p, self.ctx),
            direct_scratch_bytes(&self.p, 2048, self.ctx),
        );
        (self.dense + self.direct_bytes() + kv + sc, kv, sc)
    }
    fn size(&self, n_experts: u64) -> ModelSize {
        ModelSize {
            dense: self.dense,
            slot: self.classes.slot[0].max(self.classes.slot[1]),
            n_experts,
            ctx: self.ctx as u64,
        }
    }
    fn lease(
        &self,
        mode: DirectMode,
        mmap: MmapMode,
        limits: Limits,
        adm: Option<Admission>,
        n_exp: u64,
    ) -> Lease {
        Lease::new(
            self.p,
            self.size(n_exp),
            2048,
            self.classes,
            Some(limits),
            adm,
            &self.bank,
        )
        .with_direct_promotion(self.promo(mode, mmap), None)
    }
}

#[test]
fn profile_is_the_fixture_profile() {
    let f = fixture("admission-addons.json");
    let pf = &f["profile"];
    let p = profile(j_u64(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(
        p.kv_base,
        j_u64(&pf["KV_BASE"]) + j_u64(&pf["kv_fixed_ladder"])
    );
    let got = [
        p.compute_base,
        p.compute_per_row,
        p.compute_fixed,
        p.compute_per_ctx,
        p.io_base,
        p.io_per_row,
        p.io_fixed,
        p.second_staging_min_cache,
    ];
    let want = [
        "COMPUTE_BASE",
        "COMPUTE_PER_ROW",
        "compute_fixed",
        "compute_per_ctx",
        "IO_BASE",
        "IO_PER_ROW",
        "io_fixed",
        "SECOND_STAGING_MIN_CACHE",
    ]
    .map(|k| j_u64(&pf[k]));
    assert_eq!(got, want);
    assert_eq!((p.cache_env, p.reserve_env), (CACHE_ENV, RESERVE_ENV));
}

#[test]
fn admission_matches_the_fixture_rows() {
    let f = fixture("admission-addons.json");
    let (inputs, mut n) = (&f["inputs"], 0);
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let p = profile(j_u64(&v["kv_per_ctx"]));
        for r in v["rows"].as_array().unwrap() {
            let m = ModelSize {
                ctx: j_u64(&r["ctx"]),
                ..base(inputs, v)
            };
            let (rows, others) = (j_u64(&r["chunk_rows"]), j_u64(&r["others_gib"]) * G);
            let (h, ws) = host(inputs, others);
            let case = format!("{name} ctx {} chunk {rows} others {others}", m.ctx);
            let t = Terms::with(&p, &m, ws, rows);
            let want = &r["terms"];
            let t_got = [
                t.dense,
                t.staging_layer,
                t.kv,
                t.compute,
                t.io,
                t.driver,
                t.pool,
            ];
            let t_want = [
                "dense",
                "staging_layer",
                "kv",
                "compute",
                "io",
                "driver",
                "pool",
            ]
            .map(|k| j_u64(&want[k]));
            assert_eq!(t_got, t_want, "{case}");
            let got = admit_with(&p, &m, rows, &Limits::default(), &h, &Own::default(), ws);
            match (got, r["refused"].as_bool().unwrap()) {
                (Err(e), true) => assert!(e.reason.contains("vs budget"), "{case}: {e}"),
                (Ok(a), false) => {
                    let a_got = [
                        a.live,
                        a.budget(),
                        a.staging_layers,
                        a.fixed,
                        a.cache_slots,
                        a.cache_bytes(),
                        a.wired_limit(),
                    ];
                    let a_want = [
                        "live",
                        "budget",
                        "staging_layers",
                        "fixed",
                        "cache_slots",
                        "cache_bytes",
                        "wired_limit",
                    ]
                    .map(|k| j_u64(&r[k]));
                    assert_eq!(a_got, a_want, "{case}");
                    let bound = match r["bound"].as_str().unwrap() {
                        "Live" => Bound::Live,
                        "WorkingSet" => Bound::WorkingSet,
                        b => panic!("{case}: bound {b}"),
                    };
                    assert_eq!(a.bound, bound, "{case}");
                }
                (a, refused) => panic!("{case}: got {a:?}, refused={refused}"),
            }
            n += 1;
        }
    }
    assert_eq!(n, 2 * 5 * 3 * 3);
}

#[test]
fn pick_matches_the_fixture_ladder() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let max_ctx = j_usize(&inputs["max_context"]);
    assert_eq!(max_ctx, crate::models::hy_v4_flash::DEFAULT_CONTEXT_CAP);
    assert_eq!(j_usize(&inputs["min_context"]), MIN_CONTEXT);
    assert_eq!(j_usizes(&inputs["chunk_rows"]), CHUNKS);
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let (p, b) = (profile(j_u64(&v["kv_per_ctx"])), base(inputs, v));
        for (others, want) in v["pick"].as_object().unwrap() {
            let (h, ws) = host(inputs, others.parse::<u64>().unwrap() * G);
            let case = format!("{name} others {others}");
            let got = pick(&p, &b, max_ctx, &Limits::default(), &h, &Own::default(), ws);
            if !want["admitted"].as_bool().unwrap() {
                assert!(
                    got.unwrap_err().reason.contains("refused at context 16384"),
                    "{case}"
                );
                continue;
            }
            let got = got.unwrap();
            let a = got.admission;
            assert_eq!(
                (got.ctx, got.chunk),
                (j_usize(&want["ctx"]), j_usize(&want["chunk_rows"])),
                "{case}"
            );
            assert_eq!(
                [a.cache_bytes(), a.staging_layers, a.wired_limit()],
                [
                    j_u64(&want["cache_bytes"]),
                    j_u64(&want["staging_layers"]),
                    j_u64(&want["wired_limit"])
                ],
                "{case}"
            );
            let m = ModelSize {
                ctx: got.ctx as u64,
                ..b
            };
            assert_eq!(a.terms, Terms::with(&p, &m, ws, got.chunk as u64));
        }
    }
}

#[test]
fn pick_below_the_floor() {
    let p = profile(15_812);
    let b = ModelSize {
        dense: 11_352_192_436,
        slot: 4_663_296,
        n_experts: 1024,
        ctx: 0,
    };
    let (h, ws) = (host_mem(128 * G, 20 * G), 96_000_000_000);
    let got = pick(&p, &b, 4096, &Limits::default(), &h, &Own::default(), ws).unwrap();
    assert_eq!((got.ctx, got.chunk), (4096, 4096));
    assert_eq!(
        got.admission.terms,
        Terms::with(&p, &ModelSize { ctx: 4096, ..b }, ws, 4096)
    );
    let busy = host_mem(128 * G, 100 * G);
    let e = pick(&p, &b, 8192, &Limits::default(), &busy, &Own::default(), ws)
        .unwrap_err()
        .reason;
    assert!(e.contains("refused at context 8192"), "{e}");
}

#[test]
fn readmission_counts_the_session_kv_as_own() {
    let rig = AdmissionRig::mtp_on(50);
    assert_eq!(
        (rig.load.ctx, rig.load.admission.bound),
        (rig.max_ctx, Bound::Live)
    );
    let kv = rig.p.kv_per_ctx * rig.load.ctx as u64;
    assert!(kv > KEEP_SLACK);
    let h1 = HostMemory {
        wired: rig.others + rig.base.dense + kv,
        ..rig.h0
    };
    assert_eq!(
        rig.readmit(&h1, &own_of(rig.base.dense, kv, 0, true, 0, 0)),
        Turn::Keep
    );
    assert!(matches!(
        rig.readmit(&h1, &own_of(rig.base.dense, 0, 0, true, 0, 0)),
        Turn::StagingOnly(_)
    ));
}

#[test]
fn readmission_counts_the_free_pool_up_to_its_term() {
    let rig = AdmissionRig::mtp_on(50);
    let a = rig.load.admission;
    assert_eq!((a.bound, a.terms.pool), (Bound::Live, KEEP_SLACK));
    let (term, res, d) = (a.terms.pool, a.cache_bytes(), rig.base.dense);
    let turn = |in_use: u64, own: Own| {
        rig.readmit(
            &HostMemory {
                wired: rig.others + d + res + in_use,
                ..rig.h0
            },
            &own,
        )
    };
    assert_eq!(
        turn(term + G, own_of(d, 0, res, true, term, term)),
        Turn::Keep
    );
    assert!(matches!(
        turn(term + G, own_of(d, 0, res, true, term, 0)),
        Turn::StagingOnly(_)
    ));
    assert!(matches!(
        turn(term + 3 * G, own_of(d, 0, res, true, term + 3 * G, term)),
        Turn::StagingOnly(_)
    ));
    assert!(matches!(
        turn(3 * G, own_of(d, 0, res, true, 0, term)),
        Turn::StagingOnly(_)
    ));
    assert_eq!(
        own_of(d, 0, res, true, term + 3 * G, term).bytes(),
        d + res + term
    );
    assert_eq!(own_of(d, 0, res, false, 0, term).volatile, res);
}

#[test]
fn readmission_settles_the_bank_pages_still_counted_purgeable() {
    let gb = |x: f64| (x * 1e9) as u64;
    let hm = |w, a, p| HostMemory {
        physical: gb(137.44),
        free: 0,
        file: 0,
        anonymous: gb(a),
        purgeable: gb(p),
        wired: gb(w),
        compressor: gb(5.17),
    };
    let res = gb(54.94);
    let (held, listed) = (
        Residency {
            resident: res,
            volatile: 0,
        },
        Residency {
            resident: res,
            volatile: res,
        },
    );
    let o = own_of(gb(11.35), gb(3.83), res, true, 0, 0);
    let others = |h, bank| {
        settled(h, bank, true)
            .non_reclaimable()
            .saturating_sub(o.bytes()) as f64
            / 1e9
    };
    let normal = others(hm(79.75, 20.22, 0.47), held);
    for h in [hm(79.66, 20.15, 55.47), hm(8.94, 90.61, 55.47)] {
        assert!(others(h, held) < normal - 15.0 && (others(h, listed) - normal).abs() < 1.0);
    }
    assert!((others(hm(79.75, 70.22, 50.47), held) - normal).abs() < 0.01);
    let h = hm(79.66, 20.15, 55.47);
    assert_eq!(settled(h, listed, false), h);

    let rig = AdmissionRig::mtp_on(50);
    let res = rig.load.admission.cache_bytes();
    assert!(res > GROW_MIN);
    let lagging = HostMemory {
        wired: rig.others - 20 * G + rig.base.dense,
        anonymous: 20 * G + res,
        purgeable: res,
        ..rig.h0
    };
    let o = own_of(rig.base.dense, 0, res, true, 0, 0);
    let (t0, tr) = (
        rig.readmit_settled(lagging, res, 0, &o),
        rig.readmit_settled(lagging, res, res, &o),
    );
    assert!(matches!(t0, Turn::Rebuild(_)) && tr == Turn::Keep);
    assert_eq!(wary_growth(t0, || Ok(tr)), Turn::Keep);

    let roomy = HostMemory {
        wired: rig.others - 30 * G + rig.base.dense + res,
        anonymous: 20 * G,
        purgeable: G,
        ..rig.h0
    };
    let Turn::Rebuild(wary) = rig.readmit_settled(roomy, res, res, &o) else {
        panic!("no room");
    };
    let r_raw = rig.readmit(&roomy, &o);
    assert!(matches!(r_raw, Turn::Rebuild(_)));
    assert_eq!(
        wary_growth(r_raw, || Ok(rig.readmit_settled(roomy, res, res, &o))),
        Turn::Rebuild(wary)
    );
    assert_eq!(
        wary_growth(Turn::Keep, || panic!("no wary needed")),
        Turn::Keep
    );
}

#[test]
fn next_bank_shrinks_to_what_the_host_admits() {
    let a = AdmissionRig::mtp_on(50).load.admission;
    assert!(a.cache_slots > 0);
    let smaller = Admission {
        cache_slots: a.cache_slots / 2,
        ..a
    };
    let staging = Admission {
        staging_layers: 1,
        fixed: a.terms.fixed(1),
        cache_slots: 0,
        ..a
    };
    let never = || -> Result<Admission> { panic!("no reading needed") };
    for in_turn in [false, true] {
        assert_eq!(next_bank(Turn::Keep, &a, false, in_turn, never), None);
        assert_eq!(next_bank(Turn::Keep, &a, true, in_turn, never), Some(a));
        assert_eq!(
            next_bank(Turn::Rebuild(a), &staging, false, in_turn, never),
            Some(a)
        );
        assert_eq!(
            next_bank(Turn::Rebuild(a), &smaller, true, in_turn, never),
            Some(a)
        );
        assert_eq!(
            next_bank(Turn::Rebuild(a), &smaller, false, in_turn, never),
            (!in_turn).then_some(a)
        );
        for lost in [false, true] {
            assert_eq!(
                next_bank(Turn::StagingOnly(staging), &a, lost, in_turn, || Ok(
                    smaller
                )),
                Some(smaller)
            );
            assert_eq!(
                next_bank(Turn::StagingOnly(staging), &a, lost, in_turn, || Ok(
                    staging
                )),
                Some(staging)
            );
        }
    }
}

#[test]
fn next_bank_runs_staging_only_when_the_wary_reading_refuses() {
    let rig = AdmissionRig::mtp_on(50);
    let a = rig.load.admission;
    let res = a.cache_bytes();
    let o = own_of(rig.base.dense, 0, res, true, 0, 0);
    let h = HostMemory {
        wired: rig.h0.physical - Limits::default().reserve - (a.terms.fixed(1) + G)
            + rig.base.dense,
        anonymous: 2 * res,
        purgeable: res,
        ..rig.h0
    };
    let all = Residency {
        resident: res,
        volatile: res,
    };
    let host_turn = rig.readmit(&h, &o);
    let Turn::StagingOnly(s) = host_turn else {
        panic!("{host_turn:?}")
    };
    assert_eq!((s.cache_slots, s.staging_layers), (0, 1));
    let wary = settled(h, all, true);
    assert!(
        rig.admit(&wary, &o)
            .unwrap_err()
            .reason
            .contains("vs budget")
    );
    for in_turn in [false, true] {
        assert_eq!(
            next_bank(host_turn, &a, false, in_turn, || rig.admit(&wary, &o)),
            Some(s)
        );
    }
}

#[test]
fn flip_lag_reads_the_bank_left_in_the_count() {
    let gb = |x: f64| (x * 1e9) as u64;
    let (bank, before) = (gb(54.94), gb(0.47) + gb(54.94));
    for (pre, post, vol, want) in [
        (before, gb(55.47), bank, bank),
        (before, gb(0.47), bank, 0),
        (before, gb(0.40), bank, 0),
        (before, gb(0.60), bank, gb(0.60) - gb(0.47)),
        (before, gb(30.47), bank, gb(30.47) - gb(0.47)),
        (gb(10.0), gb(55.47), bank, 0),
        (gb(0.47), gb(0.47), 0, 0),
    ] {
        assert_eq!(flip_lag(pre, post, vol), want);
    }
}

#[test]
fn turn_start_shrink_reads_the_flip_lag() {
    let rig = AdmissionRig::mtp_on(50);
    let res = rig.load.admission.cache_bytes();
    assert!(rig.load.admission.bound == Bound::Live && res > 2 * KEEP_SLACK);
    let o = own_of(rig.base.dense, 0, res, true, 0, 0);
    for grown in [0, KEEP_SLACK + 2 * G] {
        let h = HostMemory {
            wired: rig.others + grown - 20 * G + rig.base.dense,
            anonymous: 20 * G + res,
            purgeable: res,
            ..rig.h0
        };
        let lag = flip_lag(res, h.purgeable, res);
        assert_eq!(lag, res);
        assert!(!matches!(
            rig.readmit_settled(h, res, 0, &o),
            Turn::StagingOnly(_)
        ));
        assert_eq!(
            matches!(rig.readmit_settled(h, res, lag, &o), Turn::StagingOnly(_)),
            grown > 0
        );
    }
    let foreign = 3 * KEEP_SLACK;
    let h = HostMemory {
        wired: rig.others - 20 * G + rig.base.dense,
        anonymous: 20 * G + foreign + res,
        purgeable: foreign,
        ..rig.h0
    };
    assert_eq!(flip_lag(foreign + res, h.purgeable, res), 0);
    assert_eq!(rig.readmit_settled(h, res, 0, &o), Turn::Keep);
    assert!(matches!(
        rig.readmit_settled(h, res, res, &o),
        Turn::StagingOnly(_)
    ));
}

#[test]
fn staging_charge_is_two_full_layers() {
    let f = fixture("admission-addons.json");
    let (inputs, v) = (&f["inputs"], &f["variants"]["mtp_on"]);
    let m = ModelSize {
        ctx: 262_144,
        ..base(inputs, v)
    };
    let t = Terms::with(
        &profile(j_u64(&v["kv_per_ctx"])),
        &m,
        j_u64(&inputs["ws"]),
        8192,
    );
    assert_eq!(t.staging_layer * 2, 9_550_430_208);
    let cls = fixture("expert-classes.json");
    let a = j_u64(&cls["slots"]["A"]["mlx_slot_bytes"]);
    let b = j_u64(&cls["slots"]["B"]["mlx_slot_bytes"]);
    assert_eq!(2 * 512 * (a + b), 8_965_324_800);
    assert!(t.staging_layer * 2 >= 2 * 512 * (a + b));
}

#[test]
fn qmv_batch_limit_follows_mlx() {
    for (arch, want) in [
        ("applegpu_g17s", [33, 25, 13]),
        ("applegpu_g17g", [33, 25, 13]),
        ("applegpu_g17d", [32, 18, 12]),
        ("applegpu_g16s", [13, 15, 13]),
        ("applegpu_g15s", [13, 15, 13]),
        ("applegpu_g15d", [32, 18, 12]),
        ("applegpu_g14s", [14, 10, 6]),
        ("applegpu_g13s", [14, 10, 7]),
        ("applegpu_g13g", [14, 10, 6]),
        ("applegpu_g13d", [32, 18, 12]),
        ("applegpu_g12s", [18, 12, 10]),
        ("", [18, 12, 10]),
    ] {
        let got =
            [(2048, 2048), (4096, 1024), (3072, 8192)].map(|(k, n)| qmv_batch_limit(arch, k, n));
        assert_eq!(got, want, "{arch}");
    }
    assert_eq!(qmv_batch_limit("applegpu_g17s", 2049, 16), 25);
    assert_eq!(qmv_batch_limit("applegpu_g17s", 16, 4097), 13);
}

#[test]
fn row_cap_is_one_below_the_smallest_limit() {
    let shapes = [
        (192, 512),
        (512, 256),
        (3072, 576),
        (3072, 8192),
        (1536, 15_360),
    ];
    for (arch, want) in [
        ("applegpu_g17s", 12),
        ("applegpu_g16s", 12),
        ("applegpu_g15s", 12),
        ("applegpu_g17d", 11),
        ("applegpu_g15d", 11),
        ("applegpu_g14s", 5),
        ("applegpu_g13s", 6),
    ] {
        assert_eq!(row_cap(arch, None, &shapes).unwrap(), want, "{arch}");
    }
    assert_eq!(row_cap("applegpu_g17s", Some(4), &shapes).unwrap(), 3);
    assert_eq!(row_cap("applegpu_g17s", Some(1), &shapes).unwrap(), 0);
    assert!(row_cap("applegpu_g17s", None, &[]).is_err());
}

#[test]
fn splitk_env_reads_like_atoi() {
    for (raw, want) in [
        (None, None),
        (Some("8"), Some(8)),
        (Some("  12x"), Some(12)),
        (Some("+3"), Some(3)),
        (Some("-3"), None),
        (Some("0"), None),
        (Some("abc"), None),
        (Some(""), None),
        (Some("99999999999"), None),
    ] {
        assert_eq!(splitk_min_m(raw), want, "{raw:?}");
    }
}

#[test]
fn lanes_charge_their_floor() {
    let cfg = tiny_cfg();
    let floor = lane_floor(&cfg, false);
    assert_eq!(floor, 2 * 11 * 24 * 2 + 2 * 6 * 16 * 4);
    assert_eq!(lane_floor(&cfg, true) - floor, SEEDS as u64 * 32 * 4);
    let p = profile(15_812);
    assert_eq!(
        (lane_profile(&p, 1, floor), lane_profile(&p, 0, floor)),
        (p, p)
    );
    let four = lane_profile(&p, 4, floor);
    assert_eq!(four.kv_base, p.kv_base + 3 * floor);
    assert_eq!(
        Profile {
            kv_base: p.kv_base,
            ..four
        },
        p
    );
}

#[test]
fn lane_bytes_price_the_draft_pooled_keys() {
    let cfg = tiny_cfg();
    for ctx in [0, 64, 262_144] {
        assert_eq!(lane_bytes(&cfg, false, ctx), lane_floor(&cfg, false));
        assert_eq!(
            lane_bytes(&cfg, true, ctx),
            lane_floor(&cfg, true) + (ctx / 4) as u64 * 32
        );
    }
}

#[test]
fn plan_lanes_steps_down_to_what_fits() {
    let fits = |max: usize| {
        move |n: usize| {
            if n <= max {
                Ok(n * 10)
            } else {
                Err(Error::from_reason(format!("{n} lanes")))
            }
        }
    };
    assert_eq!(
        (
            plan_lanes(4, fits(4)).unwrap(),
            plan_lanes(4, fits(2)).unwrap()
        ),
        ((4, 40), (2, 20))
    );
    assert_eq!(plan_lanes(0, fits(4)).unwrap(), (1, 10));
    assert_eq!(plan_lanes(3, fits(0)).unwrap_err().reason, "1 lanes");
}

#[test]
fn lanes_keep_the_one_lane_window_and_chunk() {
    let rig = AdmissionRig::mtp_on(50);
    let (l, own, one, mut shrunk) = (Limits::default(), Own::default(), rig.load, 0);
    for quarter in 1..=160u64 {
        let floor = quarter * G / 4;
        let admit = |n: usize| {
            let lp = lane_profile(&rig.p, n, floor);
            admit_with(&lp, &rig.m, one.chunk as u64, &l, &rig.h0, &own, rig.ws)
        };
        let (lanes, got) = lanes_at(one, LANES_MAX, admit);
        let most = (1..=LANES_MAX)
            .rev()
            .find(|&n| n == 1 || admit(n).is_ok())
            .unwrap();
        assert_eq!(
            (got.ctx, got.chunk, lanes),
            (one.ctx, one.chunk, most),
            "floor {floor}"
        );
        assert_eq!(
            got.admission,
            if lanes > 1 {
                admit(lanes).unwrap()
            } else {
                one.admission
            }
        );
        let lp_max = lane_profile(&rig.p, LANES_MAX, floor);
        if let Ok(lad) = pick(&lp_max, &rig.base, rig.max_ctx, &l, &rig.h0, &own, rig.ws)
            && (lad.ctx, lad.chunk) != (one.ctx, one.chunk)
        {
            shrunk += 1;
        }
    }
    assert!(shrunk > 0);
}

#[test]
fn cache_split_follows_the_layer_counts() {
    let c = Classes {
        layers: [19, 22],
        slot: [4_663_296, 4_091_904],
    };
    for bytes in [0, 4_663_296, 17 * G, 48 * G + 12_345, 56_299_972_608] {
        let [a, b] = c.split(bytes);
        let used = a as u64 * c.slot[0] + b as u64 * c.slot[1];
        assert!(
            used <= bytes && bytes - used < c.slot[1],
            "{bytes}: {a} {b}"
        );
        if bytes >= 17 * G {
            assert!((a as f64 / 19.0 - b as f64 / 22.0).abs() < 1.0);
        }
    }
    assert_eq!(
        Classes {
            layers: [0, 22],
            ..c
        }
        .split(10 * c.slot[1] + 1),
        [0, 10]
    );
    assert_eq!(
        Classes {
            layers: [19, 0],
            ..c
        }
        .split(10 * c.slot[0] + 1),
        [10, 0]
    );
    assert_eq!(
        Classes {
            layers: [0, 0],
            ..c
        }
        .split(G),
        [0, 0]
    );
}

#[test]
fn chunk_env_values() {
    for (raw, want) in [
        (None, None),
        (Some(" "), None),
        (Some("4096"), Some(4096)),
        (Some(" 256 "), Some(256)),
    ] {
        assert_eq!(chunk_cap(raw).unwrap(), want, "{raw:?}");
    }
    for bad in ["0", "128", "1000", "-256", "abc", "4096.0"] {
        assert!(
            chunk_cap(Some(bad)).unwrap_err().reason.contains(CHUNK_ENV),
            "{bad}"
        );
    }
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_sizes_match_the_oracle() {
    let s = real_scan();
    let sizes = Sizes::of(&s).unwrap();
    let addons = fixture("admission-addons.json");
    let inputs = &addons["inputs"];
    assert_eq!(sizes.dense_trunk, j_u64(&inputs["dense_trunk"]));
    let mtp = fixture("mtp-tensors.json");
    let dh = crate::models::hy_v4_flash::mtp::draft_head_bytes(&s).unwrap();
    assert_eq!(sizes.dense_mtp, j_u64(&mtp["totals"]["plan_bytes"]) + dh);
    let cls = fixture("expert-classes.json");
    assert_eq!(
        sizes.classes.slot,
        [
            j_u64(&cls["slots"]["A"]["mlx_slot_bytes"]),
            j_u64(&cls["slots"]["B"]["mlx_slot_bytes"])
        ]
    );
    assert_eq!(sizes.classes.layers, [19, 22]);
    let m = sizes.model(true, 262_144);
    assert_eq!(
        (m.dense, m.slot),
        (sizes.dense_trunk + sizes.dense_mtp, j_u64(&inputs["slot"]))
    );
    assert_eq!(
        (m.n_experts, sizes.model(false, 1).dense),
        (j_u64(&inputs["n_experts"]), sizes.dense_trunk)
    );
    let pf = &addons["profile"];
    assert_eq!(kv_per_token(&s.cfg, true), j_u64(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(
        kv_per_token(&s.cfg, false),
        j_u64(&pf["kv_per_ctx_mtp_off"])
    );
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_lane_floor_and_wave_shapes() {
    let s = real_scan();
    assert_eq!(
        lane_floor(&s.cfg, true),
        30 * 515 * 576 * 2 + 13 * 6 * 128 * 4 + 8 * 3072 * 4
    );
    assert_eq!(
        lane_floor(&s.cfg, false),
        30 * 515 * 576 * 2 + 12 * 6 * 128 * 4
    );
    let shapes = q8_shapes(&s, true).unwrap();
    for want in [
        (192, 512),
        (512, 256),
        (3072, 576),
        (3072, 16_384),
        (1536, 15_360),
        (13_824, 3072),
        (6144, 3072),
    ] {
        assert!(shapes.contains(&want), "{want:?}");
    }
    assert!(!q8_shapes(&s, false).unwrap().contains(&(6144, 3072)));
    assert_eq!(row_cap("applegpu_g17s", None, &shapes).unwrap(), 12);
    assert_eq!(row_cap("applegpu_g15d", None, &shapes).unwrap(), 11);
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_store_admits_at_load() {
    let s = real_scan();
    let inputs = fixture("admission-addons.json")["inputs"].clone();
    let env = TestEnvGuard::new();
    let (h, ws) = host(&inputs, 20 * G);
    env.host(h, ws);
    let opts = Options {
        max_context: None,
        mtp: true,
        chunk_cap: None,
        lanes: LANES_MAX,
        plan: CachePlan::AdmitWith(Limits::default()),
        max_memory: None,
        direct: DirectMode::Off,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let st = Store::open(&s.paths, &opts).unwrap();
    drop(env);
    assert_eq!((st.ctx, st.chunk, st.lanes), (262_144, 8192, LANES_MAX));
    assert!(st.row_cap >= 2);
    let a = st.admission().unwrap();
    assert_eq!(st.cache_slots(), st.sizes.classes.split(a.cache_bytes()));
    assert_eq!(
        (st.bank.staging_layers() as u64, st.dense_bytes),
        (a.staging_layers, st.sizes.dense_trunk)
    );
    assert!(st.model.new_session(st.ctx).is_ok());
}

#[test]
fn lease_rebuilds_keeps_and_refuses() {
    let files = BankFiles::new(1);
    let bank = files.streamed([0, 0], 1);
    let (classes, p) = (files.classes(&bank), profile(100));
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: 1024,
    };
    let limits = Limits {
        cache_cap: Some(64 * size.slot),
        ..Limits::default()
    };
    let (roomy_h, ws) = (host_mem(64 * G, 10 * G), 48 * G);
    let loaded = admit_with(&p, &size, 1024, &limits, &roomy_h, &Own::default(), ws).unwrap();
    assert_eq!((loaded.cache_slots, loaded.bound), (64, Bound::Env));
    let staging = Admission {
        staging_layers: 1,
        fixed: loaded.terms.fixed(1),
        cache_slots: 0,
        ..loaded
    };
    let lease = Lease::new(p, size, 1024, classes, Some(limits), Some(staging), &bank);
    bank.end_turn().unwrap();

    let env = TestEnvGuard::new();
    env.host(roomy_h, ws);
    lease.begin_turn(&bank, 0, 0).unwrap();
    let a = lease.admission().unwrap();
    assert!(
        lease.in_turn() && bank.in_turn() && a.cache_slots == 64 && lease.switches().len() == 1
    );
    assert_eq!(
        (bank.cache_slots(), bank.staging_layers() as u64),
        (classes.split(a.cache_bytes()), a.staging_layers)
    );
    lease.end_turn(&bank).unwrap();
    assert!(!lease.in_turn() && !bank.in_turn());

    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!((lease.admission().unwrap(), lease.switches().len()), (a, 1));
    lease.end_turn(&bank).unwrap();

    env.host(host_mem(64 * G, 60 * G), ws);
    assert!(
        lease
            .begin_turn(&bank, 0, 0)
            .unwrap_err()
            .reason
            .contains("vs budget")
    );
    assert!(!lease.in_turn() && !bank.in_turn() && lease.admission().unwrap() == a);
}

#[test]
fn lease_recheck_keeps_for_the_pool_and_shrinks_for_others() {
    let (_files, bank, lease, classes, loaded, physical, ws, others) = live_lease_rig(2);
    assert!(loaded.cache_slots > 48);
    bank.end_turn().unwrap();
    let r = bank.residency().unwrap();
    assert_eq!(r.volatile, r.resident);

    let env = TestEnvGuard::new();
    env.host(host_mem(physical, others), ws);
    env.pool(0);
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(
        (lease.admission().unwrap(), lease.switches().len()),
        (loaded, 0)
    );
    let r = bank.residency().unwrap();
    assert_eq!(r.volatile, 0);

    let pool = loaded.terms.pool;
    env.host(host_mem(physical, others + r.resident + pool + G), ws);
    env.pool(pool);
    assert!(!lease.recheck(&bank, 0, 0).unwrap());

    env.pool(0);
    env.host(host_mem(physical, r.resident), ws);
    assert!(!lease.recheck(&bank, 0, 0).unwrap());
    assert_eq!(
        (lease.admission().unwrap(), lease.switches().len()),
        (loaded, 0)
    );

    env.host(host_mem(physical, others + r.resident + pool + G), ws);
    assert!(lease.recheck(&bank, 0, 0).unwrap());
    let a = lease.admission().unwrap();
    assert!(a.cache_slots > 0 && a.cache_slots < loaded.cache_slots);
    assert_eq!(
        (bank.cache_slots(), lease.switches().len()),
        (classes.split(a.cache_bytes()), 1)
    );
    assert!(lease.in_turn() && bank.in_turn());
    lease.end_turn(&bank).unwrap();
}

#[test]
fn lease_turn_start_takes_the_flip_lag_out() {
    let (_files, bank, lease, _, loaded, physical, ws, _) = live_lease_rig(3);
    bank.touch();
    bank.end_turn().unwrap();
    let r = bank.residency().unwrap();
    assert!(r.resident > 0 && r.volatile == r.resident);

    let edge =
        physical - Limits::default().reserve + KEEP_SLACK - (loaded.fixed + loaded.cache_bytes());
    let h = HostMemory {
        physical,
        wired: edge + 1,
        anonymous: r.resident,
        purgeable: r.resident,
        ..Default::default()
    };
    let env = TestEnvGuard::new();
    env.host(h, ws);
    env.pool(0);
    env.before_flip(2 * r.resident);
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(
        (lease.admission().unwrap(), lease.switches().len()),
        (loaded, 0)
    );
    lease.end_turn(&bank).unwrap();
    assert_eq!(bank.residency().unwrap(), r);

    env.before_flip(r.resident);
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(
        lease.admission().unwrap().cache_slots < loaded.cache_slots && lease.switches().len() == 1
    );
    lease.end_turn(&bank).unwrap();
}

#[test]
fn lease_without_admission_only_flips_the_bank() {
    let files = BankFiles::new(4);
    let bank = files.streamed([2, 2], 1);
    let classes = files.classes(&bank);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0],
        n_experts: 8,
        ctx: 64,
    };
    let lease = Lease::new(profile(100), size, 64, classes, None, None, &bank);
    bank.end_turn().unwrap();
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(bank.in_turn() && !lease.in_turn() && bank.cache_slots() == [2, 2]);
    lease.end_turn(&bank).unwrap();
    assert!(!bank.in_turn() && lease.switches().is_empty());
}

#[test]
fn max_memory_caps_the_working_set_only_below_it() {
    assert_eq!(capped_ws(100 * G, None), 100 * G);
    assert_eq!(capped_ws(100 * G, Some(64 * G)), 64 * G);
    assert_eq!(capped_ws(100 * G, Some(200 * G)), 100 * G);
}

#[test]
fn direct_env_values() {
    use DirectMode as D;
    for (raw, want) in [
        (None, D::Auto),
        (Some(""), D::Auto),
        (Some("  "), D::Auto),
        (Some("auto"), D::Auto),
        (Some("AUTO"), D::Auto),
        (Some("1"), D::On),
        (Some("true"), D::On),
        (Some("TRUE"), D::On),
        (Some("0"), D::Off),
        (Some("false"), D::Off),
        (Some("FALSE"), D::Off),
    ] {
        assert_eq!(direct_mode(raw).unwrap(), want, "{raw:?}");
    }
    for bad in ["2", "yes", "no", "maybe"] {
        assert!(
            direct_mode(Some(bad))
                .unwrap_err()
                .reason
                .contains(DIRECT_ENV),
            "{bad}"
        );
    }
}

#[test]
fn direct_gate_admits_quiet_and_falls_back_when_busy_or_disabled() {
    use DirectMode as D;
    let mut rig = ModeRig::real();
    assert_eq!(
        (rig.direct_bytes, rig.sizes.total_experts()),
        (91_455_750_144, [9_728, 11_264])
    );

    let kv = rig.p.kv_base + rig.p.kv_per_ctx * rig.ctx as u64;
    let sc = direct_scratch_bytes(&rig.p, CHUNKS[0], rig.ctx);
    let (quiet, busy) = (host_mem(128 * G, 20 * G), host_mem(128 * G, 45 * G));

    let d_auto = rig.gate(D::Auto, &quiet, kv, sc);
    assert!(d_auto.admitted && d_auto.available == 128 * G - 20 * G - DIRECT_MARGIN);
    assert_eq!(
        (d_auto.margin, d_auto.scratch_bytes, d_auto.kv_and_scratch),
        (DIRECT_MARGIN, sc, kv + sc)
    );

    for (mode, h, max_m, cap, want) in [
        (D::On, quiet, None, None, true),
        (D::Off, quiet, None, None, false),
        (D::Auto, busy, None, None, false),
        (D::On, busy, None, None, false),
        (D::Auto, quiet, Some(96 * G), None, false),
        (D::Auto, quiet, None, Some(rig.direct_bytes - 1), false),
        (D::Auto, quiet, None, Some(rig.direct_bytes), true),
    ] {
        rig.max_memory = max_m;
        rig.cache_cap = cap;
        assert_eq!(
            rig.gate(mode, &h, kv, sc).admitted,
            want,
            "{mode:?} {max_m:?} {cap:?}"
        );
    }

    let ws = 96_000_000_000;
    assert_eq!(
        (
            direct_wired_limit(d_auto.need, ws),
            direct_wired_limit(50 * G, ws)
        ),
        (ws, 50 * G)
    );
}

#[test]
fn direct_gate_respects_explicit_reserve_override_and_steps_chunk_down() {
    for (raw, want) in [(None, None), (Some("  "), None), (Some("16"), Some(16 * G))] {
        assert_eq!(parse_reserve_override(raw), want);
    }
    assert_eq!(
        [
            direct_margin(None),
            direct_margin(Some(2 * G)),
            direct_margin(Some(16 * G))
        ],
        [DIRECT_MARGIN, DIRECT_MARGIN, 16 * G]
    );

    let mut rig = ModeRig::real();
    let kv_64k = direct_kv_allowance(&rig.p, rig.ctx);
    assert_eq!(
        kv_64k,
        rig.p.kv_base + rig.p.kv_per_ctx * DIRECT_KV_ALLOWANCE_TOKENS as u64
    );
    assert_eq!(
        direct_kv_allowance(&rig.p, 32_768),
        rig.p.kv_base + rig.p.kv_per_ctx * 32_768
    );

    let [s8, s4, s2] = [8192, 4096, 2048].map(|c| direct_scratch_bytes(&rig.p, c, rig.ctx));
    assert!(s8 > s4 && s4 > s2);
    let base_need = rig.dense_full + rig.direct_bytes + kv_64k;
    let [need_8k, need_4k, need_2k] = [s8, s4, s2].map(|s| base_need + s);

    // MTP-1 fits at 26 GiB others; AR fits at 28 GiB others.
    let (mc, mg) = rig.pick_direct(DirectMode::Auto, &host_mem(128 * G, 26 * G));
    assert!(mg.admitted && mc == 2048 && mg.kv_and_scratch == kv_64k + s2);

    let mut ar = rig.clone();
    ar.p = profile(14_596);
    ar.dense_full = ar.sizes.dense_trunk;
    let (ac, ag) = ar.pick_direct(DirectMode::Auto, &host_mem(128 * G, 28 * G));
    assert!(ag.admitted && ac == 2048);

    let quiet = host_mem(128 * G, 20 * G);
    let (chunk, gate) = rig.pick_direct(DirectMode::Auto, &quiet);
    assert!(gate.admitted && chunk == 8192 && gate.margin == DIRECT_MARGIN);
    assert_eq!((gate.scratch_bytes, gate.kv_and_scratch), (s8, kv_64k + s8));

    // Explicit reserve override exact boundary forces streamed fallback.
    let exact_res = quiet.physical - quiet.wired - need_2k;
    rig.reserve_override = Some(exact_res);
    let (c_ok, g_ok) = rig.pick_direct(DirectMode::Auto, &quiet);
    assert!(g_ok.admitted && c_ok == 2048 && g_ok.margin == exact_res.max(DIRECT_MARGIN));
    rig.reserve_override = Some(exact_res + 1);
    let (_, g_no) = rig.pick_direct(DirectMode::Auto, &quiet);
    assert!(!g_no.admitted && g_no.margin == (exact_res + 1).max(DIRECT_MARGIN));
    rig.reserve_override = None;

    for (cap, ok, wc, ws) in [
        (need_8k, true, 8192usize, s8),
        (need_8k - 1, true, 4096, s4),
        (need_4k, true, 4096, s4),
        (need_4k - 1, true, 2048, s2),
        (need_2k, true, 2048, s2),
        (need_2k - 1, false, 2048, s2),
    ] {
        rig.max_memory = Some(cap);
        let (gc, gg) = rig.pick_direct(DirectMode::Auto, &quiet);
        assert_eq!(
            (gg.admitted, gc, gg.scratch_bytes),
            (ok, wc, ws),
            "cap={cap}"
        );
    }
}

#[test]
fn lease_end_turn_idle_promotes_partial_to_direct_when_others_drop() {
    let rig = PromoRig::new(40 * G, 96 * G);
    let (kv_live, limits) = (2 * G, Limits::default());
    let (need_2k, kv_64k, sc_2k) = rig.need_2k();
    let size = rig.size(8);
    let adm = synth_adm(
        rig.physical,
        rig.physical - need_2k - DIRECT_MARGIN + 2 * G,
        limits.reserve,
        20 * G,
        rig.ws,
        Terms::with(&rig.p, &size, rig.ws, 2048),
        1,
        rig.dense + kv_64k + sc_2k,
        size.slot,
        4,
    );
    let lease = rig.lease(DirectMode::Auto, MmapMode::Off, limits, Some(adm), 8);
    let (env, r) = (TestEnvGuard::new(), rig.bank.residency().unwrap());

    // Turn 1: others above threshold -> no promotion.
    let high = rig.physical - need_2k - DIRECT_MARGIN + G;
    env.host(
        host_mem(rig.physical, high + rig.dense + r.resident + kv_live),
        rig.ws,
    );
    assert_eq!(
        lease.end_turn_idle(&rig.bank, rig.dense, kv_live).unwrap(),
        None
    );
    assert!(!rig.bank.is_direct() && lease.switches().is_empty());

    // Turn 2: others drop -> promotes to Direct.
    env.host(
        host_mem(rig.physical, 2 * G + rig.dense + r.resident + kv_live),
        rig.ws,
    );
    assert!(
        lease
            .end_turn_idle(&rig.bank, rig.dense, kv_live)
            .unwrap()
            .is_some()
    );
    assert!(rig.bank.is_direct() && !rig.bank.in_turn() && lease.direct_budget().is_some());
    assert_eq!(
        (
            lease.switches().len(),
            lease.admission().unwrap().staging_layers
        ),
        (1, 0)
    );
    rig.bank.wait_direct_fill().unwrap();
    assert!(rig.bank.direct_fill_complete());

    // Promoted direct bank serves identical expert bytes across every layer.
    let ne = rig.files.moe.n_experts;
    let all: Vec<u32> = (0..ne as u32).collect();
    for il in 0..3 {
        let dp = rig.bank.prepare(il, &all, &[], true).unwrap();
        let rp = rig.reference.prepare(il, &all, &[], false).unwrap();
        assert_eq!(dp.slots, all);
        for pi in 0..3 {
            let (dw, rw) = (
                dp.arrays[pi].w.to_uint32().unwrap().to_vec(),
                rp.arrays[pi].w.to_uint32().unwrap().to_vec(),
            );
            let cols = dw.len() / ne;
            for (&e, &s) in rp.experts.iter().zip(&rp.slots) {
                assert_eq!(
                    dw[e as usize * cols..(e as usize + 1) * cols],
                    rw[s as usize * cols..(s as usize + 1) * cols]
                );
            }
        }
        rig.reference.settle_all();
    }

    // Subsequent turns stay Direct and NonVolatile even if host memory spikes.
    env.host(host_mem(rig.physical, rig.physical - G), rig.ws);
    lease.begin_turn(&rig.bank, rig.dense, kv_live).unwrap();
    assert!(lease.in_turn() && rig.bank.in_turn());
    assert_eq!(
        lease.end_turn_idle(&rig.bank, rig.dense, kv_live).unwrap(),
        None
    );
    assert_eq!(
        (
            lease.switches().len(),
            rig.bank.residency().unwrap().volatile
        ),
        (1, 0)
    );
    rig.reference.end_turn().unwrap();
}

#[test]
fn lease_end_turn_idle_does_not_promote_when_disabled_capped_or_headroom_tight() {
    let env = TestEnvGuard::new();

    // 1. DirectMode::Off never promotes on a quiet host.
    let rig = PromoRig::new(40 * G, 96 * G);
    let lease = rig.lease(DirectMode::Off, MmapMode::Off, Limits::default(), None, 8);
    env.host(host_mem(rig.physical, 12 * G), rig.ws);
    assert_eq!(lease.end_turn_idle(&rig.bank, rig.dense, 0).unwrap(), None);
    assert!(!rig.bank.is_direct() && lease.switches().is_empty());

    // 2. cache_cap below direct_expert_bytes never promotes.
    let rig = PromoRig::new(40 * G, 96 * G);
    let lim = Limits {
        cache_cap: Some(rig.direct_bytes() - 1),
        ..Limits::default()
    };
    let lease = rig.lease(DirectMode::Auto, MmapMode::Off, lim, None, 8);
    assert_eq!(lease.end_turn_idle(&rig.bank, rig.dense, 0).unwrap(), None);
    assert!(!rig.bank.is_direct());

    // 3. Large live KV exceeding physical headroom blocks promotion before dropping partial bank.
    let rig = PromoRig::new(40 * G, 96 * G);
    let old = rig.bank.nbytes();
    let lease = rig.lease(DirectMode::Auto, MmapMode::Off, Limits::default(), None, 8);
    let huge_kv = 26 * G;
    env.host(host_mem(rig.physical, rig.dense + huge_kv + G), rig.ws);
    assert_eq!(
        lease.end_turn_idle(&rig.bank, rig.dense, huge_kv).unwrap(),
        None
    );
    assert!(!rig.bank.is_direct() && rig.bank.nbytes() == old && rig.direct_bytes() > 0);
}

#[test]
fn lease_direct_bank_never_shrinks_or_flips_volatile() {
    let files = BankFiles::new(5);
    let (bank, p, ws) = (files.direct(), profile(100), 64 * G);
    let classes = files.classes(&bank);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: 1024,
    };
    let terms = Terms {
        dense: 10 * G,
        staging_layer: 0,
        kv: 4 * G,
        compute: direct_scratch_bytes(&p, 1024, 1024),
        io: 0,
        driver: 0,
        pool: 0,
    };
    let adm = synth_adm(
        128 * G,
        20 * G,
        DIRECT_MARGIN,
        104 * G,
        ws,
        terms,
        0,
        80 * G,
        size.slot,
        12,
    );
    let lease = Lease::new(
        p,
        size,
        1024,
        classes,
        Some(Limits::default()),
        Some(adm),
        &bank,
    );
    assert_eq!(lease.wired_limit(&bank), Some(ws));

    let env = TestEnvGuard::new();
    env.host(host_mem(128 * G, 125 * G), ws);
    lease.begin_turn(&bank, 10 * G, 4 * G).unwrap();
    assert!(lease.in_turn() && bank.in_turn() && !lease.recheck(&bank, 10 * G, 4 * G).unwrap());
    assert!(lease.switches().is_empty());

    lease.end_turn(&bank).unwrap();
    let r = bank.residency().unwrap();
    assert!(!lease.in_turn() && !bank.in_turn() && r.resident > 0 && r.volatile == 0);

    lease.begin_turn(&bank, 10 * G, 4 * G).unwrap();
    assert!(lease.in_turn() && bank.in_turn() && lease.switches().is_empty());
    lease.end_turn(&bank).unwrap();
}

#[test]
fn shard1_nocache_and_page_cache_flags() {
    for (raw, want) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("false"), false),
        (Some("1"), true),
        (Some("true"), true),
        (Some(" TRUE "), true),
    ] {
        assert_eq!(shard1_nocache(raw), want, "{raw:?}");
    }
    let m = store_moe();
    let l1 = kquant_layer(
        &mut Rng::new(1),
        &m,
        1,
        ExpertClass::Q6kDown,
        1,
        &mut Vec::new(),
    );
    let experts = vec![None, Some(l1)];
    assert_eq!(
        shard_page_cache_for(2, &experts, 1, false),
        vec![true, false]
    );
    assert_eq!(
        shard_page_cache_for(2, &experts, 1, true),
        vec![false, false]
    );
    assert_eq!(shard_page_cache_for(1, &experts, 0, false), vec![false]);
}

#[test]
fn mmap_mode_and_prewarm_env_parsing() {
    use MmapMode as M;
    let cases = [
        (None, M::Auto),
        (Some(""), M::Auto),
        (Some("  "), M::Auto),
        (Some("auto"), M::Auto),
        (Some("AUTO"), M::Auto),
        (Some("1"), M::On),
        (Some("true"), M::On),
        (Some("TRUE"), M::On),
        (Some("0"), M::Off),
        (Some("false"), M::Off),
        (Some("FALSE"), M::Off),
    ];
    for (raw, want) in cases {
        assert_eq!(mmap_mode(raw).unwrap(), want, "{raw:?}");
        assert_eq!(mmap_prewarm_mode(raw).unwrap(), want, "{raw:?}");
    }
    assert!(
        mmap_mode(Some("yes"))
            .unwrap_err()
            .reason
            .contains(MMAP_ENV)
    );
    assert!(
        mmap_prewarm_mode(Some("no"))
            .unwrap_err()
            .reason
            .contains(MMAP_PREWARM_ENV)
    );
}

#[test]
fn mmap_admission_headroom_guard_and_precedence() {
    use DirectMode as D;
    use MmapMode as M;
    let mut rig = ModeRig::real();
    let (q128, b128, s128) = (
        host_mem(128 * G, 20 * G),
        host_mem(128 * G, 50 * G),
        host_mem(128 * G, 80 * G),
    );
    let (b192, h96) = (host_mem(192 * G, 50 * G), host_mem(96 * G, 20 * G));

    // 1. Quiet 128 GiB host + Auto/Auto admits Mmap at 4096 with wired_limit < 20 GiB.
    let ExpertResidency::Mmap { chunk, gate } = rig.select(M::Auto, D::Auto, &q128).unwrap() else {
        panic!("expected Mmap on 128 GiB quiet host");
    };
    assert_eq!(
        (chunk, gate.admitted, gate.mapped_bytes),
        (MMAP_PREFILL_CHUNK_MAX, true, rig.mapped_bytes)
    );
    assert_eq!(
        gate.need,
        gate.dense_bytes + gate.kv_and_scratch + rig.mapped_bytes / 2
    );
    let wired = gate.dense_bytes + gate.kv_and_scratch;
    assert!(wired < 20 * G && direct_wired_limit(wired, rig.ws) < 20 * G);

    assert_eq!(rig.warm(M::Auto, &q128), rig.mapped_bytes);
    assert_eq!(rig.warm(M::Off, &q128), 0);
    assert_eq!(rig.warm(M::Auto, &host_mem(128 * G, 108 * G)), 0);

    // 2. Forced MmapMode::On admits on squeezed 128 GiB host with base-only need (< 20 GiB).
    let ExpertResidency::Mmap {
        chunk: fc,
        gate: fg,
    } = rig.select(M::On, D::Auto, &s128).unwrap()
    else {
        panic!("expected MLX_HY4_MMAP=1 to force Mmap");
    };
    assert_eq!(
        (fc, fg.admitted, fg.need),
        (
            MMAP_PREFILL_CHUNK_MAX,
            true,
            fg.dense_bytes + fg.kv_and_scratch
        )
    );
    assert!(fg.need < 20 * G);

    // 3. Full mode precedence matrix.
    #[derive(Debug, PartialEq, Eq)]
    enum Kind {
        Mmap,
        Direct,
        Streamed,
    }
    let kind = |r| match r {
        ExpertResidency::Mmap { .. } => Kind::Mmap,
        ExpertResidency::Direct { .. } => Kind::Direct,
        ExpertResidency::Streamed { .. } => Kind::Streamed,
    };
    for (m, d, metal, h, want) in [
        (M::Auto, D::Auto, true, b128, Kind::Mmap),
        (M::Auto, D::Off, true, b128, Kind::Mmap),
        (M::Auto, D::On, true, b128, Kind::Mmap),
        (M::Auto, D::Auto, true, s128, Kind::Streamed),
        (M::Auto, D::Off, true, s128, Kind::Streamed),
        (M::Auto, D::On, true, s128, Kind::Streamed),
        (M::Auto, D::Auto, true, b192, Kind::Mmap),
        (M::Off, D::Auto, true, q128, Kind::Direct),
        (M::Off, D::Auto, true, b128, Kind::Streamed),
        (M::Auto, D::On, true, q128, Kind::Direct),
        (M::Auto, D::Off, true, q128, Kind::Mmap),
        (M::Auto, D::Auto, true, h96, Kind::Streamed),
        (M::Auto, D::Off, false, q128, Kind::Streamed),
        (M::On, D::Off, true, h96, Kind::Mmap),
    ] {
        rig.metal = metal;
        assert_eq!(
            kind(rig.select(m, d, &h).unwrap()),
            want,
            "{m:?} {d:?} metal={metal}"
        );
    }

    // Squeezed >=100 GiB Metal host records mmap_refusal; <100 GiB host does not.
    rig.metal = true;
    assert!(matches!(
        rig.select(M::Auto, D::Auto, &s128).unwrap(),
        ExpertResidency::Streamed {
            direct_refusal: None,
            mmap_refusal: Some(g),
        } if !g.admitted
    ));
    assert!(matches!(
        rig.select(M::Auto, D::Auto, &h96).unwrap(),
        ExpertResidency::Streamed {
            direct_refusal: None,
            mmap_refusal: None,
        }
    ));

    // 4. Forced MmapMode::On fails loud when squeezed or without Metal.
    rig.metal = true;
    assert!(
        rig.select(M::On, D::Auto, &host_mem(128 * G, 118 * G))
            .unwrap_err()
            .reason
            .contains("MLX_HY4_MMAP=1 refused")
    );
    rig.metal = false;
    assert!(
        rig.select(M::On, D::Auto, &b128)
            .unwrap_err()
            .reason
            .contains("Metal")
    );
}

#[test]
fn lease_mmap_bank_has_zero_anon_wired_under_20gb_and_never_promotes() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let files = BankFiles::new(6);
    let bank = files.mmap();
    assert!(bank.is_direct() && bank.is_mmap() && bank.nbytes() == 0);
    assert_eq!(
        (bank.resident_bytes().unwrap(), bank.residency().unwrap()),
        (0, Residency::default())
    );

    let (p, ctx, dense, ws) = (profile(15_812), 262_144usize, 10 * G, 96 * G);
    let (kv, sc) = (
        direct_kv_allowance(&p, ctx),
        direct_scratch_bytes(&p, MMAP_PREFILL_CHUNK_MAX, ctx),
    );
    let fixed = dense + kv + sc;
    assert!(fixed < 20 * G);
    let classes = files.classes(&bank);
    let size = ModelSize {
        dense,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: ctx as u64,
    };
    let terms = Terms {
        dense,
        staging_layer: 0,
        kv,
        compute: sc,
        io: 0,
        driver: 0,
        pool: 0,
    };
    let adm = synth_adm(
        128 * G,
        50 * G,
        DIRECT_MARGIN,
        74 * G,
        ws,
        terms,
        0,
        fixed,
        size.slot,
        12,
    );
    let lease = Lease::new(
        p,
        size,
        MMAP_PREFILL_CHUNK_MAX as u64,
        classes,
        Some(Limits::default()),
        Some(adm),
        &bank,
    )
    .with_direct_budget(Some(74 * G));

    let wl = lease.wired_limit(&bank).unwrap();
    assert!(wl == fixed && wl < 20 * G && !bank.in_turn());
    lease.begin_turn(&bank, dense, kv).unwrap();
    assert!(lease.in_turn() && bank.in_turn() && !lease.recheck(&bank, dense, kv).unwrap());
    assert_eq!(lease.end_turn_idle(&bank, dense, kv).unwrap(), None);
    assert!(!lease.in_turn() && !bank.in_turn() && bank.is_mmap() && lease.switches().is_empty());
}

#[test]
fn mmap_chunk_max_override_fallback_and_dense_savings_accounting() {
    assert_eq!(MMAP_PREFILL_CHUNK_MAX, 4096);
    for (raw, want) in [
        (None, 4096),
        (Some(""), 4096),
        (Some("2048"), 2048),
        (Some("3072"), 3072),
    ] {
        assert_eq!(mmap_prefill_chunk_max(raw).unwrap(), want, "{raw:?}");
    }
    for bad in ["0", "300", "abc"] {
        assert!(mmap_prefill_chunk_max(Some(bad)).is_err(), "{bad}");
    }

    let mut rig = ModeRig::real().with_savings(1_194_590_208);
    let q128 = host_mem(128 * G, 20 * G);
    let ExpertResidency::Mmap { chunk, gate } = rig
        .select_for(MmapMode::Auto, DirectMode::Auto, &q128)
        .unwrap()
    else {
        panic!("expected Mmap at 4096");
    };
    assert_eq!((chunk, gate.dense_bytes), (4096, rig.dense_mmap));
    let wired_default = gate.dense_bytes + gate.kv_and_scratch;

    rig.mmap_chunk_max = 2048;
    let ExpertResidency::Mmap {
        chunk: c2k,
        gate: g2k,
    } = rig
        .select_for(MmapMode::Auto, DirectMode::Auto, &q128)
        .unwrap()
    else {
        panic!("expected Mmap at 2048");
    };
    assert!(
        c2k == 2048 && g2k.dense_bytes == rig.dense_mmap && g2k.scratch_bytes < gate.scratch_bytes
    );

    let mid = g2k.need + (gate.need - g2k.need) / 2;
    let (fbc, fbg) = pick_mmap_with_max(
        MmapMode::Auto,
        true,
        true,
        &rig.p,
        rig.ctx,
        &q128,
        &Own::default(),
        rig.dense_mmap,
        rig.mapped_bytes,
        4096,
        None,
        Some(mid),
    );
    assert!(fbg.admitted && fbc == 2048);

    rig.mmap_chunk_max = 4096;
    let ExpertResidency::Direct { gate: dg, .. } = rig
        .select_for(MmapMode::Off, DirectMode::Auto, &q128)
        .unwrap()
    else {
        panic!("expected Direct");
    };
    assert!(dg.dense_bytes == rig.dense_full && wired_default < 15 * G);
}

#[test]
fn auto_mmap_on_metal_never_promotes_streamed_to_direct() {
    use DirectMode as D;
    use MmapMode as M;
    let (above, below) = (128 * G, 96 * G);
    for (m, d, metal, phys, is_mmap, want) in [
        (M::Auto, D::Auto, true, above, false, false),
        (M::Auto, D::Auto, true, above, true, false),
        (M::Off, D::Off, true, above, false, false),
        (M::Off, D::Auto, true, above, false, true),
        (M::Auto, D::On, true, above, false, true),
        (M::Auto, D::Auto, false, above, false, true),
        (M::Auto, D::Auto, true, below, false, true),
        (M::Auto, D::Auto, true, below, true, false),
        (M::Auto, D::Off, true, below, false, false),
    ] {
        assert_eq!(
            should_enable_direct_promotion(m, d, metal, phys, is_mmap),
            want
        );
    }

    let env = TestEnvGuard::new();
    let rig = PromoRig::new(128 * G, 96 * G);
    let lease = rig.lease(
        D::Auto,
        M::Auto,
        Limits::default(),
        None,
        rig.files.moe.n_experts as u64,
    );
    env.host(host_mem(rig.physical, 12 * G), rig.ws);
    assert_eq!(lease.end_turn_idle(&rig.bank, rig.dense, 0).unwrap(), None);
    assert!(!rig.bank.is_direct() && lease.switches().is_empty());

    let rig = PromoRig::new(128 * G, 96 * G);
    let lease = rig.lease(
        D::On,
        M::Auto,
        Limits::default(),
        None,
        rig.files.moe.n_experts as u64,
    );
    assert!(
        lease
            .end_turn_idle(&rig.bank, rig.dense, 0)
            .unwrap()
            .is_some()
            && rig.bank.is_direct()
    );
    rig.bank.wait_direct_fill().unwrap();
}

#[test]
fn below_mmap_floor_auto_metal_recovers_streamed_to_direct_on_idle() {
    let rig = PromoRig::new(96 * G, 72 * G);
    assert!(rig.physical < MMAP_MIN_PHYSICAL);
    let (db, (need_2k, _, _)) = (rig.direct_bytes(), rig.need_2k());
    let high = rig.physical - need_2k - DIRECT_MARGIN + 2 * G;

    let busy = host_mem(rig.physical, high);
    let sel = select_expert_mode(
        MmapMode::Auto,
        DirectMode::Auto,
        true,
        &rig.p,
        rig.ctx,
        &busy,
        &Own::default(),
        rig.dense,
        db,
        db,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(matches!(sel, ExpertResidency::Streamed { .. }));
    assert!(should_enable_direct_promotion(
        MmapMode::Auto,
        DirectMode::Auto,
        true,
        rig.physical,
        false
    ));

    let lease = rig.lease(
        DirectMode::Auto,
        MmapMode::Auto,
        Limits::default(),
        None,
        rig.files.moe.n_experts as u64,
    );
    let (env, r) = (TestEnvGuard::new(), rig.bank.residency().unwrap());

    env.host(
        host_mem(rig.physical, high + rig.dense + r.resident),
        rig.ws,
    );
    assert_eq!(lease.end_turn_idle(&rig.bank, rig.dense, 0).unwrap(), None);
    assert!(!rig.bank.is_direct() && lease.switches().is_empty());

    env.host(
        host_mem(rig.physical, 4 * G + rig.dense + r.resident),
        rig.ws,
    );
    assert!(
        lease
            .end_turn_idle(&rig.bank, rig.dense, 0)
            .unwrap()
            .is_some()
    );
    assert!(rig.bank.is_direct() && lease.switches().len() == 1);
    rig.bank.wait_direct_fill().unwrap();
}

#[test]
fn select_expert_mode_with_retry_recovers_mmap_after_clear_cache() {
    let rig = ModeRig::real().with_savings(1_194_590_208);

    let (cl, rd) = (Cell::new(0usize), Cell::new(0usize));
    let (_, _, res) = rig
        .retry(
            MmapMode::Auto,
            DirectMode::Auto,
            || {
                rd.set(rd.get() + 1);
                Ok((
                    host_mem(128 * G, if cl.get() == 0 { 76 * G } else { 60 * G }),
                    rig.ws,
                ))
            },
            || if cl.get() == 0 { 16 * G } else { 0 },
            || cl.set(cl.get() + 1),
        )
        .unwrap();
    assert_eq!((cl.get(), rd.get()), (1, 2));
    assert!(matches!(res, ExpertResidency::Mmap { chunk: 4096, .. }));

    let (cl, rd) = (Cell::new(0usize), Cell::new(0usize));
    let (_, _, res) = rig
        .retry(
            MmapMode::Auto,
            DirectMode::Auto,
            || {
                rd.set(rd.get() + 1);
                Ok((host_mem(128 * G, 80 * G), rig.ws))
            },
            || 0,
            || cl.set(cl.get() + 1),
        )
        .unwrap();
    assert!(
        (cl.get(), rd.get()) == (0, 1)
            && matches!(
                res,
                ExpertResidency::Streamed {
                    direct_refusal: None,
                    mmap_refusal: Some(_),
                }
            )
    );

    let cl = Cell::new(0usize);
    let (_, _, res) = rig
        .retry(
            MmapMode::On,
            DirectMode::Off,
            || {
                Ok((
                    host_mem(128 * G, if cl.get() == 0 { 118 * G } else { 50 * G }),
                    rig.ws,
                ))
            },
            || 8 * G,
            || cl.set(cl.get() + 1),
        )
        .unwrap();
    assert!(cl.get() == 1 && matches!(res, ExpertResidency::Mmap { .. }));
}

#[test]
fn mmap_rset_and_warm_budgets_bound_auto_and_max_memory() {
    use DirectMode as D;
    use MmapMode as M;

    // Busy 128 GiB host: Auto admits with partial headroom (< mapped_bytes).
    let mut rig = ModeRig::real().with_savings(1_194_590_208);
    let b128 = host_mem(128 * G, 50 * G);
    let ExpertResidency::Mmap { gate, .. } = rig.select_for(M::Auto, D::Auto, &b128).unwrap()
    else {
        panic!("expected Mmap on busy 128 GiB host");
    };
    let kv = direct_kv_allowance(&rig.p, rig.ctx);
    let page = 12 * u64::from(crate::models::hy_v4_flash::paged::BLOCK) * (576 * 2 + 128 * 2 / 4);
    let full_kv = mmap_full_kv_bytes(&rig.p, rig.ctx, 1, page, 0, 0);
    let rset_reserve = gate.dense_bytes + full_kv + gate.scratch_bytes;
    let rset_auto = mmap_rset_budget(M::Auto, gate.budget, rset_reserve);
    assert_eq!(rset_auto, gate.budget - rset_reserve);
    assert!(rset_auto <= gate.budget - gate.dense_bytes - full_kv);
    assert!(rset_auto < rig.mapped_bytes && rset_auto >= rig.mapped_bytes / 2);
    assert_eq!(mmap_rset_budget(M::On, gate.budget, rset_reserve), u64::MAX);
    assert_eq!(mmap_rset_budget(M::Off, gate.budget, rset_reserve), 0);

    // Roomy 192 GiB host capped by --max-memory bounds BOTH rset and prewarm.
    let q192 = host_mem(192 * G, 20 * G);
    rig.max_memory = Some(70 * G);
    let ExpertResidency::Mmap { gate: mg, .. } = rig.select_for(M::Auto, D::Auto, &q192).unwrap()
    else {
        panic!("expected Mmap under 70 GiB max_memory");
    };
    assert_eq!(mg.budget, 70 * G);
    let m_reserve = mg.dense_bytes + full_kv + mg.scratch_bytes;
    let m_rset = mmap_rset_budget(M::Auto, mg.budget, m_reserve);
    let m_warm = mmap_warm_budget(
        M::Auto,
        &q192,
        mg.dense_bytes,
        kv,
        mg.margin,
        rig.mapped_bytes,
        Some(mg.budget),
    );
    assert_eq!(m_rset, 70 * G - m_reserve);
    assert_eq!(m_warm, 70 * G - mg.dense_bytes - kv);
    assert!(m_rset <= 70 * G - mg.dense_bytes - full_kv && m_warm < rig.mapped_bytes);

    // Tight max_memory below prewarm floor disables background prewarm cleanly.
    assert_eq!(
        mmap_warm_budget(
            M::Auto,
            &q192,
            mg.dense_bytes,
            kv,
            mg.margin,
            rig.mapped_bytes,
            Some(mg.dense_bytes + kv + MMAP_PREWARM_MIN_BUDGET - 1),
        ),
        0
    );
}

#[test]
fn partial_auto_mmap_charges_rset_pinned_bytes_in_governor_untracked() {
    use std::collections::HashMap;

    use crate::engine::governor::governor::tests::{FakeIo, host as gov_host, lane, stats};
    use crate::engine::governor::governor::{Failure, Governor, ModelRegistration, Scope};
    use crate::engine::governor::host::Pressure;
    use crate::engine::governor::ledger::ClaimClass;
    use crate::models::hy_v4_flash::paged::BLOCK;
    use crate::models::hy_v4_flash::scan::{ENGRAM_TABLE, Located};
    use crate::models::hy_v4_flash::sched::Paged;
    use crate::models::hy_v4_flash::test_support::{
        TINY_CLASSES, TinyWeightsOpts, tiny_engram_table, tiny_weights_opts,
    };
    use crate::models::hy_v4_flash::weights::MmapDenseSpec;
    use crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks;
    use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

    // 1. Real Hy4 numbers under partial MmapMode::Auto + tight --max-memory (70 GiB).
    let mut rig = ModeRig::real().with_savings(1_194_590_208);
    rig.max_memory = Some(70 * G);
    let q192 = host_mem(192 * G, 20 * G);
    let ExpertResidency::Mmap { gate, .. } = rig
        .select_for(MmapMode::Auto, DirectMode::Auto, &q192)
        .unwrap()
    else {
        panic!("expected partial Mmap under 70 GiB max_memory");
    };
    assert_eq!(gate.budget, 70 * G);
    let page = 12 * u64::from(BLOCK) * (576 * 2 + 128 * 2 / 4); // 466,944 B
    let lanes = 1usize;
    let full_kv = mmap_full_kv_bytes(&rig.p, rig.ctx, lanes, page, 0, 0);
    let max_pool = paged_pool_max_bytes(page, rig.ctx, lanes);
    assert!(full_kv >= full_kv_allowance(&rig.p, rig.ctx) && full_kv >= max_pool);
    // Multi-lane and non-block-aligned edge checks: full KV reservation always
    // covers both profile full-ctx + extra lanes AND exact paged max_blocks + state.
    let extra_4 = 3 * 36 * MIB;
    let state_4 = 4 * 36 * MIB + LADDER_BYTES + (15_812 - 14_596) * rig.ctx as u64;
    let full_kv_4 = mmap_full_kv_bytes(&rig.p, rig.ctx, 4, page, extra_4, state_4);
    assert!(full_kv_4 >= full_kv_allowance(&rig.p, rig.ctx) + extra_4);
    assert!(full_kv_4 >= paged_pool_max_bytes(page, rig.ctx, 4) + state_4);
    let zero_base = Profile {
        kv_base: 0,
        ..rig.p
    };
    assert_eq!(
        mmap_full_kv_bytes(&zero_base, 100, 4, page, extra_4, 0),
        paged_pool_max_bytes(page, 100, 4) + extra_4
    );

    let rset_reserve = gate
        .dense_bytes
        .saturating_add(full_kv)
        .saturating_add(gate.scratch_bytes);
    let rset_pinned = mmap_rset_budget(MmapMode::Auto, gate.budget, rset_reserve);
    assert!(rset_pinned > 0 && rset_pinned < rig.mapped_bytes);
    assert!(rset_reserve + rset_pinned <= gate.budget);

    let runway = u64::from(runway_pool_extent_blocks(page)) * page;
    let kv_64k_pool = (DIRECT_KV_ALLOWANCE_TOKENS as u64 / u64::from(BLOCK)) * page;
    assert!(runway < kv_64k_pool && kv_64k_pool < max_pool);

    let io = FakeIo::new(stats(gate.dense_bytes), gov_host(64 * G, Pressure::Normal));
    let gov = Governor::new(io.clone());
    let reg = ModelRegistration {
        model_id: 1,
        hard_budget: gate.budget,
        untracked: rset_pinned,
        page_bytes: page,
    };
    let (_guard, ()) = gov.load_pool(reg, runway, || Ok(())).unwrap();
    io.with(|s| s.stats.pools = runway);
    assert_eq!(
        gov.snapshot().headroom,
        full_kv + gate.scratch_bytes - runway
    );

    // Growing KV all the way to the advertised 262,144-token window
    // (max_blocks = ctx.div_ceil(32) + lanes + 1) WITH full prefill scratch
    // active succeeds within hard_budget.
    let claim = gov.open(lane(ClaimClass::InService, 1, 0), reg.model_id);
    {
        let _scope = Scope::enter(claim.claim());
        let res = gov.try_reserve(kv_64k_pool - runway).unwrap();
        io.with(|s| s.stats.pools = kv_64k_pool);
        drop(res);

        io.with(|s| s.stats.active = gate.dense_bytes + gate.scratch_bytes);
        let res_full = gov.try_reserve(max_pool - kv_64k_pool).unwrap();
        io.with(|s| s.stats.pools = max_pool);
        drop(res_full);
        assert!(gate.dense_bytes + gate.scratch_bytes + max_pool + rset_pinned <= gate.budget);

        // An extra claim past hard_budget is properly rejected with Failure::EngineBudget.
        let leftover = gov.snapshot().headroom;
        assert_eq!(
            gov.try_reserve(leftover + page).err(),
            Some(Failure::EngineBudget)
        );
    }

    // 2. Real Store + Paged::model_registration wiring on Metal (rset_bounded on vs off).
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let cfg = tiny_cfg();
    let mut rng = Rng::new(9);
    let mut exp_file = rng.bytes(1024);
    let experts: Vec<Option<ExpertLayer>> = TINY_CLASSES
        .iter()
        .enumerate()
        .map(|(il, c)| c.map(|c| kquant_layer(&mut rng, &cfg.moe, il, c, 32, &mut exp_file)))
        .collect();
    let tmp = Temp::new("store-rset-gov");
    let exp_path = tmp.write("experts.bin", &exp_file);
    let engram_rows: u64 = cfg.engram.head_vocab_sizes.iter().sum();
    let engram_path = tmp.write(
        "engram.bin",
        &vec![0u8; engram_rows as usize * cfg.engram.head_dim],
    );
    let paths = vec![exp_path.clone(), engram_path];
    let shards = Arc::new(ShardSet::open(&paths).unwrap());
    let engram_loc = Located {
        shard: 1,
        data_offset: 0,
        info: GgufTensorInfo {
            name: ENGRAM_TABLE.into(),
            n_dims: 2,
            dims: vec![cfg.engram.head_dim as u64, engram_rows],
            tensor_type: GgufTensorType::MXFP8,
            offset: 0,
        },
    };
    let table = Arc::new(
        F8Table::open_with_options(Arc::clone(&shards), 1, &engram_loc, &cfg.engram, 1, 1).unwrap(),
    );
    let (probe, _) = KqBank::new_mmap_with_dense(
        IoPool::new(Arc::clone(&shards), 2).unwrap(),
        &cfg.moe,
        &experts,
        std::slice::from_ref(&exp_path),
        0,
        false,
        u64::MAX,
        &MmapDenseSpec::default(),
    )
    .unwrap();
    let partial_cap = probe.rset_total_bytes() / 2;
    let rset_supported = probe.has_residency_set();
    drop(probe);

    let (bank, _) = KqBank::new_mmap_with_dense(
        IoPool::new(Arc::clone(&shards), 2).unwrap(),
        &cfg.moe,
        &experts,
        std::slice::from_ref(&exp_path),
        0,
        false,
        partial_cap,
        &MmapDenseSpec::default(),
    )
    .unwrap();
    let bank = Arc::new(bank);
    let pinned = bank.rset_pinned_bytes();
    assert!(pinned <= partial_cap);
    if rset_supported {
        assert!(pinned > 0 && pinned < bank.rset_total_bytes());
    }
    let classes = Classes {
        layers: [2, 1],
        slot: bank.slot_bytes(),
    };
    let p = profile(15_812);
    let size = ModelSize {
        dense: 4 * MIB,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: cfg.moe.n_experts as u64,
        ctx: cfg.context_length as u64,
    };
    let hard_budget = size.dense + pinned + 2 * MIB;
    let b_ref = Arc::clone(&bank);
    let model = Model::new(
        cfg.clone(),
        Box::new(tiny_weights_opts(&cfg, TinyWeightsOpts::default())),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b_ref))),
        Box::new(tiny_engram_table(&cfg.engram, 9)),
    )
    .unwrap();
    let lease =
        Lease::new(p, size, 64, classes, None, None, &bank).with_direct_budget(Some(hard_budget));
    let mut store = Store {
        model,
        bank: Arc::clone(&bank),
        table,
        scan: Scan {
            paths,
            cfg: cfg.clone(),
            tensors: HashMap::new(),
            experts,
            engram_table: engram_loc,
            mtp: None,
        },
        shards,
        dense_read: 0,
        dense_bytes: size.dense,
        sizes: Sizes {
            dense_trunk: size.dense,
            dense_mtp: 0,
            classes,
            n_experts: cfg.moe.n_experts,
        },
        size,
        profile: p,
        ctx: size.ctx as usize,
        chunk: 64,
        lanes: 1,
        row_cap: 12,
        mtp: false,
        bank_alloc_s: 0.0,
        max_memory: Some(hard_budget),
        direct_budget: Some(hard_budget),
        rset_bounded: true,
        lease,
    };

    // Bounded MmapMode::Auto passes rset_pinned_bytes into ModelRegistration.untracked.
    let auto_reg = Paged::model_registration(&store, 64 * 1024);
    assert_eq!(store.governor_untracked_bytes(), pinned);
    assert_eq!(
        (auto_reg.hard_budget, auto_reg.untracked),
        (hard_budget, pinned)
    );

    let io2 = FakeIo::new(stats(size.dense), gov_host(4 * G, Pressure::Normal));
    let gov2 = Governor::new(io2.clone());
    let (_g2, ()) = gov2.load_pool(auto_reg, MIB, || Ok(())).unwrap();
    io2.with(|s| s.stats.pools = MIB);
    assert_eq!(gov2.snapshot().headroom, MIB);
    let c2 = gov2.open(lane(ClaimClass::InService, 1, 0), auto_reg.model_id);
    {
        let _scope = Scope::enter(c2.claim());
        assert_eq!(
            gov2.try_reserve(MIB + 64 * 1024).err(),
            Some(Failure::EngineBudget)
        );
    }

    // Forced MmapMode::On (rset_bounded = false) leaves untracked = 0 even with pinned bytes.
    store.rset_bounded = false;
    let on_reg = Paged::model_registration(&store, 64 * 1024);
    assert_eq!((store.governor_untracked_bytes(), on_reg.untracked), (0, 0));
}
