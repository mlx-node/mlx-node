use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::models::hy_v4_flash::bank::{class_types, geometry};
use crate::models::hy_v4_flash::config::MoeConfig;
use crate::models::hy_v4_flash::scan::{ExpertClass, ExpertLayer, Located};
use crate::moe_stream::budget::{Bound, Terms};
use crate::utils::gguf::GgufTensorInfo;

const G: u64 = 1 << 30;

fn fixture(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn u(v: &Value) -> u64 {
    v.as_u64().unwrap_or_else(|| panic!("not a u64: {v}"))
}

/// `admission-addons.json` host: `others` bytes in use, nothing of this model.
fn host(inputs: &Value, others: u64) -> (HostMemory, u64) {
    (
        HostMemory {
            physical: u(&inputs["physical"]),
            wired: others,
            ..Default::default()
        },
        u(&inputs["ws"]),
    )
}

fn base(inputs: &Value, variant: &Value) -> ModelSize {
    ModelSize {
        dense: u(&variant["dense"]),
        slot: u(&inputs["slot"]),
        n_experts: u(&inputs["n_experts"]),
        ctx: 0,
    }
}

#[test]
fn profile_is_the_fixture_profile() {
    let f = fixture("admission-addons.json");
    let pf = &f["profile"];
    let p = profile(u(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(p.kv_base, u(&pf["KV_BASE"]) + u(&pf["kv_fixed_ladder"]));
    assert_eq!(p.compute_base, u(&pf["COMPUTE_BASE"]));
    assert_eq!(p.compute_per_row, u(&pf["COMPUTE_PER_ROW"]));
    assert_eq!(p.compute_fixed, u(&pf["compute_fixed"]));
    assert_eq!(p.compute_per_ctx, u(&pf["compute_per_ctx"]));
    assert_eq!(p.io_base, u(&pf["IO_BASE"]));
    assert_eq!(p.io_per_row, u(&pf["IO_PER_ROW"]));
    assert_eq!(p.io_fixed, u(&pf["io_fixed"]));
    assert_eq!(
        p.second_staging_min_cache,
        u(&pf["SECOND_STAGING_MIN_CACHE"])
    );
    assert_eq!((p.cache_env, p.reserve_env), (CACHE_ENV, RESERVE_ENV));
}

/// Every (ctx, chunk, others) row of both MTP variants: terms, refusal,
/// staging layers, fixed, cache slots, wired limit and bound.
#[test]
fn admission_matches_the_fixture_rows() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let mut n = 0;
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let p = profile(u(&v["kv_per_ctx"]));
        for r in v["rows"].as_array().unwrap() {
            let m = ModelSize {
                ctx: u(&r["ctx"]),
                ..base(inputs, v)
            };
            let rows = u(&r["chunk_rows"]);
            let others = u(&r["others_gib"]) * G;
            let (h, ws) = host(inputs, others);
            let case = format!("{name} ctx {} chunk {rows} others {others}", m.ctx);
            let t = Terms::with(&p, &m, ws, rows);
            let want = &r["terms"];
            assert_eq!(
                [
                    t.dense,
                    t.staging_layer,
                    t.kv,
                    t.compute,
                    t.io,
                    t.driver,
                    t.pool
                ],
                [
                    "dense",
                    "staging_layer",
                    "kv",
                    "compute",
                    "io",
                    "driver",
                    "pool"
                ]
                .map(|k| u(&want[k])),
                "{case}"
            );
            let got = admit_with(&p, &m, rows, &Limits::default(), &h, &Own::default(), ws);
            match (got, r["refused"].as_bool().unwrap()) {
                (Err(e), true) => assert!(e.reason.contains("vs budget"), "{case}: {e}"),
                (Ok(a), false) => {
                    assert_eq!(a.live, u(&r["live"]), "{case}");
                    assert_eq!(a.budget(), u(&r["budget"]), "{case}");
                    assert_eq!(a.staging_layers, u(&r["staging_layers"]), "{case}");
                    assert_eq!(a.fixed, u(&r["fixed"]), "{case}");
                    assert_eq!(a.cache_slots, u(&r["cache_slots"]), "{case}");
                    assert_eq!(a.cache_bytes(), u(&r["cache_bytes"]), "{case}");
                    assert_eq!(a.wired_limit(), u(&r["wired_limit"]), "{case}");
                    let bound = match r["bound"].as_str().unwrap() {
                        "Live" => Bound::Live,
                        "WorkingSet" => Bound::WorkingSet,
                        b => panic!("{case}: bound {b}"),
                    };
                    assert_eq!(a.bound, bound, "{case}");
                }
                (a, refused) => panic!("{case}: got {a:?}, fixture refused {refused}"),
            }
            n += 1;
        }
    }
    assert_eq!(n, 2 * 5 * 3 * 3);
}

/// The context ladder and chunk rule for every `ladder_others_gib` host.
#[test]
fn pick_matches_the_fixture_ladder() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let max_ctx = u(&inputs["max_context"]) as usize;
    assert_eq!(max_ctx, crate::models::hy_v4_flash::DEFAULT_CONTEXT_CAP);
    assert_eq!(u(&inputs["min_context"]) as usize, MIN_CONTEXT);
    assert_eq!(
        inputs["chunk_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| u(c) as usize)
            .collect::<Vec<_>>(),
        CHUNKS
    );
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let p = profile(u(&v["kv_per_ctx"]));
        let b = base(inputs, v);
        for (others, want) in v["pick"].as_object().unwrap() {
            let others = others.parse::<u64>().unwrap() * G;
            let (h, ws) = host(inputs, others);
            let case = format!("{name} others {others}");
            let got = pick(&p, &b, max_ctx, &Limits::default(), &h, &Own::default(), ws);
            if !want["admitted"].as_bool().unwrap() {
                let e = got.unwrap_err().reason;
                assert!(e.contains("refused at context 16384"), "{case}: {e}");
                continue;
            }
            let got = got.unwrap();
            assert_eq!(got.ctx, u(&want["ctx"]) as usize, "{case}");
            assert_eq!(got.chunk, u(&want["chunk_rows"]) as usize, "{case}");
            let a = got.admission;
            assert_eq!(a.cache_bytes(), u(&want["cache_bytes"]), "{case}");
            assert_eq!(a.staging_layers, u(&want["staging_layers"]), "{case}");
            assert_eq!(a.wired_limit(), u(&want["wired_limit"]), "{case}");
            assert_eq!(
                a.terms,
                Terms::with(
                    &p,
                    &ModelSize {
                        ctx: got.ctx as u64,
                        ..b
                    },
                    ws,
                    got.chunk as u64
                )
            );
        }
    }
}

/// A context below every chunk charges its own length; a refused context
/// below the floor errs at once.
#[test]
fn pick_below_the_floor() {
    let p = profile(15_812);
    let b = ModelSize {
        dense: 11_352_192_436,
        slot: 4_663_296,
        n_experts: 1024,
        ctx: 0,
    };
    let h = HostMemory {
        physical: 128 * G,
        wired: 20 * G,
        ..Default::default()
    };
    let ws = 96_000_000_000;
    let got = pick(&p, &b, 4096, &Limits::default(), &h, &Own::default(), ws).unwrap();
    assert_eq!((got.ctx, got.chunk), (4096, 4096));
    let m = ModelSize { ctx: 4096, ..b };
    assert_eq!(got.admission.terms, Terms::with(&p, &m, ws, 4096));
    let busy = HostMemory {
        wired: 100 * G,
        ..h
    };
    let e = pick(&p, &b, 8192, &Limits::default(), &busy, &Own::default(), ws)
        .unwrap_err()
        .reason;
    assert!(e.contains("refused at context 8192"), "{e}");
}

/// The second turn of a 256K load bound by live memory: the sessions' KV,
/// allocated after the load's reading, is this model's and keeps the bank;
/// read as others' it would drop the bank to staging only.
#[test]
fn readmission_counts_the_session_kv_as_own() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let others = 50 * G;
    let (h0, ws) = host(inputs, others);
    let max_ctx = u(&inputs["max_context"]) as usize;
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
    assert_eq!(load.ctx, max_ctx);
    assert_eq!(load.admission.bound, Bound::Live);
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let kv = p.kv_per_ctx * load.ctx as u64;
    assert!(kv > crate::moe_stream::budget::KEEP_SLACK);
    let h1 = HostMemory {
        wired: others + b.dense + kv,
        ..h0
    };
    let turn = |own: Own| {
        readmit_with(
            &p,
            &load.admission,
            &m,
            load.chunk as u64,
            &Limits::default(),
            &h1,
            &own,
            ws,
        )
        .unwrap()
    };
    assert_eq!(turn(own_of(b.dense, kv, 0, true)), Turn::Keep);
    assert!(matches!(
        turn(own_of(b.dense, 0, 0, true)),
        Turn::StagingOnly(_)
    ));
}

/// Two staging layers charge 2 x 1024 slots at the larger slot (8.89
/// GiB), at least what the bank allocates (2 x 512 slots per class).
#[test]
fn staging_charge_is_two_full_layers() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let m = ModelSize {
        ctx: 262_144,
        ..base(inputs, v)
    };
    let t = Terms::with(&profile(u(&v["kv_per_ctx"])), &m, u(&inputs["ws"]), 8192);
    assert_eq!(t.staging_layer * 2, 9_550_430_208);
    let adm = fixture("admission.json");
    let bank = 2 * 512 * (u(&adm["inputs"]["slot_a"]) + u(&adm["inputs"]["slot_b"]));
    assert_eq!(bank, 8_965_324_800);
    assert!(t.staging_layer * 2 >= bank);
}

/// MLX's qmv batch limit per generation, size letter and shape class.
#[test]
fn qmv_batch_limit_follows_mlx() {
    let cases = [
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
    ];
    for (arch, want) in cases {
        let got =
            [(2048, 2048), (4096, 1024), (3072, 8192)].map(|(k, n)| qmv_batch_limit(arch, k, n));
        assert_eq!(got, want, "{arch}");
    }
    assert_eq!(qmv_batch_limit("applegpu_g17s", 2049, 16), 25);
    assert_eq!(qmv_batch_limit("applegpu_g17s", 16, 4097), 13);
}

/// Hy4's shape classes (small k_b / v_b, medium kv_a, large gate): 12 rows
/// on M3-M5 Max, 11 on an Ultra, 5-6 on M1/M2; the env sets every limit.
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

/// Each lane past the first adds its floor to the KV base, nothing else.
#[test]
fn lanes_charge_their_floor() {
    let cfg = crate::models::hy_v4_flash::tests_model::tiny_cfg();
    let floor = lane_floor(&cfg, false);
    assert_eq!(floor, 2 * 11 * 24 * 2 + 2 * 6 * 16 * 4 + 32 * 4);
    let p = profile(15_812);
    assert_eq!(lane_profile(&p, 1, floor), p);
    assert_eq!(lane_profile(&p, 0, floor), p);
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

/// The most lanes that admit win; one lane's refusal is the error.
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
    assert_eq!(plan_lanes(4, fits(4)).unwrap(), (4, 40));
    assert_eq!(plan_lanes(4, fits(2)).unwrap(), (2, 20));
    assert_eq!(plan_lanes(0, fits(4)).unwrap(), (1, 10));
    assert_eq!(plan_lanes(3, fits(0)).unwrap_err().reason, "1 lanes");
}

/// Lanes are fitted at the one-lane context and chunk: for every lane
/// floor the load keeps the one-lane pick's window and chunk and takes the
/// most lanes that admit there; with the ladder run per lane count some
/// floors would shrink the window or the chunk.
#[test]
fn lanes_keep_the_one_lane_window_and_chunk() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let (h, ws) = host(inputs, 50 * G);
    let (l, own) = (Limits::default(), Own::default());
    let one = pick(&p, &b, max_ctx, &l, &h, &own, ws).unwrap();
    let at = ModelSize {
        ctx: one.ctx as u64,
        ..b
    };
    let mut shrunk = 0;
    for quarter in 1..=160u64 {
        let floor = quarter * G / 4;
        let admit = |n: usize| {
            admit_with(
                &lane_profile(&p, n, floor),
                &at,
                one.chunk as u64,
                &l,
                &h,
                &own,
                ws,
            )
        };
        let (lanes, got) = lanes_at(one, LANES_MAX, admit);
        assert_eq!((got.ctx, got.chunk), (one.ctx, one.chunk), "floor {floor}");
        let most = (1..=LANES_MAX)
            .rev()
            .find(|&n| n == 1 || admit(n).is_ok())
            .unwrap();
        assert_eq!(lanes, most, "floor {floor}");
        if lanes > 1 {
            assert_eq!(got.admission, admit(lanes).unwrap(), "floor {floor}");
        } else {
            assert_eq!(got.admission, one.admission, "floor {floor}");
        }
        if let Ok(ladder) = pick(
            &lane_profile(&p, LANES_MAX, floor),
            &b,
            max_ctx,
            &l,
            &h,
            &own,
            ws,
        ) && (ladder.ctx, ladder.chunk) != (one.ctx, one.chunk)
        {
            shrunk += 1;
        }
    }
    assert!(shrunk > 0, "no floor shrinks the window or the chunk");
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
        assert!(used <= bytes, "{bytes}");
        assert!(bytes - used < c.slot[1], "{bytes}: {a} {b}");
        if bytes >= 17 * G {
            // equal slots per layer within one round
            let (pa, pb) = (a as f64 / 19.0, b as f64 / 22.0);
            assert!((pa - pb).abs() < 1.0, "{bytes}: {pa} {pb}");
        }
    }
    let only_b = Classes {
        layers: [0, 22],
        ..c
    };
    assert_eq!(only_b.split(10 * c.slot[1] + 1), [0, 10]);
    let only_a = Classes {
        layers: [19, 0],
        ..c
    };
    assert_eq!(only_a.split(10 * c.slot[0] + 1), [10, 0]);
    let none = Classes {
        layers: [0, 0],
        ..c
    };
    assert_eq!(none.split(G), [0, 0]);
}

#[test]
fn chunk_env_values() {
    assert_eq!(chunk_cap(None).unwrap(), None);
    assert_eq!(chunk_cap(Some(" ")).unwrap(), None);
    assert_eq!(chunk_cap(Some("4096")).unwrap(), Some(4096));
    assert_eq!(chunk_cap(Some(" 256 ")).unwrap(), Some(256));
    for v in ["0", "128", "1000", "-256", "abc", "4096.0"] {
        let e = chunk_cap(Some(v)).unwrap_err().reason;
        assert!(e.contains(CHUNK_ENV), "{v}: {e}");
    }
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    Scan::open(&paths).unwrap()
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_sizes_match_the_oracle() {
    let s = real_scan();
    let sizes = Sizes::of(&s).unwrap();
    let addons = fixture("admission-addons.json");
    let inputs = &addons["inputs"];
    assert_eq!(sizes.dense_trunk, u(&inputs["dense_trunk"]));
    assert_eq!(sizes.dense_mtp, u(&inputs["dense_mtp_plan"]));
    let mtp = fixture("mtp-tensors.json");
    assert_eq!(sizes.dense_mtp, u(&mtp["totals"]["plan_bytes"]));
    let adm = fixture("admission.json");
    assert_eq!(
        sizes.classes.slot,
        [u(&adm["inputs"]["slot_a"]), u(&adm["inputs"]["slot_b"])]
    );
    assert_eq!(sizes.classes.layers, [19, 22]);
    let m = sizes.model(true, 262_144);
    assert_eq!(
        (m.dense, m.slot, m.n_experts),
        (
            u(&addons["variants"]["mtp_on"]["dense"]),
            u(&inputs["slot"]),
            u(&inputs["n_experts"])
        )
    );
    assert_eq!(sizes.model(false, 1).dense, sizes.dense_trunk);
    let pf = &addons["profile"];
    assert_eq!(kv_per_token(&s.cfg, true), u(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(kv_per_token(&s.cfg, false), u(&pf["kv_per_ctx_mtp_off"]));
}

/// Real headers (CPU): the lane floor with the MTP layer and the Q8
/// shapes the row cap is taken over.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_lane_floor_and_wave_shapes() {
    let s = real_scan();
    assert_eq!(
        lane_floor(&s.cfg, true),
        30 * 515 * 576 * 2 + 13 * 6 * 128 * 4 + 3072 * 4
    );
    assert_eq!(
        lane_floor(&s.cfg, false),
        30 * 515 * 576 * 2 + 12 * 6 * 128 * 4 + 3072 * 4
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
        assert!(shapes.contains(&want), "{want:?} in {shapes:?}");
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
    TEST_HOST.with(|c| c.set(Some(host(&inputs, 20 * G))));
    let opts = Options {
        max_context: None,
        mtp: true,
        chunk_cap: None,
        lanes: LANES_MAX,
        plan: CachePlan::AdmitWith(Limits::default()),
    };
    let st = Store::open(&s.paths, &opts).unwrap();
    TEST_HOST.with(|c| c.set(None));
    assert_eq!((st.ctx, st.chunk, st.lanes), (262_144, 8192, LANES_MAX));
    assert!(st.row_cap >= 2, "row cap {}", st.row_cap);
    let a = st.admission().unwrap();
    assert_eq!(st.cache_slots(), st.sizes.classes.split(a.cache_bytes()));
    assert_eq!(st.bank.staging_layers() as u64, a.staging_layers);
    assert_eq!(st.dense_bytes, st.sizes.dense_trunk);
    assert!(st.model.new_session(st.ctx).is_ok());
}

struct Temp(PathBuf);

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn moe() -> MoeConfig {
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

fn layer(m: &MoeConfig, il: usize, class: ExpertClass) -> ExpertLayer {
    let types = class_types(class).unwrap();
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let loc = |p: usize, name: &str| Located {
        shard: 0,
        data_offset: 0,
        info: GgufTensorInfo {
            name: format!("blk.{il}.{name}.weight"),
            n_dims: 3,
            dims: dims[p].iter().map(|&d| d as u64).collect(),
            tensor_type: types[p],
            offset: 0,
        },
    };
    ExpertLayer {
        class,
        gate: loc(0, "ffn_gate_exps"),
        up: loc(1, "ffn_up_exps"),
        down: loc(2, "ffn_down_exps"),
    }
}

/// A two-class bank of 4 experts whose slots are never read.
fn tiny_bank(cache: [usize; 2], staging: usize) -> (KqBank, Classes, Temp) {
    let m = moe();
    let experts = vec![
        Some(layer(&m, 0, ExpertClass::Q6kDown)),
        Some(layer(&m, 1, ExpertClass::Q4kDown)),
        Some(layer(&m, 2, ExpertClass::Q4kDown)),
    ];
    let dir = std::env::temp_dir().join(format!("hy4-store-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("experts.bin");
    std::fs::write(&path, [0u8; 64]).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 1).unwrap();
    let bank = KqBank::new(pool, &m, &experts, cache, staging).unwrap();
    let slot = CLASSES.map(|c| {
        geometry(&m, c)
            .unwrap()
            .iter()
            .map(|g| g.slot_bytes())
            .sum::<u64>()
    });
    assert_eq!(slot, bank.slot_bytes());
    (
        bank,
        Classes {
            layers: [1, 2],
            slot,
        },
        Temp(dir),
    )
}

/// A staging-only bank is rebuilt at the first turn the host has room in,
/// kept at the next, and a refused turn leaves the bank volatile.
#[test]
fn lease_rebuilds_keeps_and_refuses() {
    let (bank, classes, _dir) = tiny_bank([0, 0], 1);
    let p = profile(100);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: 1024,
    };
    let rows = 1024;
    let cap = 64 * size.slot;
    let limits = Limits {
        cache_cap: Some(cap),
        ..Limits::default()
    };
    let roomy = (
        HostMemory {
            physical: 64 * G,
            wired: 10 * G,
            ..Default::default()
        },
        48 * G,
    );
    let loaded = admit_with(&p, &size, rows, &limits, &roomy.0, &Own::default(), roomy.1).unwrap();
    assert_eq!((loaded.cache_slots, loaded.bound), (64, Bound::Env));
    let staging_only = Admission {
        staging_layers: 1,
        fixed: loaded.terms.fixed(1),
        cache_slots: 0,
        ..loaded
    };
    let lease = Lease::new(
        p,
        size,
        rows,
        classes,
        Some(limits),
        Some(staging_only),
        &bank,
    );
    bank.end_turn().unwrap();
    TEST_HOST.with(|c| c.set(Some(roomy)));
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(lease.in_turn() && bank.in_turn());
    let a = lease.admission().unwrap();
    assert_eq!(a.cache_slots, 64);
    assert_eq!(bank.cache_slots(), classes.split(a.cache_bytes()));
    assert_eq!(bank.staging_layers() as u64, a.staging_layers);
    assert_eq!(lease.switches().len(), 1);
    lease.end_turn(&bank).unwrap();
    assert!(!lease.in_turn() && !bank.in_turn());
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(lease.admission().unwrap(), a);
    assert_eq!(lease.switches().len(), 1);
    lease.end_turn(&bank).unwrap();
    let full = (
        HostMemory {
            wired: 60 * G,
            ..roomy.0
        },
        roomy.1,
    );
    TEST_HOST.with(|c| c.set(Some(full)));
    let e = lease.begin_turn(&bank, 0, 0).unwrap_err().reason;
    TEST_HOST.with(|c| c.set(None));
    assert!(e.contains("vs budget"), "{e}");
    assert!(!lease.in_turn() && !bank.in_turn());
    assert_eq!(lease.admission().unwrap(), a);
}

/// Without limits the lease never re-admits or switches.
#[test]
fn lease_without_admission_only_flips_the_bank() {
    let (bank, classes, _dir) = tiny_bank([2, 2], 1);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0],
        n_experts: 8,
        ctx: 64,
    };
    let lease = Lease::new(profile(100), size, 64, classes, None, None, &bank);
    bank.end_turn().unwrap();
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(bank.in_turn() && !lease.in_turn());
    assert_eq!(bank.cache_slots(), [2, 2]);
    lease.end_turn(&bank).unwrap();
    assert!(!bank.in_turn());
    assert!(lease.switches().is_empty());
}
