//! The product load of the real Qwen3.8-27B UD-Q4_K_XL GGUF with its DFlash2
//! draft (ignored: it loads the 27B model). Its assets go to a temporary
//! root and its SSD tier is off, so nothing outside the test is written. The
//! load's planes lines are read back from `gguf_planes::LOAD_LINES`.
use super::*;

use crate::engine::checkpoint_source::open_gguf_main;
use crate::utils::gguf_planes::{PlanesDevice, PlanesLoad, Walk, plan_for_load};

/// The Qwen3.8-27B UD-Q4_K_XL GGUF that `MLX_TEST_QWEN38_27B_GGUF` names.
fn real_gguf() -> String {
    std::env::var("MLX_TEST_QWEN38_27B_GGUF").unwrap_or_default()
}

/// Its DFlash2 draft directory, named by `MLX_TEST_QWEN38_27B_DRAFT`.
fn real_draft() -> String {
    std::env::var("MLX_TEST_QWEN38_27B_DRAFT").unwrap_or_default()
}

fn real_model_present() -> bool {
    Path::new(&real_gguf()).is_file() && Path::new(&real_draft()).is_dir()
}
const GIB: u64 = 1 << 30;

/// The real header plans nothing for a load with no draft or under a 40 GiB
/// cap; the product load with the draft and the default activations plans
/// the 164 tensors of the plan of record on the product walk, admits them
/// before the import, writes and attaches them, keeps them under the pool
/// guard with its ledger line, and names
/// the store identity `Int8`.
#[test]
#[ignore = "loads the real 27B model"]
fn real_default_load_attaches_the_plan_of_record() -> Result<()> {
    if !real_model_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF / MLX_TEST_QWEN38_27B_DRAFT name no model");
        return Ok(());
    }
    let main = open_gguf_main(&[real_gguf().into()])?;
    let app = PlanesLoad {
        draft: true,
        clef: false,
        files: 1,
        walk: Walk::Product,
    };
    for (what, load, cap) in [
        (
            "no draft",
            PlanesLoad {
                draft: false,
                ..app
            },
            None,
        ),
        ("capped 40 GiB", app, Some(40 * GIB)),
    ] {
        let plan = plan_for_load(&main.header, &PlanesDevice::this(cap), &load);
        println!("CONDITION {what}: {}", plan.describe());
        assert!(plan.tensors.is_empty(), "{what} plans nothing");
    }

    crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    let assets = std::env::temp_dir().join(format!("mlx-planes-real-load-{}", std::process::id()));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let source = runtime.block_on(CheckpointSource::resolve_in(
        Path::new(&real_gguf()),
        crate::utils::gguf::NativeGgufFamily::Qwen35,
        Some(&assets),
    ))?;
    let started = std::time::Instant::now();
    let loaded = runtime.block_on(load_with_thread_impl(
        &source,
        Some(real_draft()),
        None,
        crate::transformer::Activations::Bf16,
        None,
        None,
        crate::engine::scheduler::DEFAULT_DECODE_SHARE,
        crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Off,
            ..Default::default()
        },
        LoadKind::Standard,
        false,
        crate::engine::load_progress::LoadProgressCell::detached(),
    ));
    source.release_lease();
    let model = loaded?;
    println!("LOAD secs={:.1}", started.elapsed().as_secs_f64());
    drop(model);
    drop(runtime);
    let _ = std::fs::remove_dir_all(&assets);

    let lines = crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let mut planes_line = None;
    let mut ledger_line = None;
    let mut identity = None;
    for line in &lines {
        println!("LOG {line}");
        if let Some((_, rest)) = line.split_once("GGUF planes: ") {
            planes_line = Some(rest.to_string());
        }
        if let Some((_, rest)) = line.split_once("Qwen3.5 memory plan: planes ") {
            ledger_line = Some(rest.to_string());
        }
        if let Some((_, rest)) = line.split_once("Qwen3.5 state store identity: ") {
            identity = Some(rest.trim_end().to_string());
        }
    }
    assert!(
        lines.iter().any(|line| line.starts_with(
            "GGUF planes admitted before the import: planes 9885491200 B in the weights: pool "
        )),
        "the import guard admits the planes before the import"
    );
    let planes_line = planes_line.expect("the load logs its planes plan");
    assert!(
        planes_line.starts_with("164 tensors, 9885491200 bytes, Product walk"),
        "{planes_line}"
    );
    let ledger_line = ledger_line.expect("the load logs its planes ledger line");
    assert!(
        ledger_line.starts_with("9885491200 B in the weights: pool "),
        "{ledger_line}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("GGUF planes dropped")),
        "the guard keeps the planes on this machine"
    );
    assert_eq!(identity.as_deref(), Some("Int8"), "the store identity");
    Ok(())
}

/// Runs on the model thread at the end of a load, before the model serves.
type LoadedHook = Box<dyn FnOnce(&mut Qwen35Inner) -> Result<()> + Send>;

static LOADED_HOOK: std::sync::Mutex<Option<LoadedHook>> = std::sync::Mutex::new(None);

pub(super) fn run_loaded_hook(inner: &mut Qwen35Inner) -> Result<()> {
    let hook = LOADED_HOOK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    match hook {
        Some(hook) => hook(inner),
        None => Ok(()),
    }
}

/// The product load of the real model and draft with `hook` run on its
/// model thread; prints the load's planes lines.
fn load_real_with(hook: LoadedHook) -> Result<()> {
    crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    *LOADED_HOOK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    let assets = std::env::temp_dir().join(format!("mlx-planes-verify-{}", std::process::id()));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let source = runtime.block_on(CheckpointSource::resolve_in(
        Path::new(&real_gguf()),
        crate::utils::gguf::NativeGgufFamily::Qwen35,
        Some(&assets),
    ))?;
    let started = std::time::Instant::now();
    let loaded = runtime.block_on(load_with_thread_impl(
        &source,
        Some(real_draft()),
        None,
        crate::transformer::Activations::Bf16,
        None,
        None,
        crate::engine::scheduler::DEFAULT_DECODE_SHARE,
        crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Off,
            ..Default::default()
        },
        LoadKind::Standard,
        false,
        crate::engine::load_progress::LoadProgressCell::detached(),
    ));
    source.release_lease();
    let model = loaded?;
    println!("LOAD secs={:.1}", started.elapsed().as_secs_f64());
    drop(model);
    drop(runtime);
    let _ = std::fs::remove_dir_all(&assets);
    for line in crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        println!("LOG {line}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod verify_route_timing {
    use super::*;
    use crate::models::quantized_linear::{
        AttachedPlanes, MLPVariant, PLANES_ROUTE_OFF, planes_calls,
    };
    use crate::models::qwen3_5::dflash2::DFLASH2_VERIFY_ROWS;
    use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;
    use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
    use crate::models::qwen3_5::paged_forward::{VerifyInput, run_paged_dflash2_verify_step};
    use crate::models::qwen3_5::prefill_kernel_timing::{
        bf16_input, fill_to, linear_caches, run_stack,
    };
    use crate::utils::gguf_planes::{PlanesDevice, PlanesLoad, PlanesPlan, PlanesTensor, Walk};
    use std::time::Instant;

    const SHALLOW: u32 = 24_576;
    const DEEP: u32 = 145_408;
    const CHUNK: u32 = 2048;
    const BLOCK: u32 = 16;
    const WINDOW: u32 = 262_144;

    struct Stack {
        kinds: Vec<crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind>,
        taps: Vec<usize>,
        adapter: PagedKVCacheAdapter,
        caches: Vec<Qwen3_5LayerCache>,
        blobs: GdnStateBlobs,
        attached: u64,
        cycle: u32,
    }

    /// One verify of 8 host ids at the stack's depth, its logits and every
    /// GDN slot evaluated, then rolled back (0 rows kept): wall ms, planes
    /// calls and the logits.
    fn cycle(
        inner: &mut Qwen35Inner,
        s: &mut Stack,
        route_on: bool,
    ) -> Result<(f64, u64, MxArray)> {
        let vocab = inner.config.vocab_size as u32;
        let ids: Vec<u32> = (0..DFLASH2_VERIFY_ROWS as u32)
            .map(|j| (s.cycle * 131 + j * 7919 + 17) % vocab)
            .collect();
        s.cycle += 1;
        let mut tape = Vec::new();
        PLANES_ROUTE_OFF.with(|off| off.set(!route_on));
        let before = planes_calls();
        let start = Instant::now();
        let built = run_paged_dflash2_verify_step(
            VerifyInput::Host(&ids),
            &inner.embedding,
            &mut inner.layers,
            &mut s.caches,
            &inner.final_norm,
            &inner.lm_head,
            &s.kinds,
            &mut s.adapter,
            &mut tape,
            0,
            &s.taps,
            &s.blobs,
            None,
            WINDOW,
        );
        PLANES_ROUTE_OFF.with(|off| off.set(false));
        let (logits, taps) = built?;
        let calls = planes_calls() - before;
        let mut outs = vec![&logits];
        outs.extend(taps.iter());
        let slots: Vec<MxArray> = s
            .caches
            .iter()
            .filter_map(|c| match c {
                Qwen3_5LayerCache::Linear(arrays) => {
                    Some((0..2).filter_map(|i| arrays.get(i).cloned()))
                }
                Qwen3_5LayerCache::FullAttention(_) => None,
            })
            .flatten()
            .collect();
        outs.extend(slots.iter());
        MxArray::eval_arrays(&outs)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        drop(outs);
        s.adapter
            .rollback_last_tokens(DFLASH2_VERIFY_ROWS as u32)
            .map_err(Error::from_reason)?;
        s.blobs.apply_views(&mut s.caches)?;
        Ok((ms, calls, logits))
    }

    fn logits_bits(logits: &MxArray) -> Result<Vec<u32>> {
        let x = logits.astype(DType::Float32)?;
        x.eval();
        Ok(x.to_float32()?.iter().map(|v| v.to_bits()).collect())
    }

    /// The walk's rows at the stack's depth: one route-off / route-on pair
    /// from the same state (EQ), two warm cycles per arm, then `rounds`
    /// rounds of one off and one on cycle, the order alternating per round
    /// (round 0 starts with the arm `start_on` names).
    fn rows(
        inner: &mut Qwen35Inner,
        s: &mut Stack,
        walk: &str,
        rounds: usize,
        start_on: bool,
    ) -> Result<()> {
        let depth = s.adapter.current_token_count();
        let (_, off_calls, off) = cycle(inner, s, false)?;
        s.cycle -= 1;
        let (_, on_calls, on) = cycle(inner, s, true)?;
        let (a, b) = (logits_bits(&off)?, logits_bits(&on)?);
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!(
            "ROUTEEQ walk={walk} depth={depth} logits_differ={differ} of {} off_calls={off_calls} on_calls={on_calls} attached={}",
            a.len(),
            s.attached
        );
        assert_eq!(off_calls, 0, "the route-off arm runs no planes");
        assert_eq!(
            on_calls, s.attached,
            "the route-on arm runs every attached projection"
        );
        for _ in 0..2 {
            for on in [false, true] {
                cycle(inner, s, on)?;
            }
        }
        for round in 0..rounds {
            let first_on = (round + usize::from(start_on)) % 2 == 1;
            let order = if first_on {
                [true, false]
            } else {
                [false, true]
            };
            let mut ms = [0.0f64; 2];
            let mut calls = [0u64; 2];
            for on in order {
                let (t, c, _) = cycle(inner, s, on)?;
                ms[usize::from(on)] = t;
                calls[usize::from(on)] = c;
            }
            println!(
                "ROUTEROW walk={walk} depth={depth} round={round} first={} off_ms={:.3} on_ms={:.3} off_calls={} on_calls={}",
                if first_on { "on" } else { "off" },
                ms[0],
                ms[1],
                calls[0],
                calls[1]
            );
        }
        Ok(())
    }

    /// Re-attaches the planes of `attached` on `walk` (the spread walk takes
    /// each tensor's walk from `spread`).
    fn reattach(
        inner: &mut Qwen35Inner,
        attached: &[(MxArray, PlanesTensor)],
        walk: Walk,
        spread: &HashMap<String, bool>,
    ) -> Result<()> {
        inner.detach_planes();
        let mut tensors = Vec::new();
        let mut arrays = HashMap::new();
        for (array, t) in attached {
            let mut t = t.clone();
            t.walk = match walk {
                Walk::Product => false,
                Walk::Spread => *spread
                    .get(&t.key)
                    .ok_or_else(|| Error::from_reason(format!("{} has no spread walk", t.key)))?,
            };
            arrays.insert(t.key.clone(), array.clone());
            tensors.push(t);
        }
        inner.attach_planes(
            &PlanesPlan {
                tensors,
                walk,
                left_out: Vec::new(),
            },
            arrays,
        )?;
        let walks = inner
            .attached_planes()
            .iter()
            .filter(|p| p.tensor.walk)
            .count();
        println!(
            "ROUTEWALK {walk:?} attached={} walk_on={walks}",
            inner.attached_planes().len()
        );
        Ok(())
    }

    /// The plain GEMMs of every attached projection at 8 rows, in one graph
    /// and one eval: median wall ms of 15 runs after 2.
    fn planned_gemms(inner: &Qwen35Inner) -> Result<()> {
        let x_hidden = bf16_input(&[1, 8, 5120], 81)?.mul_scalar(0.1)?;
        let x_ffn = bf16_input(&[1, 8, 17408], 82)?.mul_scalar(0.1)?;
        let x_mixer = bf16_input(&[1, 8, 6144], 83)?.mul_scalar(0.1)?;
        MxArray::eval_arrays(&[&x_hidden, &x_ffn, &x_mixer])?;
        let build = || -> Result<Vec<MxArray>> {
            let mut outs = Vec::new();
            let planned = |p: Option<&AttachedPlanes>| p.is_some();
            for layer in &inner.layers {
                if let crate::models::qwen3_5::decoder_layer::MLPType::Dense(
                    MLPVariant::Quantized {
                        gate_proj,
                        up_proj,
                        down_proj,
                        gate_up,
                    },
                ) = &layer.mlp
                {
                    if let Some(pair) = gate_up
                        && planned(pair.0.planes())
                    {
                        outs.push(pair.0.forward(&x_hidden)?);
                    }
                    for p in [gate_proj, up_proj] {
                        if planned(p.planes()) {
                            outs.push(p.forward(&x_hidden)?);
                        }
                    }
                    if planned(down_proj.planes()) {
                        outs.push(down_proj.forward(&x_ffn)?);
                    }
                }
                let out = match &layer.attn {
                    AttentionType::Linear(gdn) => gdn.out_proj(),
                    AttentionType::Full(attn) => attn.o_proj(),
                };
                if planned(out.planes()) {
                    outs.push(out.forward(&x_mixer)?);
                }
            }
            Ok(outs)
        };
        let mut samples = Vec::new();
        let mut calls = 0;
        for run in 0..17 {
            let outs = build()?;
            calls = outs.len();
            let refs: Vec<&MxArray> = outs.iter().collect();
            let start = Instant::now();
            MxArray::eval_arrays(&refs)?;
            if run >= 2 {
                samples.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "ROUTEGEMM calls={calls} median_ms={:.3} min_ms={:.3}",
            samples[samples.len() / 2],
            samples[0]
        );
        Ok(())
    }

    fn memory_row(tag: &str) {
        let (footprint, peak, _) = crate::models::deepseek_v41::stats::rusage();
        println!(
            "ROUTEMEM {tag} footprint={footprint} peak={peak} active={}",
            crate::array::memory::active_memory_bytes()
        );
    }

    fn rig(
        inner: &mut Qwen35Inner,
        start_on: bool,
        rounds: usize,
        spread_deep: bool,
        spread: HashMap<String, bool>,
    ) -> Result<()> {
        let setup = Instant::now();
        let attached: Vec<(MxArray, PlanesTensor)> = inner
            .attached_planes()
            .iter()
            .map(|p| (p.array.clone(), p.tensor.clone()))
            .collect();
        let walk = inner.attached_planes_plan().walk;
        println!(
            "ROUTESETUP attached={} bytes={} walk={walk:?} start={}",
            attached.len(),
            inner.attached_planes_bytes(),
            if start_on { "on" } else { "off" }
        );
        if attached.is_empty() || walk != Walk::Product {
            return Err(Error::from_reason(
                "the load attached no product-walk planes",
            ));
        }
        memory_row("loaded_with_planes");
        let cfg = inner.config.clone();
        let attn_layers = cfg.full_attention_layer_count() as u32;
        if attn_layers != 16 {
            return Err(Error::from_reason(format!(
                "{attn_layers} attention layers, not 16"
            )));
        }
        let kinds = inner.layer_kinds.clone();
        let taps = inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("no DFlash2 draft"))?
            .config
            .target_layers
            .clone();
        let pool_tokens = DEEP + 8192;
        let blocks = pool_tokens / BLOCK + 64;
        let (mut adapter, _pool) = crate::transformer::paged_int8_tests::adapter_for(
            mlx_paged_attn::PagedAttentionConfig {
                block_size: BLOCK,
                gpu_memory_mb: 8192,
                head_size: cfg.head_dim as u32,
                num_kv_heads: cfg.num_kv_heads as u32,
                num_layers: attn_layers,
                max_seq_len: Some(WINDOW),
                max_batch_size: Some(1),
            },
            blocks,
            blocks,
            mlx_paged_attn::metal::MetalDtype::Int8,
        )
        .ok_or_else(|| Error::from_reason("no int8 pool"))?;
        adapter
            .reset_for_new_request(1)
            .map_err(Error::from_reason)?;
        adapter
            .allocate_suffix_blocks(pool_tokens)
            .map_err(Error::from_reason)?;
        let kv = (
            bf16_input(
                &[CHUNK as i64, cfg.num_kv_heads as i64, cfg.head_dim as i64],
                50,
            )?,
            bf16_input(
                &[CHUNK as i64, cfg.num_kv_heads as i64, cfg.head_dim as i64],
                51,
            )?,
        );
        fill_to(&mut adapter, &kv, SHALLOW)?;
        let vocab = cfg.vocab_size as u32;
        let ids: Vec<u32> = (0..CHUNK).map(|i| (i * 7919 + 13) % vocab).collect();
        let x = inner
            .embedding
            .forward(&MxArray::from_uint32(&ids, &[1, CHUNK as i64])?)?;
        let mut caches = linear_caches(&kinds);
        run_stack(
            &mut inner.layers,
            &kinds,
            &mut caches,
            &mut adapter,
            &x,
            SHALLOW - CHUNK,
            8,
        )?;
        let tiled = cfg.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
        let blobs = GdnStateBlobs::pack(&caches, tiled)?
            .ok_or_else(|| Error::from_reason("the GDN slots did not pack"))?;
        blobs.apply_views(&mut caches)?;
        let mut s = Stack {
            kinds,
            taps,
            adapter,
            caches,
            blobs,
            attached: attached.len() as u64,
            cycle: 0,
        };
        println!(
            "ROUTESETUP secs={:.1} depth={}",
            setup.elapsed().as_secs_f64(),
            s.adapter.current_token_count()
        );

        let t0 = Instant::now();
        rows(inner, &mut s, "product", rounds, start_on)?;
        planned_gemms(inner)?;
        reattach(inner, &attached, Walk::Spread, &spread)?;
        rows(inner, &mut s, "spread", rounds, start_on)?;
        println!("ROUTETIME shallow secs={:.1}", t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        fill_to(&mut s.adapter, &kv, DEEP)?;
        println!(
            "ROUTETIME fill secs={:.1} depth={}",
            t1.elapsed().as_secs_f64(),
            s.adapter.current_token_count()
        );
        let t2 = Instant::now();
        if spread_deep {
            rows(inner, &mut s, "spread", rounds, start_on)?;
        } else {
            println!("ROUTECUT spread deep not run");
        }
        reattach(inner, &attached, Walk::Product, &spread)?;
        rows(inner, &mut s, "product", rounds, start_on)?;
        println!("ROUTETIME deep secs={:.1}", t2.elapsed().as_secs_f64());

        memory_row("rig_with_planes");
        drop(attached);
        let detached = inner.detach_planes();
        crate::array::memory::synchronize_and_clear_cache();
        memory_row("rig_without_planes");
        println!("ROUTEMEM detached_bytes={detached}");
        Ok(())
    }

    /// The whole 8-row DFlash2 verify of the real 64-block stack, route on
    /// against route off by the test switch, at 24,576 and 145,408 rows of
    /// history, on the product walk the load attached and on the spread
    /// walk (the same planes re-attached with the plan's spread walk).
    /// Rounds alternate the arm order; `start_on` names round 0's first arm.
    fn verify_route_saving(start_on: bool) -> Result<()> {
        if !real_model_present() {
            println!("SKIP: MLX_TEST_QWEN38_27B_GGUF / MLX_TEST_QWEN38_27B_DRAFT name no model");
            return Ok(());
        }
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let rounds = env("VERIFY_ROUTE_ROUNDS", 31);
        let spread_deep = env("VERIFY_ROUTE_SPREAD_DEEP", 1) == 1;
        let main = open_gguf_main(&[real_gguf().into()])?;
        let plan = plan_for_load(
            &main.header,
            &PlanesDevice::this(None),
            &PlanesLoad {
                draft: true,
                clef: false,
                files: 1,
                walk: Walk::Spread,
            },
        );
        let spread: HashMap<String, bool> = plan
            .tensors
            .iter()
            .map(|t| (t.key.clone(), t.walk))
            .collect();
        println!(
            "ROUTEPLAN spread tensors={} walk_on={} rounds={rounds} spread_deep={spread_deep}",
            spread.len(),
            spread.values().filter(|&&w| w).count()
        );
        drop(main);
        load_real_with(Box::new(move |inner: &mut Qwen35Inner| {
            rig(inner, start_on, rounds, spread_deep, spread)
        }))
    }

    #[test]
    #[ignore = "loads the real 27B model; timed rows"]
    fn verify_route_saving_starting_off() -> Result<()> {
        verify_route_saving(false)
    }

    #[test]
    #[ignore = "loads the real 27B model; timed rows"]
    fn verify_route_saving_starting_on() -> Result<()> {
        verify_route_saving(true)
    }
}
