//! The planes route against the Tiled64 route on the real Qwen3.8-27B GGUF
//! (ignored: they read the 27B file).
use super::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use crate::engine::checkpoint_source::{native_import_plans, open_gguf_main};
use crate::models::quant_dispatch::{KQuantLayout, parse_mode_str};
use crate::models::quantized_linear::{QuantizedLinear, try_build_kquant_quantized_linear_tiled};
use crate::utils::gguf::{GgufFile, NativeGgufFamily};
use crate::utils::gguf_import::{ImportProbe, RawTensors, ReadConfig, RepackBackend};
use crate::utils::gguf_planes::{
    Kernel, PlanesDevice, PlanesPlan, PlanesTensor, Role, plan_planes, prepare_planes,
};
use crate::utils::gguf_read::PageCacheMode;
use crate::utils::gguf_source::GgufSourceSet;

const FFN: u32 = 17408;
const MIXER: u32 = 6144;
/// `gguf_planes::testing::kWalkOff`.
const WALK_OFF: i32 = 6;

/// The Qwen3.8-27B UD-Q4_K_XL GGUF that `MLX_TEST_QWEN38_27B_GGUF` names.
fn real_gguf() -> String {
    std::env::var("MLX_TEST_QWEN38_27B_GGUF").unwrap_or_default()
}

fn real_gguf_present() -> bool {
    std::path::Path::new(&real_gguf()).is_file()
}

fn import_real(probe: &ImportProbe) -> Result<(RawTensors, BTreeSet<String>)> {
    let main = open_gguf_main(&[real_gguf().into()])?;
    let (path, header, identity) = &main.parts[0];
    let (_, plan, _) = native_import_plans(NativeGgufFamily::Qwen35, (path, &main.header), None)?;
    let set = GgufSourceSet::open(&[(path, header.clone(), identity)], None)?;
    let handle = &set.main[0];
    let raw = crate::utils::gguf_import::read_gguf_tensors_with(
        &handle.file,
        handle,
        &plan,
        RepackBackend::Auto,
        PageCacheMode::Auto,
        &ReadConfig::default(),
        probe,
    )?;
    Ok((raw, plan.tiled_gguf_names.clone()))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Cur,
    Pp,
    Ps,
}

impl Arm {
    fn name(self, lanes: i64) -> &'static str {
        match self {
            Arm::Cur if lanes == 1 => "cur",
            Arm::Cur => "tier",
            Arm::Pp => "Pp",
            Arm::Ps => "Ps",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Call,
    Chain,
}

enum RealWeights {
    Merged {
        pair: (QuantizedLinear, i64),
        planes: MxArray,
    },
    Pair {
        gate: QuantizedLinear,
        up: QuantizedLinear,
        pg: MxArray,
        pu: MxArray,
        up_format: u32,
    },
    Single {
        ql: QuantizedLinear,
        planes: MxArray,
    },
}

struct RealCall {
    group: String,
    tensors: Vec<PlanesTensor>,
    w: RealWeights,
}

struct LaneInputs {
    x5: MxArray,
    xd: MxArray,
    xm: MxArray,
    res: MxArray,
    x5_3: MxArray,
    xd_3: MxArray,
    xm_3: MxArray,
    res_3: MxArray,
}

fn lane_inputs(lanes: i64) -> Result<LaneInputs> {
    let rows = 8 * lanes;
    let seed = 0x5e00 + lanes as u64 * 16;
    let x5 = bf16_input(&[rows, i64::from(HIDDEN)], seed)?;
    let xd = bf16_input(&[rows, i64::from(FFN)], seed + 1)?;
    let xm = bf16_input(&[rows, i64::from(MIXER)], seed + 2)?;
    let res = bf16_input(&[rows, i64::from(HIDDEN)], seed + 3)?;
    let three = |a: &MxArray, k: u32| -> Result<MxArray> {
        let y = a.reshape(&[lanes, 8, i64::from(k)])?;
        y.eval();
        Ok(y)
    };
    Ok(LaneInputs {
        x5_3: three(&x5, HIDDEN)?,
        xd_3: three(&xd, FFN)?,
        xm_3: three(&xm, MIXER)?,
        res_3: three(&res, HIDDEN)?,
        x5,
        xd,
        xm,
        res,
    })
}

fn real_projection(raw: &RawTensors, header: &GgufFile, name: &str) -> Result<QuantizedLinear> {
    let prefix = name
        .strip_suffix(".weight")
        .ok_or_else(|| err(format!("{name} is not a weight")))?;
    let info = header
        .tensors
        .iter()
        .find(|i| i.name == name)
        .ok_or_else(|| err(format!("header has no {name}")))?;
    let get = |key: String| {
        raw.map
            .get(&key)
            .cloned()
            .ok_or_else(|| err(format!("import has no {key}")))
    };
    let params = HashMap::from([
        ("p.weight".to_string(), get(name.to_string())?),
        ("p.scales".to_string(), get(format!("{prefix}.scales"))?),
        ("p.biases".to_string(), get(format!("{prefix}.biases"))?),
    ]);
    let mode = parse_mode_str(info.tensor_type.k_quant_format().map(|f| f.mlx_mode()))
        .ok_or_else(|| err(format!("{name}: no K-quant mode")))?;
    let mut tiled = Vec::new();
    try_build_kquant_quantized_linear_tiled(
        &params,
        "p",
        mode,
        KQuantLayout::Tiled64,
        "qwen3_5",
        &mut tiled,
    )?
    .ok_or_else(|| err(format!("{name}: built nothing")))
}

fn real_calls(raw: &RawTensors, header: &GgufFile, plan: &PlanesPlan) -> Result<Vec<RealCall>> {
    let planes_at = |key: &str| {
        raw.map
            .get(key)
            .cloned()
            .ok_or_else(|| err(format!("import has no {key}")))
    };
    let mut calls = Vec::new();
    let mut i = 0;
    while i < plan.tensors.len() {
        let t = &plan.tensors[i];
        let call = match t.role {
            Role::GateUp => {
                let gate = real_projection(raw, header, &t.sources[0].0)?;
                let up = real_projection(raw, header, &t.sources[1].0)?;
                let ql = gate
                    .concat_rows(&up)?
                    .ok_or_else(|| err(format!("{}: gate|up do not merge", t.key)))?;
                RealCall {
                    group: format!("merged {}", fmt_name(t.format)),
                    tensors: vec![t.clone()],
                    w: RealWeights::Merged {
                        pair: (ql, i64::from(FFN)),
                        planes: planes_at(&t.key)?,
                    },
                }
            }
            Role::Gate => {
                let u = plan
                    .tensors
                    .get(i + 1)
                    .filter(|u| u.role == Role::Up)
                    .ok_or_else(|| err(format!("{}: no up after the gate", t.key)))?;
                RealCall {
                    group: format!("pair {}/{}", fmt_name(t.format), fmt_name(u.format)),
                    tensors: vec![t.clone(), u.clone()],
                    w: RealWeights::Pair {
                        gate: real_projection(raw, header, &t.sources[0].0)?,
                        up: real_projection(raw, header, &u.sources[0].0)?,
                        pg: planes_at(&t.key)?,
                        pu: planes_at(&u.key)?,
                        up_format: u.format,
                    },
                }
            }
            Role::Down | Role::Out | Role::O => {
                let role = match t.role {
                    Role::Down => "down",
                    Role::Out => "out",
                    _ => "o",
                };
                RealCall {
                    group: format!("{role} {}", fmt_name(t.format)),
                    tensors: vec![t.clone()],
                    w: RealWeights::Single {
                        ql: real_projection(raw, header, &t.sources[0].0)?,
                        planes: planes_at(&t.key)?,
                    },
                }
            }
            Role::Up => return Err(err(format!("{}: an up without its gate", t.key))),
        };
        i += call.tensors.len();
        calls.push(call);
    }
    Ok(calls)
}

fn swiglu_halves(y: &MxArray, axis: i32) -> Result<MxArray> {
    let gu = y.split_sections(&[i64::from(FFN)], axis)?;
    crate::nn::Activations::swiglu_compiled(&gu[0], &gu[1])
}

fn tiled(ql: &QuantizedLinear, x3: &MxArray, lanes: i64) -> Result<MxArray> {
    if lanes == 1 {
        ql.forward(x3)
    } else {
        ql.forward_lanes(x3)
    }
}

fn planes_call(c: &RealCall, inp: &LaneInputs) -> Result<MxArray> {
    let t = &c.tensors[0];
    let (f, splits, spread) = (t.format, t.splits, u32::from(t.walk));
    match &c.w {
        RealWeights::Merged { planes, .. } if t.kernel == Kernel::GateUp => {
            gate_up(&inp.x5, planes, f, FFN, HIDDEN, splits, spread)
        }
        RealWeights::Merged { planes, .. } => swiglu_halves(
            &mm(&inp.x5, planes, None, f, 2 * FFN, HIDDEN, 0, splits, spread)?,
            1,
        ),
        RealWeights::Pair {
            pg, pu, up_format, ..
        } => {
            let g = mm(&inp.x5, pg, None, f, FFN, HIDDEN, 0, splits, spread)?;
            mm(
                &inp.x5,
                pu,
                Some(&g),
                *up_format,
                FFN,
                HIDDEN,
                2,
                splits,
                spread,
            )
        }
        RealWeights::Single { planes, .. } => {
            let x = if t.k == FFN { &inp.xd } else { &inp.xm };
            mm(x, planes, Some(&inp.res), f, t.rows, t.k, 1, splits, spread)
        }
    }
}

fn build(c: &RealCall, arm: Arm, lanes: i64, inp: &LaneInputs) -> Result<MxArray> {
    match arm {
        Arm::Cur => match &c.w {
            RealWeights::Merged { pair, .. } => {
                swiglu_halves(&tiled(&pair.0, &inp.x5_3, lanes)?, 2)
            }
            RealWeights::Pair { gate, up, .. } => crate::nn::Activations::swiglu_compiled(
                &tiled(gate, &inp.x5_3, lanes)?,
                &tiled(up, &inp.x5_3, lanes)?,
            ),
            RealWeights::Single { ql, .. } => {
                let x3 = if c.tensors[0].k == FFN {
                    &inp.xd_3
                } else {
                    &inp.xm_3
                };
                inp.res_3.add(&tiled(ql, x3, lanes)?)
            }
        },
        Arm::Pp => {
            mutate(WALK_OFF);
            let out = planes_call(c, inp);
            mutate(0);
            out
        }
        Arm::Ps => planes_call(c, inp),
    }
}

fn outputs(calls: &[RealCall], arm: Arm, lanes: i64, inp: &LaneInputs) -> Result<Vec<MxArray>> {
    calls.iter().map(|c| build(c, arm, lanes, inp)).collect()
}

fn chain_ms(outs: &[MxArray]) -> Result<f64> {
    let mut handles: Vec<*mut mlx_sys::mlx_array> = outs.iter().map(|a| a.as_raw_ptr()).collect();
    // SAFETY: plain stream synchronize.
    unsafe { mlx_sys::mlx_synchronize() };
    let t0 = Instant::now();
    // SAFETY: live handles, `handles.len()` of them.
    let ok = unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), handles.len()) };
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_synchronize() };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if !ok {
        return Err(err("chain eval failed"));
    }
    Ok(ms)
}

fn cycle(
    calls: &[RealCall],
    arm: Arm,
    lanes: i64,
    inp: &LaneInputs,
    form: Form,
) -> Result<BTreeMap<String, f64>> {
    let outs = outputs(calls, arm, lanes, inp)?;
    let mut by_group: BTreeMap<String, f64> = BTreeMap::new();
    match form {
        Form::Call => {
            let seconds = gpu_seconds(&outs)?;
            let mut all = 0.0;
            for (c, s) in calls.iter().zip(&seconds) {
                *by_group.entry(c.group.clone()).or_insert(0.0) += s * 1e3;
                all += s * 1e3;
            }
            by_group.insert("ALL".to_string(), all);
        }
        Form::Chain => {
            by_group.insert("ALL".to_string(), chain_ms(&outs)?);
        }
    }
    Ok(by_group)
}

fn real_rows(process: u32, form: Form) -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    if !real_gguf_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF names no file");
        return Ok(());
    }
    let tag = match form {
        Form::Call => "call",
        Form::Chain => "chain",
    };
    let main = open_gguf_main(&[real_gguf().into()])?;
    let device = PlanesDevice::this(None);
    let plan = prepare_planes(plan_planes(&main.header, &device));
    println!("DEVICE {device:?}");
    println!("PLANDESC {}", plan.describe());
    println!("PLANID {}", plan.identity());
    assert!(!plan.tensors.is_empty(), "the app machine plans planes");
    let started = Instant::now();
    let probe = ImportProbe {
        planes: Some(plan.clone()),
        ..ImportProbe::default()
    };
    let (raw, tiled_names) = import_real(&probe)?;
    for t in &plan.tensors {
        for (source, _) in &t.sources {
            assert!(
                tiled_names.contains(source),
                "{source} is planned but not imported as Tiled64"
            );
        }
    }
    println!(
        "IMPORT tensors={} secs={:.1}",
        raw.map.len(),
        started.elapsed().as_secs_f64()
    );
    let calls = real_calls(&raw, &main.header, &plan)?;
    drop(raw);
    for c in &calls {
        for t in &c.tensors {
            println!(
                "PLANT {} | {} bytes={} kernel={:?} splits={} walk={}",
                c.group,
                t.key,
                t.bytes,
                t.kernel,
                t.splits,
                u8::from(t.walk)
            );
        }
        let holds = match &c.w {
            RealWeights::Merged { pair, .. } => pair.0.lane_tier_holds(HIDDEN.into())?,
            RealWeights::Pair { gate, up, .. } => {
                gate.lane_tier_holds(HIDDEN.into())? && up.lane_tier_holds(HIDDEN.into())?
            }
            RealWeights::Single { ql, .. } => ql.lane_tier_holds(c.tensors[0].k.into())?,
        };
        println!(
            "TIER {} | {} holds={}",
            c.group,
            c.tensors[0].key,
            u8::from(holds)
        );
    }
    println!(
        "TPLAN process={process} form={tag} calls={} tensors={} bytes={}",
        calls.len(),
        plan.tensors.len(),
        plan.bytes()
    );

    let lanes_list: &[i64] = match form {
        Form::Call => &[1, 2, 3, 4],
        Form::Chain => &[1],
    };
    let inputs: Vec<LaneInputs> = lanes_list
        .iter()
        .map(|&l| lane_inputs(l))
        .collect::<Result<_>>()?;

    let one = &inputs[0];
    let (mut pp_calls, mut ps_calls) = (0usize, 0usize);
    for c in &calls {
        let want = bits(&build(c, Arm::Cur, 1, one)?)?;
        let pp = bits(&build(c, Arm::Pp, 1, one)?)?;
        let ps = bits(&build(c, Arm::Ps, 1, one)?)?;
        let (pp_differ, pp_ulp) = compare(&pp, &want);
        let (ps_differ, ps_ulp) = compare(&ps, &want);
        pp_calls += usize::from(pp_differ != 0);
        ps_calls += usize::from(ps_differ != 0);
        let nonzero = want.iter().filter(|&&b| b & 0x7fff != 0).count();
        println!(
            "EQROW {process}-{tag} {} | {} calls_out={} nonzero={nonzero} pp_differ={pp_differ} pp_max_ulp={pp_ulp} ps_differ={ps_differ} ps_max_ulp={ps_ulp}",
            c.group,
            c.tensors[0].key,
            want.len()
        );
    }
    println!(
        "EQCOUNT {process}-{tag} calls={} pp_differ_calls={pp_calls} ps_differ_calls={ps_calls}",
        calls.len()
    );

    for (&lanes, inp) in lanes_list.iter().zip(&inputs) {
        for arm in [Arm::Cur, Arm::Pp, Arm::Ps] {
            cycle(&calls, arm, lanes, inp, form)?;
        }
    }
    let rows_started = Instant::now();
    for round in 0..16u32 {
        let arms = if (round + process).is_multiple_of(2) {
            [Arm::Cur, Arm::Pp, Arm::Ps]
        } else {
            [Arm::Ps, Arm::Pp, Arm::Cur]
        };
        for (&lanes, inp) in lanes_list.iter().zip(&inputs) {
            for arm in arms {
                let by_group = cycle(&calls, arm, lanes, inp, form)?;
                for (group, ms) in &by_group {
                    println!(
                        "SROW {tag} {process} {round} {lanes} {} {group} {ms:.6}",
                        arm.name(lanes)
                    );
                }
            }
        }
    }
    println!(
        "ROWS_DONE process={process} form={tag} secs={:.1}",
        rows_started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[test]
#[ignore = "reads the real 27B GGUF; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_real_rows_call_forward() -> Result<()> {
    real_rows(0, Form::Call)
}

#[test]
#[ignore = "reads the real 27B GGUF; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_real_rows_call_backward() -> Result<()> {
    real_rows(1, Form::Call)
}

#[test]
#[ignore = "reads the real 27B GGUF; default buffer limits"]
fn planes_real_rows_chain_forward() -> Result<()> {
    real_rows(0, Form::Chain)
}

#[test]
#[ignore = "reads the real 27B GGUF; default buffer limits"]
fn planes_real_rows_chain_backward() -> Result<()> {
    real_rows(1, Form::Chain)
}

/// Attaches each call's planes to its projections, as a load does.
fn attach_real(calls: &mut [RealCall]) -> Result<()> {
    for c in calls.iter_mut() {
        match &mut c.w {
            RealWeights::Merged { pair, planes } => {
                pair.0.attach_planes(planes.clone(), c.tensors[0].clone())?
            }
            RealWeights::Pair {
                gate, up, pg, pu, ..
            } => {
                gate.attach_planes(pg.clone(), c.tensors[0].clone())?;
                up.attach_planes(pu.clone(), c.tensors[1].clone())?;
            }
            RealWeights::Single { ql, planes } => {
                ql.attach_planes(planes.clone(), c.tensors[0].clone())?
            }
        }
    }
    Ok(())
}

/// The product's wave form of one call on `lanes` lanes: one planes call
/// over the lanes' stacked rows (3 lanes: 32 rows with a zero block).
fn wave_planes(c: &RealCall, lanes: i64, inp: &LaneInputs) -> Result<MxArray> {
    use crate::models::quantized_linear::{MLPVariant, WaveRows};
    let w = WaveRows::of_lanes(lanes);
    let none = || {
        err(format!(
            "{}: the wave form took no planes",
            c.tensors[0].key
        ))
    };
    match &c.w {
        RealWeights::Merged { pair, .. } => {
            let x2 = w.stack(&inp.x5_3)?;
            let a = MLPVariant::planes_activated_rows(&x2, &pair.0, &pair.0, Some(pair))?
                .ok_or_else(none)?;
            w.unstack(&a)
        }
        RealWeights::Pair { gate, up, .. } => {
            let x2 = w.stack(&inp.x5_3)?;
            let a = MLPVariant::planes_activated_rows(&x2, gate, up, None)?.ok_or_else(none)?;
            w.unstack(&a)
        }
        RealWeights::Single { ql, .. } => {
            let x3 = if c.tensors[0].k == FFN {
                &inp.xd_3
            } else {
                &inp.xm_3
            };
            ql.forward_verify_lanes_residual(x3, &inp.res_3)?
                .ok_or_else(none)
        }
    }
}

fn build_wave(c: &RealCall, arm: Arm, lanes: i64, inp: &LaneInputs) -> Result<MxArray> {
    match arm {
        Arm::Cur => build(c, Arm::Cur, lanes, inp),
        Arm::Pp => {
            mutate(WALK_OFF);
            let out = wave_planes(c, lanes, inp);
            mutate(0);
            out
        }
        Arm::Ps => wave_planes(c, lanes, inp),
    }
}

/// The wave rows: per call (one command buffer each), lanes 2, 3, 4, the
/// Tiled64 lane tier (`tier`) against the product's wave form with the
/// product walk (`Pp`) and the spread walk (`Ps`); 16 rounds, `process` 0
/// starts with the tier, 1 with the planes.
fn wave_rows(process: u32) -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    if !real_gguf_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF names no file");
        return Ok(());
    }
    let main = open_gguf_main(&[real_gguf().into()])?;
    let device = PlanesDevice::this(None);
    let plan = prepare_planes(plan_planes(&main.header, &device));
    println!("DEVICE {device:?}");
    println!("PLANDESC {}", plan.describe());
    let started = Instant::now();
    let probe = ImportProbe {
        planes: Some(plan.clone()),
        ..ImportProbe::default()
    };
    let (raw, _) = import_real(&probe)?;
    println!(
        "IMPORT tensors={} secs={:.1}",
        raw.map.len(),
        started.elapsed().as_secs_f64()
    );
    let mut calls = real_calls(&raw, &main.header, &plan)?;
    drop(raw);
    attach_real(&mut calls)?;
    println!(
        "WPLAN process={process} calls={} tensors={} bytes={}",
        calls.len(),
        plan.tensors.len(),
        plan.bytes()
    );
    let lanes_list = [2i64, 3, 4];
    let inputs: Vec<LaneInputs> = lanes_list
        .iter()
        .map(|&l| lane_inputs(l))
        .collect::<Result<_>>()?;
    for (&lanes, inp) in lanes_list.iter().zip(&inputs) {
        let (mut pp_calls, mut ps_calls) = (0usize, 0usize);
        for c in &calls {
            let want = bits(&build_wave(c, Arm::Cur, lanes, inp)?)?;
            let (pp_differ, _) = compare(&bits(&build_wave(c, Arm::Pp, lanes, inp)?)?, &want);
            let (ps_differ, _) = compare(&bits(&build_wave(c, Arm::Ps, lanes, inp)?)?, &want);
            pp_calls += usize::from(pp_differ != 0);
            ps_calls += usize::from(ps_differ != 0);
        }
        println!(
            "WEQ process={process} lanes={lanes} calls={} pp_differ_calls={pp_calls} ps_differ_calls={ps_calls}",
            calls.len()
        );
    }
    let cycle_wave = |arm: Arm, lanes: i64, inp: &LaneInputs| -> Result<BTreeMap<String, f64>> {
        let outs = calls
            .iter()
            .map(|c| build_wave(c, arm, lanes, inp))
            .collect::<Result<Vec<_>>>()?;
        let seconds = gpu_seconds(&outs)?;
        let mut by_group: BTreeMap<String, f64> = BTreeMap::new();
        let mut all = 0.0;
        for (c, s) in calls.iter().zip(&seconds) {
            *by_group.entry(c.group.clone()).or_insert(0.0) += s * 1e3;
            all += s * 1e3;
        }
        by_group.insert("ALL".to_string(), all);
        Ok(by_group)
    };
    for (&lanes, inp) in lanes_list.iter().zip(&inputs) {
        for arm in [Arm::Cur, Arm::Pp, Arm::Ps] {
            cycle_wave(arm, lanes, inp)?;
        }
    }
    let rows_started = Instant::now();
    for round in 0..16u32 {
        let arms = if (round + process).is_multiple_of(2) {
            [Arm::Cur, Arm::Pp, Arm::Ps]
        } else {
            [Arm::Ps, Arm::Pp, Arm::Cur]
        };
        for (&lanes, inp) in lanes_list.iter().zip(&inputs) {
            for arm in arms {
                for (group, ms) in &cycle_wave(arm, lanes, inp)? {
                    println!(
                        "SROW wave {process} {round} {lanes} {} {group} {ms:.6}",
                        arm.name(lanes)
                    );
                }
            }
        }
    }
    println!(
        "ROWS_DONE process={process} form=wave secs={:.1}",
        rows_started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[test]
#[ignore = "reads the real 27B GGUF; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_real_wave_rows_forward() -> Result<()> {
    wave_rows(0)
}

#[test]
#[ignore = "reads the real 27B GGUF; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_real_wave_rows_backward() -> Result<()> {
    wave_rows(1)
}
