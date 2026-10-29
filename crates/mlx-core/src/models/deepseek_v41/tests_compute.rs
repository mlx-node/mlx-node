// Compute memory vs prompt length. A staging-only (expert cache 0)
// prefill of a real prompt of T ids (`$DSV41_PROMPTS/compute-<T>.json`) with
// last-row logits (`forward_last`, the product prefill), then one greedy
// step. Records the peak footprint (sampled every 2 ms), MLX peak active
// bytes, KV bytes, the HC carry and the attention scratch to
// `$DSV41_COMPUTE_OUT/compute-<T>.json`; `compute(T) = peak - after load -
// KV(T)` is fitted out of repo (`tools/s14c/fit.py`). Run with
// `MLX_DSV41_EXPERT_CACHE_GB=0`, `MLX_DSV41_STATS`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde_json::json;

use super::stats::Counters;
use crate::array::{get_active_memory, get_peak_memory};

/// Max of this process's footprint, polled on a thread until dropped.
struct FootprintMax {
    stop: Arc<AtomicBool>,
    max: Arc<AtomicU64>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl FootprintMax {
    fn start() -> Self {
        let (stop, max) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
        );
        let (s, m) = (stop.clone(), max.clone());
        let join = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                m.fetch_max(super::stats::rusage().0, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        Self {
            stop,
            max,
            join: Some(join),
        }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            j.join().unwrap();
        }
        self.max
            .load(Ordering::Relaxed)
            .max(super::stats::rusage().0)
    }
}

impl Drop for FootprintMax {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Bytes of every per-session cache array (window rows, compressed KV,
/// index keys, compressor tails).
fn session_kv_bytes(sess: &super::model::Session) -> (u64, u64, u64) {
    let nb = |a: &Option<MxArray>| a.as_ref().map_or(0, |a| a.nbytes() as u64);
    let win: u64 = sess
        .wins
        .iter()
        .map(|w| w.rows().map_or(0, |r| r.nbytes() as u64))
        .sum();
    let (mut ckv, mut ik) = (0, 0);
    for c in sess.sources.iter().flatten() {
        ckv += nb(&c.kv) + nb(&c.tail.kv) + nb(&c.tail.score);
        ik += nb(&c.index_k);
    }
    (win, ckv, ik)
}

fn real_compute(t: usize) {
    const GIB: f64 = (1u64 << 30) as f64;
    let dir = PathBuf::from(std::env::var("DSV41_PROMPTS").expect("DSV41_PROMPTS"));
    let pj: Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("compute-{t}.json"))).unwrap())
            .unwrap();
    let ids: Vec<u32> = pj["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids.len(), t);
    let mut store = e2e_store();
    // the greedy step after a T-token prefill is measured one token past the
    // context limit at T = 16K, which the product refuses
    store.model.ctx_limit = store.model.ctx_limit.map(|l| l.max(t + 1));
    let m = &store.model;
    let (hc, dim, n_layers) = (m.cfg.hc_mult, m.cfg.dim, m.cfg.n_layers);
    crate::array::synchronize();
    let fp_load = super::stats::rusage().0;
    let (act_load, pool_load) = (get_active_memory(), get_cache_memory());
    let (_, swap0) = super::stats::host_pressure();
    let c0 = Counters::now(&store.bank);
    let mut fails = Vec::new();

    let mut sess = m.new_session();
    crate::array::reset_peak_memory();
    let sampler = FootprintMax::start();
    let t0 = std::time::Instant::now();
    let logits = m.forward_last(&mut sess, &ids, None, None).unwrap();
    let last = logits.to_float32().unwrap().to_vec();
    let ttft = t0.elapsed().as_secs_f64();
    let fp_peak = sampler.finish();
    let mlx_peak = get_peak_memory();
    let pool_prefill = get_cache_memory();
    let c_pre = Counters::now(&store.bank).since(&c0);
    let (win, ckv, ik) = session_kv_bytes(&sess);
    let kv = win + ckv + ik;
    let act_after = get_active_memory();
    let pick = super::generate::argmax(&last);
    if !last.iter().all(|v| v.is_finite()) {
        fails.push("prefill logits not finite".into());
    }
    if sess.tokens.len() != t {
        fails.push(format!("session holds {} tokens", sess.tokens.len()));
    }
    for (l, c) in sess.sources.iter().enumerate() {
        if let Some(c) = c {
            let rows = c.kv.as_ref().map_or(0, |a| a.shape_at(0).unwrap() as usize);
            if rows != t / c.ratio {
                fails.push(format!("L{l} compressed rows {rows} != {t} / {}", c.ratio));
            }
        }
    }

    // one greedy step at context T
    crate::array::reset_peak_memory();
    let sampler = FootprintMax::start();
    let t1 = std::time::Instant::now();
    let step = m.forward(&mut sess, &[pick], None, None).unwrap();
    let step_row = step.to_float32().unwrap().to_vec();
    let step_s = t1.elapsed().as_secs_f64();
    let fp_step = sampler.finish();
    let mlx_step = get_peak_memory();
    if !step_row.iter().all(|v| v.is_finite()) {
        fails.push("step logits not finite".into());
    }
    let c = Counters::now(&store.bank).since(&c0);
    let pool_end = get_cache_memory();
    for (what, v) in [("prefill", pool_prefill), ("step", pool_end)] {
        if v > 2.0 * GIB {
            fails.push(format!(
                "MLX cache {:.2} GiB after the {what}",
                v / GIB
            ));
        }
    }
    let ratio = c.fill_bytes as f64 / (c.fill_slots * EXPERT_BYTES).max(1) as f64;
    if !(1.0..=1.005).contains(&ratio) {
        fails.push(format!("expert bytes ratio {ratio}"));
    }
    if c_pre.fill_slots > (n_layers * m.cfg.n_routed_experts) as u64 {
        fails.push(format!(
            "prefill read {} experts > one per (layer, expert)",
            c_pre.fill_slots
        ));
    }

    let carry = (t * hc * dim * 4 + t * hc * 4) as u64;
    let scratch = m.scratch_peak() as u64;
    let compute_mlx = mlx_peak - act_load - kv as f64;
    let compute_fp = fp_peak as f64 - fp_load as f64 - kv as f64;
    let (lvl, swap) = super::stats::host_pressure();
    let swap_d = swap.zip(swap0).map_or(0, |(a, b)| a as i64 - b as i64);
    let line = super::stats::line(
        "s14c",
        t,
        2,
        ttft,
        &[step_s],
        &c,
        t0.elapsed().as_secs_f64(),
        store.dense_read + c.fill_bytes + c.engram_bytes,
        0,
        super::stats::rusage().1,
        swap_d,
        lvl,
    );
    super::stats::emit(&line).unwrap();
    let doc = json!({
        "T": t,
        "dtype": "f32",
        "ttft_s": ttft,
        "prefill_tok_s": t as f64 / ttft,
        "step_s": step_s,
        "prefill_argmax": pick,
        "step_argmax": super::generate::argmax(&step_row),
        "footprint_after_load": fp_load,
        "footprint_peak_prefill": fp_peak,
        "footprint_peak_step": fp_step,
        "footprint_lifetime_peak": super::stats::rusage().1,
        "mlx_active_after_load": act_load,
        "mlx_pool_after_load": pool_load,
        "mlx_peak_prefill": mlx_peak,
        "mlx_active_after_prefill": act_after,
        "mlx_pool_after_prefill": pool_prefill,
        "mlx_peak_step": mlx_step,
        "kv_window": win,
        "kv_compressed": ckv,
        "kv_index_k": ik,
        "kv_bytes": kv,
        "hc_carry_bytes": carry,
        "attn_scratch_peak": scratch,
        "compute_mlx": compute_mlx,
        "compute_footprint": compute_fp,
        "dense_bytes": store.dense_bytes,
        "bank_bytes": store.bank.nbytes(),
        "prefill_fill_slots": c_pre.fill_slots,
        "prefill_fill_bytes": c_pre.fill_bytes,
        "prefill_fill_s": c_pre.fill_ns as f64 * 1e-9,
        "prefill_engram_bytes": c_pre.engram_bytes,
        "prefill_engram_s": c_pre.engram_ns as f64 * 1e-9,
        "prefill_router_wait_s": c_pre.router_ns as f64 * 1e-9,
        "bytes_read_total": store.dense_read + c.fill_bytes + c.engram_bytes,
        "swap_delta": swap_d,
        "stats": line,
        "fails": fails,
    });
    println!(
        "compute T={t}: ttft {ttft:.1} s ({:.1} tok/s prefill), step {step_s:.2} s; footprint load {:.2} peak {:.2} GiB, MLX active load {:.2} peak {:.2} GiB; KV {:.3} GiB (window {win}, ckv {ckv}, index-K {ik}); HC carry {:.3} GiB; attn scratch {:.3} GiB; compute MLX {:.3} GiB, footprint {:.3} GiB; experts {} read {:.1} GB in {:.1} s, engram {:.2} GB",
        t as f64 / ttft,
        fp_load as f64 / GIB,
        fp_peak as f64 / GIB,
        act_load / GIB,
        mlx_peak / GIB,
        kv as f64 / GIB,
        carry as f64 / GIB,
        scratch as f64 / GIB,
        compute_mlx / GIB,
        compute_fp / GIB,
        c_pre.fill_slots,
        c_pre.fill_bytes as f64 / 1e9,
        c_pre.fill_ns as f64 * 1e-9,
        c_pre.engram_bytes as f64 / 1e9,
    );
    println!("compute json: {doc}");
    if let Some(out) = std::env::var_os("DSV41_COMPUTE_OUT") {
        let p = PathBuf::from(out).join(format!("compute-{t}.json"));
        std::fs::write(&p, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
    }
    finish(&format!("real_compute_{t}"), &fails);
}

#[test]
#[ignore = "real 9 shards + $DSV41_PROMPTS; MLX_DSV41_EXPERT_CACHE_GB=0; ~20 GiB"]
fn real_compute_1k() {
    real_compute(1024);
}

#[test]
#[ignore = "real 9 shards + $DSV41_PROMPTS; MLX_DSV41_EXPERT_CACHE_GB=0; ~25 GiB"]
fn real_compute_8k() {
    real_compute(8192);
}

#[test]
#[ignore = "real 9 shards + $DSV41_PROMPTS; MLX_DSV41_EXPERT_CACHE_GB=0; ~35 GiB"]
fn real_compute_16k() {
    real_compute(16384);
}
