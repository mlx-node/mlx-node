use crate::array::MxArray;
use crate::models::qwen3_5::int8_gemm;
use crate::nn::Activations;
use crate::nn::Linear;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Default minimum `M = batch * seq_len` at which the int8 W8A8 prefill GEMM is
/// worth taking. Below this (notably `M == 1` decode), per-token activation
/// quant overhead dominates and int8 regresses vs bf16, so we fall through to
/// the bf16 path. Override via `MLX_INT8_PREFILL_MIN_M`.
const INT8_PREFILL_MIN_M_DEFAULT: i64 = 256;

/// Returns `true` when `MLX_INT8_PREFILL` is set to a truthy value (non-empty,
/// not "0"/"false"). The int8 W8A8 prefill path is OFF by default — the bf16
/// fused path is the unchanged default.
fn int8_prefill_enabled() -> bool {
    match std::env::var("MLX_INT8_PREFILL") {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        }
        Err(_) => false,
    }
}

/// The `M` threshold (`batch * seq_len`) at or above which the int8 path is
/// taken. Reads `MLX_INT8_PREFILL_MIN_M`, falling back to the default.
fn int8_prefill_min_m() -> i64 {
    std::env::var("MLX_INT8_PREFILL_MIN_M")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|&v| v >= 0)
        .unwrap_or(INT8_PREFILL_MIN_M_DEFAULT)
}

/// Multi-Layer Perceptron with SwiGLU activation.
///
/// Uses the gated linear unit activation popularized by models like Llama and Qwen:
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// This is more expressive than standard FFN and is the default in modern LLMs.
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// E39: pre-stacked `[w_gate; w_up]` then transposed to `[hidden, 2*intermediate]`.
    /// Populated by `finalize_gate_up()` after weights are loaded. When present,
    /// `forward()` uses `mlx_swiglu_mlp_forward_stacked` (one matmul instead of two
    /// plus the per-call transposes baked in).
    gate_up_proj_wt: Option<MxArray>,
    /// E39: pre-transposed down_proj weight `[hidden, intermediate]`. Same idea:
    /// hoist the per-forward transpose to load time.
    down_proj_wt: Option<MxArray>,
    /// Stage 3 (NA int8 W8A8 prefill): opaque int8 quant of the fused
    /// `[gate; up]` weight `[2*intermediate, hidden]` (rows = output channels =
    /// `N`, matching `quantize_weight_int8`'s `[N, K]` contract). Populated by
    /// `finalize_gate_up()` ONLY when `MLX_INT8_PREFILL` is truthy. `None` keeps
    /// the bf16 path as the unchanged default.
    gate_up_w_i8: Option<MxArray>,
    /// Per-output-channel f32 scale `[2*intermediate]` for `gate_up_w_i8`.
    gate_up_s_w: Option<MxArray>,
    /// Opaque int8 quant of the down_proj weight `[hidden, intermediate]`
    /// (`[N=hidden, K=intermediate]`).
    down_w_i8: Option<MxArray>,
    /// Per-output-channel f32 scale `[hidden]` for `down_w_i8`.
    down_s_w: Option<MxArray>,
}

impl MLP {
    /// Creates a new MLP (SwiGLU) layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension (typically 4x or more of hidden_size)
    pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
        // All three projections have no bias (standard in modern architectures)
        let gate_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj_wt: None,
            down_proj_wt: None,
            gate_up_w_i8: None,
            gate_up_s_w: None,
            down_w_i8: None,
            down_s_w: None,
        })
    }

    /// E39: precompute the stacked `[gate; up]^T` weight and the transposed
    /// `down_proj^T` weight once after all three projection weights are loaded.
    /// Forward will then use `mlx_swiglu_mlp_forward_stacked`, which does ONE
    /// (x @ wgu_t) matmul instead of two separate (x @ w_gate.T) + (x @ w_up.T)
    /// matmuls, and reads pre-transposed weights so the per-call `transpose`
    /// graph nodes vanish.
    ///
    /// Safe to call repeatedly (idempotent — overwrites). Callers from the
    /// persistence layer should invoke it once after the gate/up/down weights
    /// for a given layer have all been set.
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        let w_gate = self.gate_proj.get_weight();
        let w_up = self.up_proj.get_weight();
        let w_down = self.down_proj.get_weight();
        // gate, up: [intermediate, hidden] → stacked: [2*intermediate, hidden] →
        // transpose to [hidden, 2*intermediate] for the matmul x @ wgu_t.
        let stacked = MxArray::concatenate(&w_gate, &w_up, 0)?;
        let wgu_t = stacked.transpose(Some(&[1, 0]))?;
        wgu_t.eval();
        // down: [hidden, intermediate] → [intermediate, hidden]
        let wd_t = w_down.transpose(Some(&[1, 0]))?;
        wd_t.eval();
        self.gate_up_proj_wt = Some(wgu_t);
        self.down_proj_wt = Some(wd_t);

        // Stage 3 (NA int8 W8A8 prefill, opt-in via MLX_INT8_PREFILL): quantize
        // the fused gate_up and down weights to int8 ONCE at load.
        //
        // LAYOUT: `quantize_weight_int8` expects `[N, K]` with rows = output
        // channels so its `s_w[N]` scale broadcasts onto the GEMM accumulator
        // `acc[M, N]`. The UN-transposed `stacked` is `[2*intermediate, hidden]`
        // = `[N=2*intermediate, K=hidden]`, and `w_down` is
        // `[hidden, intermediate]` = `[N=hidden, K=intermediate]` — both already
        // in `[N, K]`, so we pass them straight in (the `*_wt` transposed forms
        // are for the bf16 matmul ONLY and must NOT be quantized here).
        //
        // Stage 4b: `quantize_weight_int8` now ALSO hoists the per-forward
        // transpose — it returns the opaque int8 weight already in the `[K, N]`
        // kernel layout, so `try_forward_int8` does zero weight reshaping. The
        // stored orientation is opaque to Rust; we still pass the `[N, K]` input.
        //
        // SCOPE: int8 prefill is DENSE qwen3_5 ONLY. qwen3_5_moe routes its MLP
        // through the per-expert SwitchMLP / gather_qmm persistence path and never
        // calls this `finalize_gate_up()`, so `MLX_INT8_PREFILL` is a silent bf16
        // no-op on MoE (intended — not wired for MoE; documented, not fixed).
        //
        // MEMORY: with the flag ON, BOTH the bf16 stacked/transposed weight
        // (`gate_up_proj_wt` / `down_proj_wt`, above) AND the int8 weight stay
        // resident — the bf16 form is the per-call `Err`-fallback target in
        // `try_forward_int8`. Opt-in trades extra weight memory for prefill speed.
        //
        // VALIDATED REGIME: opt-in, greedy (T=0), bf16 models, prefill M>=256.
        // Sampling / MTP / long-context (>8k) are NOT yet validation-gated, which
        // is why this path is deliberately default-OFF.
        self.gate_up_w_i8 = None;
        self.gate_up_s_w = None;
        self.down_w_i8 = None;
        self.down_s_w = None;
        if int8_prefill_enabled() {
            // Fail-soft: if quant fails (e.g. unsupported shape), leave the
            // fields None so forward() stays on the unchanged bf16 path.
            if let Ok((gu_i8, gu_s)) = int8_gemm::quantize_weight_int8(&stacked) {
                gu_i8.eval();
                gu_s.eval();
                self.gate_up_w_i8 = Some(gu_i8);
                self.gate_up_s_w = Some(gu_s);
            }
            if let Ok((d_i8, d_s)) = int8_gemm::quantize_weight_int8(&w_down) {
                d_i8.eval();
                d_s.eval();
                self.down_w_i8 = Some(d_i8);
                self.down_s_w = Some(d_s);
            }
        }
        Ok(())
    }

    /// Stage 3 NA int8 W8A8 prefill MLP path.
    ///
    /// Returns:
    ///   * `Ok(Some(out))` — the int8 path ran and produced the MLP output.
    ///   * `Ok(None)`      — not eligible (flag off / no int8 weights / `M`
    ///     below threshold / a fail-soft int8-op `Err`); caller must use bf16.
    ///   * `Err`           — only a genuine non-int8 error (reshape/shape).
    ///
    /// Pipeline (mirrors `mlx_swiglu_mlp_forward_stacked`):
    ///   `x[B,T,hidden]` → `[M, hidden]`
    ///   → `gate_up = int8_w8a8(x, gate_up_w_i8, gate_up_s_w)` `[M, 2*inter]`
    ///   → split → `swiglu = silu(gate) * up` (bf16, no f32 promotion)
    ///   → `out = int8_w8a8(swiglu, down_w_i8, down_s_w)` `[M, hidden]`
    ///   → reshape back to the original leading dims.
    ///
    /// The int8 op narrows its result to bf16 internally, so the residual add in
    /// the caller is not promoted to f32.
    fn try_forward_int8(&self, x: &MxArray) -> Result<Option<MxArray>> {
        // Gate 1: flag + quantized weights present.
        let (Some(gu_i8), Some(gu_s), Some(d_i8), Some(d_s)) = (
            &self.gate_up_w_i8,
            &self.gate_up_s_w,
            &self.down_w_i8,
            &self.down_s_w,
        ) else {
            return Ok(None);
        };
        if !int8_prefill_enabled() {
            return Ok(None);
        }

        // Gate 2: M = product of leading dims (everything but the last). For
        // `[B, T, hidden]`, M = B*T; for already-2D `[M, hidden]`, M = M.
        let shape = x.shape()?;
        let dims: &[i64] = &shape;
        if dims.len() < 2 {
            return Ok(None);
        }
        let hidden = dims[dims.len() - 1];
        let m: i64 = dims[..dims.len() - 1].iter().product();
        // M == 1 (decode) and small prefill regress vs bf16 → fall through.
        if m < int8_prefill_min_m() {
            return Ok(None);
        }

        // Reshape to 2D [M, hidden] for the int8 GEMM.
        let x2d = x.reshape(&[m, hidden])?;

        // gate_up: int8 W8A8. On Err (e.g. gen<17 / K%16!=0) fall back to bf16.
        let gate_up = match int8_gemm::int8_w8a8_matmul(&x2d, gu_i8, gu_s) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        // gate_up: [M, 2*intermediate] → split halves.
        let two_inter = gate_up.shape_at(1)?;
        let intermediate = two_inter / 2;
        let gate = gate_up.slice(&[0, 0], &[m, intermediate])?;
        let up = gate_up.slice(&[0, intermediate], &[m, two_inter])?;

        // swiglu = silu(gate) * up. `Activations::silu` preserves bf16 dtype, so
        // `gated` stays bf16 (no f32 promotion).
        let gate_act = Activations::silu(&gate)?;
        let gated = gate_act.mul(&up)?;

        // down: int8 W8A8 → [M, hidden]. On Err fall back to bf16.
        let out2d = match int8_gemm::int8_w8a8_matmul(&gated, d_i8, d_s) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        // Optional debug breadcrumb so a smoke test can confirm the int8 path
        // actually fired (gated so it never pollutes the default path).
        if std::env::var("MLX_INT8_PREFILL_DEBUG").is_ok() {
            eprintln!("[int8-prefill] fired: M={m} hidden={hidden} two_inter={two_inter}");
        }

        // Reshape [M, hidden] back to the original leading dims (mirror bf16).
        let mut out_shape: Vec<i64> = dims[..dims.len() - 1].to_vec();
        out_shape.push(hidden);
        let out = out2d.reshape(&out_shape)?;
        Ok(Some(out))
    }

    /// Forward pass: down(silu(gate(x)) * up(x))
    ///
    /// Uses fused C++ implementation for maximum performance (1 FFI call vs 8).
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        // Stage 3 (NA int8 W8A8 prefill, opt-in): route the fused gate_up and
        // down matmuls through the int8 GEMM when enabled, eligible, and the
        // weights are quantized. Any failure inside falls through to the bf16
        // path below (returns Ok(None)). OFF by default ⇒ zero change.
        if let Some(out) = self.try_forward_int8(x)? {
            return Ok(out);
        }

        // E39: fast path — pre-stacked + pre-transposed weights.
        // Env-toggle MLX_DISABLE_E39_STACKED_MLP=1 reverts to the legacy
        // two-matmul path for A/B testing.
        if let (Some(wgu_t), Some(wd_t)) = (&self.gate_up_proj_wt, &self.down_proj_wt)
            && std::env::var("MLX_DISABLE_E39_STACKED_MLP").is_err()
        {
            let handle = unsafe {
                sys::mlx_swiglu_mlp_forward_stacked(x.handle.0, wgu_t.handle.0, wd_t.handle.0)
            };
            return MxArray::from_handle(handle, "swiglu_mlp_forward_stacked");
        }

        // Legacy path: two matmuls with per-call transposes.
        let w_gate = self.gate_proj.get_weight();
        let w_up = self.up_proj.get_weight();
        let w_down = self.down_proj.get_weight();

        let handle = unsafe {
            sys::mlx_swiglu_mlp_forward(x.handle.0, w_gate.handle.0, w_up.handle.0, w_down.handle.0)
        };
        MxArray::from_handle(handle, "swiglu_mlp_forward")
    }

    /// Forward pass with cached intermediates for backward pass
    ///
    /// Returns: [output, gate, up, gate_act, gated]
    /// - output: final output
    /// - gate: gate_proj(x)
    /// - up: up_proj(x)
    /// - gate_act: silu(gate)
    /// - gated: gate_act * up
    #[cfg(test)]
    pub fn forward_with_cache(&self, x: &MxArray) -> Result<Vec<MxArray>> {
        // Compute gate and up projections
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // Apply SiLU activation to gate
        let gate_act = Activations::silu(&gate)?;

        // Element-wise multiplication
        let gated = gate_act.mul(&up)?;

        // Down projection
        let output = self.down_proj.forward(&gated)?;

        Ok(vec![output, gate, up, gate_act, gated])
    }

    // Weight setters for loading pretrained models

    pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.gate_proj.set_weight(weight)?;
        // Invalidate the E39 stacked cache — caller must call finalize_gate_up().
        self.gate_up_proj_wt = None;
        // Stage 3: also invalidate the int8 quant of the fused gate_up weight.
        self.gate_up_w_i8 = None;
        self.gate_up_s_w = None;
        Ok(())
    }

    pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.up_proj.set_weight(weight)?;
        self.gate_up_proj_wt = None;
        self.gate_up_w_i8 = None;
        self.gate_up_s_w = None;
        Ok(())
    }

    pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.down_proj.set_weight(weight)?;
        self.down_proj_wt = None;
        self.down_w_i8 = None;
        self.down_s_w = None;
        Ok(())
    }

    // Mutable projection accessors.
    //
    // Expose the underlying `Linear`s so a persistence layer can drive
    // affine-quantized loads (`Linear::load_quantized`) or plain bf16 loads
    // uniformly without this shared module needing to know about each model's
    // quantization scheme. Each accessor invalidates the E39 stacked-MLP cache
    // (`gate_up_proj_wt` / `down_proj_wt`), because a caller obtaining a `&mut
    // Linear` may replace the weight; a stale stacked cache would otherwise be
    // served by `forward()`. The caller must re-run `finalize_gate_up()` if it
    // wants the stacked fast path after mutating a projection. This mirrors the
    // invalidation already done by `set_{gate,up,down}_proj_weight`.

    pub fn gate_proj_mut(&mut self) -> &mut Linear {
        self.gate_up_proj_wt = None;
        self.gate_up_w_i8 = None;
        self.gate_up_s_w = None;
        &mut self.gate_proj
    }

    pub fn up_proj_mut(&mut self) -> &mut Linear {
        self.gate_up_proj_wt = None;
        self.gate_up_w_i8 = None;
        self.gate_up_s_w = None;
        &mut self.up_proj
    }

    pub fn down_proj_mut(&mut self) -> &mut Linear {
        self.down_proj_wt = None;
        self.down_w_i8 = None;
        self.down_s_w = None;
        &mut self.down_proj
    }

    // Weight getters for backward pass

    pub fn get_gate_proj_weight(&self) -> MxArray {
        self.gate_proj.get_weight()
    }

    pub fn get_up_proj_weight(&self) -> MxArray {
        self.up_proj.get_weight()
    }

    pub fn get_down_proj_weight(&self) -> MxArray {
        self.down_proj.get_weight()
    }
}

impl Clone for MLP {
    fn clone(&self) -> Self {
        Self {
            gate_proj: self.gate_proj.clone(),
            up_proj: self.up_proj.clone(),
            down_proj: self.down_proj.clone(),
            gate_up_proj_wt: self.gate_up_proj_wt.clone(),
            down_proj_wt: self.down_proj_wt.clone(),
            gate_up_w_i8: self.gate_up_w_i8.clone(),
            gate_up_s_w: self.gate_up_s_w.clone(),
            down_w_i8: self.down_w_i8.clone(),
            down_s_w: self.down_s_w.clone(),
        }
    }
}

// =================== NA int8 W8A8 prefill: forward-path wiring ===================
// Review N1: ONE integration test that exercises the WIRED forward control flow
// (`finalize_gate_up` + `forward` -> `try_forward_int8`) end-to-end WITHOUT a model
// file, using small synthetic bf16 weights. Lives in `mlp.rs` (not the sibling
// `mlp_test.rs` / `int8_gemm.rs`) so it can assert the PRIVATE int8 fields are
// invalidated after a weight setter — the true (re-quantized / None) check.
//
// It mutates `MLX_INT8_PREFILL`, so it serializes on a private lock and restores
// the var via an RAII guard. Run serially is also safe:
//   cargo test -p mlx-core --lib mlp::int8_forward_wiring -- --test-threads=1
#[cfg(test)]
mod int8_forward_wiring {
    use super::*;
    use crate::array::DType;
    use std::sync::Mutex;

    // Serializes any test in this module that toggles MLX_INT8_PREFILL so a
    // concurrent test never observes another's setting.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard: restores `MLX_INT8_PREFILL` on drop (even on panic).
    struct EnvGuard {
        prev: Option<String>,
    }
    impl EnvGuard {
        fn set(value: &str) -> Self {
            let prev = std::env::var("MLX_INT8_PREFILL").ok();
            // SAFETY: holders serialize on ENV_LOCK; no concurrent env access.
            unsafe {
                std::env::set_var("MLX_INT8_PREFILL", value);
            }
            Self { prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: see EnvGuard::set.
            unsafe {
                match self.prev.take() {
                    Some(v) => std::env::set_var("MLX_INT8_PREFILL", v),
                    None => std::env::remove_var("MLX_INT8_PREFILL"),
                }
            }
        }
    }

    fn gpu_gen() -> i32 {
        unsafe { sys::mlx_gpu_architecture_gen() }
    }

    /// Deterministic LCG int in [lo,hi] (avoids the reserved `gen` ident).
    fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let span = (hi - lo + 1) as u64;
        lo + ((*state >> 33) % span) as i32
    }

    /// Deterministic bf16 array of the given shape with small magnitudes
    /// (~[-0.2,0.2]) so the per-token quant absmax/round/clip paths are hit.
    fn rand_bf16_shape(state: &mut u64, shape: &[i64]) -> MxArray {
        let n: i64 = shape.iter().product();
        let mut f = vec![0f32; n as usize];
        for v in f.iter_mut() {
            *v = next_int(state, -200, 200) as f32 / 1000.0;
        }
        MxArray::from_float32(&f, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    /// 2D `[rows, cols]` convenience (weights).
    fn rand_bf16(state: &mut u64, rows: i64, cols: i64) -> MxArray {
        rand_bf16_shape(state, &[rows, cols])
    }

    /// Build an MLP with deterministic bf16 weights. Uses the SAME shape class
    /// as `int8_gemm::tests::s2_w8a8_cosine_parity` (hidden=2560, intermediate=
    /// 2560 — K=2560 %16==0 for BOTH the gate_up (K=hidden) and down (K=inter)
    /// GEMMs, and N >= the 128x64 MPP tile) so the NA int8 kernels get realistic,
    /// well-supported extents (tiny hidden like 64 trips an MPP matmul2d edge and
    /// aborts via a foreign C++ exception). The forward cost at M=256 is trivial.
    const HIDDEN: u32 = 2560;
    const INTER: u32 = 2560;
    fn build_mlp(state: &mut u64) -> MLP {
        let hidden = HIDDEN;
        let inter = INTER;
        let mut mlp = MLP::new(hidden, inter).unwrap();
        // gate/up: [inter,hidden]; down: [hidden,inter].
        let w_gate = rand_bf16(state, inter as i64, hidden as i64);
        let w_up = rand_bf16(state, inter as i64, hidden as i64);
        let w_down = rand_bf16(state, hidden as i64, inter as i64);
        mlp.set_gate_proj_weight(&w_gate).unwrap();
        mlp.set_up_proj_weight(&w_up).unwrap();
        mlp.set_down_proj_weight(&w_down).unwrap();
        mlp
    }

    /// Per-row min cosine of two [M,N] f32 buffers.
    fn min_row_cosine(a: &[f32], b: &[f32], m: usize, n: usize) -> f64 {
        let mut min_cos = f64::INFINITY;
        for mi in 0..m {
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for ni in 0..n {
                let x = a[mi * n + ni] as f64;
                let y = b[mi * n + ni] as f64;
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
            min_cos = min_cos.min(dot / denom);
        }
        min_cos
    }

    // This test validates the forward CONTROL FLOW (gating / threshold fall-
    // through / setter invalidation), NOT the W8A8 numeric quality:
    //   (a) M>=256: the int8 path FIRES — produces a finite, correctly-shaped
    //       bf16 output that DIFFERS from the bf16 forward (proving the int8
    //       branch actually ran rather than silently falling through). The
    //       per-row cosine vs bf16 is COMPUTED and printed as a diagnostic; the
    //       numeric-accuracy gate (cosine >= 0.999) lives in the dedicated
    //       `int8_gemm::tests::{s2,qkvz}_w8a8_cosine_parity` tests, so it is not
    //       re-asserted here (a wiring test must not double as the numeric gate).
    //   (b) M<256 (M=1): the int8 path FALLS THROUGH — byte-identical to bf16.
    //   (c) a weight setter INVALIDATES the int8 fields (set to None).
    #[test]
    fn int8_forward_wiring() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if gpu_gen() < 17 {
            eprintln!(
                "[int8-wire] SKIP: gpu gen {} < 17 (NA needs M5+)",
                gpu_gen()
            );
            return;
        }
        let hidden: i64 = HIDDEN as i64;
        const SEED: u64 = 0x1278_a1be_0000_0001; // fixed seed for reproducibility
        let mut state: u64 = SEED;

        // --- Reference bf16 outputs: build + finalize with the flag OFF so the
        // int8 fields stay None and `forward()` uses the unchanged bf16 path. ---
        {
            // Ensure OFF for the reference build/finalize.
            let _g = EnvGuard::set("0");
            let mut ref_mlp = build_mlp(&mut state);
            ref_mlp.finalize_gate_up().unwrap();
            assert!(
                ref_mlp.gate_up_w_i8.is_none() && ref_mlp.down_w_i8.is_none(),
                "flag OFF must leave int8 weights None"
            );
            // 3D inputs [B=1, T, hidden]: MLP::forward's stacked bf16 path slices
            // a 3D gate_up, so the input must be 3D. M = B*T drives the int8 gate.
            // x256: T=256 (M=256, at/above the 256 threshold → int8 fires);
            // x1:   T=1   (M=1, below threshold → bf16 fall-through).
            let x256 = rand_bf16_shape(&mut state, &[1, 256, hidden]);
            let x1 = rand_bf16_shape(&mut state, &[1, 1, hidden]);
            let y_ref_256 = ref_mlp.forward(&x256).unwrap();
            let y_ref_1 = ref_mlp.forward(&x1).unwrap();
            y_ref_256.eval();
            y_ref_1.eval();
            let ref256 = y_ref_256
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let ref1 = y_ref_1
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();

            // --- int8 build/finalize with the flag ON. SAME weights + inputs
            // (reseed the weight LCG to the build seed used above). ---
            let _g_on = EnvGuard::set("1");
            // Rebuild from a fresh, identical seed so weights match the reference.
            let mut state2: u64 = SEED;
            let mut mlp = build_mlp(&mut state2);
            mlp.finalize_gate_up().unwrap();
            // Flag ON + supported shape -> int8 weights MUST be populated.
            assert!(
                mlp.gate_up_w_i8.is_some()
                    && mlp.gate_up_s_w.is_some()
                    && mlp.down_w_i8.is_some()
                    && mlp.down_s_w.is_some(),
                "flag ON must populate int8 weights at finalize"
            );

            // (a) M=256: the int8 path FIRES. Assert WIRING invariants only:
            // correct shape + dtype, all-finite output, and that it DIFFERS from
            // the bf16 forward (so we know the int8 branch ran, not a silent bf16
            // fall-through). Cosine vs bf16 is printed as a diagnostic; the
            // numeric-quality gate is s2/qkvz_w8a8_cosine_parity (NOT here).
            let y_256 = mlp.forward(&x256).unwrap();
            y_256.eval();
            assert_eq!(
                y_256.dtype().unwrap(),
                DType::BFloat16,
                "int8 out must be bf16"
            );
            assert_eq!(y_256.ndim().unwrap(), 3, "int8 out must be [B,T,hidden]");
            assert_eq!(y_256.shape_at(0).unwrap(), 1, "int8 out B must be 1");
            assert_eq!(y_256.shape_at(1).unwrap(), 256, "int8 out T must be M=256");
            assert_eq!(
                y_256.shape_at(2).unwrap(),
                hidden,
                "int8 out hidden dim must match"
            );
            let got256 = y_256.astype(DType::Float32).unwrap().to_float32().unwrap();
            assert_eq!(got256.len(), ref256.len());
            assert!(
                got256.iter().all(|v| v.is_finite()),
                "int8 path produced non-finite output"
            );
            let differs = got256
                .iter()
                .zip(ref256.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                differs,
                "int8 path output is byte-identical to bf16 — int8 branch did not fire at M=256"
            );
            let min_cos = min_row_cosine(&got256, &ref256, 256, hidden as usize);
            eprintln!(
                "[int8-wire] (a) M=256 int8 path FIRED (shape/dtype/finite ok, \
                 differs from bf16); diagnostic min_row_cos = {min_cos:.6} \
                 (numeric gate: s2/qkvz_w8a8_cosine_parity)"
            );

            // (b) M=1: int8 path falls through to bf16 -> byte-identical.
            let y_1 = mlp.forward(&x1).unwrap();
            y_1.eval();
            let got1 = y_1.astype(DType::Float32).unwrap().to_float32().unwrap();
            assert_eq!(got1.len(), ref1.len());
            let mut bad1 = 0usize;
            for i in 0..got1.len() {
                if got1[i].to_bits() != ref1[i].to_bits() {
                    bad1 += 1;
                }
            }
            eprintln!("[int8-wire] (b) M=1 byte-diffs vs bf16 = {bad1}");
            assert_eq!(bad1, 0, "M<256 must be byte-identical to bf16 forward");

            // (c) a weight setter invalidates the int8 fields. w_new is the
            // gate_proj shape [out=inter, in=hidden]; down takes its transpose
            // [out=hidden, in=inter] (both square here).
            let w_new = rand_bf16(&mut state2, INTER as i64, hidden);
            mlp.set_gate_proj_weight(&w_new).unwrap();
            assert!(
                mlp.gate_up_w_i8.is_none() && mlp.gate_up_s_w.is_none(),
                "set_gate_proj_weight must invalidate gate_up int8 fields"
            );
            mlp.set_down_proj_weight(&w_new.transpose(Some(&[1, 0])).unwrap())
                .unwrap();
            assert!(
                mlp.down_w_i8.is_none() && mlp.down_s_w.is_none(),
                "set_down_proj_weight must invalidate down int8 fields"
            );
            eprintln!("[int8-wire] (c) weight setters invalidated int8 fields");
        }
    }
}
