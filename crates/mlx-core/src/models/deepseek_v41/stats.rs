//! Per-token stats line (`MLX_DSV41_STATS=<file>`, one JSON object per
//! generated token) and the process-wide wait counters it reads: router ids
//! to the host, engram row reads. Expert fill waits come from the bank.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

pub(crate) static ROUTER_WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static ENGRAM_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static ENGRAM_BYTES: AtomicU64 = AtomicU64::new(0);

/// Wait counters and read bytes so far (monotonic, process-wide).
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub router_ns: u64,
    pub engram_ns: u64,
    pub engram_bytes: u64,
    pub fill_ns: u64,
    pub fill_bytes: u64,
    /// Experts read into slots (misses; all projections of one counted once).
    pub fill_slots: u64,
    /// Time blocked on expert reads (< `fill_ns` when reads overlap GPU work).
    pub fill_wait_ns: u64,
    /// Expert uses (distinct experts per layer step) and cache hits among them.
    pub uses: u64,
    pub hits: u64,
}

impl Counters {
    pub fn now(bank: &crate::moe_stream::bank::StagingBank) -> Self {
        let b = bank.stats();
        Self {
            router_ns: ROUTER_WAIT_NS.load(Ordering::Relaxed),
            engram_ns: ENGRAM_NS.load(Ordering::Relaxed),
            engram_bytes: ENGRAM_BYTES.load(Ordering::Relaxed),
            fill_ns: b.ns,
            fill_bytes: b.bytes,
            fill_slots: b.slots as u64,
            fill_wait_ns: b.wait_ns,
            uses: b.uses as u64,
            hits: b.hits as u64,
        }
    }

    pub fn since(&self, before: &Self) -> Self {
        Self {
            router_ns: self.router_ns - before.router_ns,
            engram_ns: self.engram_ns - before.engram_ns,
            engram_bytes: self.engram_bytes - before.engram_bytes,
            fill_ns: self.fill_ns - before.fill_ns,
            fill_bytes: self.fill_bytes - before.fill_bytes,
            fill_slots: self.fill_slots - before.fill_slots,
            fill_wait_ns: self.fill_wait_ns - before.fill_wait_ns,
            uses: self.uses - before.uses,
            hits: self.hits - before.hits,
        }
    }
}

/// This process: (current footprint, lifetime peak footprint, disk bytes read).
#[cfg(target_os = "macos")]
pub fn rusage() -> (u64, u64, u64) {
    // SAFETY: rusage_info_v4 is plain integers; all-zero is valid.
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a writable rusage_info_v4 for the V4 flavor.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            (&raw mut info).cast(),
        )
    };
    if rc != 0 {
        return (0, 0, 0);
    }
    (
        info.ri_phys_footprint,
        info.ri_lifetime_max_phys_footprint,
        info.ri_diskio_bytesread,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn rusage() -> (u64, u64, u64) {
    (0, 0, 0)
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut v = [0u8; 8];
    let mut len = v.len();
    // SAFETY: `v` is writable for `len` bytes; the name is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(
            c.as_ptr(),
            v.as_mut_ptr().cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (rc, len) {
        (0, 4) => Some(u64::from(u32::from_le_bytes([v[0], v[1], v[2], v[3]]))),
        (0, 8) => Some(u64::from_le_bytes(v)),
        _ => None,
    }
}

/// (`kern.memorystatus_level`, `vm.swapusage` used bytes).
#[cfg(target_os = "macos")]
pub fn host_pressure() -> (Option<u64>, Option<u64>) {
    let swap = {
        // SAFETY: xsw_usage is plain integers; all-zero is valid.
        let mut s: libc::xsw_usage = unsafe { std::mem::zeroed() };
        let mut len = std::mem::size_of::<libc::xsw_usage>();
        // SAFETY: `s` is writable for `len` bytes; the name is NUL-terminated.
        let rc = unsafe {
            libc::sysctlbyname(
                c"vm.swapusage".as_ptr(),
                (&raw mut s).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(s.xsu_used)
    };
    (sysctl_u64("kern.memorystatus_level"), swap)
}

#[cfg(not(target_os = "macos"))]
pub fn host_pressure() -> (Option<u64>, Option<u64>) {
    (None, None)
}

/// Appends `line` as one JSON line to `$MLX_DSV41_STATS` when set.
pub fn emit(line: &Value) -> std::io::Result<()> {
    let Some(p) = std::env::var_os("MLX_DSV41_STATS") else {
        return Ok(());
    };
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)?;
    writeln!(f, "{line}")
}

/// The plan's stats line (S14a) from one generation's numbers.
#[allow(clippy::too_many_arguments)]
pub fn line(
    stage: &str,
    prompt_tok: usize,
    gen_tok: usize,
    ttft_s: f64,
    decode_s: &[f64],
    c: &Counters,
    wall_s: f64,
    bytes_read_total: u64,
    cache_slots: usize,
    peak_footprint: u64,
    swapout_delta: i64,
    memstatus_min: Option<u64>,
) -> Value {
    const GIB: f64 = (1u64 << 30) as f64;
    let mut d = decode_s.to_vec();
    d.sort_by(f64::total_cmp);
    let med = d.get(d.len() / 2).copied().unwrap_or(0.0);
    let (io_s, eng_s) = (c.fill_ns as f64 * 1e-9, c.engram_ns as f64 * 1e-9);
    let toks = gen_tok.max(1) as f64;
    json!({
        "stage": stage,
        "prompt_tok": prompt_tok,
        "gen_tok": gen_tok,
        "ttft_s": ttft_s,
        "prefill_tok_s": prompt_tok as f64 / ttft_s.max(1e-9),
        "decode_tok_s_median": if med > 0.0 { 1.0 / med } else { 0.0 },
        // cache hits per use; bytes: every use served without a read (cache
        // hits and still-valid staging copies; all experts have one size)
        "hit_rate_uses": c.hits as f64 / c.uses.max(1) as f64,
        "hit_rate_bytes": 1.0 - c.fill_slots as f64 / c.uses.max(1) as f64,
        "miss_gb_per_tok": c.fill_bytes as f64 / 1e9 / toks,
        "read_gbps": c.fill_bytes as f64 / 1e9 / io_s.max(1e-9),
        "t_c_ms": (wall_s - io_s - eng_s) * 1e3 / toks,
        "t_io_ms": io_s * 1e3 / toks,
        "t_io_wait_ms": c.fill_wait_ns as f64 * 1e-6 / toks,
        "t_engram_ms": eng_s * 1e3 / toks,
        "t_router_wait_ms": c.router_ns as f64 * 1e-6 / toks,
        "bytes_read_total": bytes_read_total,
        "cache_slots": cache_slots,
        "staging_layers": 1,
        "peak_footprint_gib": peak_footprint as f64 / GIB,
        "mlx_cache_gib": crate::array::get_cache_memory() / GIB,
        "swapout_delta_mib": swapout_delta as f64 / (1u64 << 20) as f64,
        "memstatus_min": memstatus_min,
        "cold": c.hits == 0,
    })
}
