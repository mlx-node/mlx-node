use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use super::*;
use crate::array::{DType, MxArray};
use crate::utils::gguf::{GgufFile, GgufMetaValue, GgufTensorType, parse_gguf, parse_gguf_reader};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::{
    GpuRepackKind, OutLayout, RepackDst, gpu_repack_available, gpu_repack_band,
};
use crate::utils::gguf_test_fixtures::build_minimal_gguf;

const N: usize = 7;
const OUT: usize = 256;
const K: usize = 256;
/// File offset of the fixture tensor, as the real file's first expert tensor.
const ABS: u64 = 1600;
const PER: usize = OUT * K / 32 * 17;
const SENTINEL: u8 = 0xAB;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("moe-stream-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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

struct Fixture {
    _dir: Temp,
    path: PathBuf,
    header: GgufFile,
    payload: Vec<u8>,
}

/// One MXFP4 `[N, OUT, K]` expert tensor at file offset 1600.
fn fixture() -> Fixture {
    let dir = Temp::new();
    let path = dir.0.join("experts.gguf");
    let payload = random_bytes(N * PER, 7);
    let mut pad = String::new();
    loop {
        let bytes = build_minimal_gguf(
            &[("general.pad", GgufMetaValue::String(pad.clone()))],
            &[(
                "blk.0.ffn_up_exps.weight",
                &[K as u64, OUT as u64, N as u64],
                GgufTensorType::MXFP4,
                &payload,
            )],
        );
        std::fs::write(&path, bytes).unwrap();
        let header = parse_gguf(&path).unwrap();
        let at = header.data_offset + header.tensors[0].offset;
        if at == ABS {
            return Fixture {
                _dir: dir,
                path,
                header,
                payload,
            };
        }
        assert!(at < ABS, "fixture header already past {ABS}");
        pad.push_str(&"x".repeat((ABS - at) as usize));
    }
}

struct Bank {
    dst: RepackDst,
    repack: SlotRepack,
}

fn bank(f: &Fixture) -> Bank {
    let (dst, ptrs) = RepackDst::alloc(&f.header.tensors[0], OutLayout::RowMajor).unwrap();
    // SAFETY: fresh CPU-visible buffers of these sizes, no GPU work yet.
    unsafe {
        std::ptr::write_bytes(ptrs.weight.0, SENTINEL, N * OUT * K / 2);
        std::ptr::write_bytes(ptrs.scales.unwrap().0, SENTINEL, N * OUT * K / 32);
    }
    Bank {
        repack: SlotRepack::from_dst(ptrs, OUT).unwrap(),
        dst,
    }
}

/// `(weight bytes, scale bytes)` of each expert / slot.
fn slots(dst: &RepackDst) -> Vec<(Vec<u8>, Vec<u8>)> {
    let s = dst.scales.as_ref().unwrap();
    MxArray::eval_arrays_with_context(&[&dst.weight, s], "moe_stream test").unwrap();
    assert_eq!(s.dtype().unwrap(), DType::Uint8);
    let w: Vec<u8> = dst
        .weight
        .to_uint32()
        .unwrap()
        .to_vec()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let s = s.to_uint8().unwrap();
    let (wn, sn) = (w.len() / N, s.len() / N);
    (0..N)
        .map(|i| {
            (
                w[i * wn..(i + 1) * wn].to_vec(),
                s[i * sn..(i + 1) * sn].to_vec(),
            )
        })
        .collect()
}

/// The whole-file loader's CPU repack and the GPU `Mxfp4` repack of the
/// whole tensor, per expert; both must agree.
fn references(f: &Fixture) -> Vec<(Vec<u8>, Vec<u8>)> {
    let info = &f.header.tensors[0];
    let (cpu, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor).unwrap();
    decode_affine_band_into(GgufTensorType::MXFP4, &f.payload, 0, N * OUT, K, &ptrs).unwrap();
    let cpu = slots(&cpu);
    if cfg!(target_os = "macos") {
        assert!(gpu_repack_available(), "no GPU repack kernels on macOS");
        let staging = MxArray::from_uint8(&f.payload, &[f.payload.len() as i64]).unwrap();
        let (gdst, _) =
            RepackDst::alloc_for_kind(info, GpuRepackKind::Mxfp4, OutLayout::RowMajor).unwrap();
        let gpu = gpu_repack_band(
            &staging,
            0,
            GpuRepackKind::Mxfp4,
            OutLayout::RowMajor,
            K as u32,
            0,
            (N * OUT) as u32,
            gdst,
        )
        .unwrap();
        assert_eq!(slots(&gpu), cpu, "GPU Mxfp4 repack != CPU loader");
    }
    cpu
}

fn perm(e: usize) -> usize {
    (3 * e + 2) % N
}

fn expert_reads(f: &Fixture, b: &Bank) -> Vec<Read> {
    (0..N)
        .map(|e| {
            let ext = Extent::expert(0, f.header.data_offset, &f.header.tensors[0], e).unwrap();
            Read::slot(ext, &b.repack, perm(e))
        })
        .collect()
}

#[test]
fn slots_match_loader_and_gpu_repack() {
    let f = fixture();
    let want = references(&f);
    let shards = Arc::new(ShardSet::open(&[&f.path]).unwrap());
    let pool = IoPool::new(Arc::clone(&shards), 3).unwrap();
    let b = bank(&f);
    let mut reads = expert_reads(&f, &b);
    assert_eq!(reads[0].extent.offset, ABS);
    assert_eq!(reads[1].extent.offset, ABS + PER as u64);
    let low = reads.split_off(4);
    // SAFETY: `b` outlives both tickets; one read per slot.
    let (t_low, t_high) = unsafe {
        (
            pool.submit(low, Priority::Low).unwrap(),
            pool.submit(reads, Priority::High).unwrap(),
        )
    };
    let (s_low, s_high) = (t_low.wait().unwrap(), t_high.wait().unwrap());
    assert_eq!((s_low.reads, s_high.reads), (3, 4));
    assert_eq!(s_low.bytes + s_high.bytes, (N * PER) as u64);
    let got = slots(&b.dst);
    for (e, w) in want.iter().enumerate() {
        assert_eq!(&got[perm(e)], w, "expert {e} in slot {}", perm(e));
    }

    let mut raw = vec![0u8; 333];
    let ext = Extent {
        shard: 0,
        offset: ABS + 5,
        len: raw.len() as u64,
    };
    // SAFETY: `raw` outlives the ticket.
    let st = unsafe { pool.submit(vec![Read::raw(ext, raw.as_mut_ptr())], Priority::Low) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(st.bytes, 333);
    assert_eq!(raw, f.payload[5..338]);
}

fn assert_unpublished(b: &Bank, want: &[(Vec<u8>, Vec<u8>)], failed: &[usize]) {
    let got = slots(&b.dst);
    for (e, w) in want.iter().enumerate() {
        let slot = &got[perm(e)];
        if failed.contains(&e) {
            assert!(
                slot.0.iter().chain(&slot.1).all(|&x| x == SENTINEL),
                "expert {e}: slot written after a failed read"
            );
        } else {
            assert_eq!(slot, w, "expert {e}");
        }
    }
}

#[test]
fn short_read_fails_and_leaves_slot_unwritten() {
    let f = fixture();
    let want = references(&f);
    let shards = Arc::new(ShardSet::open(&[&f.path]).unwrap());
    let pool = IoPool::new(Arc::clone(&shards), 4).unwrap();
    let b = bank(&f);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&f.path)
        .unwrap()
        .set_len(ABS + (5 * PER + PER / 2) as u64)
        .unwrap();
    // SAFETY: `b` outlives the ticket; one read per slot.
    let err = unsafe { pool.submit(expert_reads(&f, &b), Priority::High) }
        .unwrap()
        .wait()
        .unwrap_err()
        .reason;
    assert!(err.contains("2 of 7 reads failed"), "{err}");
    assert!(err.contains("short read"), "{err}");
    assert_unpublished(&b, &want, &[5, 6]);
    assert!(shards.recheck().is_err());
}

#[test]
fn eio_fails_and_leaves_slot_unwritten() {
    let f = fixture();
    let want = references(&f);
    let shards = Arc::new(ShardSet::open(&[&f.path]).unwrap());
    // Inside expert 3's 16 KiB-padded window only (one expert > one page).
    *lock(&shards.fault) = Some((0, ABS + 3 * PER as u64 + 12_000));
    let pool = IoPool::new(Arc::clone(&shards), 4).unwrap();
    let b = bank(&f);
    // SAFETY: `b` outlives the ticket; one read per slot.
    let err = unsafe { pool.submit(expert_reads(&f, &b), Priority::Low) }
        .unwrap()
        .wait()
        .unwrap_err()
        .reason;
    assert!(err.contains("1 of 7 reads failed"), "{err}");
    assert!(
        err.contains(&std::io::Error::from_raw_os_error(libc::EIO).to_string()),
        "{err}"
    );
    assert_unpublished(&b, &want, &[3]);
    shards.recheck().unwrap();
}

#[test]
fn submit_refuses_bad_reads() {
    let f = fixture();
    let shards = Arc::new(ShardSet::open(&[&f.path]).unwrap());
    let pool = IoPool::new(shards, 4).unwrap();
    let b = bank(&f);
    let good = Extent::expert(0, f.header.data_offset, &f.header.tensors[0], 6).unwrap();
    let file_len = std::fs::metadata(&f.path).unwrap().len();
    let cases: Vec<(&str, Vec<Read>)> = vec![
        (
            "outside",
            vec![Read::slot(
                Extent {
                    offset: file_len - 10,
                    ..good
                },
                &b.repack,
                0,
            )],
        ),
        (
            "shard",
            vec![Read::slot(Extent { shard: 1, ..good }, &b.repack, 0)],
        ),
        (
            "bytes, the slot takes",
            vec![Read::slot(Extent { len: 17, ..good }, &b.repack, 0)],
        ),
        ("slot 7 of 7", vec![Read::slot(good, &b.repack, N)]),
        (
            "read twice",
            vec![
                Read::slot(good, &b.repack, 2),
                Read::slot(good, &b.repack, 2),
            ],
        ),
        ("null raw", vec![Read::raw(good, std::ptr::null_mut())]),
    ];
    for (want, reads) in cases {
        // SAFETY: refused before anything is queued.
        let err = unsafe { pool.submit(reads, Priority::High) }
            .err()
            .unwrap_or_else(|| panic!("{want}: accepted"))
            .reason;
        assert!(err.contains(want), "{want}: {err}");
    }
    let got = slots(&b.dst);
    assert!(
        got.iter()
            .all(|(w, s)| w.iter().chain(s).all(|&x| x == SENTINEL))
    );
    // SAFETY: `b` outlives the ticket.
    let st = unsafe { pool.submit(vec![Read::slot(good, &b.repack, 1)], Priority::Low) }
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(st.bytes, PER as u64);
}

#[test]
fn threads_env_range() {
    assert_eq!(threads_from(None).unwrap(), DEFAULT_THREADS);
    assert_eq!(threads_from(Some("")).unwrap(), DEFAULT_THREADS);
    assert_eq!(threads_from(Some("4")).unwrap(), 4);
    assert_eq!(threads_from(Some(" 18 ")).unwrap(), 18);
    for bad in ["3", "19", "0", "x", "-4"] {
        assert!(threads_from(Some(bad)).is_err(), "{bad}");
    }
}

/// Independent MXFP4 -> MLX `mxfp4` packing: codes `j` / `j + 16` are the
/// low / high nibble of byte `1 + j`, 8 codes per u32 LSB first; byte 0 is
/// the E8M0 scale.
fn ref_mxfp4(blocks: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut w = Vec::with_capacity(blocks.len() / 17 * 16);
    let mut s = Vec::with_capacity(blocks.len() / 17);
    for b in blocks.as_chunks::<17>().0 {
        s.push(b[0]);
        let mut codes = [0u32; 32];
        for j in 0..16 {
            codes[j] = u32::from(b[1 + j] & 15);
            codes[j + 16] = u32::from(b[1 + j] >> 4);
        }
        for word in codes.as_chunks::<8>().0 {
            let v = word
                .iter()
                .enumerate()
                .fold(0u32, |a, (i, &c)| a | c << (4 * i));
            w.extend_from_slice(&v.to_le_bytes());
        }
    }
    (w, s)
}

struct RealProj {
    shard: usize,
    data_offset: u64,
    info: crate::utils::gguf::GgufTensorInfo,
}

impl RealProj {
    fn rows(&self) -> usize {
        self.info.mlx_shape()[1] as usize
    }
    fn k(&self) -> usize {
        self.info.mlx_shape()[2] as usize
    }
    fn extent(&self, e: usize) -> Extent {
        Extent::expert(self.shard, self.data_offset, &self.info, e).unwrap()
    }
}

struct VecBank {
    w: Vec<u32>,
    s: Vec<u8>,
    repack: SlotRepack,
    wn: usize,
    sn: usize,
}

impl VecBank {
    fn new(slots: usize, rows: usize, k: usize) -> Self {
        let (wn, sn) = (rows * k / 8, rows * k / 32);
        let mut w = vec![0u32; slots * wn];
        let mut s = vec![0u8; slots * sn];
        // SAFETY: the vectors hold exactly these sizes and are never resized.
        let repack =
            unsafe { SlotRepack::mxfp4(w.as_mut_ptr(), s.as_mut_ptr(), slots, rows, k) }.unwrap();
        Self {
            w,
            s,
            repack,
            wn,
            sn,
        }
    }
    fn slot(&self, i: usize) -> (Vec<u8>, Vec<u8>) {
        let w = &self.w[i * self.wn..(i + 1) * self.wn];
        (
            w.iter().flat_map(|x| x.to_le_bytes()).collect(),
            self.s[i * self.sn..(i + 1) * self.sn].to_vec(),
        )
    }
}

fn disk_bytes_read() -> u64 {
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
    assert_eq!(rc, 0);
    info.ri_diskio_bytesread
}

/// `(resident, total)` pages of `extents` in the page cache (expert shards
/// only; nothing is touched).
fn resident_pages(shards: &ShardSet, extents: &[Extent]) -> (usize, usize) {
    // SAFETY: plain sysconf query.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let (mut resident, mut pages) = (0usize, 0usize);
    for e in extents {
        assert!(
            !shards
                .path(e.shard)
                .unwrap()
                .to_string_lossy()
                .contains("-00008-of-")
        );
        let start = e.offset / page * page;
        let len = (e.offset + e.len - start) as usize;
        let fd = {
            use std::os::fd::AsRawFd;
            shards.file(e.shard).unwrap().as_raw_fd()
        };
        // SAFETY: read-only shared mapping of a range inside the file, only
        // passed to mincore and unmapped below; no page is touched.
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                start as libc::off_t,
            );
            assert_ne!(p, libc::MAP_FAILED);
            let n = len.div_ceil(page as usize);
            let mut vec = vec![0 as libc::c_char; n];
            assert_eq!(libc::mincore(p, len, vec.as_mut_ptr()), 0);
            resident += vec.iter().filter(|&&v| v & 1 != 0).count();
            pages += n;
            libc::munmap(p, len);
        }
    }
    (resident, pages)
}

/// Real shards 2-7 (`DSV41_DIR`): pool slots == plain `pread` + independent
/// repack for 64 random experts; then a decode-shaped run (6 misses x 40
/// layers, barrier per layer) prints the rate; then a `mincore` probe of the
/// last layer's extents. Shard 8 is never opened.
#[test]
#[ignore = "needs DSV41_DIR with the real DeepSeek-V4.1 shards"]
fn real_smoke() {
    let dir = PathBuf::from(std::env::var("DSV41_DIR").expect("DSV41_DIR"));
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            (2..=7).any(|i| n.ends_with(&format!("-0000{i}-of-00009.gguf")))
        })
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 6, "{paths:?}");
    assert!(
        paths
            .iter()
            .all(|p| !p.to_string_lossy().contains("-00008-of-"))
    );
    let shards = Arc::new(ShardSet::open(&paths).unwrap());
    let mut layers: BTreeMap<usize, [Option<RealProj>; 3]> = BTreeMap::new();
    for i in 0..shards.len() {
        let h = parse_gguf_reader(std::io::BufReader::new(shards.file(i).unwrap())).unwrap();
        for t in h.tensors {
            let Some(rest) = t.name.strip_prefix("blk.") else {
                continue;
            };
            let (l, proj) = rest.split_once('.').unwrap();
            let l: usize = l.parse().unwrap();
            let p = match proj {
                "ffn_gate_exps.weight" => 0,
                "ffn_up_exps.weight" => 1,
                "ffn_down_exps.weight" => 2,
                _ => continue,
            };
            assert_eq!(t.tensor_type, GgufTensorType::MXFP4, "{}", t.name);
            assert_eq!(t.mlx_shape()[0], 384, "{}", t.name);
            layers.entry(l).or_default()[p] = Some(RealProj {
                shard: i,
                data_offset: h.data_offset,
                info: t,
            });
        }
    }
    assert_eq!(layers.len(), 40);
    let layers: Vec<[RealProj; 3]> = layers
        .into_values()
        .map(|ps| ps.map(|p| p.expect("gate/up/down per layer")))
        .collect();
    let pool = IoPool::from_env(Arc::clone(&shards)).unwrap();
    println!(
        "real_smoke: {} reader threads, profile debug_assertions={}",
        pool.threads(),
        cfg!(debug_assertions)
    );
    let mut rng = 0x5EED_u64;
    let mut next = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as usize
    };
    let plain: Vec<File> = paths.iter().map(|p| open_nocache(p).unwrap()).collect();
    let plain_read = |e: Extent| {
        let mut buf = vec![0u8; e.len as usize];
        plain[e.shard].read_exact_at(&mut buf, e.offset).unwrap();
        buf
    };

    // A: 64 random experts, 8 per batch.
    let (mut a_bytes, mut checked) = (0u64, 0usize);
    for _ in 0..8 {
        let picks: Vec<(usize, usize)> = (0..8).map(|_| (next(40), next(384))).collect();
        let banks: Vec<VecBank> = (0..3)
            .map(|p| VecBank::new(8, layers[0][p].rows(), layers[0][p].k()))
            .collect();
        let mut reads = Vec::new();
        for (slot, &(l, e)) in picks.iter().enumerate() {
            for p in 0..3 {
                assert_eq!(
                    (layers[l][p].rows(), layers[l][p].k()),
                    (layers[0][p].rows(), layers[0][p].k())
                );
                reads.push(Read::slot(layers[l][p].extent(e), &banks[p].repack, slot));
            }
        }
        // SAFETY: `banks` outlive the ticket; one read per slot.
        let st = unsafe { pool.submit(reads, Priority::Low) }
            .unwrap()
            .wait()
            .unwrap();
        a_bytes += st.bytes;
        for (slot, &(l, e)) in picks.iter().enumerate() {
            for p in 0..3 {
                let want = ref_mxfp4(&plain_read(layers[l][p].extent(e)));
                assert!(banks[p].slot(slot) == want, "layer {l} expert {e} proj {p}");
                checked += 1;
            }
        }
    }
    println!("real_smoke A: {checked} extents ({a_bytes} B) == plain pread + independent repack");

    // B: decode-shaped, 6 misses per layer, barrier per layer.
    let banks: Vec<VecBank> = (0..3)
        .map(|p| VecBank::new(6, layers[0][p].rows(), layers[0][p].k()))
        .collect();
    let (mut total, mut layer_ms) = (IoStats::default(), Vec::new());
    let (mut last, mut before_res) = (Vec::new(), 0);
    let disk0 = disk_bytes_read();
    let t0 = Instant::now();
    for (l, ps) in layers.iter().enumerate() {
        let mut picks: Vec<usize> = Vec::new();
        while picks.len() < 6 {
            let e = next(384);
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        let reads: Vec<Read> = picks
            .iter()
            .enumerate()
            .flat_map(|(slot, &e)| (0..3).map(move |p| (slot, e, p)))
            .map(|(slot, e, p)| Read::slot(ps[p].extent(e), &banks[p].repack, slot))
            .collect();
        if l == 39 {
            last = reads.iter().map(|r| r.extent).collect();
            before_res = resident_pages(&shards, &last).0;
        }
        // SAFETY: `banks` outlive the ticket; one read per slot.
        let st = unsafe { pool.submit(reads, Priority::High) }
            .unwrap()
            .wait()
            .unwrap();
        layer_ms.push(st.ns as f64 / 1e6);
        total.reads += st.reads;
        total.bytes += st.bytes;
        total.read_ns += st.read_ns;
        total.repack_ns += st.repack_ns;
        total.max_latency_ns = total.max_latency_ns.max(st.max_latency_ns);
        if l == 0 || l == 39 {
            for p in 0..3 {
                assert!(
                    banks[p].slot(5) == ref_mxfp4(&plain_read(ps[p].extent(picks[5]))),
                    "B layer {l} proj {p}"
                );
            }
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let disk = disk_bytes_read() - disk0;
    println!(
        "real_smoke B: {:.3} GB from disk (rest served by the unified buffer cache) = {:.2} GB/s disk",
        disk as f64 / 1e9,
        disk as f64 / 1e9 / wall
    );
    layer_ms.sort_by(f64::total_cmp);
    println!(
        "real_smoke B: {} reads, {:.3} GB in {:.2} s = {:.2} GB/s; per layer p50 {:.1} ms max {:.1} ms; \
         max read latency {:.1} ms; read {:.0} ms, CPU repack {:.0} ms ({:.2} ms per expert matrix) summed over threads",
        total.reads,
        total.bytes as f64 / 1e9,
        wall,
        total.bytes as f64 / 1e9 / wall,
        layer_ms[layer_ms.len() / 2],
        layer_ms[layer_ms.len() - 1],
        total.max_latency_ns as f64 / 1e6,
        total.read_ns as f64 / 1e6,
        total.repack_ns as f64 / 1e6,
        total.repack_ns as f64 / 1e6 / total.reads as f64,
    );
    assert_eq!(total.bytes, 40 * 6 * 3 * 6_266_880);

    // C: does an F_NOCACHE read leave the extent in the page cache?
    let one = VecBank::new(1, layers[0][0].rows(), layers[0][0].k());
    for (l, e) in [(5, 57), (12, 158), (19, 259), (26, 360)] {
        let ext = layers[l][0].extent(e);
        let (before, n) = resident_pages(&shards, &[ext]);
        let d0 = disk_bytes_read();
        // SAFETY: `one` outlives the ticket.
        unsafe { pool.submit(vec![Read::slot(ext, &one.repack, 0)], Priority::High) }
            .unwrap()
            .wait()
            .unwrap();
        let disk = disk_bytes_read() - d0;
        println!(
            "real_smoke C: layer {l} expert {e} gate (offset % page {}): {before} of {n} pages resident before, \
             {} after; {disk} B from disk",
            ext.offset % 16384,
            resident_pages(&shards, &[ext]).0
        );
    }

    let (resident, pages) = resident_pages(&shards, &last);
    println!(
        "real_smoke mincore: last layer's 18 extents {before_res} of {pages} pages resident before the reads, \
         {resident} after"
    );
}
