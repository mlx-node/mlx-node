//! Mapped expert pack v3: one file per routed `(layer, tensor)` holding the
//! three quantized arrays as page-aligned sections with experts in id order,
//! so a section can back a bank directly (slot id == expert id) or be read by
//! expert range. Built once beside the per-expert chunk cache from its
//! verified chunks, else from the checkpoint through the ordinary importer.
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use super::packed_cache::{self, ArrayInfo, Header, VerifiedChunk};

pub(super) const DIR_PREFIX: &str = ".mlx-qwen4-mapped-v3-";
/// Apple page size: every section offset and length is a multiple, so a
/// section maps as one no-copy buffer.
pub(super) const PAGE: u64 = 16 << 10;
const MAGIC: &[u8; 8] = b"Q4MAP03\0";
const VERSION: u32 = 3;
/// Experts fetched and written per pass; bounds builder memory to tens of MiB.
const BUILD_GROUP: usize = 16;
const VERIFY_CHUNK: usize = 8 << 20;
const STAMP_EVERY: Duration = Duration::from_secs(5);

fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
}

fn round_up(n: u64) -> u64 {
    n.div_ceil(PAGE) * PAGE
}

fn element_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "U32" | "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        "U8" | "I8" => Some(1),
        _ => None,
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(super) struct SectionInfo {
    pub(super) dtype: String,
    /// `[experts * rows_per_expert, columns]`: the bank shape a mapping yields.
    pub(super) shape: [u64; 2],
    /// Bytes of one expert; expert `e` starts at `offset + e * record_bytes`.
    pub(super) record_bytes: u64,
    pub(super) offset: u64,
    /// Page-rounded; the tail past `experts * record_bytes` is zero.
    pub(super) length: u64,
    /// Over the `experts * record_bytes` payload bytes, not the padding.
    pub(super) sha256: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(super) struct MappedHeader {
    pub(super) identity: String,
    pub(super) name: String,
    pub(super) experts: u64,
    pub(super) rows_per_expert: u64,
    pub(super) width: u64,
    pub(super) group: i32,
    pub(super) bits: i32,
    pub(super) mode: String,
    pub(super) sections: Vec<SectionInfo>,
    pub(super) file_bytes: u64,
}

impl MappedHeader {
    fn payload(&self, section: &SectionInfo) -> u64 {
        section.record_bytes * self.experts
    }

    fn validate(&self) -> Result<()> {
        let bad = |what: &str| Err(err(format!("Invalid mapped pack header: {what}")));
        if self.identity.len() != 64 || self.name.is_empty() || self.name.contains('/') {
            return bad("identity or name");
        }
        if self.experts == 0 || self.rows_per_expert == 0 || self.width == 0 {
            return bad("geometry");
        }
        if !(2..=3).contains(&self.sections.len()) {
            return bad("section count");
        }
        let mut cursor = PAGE;
        for s in &self.sections {
            let Some(elem) = element_bytes(&s.dtype) else {
                return bad("dtype");
            };
            if s.offset != cursor || !s.length.is_multiple_of(PAGE) || s.length == 0 {
                return bad("section placement");
            }
            let payload = s
                .record_bytes
                .checked_mul(self.experts)
                .ok_or_else(|| err("Invalid mapped pack header: payload overflow"))?;
            if s.record_bytes == 0 || payload > s.length || s.length - payload >= PAGE {
                return bad("section length");
            }
            if s.shape[0] != self.experts * self.rows_per_expert
                || s.shape[0]
                    .checked_mul(s.shape[1])
                    .and_then(|n| n.checked_mul(elem))
                    != Some(payload)
                || s.sha256.len() != 64
            {
                return bad("section shape");
            }
            cursor = s
                .offset
                .checked_add(s.length)
                .ok_or_else(|| err("Invalid mapped pack header: offset overflow"))?;
        }
        if cursor != self.file_bytes {
            return bad("file size");
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(self).map_err(err)?;
        if json.len() as u64 > PAGE - 48 {
            return Err(err("Mapped pack header exceeds its page"));
        }
        let mut page = Vec::with_capacity(PAGE as usize);
        page.extend_from_slice(MAGIC);
        page.extend_from_slice(&VERSION.to_le_bytes());
        page.extend_from_slice(&(json.len() as u32).to_le_bytes());
        page.extend_from_slice(&Sha256::digest(&json));
        page.extend_from_slice(&json);
        page.resize(PAGE as usize, 0);
        Ok(page)
    }

    fn decode(page: &[u8]) -> Result<Self> {
        if page.len() != PAGE as usize || &page[..8] != MAGIC {
            return Err(err("Invalid mapped pack magic"));
        }
        let version = u32::from_le_bytes(page[8..12].try_into().map_err(err)?);
        let len = u32::from_le_bytes(page[12..16].try_into().map_err(err)?) as usize;
        if version != VERSION || len > PAGE as usize - 48 {
            return Err(err("Unsupported mapped pack header"));
        }
        let json = &page[48..48 + len];
        if Sha256::digest(json)[..] != page[16..48] {
            return Err(err("Mapped pack header checksum mismatch"));
        }
        let header: Self = serde_json::from_slice(json).map_err(err)?;
        header.validate()?;
        Ok(header)
    }
}

/// One expert of one tensor in chunk encoding: the arrays' `ArrayInfo`s and
/// their concatenated little-endian bytes.
#[derive(Clone)]
pub(super) struct ExpertRecord {
    pub(super) group: i32,
    pub(super) bits: i32,
    pub(super) mode: String,
    pub(super) arrays: Vec<ArrayInfo>,
    pub(super) data: Vec<u8>,
}

impl ExpertRecord {
    pub(super) fn from_chunk(chunk: VerifiedChunk) -> Self {
        Self {
            group: chunk.header.group,
            bits: chunk.header.bits,
            mode: chunk.header.mode,
            arrays: chunk.header.arrays,
            data: chunk.data,
        }
    }

    pub(super) fn from_weight(weight: &super::weights::Weight) -> Result<Self> {
        let (arrays, data) = packed_cache::encode_weight(weight)?;
        Ok(Self {
            group: weight.group,
            bits: weight.bits,
            mode: weight.mode.clone(),
            arrays,
            data,
        })
    }

    fn parts(&self) -> Result<Vec<&[u8]>> {
        let mut rest = self.data.as_slice();
        let mut parts = Vec::with_capacity(self.arrays.len());
        for a in &self.arrays {
            let (part, tail) = rest
                .split_at_checked(a.bytes)
                .ok_or_else(|| err("Expert record is shorter than its arrays"))?;
            parts.push(part);
            rest = tail;
        }
        if !rest.is_empty() {
            return Err(err("Expert record has trailing bytes"));
        }
        Ok(parts)
    }

    fn same_geometry(&self, other: &Self) -> bool {
        self.group == other.group
            && self.bits == other.bits
            && self.mode == other.mode
            && self.arrays == other.arrays
    }
}

/// Where the builder takes expert records from. `cached` may run on worker
/// threads; `import` creates MLX arrays and runs on the calling thread only.
pub(super) trait ExpertSource: Sync {
    fn name(&self) -> &str;
    fn experts(&self) -> usize;
    fn rows_per_expert(&self) -> usize;
    fn width(&self) -> usize;
    fn cached(&self, expert: usize) -> Result<Option<ExpertRecord>>;
    fn import(&self, expert: usize) -> Result<ExpertRecord>;
}

fn file_name(name: &str) -> String {
    format!("{name}.mapped")
}

fn tmp_name(name: &str) -> String {
    format!(".{name}.tmp")
}

/// Builder writes only: 74 GiB written through the page cache would evict
/// the pages the running model reads. Reads must keep the page cache, the
/// tier that serves warm re-reads; F_NOCACHE reads lost every warm row.
fn set_nocache(file: &File) {
    // SAFETY: `file` owns an open descriptor; F_NOCACHE only changes caching.
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
}

pub(super) struct LayerFile {
    file: File,
    pub(super) header: MappedHeader,
}

impl LayerFile {
    /// `None` when the file is absent; `Err` when present but not a valid
    /// pack for `name`/`identity` (truncated, foreign, corrupt).
    fn open(dirfd: &File, name: &str, identity: &str, verify: bool) -> Result<Option<Self>> {
        let file = match packed_cache::dir_open_at(
            dirfd,
            OsStr::new(&file_name(name)),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        ) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(err(e)),
        };
        let len = file.metadata().map_err(err)?.len();
        if len < PAGE {
            return Err(err("Mapped pack file is shorter than its header"));
        }
        let mut page = vec![0u8; PAGE as usize];
        file.read_exact_at(&mut page, 0).map_err(err)?;
        let header = MappedHeader::decode(&page)?;
        if header.name != name || header.identity != identity {
            return Err(err(
                "Mapped pack file belongs to another tensor or checkpoint",
            ));
        }
        if header.file_bytes != len {
            return Err(err("Mapped pack file size differs from its header"));
        }
        let layer = Self { file, header };
        if verify {
            layer.verify_sections()?;
        }
        Ok(Some(layer))
    }

    /// Re-hash every section payload against the header.
    pub(super) fn verify_sections(&self) -> Result<()> {
        verify_sections(&self.file, &self.header)
    }

    /// Bytes of experts `first..first + count` of every section, in section
    /// order, each slice contiguous in expert order.
    pub(super) fn read_range(&self, first: usize, count: usize) -> Result<Vec<Vec<u8>>> {
        let h = &self.header;
        if count == 0
            || (first as u64)
                .checked_add(count as u64)
                .is_none_or(|end| end > h.experts)
        {
            return Err(err("Mapped pack expert range is outside the tensor"));
        }
        let mut out = Vec::with_capacity(h.sections.len());
        for s in &h.sections {
            let bytes = s.record_bytes * count as u64;
            let mut buf = vec![0u8; bytes as usize];
            self.file
                .read_exact_at(&mut buf, s.offset + s.record_bytes * first as u64)
                .map_err(err)?;
            out.push(buf);
        }
        Ok(out)
    }

    /// One expert in the chunk encoding the slot path already imports.
    fn read_expert(&self, expert: usize) -> Result<VerifiedChunk> {
        let h = &self.header;
        let total: u64 = h.sections.iter().map(|s| s.record_bytes).sum();
        let mut data = Vec::with_capacity(total as usize);
        let mut arrays = Vec::with_capacity(h.sections.len());
        for (s, part) in h.sections.iter().zip(self.read_range(expert, 1)?) {
            arrays.push(ArrayInfo {
                dtype: s.dtype.clone(),
                shape: vec![h.rows_per_expert as i64, s.shape[1] as i64],
                bytes: part.len(),
            });
            data.extend_from_slice(&part);
        }
        Ok(VerifiedChunk {
            header: Header {
                key: format!("{}#{expert}", h.name),
                group: h.group,
                bits: h.bits,
                mode: h.mode.clone(),
                arrays,
            },
            data,
            file_len: total,
        })
    }
}

fn verify_sections(file: &File, header: &MappedHeader) -> Result<()> {
    let mut buf = vec![0u8; VERIFY_CHUNK];
    for s in &header.sections {
        let mut hasher = Sha256::new();
        let mut remaining = header.payload(s);
        let mut at = s.offset;
        while remaining > 0 {
            let n = remaining.min(VERIFY_CHUNK as u64) as usize;
            file.read_exact_at(&mut buf[..n], at).map_err(err)?;
            hasher.update(&buf[..n]);
            at += n as u64;
            remaining -= n as u64;
        }
        if hex(&hasher.finalize()) != s.sha256 {
            return Err(err(format!(
                "Mapped pack section checksum mismatch in {} ({})",
                header.name, s.dtype
            )));
        }
    }
    Ok(())
}

/// Fetch `first..first + count`: the chunk fast path on worker threads, the
/// importer on this thread for whatever the cache lacks. Every record must
/// share the first expert's geometry.
fn fetch_group(
    source: &dyn ExpertSource,
    first: usize,
    count: usize,
    prototype: Option<&ExpertRecord>,
) -> Result<Vec<ExpertRecord>> {
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 8)
        .min(count);
    let mut records: Vec<Option<ExpertRecord>> = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..workers)
            .map(|w| {
                scope.spawn(move || {
                    (w..count)
                        .step_by(workers)
                        .map(|i| (i, source.cached(first + i).ok().flatten()))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut out: Vec<Option<ExpertRecord>> = (0..count).map(|_| None).collect();
        for job in jobs {
            for (i, record) in job
                .join()
                .map_err(|_| err("Mapped pack reader worker panicked"))?
            {
                out[i] = record;
            }
        }
        Ok::<_, Error>(out)
    })?;
    let mut out = Vec::with_capacity(count);
    for (i, slot) in records.iter_mut().enumerate() {
        let record = match slot.take() {
            Some(record) => record,
            None => source.import(first + i)?,
        };
        if let Some(prototype) = prototype.or(out.first())
            && !record.same_geometry(prototype)
        {
            return Err(err(format!(
                "Expert {} of {} differs in quantization geometry",
                first + i,
                source.name()
            )));
        }
        out.push(record);
    }
    Ok(out)
}

/// Build one tensor's file in the pinned dir: a flock-held `.tmp`, the
/// sections written in expert order with running checksums, the header page,
/// fsync, a read-back verification, then the atomic rename. `None` when
/// another builder holds this tensor's staging file.
fn build_file(
    dirfd: &File,
    identity: &str,
    source: &dyn ExpertSource,
) -> Result<Option<MappedHeader>> {
    let name = source.name();
    let tmp = tmp_name(name);
    let staging = packed_cache::dir_open_at(
        dirfd,
        OsStr::new(&tmp),
        libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0o644,
    )
    .map_err(err)?;
    // SAFETY: `staging` owns an open descriptor for the flock's lifetime.
    if unsafe { libc::flock(staging.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Ok(None);
    }
    let result = write_file(dirfd, &staging, identity, source);
    if result.is_err() {
        let _ = packed_cache::dir_unlinkat(dirfd, OsStr::new(&tmp));
    }
    result.map(Some)
}

fn write_file(
    dirfd: &File,
    staging: &File,
    identity: &str,
    source: &dyn ExpertSource,
) -> Result<MappedHeader> {
    let name = source.name();
    let experts = source.experts();
    let rows = source.rows_per_expert();
    if experts == 0 || rows == 0 {
        return Err(err(format!("{name} has no experts to pack")));
    }
    staging.set_len(0).map_err(err)?;
    set_nocache(staging);
    let first = fetch_group(source, 0, BUILD_GROUP.min(experts), None)?;
    let prototype = first[0].clone();
    let prototype = &prototype;
    let parts = prototype.parts()?;
    if !(2..=3).contains(&parts.len()) {
        return Err(err(format!(
            "{name} expert record has {} arrays",
            parts.len()
        )));
    }
    let mut sections = Vec::with_capacity(parts.len());
    let mut cursor = PAGE;
    for a in &prototype.arrays {
        let elem = element_bytes(&a.dtype).ok_or_else(|| err("Unsupported expert dtype"))?;
        if a.shape.len() != 2 || a.shape[0] != rows as i64 || a.shape[1] <= 0 {
            return Err(err(format!("{name} expert arrays are not [rows, columns]")));
        }
        let cols = a.shape[1] as u64;
        if rows as u64 * cols * elem != a.bytes as u64 {
            return Err(err(format!(
                "{name} expert array bytes disagree with its shape"
            )));
        }
        let payload = a.bytes as u64 * experts as u64;
        sections.push(SectionInfo {
            dtype: a.dtype.clone(),
            shape: [experts as u64 * rows as u64, cols],
            record_bytes: a.bytes as u64,
            offset: cursor,
            length: round_up(payload),
            sha256: String::new(),
        });
        cursor += round_up(payload);
    }
    let file_bytes = cursor;
    if !packed_cache::has_disk_space(dirfd, file_bytes) {
        return Err(err(format!(
            "Not enough disk space for {name} ({:.2} GiB plus the 16 GiB reserve)",
            file_bytes as f64 / (1u64 << 30) as f64
        )));
    }
    staging.set_len(file_bytes).map_err(err)?;
    let mut hashers: Vec<Sha256> = sections.iter().map(|_| Sha256::new()).collect();
    let mut pending = Some(first);
    let mut start = 0usize;
    while start < experts {
        let count = BUILD_GROUP.min(experts - start);
        let group = match pending.take() {
            Some(group) => group,
            None => fetch_group(source, start, count, Some(prototype))?,
        };
        for (s, section) in sections.iter().enumerate() {
            let mut slab = Vec::with_capacity(section.record_bytes as usize * count);
            for record in &group {
                slab.extend_from_slice(record.parts()?[s]);
            }
            staging
                .write_all_at(&slab, section.offset + section.record_bytes * start as u64)
                .map_err(err)?;
            hashers[s].update(&slab);
        }
        start += count;
    }
    for (section, hasher) in sections.iter_mut().zip(hashers) {
        section.sha256 = hex(&hasher.finalize());
    }
    let header = MappedHeader {
        identity: identity.into(),
        name: name.into(),
        experts: experts as u64,
        rows_per_expert: rows as u64,
        width: source.width() as u64,
        group: prototype.group,
        bits: prototype.bits,
        mode: prototype.mode.clone(),
        sections,
        file_bytes,
    };
    header.validate()?;
    staging.write_all_at(&header.encode()?, 0).map_err(err)?;
    staging.sync_all().map_err(err)?;
    // Read back what landed on disk, not what was handed to the kernel.
    let mut page = vec![0u8; PAGE as usize];
    staging.read_exact_at(&mut page, 0).map_err(err)?;
    if MappedHeader::decode(&page)? != header
        || staging.metadata().map_err(err)?.len() != file_bytes
    {
        return Err(err(format!("{name} header read-back differs")));
    }
    verify_sections(staging, &header)?;
    packed_cache::dir_renameat(dirfd, &tmp_name(name), &file_name(name)).map_err(err)?;
    Ok(header)
}

pub(super) struct BuildReport {
    pub(super) opened: usize,
    pub(super) built: usize,
    pub(super) skipped: usize,
}

pub(super) struct MappedPack {
    dir: PathBuf,
    files: HashMap<String, LayerFile>,
    stamp: Mutex<Instant>,
    reads: AtomicU64,
    bytes: AtomicU64,
}

impl MappedPack {
    /// Open every valid tensor file under `dir`, rebuild invalid ones and
    /// build missing ones from `sources`. Building stops at the first
    /// failure (disk, source) and the files that exist stay usable; a tensor
    /// another process is building right now is left out of this pack.
    /// Sections are hashed when built and read back; an open checks the
    /// header and the file size only.
    pub(super) fn open_or_build(
        dir: &Path,
        identity: &str,
        manifest: Option<&[u8]>,
        sources: &[&dyn ExpertSource],
    ) -> Result<(Self, BuildReport)> {
        Self::open_or_build_with(dir, identity, manifest, sources, false)
    }

    /// `verify` re-hashes every section of every file it opens.
    #[cfg(test)]
    pub(super) fn open_or_build_verifying(
        dir: &Path,
        identity: &str,
        manifest: Option<&[u8]>,
        sources: &[&dyn ExpertSource],
    ) -> Result<(Self, BuildReport)> {
        Self::open_or_build_with(dir, identity, manifest, sources, true)
    }

    fn open_or_build_with(
        dir: &Path,
        identity: &str,
        manifest: Option<&[u8]>,
        sources: &[&dyn ExpertSource],
        verify: bool,
    ) -> Result<(Self, BuildReport)> {
        packed_cache::unlink_if_symlink(dir);
        std::fs::create_dir_all(dir).map_err(err)?;
        let dirfd = packed_cache::open_dir_nofollow(dir)
            .ok_or_else(|| err(format!("{} is not a real directory", dir.display())))?;
        if let Some(manifest) = manifest
            && packed_cache::dir_stat(&dirfd, OsStr::new(".manifest"))
                .is_none_or(|stat| (stat.st_mode & libc::S_IFMT) != libc::S_IFREG)
        {
            packed_cache::dir_write_file(&dirfd, ".manifest", manifest).map_err(err)?;
        }
        let mut report = BuildReport {
            opened: 0,
            built: 0,
            skipped: 0,
        };
        let mut files = HashMap::with_capacity(sources.len());
        let mut building = true;
        let mut layer: Option<(String, u64, Duration)> = None;
        for source in sources {
            let name = source.name();
            match LayerFile::open(&dirfd, name, identity, verify) {
                Ok(Some(file)) => {
                    report.opened += 1;
                    files.insert(name.to_owned(), file);
                    continue;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("qwen4 mapped pack: rebuilding {name}: {e}");
                    let _ = packed_cache::dir_unlinkat(&dirfd, OsStr::new(&file_name(name)));
                }
            }
            if !building {
                report.skipped += 1;
                continue;
            }
            let started = Instant::now();
            match build_file(&dirfd, identity, *source) {
                Ok(Some(header)) => {
                    let elapsed = started.elapsed();
                    let prefix = name.split(".ffn_").next().unwrap_or(name).to_owned();
                    match &mut layer {
                        Some((current, bytes, time)) if *current == prefix => {
                            *bytes += header.file_bytes;
                            *time += elapsed;
                        }
                        _ => {
                            if let Some((current, bytes, time)) = layer.take() {
                                log_layer(&current, bytes, time);
                            }
                            layer = Some((prefix, header.file_bytes, elapsed));
                        }
                    }
                    match LayerFile::open(&dirfd, name, identity, false)? {
                        Some(file) => {
                            report.built += 1;
                            files.insert(name.to_owned(), file);
                        }
                        None => return Err(err(format!("{name} vanished after its build"))),
                    }
                }
                Ok(None) => {
                    warn!("qwen4 mapped pack: {name} is being built by another process");
                    report.skipped += 1;
                }
                Err(e) => {
                    warn!("qwen4 mapped pack: build of {name} failed: {e}");
                    building = false;
                    report.skipped += 1;
                }
            }
        }
        if let Some((current, bytes, time)) = layer.take() {
            log_layer(&current, bytes, time);
        }
        Ok((
            Self {
                dir: dir.to_path_buf(),
                files,
                stamp: Mutex::new(Instant::now()),
                reads: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
            },
            report,
        ))
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn len(&self) -> usize {
        self.files.len()
    }

    /// Whole experts served by range so far.
    pub(super) fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    pub(super) fn bytes_read(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn file(&self, name: &str) -> Option<&LayerFile> {
        self.files.get(name)
    }

    /// Keep the dir's `.manifest` mtime recent and report the read counters,
    /// at most once per `STAMP_EVERY` instead of per read.
    fn touch(&self) {
        if let Ok(mut last) = self.stamp.lock()
            && last.elapsed() >= STAMP_EVERY
        {
            *last = Instant::now();
            packed_cache::stamp_mtime(&self.dir.join(".manifest"));
            info!(
                "qwen4 mapped pack: {} expert reads, {:.2} GiB by range so far",
                self.reads(),
                self.bytes_read() as f64 / (1u64 << 30) as f64
            );
        }
    }

    /// The slot path's expert batch: every request must be one whole expert
    /// of a packed tensor, else `None` and the caller keeps its own reader.
    /// A read failure also yields `None`: the chunk reader still has the
    /// bytes, and a corrupt file is reported at the next load's open.
    pub(super) fn read_batch(
        &self,
        requests: &[(String, usize, usize)],
    ) -> Result<Option<Vec<VerifiedChunk>>> {
        let mut plan = Vec::with_capacity(requests.len());
        for (name, start, rows) in requests {
            let Some(file) = self.files.get(name) else {
                return Ok(None);
            };
            let h = &file.header;
            if *rows as u64 != h.rows_per_expert
                || !(*start as u64).is_multiple_of(h.rows_per_expert)
            {
                return Ok(None);
            }
            let expert = *start as u64 / h.rows_per_expert;
            if expert >= h.experts {
                return Err(err("Qwen4 expert id exceeds the mapped pack"));
            }
            plan.push((file, expert as usize));
        }
        if plan.is_empty() {
            return Ok(Some(Vec::new()));
        }
        self.touch();
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, 8)
            .min(plan.len());
        let plan = &plan;
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..workers)
                .map(|w| {
                    scope.spawn(move || {
                        (w..plan.len())
                            .step_by(workers)
                            .map(|i| {
                                let (file, expert) = plan[i];
                                (i, file.read_expert(expert))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut out: Vec<Option<VerifiedChunk>> = (0..plan.len()).map(|_| None).collect();
            for job in jobs {
                for (i, chunk) in job
                    .join()
                    .map_err(|_| err("Mapped pack reader worker panicked"))?
                {
                    match chunk {
                        Ok(chunk) => out[i] = Some(chunk),
                        Err(e) => {
                            warn!("qwen4 mapped pack: read failed, using chunks: {e}");
                            return Ok(None);
                        }
                    }
                }
            }
            // The caller pairs these positionally with its requests: a gap
            // would shift every later expert into the wrong slot.
            let chunks = out
                .into_iter()
                .collect::<Option<Vec<VerifiedChunk>>>()
                .ok_or_else(|| err("Mapped pack batch left a request unserved"))?;
            self.reads.fetch_add(chunks.len() as u64, Ordering::Relaxed);
            self.bytes.fetch_add(
                chunks.iter().map(|c| c.file_len).sum::<u64>(),
                Ordering::Relaxed,
            );
            Ok(Some(chunks))
        })
    }
}

fn log_layer(layer: &str, bytes: u64, time: Duration) {
    let gib = bytes as f64 / (1u64 << 30) as f64;
    let secs = time.as_secs_f64().max(1e-9);
    info!(
        "qwen4 mapped pack: built {layer} {gib:.2} GiB in {secs:.1} s ({:.2} GiB/s)",
        gib / secs
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MxArray;
    use crate::models::qwen4_exp::weights::Weight;
    use std::fs;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("qwen4-mapped-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const IDENTITY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    /// A fixture bank of `e` experts with `n` rows each: q4k codes/scales/
    /// biases, or an affine 5-bit triple, in the shapes the slot path uses.
    fn bank(e: usize, n: usize, k: usize, affine: bool, seed: u32) -> Weight {
        let bits = if affine { 5 } else { 4 };
        Weight {
            values: MxArray::from_uint32(
                &(0..e * n * k * bits / 32)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(seed))
                    .collect::<Vec<_>>(),
                &[(e * n) as i64, (k * bits / 32) as i64],
            )
            .unwrap(),
            scales: Some(if affine {
                MxArray::from_float16(
                    &(0..e * n * k / 32)
                        .map(|i| half::f16::from_f32(0.001 * (i % 11) as f32).to_bits())
                        .collect::<Vec<_>>(),
                    &[(e * n) as i64, (k / 32) as i64],
                )
                .unwrap()
            } else {
                MxArray::from_uint8(
                    &(0..e * n * k / 16)
                        .map(|i| ((i + seed as usize) % 63 + 1) as u8)
                        .collect::<Vec<_>>(),
                    &[(e * n) as i64, (k / 16) as i64],
                )
                .unwrap()
            }),
            biases: Some(
                MxArray::from_float16(
                    &(0..e * n * k / if affine { 32 } else { 128 })
                        .map(|i| half::f16::from_f32(0.0001 * (i % 7) as f32).to_bits())
                        .collect::<Vec<_>>(),
                    &[(e * n) as i64, (k / if affine { 32 } else { 128 }) as i64],
                )
                .unwrap(),
            ),
            dense_bf16: None,
            group: 32,
            bits: bits as i32,
            mode: if affine {
                "affine".into()
            } else {
                "q4k".into()
            },
        }
    }

    /// Serves one bank: experts below `cached_below` through the fast path,
    /// the rest through `import`; counts both so a test can tell them apart.
    /// Records are cut from the whole bank's row-major bytes, independently
    /// of the builder's own slicing.
    struct BankSource {
        name: String,
        bits: i32,
        mode: String,
        width: usize,
        arrays: Vec<ArrayInfo>,
        data: Vec<u8>,
        experts: usize,
        rows: usize,
        cached_below: usize,
        cached_calls: std::sync::atomic::AtomicUsize,
        imports: std::sync::atomic::AtomicUsize,
    }
    impl BankSource {
        fn new(name: &str, bank: Weight, experts: usize, cached_below: usize) -> Self {
            let rows = bank.values.shape().unwrap()[0] as usize / experts;
            let width = bank.values.shape().unwrap()[1] as usize * 32 / bank.bits as usize;
            let (arrays, data) = packed_cache::encode_weight(&bank).unwrap();
            Self {
                name: name.into(),
                bits: bank.bits,
                mode: bank.mode.clone(),
                width,
                arrays,
                data,
                experts,
                rows,
                cached_below,
                cached_calls: Default::default(),
                imports: Default::default(),
            }
        }
        fn record(&self, expert: usize) -> ExpertRecord {
            let mut arrays = Vec::new();
            let mut data = Vec::new();
            let mut base = 0usize;
            for a in &self.arrays {
                let record = a.bytes / self.experts;
                data.extend_from_slice(
                    &self.data[base + expert * record..base + (expert + 1) * record],
                );
                arrays.push(ArrayInfo {
                    dtype: a.dtype.clone(),
                    shape: vec![self.rows as i64, a.shape[1]],
                    bytes: record,
                });
                base += a.bytes;
            }
            ExpertRecord {
                group: 32,
                bits: self.bits,
                mode: self.mode.clone(),
                arrays,
                data,
            }
        }
    }
    impl ExpertSource for BankSource {
        fn name(&self) -> &str {
            &self.name
        }
        fn experts(&self) -> usize {
            self.experts
        }
        fn rows_per_expert(&self) -> usize {
            self.rows
        }
        fn width(&self) -> usize {
            self.width
        }
        fn cached(&self, expert: usize) -> Result<Option<ExpertRecord>> {
            self.cached_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok((expert < self.cached_below).then(|| self.record(expert)))
        }
        fn import(&self, expert: usize) -> Result<ExpertRecord> {
            self.imports
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self.record(expert))
        }
    }

    fn sources(cached_below: usize) -> Vec<BankSource> {
        // 37 experts: more than two build groups with a ragged tail.
        vec![
            BankSource::new(
                "blk.0.ffn_gate_exps.weight",
                bank(37, 4, 256, false, 71),
                37,
                cached_below,
            ),
            BankSource::new(
                "blk.0.ffn_up_exps.weight",
                bank(37, 4, 256, false, 197),
                37,
                cached_below,
            ),
            BankSource::new(
                "blk.0.ffn_down_exps.weight",
                bank(37, 8, 128, true, 13),
                37,
                cached_below,
            ),
        ]
    }

    fn dyn_sources(sources: &[BankSource]) -> Vec<&dyn ExpertSource> {
        sources.iter().map(|s| s as &dyn ExpertSource).collect()
    }

    fn section_bytes(path: &Path, s: &SectionInfo, payload: u64) -> Vec<u8> {
        let file = File::open(path).unwrap();
        let mut buf = vec![0u8; s.length as usize];
        file.read_exact_at(&mut buf, s.offset).unwrap();
        assert!(
            buf[payload as usize..].iter().all(|&b| b == 0),
            "section padding must be zero"
        );
        buf.truncate(payload as usize);
        buf
    }

    #[test]
    fn sections_equal_concatenated_expert_payloads_in_id_order_and_are_page_aligned() {
        let temp = Temp::new();
        let dir = temp.0.join(format!("{DIR_PREFIX}{IDENTITY}"));
        let banks = sources(20);
        let (pack, report) =
            MappedPack::open_or_build(&dir, IDENTITY, Some(b"path_hex=00\n"), &dyn_sources(&banks))
                .unwrap();
        assert_eq!((report.opened, report.built, report.skipped), (0, 3, 0));
        assert_eq!(pack.len(), 3);
        assert!(dir.join(".manifest").is_file());
        for source in &banks {
            assert_eq!(
                source.imports.load(std::sync::atomic::Ordering::Relaxed),
                17
            );
            assert_eq!(
                source
                    .cached_calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                37
            );
            let file = pack.file(&source.name).unwrap();
            let h = &file.header;
            assert_eq!((h.experts, h.rows_per_expert), (37, source.rows as u64));
            assert_eq!(
                (h.group, h.bits, h.mode.as_str()),
                (32, source.bits, source.mode.as_str())
            );
            assert_eq!(h.width as usize, source.width);
            assert_eq!(h.sections.len(), 3);
            let path = dir.join(file_name(&source.name));
            assert_eq!(fs::metadata(&path).unwrap().len(), h.file_bytes);
            assert_eq!(h.file_bytes % PAGE, 0);
            let mut expected: Vec<Vec<u8>> = vec![Vec::new(); 3];
            for expert in 0..37 {
                for (part, bytes) in source.record(expert).parts().unwrap().iter().enumerate() {
                    expected[part].extend_from_slice(bytes);
                }
            }
            for (s, section) in h.sections.iter().enumerate() {
                assert_eq!(
                    section.offset % PAGE,
                    0,
                    "{} section {s} offset",
                    source.name
                );
                assert_eq!(
                    section.length % PAGE,
                    0,
                    "{} section {s} length",
                    source.name
                );
                let payload = section.record_bytes * 37;
                assert_eq!(expected[s].len() as u64, payload);
                let got = section_bytes(&path, section, payload);
                assert_eq!(got, expected[s], "{} section {s} bytes", source.name);
                assert_eq!(hex(&Sha256::digest(&got)), section.sha256);
                // A range read returns the same bytes the file holds.
                let range = file.read_range(5, 9).unwrap();
                let rec = section.record_bytes as usize;
                assert_eq!(range[s], expected[s][5 * rec..14 * rec]);
            }
            // Expert reads come back in the chunk encoding, array by array.
            let chunk = file.read_expert(36).unwrap();
            let record = source.record(36);
            assert_eq!(chunk.data, record.data);
            assert_eq!(chunk.header.arrays, record.arrays);
            assert_eq!(
                (chunk.header.group, chunk.header.bits, &chunk.header.mode),
                (record.group, record.bits, &record.mode)
            );
            assert!(
                file.read_range(30, 8).is_err(),
                "range past the last expert"
            );
            file.verify_sections().unwrap();
        }
        // Mutation: one flipped payload byte fails verification and is
        // rebuilt byte-identical at the next open.
        let name = &banks[1].name;
        let path = dir.join(file_name(name));
        let original = fs::read(&path).unwrap();
        let section = &pack.file(name).unwrap().header.sections[1];
        let mut flipped = original.clone();
        flipped[section.offset as usize + 7] ^= 0x40;
        drop(pack);
        fs::write(&path, &flipped).unwrap();
        let dirfd = packed_cache::open_dir_nofollow(&dir).unwrap();
        let reopened = LayerFile::open(&dirfd, name, IDENTITY, false)
            .unwrap()
            .unwrap();
        let error = reopened.verify_sections().unwrap_err().reason;
        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(LayerFile::open(&dirfd, name, IDENTITY, true).is_err());
        drop(reopened);
        drop(dirfd);
        let (_, report) =
            MappedPack::open_or_build_verifying(&dir, IDENTITY, None, &dyn_sources(&banks))
                .unwrap();
        assert_eq!((report.opened, report.built), (2, 1));
        assert_eq!(fs::read(&path).unwrap(), original);
        // A header that names another checkpoint or tensor is foreign.
        let other = Temp::new();
        let other_dir = other.0.join(format!("{DIR_PREFIX}{IDENTITY}"));
        fs::create_dir_all(&other_dir).unwrap();
        fs::copy(
            &path,
            other_dir.join(file_name("blk.0.ffn_gate_exps.weight")),
        )
        .unwrap();
        fs::copy(&path, other_dir.join(file_name(name))).unwrap();
        let other_fd = packed_cache::open_dir_nofollow(&other_dir).unwrap();
        let foreign = |tensor: &str, identity: &str| {
            LayerFile::open(&other_fd, tensor, identity, false)
                .err()
                .map(|e| e.reason)
        };
        assert!(
            foreign("blk.0.ffn_gate_exps.weight", IDENTITY)
                .is_some_and(|r| r.contains("another tensor or checkpoint")),
            "a file whose header names another tensor"
        );
        assert!(
            foreign(name, &IDENTITY.replace('0', "1"))
                .is_some_and(|r| r.contains("another tensor or checkpoint")),
            "a file whose header names another checkpoint"
        );
        assert!(
            LayerFile::open(&other_fd, name, IDENTITY, false)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn builder_resumes_from_a_partial_directory_and_rebuilds_truncated_files() {
        let temp = Temp::new();
        let complete = temp.0.join(format!("a/{DIR_PREFIX}{IDENTITY}"));
        let banks = sources(37);
        let (_, report) =
            MappedPack::open_or_build(&complete, IDENTITY, None, &dyn_sources(&banks)).unwrap();
        assert_eq!((report.opened, report.built), (0, 3));
        for source in &banks {
            // Every expert came from the fast path; nothing was imported.
            assert_eq!(source.imports.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
        let partial = temp.0.join(format!("b/{DIR_PREFIX}{IDENTITY}"));
        fs::create_dir_all(&partial).unwrap();
        let [gate, up, down] = ["gate", "up", "down"].map(|p| format!("blk.0.ffn_{p}_exps.weight"));
        fs::copy(
            complete.join(file_name(&gate)),
            partial.join(file_name(&gate)),
        )
        .unwrap();
        let truncated = fs::read(complete.join(file_name(&up))).unwrap();
        fs::write(
            partial.join(file_name(&up)),
            &truncated[..truncated.len() - 1000],
        )
        .unwrap();
        fs::write(partial.join(tmp_name(&down)), b"half-written staging file").unwrap();
        let stale = partial.join(tmp_name(&down));
        let (pack, report) =
            MappedPack::open_or_build(&partial, IDENTITY, None, &dyn_sources(&banks)).unwrap();
        assert_eq!((report.opened, report.built, report.skipped), (1, 2, 0));
        assert_eq!(pack.len(), 3);
        assert!(
            !stale.exists(),
            "the stale staging file is replaced by the final file"
        );
        for name in [&gate, &up, &down] {
            assert_eq!(
                fs::read(partial.join(file_name(name))).unwrap(),
                fs::read(complete.join(file_name(name))).unwrap(),
                "{name} must be byte-identical after the resumed build"
            );
        }
        // A staging file another builder holds locked is left alone.
        let held = packed_cache::dir_open_at(
            &packed_cache::open_dir_nofollow(&partial).unwrap(),
            OsStr::new(&tmp_name(&gate)),
            libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC,
            0o644,
        )
        .unwrap();
        assert_eq!(
            unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        fs::remove_file(partial.join(file_name(&gate))).unwrap();
        let (pack, report) =
            MappedPack::open_or_build(&partial, IDENTITY, None, &dyn_sources(&banks)).unwrap();
        assert_eq!((report.opened, report.built, report.skipped), (2, 0, 1));
        assert!(pack.file(&gate).is_none());
        assert!(
            pack.read_batch(&[(gate.clone(), 0, 4), (up.clone(), 0, 4)])
                .unwrap()
                .is_none()
        );
        drop(held);
    }

    #[test]
    fn read_batch_serves_whole_experts_only() {
        let temp = Temp::new();
        let dir = temp.0.join(format!("{DIR_PREFIX}{IDENTITY}"));
        let banks = sources(37);
        let (pack, _) =
            MappedPack::open_or_build(&dir, IDENTITY, None, &dyn_sources(&banks)).unwrap();
        let requests = vec![
            ("blk.0.ffn_gate_exps.weight".to_owned(), 3 * 4, 4),
            ("blk.0.ffn_up_exps.weight".to_owned(), 3 * 4, 4),
            ("blk.0.ffn_down_exps.weight".to_owned(), 3 * 8, 8),
            ("blk.0.ffn_gate_exps.weight".to_owned(), 36 * 4, 4),
        ];
        let chunks = pack.read_batch(&requests).unwrap().unwrap();
        assert_eq!(chunks.len(), 4);
        for (chunk, (source, expert)) in chunks.iter().zip([
            (&banks[0], 3),
            (&banks[1], 3),
            (&banks[2], 3),
            (&banks[0], 36),
        ]) {
            let record = source.record(expert);
            assert_eq!(chunk.data, record.data);
            assert_eq!(chunk.header.arrays, record.arrays);
        }
        // A partial expert, an unknown tensor and an unaligned start fall back.
        assert!(
            pack.read_batch(&[("blk.0.ffn_gate_exps.weight".into(), 0, 2)])
                .unwrap()
                .is_none()
        );
        assert!(
            pack.read_batch(&[("blk.1.ffn_gate_exps.weight".into(), 0, 4)])
                .unwrap()
                .is_none()
        );
        assert!(
            pack.read_batch(&[("blk.0.ffn_gate_exps.weight".into(), 2, 4)])
                .unwrap()
                .is_none()
        );
        assert!(
            pack.read_batch(&[("blk.0.ffn_gate_exps.weight".into(), 37 * 4, 4)])
                .is_err()
        );
    }
}
