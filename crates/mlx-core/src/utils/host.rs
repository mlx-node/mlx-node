#[cfg(target_os = "macos")]
pub(crate) fn physical_memory() -> Option<u64> {
    let mut bytes = 0u64;
    let mut size = std::mem::size_of_val(&bytes);
    // SAFETY: `hw.memsize` is a u64 and `size` holds its byte length.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && bytes > 0).then_some(bytes)
}

#[cfg(target_os = "linux")]
fn raw_physical_memory() -> Option<u64> {
    // SAFETY: sysconf only reads system configuration.
    let (pages, page) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    if pages <= 0 || page <= 0 {
        return None;
    }
    (pages as u64).checked_mul(page as u64)
}

#[cfg(target_os = "linux")]
pub(crate) fn physical_memory() -> Option<u64> {
    let raw = raw_physical_memory()?;
    Some(read_linux_cgroup_v2_memory().map_or(raw, |cg| raw.min(cg.limit)))
}

#[cfg(target_os = "windows")]
pub(crate) fn physical_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    // SAFETY: `dwLength` is initialized to `size_of::<MEMORYSTATUSEX>()`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn physical_memory() -> Option<u64> {
    None
}

/// One host memory snapshot, in bytes (family-neutral; callers pass the
/// value on, so tests build it by hand).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMemory {
    pub physical: u64,
    pub free: u64,
    /// Pageable file-backed pages the kernel can drop on demand.
    pub file: u64,
    /// Pageable anonymous pages of every process.
    pub anonymous: u64,
    pub purgeable: u64,
    pub wired: u64,
    /// Physical pages the compressor pool occupies.
    pub compressor: u64,
}

impl HostMemory {
    /// Free plus clean file cache: what the kernel hands out without
    /// touching another process's data.
    pub fn available(&self) -> u64 {
        self.free.saturating_add(self.file)
    }

    /// `host_used` of the guard: wired + compressor + anonymous - purgeable.
    pub fn non_reclaimable(&self) -> u64 {
        self.wired
            .saturating_add(self.compressor)
            .saturating_add(self.anonymous.saturating_sub(self.purgeable))
    }
}

/// Raw Mach VM page counts and page size from one `host_statistics64` read.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MacosVmStats {
    pub page_size: u64,
    pub free: u64,
    pub speculative: u64,
    pub file_backed: u64,
    pub purgeable: u64,
    pub anonymous: u64,
    pub wired: u64,
    pub compressor: u64,
    pub compressed: u64,
}

/// Sample raw Mach VM page counts and host page size via `host_statistics64`.
#[cfg(target_os = "macos")]
pub(crate) fn sample_macos_vm_stats() -> Option<MacosVmStats> {
    unsafe extern "C" {
        fn mach_host_self() -> libc::mach_port_t;
        static mach_task_self_: libc::mach_port_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
        fn host_page_size(
            host: libc::mach_port_t,
            size: *mut libc::vm_size_t,
        ) -> libc::kern_return_t;
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let mut page_size: libc::vm_size_t = 0;
    // SAFETY: plain Mach calls on this task's host port; `stats` holds
    // `count` integers and the port is released after the read.
    let (ok, stats) = unsafe {
        let host = mach_host_self();
        let page_ok = host_page_size(host, &mut page_size);
        let ok = if page_ok == 0 {
            libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast(),
                &mut count,
            )
        } else {
            page_ok
        };
        mach_port_deallocate(mach_task_self_, host);
        (ok, stats.assume_init())
    };
    if ok != 0 || page_size == 0 {
        return None;
    }
    Some(MacosVmStats {
        page_size: page_size as u64,
        free: u64::from(stats.free_count),
        speculative: u64::from(stats.speculative_count),
        file_backed: u64::from(stats.external_page_count),
        purgeable: u64::from(stats.purgeable_count),
        anonymous: u64::from(stats.internal_page_count),
        wired: u64::from(stats.wire_count),
        compressor: u64::from(stats.compressor_page_count),
        compressed: stats.total_uncompressed_pages_in_compressor,
    })
}

#[cfg(target_os = "macos")]
pub fn host_memory() -> Option<HostMemory> {
    let stats = sample_macos_vm_stats()?;
    let page = stats.page_size;
    Some(HostMemory {
        physical: physical_memory()?,
        free: stats.free * page,
        file: stats.file_backed * page,
        anonymous: stats.anonymous * page,
        purgeable: stats.purgeable * page,
        wired: stats.wired * page,
        compressor: stats.compressor * page,
    })
}

#[cfg(target_os = "windows")]
pub fn host_memory() -> Option<HostMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is a plain Win32 POD struct; zeroed with
    // `dwLength` initialized before calling `GlobalMemoryStatusEx`.
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.ullTotalPhys == 0 {
        return None;
    }
    let physical = status.ullTotalPhys;
    // Windows `ullAvailPhys` includes free and standby (reclaimable) pages,
    // matching macOS's `free + file`.
    let free = status.ullAvailPhys.min(physical);
    let non_reclaimable = physical.saturating_sub(free);
    Some(HostMemory {
        physical,
        free,
        file: 0,
        anonymous: non_reclaimable,
        purgeable: 0,
        wired: 0,
        compressor: 0,
    })
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_meminfo(text: &str, physical: u64) -> Option<HostMemory> {
    if physical == 0 {
        return None;
    }
    let mut mem_free: Option<u64> = None;
    let mut mem_available: Option<u64> = None;
    let mut buffers = 0u64;
    let mut cached = 0u64;
    let mut sreclaimable = 0u64;

    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let Some(value_str) = parts.next() else {
            continue;
        };
        let Ok(kib) = value_str.parse::<u64>() else {
            continue;
        };
        if let Some(unit) = parts.next()
            && !unit.eq_ignore_ascii_case("kB")
        {
            continue;
        }
        let bytes = kib.saturating_mul(1024);
        match key.trim() {
            "MemFree" => mem_free = Some(bytes),
            "MemAvailable" => mem_available = Some(bytes),
            "Buffers" => buffers = bytes,
            "Cached" => cached = bytes,
            "SReclaimable" => sreclaimable = bytes,
            _ => {}
        }
    }

    let mem_free = mem_free?;
    let raw_available = mem_available.unwrap_or_else(|| {
        mem_free
            .saturating_add(buffers)
            .saturating_add(cached)
            .saturating_add(sreclaimable)
    });
    let available = raw_available.min(physical);
    let free = mem_free.min(available);
    let file = available.saturating_sub(free);
    let anonymous = physical.saturating_sub(available);
    Some(HostMemory {
        physical,
        free,
        file,
        anonymous,
        purgeable: 0,
        wired: 0,
        compressor: 0,
    })
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinuxCgroupMemory {
    /// Tightest finite `memory.max` across the process's cgroup v2 and its ancestors.
    limit: u64,
    /// Tightest available headroom (`limit - (current - inactive_file)`) across bounded ancestors.
    available: u64,
}

#[cfg(any(target_os = "linux", test))]
fn normalize_cgroup_rel_path(path: &str) -> Option<&str> {
    let rel = path.strip_prefix('/')?.trim_matches('/');
    if rel.split('/').any(|seg| seg == "." || seg == "..") {
        return None;
    }
    Some(rel)
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_v2_rel_path(proc_self_cgroup: &str) -> Option<&str> {
    for line in proc_self_cgroup.lines() {
        if let Some(path) = line.strip_prefix("0::") {
            return normalize_cgroup_rel_path(path);
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn unescape_mountinfo_field(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && (b'0'..=b'3').contains(&bytes[i + 1])
            && (b'0'..=b'7').contains(&bytes[i + 2])
            && (b'0'..=b'7').contains(&bytes[i + 3])
        {
            let val =
                ((bytes[i + 1] - b'0') << 6) | ((bytes[i + 2] - b'0') << 3) | (bytes[i + 3] - b'0');
            out.push(val);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

#[cfg(any(target_os = "linux", test))]
fn resolve_cgroup_v2_mounts(
    proc_self_cgroup: &str,
    proc_self_mountinfo: &str,
) -> Vec<(std::path::PathBuf, std::path::PathBuf)> {
    let Some(rel) = parse_cgroup_v2_rel_path(proc_self_cgroup) else {
        return Vec::new();
    };
    let mut matches: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();

    for line in proc_self_mountinfo.lines() {
        let Some((pre, post)) = line.split_once(" - ") else {
            continue;
        };
        if post.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let mut pre_fields = pre.split_whitespace();
        let (
            Some(_mount_id),
            Some(_parent_id),
            Some(_major_minor),
            Some(raw_root),
            Some(raw_mount_point),
        ) = (
            pre_fields.next(),
            pre_fields.next(),
            pre_fields.next(),
            pre_fields.next(),
            pre_fields.next(),
        )
        else {
            continue;
        };

        let root_unescaped = unescape_mountinfo_field(raw_root);
        let Some(mount_root) = normalize_cgroup_rel_path(&root_unescaped) else {
            continue;
        };
        let mount_point = std::path::PathBuf::from(unescape_mountinfo_field(raw_mount_point));
        if !mount_point.has_root() {
            continue;
        }

        let leaf_rel = if mount_root.is_empty() {
            rel
        } else if rel == mount_root {
            ""
        } else if let Some(rest) = rel
            .strip_prefix(mount_root)
            .and_then(|rest| rest.strip_prefix('/'))
        {
            rest.trim_start_matches('/')
        } else {
            continue;
        };

        let start = if leaf_rel.is_empty() {
            mount_point.clone()
        } else {
            mount_point.join(leaf_rel)
        };
        let pair = (start, mount_point);
        if !matches.contains(&pair) {
            matches.push(pair);
        }
    }

    matches
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_memory_max(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok().filter(|&bytes| bytes > 0)
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_memory_stat_reclaimable(text: &str) -> u64 {
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("inactive_file")
            && let Some(val) = parts.next()
            && parts.next().is_none()
            && let Ok(bytes) = val.parse::<u64>()
        {
            return bytes;
        }
    }
    0
}

#[cfg(any(target_os = "linux", test))]
fn combine_cgroup_ancestors(levels: &[(Option<u64>, u64, u64)]) -> Option<LinuxCgroupMemory> {
    let mut min_limit = u64::MAX;
    let mut min_avail = u64::MAX;
    let mut found = false;

    for &(memory_max, memory_current, inactive_file) in levels {
        let Some(limit) = memory_max.filter(|&bytes| bytes > 0) else {
            continue;
        };
        let non_reclaimable = memory_current.saturating_sub(inactive_file);
        let avail = limit.saturating_sub(non_reclaimable);
        min_limit = min_limit.min(limit);
        min_avail = min_avail.min(avail);
        found = true;
    }

    found.then_some(LinuxCgroupMemory {
        limit: min_limit,
        available: min_avail.min(min_limit),
    })
}

#[cfg(any(target_os = "linux", test))]
fn apply_cgroup_memory_limit(host: HostMemory, cgroup: Option<LinuxCgroupMemory>) -> HostMemory {
    let Some(cg) = cgroup else {
        return host;
    };
    let physical = host.physical.min(cg.limit);
    let available = host.available().min(cg.available).min(physical);
    let free = host.free.min(available);
    let file = available.saturating_sub(free);
    let anonymous = physical.saturating_sub(available);
    HostMemory {
        physical,
        free,
        file,
        anonymous,
        purgeable: 0,
        wired: 0,
        compressor: 0,
    }
}

#[cfg(any(target_os = "linux", test))]
fn read_cgroup_v2_hierarchy(
    start: &std::path::Path,
    root: &std::path::Path,
) -> Option<LinuxCgroupMemory> {
    let mut levels = Vec::new();
    let mut current_dir = Some(start);
    while let Some(dir) = current_dir {
        if !dir.starts_with(root) {
            break;
        }
        if let Some(limit) = std::fs::read_to_string(dir.join("memory.max"))
            .ok()
            .and_then(|s| parse_cgroup_memory_max(&s))
        {
            let (current, inactive_file) = match std::fs::read_to_string(dir.join("memory.current"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
            {
                Some(current) => {
                    let inactive_file = std::fs::read_to_string(dir.join("memory.stat"))
                        .ok()
                        .map_or(0, |s| parse_cgroup_memory_stat_reclaimable(&s));
                    (current, inactive_file)
                }
                None => (limit, 0),
            };
            levels.push((Some(limit), current, inactive_file));
        }
        if dir == root {
            break;
        }
        current_dir = dir.parent();
    }
    combine_cgroup_ancestors(&levels)
}

#[cfg(any(target_os = "linux", test))]
fn read_cgroup_v2_mounts(
    mounts: &[(std::path::PathBuf, std::path::PathBuf)],
) -> Option<LinuxCgroupMemory> {
    let mut min_limit: Option<u64> = None;
    let mut min_available: Option<u64> = None;
    for (start, root) in mounts {
        if let Some(cg) = read_cgroup_v2_hierarchy(start, root) {
            min_limit = Some(min_limit.map_or(cg.limit, |acc| acc.min(cg.limit)));
            min_available = Some(min_available.map_or(cg.available, |acc| acc.min(cg.available)));
        }
    }
    let limit = min_limit?;
    let available = min_available.unwrap_or(limit).min(limit);
    Some(LinuxCgroupMemory { limit, available })
}

#[cfg(target_os = "linux")]
fn read_linux_cgroup_v2_memory() -> Option<LinuxCgroupMemory> {
    let cgroup_text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut mounts = std::fs::read_to_string("/proc/self/mountinfo")
        .ok()
        .map(|mountinfo| resolve_cgroup_v2_mounts(&cgroup_text, &mountinfo))
        .unwrap_or_default();
    if mounts.is_empty() {
        let rel = parse_cgroup_v2_rel_path(&cgroup_text)?;
        let root = std::path::PathBuf::from("/sys/fs/cgroup");
        let start = if rel.is_empty() {
            root.clone()
        } else {
            root.join(rel)
        };
        mounts.push((start, root));
    }
    read_cgroup_v2_mounts(&mounts)
}

#[cfg(target_os = "linux")]
pub fn host_memory() -> Option<HostMemory> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let host = parse_linux_meminfo(&text, raw_physical_memory()?)?;
    Some(apply_cgroup_memory_limit(
        host,
        read_linux_cgroup_v2_memory(),
    ))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn host_memory() -> Option<HostMemory> {
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_memory_is_known_on_supported_hosts() {
        let bytes = physical_memory();
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            assert!(bytes.is_some_and(|b| b >= 1 << 30), "{bytes:?}");
        }
    }

    #[test]
    fn host_memory_fields_sane() {
        let Some(h) = host_memory() else {
            if cfg!(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows"
            )) {
                panic!("host_memory failed on supported host");
            }
            return;
        };
        println!("{h:?} non_reclaimable {}", h.non_reclaimable());
        assert_eq!(Some(h.physical), physical_memory());
        assert!(
            (!cfg!(target_os = "macos") || h.wired > 0) && h.anonymous > 0 && h.available() > 0,
            "{h:?}"
        );
        let parts = h.free + h.file + h.anonymous + h.wired + h.compressor;
        assert!(parts <= h.physical + (h.physical >> 3), "{h:?}");
        assert!(h.non_reclaimable() < h.physical, "{h:?}");
    }

    #[test]
    fn parse_linux_meminfo_normal() {
        let physical = 128u64 << 30;
        let sample = "\
MemTotal:       134217728 kB
MemFree:         40000000 kB
MemAvailable:    90000000 kB
Buffers:          5000000 kB
Cached:          40000000 kB
SReclaimable:     5000000 kB
";
        let h = parse_linux_meminfo(sample, physical).expect("valid meminfo");
        assert_eq!(h.physical, physical);
        assert_eq!(h.free, 40_000_000 * 1024);
        assert_eq!(h.file, 50_000_000 * 1024);
        assert_eq!(h.available(), 90_000_000 * 1024);
        assert_eq!(h.anonymous, physical - 90_000_000 * 1024);
        assert_eq!(h.non_reclaimable(), physical - 90_000_000 * 1024);
        assert_eq!(h.purgeable, 0);
        assert_eq!(h.wired, 0);
        assert_eq!(h.compressor, 0);
    }

    #[test]
    fn parse_linux_meminfo_fallback_without_mem_available() {
        let physical = 64u64 << 30;
        let sample = "\
MemTotal:       67108864 kB
MemFree:        10000000 kB
Buffers:         1000000 kB
Cached:         20000000
SReclaimable:    3000000 kB
";
        let h = parse_linux_meminfo(sample, physical).expect("fallback meminfo");
        let expected_available = (10_000_000 + 1_000_000 + 20_000_000 + 3_000_000) * 1024;
        assert_eq!(h.free, 10_000_000 * 1024);
        assert_eq!(h.file, (1_000_000 + 20_000_000 + 3_000_000) * 1024);
        assert_eq!(h.available(), expected_available);
        assert_eq!(h.anonymous, physical - expected_available);
    }

    #[test]
    fn parse_linux_meminfo_clamps_to_physical() {
        let physical = 16u64 << 30;
        let sample = "\
MemFree:        10000000 kB
MemAvailable:   99999999 kB
";
        let h = parse_linux_meminfo(sample, physical).expect("clamped meminfo");
        assert_eq!(h.available(), physical);
        assert_eq!(h.free, 10_000_000 * 1024);
        assert_eq!(h.file, physical - 10_000_000 * 1024);
        assert_eq!(h.anonymous, 0);

        let free_gt_avail = "\
MemFree:        12000000 kB
MemAvailable:   10000000 kB
";
        let h2 = parse_linux_meminfo(free_gt_avail, physical).expect("clamped free");
        assert_eq!(h2.available(), 10_000_000 * 1024);
        assert_eq!(h2.free, 10_000_000 * 1024);
        assert_eq!(h2.file, 0);
        assert_eq!(h2.anonymous, physical - 10_000_000 * 1024);
    }

    #[test]
    fn parse_linux_meminfo_missing_mem_free_returns_none() {
        let physical = 16u64 << 30;
        let no_free = "\
MemTotal:       16777216 kB
MemAvailable:    8000000 kB
";
        assert_eq!(parse_linux_meminfo(no_free, physical), None);
        assert_eq!(parse_linux_meminfo("MemFree: 1000 kB\n", 0), None);
    }

    #[test]
    fn parse_cgroup_v2_rel_path_cases() {
        assert_eq!(parse_cgroup_v2_rel_path("0::/\n"), Some(""));
        assert_eq!(
            parse_cgroup_v2_rel_path("0::/docker/123\n"),
            Some("docker/123")
        );
        assert_eq!(
            parse_cgroup_v2_rel_path("0::/docker/123/\n"),
            Some("docker/123")
        );
        assert_eq!(
            parse_cgroup_v2_rel_path("0::/system.slice/foo:bar.service\n"),
            Some("system.slice/foo:bar.service")
        );
        let hybrid = "\
12:memory:/kubepods/pod123
11:cpu,cpuacct:/kubepods/pod123
0::/kubepods/pod123
";
        assert_eq!(parse_cgroup_v2_rel_path(hybrid), Some("kubepods/pod123"));
        let v1_only = "\
12:memory:/docker/123
11:cpu,cpuacct:/docker/123
";
        assert_eq!(parse_cgroup_v2_rel_path(v1_only), None);
        assert_eq!(parse_cgroup_v2_rel_path(""), None);
        assert_eq!(parse_cgroup_v2_rel_path("0::\n"), None);
        assert_eq!(parse_cgroup_v2_rel_path("0::docker/123\n"), None);
        assert_eq!(parse_cgroup_v2_rel_path("0::/../escape\n"), None);
        assert_eq!(parse_cgroup_v2_rel_path("0::/docker/../host\n"), None);
        assert_eq!(parse_cgroup_v2_rel_path("0::/./docker\n"), None);
    }

    #[test]
    fn parse_cgroup_memory_max_and_stat_cases() {
        assert_eq!(parse_cgroup_memory_max("max\n"), None);
        assert_eq!(parse_cgroup_memory_max("0\n"), None);
        assert_eq!(parse_cgroup_memory_max(""), None);
        assert_eq!(parse_cgroup_memory_max("not-a-number\n"), None);
        assert_eq!(parse_cgroup_memory_max("34359738368\n"), Some(32u64 << 30));

        let stat = "\
anon 3964928
file 13481406464
kernel 297738240
slab_reclaimable 536870912
inactive_file 2147483648
active_file 5383471104
";
        assert_eq!(parse_cgroup_memory_stat_reclaimable(stat), 2u64 << 30);
        assert_eq!(parse_cgroup_memory_stat_reclaimable("anon 123\n"), 0);
    }

    #[test]
    fn combine_cgroup_ancestors_cases() {
        const GIB: u64 = 1u64 << 30;
        assert_eq!(combine_cgroup_ancestors(&[]), None);
        assert_eq!(
            combine_cgroup_ancestors(&[(None, 10 * GIB, GIB), (None, 20 * GIB, 2 * GIB)]),
            None
        );
        assert_eq!(combine_cgroup_ancestors(&[(Some(0), GIB, 0)]), None);

        // Single bounded cgroup.
        assert_eq!(
            combine_cgroup_ancestors(&[(Some(32 * GIB), 28 * GIB, 2 * GIB)]),
            Some(LinuxCgroupMemory {
                limit: 32 * GIB,
                available: 6 * GIB,
            })
        );

        // Nested ancestor tighter than child.
        assert_eq!(
            combine_cgroup_ancestors(&[
                (Some(64 * GIB), 10 * GIB, 2 * GIB),
                (Some(32 * GIB), 20 * GIB, 4 * GIB),
            ]),
            Some(LinuxCgroupMemory {
                limit: 32 * GIB,
                available: 16 * GIB,
            })
        );

        // Unbounded child inside bounded ancestor.
        assert_eq!(
            combine_cgroup_ancestors(&[(None, 4 * GIB, GIB), (Some(16 * GIB), 14 * GIB, 2 * GIB)]),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 4 * GIB,
            })
        );

        // Child more pressured than parent.
        assert_eq!(
            combine_cgroup_ancestors(&[
                (Some(16 * GIB), 15 * GIB, GIB),
                (Some(64 * GIB), 30 * GIB, 5 * GIB),
            ]),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 2 * GIB,
            })
        );

        // Saturating arithmetic on transiently over-limit or noisy readings.
        assert_eq!(
            combine_cgroup_ancestors(&[(Some(16 * GIB), 18 * GIB, 0)]),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 0,
            })
        );
        assert_eq!(
            combine_cgroup_ancestors(&[(Some(16 * GIB), 4 * GIB, 6 * GIB)]),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 16 * GIB,
            })
        );
    }

    #[test]
    fn apply_cgroup_memory_limit_bounds_large_host_snapshot() {
        const MIB: u64 = 1u64 << 20;
        const GIB: u64 = 1u64 << 30;
        let host = HostMemory {
            physical: 128 * GIB,
            free: 100 * GIB,
            file: 0,
            anonymous: 28 * GIB,
            purgeable: 0,
            wired: 0,
            compressor: 0,
        };

        // Unbounded leaves host snapshot untouched.
        assert_eq!(apply_cgroup_memory_limit(host, None), host);

        // 128 GiB host with 100 GiB free inside a 32 GiB cgroup with 28 GiB current
        // and 2 GiB inactive_file => physical = 32 GiB, available = 6 GiB, non_reclaimable = 26 GiB.
        let cg = combine_cgroup_ancestors(&[(Some(32 * GIB), 28 * GIB, 2 * GIB)]);
        let bounded = apply_cgroup_memory_limit(host, cg);
        assert_eq!(bounded.physical, 32 * GIB);
        assert_eq!(bounded.available(), 6 * GIB);
        assert_eq!(bounded.free, 6 * GIB);
        assert_eq!(bounded.file, 0);
        assert_eq!(bounded.anonymous, 26 * GIB);
        assert_eq!(bounded.non_reclaimable(), 26 * GIB);
        assert_eq!(bounded.purgeable, 0);
        assert_eq!(bounded.wired, 0);
        assert_eq!(bounded.compressor, 0);
        assert_eq!(
            bounded.free + bounded.file + bounded.anonymous + bounded.wired + bounded.compressor,
            bounded.physical
        );

        // Preserves free/file split when host.free < bounded available.
        let cached_host = HostMemory {
            physical: 128 * GIB,
            free: 2 * GIB,
            file: 98 * GIB,
            anonymous: 28 * GIB,
            purgeable: 0,
            wired: 0,
            compressor: 0,
        };
        let bounded_cached = apply_cgroup_memory_limit(cached_host, cg);
        assert_eq!(bounded_cached.physical, 32 * GIB);
        assert_eq!(bounded_cached.free, 2 * GIB);
        assert_eq!(bounded_cached.file, 4 * GIB);
        assert_eq!(bounded_cached.available(), 6 * GIB);
        assert_eq!(bounded_cached.non_reclaimable(), 26 * GIB);

        // Nearly exhausted cgroup drops available accordingly.
        let tight_cg = combine_cgroup_ancestors(&[(Some(32 * GIB), 32 * GIB - 512 * MIB, 0)]);
        let tight = apply_cgroup_memory_limit(host, tight_cg);
        assert_eq!(tight.physical, 32 * GIB);
        assert_eq!(tight.available(), 512 * MIB);
        assert_eq!(tight.non_reclaimable(), 32 * GIB - 512 * MIB);

        // Host pressure still bounds available when host has less headroom than cgroup.
        let pressured_host = HostMemory {
            physical: 128 * GIB,
            free: GIB,
            file: 3 * GIB,
            anonymous: 124 * GIB,
            purgeable: 0,
            wired: 0,
            compressor: 0,
        };
        let host_bound = apply_cgroup_memory_limit(pressured_host, cg);
        assert_eq!(host_bound.physical, 32 * GIB);
        assert_eq!(host_bound.available(), 4 * GIB);
        assert_eq!(host_bound.free, GIB);
        assert_eq!(host_bound.file, 3 * GIB);
        assert_eq!(host_bound.non_reclaimable(), 28 * GIB);

        // Cgroup limit larger than host physical never inflates physical.
        let loose_cg = combine_cgroup_ancestors(&[(Some(256 * GIB), 10 * GIB, 0)]);
        let loose = apply_cgroup_memory_limit(host, loose_cg);
        assert_eq!(loose.physical, 128 * GIB);
        assert_eq!(loose.available(), 100 * GIB);
    }

    #[test]
    fn read_cgroup_v2_hierarchy_walks_ancestors() {
        const GIB: u64 = 1u64 << 30;
        struct TempDir(std::path::PathBuf);
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let root =
            std::env::temp_dir().join(format!("mlx-core-cgroup-v2-test-{}", std::process::id()));
        let _guard = TempDir(root.clone());
        let parent = root.join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).expect("create test cgroup dirs");

        // Unbounded child + unbounded parent + root without memory.max => None.
        std::fs::write(parent.join("memory.max"), "max\n").expect("write parent max");
        std::fs::write(child.join("memory.max"), "max\n").expect("write child max");
        assert_eq!(read_cgroup_v2_hierarchy(&child, &root), None);

        // Parent bounded, child unbounded => parent limit applies.
        std::fs::write(parent.join("memory.max"), format!("{}\n", 32 * GIB))
            .expect("write parent limit");
        std::fs::write(parent.join("memory.current"), format!("{}\n", 20 * GIB))
            .expect("write parent current");
        std::fs::write(
            parent.join("memory.stat"),
            format!("anon 1024\ninactive_file {}\n", 4 * GIB),
        )
        .expect("write parent stat");
        assert_eq!(
            read_cgroup_v2_hierarchy(&child, &root),
            Some(LinuxCgroupMemory {
                limit: 32 * GIB,
                available: 16 * GIB,
            })
        );

        // Child has looser limit (64 GiB) but tighter headroom (6 GiB) than parent (32 GiB / 16 GiB).
        std::fs::write(child.join("memory.max"), format!("{}\n", 64 * GIB))
            .expect("write child limit");
        std::fs::write(child.join("memory.current"), format!("{}\n", 60 * GIB))
            .expect("write child current");
        std::fs::write(
            child.join("memory.stat"),
            format!("inactive_file {}\n", 2 * GIB),
        )
        .expect("write child stat");
        assert_eq!(
            read_cgroup_v2_hierarchy(&child, &root),
            Some(LinuxCgroupMemory {
                limit: 32 * GIB,
                available: 6 * GIB,
            })
        );

        // Container-namespaced root with finite memory.max directly at root.
        std::fs::write(root.join("memory.max"), format!("{}\n", 16 * GIB))
            .expect("write root limit");
        std::fs::write(root.join("memory.current"), format!("{}\n", 12 * GIB))
            .expect("write root current");
        assert_eq!(
            read_cgroup_v2_hierarchy(&root, &root),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 4 * GIB,
            })
        );
    }

    #[test]
    fn resolve_cgroup_v2_mount_cases() {
        use std::path::PathBuf;

        // Standard root `/` mount at `/sys/fs/cgroup` (ignoring v1 `cgroup` mounts).
        let standard_mountinfo = "\
28 22 0:25 / /sys/fs/cgroup/memory rw,nosuid - cgroup cgroup rw,memory
29 22 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate
";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/\n", standard_mountinfo),
            vec![(
                PathBuf::from("/sys/fs/cgroup"),
                PathBuf::from("/sys/fs/cgroup"),
            )]
        );
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/docker/123\n", standard_mountinfo),
            vec![(
                PathBuf::from("/sys/fs/cgroup/docker/123"),
                PathBuf::from("/sys/fs/cgroup"),
            )]
        );

        // Subtree bind-mount `/batch` mounted at `/sys/fs/cgroup`:
        // process in `0::/batch/model` resolves to `/sys/fs/cgroup/model`, NOT `/sys/fs/cgroup/batch/model`.
        let subtree_mountinfo = "\
30 22 0:26 /other /mnt/other rw shared:5 - cgroup2 cgroup2 rw
31 22 0:26 /batch /sys/fs/cgroup rw,nosuid shared:6 - cgroup2 cgroup2 rw,nsdelegate
";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/batch/model\n", subtree_mountinfo),
            vec![(
                PathBuf::from("/sys/fs/cgroup/model"),
                PathBuf::from("/sys/fs/cgroup"),
            )]
        );
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/batch\n", subtree_mountinfo),
            vec![(
                PathBuf::from("/sys/fs/cgroup"),
                PathBuf::from("/sys/fs/cgroup"),
            )]
        );

        // Non-matching prefix `/other` and non-segment prefix `/batch` vs `/batch2` are ignored.
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/batch2/model\n", subtree_mountinfo),
            Vec::<(PathBuf, PathBuf)>::new()
        );
        let non_matching = "30 22 0:26 /other /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/batch/model\n", non_matching),
            Vec::<(PathBuf, PathBuf)>::new()
        );

        // Layered mounts return ALL matching views so tighter ancestor caps
        // visible only through a broader mount are never missed.
        let layered_mountinfo = "\
29 22 0:26 / /host/cgroup rw - cgroup2 cgroup2 rw
31 22 0:26 /batch /sys/fs/cgroup rw - cgroup2 cgroup2 rw
";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/batch/model\n", layered_mountinfo),
            vec![
                (
                    PathBuf::from("/host/cgroup/batch/model"),
                    PathBuf::from("/host/cgroup"),
                ),
                (
                    PathBuf::from("/sys/fs/cgroup/model"),
                    PathBuf::from("/sys/fs/cgroup"),
                ),
            ]
        );

        // Custom mount point `/mnt/cg2` (with no optional fields before ` - `).
        let custom_mountinfo = "40 22 0:26 / /mnt/cg2 rw,relatime - cgroup2 cgroup2 rw\n";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/app/worker\n", custom_mountinfo),
            vec![(
                PathBuf::from("/mnt/cg2/app/worker"),
                PathBuf::from("/mnt/cg2")
            )]
        );

        // Octal-escaped spaces (`\040`), tabs (`\011`), newlines (`\012`), and backslashes (`\134`).
        let escaped_mountinfo =
            "41 22 0:26 /my\\040batch /mnt/my\\040cg2 rw - cgroup2 cgroup2 rw\n";
        assert_eq!(
            resolve_cgroup_v2_mounts("0::/my batch/model\n", escaped_mountinfo),
            vec![(
                PathBuf::from("/mnt/my cg2/model"),
                PathBuf::from("/mnt/my cg2"),
            )]
        );
        assert_eq!(
            unescape_mountinfo_field("/a\\040b\\011c\\012d\\134e\\134040"),
            "/a b\tc\nd\\e\\040"
        );
    }

    #[test]
    fn read_cgroup_v2_subtree_mount_enforces_tighter_leaf() {
        const GIB: u64 = 1u64 << 30;
        struct TempDir(std::path::PathBuf);
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let mount_root = std::env::temp_dir().join(format!(
            "mlx-core-cgroup-v2-subtree-test-{}",
            std::process::id()
        ));
        let _guard = TempDir(mount_root.clone());
        let leaf = mount_root.join("model");
        std::fs::create_dir_all(&leaf).expect("create subtree leaf dir");

        // Mount root has a loose 64 GiB limit; leaf `model` enforces a tighter 16 GiB limit.
        std::fs::write(mount_root.join("memory.max"), format!("{}\n", 64 * GIB))
            .expect("write root limit");
        std::fs::write(mount_root.join("memory.current"), format!("{}\n", 10 * GIB))
            .expect("write root current");
        std::fs::write(leaf.join("memory.max"), format!("{}\n", 16 * GIB))
            .expect("write leaf limit");
        std::fs::write(leaf.join("memory.current"), format!("{}\n", 12 * GIB))
            .expect("write leaf current");
        std::fs::write(
            leaf.join("memory.stat"),
            format!("inactive_file {}\n", 2 * GIB),
        )
        .expect("write leaf stat");

        let escape_mount = |path: &std::path::Path| {
            let mut out = String::new();
            for ch in path.to_string_lossy().chars() {
                match ch {
                    '\\' => out.push_str("\\134"),
                    ' ' => out.push_str("\\040"),
                    '\t' => out.push_str("\\011"),
                    '\n' => out.push_str("\\012"),
                    _ => out.push(ch),
                }
            }
            out
        };
        let escaped_mount = escape_mount(&mount_root);
        let mountinfo = format!(
            "30 22 0:26 /other /mnt/other rw - cgroup2 cgroup2 rw\n\
             31 22 0:26 /batch {escaped_mount} rw,nosuid shared:5 - cgroup2 cgroup2 rw\n"
        );
        let mounts = resolve_cgroup_v2_mounts("0::/batch/model\n", &mountinfo);
        assert_eq!(mounts, vec![(leaf.clone(), mount_root.clone())]);
        assert_eq!(
            read_cgroup_v2_mounts(&mounts),
            Some(LinuxCgroupMemory {
                limit: 16 * GIB,
                available: 6 * GIB,
            })
        );

        // Layered mount where a broader `/` mount exposes a tighter 12 GiB root
        // ancestor than the `/batch` subtree mount's 16 GiB leaf.
        let broad_root = std::env::temp_dir().join(format!(
            "mlx-core-cgroup-v2-broad-test-{}",
            std::process::id()
        ));
        let _broad_guard = TempDir(broad_root.clone());
        let broad_leaf = broad_root.join("batch/model");
        std::fs::create_dir_all(&broad_leaf).expect("create broad leaf dir");
        std::fs::write(broad_root.join("memory.max"), format!("{}\n", 12 * GIB))
            .expect("write broad root limit");
        std::fs::write(broad_root.join("memory.current"), format!("{}\n", 9 * GIB))
            .expect("write broad root current");
        let escaped_broad = escape_mount(&broad_root);
        let layered_mountinfo = format!(
            "29 22 0:26 / {escaped_broad} rw - cgroup2 cgroup2 rw\n\
             31 22 0:26 /batch {escaped_mount} rw - cgroup2 cgroup2 rw\n"
        );
        let layered_mounts = resolve_cgroup_v2_mounts("0::/batch/model\n", &layered_mountinfo);
        assert_eq!(layered_mounts.len(), 2);
        assert_eq!(
            read_cgroup_v2_mounts(&layered_mounts),
            Some(LinuxCgroupMemory {
                limit: 12 * GIB,
                available: 3 * GIB,
            })
        );
    }

    #[test]
    fn non_reclaimable_formula() {
        let h = HostMemory {
            physical: 128 << 30,
            free: 1,
            file: 2,
            anonymous: 40 << 30,
            purgeable: 3 << 30,
            wired: 7 << 30,
            compressor: 5 << 30,
        };
        assert_eq!(h.available(), 3);
        assert_eq!(h.non_reclaimable(), (7 + 5 + 40 - 3) << 30);
        let under = HostMemory {
            purgeable: 50 << 30,
            ..h
        };
        assert_eq!(under.non_reclaimable(), 12 << 30);
    }
}
