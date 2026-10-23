//! Content-based Metal cache shared by the build script and CPU regressions.
use std::path::PathBuf;

/// One `.metal` → `.air` compile. Arguments exclude input/output/depfile paths.
pub struct AirJob {
    pub src: PathBuf,
    pub air: PathBuf,
    pub args: Vec<String>,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Include the current job's source identity independently of saved depfile
/// paths. A shared Cargo target may contain a stamp from another checkout
/// whose old dependencies still exist and have not changed.
pub fn air_stamp(job: &AirJob, toolchain: &str, deps: &[PathBuf]) -> Option<String> {
    let source = job.src.canonicalize().ok()?;
    // A compiler-produced depfile must include this job's source. Otherwise
    // never certify an AIR compiled from a different source tree.
    if !deps
        .iter()
        .any(|p| p.canonicalize().ok().as_ref() == Some(&source))
    {
        return None;
    }
    let mut stamp = format!(
        "toolchain {toolchain}\nargs {}\nsource {}\n",
        job.args.join(" "),
        source.display()
    );
    for dep in deps {
        let bytes = std::fs::read(dep).ok()?;
        stamp.push_str(&format!("{:016x} {}\n", fnv1a(&bytes), dep.display()));
    }
    Some(stamp)
}

pub fn air_is_current(job: &AirJob, toolchain: &str) -> bool {
    let stamp_path = job.air.with_extension("air.stamp");
    let (Ok(stamp), true) = (std::fs::read_to_string(stamp_path), job.air.is_file()) else {
        return false;
    };
    // Old two-line headers did not bind a stamp to its current source job.
    if !stamp
        .lines()
        .nth(2)
        .is_some_and(|line| line.starts_with("source "))
    {
        return false;
    }
    let deps: Vec<PathBuf> = stamp
        .lines()
        .skip(3)
        .filter_map(|line| line.split_once(' ').map(|(_, path)| PathBuf::from(path)))
        .collect();
    air_stamp(job, toolchain, &deps).as_deref() == Some(stamp.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "mlx-metal-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn write(&self, name: &str, bytes: &str) -> PathBuf {
            let p = self.0.join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, bytes).unwrap();
            p
        }
        fn job(&self, source: &str) -> AirJob {
            AirJob {
                src: self.write(source, "kernel source"),
                air: self.write("shared-target/kernel.air", "compiled air"),
                args: vec!["-fno-fast-math".into()],
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn save(job: &AirJob, deps: &[PathBuf]) {
        std::fs::write(
            job.air.with_extension("air.stamp"),
            air_stamp(job, "metal-v1", deps).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn another_checkout_in_shared_target_invalidates_unchanged_old_dependencies() {
        let f = Fixture::new();
        let old = f.job("baseline/kernel.metal");
        let header = f.write("baseline/header.h", "old header");
        let deps = vec![old.src.clone(), header];
        save(&old, &deps);
        assert!(air_is_current(&old, "metal-v1"));
        let current = f.job("worktree/kernel.metal");
        // Both files still exist with identical bytes. Source identity alone
        // must invalidate the baseline artifact before current-tree edits.
        assert!(!air_is_current(&current, "metal-v1"));
        assert!(air_stamp(&current, "metal-v1", &deps).is_none());
        std::fs::write(&current.src, "new kernel ABI").unwrap();
        assert!(!air_is_current(&current, "metal-v1"));
    }

    #[test]
    fn source_header_flags_toolchain_and_missing_air_invalidate() {
        let f = Fixture::new();
        let mut job = f.job("source/kernel.metal");
        let header = f.write("source/header.h", "header");
        let deps = vec![job.src.clone(), header.clone()];
        save(&job, &deps);
        assert!(air_is_current(&job, "metal-v1"));
        assert!(!air_is_current(&job, "metal-v2"));
        job.args.push("-DNEW_ABI=1".into());
        assert!(!air_is_current(&job, "metal-v1"));
        job.args.pop();
        std::fs::write(&header, "changed header").unwrap();
        assert!(!air_is_current(&job, "metal-v1"));
        save(&job, &deps);
        std::fs::write(&job.src, "changed source").unwrap();
        assert!(!air_is_current(&job, "metal-v1"));
        save(&job, &deps);
        std::fs::remove_file(&job.air).unwrap();
        assert!(!air_is_current(&job, "metal-v1"));
    }

    #[test]
    fn legacy_stamp_is_never_reused() {
        let f = Fixture::new();
        let job = f.job("source/kernel.metal");
        let deps = vec![job.src.clone()];
        let legacy = air_stamp(&job, "metal-v1", &deps)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("source "))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(job.air.with_extension("air.stamp"), legacy).unwrap();
        assert!(!air_is_current(&job, "metal-v1"));
    }
}
