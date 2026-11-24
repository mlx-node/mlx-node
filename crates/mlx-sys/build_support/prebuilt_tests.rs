use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn unique_temp_dir(prefix: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "{prefix}-{}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(prefix: &str) -> Self {
        let path = unique_temp_dir(prefix);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn lock(names: &[&'static str]) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = names
            .iter()
            .map(|&name| (name, env::var_os(name)))
            .collect();
        unsafe {
            for &name in names {
                env::remove_var(name);
            }
        }
        Self { saved, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
    }
}

struct Fixture {
    root: PathBuf,
    _env: EnvGuard,
}

impl Fixture {
    fn new() -> Self {
        let env_guard =
            EnvGuard::lock(&["MLX_PREBUILT", "MLX_PREBUILT_DIR", "MLX_PREBUILT_EXPORT"]);
        let root = unique_temp_dir("mlx-explicit-prebuilt");
        for directory in ["cmake", "mlx", "artifact/lib", "artifact/include/mlx"] {
            std::fs::create_dir_all(root.join(directory)).unwrap();
        }
        std::fs::write(root.join("mlx/CMakeLists.txt"), "fixture\n").unwrap();
        std::fs::write(root.join("cmake/build-input.cmake"), "build input v1\n").unwrap();
        std::fs::write(root.join("artifact/lib/libmlx.a"), []).unwrap();
        for arguments in [
            vec!["init", "-q"],
            vec!["add", "CMakeLists.txt"],
            vec![
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
        ] {
            let output = Command::new("git")
                .args(arguments)
                .current_dir(root.join("mlx"))
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        unsafe {
            env::set_var("MLX_PREBUILT_DIR", root.join("artifact"));
        }
        Self {
            root,
            _env: env_guard,
        }
    }

    fn manifest(&self, override_field: Option<(&str, &str)>) {
        let (head, dirty) = submodule_state(&self.root.join("mlx")).unwrap();
        assert!(!dirty);
        let defines = portable_cmake_defines(true, "macos", "aarch64", Some("26.2"));
        let key = prebuilt_build_key(&self.root, &defines).unwrap();
        let fields = [
            ("mlx_commit", head.as_str()),
            ("triple", "aarch64-apple-darwin"),
            ("build_key", key.as_str()),
        ];
        let body = fields
            .iter()
            .map(|&(field, value)| {
                let value = override_field
                    .filter(|&(name, _)| name == field)
                    .map_or(value, |(_, value)| value);
                format!("\"{field}\":\"{value}\"")
            })
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            self.root.join("artifact/manifest.json"),
            format!("{{{body}}}"),
        )
        .unwrap();
    }

    fn resolve(&self) -> MlxSource {
        resolve_mlx_source(
            &self.root,
            &self.root.join("mlx"),
            "aarch64-apple-darwin",
            true,
            Some("26.2"),
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// Run this environment-owning build-script test in its own process.
#[test]
fn explicit_metal_prebuilt_requires_current_source_and_build_key() {
    let fixture = Fixture::new();
    assert!(
        std::panic::catch_unwind(|| fixture.resolve()).is_err(),
        "manifestless artifact accepted"
    );
    fixture.manifest(None);
    assert!(matches!(fixture.resolve(), MlxSource::Prebuilt { .. }));
    for field in ["mlx_commit", "triple", "build_key"] {
        fixture.manifest(Some((field, "stale")));
        assert!(
            std::panic::catch_unwind(|| fixture.resolve()).is_err(),
            "stale {field} accepted"
        );
    }
    fixture.manifest(None);
    std::fs::write(
        fixture.root.join("cmake/build-input.cmake"),
        "build input v2\n",
    )
    .unwrap();
    assert!(
        std::panic::catch_unwind(|| fixture.resolve()).is_err(),
        "old build inputs accepted after cmake changed"
    );
    fixture.manifest(None);
    assert!(matches!(fixture.resolve(), MlxSource::Prebuilt { .. }));
    std::fs::write(fixture.root.join("mlx/CMakeLists.txt"), "modified source\n").unwrap();
    assert!(
        std::panic::catch_unwind(|| fixture.resolve()).is_err(),
        "modified source accepted"
    );
    unsafe { env::set_var("MLX_PREBUILT", "0") };
    assert!(
        matches!(fixture.resolve(), MlxSource::Source { .. }),
        "source kill switch must override an incompatible explicit artifact"
    );
}

#[test]
fn linux_cuda_portable_defines_and_layout() {
    let defines = portable_cmake_defines(false, "linux", "aarch64", None);
    for (key, value) in [
        ("MLX_BUILD_CUDA", "ON"),
        ("CMAKE_CUDA_RUNTIME_LIBRARY", "Shared"),
        ("CMAKE_INSTALL_LIBDIR", "lib"),
        ("CMAKE_DISABLE_FIND_PACKAGE_NCCL", "ON"),
    ] {
        assert!(
            defines.iter().any(|(k, v)| k == key && v == value),
            "missing define {key}={value} in {defines:?}"
        );
    }
    assert!(
        defines
            .iter()
            .any(|(k, v)| k == "CMAKE_CUDA_ARCHITECTURES" && !v.is_empty()),
        "missing CMAKE_CUDA_ARCHITECTURES in {defines:?}"
    );

    let triple = "aarch64-unknown-linux-gnu";
    assert!(is_linux_triple(triple));

    let dir = TempDir::new("mlx-linux-layout");
    std::fs::create_dir_all(dir.path().join("lib")).unwrap();
    std::fs::create_dir_all(dir.path().join("include/mlx")).unwrap();
    std::fs::write(dir.path().join("lib/libmlx.a"), []).unwrap();
    assert!(
        !prebuilt_layout_complete(dir.path(), triple),
        "Linux layout must require CUDA and fmt include dirs"
    );

    let required_includes = ["fmt", "cccl", "cute", "cutlass"];
    for subdir in required_includes {
        std::fs::create_dir_all(dir.path().join("include").join(subdir)).unwrap();
    }
    assert!(
        prebuilt_layout_complete(dir.path(), triple),
        "Linux layout should be complete without lib/mlx.metallib"
    );
    assert!(
        !prebuilt_layout_complete(dir.path(), "aarch64-apple-darwin"),
        "macOS layout must still require lib/mlx.metallib"
    );

    for missing in required_includes {
        let path = dir.path().join("include").join(missing);
        std::fs::remove_dir_all(&path).unwrap();
        assert!(
            !prebuilt_layout_complete(dir.path(), triple),
            "Linux layout accepted without include/{missing}"
        );
        std::fs::create_dir_all(&path).unwrap();
    }
    std::fs::remove_file(dir.path().join("lib/libmlx.a")).unwrap();
    assert!(
        !prebuilt_layout_complete(dir.path(), triple),
        "Linux layout accepted without lib/libmlx.a"
    );
}

#[test]
fn explicit_linux_prebuilt_requires_cuda_headers() {
    let _env = EnvGuard::lock(&["MLX_PREBUILT", "MLX_PREBUILT_DIR", "MLX_PREBUILT_EXPORT"]);
    let root = TempDir::new("mlx-linux-explicit");
    let artifact = root.path().join("artifact");
    std::fs::create_dir_all(artifact.join("lib")).unwrap();
    std::fs::create_dir_all(artifact.join("include/mlx")).unwrap();
    std::fs::write(artifact.join("lib/libmlx.a"), []).unwrap();
    unsafe {
        env::set_var("MLX_PREBUILT_DIR", &artifact);
    }

    let resolve = || {
        resolve_mlx_source(
            root.path(),
            &root.path().join("mlx"),
            "aarch64-unknown-linux-gnu",
            false,
            None,
        )
    };
    assert!(
        std::panic::catch_unwind(resolve).is_err(),
        "incomplete Linux MLX_PREBUILT_DIR accepted"
    );

    for subdir in ["fmt", "cccl", "cute", "cutlass"] {
        std::fs::create_dir_all(artifact.join("include").join(subdir)).unwrap();
    }
    assert!(matches!(resolve(), MlxSource::Prebuilt { .. }));
}

#[test]
fn stage_linux_cuda_includes_stages_out_dir_and_profile_dir() {
    let root = TempDir::new("mlx-linux-stage");
    let mlx_root = root.path().join("mlx-root");
    let profile_dir = root.path().join("target/release");
    let profile_include = profile_dir.join("include");
    let out_dir = profile_dir.join("build/mlx-sys-abc/out");
    std::fs::create_dir_all(mlx_root.join("include")).unwrap();
    std::fs::create_dir_all(&out_dir).unwrap();

    assert!(
        stage_linux_cuda_includes(&mlx_root, &out_dir).is_err(),
        "staging must fail when CUDA include subdirs are missing"
    );

    // Pre-existing sibling headers inside `<profile>/include/` must never be wiped.
    let sibling_header = profile_include.join("other_lib/keep.h");
    std::fs::create_dir_all(sibling_header.parent().unwrap()).unwrap();
    std::fs::write(&sibling_header, "keep me").unwrap();

    for subdir in LINUX_CUDA_INCLUDE_DIRS {
        let dir = mlx_root.join("include").join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker.h"), format!("v1-{subdir}")).unwrap();
    }

    stage_linux_cuda_includes(&mlx_root, &out_dir).unwrap();
    // Second call must be idempotent (skips work when symlink already matches).
    stage_linux_cuda_includes(&mlx_root, &out_dir).unwrap();

    for staged in [out_dir.join("cuda-include"), profile_include.clone()] {
        for subdir in LINUX_CUDA_INCLUDE_DIRS {
            assert_eq!(
                std::fs::read_to_string(staged.join(subdir).join("marker.h")).unwrap(),
                format!("v1-{subdir}"),
                "missing or mismatched staged header under {}",
                staged.join(subdir).display()
            );
        }
    }
    assert_eq!(
        std::fs::read_to_string(&sibling_header).unwrap(),
        "keep me",
        "pre-existing sibling header under <profile>/include was wiped"
    );

    // Re-running with a new mlx_root updates MLX-owned subdirs while preserving siblings.
    let mlx_root_v2 = root.path().join("mlx-root-v2");
    for subdir in LINUX_CUDA_INCLUDE_DIRS {
        let dir = mlx_root_v2.join("include").join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker.h"), format!("v2-{subdir}")).unwrap();
    }
    stage_linux_cuda_includes(&mlx_root_v2, &out_dir).unwrap();

    for staged in [out_dir.join("cuda-include"), profile_include.clone()] {
        for subdir in LINUX_CUDA_INCLUDE_DIRS {
            assert_eq!(
                std::fs::read_to_string(staged.join(subdir).join("marker.h")).unwrap(),
                format!("v2-{subdir}"),
                "stale staged header under {}",
                staged.join(subdir).display()
            );
        }
    }
    assert_eq!(
        std::fs::read_to_string(&sibling_header).unwrap(),
        "keep me",
        "pre-existing sibling header under <profile>/include was wiped on re-run"
    );

    // Migrating from a legacy `<profile>/include` symlink replaces only the symlink
    // with a real directory and never touches the old symlink target.
    #[cfg(unix)]
    {
        let legacy_profile = root.path().join("target/debug");
        let legacy_out = legacy_profile.join("build/mlx-sys-def/out");
        let legacy_include = legacy_profile.join("include");
        std::fs::create_dir_all(&legacy_out).unwrap();
        std::os::unix::fs::symlink(mlx_root.join("include"), &legacy_include).unwrap();

        stage_linux_cuda_includes(&mlx_root_v2, &legacy_out).unwrap();

        let meta = std::fs::symlink_metadata(&legacy_include).unwrap();
        assert!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "legacy <profile>/include symlink must be replaced with a real directory"
        );
        for subdir in LINUX_CUDA_INCLUDE_DIRS {
            assert_eq!(
                std::fs::read_to_string(legacy_include.join(subdir).join("marker.h")).unwrap(),
                format!("v2-{subdir}")
            );
            assert_eq!(
                std::fs::read_to_string(mlx_root.join("include").join(subdir).join("marker.h"))
                    .unwrap(),
                format!("v1-{subdir}"),
                "old symlink target was modified"
            );
        }

        // An unrelated `<profile>/include` symlink must NOT be unlinked.
        let other_profile = root.path().join("target/other");
        let other_out = other_profile.join("build/mlx-sys-ghi/out");
        let other_include = other_profile.join("include");
        let foreign_target = root.path().join("foreign-headers");
        std::fs::create_dir_all(&other_out).unwrap();
        std::fs::create_dir_all(&foreign_target).unwrap();
        std::fs::write(foreign_target.join("foreign.h"), "foreign").unwrap();
        std::os::unix::fs::symlink(&foreign_target, &other_include).unwrap();

        stage_linux_cuda_includes(&mlx_root_v2, &other_out).unwrap();

        let other_meta = std::fs::symlink_metadata(&other_include).unwrap();
        assert!(
            other_meta.file_type().is_symlink(),
            "unrelated <profile>/include symlink must be preserved"
        );
        assert_eq!(
            std::fs::read_link(&other_include).unwrap(),
            foreign_target,
            "unrelated <profile>/include symlink target changed"
        );
        assert_eq!(
            std::fs::read_to_string(other_include.join("foreign.h")).unwrap(),
            "foreign"
        );
    }
}
