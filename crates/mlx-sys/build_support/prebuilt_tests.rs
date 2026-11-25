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

fn mlx_sys_manifest_dir() -> PathBuf {
    if let Some(dir) = option_env!("CARGO_MANIFEST_DIR") {
        return PathBuf::from(dir);
    }
    Path::new(file!())
        .parent()
        .and_then(Path::parent)
        .expect("prebuilt_tests.rs lives under crates/mlx-sys/build_support")
        .to_path_buf()
}

fn populate_complete_windows_layout(root: &Path) {
    for dir in [
        "lib",
        "include/mlx",
        "include/fmt",
        "include/cccl",
        "include/cute",
        "include/cutlass",
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    std::fs::write(root.join("openblas.dll"), b"dll-bytes").unwrap();
    std::fs::write(root.join("lib/mlx.lib"), b"mlx-lib").unwrap();
    std::fs::write(root.join("lib/openblas.lib"), b"openblas-lib").unwrap();
}

#[test]
fn prebuilt_build_key_is_lf_crlf_invariant_and_matches_pinned_keys() {
    let lf_dir = TempDir::new("mlx-build-key-lf");
    let crlf_dir = TempDir::new("mlx-build-key-crlf");
    for (dir, eol) in [(lf_dir.path(), "\n"), (crlf_dir.path(), "\r\n")] {
        std::fs::create_dir_all(dir.join("cmake/nested")).unwrap();
        std::fs::write(
            dir.join("cmake/first.cmake"),
            format!("set(FOO ON){eol}set(BAR OFF){eol}"),
        )
        .unwrap();
        std::fs::write(
            dir.join("cmake/nested/second.cmake"),
            format!("# comment{eol}set(BAZ 1){eol}"),
        )
        .unwrap();
    }
    let sample_defines = vec![("MLX_BUILD_CUDA".to_string(), "ON".to_string())];
    let lf_key = prebuilt_build_key(lf_dir.path(), &sample_defines).unwrap();
    let crlf_key = prebuilt_build_key(crlf_dir.path(), &sample_defines).unwrap();
    assert_eq!(
        lf_key, crlf_key,
        "CRLF cmake files must hash identically to LF"
    );

    let manifest_dir = mlx_sys_manifest_dir();
    let darwin_defines = portable_cmake_defines(true, "macos", "aarch64", Some("26.2"));
    let darwin_key = prebuilt_build_key(&manifest_dir, &darwin_defines).unwrap();
    assert_eq!(
        &darwin_key[..8],
        "3d951e0b",
        "macOS prebuilt build key regressed: {darwin_key}"
    );

    // Pin CUDA architectures to "121a" directly in defines so the regression
    // check stays immune to any ambient MLX_CUDA_ARCHITECTURES env var.
    let pin_cuda_arch = |defines: Vec<(String, String)>| -> Vec<(String, String)> {
        defines
            .into_iter()
            .map(|(k, v)| {
                if matches!(
                    k.as_str(),
                    "MLX_CUDA_ARCHITECTURES" | "CMAKE_CUDA_ARCHITECTURES"
                ) {
                    (k, "121a".to_string())
                } else {
                    (k, v)
                }
            })
            .collect()
    };
    let linux_defines = pin_cuda_arch(portable_cmake_defines(false, "linux", "aarch64", None));
    let linux_key = prebuilt_build_key(&manifest_dir, &linux_defines).unwrap();
    assert_eq!(
        &linux_key[..8],
        "23ca1e3b",
        "Linux prebuilt build key regressed: {linux_key}"
    );

    let windows_defines = pin_cuda_arch(portable_cmake_defines(false, "windows", "aarch64", None));
    let windows_key = prebuilt_build_key(&manifest_dir, &windows_defines).unwrap();
    assert_eq!(
        &windows_key[..8],
        "5db20dc6",
        "Windows prebuilt build key regressed: {windows_key}"
    );
}

#[test]
fn prebuilt_layout_complete_validates_windows_runtime_assets() {
    let triple = "aarch64-pc-windows-msvc";
    let dir = TempDir::new("mlx-win-layout");
    populate_complete_windows_layout(dir.path());
    assert!(prebuilt_layout_complete(dir.path(), triple));

    // Missing openblas.dll fails.
    let missing_dll = TempDir::new("mlx-win-layout-no-dll");
    populate_complete_windows_layout(missing_dll.path());
    std::fs::remove_file(missing_dll.path().join("openblas.dll")).unwrap();
    assert!(!prebuilt_layout_complete(missing_dll.path(), triple));

    // Missing include/cute fails.
    let missing_cute = TempDir::new("mlx-win-layout-no-cute");
    populate_complete_windows_layout(missing_cute.path());
    std::fs::remove_dir_all(missing_cute.path().join("include/cute")).unwrap();
    assert!(!prebuilt_layout_complete(missing_cute.path(), triple));

    // Missing lib/openblas.lib fails.
    let missing_lib = TempDir::new("mlx-win-layout-no-openblas-lib");
    populate_complete_windows_layout(missing_lib.path());
    std::fs::remove_file(missing_lib.path().join("lib/openblas.lib")).unwrap();
    assert!(!prebuilt_layout_complete(missing_lib.path(), triple));
}

#[test]
fn export_prebuilt_windows_copies_openblas_dll_and_fails_when_missing() {
    let triple = "aarch64-pc-windows-msvc";
    let tmp = TempDir::new("mlx-win-export");
    let dst = tmp.path().join("cmake-install");
    let export_dir = tmp.path().join("exported");

    for dir in [
        "lib",
        "include/mlx",
        "include/cccl",
        "include/cute",
        "include/cutlass",
        "build/_deps/fmt-src/include/fmt",
    ] {
        std::fs::create_dir_all(dst.join(dir)).unwrap();
    }
    std::fs::write(dst.join("lib/mlx.lib"), b"mlx-lib").unwrap();
    std::fs::write(dst.join("lib/openblas.lib"), b"openblas-lib").unwrap();
    std::fs::write(dst.join("openblas.dll"), b"openblas-dll-payload").unwrap();
    std::fs::write(dst.join("include/mlx/mlx.h"), b"// mlx").unwrap();
    std::fs::write(dst.join("include/cccl/cuda.h"), b"// cccl").unwrap();
    std::fs::write(dst.join("include/cute/tensor.hpp"), b"// cute").unwrap();
    std::fs::write(dst.join("include/cutlass/cutlass.h"), b"// cutlass").unwrap();
    std::fs::write(
        dst.join("build/_deps/fmt-src/include/fmt/core.h"),
        b"// fmt",
    )
    .unwrap();

    export_prebuilt(
        &dst,
        &export_dir,
        "6dc14b919fc1966547456dff1aeb13f283ea93b5",
        "mlx-prebuilt-6dc14b919fc1-5db20dc6",
        "5db20dc600000000",
        triple,
        "",
        "",
        &[],
    )
    .expect("complete Windows cmake install should export cleanly");

    assert_eq!(
        std::fs::read(export_dir.join("openblas.dll")).unwrap(),
        b"openblas-dll-payload"
    );
    assert!(prebuilt_layout_complete(&export_dir, triple));

    // Removing openblas.dll must make export fail.
    std::fs::remove_file(dst.join("openblas.dll")).unwrap();
    let err = export_prebuilt(
        &dst,
        &export_dir,
        "6dc14b919fc1966547456dff1aeb13f283ea93b5",
        "mlx-prebuilt-6dc14b919fc1-5db20dc6",
        "5db20dc600000000",
        triple,
        "",
        "",
        &[],
    )
    .expect_err("export without openblas.dll must fail");
    assert!(
        err.to_string().contains("openblas.dll"),
        "expected openblas.dll in error, got: {err}"
    );
}

#[test]
fn cargo_build_unit_hash_handles_legacy_and_cargo_1_100_layouts() {
    assert_eq!(
        cargo_build_unit_hash(Path::new(
            "/repo/target/aarch64-pc-windows-msvc/release/build/mlx-sys-0123456789abcdef/out"
        ))
        .as_deref(),
        Some("0123456789abcdef")
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new(
            "/repo/target/aarch64-pc-windows-msvc/release/build/mlx-sys/0123456789abcdef/out"
        ))
        .as_deref(),
        Some("0123456789abcdef")
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("mlx-sys-deadbeef/out")).as_deref(),
        Some("deadbeef")
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("mlx-sys/deadbeef/out")).as_deref(),
        Some("deadbeef")
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("/repo/target/release/build/other-deadbeef/out")),
        None
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("/repo/target/release/build/mlx-sys-DEADBEEF/out")),
        None
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("/repo/target/release/build/mlx-sys-/out")),
        None
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("/repo/target/release/build/mlx-sys/deadbeef/run")),
        None
    );
    assert_eq!(
        cargo_build_unit_hash(Path::new("/repo/target/release/build/other/deadbeef/out")),
        None
    );
}
