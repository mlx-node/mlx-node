use super::*;

struct Fixture {
    root: PathBuf,
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl Fixture {
    fn new() -> Self {
        let root = env::temp_dir().join(format!(
            "mlx-explicit-prebuilt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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
        let names = ["MLX_PREBUILT", "MLX_PREBUILT_DIR", "MLX_PREBUILT_EXPORT"];
        let saved = names.map(|name| (name, env::var_os(name))).to_vec();
        unsafe {
            for name in names {
                env::remove_var(name);
            }
            env::set_var("MLX_PREBUILT_DIR", root.join("artifact"));
        }
        Self { root, saved }
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
        for (name, value) in &self.saved {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
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
