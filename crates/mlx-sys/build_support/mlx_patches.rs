//! Apply tracked MLX patches to a generated tree, preserving the checkout.
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn remove_entry(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

fn excluded(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git" | "build" | ".cache" | "__pycache__")
    )
}

fn sync_tree(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(destination)? {
        let entry = entry?;
        let name = entry.file_name();
        if excluded(&name) || fs::symlink_metadata(source.join(&name)).is_err() {
            remove_entry(&entry.path())?;
        }
    }
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if excluded(&name) {
            continue;
        }
        let src = entry.path();
        let dst = destination.join(name);
        let kind = entry.file_type()?;
        let old = fs::symlink_metadata(&dst).ok();
        if kind.is_symlink() {
            let target = fs::read_link(&src)?;
            if target.is_absolute() {
                return Err(io::Error::other(format!(
                    "absolute MLX source symlink: {}",
                    src.display()
                )));
            }
            if fs::read_link(&dst).ok().as_ref() != Some(&target) {
                if old.is_some() {
                    remove_entry(&dst)?;
                }
                #[cfg(unix)]
                std::os::unix::fs::symlink(target, &dst)?;
                #[cfg(not(unix))]
                return Err(io::Error::other(
                    "MLX source symlinks require a Unix build host",
                ));
            }
        } else if kind.is_dir() {
            if old.is_some_and(|metadata| !metadata.is_dir()) {
                remove_entry(&dst)?;
            }
            sync_tree(&src, &dst)?;
        } else if kind.is_file() {
            if old.as_ref().is_some_and(|metadata| !metadata.is_file()) {
                remove_entry(&dst)?;
            }
            let bytes = fs::read(&src)?;
            if fs::read(&dst).ok().as_deref() != Some(bytes.as_slice()) {
                fs::copy(&src, &dst)?;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let permissions = fs::metadata(&src)?.permissions();
                if fs::metadata(&dst)?.permissions().mode() != permissions.mode() {
                    fs::set_permissions(&dst, permissions)?;
                }
            }
        }
    }
    Ok(())
}

pub fn prepare(checkout: &Path, patches: &Path, out: &Path) -> io::Result<PathBuf> {
    let stage = out.join("mlx-patch-stage");
    let generated = out.join("mlx-patched-source");
    if fs::symlink_metadata(&stage).is_ok() {
        remove_entry(&stage)?;
    }
    sync_tree(checkout, &stage)?;
    let mut files = fs::read_dir(patches)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    files.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "patch")
    });
    files.sort();
    for patch in files {
        let patch = fs::canonicalize(patch)?;
        // Stop Git's repository discovery at OUT_DIR. The generated tree has
        // no index, so only its files can be changed by this ordinary apply.
        let result = Command::new("git")
            .current_dir(&stage)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env("GIT_CEILING_DIRECTORIES", fs::canonicalize(out)?)
            .args(["apply", "--whitespace=error-all"])
            .arg(&patch)
            .output()?;
        if !result.status.success() {
            return Err(io::Error::other(format!(
                "MLX patch {} no longer applies: {}",
                patch.display(),
                String::from_utf8_lossy(&result.stderr)
            )));
        }
    }
    // Copy only changed contents, so bridge edits do not rebuild all MLX.
    // Reconstructing the staging tree also removes previously applied patches.
    sync_tree(&stage, &generated)?;
    fs::remove_dir_all(stage)?;
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "mlx-patches-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            for dir in ["source", "patches", "out"] {
                fs::create_dir_all(root.join(dir)).unwrap();
            }
            fs::write(root.join("source/header.h"), "before\n").unwrap();
            fs::write(root.join("patches/001.patch"),
                "diff --git a/header.h b/header.h\n--- a/header.h\n+++ b/header.h\n@@ -1 +1 @@\n-before\n+after\n").unwrap();
            Self(root)
        }
        fn prepare(&self) -> io::Result<PathBuf> {
            prepare(
                &self.0.join("source"),
                &self.0.join("patches"),
                &self.0.join("out"),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn patch_is_reproducible_without_changing_checkout_or_unchanged_mtimes() {
        let f = Fixture::new();
        let generated = f.prepare().unwrap();
        assert_eq!(
            fs::read_to_string(f.0.join("source/header.h")).unwrap(),
            "before\n"
        );
        assert_eq!(
            fs::read_to_string(generated.join("header.h")).unwrap(),
            "after\n"
        );
        let modified = fs::metadata(generated.join("header.h"))
            .unwrap()
            .modified()
            .unwrap();
        f.prepare().unwrap();
        assert_eq!(
            fs::metadata(generated.join("header.h"))
                .unwrap()
                .modified()
                .unwrap(),
            modified
        );
        fs::write(f.0.join("source/extra.h"), "new\n").unwrap();
        f.prepare().unwrap();
        assert_eq!(
            fs::read_to_string(generated.join("extra.h")).unwrap(),
            "new\n"
        );
        fs::remove_file(f.0.join("source/extra.h")).unwrap();
        fs::remove_file(f.0.join("patches/001.patch")).unwrap();
        f.prepare().unwrap();
        assert!(!generated.join("extra.h").exists());
        assert_eq!(
            fs::read_to_string(generated.join("header.h")).unwrap(),
            "before\n"
        );
    }
    #[test]
    fn changed_patch_context_fails_without_touching_previous_tree() {
        let f = Fixture::new();
        let generated = f.prepare().unwrap();
        fs::write(f.0.join("source/header.h"), "incompatible\n").unwrap();
        assert!(
            f.prepare()
                .unwrap_err()
                .to_string()
                .contains("no longer applies")
        );
        assert_eq!(
            fs::read_to_string(generated.join("header.h")).unwrap(),
            "after\n"
        );
        assert_eq!(
            fs::read_to_string(f.0.join("source/header.h")).unwrap(),
            "incompatible\n"
        );
    }
}
