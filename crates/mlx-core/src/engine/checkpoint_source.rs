#![allow(dead_code)]
//! Where a model's weights come from: a safetensors directory, or a GGUF read
//! into memory at load beside an asset-only directory (`config.json`,
//! tokenizer, templates and `gguf-source.json`).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};
use serde_json::{Value, json};

use crate::array::DType;
use crate::utils::gguf::{
    GgufFile, NativeGgufFamily, native_gguf_unlease_in_use, parse_gguf, prepare_gguf_assets,
    prepare_gguf_assets_in, resolve_gemma4_gguf_source, resolve_muse_glimmer_gguf_source,
    source_file_identity_digest,
};
use crate::utils::gguf_import::PreloadInput;
use crate::utils::gguf_plan::{
    ImportOptions, ImportPlan, OutputRole, annotate_gguf_header, plan_gguf_import,
};
use crate::utils::gguf_source::{GgufFileIdentity, file_identity};

pub(crate) const GGUF_SOURCE_FILE: &str = "gguf-source.json";

/// Canonical path, the header parsed once, and the identity it was parsed under.
pub(crate) type GgufSourceFile = (PathBuf, Arc<GgufFile>, GgufFileIdentity);

pub(crate) enum CheckpointSource {
    Dir(PathBuf),
    Gguf(Box<GgufCheckpoint>),
}

pub(crate) struct GgufCheckpoint {
    /// Prepared asset entry, or the directory holding `gguf-source.json`
    /// the caller passed (a host config overlay).
    pub asset_dir: PathBuf,
    pub main: GgufSourceFile,
    pub companion: Option<GgufSourceFile>,
    pub plans: (ImportPlan, Option<ImportPlan>),
    pub opts: ImportOptions,
    pub family: NativeGgufFamily,
    /// `asset_dir` carries a lease this load must release on failure.
    pub leased: bool,
}

fn is_gguf_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

impl CheckpointSource {
    pub(crate) async fn resolve(path: &Path, family: NativeGgufFamily) -> Result<Self> {
        Self::resolve_in(path, family, None).await
    }

    /// `cache_root` None = the default asset root.
    pub(crate) async fn resolve_in(
        path: &Path,
        family: NativeGgufFamily,
        cache_root: Option<&Path>,
    ) -> Result<Self> {
        if path.is_dir() && path.join(GGUF_SOURCE_FILE).is_file() {
            return Ok(Self::Gguf(Box::new(GgufCheckpoint::from_asset_dir(
                path, family,
            )?)));
        }
        let gguf = match family {
            NativeGgufFamily::Qwen35 | NativeGgufFamily::Qwen35Moe | NativeGgufFamily::Clef => {
                is_gguf_file(path).then(|| path.to_path_buf())
            }
            NativeGgufFamily::Gemma4 => resolve_gemma4_gguf_source(path)?,
            NativeGgufFamily::MuseGlimmer => resolve_muse_glimmer_gguf_source(path)?,
        };
        let Some(gguf) = gguf else {
            return Ok(Self::Dir(path.to_path_buf()));
        };
        let checkpoint = match cache_root {
            Some(root) => prepare_gguf_assets_in(&gguf, family, root).await?,
            None => prepare_gguf_assets(&gguf, family).await?,
        };
        Ok(Self::Gguf(Box::new(checkpoint)))
    }

    pub(crate) fn asset_dir(&self) -> &Path {
        match self {
            Self::Dir(dir) => dir,
            Self::Gguf(checkpoint) => &checkpoint.asset_dir,
        }
    }

    pub(crate) fn release_lease(&self) {
        if let Self::Gguf(checkpoint) = self
            && checkpoint.leased
        {
            native_gguf_unlease_in_use(&checkpoint.asset_dir);
        }
    }
}

impl GgufCheckpoint {
    pub(crate) fn preload_input(&self, load_companion: bool) -> PreloadInput {
        let file = |(path, header, identity): &GgufSourceFile, plan: &ImportPlan| {
            (path.clone(), header.clone(), identity.clone(), plan.clone())
        };
        PreloadInput::new(
            file(&self.main, &self.plans.0),
            self.companion
                .as_ref()
                .zip(self.plans.1.as_ref())
                .map(|(companion, plan)| file(companion, plan)),
            self.family,
            load_companion,
            self.opts.clone(),
        )
    }

    /// The asset `config.json` must mark exactly the prefixes the plan tiles,
    /// or the builder would read t64 bytes as row-major (or the reverse).
    pub(crate) fn check_asset_config(&self) -> Result<()> {
        let path = self.asset_dir.join("config.json");
        let config: Value =
            serde_json::from_slice(&fs::read(&path).map_err(|e| {
                Error::from_reason(format!("Failed to read {}: {e}", path.display()))
            })?)
            .map_err(|e| Error::from_reason(format!("Failed to parse {}: {e}", path.display())))?;
        let marked = t64_prefixes(&config);
        if marked != self.plans.0.tiled_prefixes {
            return Err(Error::from_reason(format!(
                "'{}' marks {} t64 prefixes but the GGUF import tiles {}; the asset directory \
                 does not belong to '{}'",
                path.display(),
                marked.len(),
                self.plans.0.tiled_prefixes.len(),
                self.main.0.display()
            )));
        }
        Ok(())
    }

    /// A directory holding `gguf-source.json`: re-open the recorded GGUFs and
    /// refuse if either changed since prepare.
    fn from_asset_dir(dir: &Path, family: NativeGgufFamily) -> Result<Self> {
        let record = read_gguf_source_json(dir)?;
        let main = open_recorded(&record.main)?;
        family.check_main_header(&main.1)?;
        let companion = record.companion.as_ref().map(open_recorded).transpose()?;
        let (opts, main_plan, companion_plan) = native_import_plans(
            family,
            (&main.0, &main.1),
            companion
                .as_ref()
                .map(|(path, header, _)| (path.as_path(), &**header)),
        )?;
        Ok(Self {
            asset_dir: dir.to_path_buf(),
            main,
            companion,
            plans: (main_plan, companion_plan),
            opts,
            family,
            leased: false,
        })
    }
}

fn companion_role(family: NativeGgufFamily) -> OutputRole {
    OutputRole::Companion {
        filename: match family {
            NativeGgufFamily::MuseGlimmer => "draft.safetensors",
            NativeGgufFamily::Qwen35
            | NativeGgufFamily::Qwen35Moe
            | NativeGgufFamily::Clef
            | NativeGgufFamily::Gemma4 => "vision.safetensors",
        }
        .to_string(),
    }
}

/// The plans native prepare's converter calls made: bf16, K-quant import, the
/// Qwen3.5 tiled GDN layout for the Qwen3.5 families, the sibling
/// `config.json` authoritative when present; the companion takes the main
/// GGUF's directory as its config source. Returns the options the transform
/// runs with.
pub(crate) fn native_import_plans(
    family: NativeGgufFamily,
    main: (&Path, &GgufFile),
    companion: Option<(&Path, &GgufFile)>,
) -> Result<(ImportOptions, ImportPlan, Option<ImportPlan>)> {
    let parent = main.0.parent().unwrap_or(Path::new("."));
    let base = ImportOptions {
        dtype: Some(DType::BFloat16),
        import_k_quants: true,
        ..ImportOptions::default()
    };
    let main_opts = ImportOptions {
        native_qwen35_layout: matches!(
            family,
            NativeGgufFamily::Qwen35 | NativeGgufFamily::Qwen35Moe | NativeGgufFamily::Clef
        ),
        config_source_dir: parent
            .join("config.json")
            .is_file()
            .then(|| parent.to_path_buf()),
        ..base.clone()
    };
    let companion_opts = ImportOptions {
        config_source_dir: Some(parent.to_path_buf()),
        ..base
    };
    let main_plan = plan_gguf_import(main.1, main.0, &main_opts, OutputRole::Primary)?;
    let companion_plan = companion
        .map(|(path, header)| {
            plan_gguf_import(header, path, &companion_opts, companion_role(family))
        })
        .transpose()?;
    Ok((main_opts, main_plan, companion_plan))
}

/// Parse a GGUF header once, bracketed by two stats so the header, its digest
/// and the identity all describe one file version.
pub(crate) fn open_gguf_source(path: &Path) -> Result<GgufSourceFile> {
    let stat = |path: &Path| {
        fs::metadata(path).map_err(|e| {
            Error::from_reason(format!("Failed to stat GGUF '{}': {e}", path.display()))
        })
    };
    let before = stat(path)?;
    let mut header = parse_gguf(path)?;
    let identity = file_identity(path, &header, &before)?;
    if source_file_identity_digest(&identity.path, &stat(&identity.path)?)
        != identity.source_identity
    {
        return Err(Error::from_reason(format!(
            "GGUF '{}' changed while its header was read; retry the load",
            identity.path.display()
        )));
    }
    annotate_gguf_header(&mut header);
    Ok((identity.path.clone(), Arc::new(header), identity))
}

fn open_recorded(recorded: &GgufFileIdentity) -> Result<GgufSourceFile> {
    let opened = open_gguf_source(&recorded.path)?;
    if opened.2 != *recorded {
        return Err(Error::from_reason(format!(
            "GGUF '{}' changed since prepare; retry the load",
            recorded.path.display()
        )));
    }
    Ok(opened)
}

pub(crate) struct GgufSourceRecord {
    pub main: GgufFileIdentity,
    pub companion: Option<GgufFileIdentity>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn path_text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        Error::from_reason(format!("GGUF path '{}' is not valid UTF-8", path.display()))
    })
}

fn identity_json(identity: &GgufFileIdentity) -> Value {
    json!({
        "source_identity": identity.source_identity,
        "header_digest": hex(&identity.header_digest),
    })
}

pub(crate) fn write_gguf_source_json(
    dir: &Path,
    main: &GgufFileIdentity,
    companion: Option<&GgufFileIdentity>,
) -> Result<()> {
    let record = json!({
        "main": path_text(&main.path)?,
        "companion": companion.map(|c| path_text(&c.path)).transpose()?,
        "main_identity": identity_json(main),
        "companion_identity": companion.map(identity_json),
    });
    let text = serde_json::to_string_pretty(&record)
        .map_err(|e| Error::from_reason(format!("Failed to serialize {GGUF_SOURCE_FILE}: {e}")))?;
    fs::write(dir.join(GGUF_SOURCE_FILE), text)
        .map_err(|e| Error::from_reason(format!("Failed to write {GGUF_SOURCE_FILE}: {e}")))
}

pub(crate) fn read_gguf_source_json(dir: &Path) -> Result<GgufSourceRecord> {
    let path = dir.join(GGUF_SOURCE_FILE);
    let corrupt =
        |what: &str| Error::from_reason(format!("'{}' is corrupt: {what}", path.display()));
    let record: Value = serde_json::from_slice(
        &fs::read(&path)
            .map_err(|e| Error::from_reason(format!("Failed to read {}: {e}", path.display())))?,
    )
    .map_err(|e| corrupt(&e.to_string()))?;
    let identity = |path_key: &str, identity_key: &str| -> Result<Option<GgufFileIdentity>> {
        let file = match record.get(path_key) {
            None => return Err(corrupt(&format!("no '{path_key}'"))),
            Some(Value::Null) => return Ok(None),
            Some(value) => value
                .as_str()
                .ok_or_else(|| corrupt(&format!("'{path_key}' is not a string")))?,
        };
        let file = PathBuf::from(file);
        if !file.is_absolute() {
            return Err(corrupt(&format!("'{path_key}' is not an absolute path")));
        }
        let fields = record
            .get(identity_key)
            .filter(|value| value.is_object())
            .ok_or_else(|| corrupt(&format!("no '{identity_key}'")))?;
        let source_identity = fields
            .get("source_identity")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt(&format!("'{identity_key}.source_identity'")))?
            .to_string();
        let header_digest = fields
            .get("header_digest")
            .and_then(Value::as_str)
            .and_then(unhex32)
            .ok_or_else(|| corrupt(&format!("'{identity_key}.header_digest'")))?;
        Ok(Some(GgufFileIdentity {
            path: file,
            source_identity,
            header_digest,
        }))
    };
    Ok(GgufSourceRecord {
        main: identity("main", "main_identity")?.ok_or_else(|| corrupt("'main' is null"))?,
        companion: identity("companion", "companion_identity")?,
    })
}

/// Prefixes a `config.json` marks `"layout": "t64"`.
pub(crate) fn t64_prefixes(config: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for block in ["quantization", "quantization_config"] {
        if let Some(entries) = config.get(block).and_then(Value::as_object) {
            for (key, value) in entries {
                if value.get("layout").and_then(Value::as_str) == Some("t64") {
                    out.insert(key.clone());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::utils::gguf::{native_gguf_unlease_in_use, sweep_native_gguf_cache};
    use crate::utils::gguf_test_fixtures::{
        IMPORT_FIXTURE_NAMES, ImportFixture, ImportFixtureFamily, write_import_fixture,
    };

    fn work_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mlx-ckpt-src-{tag}-{}", uuid::Uuid::new_v4()))
    }

    fn golden_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/testdata/gguf_import_golden")
    }

    fn family(fx: &ImportFixture) -> NativeGgufFamily {
        match fx.family {
            ImportFixtureFamily::Qwen35 => NativeGgufFamily::Qwen35,
            ImportFixtureFamily::Qwen35Moe => NativeGgufFamily::Qwen35Moe,
            ImportFixtureFamily::Gemma4 => NativeGgufFamily::Gemma4,
            ImportFixtureFamily::MuseGlimmer => NativeGgufFamily::MuseGlimmer,
        }
    }

    fn names(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    fn set_mtime(path: &Path, age_secs: u64) {
        let time = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(time))
            .unwrap();
    }

    const TWO_DAYS: u64 = 2 * 24 * 3600;

    #[tokio::test]
    async fn prepare_gguf_assets_writes_no_weights() {
        let work = work_dir("assets");
        for name in IMPORT_FIXTURE_NAMES {
            let fx = write_import_fixture(name, &work.join("src").join(name));
            let ckpt =
                prepare_gguf_assets_in(&fx.main, family(&fx), &work.join("cache").join(name))
                    .await
                    .unwrap_or_else(|e| panic!("{name}: {}", e.reason));
            let dir = ckpt.asset_dir.clone();
            let entry = dir.file_name().unwrap().to_string_lossy().into_owned();
            assert!(entry.contains("-v8-"), "{name}: entry key {entry}");
            let marker = fs::read_to_string(dir.join(".complete")).unwrap();
            assert!(marker.starts_with("format=8\n"), "{name}: marker\n{marker}");

            let files = names(&dir);
            let weights: Vec<&String> = files
                .iter()
                .filter(|n| n.ends_with(".safetensors"))
                .collect();
            assert!(weights.is_empty(), "{name}: weight files {weights:?}");
            for required in ["config.json", GGUF_SOURCE_FILE, ".complete"] {
                assert!(
                    files.contains(required),
                    "{name}: no {required} in {files:?}"
                );
            }

            let golden = golden_root().join(name);
            let want: BTreeSet<String> = names(&golden)
                .into_iter()
                .filter(|n| {
                    n != "fixture.json" && n != "t64.json" && !n.ends_with(".manifest.json")
                })
                .collect();
            let got: BTreeSet<String> = files
                .iter()
                .filter(|n| !n.starts_with('.') && *n != GGUF_SOURCE_FILE)
                .cloned()
                .collect();
            assert_eq!(got, want, "{name}: asset file set");
            assert_eq!(
                got.contains("tokenizer.json"),
                golden.join("tokenizer.json").is_file(),
                "{name}: tokenizer.json"
            );
            for file in &want {
                assert!(
                    fs::read(dir.join(file)).unwrap() == fs::read(golden.join(file)).unwrap(),
                    "{name}: {file} differs from the base golden"
                );
            }
            ckpt.check_asset_config()
                .unwrap_or_else(|e| panic!("{name}: {}", e.reason));
            native_gguf_unlease_in_use(&dir);
        }
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn gguf_source_json_has_canonical_paths_and_identity() {
        let work = work_dir("source-json");
        for name in ["muse-draft", "qwen35-dense-untied"] {
            let fx = write_import_fixture(name, &work.join("src").join(name));
            let ckpt = prepare_gguf_assets_in(&fx.main, family(&fx), &work.join("cache"))
                .await
                .unwrap_or_else(|e| panic!("{name}: {}", e.reason));
            let record = read_gguf_source_json(&ckpt.asset_dir).unwrap();
            let raw: Value =
                serde_json::from_slice(&fs::read(ckpt.asset_dir.join(GGUF_SOURCE_FILE)).unwrap())
                    .unwrap();

            let expect = |path: &Path| {
                let canonical = path.canonicalize().unwrap();
                let header = parse_gguf(&canonical).unwrap();
                file_identity(&canonical, &header, &fs::metadata(&canonical).unwrap()).unwrap()
            };
            let main = expect(&fx.main);
            assert!(main.path.is_absolute(), "{name}: main path");
            assert_eq!(record.main, main, "{name}: main identity");
            assert_eq!(
                raw["main"].as_str(),
                main.path.to_str(),
                "{name}: main path"
            );
            assert_eq!(ckpt.main.2, main, "{name}: checkpoint main identity");
            match &fx.companion {
                Some(companion) => {
                    let companion = expect(companion);
                    assert_eq!(
                        record.companion.as_ref(),
                        Some(&companion),
                        "{name}: companion identity"
                    );
                    assert_eq!(
                        raw["companion"].as_str(),
                        companion.path.to_str(),
                        "{name}: companion path"
                    );
                    assert!(raw["companion_identity"].is_object(), "{name}");
                }
                None => {
                    assert!(record.companion.is_none(), "{name}: companion");
                    assert!(raw["companion"].is_null(), "{name}: companion");
                    assert!(raw["companion_identity"].is_null(), "{name}");
                }
            }
            native_gguf_unlease_in_use(&ckpt.asset_dir);
        }
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn concurrent_prepare_assets_publish_one_entry() {
        let work = work_dir("concurrent");
        let fx = write_import_fixture("qwen35-dense-untied", &work.join("src"));
        let cache = work.join("cache");
        let (left, right) = tokio::join!(
            prepare_gguf_assets_in(&fx.main, NativeGgufFamily::Qwen35, &cache),
            prepare_gguf_assets_in(&fx.main, NativeGgufFamily::Qwen35, &cache)
        );
        let (left, right) = (left.unwrap(), right.unwrap());
        assert_eq!(left.asset_dir, right.asset_dir);
        assert!(left.asset_dir.join(".complete").is_file());
        assert!(left.asset_dir.join(GGUF_SOURCE_FILE).is_file());
        let entries = names(&cache);
        let dirs: Vec<&String> = entries
            .iter()
            .filter(|n| !n.starts_with('.') && cache.join(n).is_dir())
            .collect();
        assert_eq!(dirs.len(), 1, "published entries {dirs:?}");
        assert!(
            !entries.iter().any(|n| n.contains(".staging-")),
            "staging survived publication: {entries:?}"
        );
        native_gguf_unlease_in_use(&left.asset_dir);
        native_gguf_unlease_in_use(&right.asset_dir);
        fs::remove_dir_all(&work).ok();
    }

    // The std Mutex guard is held across awaits on purpose: it serializes the
    // cap env and the sweep against other sweep tests for the whole test.
    #[allow(clippy::await_holding_lock)]
    #[cfg(unix)]
    #[tokio::test]
    async fn assets_lease_blocks_sweep() {
        let _env_guard = crate::utils::gguf::sweep_env_lock().lock().unwrap();
        let _env = crate::utils::gguf::SweepEnvVar::set_max_gb("0");
        let work = work_dir("lease");
        let fx = write_import_fixture("qwen35-dense-untied", &work.join("src"));
        let cache = work.join("cache");
        let ckpt = prepare_gguf_assets_in(&fx.main, NativeGgufFamily::Qwen35, &cache)
            .await
            .unwrap();
        let dir = ckpt.asset_dir.clone();
        assert!(ckpt.leased);
        // Source changed and the marker aged: stale unless leased.
        set_mtime(&fx.main, 3600);
        set_mtime(&dir.join(".complete"), TWO_DAYS);
        sweep_native_gguf_cache(&cache, "unrelated-key");
        assert!(dir.is_dir(), "the sweep removed a leased asset entry");
        native_gguf_unlease_in_use(&dir);
        sweep_native_gguf_cache(&cache, "unrelated-key");
        assert!(!dir.exists(), "the stale entry survived without its lease");
        fs::remove_dir_all(&work).ok();
    }

    #[allow(clippy::await_holding_lock)]
    #[cfg(unix)]
    #[tokio::test]
    async fn sweep_keeps_v7_and_v8_until_cutover() {
        let _env_guard = crate::utils::gguf::sweep_env_lock().lock().unwrap();
        let _env = crate::utils::gguf::SweepEnvVar::set_max_gb("0");
        let work = work_dir("formats");
        let fx = write_import_fixture("qwen35-dense-untied", &work.join("src"));
        let cache = work.join("cache");
        let ckpt = prepare_gguf_assets_in(&fx.main, NativeGgufFamily::Qwen35, &cache)
            .await
            .unwrap();
        let marker = fs::read_to_string(ckpt.asset_dir.join(".complete")).unwrap();
        assert!(marker.starts_with("format=8\n"), "{marker}");
        let entry = |format: u32| {
            let dir = cache.join(format!(
                "aged-1-1-v{format}-{}-{}",
                "a".repeat(64),
                "b".repeat(64)
            ));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("config.json"), "{}").unwrap();
            let path = dir.join(".complete");
            fs::write(
                &path,
                marker.replacen("format=8\n", &format!("format={format}\n"), 1),
            )
            .unwrap();
            set_mtime(&path, TWO_DAYS);
            dir
        };
        let (v6, v7, v8) = (entry(6), entry(7), entry(8));
        sweep_native_gguf_cache(&cache, "unrelated-key");
        assert!(!v6.exists(), "an aged v6 entry must be swept (control)");
        assert!(v7.is_dir(), "aged v7 entry swept before the cutover");
        assert!(v8.is_dir(), "aged v8 entry swept");
        native_gguf_unlease_in_use(&ckpt.asset_dir);
        fs::remove_dir_all(&work).ok();
    }

    enum Want {
        Gguf,
        Dir,
        Err(&'static str),
    }

    async fn check(path: &Path, fam: NativeGgufFamily, cache: &Path, want: Want, label: &str) {
        let got = CheckpointSource::resolve_in(path, fam, Some(cache)).await;
        match (got, want) {
            (Ok(CheckpointSource::Dir(dir)), Want::Dir) => assert_eq!(dir, path, "{label}"),
            (Ok(CheckpointSource::Gguf(ckpt)), Want::Gguf) => {
                assert!(
                    ckpt.asset_dir.join(GGUF_SOURCE_FILE).is_file(),
                    "{label}: asset dir has no {GGUF_SOURCE_FILE}"
                );
                if path.is_dir() && path.join(GGUF_SOURCE_FILE).is_file() {
                    assert_eq!(ckpt.asset_dir, path, "{label}: asset dir");
                    assert!(!ckpt.leased, "{label}: a caller dir is not leased");
                } else {
                    assert!(ckpt.leased, "{label}: prepared entry lease");
                    native_gguf_unlease_in_use(&ckpt.asset_dir);
                }
            }
            (Err(e), Want::Err(text)) => {
                assert!(e.reason.contains(text), "{label}: {}", e.reason)
            }
            (Ok(CheckpointSource::Dir(_)), _) => panic!("{label}: resolved to Dir"),
            (Ok(CheckpointSource::Gguf(_)), _) => panic!("{label}: resolved to Gguf"),
            (Err(e), _) => panic!("{label}: {}", e.reason),
        }
    }

    #[tokio::test]
    async fn checkpoint_source_resolve_table() {
        use NativeGgufFamily::*;
        let work = work_dir("resolve");
        let cache = work.join("cache");
        let plain = work.join("plain");
        fs::create_dir_all(&plain).unwrap();
        fs::write(plain.join("config.json"), "{}").unwrap();
        let mut qwen_assets = None;
        for (name, fam, dir_scanned) in [
            ("qwen35-dense-untied", Qwen35, false),
            ("qwen35moe-3d-experts", Qwen35Moe, false),
            ("gemma4-mmproj", Gemma4, true),
            ("muse-draft", MuseGlimmer, true),
        ] {
            let src = work.join("src").join(name);
            let fx = write_import_fixture(name, &src);
            check(&fx.main, fam, &cache, Want::Gguf, &format!("{name}: .gguf")).await;
            let ckpt = prepare_gguf_assets_in(&fx.main, fam, &cache).await.unwrap();
            check(
                &ckpt.asset_dir,
                fam,
                &cache,
                Want::Gguf,
                &format!("{name}: gguf-source dir"),
            )
            .await;
            let scan = if dir_scanned { Want::Gguf } else { Want::Dir };
            check(
                &src,
                fam,
                &cache,
                scan,
                &format!("{name}: dir with one .gguf"),
            )
            .await;
            check(
                &plain,
                fam,
                &cache,
                Want::Dir,
                &format!("{name}: plain dir"),
            )
            .await;
            if fam == Qwen35 {
                qwen_assets = Some((fx.main.clone(), src.clone(), ckpt.asset_dir.clone()));
            } else {
                native_gguf_unlease_in_use(&ckpt.asset_dir);
            }
        }
        // No CLEF fixture: a non-CLEF GGUF reaching the family's header check
        // shows which branch ran.
        let (qwen_gguf, qwen_src, qwen_assets) = qwen_assets.unwrap();
        let clef = "CLEF load requires";
        check(&qwen_gguf, Clef, &cache, Want::Err(clef), "clef: .gguf").await;
        check(
            &qwen_assets,
            Clef,
            &cache,
            Want::Err(clef),
            "clef: gguf-source dir",
        )
        .await;
        check(
            &qwen_src,
            Clef,
            &cache,
            Want::Dir,
            "clef: dir with one .gguf",
        )
        .await;
        check(&plain, Clef, &cache, Want::Dir, "clef: plain dir").await;
        native_gguf_unlease_in_use(&qwen_assets);
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn checkpoint_identity_matches_prepare() {
        let work = work_dir("identity");
        let fx = write_import_fixture("muse-draft", &work.join("src"));
        let ckpt =
            prepare_gguf_assets_in(&fx.main, NativeGgufFamily::MuseGlimmer, &work.join("cache"))
                .await
                .unwrap();
        let identity = |path: &Path| {
            let canonical = path.canonicalize().unwrap();
            let header = parse_gguf(&canonical).unwrap();
            file_identity(&canonical, &header, &fs::metadata(&canonical).unwrap()).unwrap()
        };
        let main = identity(&fx.main);
        let companion = identity(fx.companion.as_ref().unwrap());
        assert_eq!(ckpt.main.2, main);
        assert_eq!(ckpt.companion.as_ref().map(|c| &c.2), Some(&companion));
        let input = ckpt.preload_input(true);
        assert_eq!(input.main.2, main, "preload main identity");
        assert_eq!(
            input.companion.as_ref().map(|c| &c.2),
            Some(&companion),
            "preload companion identity"
        );
        // A later change must not leak into the snapshot taken at prepare.
        set_mtime(&fx.main, 3600);
        assert_ne!(identity(&fx.main), main, "mtime change kept the identity");
        let input = ckpt.preload_input(true);
        assert_eq!(input.main.2, main, "preload_input re-read the source");
        native_gguf_unlease_in_use(&ckpt.asset_dir);
        fs::remove_dir_all(&work).ok();
    }
}
