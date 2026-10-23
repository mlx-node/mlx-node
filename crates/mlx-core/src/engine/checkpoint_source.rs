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
    resolve_gemma4_gguf_source, resolve_muse_glimmer_gguf_source, source_file_identity_digest,
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
        match gguf {
            Some(gguf) => Ok(Self::Gguf(Box::new(
                prepare_gguf_assets(&gguf, family).await?,
            ))),
            None => Ok(Self::Dir(path.to_path_buf())),
        }
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
