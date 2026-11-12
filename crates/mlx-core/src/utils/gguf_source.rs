use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};
use sha2::{Digest, Sha256};

use crate::cold_tier::ShardIdentity;
use crate::utils::gguf::{
    GgufFile, GgufMetaValue, resolve_gguf_shards, source_file_identity_digest,
};

pub(crate) const GGUF_IMPORT_CONTRACT: &str = "gguf-import:v8";

pub(crate) struct GgufFileHandle {
    pub path: PathBuf,
    pub file: fs::File,
    pub header: Arc<GgufFile>,
    pub len: u64,
    pub stat: ShardIdentity,
    pub identity: GgufFileIdentity,
}

pub(crate) struct GgufSourceSet {
    /// Every part of the main GGUF, in part order.
    pub main: Vec<GgufFileHandle>,
    pub companion: Option<GgufFileHandle>,
}

/// One GGUF file a set opens: path, its own header, the identity it was
/// prepared under.
pub(crate) type GgufPartRef<'a> = (&'a Path, Arc<GgufFile>, &'a GgufFileIdentity);

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GgufFileIdentity {
    pub path: PathBuf,
    pub source_identity: String,
    pub header_digest: [u8; 32],
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GgufIdentity {
    pub files: Vec<GgufFileIdentity>,
    pub contract: &'static str,
}

impl GgufIdentity {
    pub(crate) fn new(main: &[GgufFileIdentity], companion: Option<&GgufFileIdentity>) -> Self {
        Self {
            files: main.iter().chain(companion).cloned().collect(),
            contract: GGUF_IMPORT_CONTRACT,
        }
    }
}

fn changed_since_prepare(path: &Path) -> Error {
    Error::from_reason(format!(
        "GGUF '{}' changed since prepare; retry the load",
        path.display()
    ))
}

impl GgufFileHandle {
    fn open(path: &Path, header: Arc<GgufFile>, expected: &GgufFileIdentity) -> Result<Self> {
        let canonical = fs::canonicalize(path).map_err(|e| {
            Error::from_reason(format!("Failed to open GGUF '{}': {e}", path.display()))
        })?;
        if canonical != expected.path {
            return Err(changed_since_prepare(&canonical));
        }
        let file = fs::File::open(&canonical).map_err(|e| {
            Error::from_reason(format!(
                "Failed to open GGUF '{}': {e}",
                canonical.display()
            ))
        })?;
        let meta = file.metadata().map_err(|e| {
            Error::from_reason(format!(
                "Failed to stat GGUF '{}': {e}",
                canonical.display()
            ))
        })?;
        if source_file_identity_digest(&canonical, &meta) != expected.source_identity {
            return Err(changed_since_prepare(&canonical));
        }
        check_tensor_bounds(&header, meta.len(), &canonical)?;
        Ok(Self {
            path: canonical,
            len: meta.len(),
            stat: ShardIdentity::from_metadata(&meta),
            identity: expected.clone(),
            header,
            file,
        })
    }

    fn recheck(&self) -> Result<()> {
        let meta = self.file.metadata().map_err(|e| {
            Error::from_reason(format!(
                "Failed to stat GGUF '{}': {e}",
                self.path.display()
            ))
        })?;
        if ShardIdentity::from_metadata(&meta) != self.stat
            || source_file_identity_digest(&self.path, &meta) != self.identity.source_identity
        {
            return Err(Error::from_reason(format!(
                "GGUF '{}' changed during the load; retry the load",
                self.path.display()
            )));
        }
        Ok(())
    }
}

impl GgufSourceSet {
    /// The main parts must form one GGUF (Splash `GgufFile` over every part);
    /// the companion is one file.
    pub(crate) fn open(
        main: &[GgufPartRef<'_>],
        companion: Option<GgufPartRef<'_>>,
    ) -> Result<Self> {
        match main {
            [] => return Err(Error::from_reason("a GGUF needs at least one file")),
            [(path, header, _)] => reject_split(header, path)?,
            parts => check_split_parts(
                &parts
                    .iter()
                    .map(|(path, header, _)| (*path, &**header))
                    .collect::<Vec<_>>(),
            )?,
        }
        if let Some((path, header, _)) = &companion {
            reject_split(header, path)?;
        }
        let main = main
            .iter()
            .map(|(path, header, identity)| GgufFileHandle::open(path, header.clone(), identity))
            .collect::<Result<Vec<_>>>()?;
        let companion = companion
            .map(|(path, header, identity)| GgufFileHandle::open(path, header, identity))
            .transpose()?;
        Ok(Self { main, companion })
    }

    pub(crate) fn recheck(&self) -> Result<()> {
        for part in &self.main {
            part.recheck()?;
        }
        if let Some(companion) = &self.companion {
            companion.recheck()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn identity(&self) -> GgufIdentity {
        GgufIdentity::new(
            &self
                .main
                .iter()
                .map(|h| h.identity.clone())
                .collect::<Vec<_>>(),
            self.companion.as_ref().map(|h| &h.identity),
        )
    }

    #[cfg(test)]
    pub(crate) fn data_bytes(&self) -> u64 {
        self.main
            .iter()
            .chain(self.companion.as_ref())
            .map(|h| h.len.saturating_sub(h.header.data_offset))
            .sum()
    }
}

/// The parts of the GGUF `path` names, in part order and canonical: `path`
/// alone, or the first part of `NAME-0000K-of-0000N.gguf` and its siblings
/// found by name beside it (before any link is resolved).
pub(crate) fn resolve_main_parts(path: &Path) -> Result<Vec<PathBuf>> {
    let canonical = |p: &Path| {
        fs::canonicalize(p).map_err(|e| {
            Error::from_reason(format!("Failed to resolve GGUF '{}': {e}", p.display()))
        })
    };
    let header = super::gguf::parse_gguf(path)?;
    let count = split_key(&header, "split.count").unwrap_or(1);
    if count <= 1 {
        return Ok(vec![canonical(path)?]);
    }
    let Some((first, _)) = split_names(path, count) else {
        return Err(lone_split_part(&header, path, count));
    };
    let part = split_key(&header, "split.no").unwrap_or(0);
    if part != 0 {
        return Err(Error::from_reason(format!(
            "GGUF '{}' is part {} of a split GGUF of {count} parts; pass its first part '{}'",
            path.display(),
            part.saturating_add(1),
            first.display()
        )));
    }
    let count = u32::try_from(count).map_err(|_| {
        Error::from_reason(format!("GGUF '{}': invalid split.count", path.display()))
    })?;
    resolve_gguf_shards(path, count)
        .map_err(|e| {
            Error::from_reason(format!(
                "GGUF '{}' is part 1 of a split GGUF of {count} parts and the set is \
                 incomplete: {}",
                path.display(),
                e.reason
            ))
        })?
        .iter()
        .map(|p| canonical(p))
        .collect()
}

/// Splash `GgufFile` constructor over several parts: part 1's `split.count`
/// is the number of parts, a later part's `split.no` (U16 or U32 when
/// present) is its place, `split.tensors.count` is the tensors of all parts,
/// and no tensor name repeats.
pub(crate) fn check_split_parts(parts: &[(&Path, &GgufFile)]) -> Result<()> {
    let [(first_path, first), rest @ ..] = parts else {
        return Err(Error::from_reason("a GGUF needs at least one file"));
    };
    let count = split_key(first, "split.count").unwrap_or(1);
    if count != parts.len() as u64 {
        return Err(Error::from_reason(format!(
            "GGUF '{}': the GGUF's split.count does not match its {} files",
            first_path.display(),
            parts.len()
        )));
    }
    for (place, (path, header)) in rest.iter().enumerate() {
        let place = place as u64 + 1;
        let number = match header.metadata.get("split.no") {
            None => continue,
            Some(GgufMetaValue::Uint16(v)) => u64::from(*v),
            Some(GgufMetaValue::Uint32(v)) => u64::from(*v),
            Some(_) => {
                return Err(Error::from_reason(format!(
                    "GGUF '{}': invalid split.no",
                    path.display()
                )));
            }
        };
        if number != place {
            return Err(Error::from_reason(format!(
                "GGUF split file '{}' is out of order",
                path.display()
            )));
        }
    }
    let total: u64 = parts.iter().map(|(_, h)| h.tensors.len() as u64).sum();
    if split_key(first, "split.tensors.count").unwrap_or(total) != total {
        return Err(Error::from_reason(format!(
            "GGUF '{}': the GGUF's split files hold another number of tensors than \
             split.tensors.count",
            first_path.display()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for (_, header) in parts {
        for tensor in &header.tensors {
            if !seen.insert(tensor.name.as_str()) {
                return Err(Error::from_reason(format!(
                    "duplicate GGUF tensor: {}",
                    tensor.name
                )));
            }
        }
    }
    Ok(())
}

/// The one header the plan reads for a GGUF of `parts`: part 1's metadata
/// without the `split.*` keys and the tensors of every part in part order.
/// No byte is read through it; each part's reads use that part's header.
pub(crate) fn merge_split_headers(parts: &[(&Path, &GgufFile)]) -> Result<GgufFile> {
    match parts {
        [(path, header)] => reject_split(header, path)?,
        parts => check_split_parts(parts)?,
    }
    let (_, first) = parts[0];
    let mut metadata = first.metadata.clone();
    if parts.len() > 1 {
        metadata.retain(|key, _| !key.starts_with("split."));
    }
    let tensors: Vec<_> = parts
        .iter()
        .flat_map(|(_, h)| h.tensors.iter().cloned())
        .collect();
    Ok(GgufFile {
        version: first.version,
        tensor_count: tensors.len() as u64,
        metadata,
        tensors,
        alignment: first.alignment,
        data_offset: first.data_offset,
    })
}

pub(crate) fn check_tensor_bounds(h: &GgufFile, file_len: u64, path: &Path) -> Result<()> {
    if h.data_offset > file_len {
        return Err(Error::from_reason(format!(
            "GGUF '{}' is truncated: tensor data starts at {} but the file holds {file_len} bytes",
            path.display(),
            h.data_offset
        )));
    }
    for t in &h.tensors {
        let size = t.data_size()?;
        let end = h
            .data_offset
            .checked_add(t.offset)
            .and_then(|start| start.checked_add(size))
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "GGUF '{}' tensor '{}' offset {} + {size} bytes overflows u64",
                    path.display(),
                    t.name,
                    t.offset
                ))
            })?;
        if end > file_len {
            return Err(Error::from_reason(format!(
                "GGUF '{}' is truncated: tensor '{}' ends at byte {end} but the file holds {file_len} bytes",
                path.display(),
                t.name
            )));
        }
    }
    Ok(())
}

pub(crate) fn reject_split(h: &GgufFile, path: &Path) -> Result<()> {
    let count = split_key(h, "split.count").unwrap_or(1);
    if count > 1 {
        return Err(lone_split_part(h, path, count));
    }
    // split.count 0 is what llama-gguf-split --merge writes into its one-file output.
    if let Some(total) = split_key(h, "split.tensors.count")
        && total != h.tensors.len() as u64
    {
        return Err(Error::from_reason(format!(
            "GGUF '{}' holds {} tensors but its split.tensors.count is {total}; the file is incomplete",
            path.display(),
            h.tensors.len()
        )));
    }
    Ok(())
}

/// A metadata integer as Splash's `GgufFile` keeps it in `unsigneds`: signed
/// values cast, a bool as 0 or 1, any other type absent.
fn split_key(h: &GgufFile, key: &str) -> Option<u64> {
    Some(match h.metadata.get(key)? {
        GgufMetaValue::Uint8(v) => u64::from(*v),
        GgufMetaValue::Uint16(v) => u64::from(*v),
        GgufMetaValue::Uint32(v) => u64::from(*v),
        GgufMetaValue::Uint64(v) => *v,
        GgufMetaValue::Int8(v) => *v as u64,
        GgufMetaValue::Int16(v) => *v as u64,
        GgufMetaValue::Int32(v) => *v as u64,
        GgufMetaValue::Int64(v) => *v as u64,
        GgufMetaValue::Bool(v) => u64::from(*v),
        _ => return None,
    })
}

fn lone_split_part(h: &GgufFile, path: &Path, count: u64) -> Error {
    let part = split_key(h, "split.no").unwrap_or(0).saturating_add(1);
    let mut text = format!(
        "GGUF '{}' is part {part} of a split GGUF of {count} parts. This path reads one GGUF \
         file, so this part alone would miss the other parts' tensors.",
        path.display()
    );
    match split_names(path, count) {
        Some((first, merged)) => {
            match resolve_gguf_shards(&first, u32::try_from(count).unwrap_or(u32::MAX)) {
                Ok(parts) if first.is_file() => {
                    let names: Vec<String> = parts
                        .iter()
                        .map(|p| {
                            p.file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .into_owned()
                        })
                        .collect();
                    text += &format!(" The parts: {}.", names.join(", "));
                }
                Ok(_) => text += &format!(" Its first part '{}' is missing.", first.display()),
                Err(e) => text += &format!(" The set is incomplete: {}.", e.reason),
            }
            text += &format!(
                " Merge the parts into one file and pass that file: \
                 llama-gguf-split --merge '{}' '{}'",
                first.display(),
                merged.display()
            );
        }
        None => {
            text += " Its file name is not NAME-0000K-of-0000N.gguf, so its other parts cannot \
                     be found by name. Merge the parts with llama-gguf-split --merge and pass \
                     the merged file."
        }
    }
    Error::from_reason(text)
}

/// The first part and a merged-file name for a part named `NAME-0000K-of-0000N.gguf`.
fn split_names(path: &Path, count: u64) -> Option<(PathBuf, PathBuf)> {
    let name = path.file_name()?.to_str()?;
    let (stem, extension) = name.rsplit_once('.')?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, number) = head.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !extension.eq_ignore_ascii_case("gguf") || !digits(number) || total != format!("{count:05}")
    {
        return None;
    }
    Some((
        path.with_file_name(format!("{prefix}-00001-of-{total}.{extension}")),
        path.with_file_name(format!("{prefix}.{extension}")),
    ))
}

/// `NAME-0000K-of-0000N.gguf` (any `.gguf` case) as `(NAME, K, N)`.
pub(crate) fn split_part(name: &str) -> Option<(&str, u64, u64)> {
    let (stem, extension) = name.rsplit_once('.')?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, number) = head.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !extension.eq_ignore_ascii_case("gguf") || !digits(number) || !digits(total) {
        return None;
    }
    Some((prefix, number.parse().ok()?, total.parse().ok()?))
}

/// Splash `findTargetGgufs`: the GGUF of a target directory, one file or the
/// parts `NAME-0000K-of-0000N.gguf` of one NAME, K = 1..N, in part order.
pub(crate) fn find_target_ggufs(dir: &Path) -> Result<Vec<PathBuf>> {
    let listing = fs::read_dir(dir).map_err(|e| {
        Error::from_reason(format!(
            "cannot list target directory {}: {e}",
            dir.display()
        ))
    })?;
    let mut found = Vec::new();
    for entry in listing {
        let path = entry
            .map_err(|e| {
                Error::from_reason(format!(
                    "cannot list target directory {}: {e}",
                    dir.display()
                ))
            })?
            .path();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            found.push(path);
        }
    }
    if found.is_empty() {
        return Err(Error::from_reason(format!(
            "target directory holds no GGUF: {}",
            dir.display()
        )));
    }
    if found.len() == 1 {
        return Ok(found);
    }
    found.sort();
    let not_split = || {
        Error::from_reason(format!(
            "target directory holds more than one GGUF, and not the parts of one split GGUF: {}",
            dir.display()
        ))
    };
    let mut parts = Vec::with_capacity(found.len());
    for path in &found {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(not_split)?;
        parts.push(split_part(name).ok_or_else(not_split)?);
    }
    let names: std::collections::BTreeSet<&str> = parts.iter().map(|(name, _, _)| *name).collect();
    if names.len() > 1 {
        return Err(Error::from_reason(format!(
            "target directory mixes the parts of several split GGUFs ({}); a split GGUF is the \
             parts of one name, NAME-0000K-of-0000N.gguf: {}",
            names.into_iter().collect::<Vec<_>>().join(", "),
            dir.display()
        )));
    }
    for (index, (_, number, count)) in parts.iter().enumerate() {
        if *number != index as u64 + 1 || *count != parts.len() as u64 {
            return Err(not_split());
        }
    }
    Ok(found)
}

pub(crate) fn file_identity(
    path: &Path,
    header: &GgufFile,
    meta: &fs::Metadata,
) -> Result<GgufFileIdentity> {
    let canonical = fs::canonicalize(path).map_err(|e| {
        Error::from_reason(format!("Failed to resolve GGUF '{}': {e}", path.display()))
    })?;
    let header_len = usize::try_from(header.data_offset).map_err(|_| {
        Error::from_reason(format!(
            "GGUF '{}' header size {} does not fit in memory",
            canonical.display(),
            header.data_offset
        ))
    })?;
    if header.data_offset > meta.len() {
        return Err(Error::from_reason(format!(
            "GGUF '{}' is truncated: header ends at {} but the file holds {} bytes",
            canonical.display(),
            header.data_offset,
            meta.len()
        )));
    }
    let mut bytes = vec![0u8; header_len];
    fs::File::open(&canonical)
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(|e| {
            Error::from_reason(format!(
                "Failed to read GGUF header '{}': {e}",
                canonical.display()
            ))
        })?;
    Ok(GgufFileIdentity {
        source_identity: source_file_identity_digest(&canonical, meta),
        header_digest: Sha256::digest(&bytes).into(),
        path: canonical,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gguf::{GgufMetaValue, GgufTensorType, parse_gguf};
    use crate::utils::gguf_test_fixtures::{
        build_minimal_gguf, write_merged_gguf, write_split_gguf,
    };

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("gguf-source-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn f32_bytes(n: usize, seed: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed + i as f32).to_le_bytes())
            .collect()
    }

    fn write_gguf(path: &Path, metadata: &[(&str, GgufMetaValue)], seed: f32) {
        let a = f32_bytes(64, seed);
        let b = f32_bytes(48, seed + 100.0);
        let bytes = build_minimal_gguf(
            metadata,
            &[
                ("a.weight", &[8, 8], GgufTensorType::F32, &a),
                ("b.weight", &[48], GgufTensorType::F32, &b),
            ],
        );
        fs::write(path, bytes).unwrap();
    }

    fn identity_of(path: &Path) -> (Arc<GgufFile>, GgufFileIdentity) {
        let header = Arc::new(parse_gguf(path).unwrap());
        let meta = fs::metadata(path).unwrap();
        let id = file_identity(path, &header, &meta).unwrap();
        (header, id)
    }

    fn err_text<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.reason.clone(),
        }
    }

    #[test]
    fn truncated_file_rejected_at_open() {
        let dir = Temp::new();
        let path = dir.0.join("m.gguf");
        write_gguf(&path, &[], 1.0);
        let header = parse_gguf(&path).unwrap();
        let last_end = header
            .tensors
            .iter()
            .map(|t| header.data_offset + t.offset + t.data_size().unwrap())
            .max()
            .unwrap();
        check_tensor_bounds(&header, last_end, &path).unwrap();
        let text = err_text(check_tensor_bounds(&header, last_end - 1, &path));
        assert!(text.contains("truncated"), "{text}");

        let full = fs::read(&path).unwrap();
        fs::write(&path, &full[..usize::try_from(last_end - 1).unwrap()]).unwrap();
        let header = Arc::new(header);
        let meta = fs::metadata(&path).unwrap();
        let id = file_identity(&path, &header, &meta).unwrap();
        let text = err_text(GgufSourceSet::open(&[(&path, header, &id)], None));
        assert!(text.contains("truncated"), "{text}");
    }

    const TINY: [(&str, &[u64]); 3] = [("a.weight", &[4]), ("b.weight", &[4]), ("c.weight", &[4])];
    const TINY_NAMES: [&str; 3] = [
        "tiny-00001-of-00003.gguf",
        "tiny-00002-of-00003.gguf",
        "tiny-00003-of-00003.gguf",
    ];

    fn tiny_metadata() -> Vec<(&'static str, GgufMetaValue)> {
        vec![(
            "general.architecture",
            GgufMetaValue::String("qwen35".to_string()),
        )]
    }

    fn split_err(path: &Path) -> String {
        err_text(reject_split(&parse_gguf(path).unwrap(), path))
    }

    #[test]
    fn a_lone_split_part_names_its_parts() {
        let dir = Temp::new();
        let parts = write_split_gguf(&dir.0, "tiny", &tiny_metadata(), &TINY, true);

        let text = split_err(&parts[0]);
        assert!(text.contains("part 1 of a split GGUF of 3 parts"), "{text}");
        for name in TINY_NAMES {
            assert!(text.contains(name), "{text}");
        }
        assert!(text.contains("llama-gguf-split --merge"), "{text}");
        let (header, id) = identity_of(&parts[0]);
        let text = err_text(GgufSourceSet::open(&[(&parts[0], header, &id)], None));
        for name in TINY_NAMES {
            assert!(text.contains(name), "load seam: {text}");
        }

        let text = split_err(&parts[1]);
        assert!(text.contains("part 2 of a split GGUF of 3 parts"), "{text}");
        assert!(text.contains(TINY_NAMES[0]), "{text}");

        let lonely = dir.0.join("lonely.gguf");
        fs::copy(&parts[0], &lonely).unwrap();
        let text = split_err(&lonely);
        assert!(text.contains("3 parts"), "{text}");
        assert!(text.contains("NAME-0000K-of-0000N"), "{text}");

        fs::remove_file(&parts[2]).unwrap();
        let text = split_err(&parts[0]);
        assert!(text.contains("tiny-00003-of-00003"), "{text}");
    }

    #[test]
    fn a_merged_gguf_reads_as_one_file() {
        let dir = Temp::new();
        let merged = dir.0.join("tiny.gguf");
        write_merged_gguf(&merged, &tiny_metadata(), &TINY);
        let (header, id) = identity_of(&merged);
        reject_split(&header, &merged).unwrap();
        GgufSourceSet::open(&[(&merged, header, &id)], None).unwrap();

        let plain = dir.0.join("plain.gguf");
        write_gguf(&plain, &[], 1.0);
        reject_split(&parse_gguf(&plain).unwrap(), &plain).unwrap();

        let short = dir.0.join("short.gguf");
        write_gguf(
            &short,
            &[
                ("split.count", GgufMetaValue::Uint32(1)),
                ("split.tensors.count", GgufMetaValue::Int32(5)),
            ],
            1.0,
        );
        let text = split_err(&short);
        assert!(text.contains("split.tensors.count is 5"), "{text}");

        // Splash keeps only integer keys: a String split.count is absent, an Int16 one is read.
        let text_count = dir.0.join("text-count.gguf");
        write_gguf(
            &text_count,
            &[("split.count", GgufMetaValue::String("3".to_string()))],
            1.0,
        );
        reject_split(&parse_gguf(&text_count).unwrap(), &text_count).unwrap();
        let short_count = dir.0.join("short-count.gguf");
        write_gguf(
            &short_count,
            &[("split.count", GgufMetaValue::Int16(3))],
            1.0,
        );
        let text = split_err(&short_count);
        assert!(text.contains("of 3 parts"), "{text}");
        let short_total = dir.0.join("short-total.gguf");
        write_gguf(
            &short_total,
            &[("split.tensors.count", GgufMetaValue::Int16(5))],
            1.0,
        );
        let text = split_err(&short_total);
        assert!(text.contains("split.tensors.count is 5"), "{text}");
    }

    #[test]
    fn open_rejects_changed_source() {
        let dir = Temp::new();
        let path = dir.0.join("m.gguf");
        write_gguf(&path, &[], 1.0);
        let (header, id) = identity_of(&path);
        let set = GgufSourceSet::open(&[(&path, header.clone(), &id)], None).unwrap();
        set.recheck().unwrap();
        assert_eq!(set.identity().files, vec![id.clone()]);
        assert_eq!(set.identity().contract, GGUF_IMPORT_CONTRACT);
        drop(set);

        let other = dir.0.join("other.gguf");
        write_gguf(&other, &[], 7.0);
        fs::rename(&other, &path).unwrap();
        let text = err_text(GgufSourceSet::open(&[(&path, header.clone(), &id)], None));
        assert!(text.contains("changed since prepare"), "{text}");

        let (header, id) = identity_of(&path);
        let set = GgufSourceSet::open(&[(&path, header, &id)], None).unwrap();
        {
            use std::io::Write;
            let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0u8; 16]).unwrap();
        }
        let text = err_text(set.recheck());
        assert!(text.contains("changed during the load"), "{text}");
    }

    #[test]
    fn source_identity_changes_on_touch_and_len() {
        let dir = Temp::new();
        let path = dir.0.join("m.gguf");
        write_gguf(&path, &[], 1.0);
        let (header, before) = identity_of(&path);

        std::thread::sleep(std::time::Duration::from_millis(20));
        let f = fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_modified(std::time::SystemTime::now()).unwrap();
        drop(f);
        let touched = file_identity(&path, &header, &fs::metadata(&path).unwrap()).unwrap();
        assert_ne!(before.source_identity, touched.source_identity);
        assert_eq!(before.header_digest, touched.header_digest);

        {
            use std::io::Write;
            let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0u8; 16]).unwrap();
        }
        let appended = file_identity(&path, &header, &fs::metadata(&path).unwrap()).unwrap();
        assert_ne!(touched.source_identity, appended.source_identity);
    }

    #[test]
    fn data_bytes_counts_payload_after_header() {
        let dir = Temp::new();
        let path = dir.0.join("m.gguf");
        write_gguf(&path, &[], 1.0);
        let (header, id) = identity_of(&path);
        let set = GgufSourceSet::open(&[(&path, header.clone(), &id)], None).unwrap();
        assert_eq!(
            set.data_bytes(),
            fs::metadata(&path).unwrap().len() - header.data_offset
        );
    }

    fn check_files(paths: &[PathBuf]) -> Result<()> {
        let headers: Vec<GgufFile> = paths.iter().map(|p| parse_gguf(p).unwrap()).collect();
        check_split_parts(
            &paths
                .iter()
                .map(PathBuf::as_path)
                .zip(&headers)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn split_parts_follow_the_gguf_file_rules() {
        use crate::utils::gguf_test_fixtures::GgufParts;
        let dir = Temp::new();
        let parts = write_split_gguf(&dir.0, "tiny", &tiny_metadata(), &TINY, true);
        let canonical: Vec<PathBuf> = parts.iter().map(|p| p.canonicalize().unwrap()).collect();
        assert_eq!(resolve_main_parts(&parts[0]).unwrap(), canonical);
        check_files(&parts).unwrap();
        let merged = {
            let headers: Vec<GgufFile> = parts.iter().map(|p| parse_gguf(p).unwrap()).collect();
            merge_split_headers(
                &parts
                    .iter()
                    .map(PathBuf::as_path)
                    .zip(&headers)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        assert_eq!(merged.tensors.len(), 3);
        assert!(merged.metadata.keys().all(|k| !k.starts_with("split.")));
        reject_split(&merged, &parts[0]).unwrap();

        let text = err_text(resolve_main_parts(&parts[1]));
        assert!(text.contains("pass its first part"), "{text}");
        assert!(text.contains(TINY_NAMES[0]), "{text}");

        // A Hugging Face snapshot links each part to a blob of another name.
        let blobs = dir.0.join("blobs");
        let snapshot = dir.0.join("snapshot");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        for (i, part) in parts.iter().enumerate() {
            let blob = blobs.join(format!("blob{i}"));
            fs::copy(part, &blob).unwrap();
            std::os::unix::fs::symlink(&blob, snapshot.join(TINY_NAMES[i])).unwrap();
        }
        let linked = resolve_main_parts(&snapshot.join(TINY_NAMES[0])).unwrap();
        assert_eq!(linked.len(), 3);
        assert_eq!(linked[2], blobs.join("blob2").canonicalize().unwrap());

        let edit = |part: usize, change: &dyn Fn(&mut GgufParts)| -> Vec<PathBuf> {
            let case = dir.0.join(format!("case-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&case).unwrap();
            parts
                .iter()
                .enumerate()
                .map(|(i, src)| {
                    let mut file = GgufParts::read(src);
                    if i == part {
                        change(&mut file);
                    }
                    let path = case.join(TINY_NAMES[i]);
                    file.write(&path);
                    path
                })
                .collect()
        };
        let swapped = {
            let mut set = edit(1, &|p| p.set("split.no", GgufMetaValue::Uint16(2)));
            let third = edit(2, &|p| p.set("split.no", GgufMetaValue::Uint16(1)));
            set[2] = third[2].clone();
            set
        };
        let text = err_text(check_files(&swapped));
        assert!(text.contains("is out of order"), "{text}");

        let duplicate = edit(1, &|p| p.tensors[0].0 = "a.weight".into());
        let text = err_text(check_files(&duplicate));
        assert!(text.contains("duplicate GGUF tensor: a.weight"), "{text}");

        let short = edit(0, &|p| {
            p.set("split.tensors.count", GgufMetaValue::Int32(4))
        });
        let text = err_text(check_files(&short));
        assert!(text.contains("split.tensors.count"), "{text}");

        let wide = edit(1, &|p| p.set("split.no", GgufMetaValue::Int32(1)));
        let text = err_text(check_files(&wide));
        assert!(text.contains("invalid split.no"), "{text}");

        let unnumbered = edit(1, &|p| p.metadata.retain(|(k, _)| k != "split.no"));
        check_files(&unnumbered).unwrap();

        let text = err_text(check_files(&parts[..2]));
        assert!(
            text.contains("split.count does not match its 2 files"),
            "{text}"
        );

        let plain = dir.0.join("plain.gguf");
        write_gguf(&plain, &[], 1.0);
        let (main_header, main_id) = identity_of(&plain);
        let (part_header, part_id) = identity_of(&parts[0]);
        let text = err_text(GgufSourceSet::open(
            &[(&plain, main_header, &main_id)],
            Some((&parts[0], part_header, &part_id)),
        ));
        assert!(
            text.contains("part 1 of a split GGUF of 3 parts"),
            "companion: {text}"
        );

        fs::remove_file(&parts[2]).unwrap();
        let text = err_text(resolve_main_parts(&parts[0]));
        assert!(text.contains("tiny-00003-of-00003"), "{text}");
    }
}
