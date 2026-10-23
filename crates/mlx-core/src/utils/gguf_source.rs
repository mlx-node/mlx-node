use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};
use sha2::{Digest, Sha256};

use crate::cold_tier::ShardIdentity;
use crate::utils::gguf::{GgufFile, source_file_identity_digest};

#[cfg(test)]
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
    pub main: GgufFileHandle,
    pub companion: Option<GgufFileHandle>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GgufFileIdentity {
    pub path: PathBuf,
    pub source_identity: String,
    pub header_digest: [u8; 32],
}

#[cfg(test)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GgufIdentity {
    pub files: Vec<GgufFileIdentity>,
    pub contract: &'static str,
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
        reject_split(&header, &canonical)?;
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
    pub(crate) fn open(
        main: (&Path, Arc<GgufFile>, &GgufFileIdentity),
        companion: Option<(&Path, Arc<GgufFile>, &GgufFileIdentity)>,
    ) -> Result<Self> {
        let main = GgufFileHandle::open(main.0, main.1, main.2)?;
        let companion = companion
            .map(|(path, header, identity)| GgufFileHandle::open(path, header, identity))
            .transpose()?;
        Ok(Self { main, companion })
    }

    pub(crate) fn recheck(&self) -> Result<()> {
        self.main.recheck()?;
        if let Some(companion) = &self.companion {
            companion.recheck()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn identity(&self) -> GgufIdentity {
        GgufIdentity {
            files: std::iter::once(&self.main)
                .chain(self.companion.as_ref())
                .map(|h| h.identity.clone())
                .collect(),
            contract: GGUF_IMPORT_CONTRACT,
        }
    }

    #[cfg(test)]
    pub(crate) fn data_bytes(&self) -> u64 {
        std::iter::once(&self.main)
            .chain(self.companion.as_ref())
            .map(|h| h.len.saturating_sub(h.header.data_offset))
            .sum()
    }
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
    let Some(value) = h.metadata.get("split.count") else {
        return Ok(());
    };
    match value.as_u32() {
        Some(1) => Ok(()),
        Some(n) if n > 1 => Err(Error::from_reason(format!(
            "GGUF '{}' is split into {n} files; split GGUFs are not supported. Merge them first \
             (llama-gguf-split --merge) and load the merged file.",
            path.display()
        ))),
        _ => Err(Error::from_reason(format!(
            "GGUF '{}' has an invalid split.count",
            path.display()
        ))),
    }
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
    use crate::utils::gguf_test_fixtures::build_minimal_gguf;

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
        let text = err_text(GgufSourceSet::open((&path, header, &id), None));
        assert!(text.contains("truncated"), "{text}");
    }

    #[test]
    fn split_gguf_rejected() {
        let dir = Temp::new();
        let path = dir.0.join("m-00001-of-00002.gguf");
        write_gguf(&path, &[("split.count", GgufMetaValue::Uint32(2))], 1.0);
        let (header, id) = identity_of(&path);
        let text = err_text(GgufSourceSet::open((&path, header.clone(), &id), None));
        assert!(text.contains("split into 2 files"), "{text}");
        assert!(reject_split(&header, &path).is_err());

        let one = dir.0.join("one.gguf");
        write_gguf(&one, &[("split.count", GgufMetaValue::Uint32(1))], 1.0);
        let (header, id) = identity_of(&one);
        GgufSourceSet::open((&one, header, &id), None).unwrap();
    }

    #[test]
    fn open_rejects_changed_source() {
        let dir = Temp::new();
        let path = dir.0.join("m.gguf");
        write_gguf(&path, &[], 1.0);
        let (header, id) = identity_of(&path);
        let set = GgufSourceSet::open((&path, header.clone(), &id), None).unwrap();
        set.recheck().unwrap();
        assert_eq!(set.identity().files, vec![id.clone()]);
        assert_eq!(set.identity().contract, GGUF_IMPORT_CONTRACT);
        drop(set);

        let other = dir.0.join("other.gguf");
        write_gguf(&other, &[], 7.0);
        fs::rename(&other, &path).unwrap();
        let text = err_text(GgufSourceSet::open((&path, header.clone(), &id), None));
        assert!(text.contains("changed since prepare"), "{text}");

        let (header, id) = identity_of(&path);
        let set = GgufSourceSet::open((&path, header, &id), None).unwrap();
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
        let set = GgufSourceSet::open((&path, header.clone(), &id), None).unwrap();
        assert_eq!(
            set.data_bytes(),
            fs::metadata(&path).unwrap().len() - header.data_offset
        );
    }
}
