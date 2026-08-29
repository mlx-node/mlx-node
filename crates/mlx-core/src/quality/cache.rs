//! On-disk teacher cache for `mlx eval`.
//!
//! The reference runs ONCE; every candidate checkpoint is then scored
//! against what it wrote. One safetensors file per sequence plus a `meta.json`
//! describing the run.
//!
//! `tokens` is stored with the distribution and score mode reads its token ids
//! FROM THE CACHE — it never re-tokenizes the dataset. That is the only thing
//! that makes a stale cache impossible to misread: a dataset or tokenizer edit
//! cannot silently produce a comparison against different text. The remaining
//! staleness surface (a different teacher, or a candidate that cannot answer
//! for the cached ids) is covered by [`EvalCacheMeta`], which every score run
//! checks and echoes into its report. See [`EvalIdentity`] for why vocabulary
//! WIDTH alone is not enough.
//!
//! # One writer per cache directory
//!
//! A cache directory takes ONE capture at a time. Rows are written under fixed
//! names (`row_path`), so two captures running against the same `--cache` in
//! separate processes interleave their rows, and whichever calls `write_meta`
//! last publishes metadata describing a mixed row set.
//!
//! `mlx eval`'s guard is process-local, so it does not serialize that. What IS
//! covered: a SCORE overlapping a capture, because the capture removes
//! `meta.json` before its first row and stamps a fresh
//! [`EvalCacheMeta::generation`] after its last, and the score re-reads the
//! metadata after its row loop and refuses if either moved.
//!
//! Closing the capture-versus-capture case needs generation-scoped row paths
//! and an atomic publish, the shape `mlx-paged-attn`'s cold cache uses. That is
//! deliberately not built here: this is a single-user developer tool, two
//! concurrent captures mean two checkpoints resident at once, and this repo
//! takes no cross-process locks anywhere. Give each capture its own `--cache`
//! directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use sha2::{Digest, Sha256};

use crate::array::MxArray;
use crate::utils::safetensors::{SafeTensorsFile, save_safetensors};

use super::scoring::{TeacherLogits, TeacherRow};

/// What a checkpoint IS, for the purpose of deciding whether two of them can be
/// compared over the same cached token ids.
///
/// Vocabulary WIDTH is not identity. Score reuses the cached ids for both the
/// forward and the targets, and `score_chunk` indexes the candidate's logits
/// directly with the teacher's cached vocabulary indices — so under a different
/// id-to-token map the NLL is a real number measured on text the candidate
/// never saw, and the KL compares two distributions over different alphabets.
/// Both come out finite and plausible, which is what makes the mismatch worth
/// refusing rather than reporting.
///
/// The tokenizer digest is over the file bytes. `mlx convert` copies
/// `tokenizer.json` verbatim, so a quantized checkpoint still matches the bf16
/// teacher it came from — the pairing this tool exists for.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvalIdentity {
    pub model_type: String,
    pub tokenizer_sha256: String,
}

impl EvalIdentity {
    /// Read `<model_path>/config.json` and `<model_path>/tokenizer.json`.
    pub fn read(model_path: &str) -> Result<Self> {
        let model_type = crate::calibration::napi::read_model_type(model_path)?;
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let bytes = std::fs::read(&tokenizer_path)
            .map_err(|e| Error::from_reason(format!("read {}: {e}", tokenizer_path.display())))?;
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:eval-identity:v1\0");
        hasher.update(&bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        let mut tokenizer_sha256 = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(tokenizer_sha256, "{byte:02x}");
        }
        Ok(Self {
            model_type,
            tokenizer_sha256,
        })
    }

    /// A deterministic identity for tests, so a fixture cache and a fixture
    /// candidate match without a checkpoint on disk.
    #[cfg(test)]
    pub fn fixture(tag: &str) -> Self {
        Self {
            model_type: "qwen3_5".to_string(),
            tokenizer_sha256: format!("{tag:0>64}"),
        }
    }

    /// Reject a digest that is not 64 lowercase hex characters.
    ///
    /// `meta.json` is plain text on disk: a partial write or a hand edit can
    /// leave a short string here, and the mismatch message takes a prefix of
    /// it. Checked before that slice so a malformed cache is an error rather
    /// than a panic on the model thread.
    fn require_well_formed_digest(&self) -> Result<()> {
        let digest = &self.tokenizer_sha256;
        if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::from_reason(format!(
                "teacher cache metadata is malformed: identity.tokenizer_sha256 is not a \
                 64-character hex digest (got {} characters). Re-capture the cache.",
                digest.len()
            )));
        }
        Ok(())
    }

    /// Refuse a candidate the cached rows cannot answer for.
    pub fn require_match(&self, cached: &Self) -> Result<()> {
        if self.model_type != cached.model_type {
            return Err(Error::from_reason(format!(
                "teacher cache was captured on model_type \"{}\" but this checkpoint is \
                 \"{}\" — the two are not comparable",
                cached.model_type, self.model_type
            )));
        }
        if self.tokenizer_sha256 != cached.tokenizer_sha256 {
            // The cached digest comes off disk, so it can be truncated or hand
            // edited. Report that as the corrupt cache it is; slicing a prefix
            // out of it unchecked would panic on the model thread instead.
            cached.require_well_formed_digest()?;
            return Err(Error::from_reason(format!(
                "teacher cache was captured on a different tokenizer ({}... vs {}...): the \
                 cached token ids and top-K indices mean different tokens to this checkpoint, \
                 so every metric would be measured on the wrong text. Re-capture against this \
                 checkpoint's teacher.",
                &cached.tokenizer_sha256[..DIGEST_PREFIX],
                &self.tokenizer_sha256[..DIGEST_PREFIX]
            )));
        }
        Ok(())
    }
}

/// What the teacher capture was, so a score run can refuse a cache that cannot
/// answer for the candidate in front of it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalCacheMeta {
    pub teacher_path: String,
    /// What the teacher was, so a candidate that cannot be compared to it is
    /// refused rather than silently scored. See [`EvalIdentity`].
    pub identity: EvalIdentity,
    /// Unique to ONE capture. Rows are overwritten in place under fixed names,
    /// so a score that re-reads this metadata after its row loop cannot tell
    /// two captures apart by their contents — the same teacher over a different
    /// dataset produces identical fields. A fresh value per capture makes that
    /// re-read an identity check rather than a shape comparison.
    pub generation: String,
    /// The teacher was itself quantized. `mlx eval` does not refuse such a
    /// teacher — anchoring on a released quantized checkpoint is a real
    /// comparison — but every number then measures divergence from that
    /// checkpoint, not from the bf16 model, so the cache carries the fact.
    #[serde(default)]
    pub teacher_quantized: bool,
    pub vocab_size: i32,
    pub seq_len: u32,
    pub top_k: u32,
    pub rows: u32,
    pub positions: u64,
}

/// Characters of a digest shown in a mismatch message.
const DIGEST_PREFIX: usize = 12;

/// A value no other capture will produce: this process, and when it wrote.
pub fn new_generation() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}", std::process::id())
}

fn meta_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("meta.json")
}

fn row_path(cache_dir: &Path, index: u32) -> PathBuf {
    cache_dir.join(format!("{index:010}.safetensors"))
}

/// Drop the cache's identity before a capture writes its first row.
///
/// Rows are overwritten in place under fixed names, so a capture that dies
/// partway leaves this teacher's rows in front of the previous teacher's. A
/// surviving `meta.json` would let `score` read that mixture under the previous
/// teacher's identity, at the previous teacher's row count, and report a number
/// that looks valid. Without it, `read_meta` fails and says to re-capture.
pub fn invalidate_meta(cache_dir: &Path) -> Result<()> {
    match std::fs::remove_file(meta_path(cache_dir)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::from_reason(format!(
            "remove {}: {e}",
            meta_path(cache_dir).display()
        ))),
    }
}

pub fn write_meta(cache_dir: &Path, meta: &EvalCacheMeta) -> Result<()> {
    let body = serde_json::to_string_pretty(meta)
        .map_err(|e| Error::from_reason(format!("serialize eval cache meta: {e}")))?;
    std::fs::write(meta_path(cache_dir), body)
        .map_err(|e| Error::from_reason(format!("write {}: {e}", meta_path(cache_dir).display())))
}

pub fn read_meta(cache_dir: &Path) -> Result<EvalCacheMeta> {
    let path = meta_path(cache_dir);
    let body = std::fs::read_to_string(&path).map_err(|e| {
        Error::from_reason(format!(
            "read {}: {e} — run `mlx eval cache` first",
            path.display()
        ))
    })?;
    serde_json::from_str(&body)
        .map_err(|e| Error::from_reason(format!("parse {}: {e}", path.display())))
}

/// Write one captured sequence. Tensor dtypes are exactly what the reducers
/// produced — `indices`/`argmax` stay U32 and `tokens` is written as U32, so
/// nothing round-trips through a float on the way to disk.
pub fn write_row(cache_dir: &Path, index: u32, row: &TeacherRow) -> Result<()> {
    let mut tensors: HashMap<String, MxArray> = HashMap::new();
    tensors.insert(
        "tokens".to_string(),
        MxArray::from_uint32(&row.tokens, &[row.tokens.len() as i64])?,
    );
    tensors.insert("logits".to_string(), row.logits.logits.clone());
    tensors.insert("indices".to_string(), row.logits.indices.clone());
    tensors.insert("lse".to_string(), row.logits.lse.clone());
    tensors.insert(
        "target_logprob".to_string(),
        row.logits.target_logprob.clone(),
    );
    tensors.insert("argmax".to_string(), row.logits.argmax.clone());
    save_safetensors(row_path(cache_dir, index), &mut tensors, None)
}

pub fn read_row(cache_dir: &Path, index: u32) -> Result<TeacherRow> {
    let path = row_path(cache_dir, index);
    let file = SafeTensorsFile::load(&path)?;
    let mut tensors = file.load_tensors(&path)?;

    let mut take = |name: &str| -> Result<MxArray> {
        tensors.remove(name).ok_or_else(|| {
            Error::from_reason(format!("{}: teacher row has no \"{name}\"", path.display()))
        })
    };
    let tokens = take("tokens")?.to_uint32()?.to_vec();
    Ok(TeacherRow {
        logits: TeacherLogits {
            logits: take("logits")?,
            indices: take("indices")?,
            lse: take("lse")?,
            target_logprob: take("target_logprob")?,
            argmax: take("argmax")?,
        },
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mlx_eval_cache_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A cached row must come back with every tensor, at the dtype and shape the
    /// scorer expects, and with `tokens` byte-identical — the ids are the whole
    /// reason score mode never re-tokenizes.
    /// A capture that dies partway must leave nothing `score` will read. Rows
    /// keep fixed names and are overwritten in place, so a surviving `meta.json`
    /// would let the new teacher's early rows be scored alongside the old
    /// teacher's late ones, under the old teacher's identity.
    /// A truncated digest in `meta.json` is an error, not a panic.
    ///
    /// `meta.json` is plain text on disk. The mismatch message shows a prefix
    /// of the cached digest, so a short value there would panic on the model
    /// thread — killing the operation instead of reporting a corrupt cache.
    #[test]
    fn a_malformed_cached_digest_is_an_error_not_a_panic() {
        let candidate = EvalIdentity::fixture("a");
        for broken in ["", "abc", "zz".repeat(32).as_str()] {
            let cached = EvalIdentity {
                model_type: "qwen3_5".to_string(),
                tokenizer_sha256: broken.to_string(),
            };
            let Err(err) = candidate.require_match(&cached) else {
                panic!("a malformed digest {broken:?} must be refused");
            };
            assert!(
                err.reason.contains("64-character hex digest"),
                "message must name the cause: {}",
                err.reason
            );
        }

        // A well-formed digest still reports the ordinary mismatch.
        let Err(err) = candidate.require_match(&EvalIdentity::fixture("b")) else {
            panic!("a different well-formed digest must be refused");
        };
        assert!(
            err.reason.contains("different tokenizer"),
            "message must name the cause: {}",
            err.reason
        );
    }

    #[test]
    fn invalidate_meta_leaves_no_readable_cache() {
        let dir = scratch("invalidate");
        let meta = EvalCacheMeta {
            teacher_path: "/models/teacher-a".to_string(),
            identity: EvalIdentity::fixture("a"),
            generation: new_generation(),
            teacher_quantized: false,
            vocab_size: 32,
            seq_len: 8,
            top_k: 4,
            rows: 3,
            positions: 21,
        };
        write_meta(&dir, &meta).unwrap();
        read_meta(&dir).expect("meta must be readable before invalidation");

        invalidate_meta(&dir).expect("invalidation must succeed");
        assert!(
            read_meta(&dir).is_err(),
            "score must fail loudly on a cache whose capture did not finish"
        );
        // Idempotent: a first-ever capture has no meta to remove.
        invalidate_meta(&dir).expect("invalidation must tolerate a missing meta");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn teacher_row_survives_a_safetensors_round_trip() {
        let dir = scratch("roundtrip");
        let tokens: Vec<u32> = vec![7, 11, 13, 17, 19];
        let p = tokens.len() as i64 - 1;
        let k = 3i64;

        let row = TeacherRow {
            tokens: tokens.clone(),
            logits: TeacherLogits {
                logits: MxArray::from_float32(&vec![0.5f32; (p * k) as usize], &[p, k]).unwrap(),
                indices: MxArray::from_uint32(&vec![2u32; (p * k) as usize], &[p, k]).unwrap(),
                lse: MxArray::from_float32(&vec![1.25f32; p as usize], &[p]).unwrap(),
                target_logprob: MxArray::from_float32(&vec![-2.5f32; p as usize], &[p]).unwrap(),
                argmax: MxArray::from_uint32(&vec![4u32; p as usize], &[p]).unwrap(),
            },
        };
        write_row(&dir, 0, &row).unwrap();

        let back = read_row(&dir, 0).unwrap();
        assert_eq!(back.tokens, tokens, "token ids must survive byte-identical");
        assert_eq!(back.logits.positions().unwrap(), p);
        assert_eq!(back.logits.support().unwrap(), k);
        assert_eq!(back.logits.logits.dtype().unwrap(), DType::Float32);
        assert_eq!(
            back.logits.indices.dtype().unwrap(),
            DType::Uint32,
            "indices must stay U32 — an astype here would silently cost ids above 2^24"
        );
        assert_eq!(back.logits.argmax.dtype().unwrap(), DType::Uint32);
        assert_eq!(back.logits.lse.dtype().unwrap(), DType::Float32);
        assert_eq!(back.logits.target_logprob.to_float32().unwrap()[0], -2.5);

        let meta = EvalCacheMeta {
            teacher_path: "/models/teacher".to_string(),
            identity: EvalIdentity::fixture("a"),
            generation: new_generation(),
            teacher_quantized: false,
            vocab_size: 32,
            seq_len: 5,
            top_k: 3,
            rows: 1,
            positions: p as u64,
        };
        write_meta(&dir, &meta).unwrap();
        let meta_back = read_meta(&dir).unwrap();
        assert_eq!(meta_back.teacher_path, meta.teacher_path);
        assert_eq!(meta_back.vocab_size, 32);
        assert_eq!(meta_back.positions, p as u64);

        std::fs::remove_dir_all(&dir).ok();
    }
}
