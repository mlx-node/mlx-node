//! One GGUF part's header. The shared parser keeps every key except bool
//! arrays: `read_meta_array` (utils/gguf.rs) skips element type 7 and yields
//! an empty `ArrayU32`, so `sliding_window_pattern` and `indexer.types` are
//! read here by a second pass over the key section.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use napi::{Error, Result};

use crate::moe_stream::io::open_nocache;
use crate::utils::gguf::{GgufFile, parse_gguf_reader};

const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// Same bound as the shared parser's `MAX_GGUF_ALLOC`.
const MAX_LEN: u64 = 256 * 1024 * 1024;
const MAX_NESTING: usize = 8;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 GGUF header: {msg}"))
}

pub struct Header {
    pub gguf: GgufFile,
    /// Every bool-array key of the part, by full key name.
    pub bools: HashMap<String, Vec<bool>>,
    pub file_len: u64,
}

impl Header {
    pub fn read(path: &Path) -> Result<Self> {
        let file = open_nocache(path)?;
        let file_len = file
            .metadata()
            .map_err(|e| bad(format!("'{}': {e}", path.display())))?
            .len();
        let mut r = BufReader::with_capacity(1 << 20, file);
        let (gguf, bools) =
            parse(&mut r).map_err(|e| bad(format!("'{}': {}", path.display(), e.reason)))?;
        Ok(Self {
            gguf,
            bools,
            file_len,
        })
    }
}

/// Both passes over one reader positioned anywhere.
pub fn parse<R: Read + Seek>(r: &mut R) -> Result<(GgufFile, HashMap<String, Vec<bool>>)> {
    r.seek(SeekFrom::Start(0))
        .map_err(|e| bad(format!("seek: {e}")))?;
    let gguf = parse_gguf_reader(&mut *r)?;
    r.seek(SeekFrom::Start(0))
        .map_err(|e| bad(format!("seek: {e}")))?;
    let bools = bool_arrays(r)?;
    Ok((gguf, bools))
}

/// Walks the key section of a header the shared parser already accepted.
fn bool_arrays(r: &mut impl Read) -> Result<HashMap<String, Vec<bool>>> {
    let mut fixed = [0u8; 24];
    read_exact(r, &mut fixed)?;
    let mut n_kv_bytes = [0u8; 8];
    n_kv_bytes.copy_from_slice(&fixed[16..24]);
    let n_kv = u64::from_le_bytes(n_kv_bytes);
    let mut out = HashMap::new();
    for _ in 0..n_kv {
        let key = read_string(r)?;
        let ty = read_u32(r)?;
        if ty != T_ARRAY {
            skip_value(r, ty, 0)?;
            continue;
        }
        let elem = read_u32(r)?;
        let len = read_len(r)?;
        if elem == T_BOOL {
            let mut raw = vec![0u8; len as usize];
            read_exact(r, &mut raw)?;
            out.insert(key, raw.into_iter().map(|b| b != 0).collect());
        } else {
            skip_array(r, elem, len, 1)?;
        }
    }
    Ok(out)
}

fn skip_value(r: &mut impl Read, ty: u32, depth: usize) -> Result<()> {
    match ty {
        T_STRING => {
            let len = read_len(r)?;
            skip(r, len)
        }
        T_ARRAY => {
            let elem = read_u32(r)?;
            let len = read_len(r)?;
            skip_array(r, elem, len, depth + 1)
        }
        _ => skip(r, scalar_size(ty)?),
    }
}

fn skip_array(r: &mut impl Read, elem: u32, len: u64, depth: usize) -> Result<()> {
    if depth > MAX_NESTING {
        return Err(bad("arrays nested too deep"));
    }
    match elem {
        T_STRING | T_ARRAY => {
            for _ in 0..len {
                skip_value(r, elem, depth)?;
            }
            Ok(())
        }
        _ => skip(r, len * scalar_size(elem)?),
    }
}

fn scalar_size(ty: u32) -> Result<u64> {
    match ty {
        T_U8 | T_I8 | T_BOOL => Ok(1),
        T_U16 | T_I16 => Ok(2),
        T_U32 | T_I32 | T_F32 => Ok(4),
        T_U64 | T_I64 | T_F64 => Ok(8),
        other => Err(bad(format!("unknown value type {other}"))),
    }
}

fn skip(r: &mut impl Read, n: u64) -> Result<()> {
    let copied = std::io::copy(&mut r.by_ref().take(n), &mut std::io::sink())
        .map_err(|e| bad(format!("read: {e}")))?;
    if copied != n {
        return Err(bad("header ends inside a value"));
    }
    Ok(())
}

fn read_exact(r: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| bad(format!("read: {e}")))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_len(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    let len = u64::from_le_bytes(b);
    if len > MAX_LEN {
        return Err(bad(format!("length {len} exceeds {MAX_LEN}")));
    }
    Ok(len)
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_len(r)?;
    let mut raw = vec![0u8; len as usize];
    read_exact(r, &mut raw)?;
    String::from_utf8(raw).map_err(|_| bad("key is not UTF-8"))
}
