//! Strict checkpoint metadata shared by reranker backends. Family-specific
//! tensor names, projector layouts and shape contracts stay in their adapters.

use napi::{Error, Result};
use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, Visitor},
};
use serde_json::Value;
use std::{collections::HashMap, fmt, io::Read, marker::PhantomData, path::Path};

fn invalid(message: impl fmt::Display) -> Error {
    Error::from_reason(format!("rerank_invalid_model: {message}"))
}

// serde's default maps accept repeated JSON keys. Check before MLX's lazy
// reader so duplicate tensors or index entries cannot select the last value.
pub(super) struct UniqueMap<V>(pub(super) HashMap<String, V>);
impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct MapVisitor<V>(PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for MapVisitor<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut out = HashMap::new();
                while let Some((key, value)) = map.next_entry::<String, V>()? {
                    if out.insert(key.clone(), value).is_some() {
                        return Err(M::Error::custom(format!("duplicate key {key}")));
                    }
                }
                Ok(UniqueMap(out))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

pub(super) fn validate_header(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path).map_err(invalid)?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(invalid)?;
    let size = u64::from_le_bytes(size);
    // The safetensors specification's header limit, independent of model size.
    if size > 100_000_000 || size > file.metadata().map_err(invalid)?.len().saturating_sub(8) {
        return Err(invalid("invalid safetensors header length"));
    }
    let mut bytes = vec![0; size as usize];
    file.read_exact(&mut bytes).map_err(invalid)?;
    serde_json::from_slice::<UniqueMap<Value>>(&bytes).map_err(invalid)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_header_and_index_keys_are_rejected() {
        assert!(serde_json::from_str::<UniqueMap<Value>>(r#"{"x":{},"x":{}}"#).is_err());
        assert!(serde_json::from_str::<UniqueMap<String>>(r#"{"x":"a","x":"b"}"#).is_err());
        assert!(serde_json::from_str::<UniqueMap<String>>(r#"{"x":"a","y":"b"}"#).is_ok());
    }
}
