//! Routed-expert streaming from GGUF shards on SSD: shard descriptors, a
//! persistent `pread` pool, and the CPU repack of MXFP4 extents straight into
//! expert bank slots. Family-neutral; the bank and the stores live with their
//! callers.

pub mod bank;
pub mod io;
pub mod repack;
