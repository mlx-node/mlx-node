//! Re-export shim for [`crate::disk_tier::store`].

pub(crate) use crate::disk_tier::store::*;

#[cfg(test)]
mod part_page_tests;
