//! Row-addressable embedding / PLE lookup for `Store`: Metal shared-buffer
//! `WindowBank` LRU, CUDA batched raw-row gather, and background row prefetch.

use super::super::row_fetch::{Fetcher, Job};
use super::{LookupStats, MAX_READ_BYTES, Store, Weight, array_bytes, err};
use crate::array::MxArray;
use napi::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Reader threads for token/PLE rows: 17 random 90 B-2.7 KB reads per token.
/// Direct expert fills yield while these rows are pending.
const ROW_READERS: usize = 32;

/// One lookup window's Metal buffer, shared by the rows it imported. It is
/// released with its last row view, so it is charged whole until then.
pub(super) struct WindowBank {
    pub(super) bytes: u64,
    pub(super) views: usize,
}

impl Store {
    /// Gather a bounded lookup window, sharing duplicate rows. Decoded rows
    /// use the existing Store-owned LRU and its byte accounting, so no second
    /// cache is reserved. The gather stays lazy: the layer that consumes it
    /// submits the imports and dequantizations with its own graph instead of
    /// draining the GPU at the token start.
    pub fn lookup_rows(&mut self, name: &str, ids: &[usize]) -> Result<MxArray> {
        let d = self.descriptor(name)?;
        let total_rows = d.rows()?;
        let width = d.width()?;
        if ids.is_empty() || ids.iter().any(|&id| id >= total_rows) {
            return Err(err("Lookup window exceeds tensor or staging bounds"));
        }
        if super::super::cuda_qwen4_fast_enabled()
            && !self.banks.contains_key(name)
            && (self.plan.direct || total_rows > 1024)
        {
            // Keep each GPU import/dequant staging batch within MAX_READ_BYTES
            // and 65,535 rows so C=6144..8192 prefill chunks (e.g. PLE 16*M =
            // 131,072 rows, token_embd 8192*2560*4 = 80 MiB) work seamlessly.
            let max_rows = ((MAX_READ_BYTES / ((width as u64).saturating_mul(4).max(1))) as usize)
                .clamp(1, 65535);
            if ids.len() <= max_rows {
                return self.lookup_rows_cuda_batched(name, ids);
            }
            let mut parts = Vec::with_capacity(ids.len().div_ceil(max_rows));
            for chunk in ids.chunks(max_rows) {
                parts.push(self.lookup_rows_cuda_batched(name, chunk)?);
            }
            return MxArray::concatenate_many(parts.iter().collect(), Some(0));
        }
        if ids.len() > 65536
            || (ids.len() as u64)
                .saturating_mul(width as u64)
                .saturating_mul(4)
                > MAX_READ_BYTES
        {
            return Err(err("Lookup window exceeds tensor or staging bounds"));
        }
        let mut unique = HashMap::new();
        let mut rows = Vec::new();
        let mut order = Vec::with_capacity(ids.len());
        let mut imported = Vec::new();
        let banked = self.banks.contains_key(name);
        for &id in ids {
            let slot = if let Some(&slot) = unique.get(&id) {
                slot
            } else {
                LookupStats::count(&self.lookup.lookups);
                let miss = !banked && !self.cache.contains_key(&(name.to_string(), id, 1));
                if miss {
                    LookupStats::count(&self.lookup.misses);
                }
                let row = self.read_impl(name, id, 1, true)?.dense_pending()?;
                let slot = rows.len();
                rows.push(row);
                if miss {
                    imported.push((id, slot));
                }
                unique.insert(id, slot);
                slot
            };
            order.push(slot as i32);
        }
        if let [row] = rows.as_slice()
            && order.len() == 1
        {
            return Ok(row.clone());
        }
        // The rows this window imported and still holds share one Metal
        // buffer: each buffer is a Metal resource and MLX aborts at ~499k of
        // them, which one buffer per row reached on the third fresh
        // 10k-token prompt. Rows the LRU already held keep their buffer, so
        // no buffer carries a slot nothing points to, and a window's buffer
        // is charged whole until its last row view is evicted. The window
        // then takes that buffer as ONE concatenation input beside the held
        // rows: one input per imported row again would add a copy dispatch
        // per import to every layer of every decode token (measured 11 us
        // per 2-hit/6-miss lookup, 0.5 ms per token over 48 layers).
        imported.retain(|&(id, _)| self.cache.contains_key(&(name.to_string(), id, 1)));
        if imported.len() > 1 {
            let fresh = MxArray::concatenate_many(
                imported.iter().map(|&(_, slot)| &rows[slot]).collect(),
                Some(0),
            )?;
            let bytes = array_bytes(&fresh)?;
            let id = self.next_window_bank;
            self.next_window_bank += 1;
            let mut rehomed = 0;
            let mut position = vec![0i32; rows.len()];
            let mut is_imported = vec![false; rows.len()];
            for (i, &(row, slot)) in imported.iter().enumerate() {
                let key = (name.to_string(), row, 1);
                let view = fresh.slice_axis(0, i as i64, i as i64 + 1)?;
                let entry = self
                    .cache
                    .get_mut(&key)
                    .ok_or_else(|| err("Qwen4 decoded row left the LRU inside its window"))?;
                rehomed += entry.bytes()?;
                *entry = Arc::new(Weight {
                    values: view,
                    dense_bf16: None,
                    scales: None,
                    biases: None,
                    group: 0,
                    bits: 0,
                    mode: String::new(),
                    q8_symmetric_128: false,
                    qlinear: None,
                });
                self.row_bank.insert(key, id);
                position[slot] = i as i32;
                is_imported[slot] = true;
            }
            self.window_banks.insert(
                id,
                WindowBank {
                    bytes,
                    views: imported.len(),
                },
            );
            self.cache_bytes = self.cache_bytes - rehomed + bytes;
            self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
            let mut parts = vec![&fresh];
            for (slot, row) in rows.iter().enumerate() {
                if !is_imported[slot] {
                    position[slot] = (imported.len() + parts.len() - 1) as i32;
                    parts.push(row);
                }
            }
            let order: Vec<i32> = order.iter().map(|&slot| position[slot as usize]).collect();
            let window = if parts.len() == 1 {
                fresh
            } else {
                MxArray::concatenate_many(parts, Some(0))?
            };
            return window.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0);
        }
        let window = MxArray::concatenate_many(rows.iter().collect(), Some(0))?;
        window.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0)
    }

    /// CUDA batched row gather: collects raw quantized bytes for every unique
    /// row (from the host raw-row cache, parallel `Fetcher::take_many`, or
    /// ascending pread), packs them into ONE contiguous buffer, and runs ONE
    /// `import_bytes` + ONE GPU `dequantize` (eliminating 16,384 per-row GPU
    /// dequantize + `concatenate_many` kernels on 1024-token prefill chunks).
    fn lookup_rows_cuda_batched(&mut self, name: &str, ids: &[usize]) -> Result<MxArray> {
        let d = self.descriptor(name)?.clone();
        let row_bytes = d.row_bytes()? as usize;
        let mut unique_map: HashMap<usize, i32> = HashMap::with_capacity(ids.len());
        let mut unique_ids: Vec<usize> = Vec::with_capacity(ids.len());
        let mut order: Vec<i32> = Vec::with_capacity(ids.len());
        let mut identity = true;
        for (pos, &id) in ids.iter().enumerate() {
            let slot = if let Some(&slot) = unique_map.get(&id) {
                identity = false;
                slot
            } else {
                let slot = unique_ids.len() as i32;
                if slot as usize != pos {
                    identity = false;
                }
                unique_ids.push(id);
                unique_map.insert(id, slot);
                slot
            };
            order.push(slot);
        }

        let mut missing: Vec<(usize, usize)> = Vec::new();
        for (slot, &row_id) in unique_ids.iter().enumerate() {
            LookupStats::count(&self.lookup.lookups);
            if !self.cuda_raw_rows.contains_key(&(name.to_string(), row_id)) {
                LookupStats::count(&self.lookup.misses);
                missing.push((slot, row_id));
            }
        }

        if !missing.is_empty() {
            let fetched = if let Some(fetcher) = self.rows.as_ref() {
                let keys: Vec<(String, usize)> = missing
                    .iter()
                    .map(|&(_, row_id)| (name.to_string(), row_id))
                    .collect();
                Some(fetcher.take_many(&keys))
            } else {
                None
            };

            let mut need_read: Vec<usize> = Vec::new();
            let mut raw_limit = 256u64 << 20;
            if self.cache_limit > 0 {
                raw_limit = raw_limit.min((self.cache_limit / 4).max(16 << 20));
            }

            let mut ready_rows: Vec<(usize, Vec<u8>)> = Vec::with_capacity(missing.len());
            if let Some(fetched) = fetched {
                for ((_, row_id), item) in missing.iter().zip(fetched) {
                    match item {
                        Some((Ok(bytes), late)) if bytes.len() == row_bytes => {
                            LookupStats::count(if late {
                                &self.lookup.late
                            } else {
                                &self.lookup.prefetched
                            });
                            ready_rows.push((*row_id, bytes));
                        }
                        other => {
                            if other.is_some() {
                                LookupStats::count(&self.lookup.late);
                            }
                            need_read.push(*row_id);
                        }
                    }
                }
            } else {
                need_read.extend(missing.iter().map(|&(_, row_id)| row_id));
            }

            if !need_read.is_empty() {
                need_read.sort_unstable();
                let mut used_workers = false;
                #[cfg(not(test))]
                let allow_workers = true;
                #[cfg(test)]
                let allow_workers = self.row_readers;
                if allow_workers && need_read.len() >= 8 {
                    if self.rows.is_none() {
                        self.rows = Some(
                            Fetcher::new(ROW_READERS, Arc::clone(&self.rows_pending))
                                .map_err(err)?,
                        );
                    }
                    let file = self.file(&d.path)?;
                    let path = self.path_arc(&d.path);
                    let mut jobs = Vec::with_capacity(need_read.len());
                    let mut keys = Vec::with_capacity(need_read.len());
                    for &row_id in &need_read {
                        let (offset, len) = d.range(row_id, 1)?;
                        let key = (name.to_string(), row_id);
                        jobs.push(Job {
                            key: key.clone(),
                            file: Arc::clone(&file),
                            path: Some(Arc::clone(&path)),
                            offset,
                            len: len as usize,
                        });
                        keys.push(key);
                    }
                    if let Some(fetcher) = self.rows.as_ref() {
                        fetcher.request(jobs);
                        let got = fetcher.take_many(&keys);
                        let mut fallback = Vec::new();
                        for (&row_id, item) in need_read.iter().zip(got) {
                            match item {
                                Some((Ok(bytes), _)) if bytes.len() == row_bytes => {
                                    ready_rows.push((row_id, bytes));
                                }
                                _ => fallback.push(row_id),
                            }
                        }
                        need_read = fallback;
                        used_workers = true;
                    }
                }
                if !need_read.is_empty() {
                    let _ = used_workers;
                    let file = self.file(&d.path)?;
                    for row_id in need_read {
                        let bytes = d.read_rows(&file, row_id, 1)?;
                        ready_rows.push((row_id, bytes));
                    }
                }
            }

            let total_read_bytes = (ready_rows.len() as u64).saturating_mul(row_bytes as u64);
            self.count_bytes_read(total_read_bytes);

            for (row_id, bytes) in ready_rows {
                let key = (name.to_string(), row_id);
                if !self.cuda_raw_rows.contains_key(&key) {
                    let b = bytes.len() as u64;
                    while self.cuda_raw_rows_bytes.saturating_add(b) > raw_limit {
                        let Some(evict) = self.cuda_raw_rows_order.pop_front() else {
                            break;
                        };
                        if let Some(old) = self.cuda_raw_rows.remove(&evict) {
                            self.cuda_raw_rows_bytes =
                                self.cuda_raw_rows_bytes.saturating_sub(old.len() as u64);
                        }
                    }
                    self.cuda_raw_rows.insert(key.clone(), bytes);
                    self.cuda_raw_rows_order.push_back(key);
                    self.cuda_raw_rows_bytes = self.cuda_raw_rows_bytes.saturating_add(b);
                }
            }
        }

        let mut flat = Vec::with_capacity(unique_ids.len().saturating_mul(row_bytes));
        for &row_id in &unique_ids {
            let key = (name.to_string(), row_id);
            let bytes = self
                .cuda_raw_rows
                .get(&key)
                .ok_or_else(|| err("CUDA batched lookup lost a cached raw row"))?;
            flat.extend_from_slice(bytes);
        }
        let dense = d
            .import_bytes_with_cache(&flat, unique_ids.len(), false)?
            .dense_pending()?;
        if identity {
            Ok(dense)
        } else {
            dense.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0)
        }
    }

    /// Start reading rows `(tensor, row)` on the reader threads; rows the
    /// LRU or a bank already holds are skipped. `read_impl` takes the bytes
    /// when it imports the row.
    pub fn prefetch_rows(&mut self, keys: &[(String, usize)]) -> Result<()> {
        #[cfg(test)]
        if !self.row_readers {
            return Ok(());
        }
        let mut seen = HashSet::new();
        let mut jobs = Vec::new();
        for (name, row) in keys {
            if !seen.insert((name.as_str(), *row))
                || self.banks.contains_key(name)
                || self.cache.contains_key(&(name.clone(), *row, 1))
                || self.cuda_raw_rows.contains_key(&(name.clone(), *row))
            {
                continue;
            }
            let d = self.descriptor(name)?.clone();
            let (offset, len) = d.range(*row, 1)?;
            let file = self.file(&d.path)?;
            let path = self.path_arc(&d.path);
            jobs.push(Job {
                key: (name.clone(), *row),
                file,
                path: Some(path),
                offset,
                len: len as usize,
            });
        }
        if jobs.is_empty() {
            return Ok(());
        }
        // One ascending sweep per table instead of a random walk of the
        // checkpoint: the rows of a window are read in file order.
        jobs.sort_by_key(|job| (Arc::as_ptr(&job.file) as usize, job.offset));
        if self.rows.is_none() {
            self.rows =
                Some(Fetcher::new(ROW_READERS, Arc::clone(&self.rows_pending)).map_err(err)?);
        }
        if let Some(fetcher) = &self.rows {
            fetcher.request(jobs);
        }
        Ok(())
    }

    /// Finish the reads in flight, then stop the reader threads: the rows
    /// they fetched stay ready, a row requested later is read late.
    #[cfg(test)]
    pub(in crate::models) fn pause_rows(&mut self) {
        if let Some(fetcher) = &mut self.rows {
            fetcher.wait_idle();
            fetcher.pause();
        }
    }
}
