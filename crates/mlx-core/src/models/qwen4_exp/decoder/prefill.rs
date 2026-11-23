use super::cuda_deferred::CudaGraphsGuard;
use super::*;

/// What every layer of one carried-window pass shares: the prompt length
/// before its first window, and when its page writes and layers submit.
#[derive(Clone, Copy)]
struct GroupPass {
    base: usize,
    async_window: bool,
    submit_span: usize,
    defer_final: bool,
}

/// The most windows a layer-major group holds on a store whose slots can
/// miss.
pub(in super::super) const PARTIAL_GROUP_WINDOWS: usize = 4;

/// How the layer-major groups of a prompt are bounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum GroupLimit {
    /// One group per run of whole windows, cut only at the deepest
    /// checkpoint and the replay point inside it (a direct fill).
    Whole,
    /// At most this many windows per group, cut at every state point (a
    /// store whose slots can miss).
    Windows(usize),
}

impl GroupLimit {
    /// The most windows one group holds.
    pub(in super::super) fn per_group(self) -> usize {
        match self {
            Self::Whole => usize::MAX,
            Self::Windows(windows) => windows,
        }
    }
}

impl Decoder {
    /// The rows of the next prefill pass after `history`: the admitted
    /// window, halved where it would outgrow a refused gate's reservation.
    pub fn prefill_pass_window(&self) -> usize {
        self.prefill_rows_at(self.history.len())
    }

    /// The rows of a prefill pass that starts after `history_len` tokens.
    pub(in super::super) fn prefill_rows_at(&self, history_len: usize) -> usize {
        if self.paged.is_none()
            && self.host_prefill_scratch.is_none()
            && self.weights.plan.direct
            && self.scope.is_empty()
            && super::super::cuda_qwen4_fast_enabled()
        {
            return self.cuda_prefill_chunk_cap();
        }
        let window = self.prefill_slice_size();
        window.min(super::super::memory::prefill_window(
            &self.config,
            self.host_prefill_scratch,
            history_len + window,
        ))
    }

    /// Flat CUDA direct prefill chunk size (defaults to 4096 so 73 GB of routed
    /// experts stream 2× fewer times on long prompts; override via
    /// `MLX_QWEN4_PREFILL_CHUNK` in `256..=8192`).
    pub(in super::super) fn cuda_prefill_chunk_cap(&self) -> usize {
        #[cfg(test)]
        if let Some(pin) = self.prefill_window_pin {
            return pin;
        }
        static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("MLX_QWEN4_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|v| v.clamp(256, 8192))
                .unwrap_or(4096)
        })
    }

    pub fn prefill_slice_size(&self) -> usize {
        self.prefill_chunk_size
    }

    /// Traverse a bounded prompt chunk layer by layer, sharing weight reads
    /// and projections across its tokens. Verification retains singleton
    /// arithmetic and captures state at every accepted frontier.
    #[cfg(test)]
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        self.prefill_chunk_ahead(tokens, &[], embeddings, project_logits)
    }

    /// `prefill_chunk` whose caller knows the window after this one:
    /// `next`'s rows are requested while this window runs.
    pub fn prefill_chunk_ahead(
        &mut self,
        tokens: &[u32],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        if cfg!(target_os = "macos") {
            self.weights.check_forward_headroom()?;
        }
        self.rotary_tables.get_mut().clear();
        let started = std::time::Instant::now();
        let result = self.count_prefill(|decoder| {
            decoder.prefill_group_with_device_routes(&[tokens], next, embeddings, project_logits)
        });
        self.rotary_tables.get_mut().clear();
        if result.is_err() {
            self.reset();
        } else {
            self.observe_pass(tokens.len(), started.elapsed());
        }
        result
    }

    pub(super) fn prefill_chunk_inner(
        &mut self,
        tokens: &[u32],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        self.prefill_group_inner(&[tokens], next, embeddings, project_logits)
    }

    /// The windows of `tokens` a group runs and the tail that runs after it:
    /// slices of the admitted window, except a final one shorter than both
    /// the slice and 256 rows, which fails the tentative gate and would drain
    /// the GPU inside every sweep.
    pub fn group_windows<'t>(&self, tokens: &'t [u32]) -> (Vec<&'t [u32]>, &'t [u32]) {
        let slice = self.prefill_slice_size().max(1);
        let mut windows: Vec<&[u32]> = tokens.chunks(slice).collect();
        let tail = match windows.last() {
            Some(last) if last.len() < slice && last.len() < 256 => {
                let tail = *last;
                windows.pop();
                tail
            }
            _ => &tokens[tokens.len()..],
        };
        (windows, tail)
    }

    /// How the layer-major groups of a prompt of `suffix` new tokens after
    /// `start` rows are bounded; `None` runs it one window at a time.
    /// During a direct fill every window is a tentative-eligible slice and
    /// the residual streams fit beside the direct turn floor (live host):
    /// one group. On a store whose slots can miss, the most windows (up to
    /// [`PARTIAL_GROUP_WINDOWS`]) whose group scratch fits beside one pass's
    /// scratch at the prompt's last row inside the transient the memory plan
    /// priced; no live host read.
    pub(in super::super) fn group_limit(&self, start: usize, suffix: usize) -> Option<GroupLimit> {
        let slice = self.prefill_slice_size().max(1);
        let windows = suffix / slice + usize::from(suffix % slice >= 256);
        if windows < 2 || self.verification.is_some() {
            return None;
        }
        #[cfg(test)]
        if let Some(pin) = self.group_pin {
            return pin.then_some(GroupLimit::Whole);
        }
        if self.weights.direct_fill_pass() {
            if slice < 256 {
                return None;
            }
            let host = super::super::memory::host_memory().ok()?;
            let need = 22_400 * suffix as u64
                + super::super::memory::prefill_scratch(
                    &self.config,
                    super::super::MAX_PREFILL_CHUNK,
                    self.context_window,
                )
                + super::super::memory::DIRECT_TURN_FLOOR;
            return (need <= host.available()).then_some(GroupLimit::Whole);
        }
        if !self.weights.slot_store() {
            return None;
        }
        let planned = self.planned_scratch?;
        let pass = super::super::memory::prefill_scratch(&self.config, slice, start + suffix);
        (2..=windows.min(PARTIAL_GROUP_WINDOWS))
            .rev()
            .find(|&k| {
                pass.saturating_add(super::super::memory::group_scratch(&self.config, k, slice))
                    <= planned
            })
            .map(GroupLimit::Windows)
    }

    /// Pre-reserve flat BF16 KV and QSA indexer key buffers ONCE per prompt
    /// across all full-attention layers so multi-chunk prefills never trigger
    /// repeated KV reallocation + `concatenate` copies mid-prompt.
    fn pre_reserve_flat_kv_for_prefill(&mut self, prompt_len: usize) -> Result<()> {
        if !super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || !self.weights.embedding_is_bf16()
            || prompt_len < 64
        {
            return Ok(());
        }
        let kv_heads = self.config.num_key_value_heads as i64;
        let head_dim = self.config.head_dim as i64;
        let idx_dim = self.config.indexer_head_dim as i64;
        let ratio = self.config.indexer_compress_ratio.max(1);
        let add_tokens = prompt_len + 128;
        let mut grown: Vec<MxArray> = Vec::new();
        for (layer, cache) in self.caches.iter_mut().enumerate() {
            if self.config.linear(layer) {
                continue;
            }
            let old_kv_cap = match &cache.keys {
                Some(k) => k.shape_at(2)? as usize,
                None => 0,
            };
            let (_, k_buf, v_buf) =
                cache.ensure_flat_kv_capacity(kv_heads, head_dim, add_tokens, DType::BFloat16)?;
            if k_buf.shape_at(2)? as usize > old_kv_cap {
                grown.push(k_buf);
                grown.push(v_buf);
            }
            let add_blocks = (cache.index_tail.len() + add_tokens) / ratio + 1;
            if let Some(idx_buf) =
                cache.ensure_flat_index_capacity(idx_dim, add_blocks, DType::BFloat16)?
            {
                grown.push(idx_buf);
            }
        }
        if !grown.is_empty() {
            let refs: Vec<&MxArray> = grown.iter().collect();
            self.submit_state(&refs, "qwen4::prefill::reserve_flat_kv")?;
        }
        Ok(())
    }

    /// Choose the next prefill chunk length, balancing the final two chunks on
    /// flat CUDA direct prefill so a prompt never streams 48L of routed experts
    /// for a tiny tail chunk (e.g. `remaining < cap + cap/3` splits evenly).
    fn next_prefill_chunk_len(remaining: usize, window: usize, balance_tail: bool) -> usize {
        if balance_tail && window >= 512 && remaining > window && remaining < window + window / 3 {
            remaining.div_ceil(2)
        } else {
            remaining.min(window)
        }
    }

    /// RAII guard that disables MLX CUDA Graph capture during large prefill
    /// passes (`M > 16`) so prefill avoids graph-capture/update stalls while
    /// preserving CUDA Graph capture for low-latency decode (`M <= 16`).
    fn cuda_prefill_graphs_guard(active: bool) -> CudaGraphsGuard {
        CudaGraphsGuard::disable_if(active && super::super::cuda_qwen4_fast_enabled())
    }

    /// A whole prompt: layer-major groups (the last with the tail) when
    /// [`Decoder::group_limit`] allows them, else one window at a time with
    /// the next window's rows requested while this one runs. The admitted
    /// window is read before every pass.
    pub fn prefill_prompt(&mut self, tokens: &[u32]) -> Result<MxArray> {
        let _cuda_graphs_guard = Self::cuda_prefill_graphs_guard(tokens.len() > 16);
        self.pre_reserve_flat_kv_for_prefill(tokens.len())?;
        if let Some(limit) = self.group_limit(self.history.len(), tokens.len()) {
            let (windows, tail) = self.group_windows(tokens);
            let groups: Vec<_> = windows.chunks(limit.per_group()).collect();
            let mut logits = None;
            for (k, group) in groups.iter().enumerate() {
                let last = k + 1 == groups.len();
                let tail = if last { tail } else { &[] };
                logits = Some(self.prefill_group(group, tail, last, false)?.0);
            }
            return logits.ok_or_else(|| Error::from_reason("Empty qwen4_exp prefill"));
        }
        let balance_tail = crate::accelerator::cuda_ops_available()
            && self.weights.plan.direct
            && self.scope.is_empty()
            && super::super::cuda_qwen4_fast_enabled();
        let mut logits = None;
        let mut start = 0;
        while start < tokens.len() {
            let window = self.prefill_pass_window();
            let take = Self::next_prefill_chunk_len(tokens.len() - start, window, balance_tail);
            let end = start + take;
            let next_take = Self::next_prefill_chunk_len(tokens.len() - end, window, balance_tail);
            let next_end = end + next_take;
            logits = Some(self.prefill_chunk_ahead(
                &tokens[start..end],
                &tokens[end..next_end],
                None,
                end == tokens.len(),
            )?);
            start = end;
        }
        logits.ok_or_else(|| Error::from_reason("Empty qwen4_exp prefill"))
    }

    /// One layer-major group over `windows`, then `tail` as its own pass on
    /// the group's final state. Returns the logits of the last row and, when
    /// `views` is set, one hidden view per token of every window (what the
    /// MTP observer takes per window); `last_chunk_hidden` holds the last
    /// pass's views as after any pass. An error resets the decoder.
    pub fn prefill_group(
        &mut self,
        windows: &[&[u32]],
        tail: &[u32],
        project_logits: bool,
        views: bool,
    ) -> Result<(MxArray, Vec<Vec<MxArray>>)> {
        if windows.is_empty() {
            return Err(Error::from_reason("Qwen4 prefill group has no windows"));
        }
        if cfg!(target_os = "macos") {
            self.weights.check_forward_headroom()?;
        }
        let rows: Vec<usize> = windows.iter().map(|w| w.len()).collect();
        let total: usize = rows.iter().sum();
        self.rotary_tables.get_mut().clear();
        self.rotary_tables.get_mut().reserve(windows.len());
        if windows.len() > 1 {
            // Writes a failed turn left lazy would otherwise ride on this
            // group's first gather of the same layer.
            if let Some(adapter) = &mut self.paged {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
        }
        let started = std::time::Instant::now();
        let result = self.count_prefill(|decoder| {
            decoder.prefill_group_with_device_routes(
                windows,
                tail,
                None,
                project_logits && tail.is_empty(),
            )
        });
        self.rotary_tables.get_mut().clear();
        let logits = match result {
            Ok(logits) => logits,
            Err(error) => {
                self.group_streams.clear();
                self.reset();
                return Err(error);
            }
        };
        self.note("qwen4::group::end");
        if windows.len() > 1 {
            let live = &self.weights.live;
            live.group_passes.fetch_add(1, Ordering::Relaxed);
            live.group_windows
                .fetch_add(windows.len() as u64, Ordering::Relaxed);
        }
        let streams = std::mem::take(&mut self.group_streams);
        let views = if views && windows.len() > 1 {
            streams
                .iter()
                .zip(&rows)
                .map(|(stream, &n)| {
                    (0..n)
                        .map(|j| stream.slice_axis(1, j as i64, j as i64 + 1))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else if views {
            vec![self.last_chunk_hidden.clone()]
        } else {
            Vec::new()
        };
        let elapsed = started.elapsed();
        if windows.len() > 1 {
            for &n in &rows {
                self.observe_pass(n, elapsed.mul_f64(n as f64 / total as f64));
            }
        } else {
            self.observe_pass(total, elapsed);
        }
        if tail.is_empty() {
            return Ok((logits, views));
        }
        self.note("qwen4::tail::start");
        let logits = self.prefill_chunk_ahead(tail, &[], None, project_logits)?;
        Ok((logits, views))
    }

    /// Slot misses, fill read bytes and fill microseconds since load, for a
    /// pass's debug line.
    fn fill_records(&self) -> [u64; 3] {
        let live = &self.weights.live;
        [
            live.slot_misses.load(Ordering::Relaxed),
            live.fill_read_bytes.load(Ordering::Relaxed),
            live.fill_micros.load(Ordering::Relaxed),
        ]
    }

    /// Runs one prefill pass and adds what it moved of the fill, miss, read
    /// and replay counters to their prefill shares.
    fn count_prefill<T>(&mut self, pass: impl FnOnce(&mut Self) -> T) -> T {
        let read = |live: &super::super::live::Live| {
            live.prefill_shares()
                .map(|(total, _)| total.load(Ordering::Relaxed))
        };
        let before = read(&self.weights.live);
        let out = pass(self);
        let live = &self.weights.live;
        for ((_, share), (after, before)) in live
            .prefill_shares()
            .into_iter()
            .zip(read(live).into_iter().zip(before))
        {
            share.fetch_add(after.saturating_sub(before), Ordering::Relaxed);
        }
        out
    }

    /// Window `w` of a group starts a layer: `history` is the prompt before
    /// it and the page cursor its base, as at the start of its own pass.
    fn set_cursor(&mut self, base: usize, windows: &[&[u32]], w: usize) {
        self.history.truncate(base);
        for window in &windows[..w] {
            self.history.extend_from_slice(window);
        }
        self.paged_chunk_start = self.history.len();
    }

    /// A group starts: every window's embedding rows, then every window's
    /// PLE rows, then the tail's, three requests in that order so sweep 0
    /// never queues behind sweep 1's rows and the tail's behind neither.
    fn prefetch_group_rows(&mut self, windows: &[&[u32]], tail: &[u32]) -> Result<()> {
        let embedding = self.key("embed_tokens.weight", "token_embd.weight");
        let keys: Vec<_> = windows
            .iter()
            .flat_map(|w| w.iter().map(|&t| (embedding.clone(), t as usize)))
            .collect();
        self.weights.prefetch_rows(&keys)?;
        let mut history = self.history.clone();
        let mut keys = Vec::new();
        for window in windows {
            keys.extend(self.ple_row_keys_after(&history, window)?);
            history.extend_from_slice(window);
        }
        self.weights.prefetch_rows(&keys)?;
        if tail.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<_> = tail
            .iter()
            .map(|&t| (embedding.clone(), t as usize))
            .collect();
        keys.extend(self.ple_row_keys_after(&history, tail)?);
        self.weights.prefetch_rows(&keys)
    }

    /// Layer `i` of a window of `rows` tokens at `first` is done: a full
    /// attention layer's keys and values go to the pool, lazily, and are
    /// evaluated where a plain pass evaluates them.
    fn write_window_kv(
        &mut self,
        i: usize,
        cache: &mut LayerCache,
        rows: usize,
        first: usize,
        pass: GroupPass,
    ) -> Result<()> {
        let GroupPass {
            base: floor,
            async_window,
            submit_span,
            defer_final,
        } = pass;
        if self.config.linear(i) {
            return Ok(());
        }
        let Some(adapter) = &mut self.paged else {
            return Ok(());
        };
        let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
        let shape = [
            rows as i64,
            self.config.num_key_value_heads as i64,
            self.config.head_dim as i64,
        ];
        let k = cache
            .keys
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no attention keys"))?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&shape)?;
        let v = cache
            .values
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no attention values"))?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&shape)?;
        super::super::paged::write_rows(adapter, layer, &k, &v, first as u32, floor as u32)?;
        let layers = self.config.num_hidden_layers;
        if self.verification.is_none()
            && ((rows > 1 && !async_window)
                || (i + 1).is_multiple_of(submit_span)
                || i + 1 == layers)
            && !(defer_final && i + 1 == layers)
        {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }

    /// Layer `i` of a carried-window pass over `windows`, each window of a
    /// group at its own cursor; `streams` holds each window's residual
    /// stream (window 0's from the start, the others' from layer 0 on).
    /// Under a union lease every window's attention, K/V write, hyper mix
    /// and routes run first, then the routed experts of all windows over one
    /// lease plan, then each window's shared expert and injection.
    fn group_layer(
        &mut self,
        i: usize,
        cache: &mut LayerCache,
        windows: &[&[u32]],
        streams: &mut Vec<MxArray>,
        prefill_normed: &mut Option<MxArray>,
        pass: GroupPass,
    ) -> Result<()> {
        let group = windows.len() > 1;
        let union = self.union_lease(windows, i)?;
        let carry_next_norm = !group && !union && super::super::cuda_qwen4_fast_enabled();
        let mut routed = Vec::new();
        let mut inputs = Vec::new();
        for (w, tokens_w) in windows.iter().enumerate() {
            if group {
                if w > 0 {
                    self.check_cancelled()?;
                }
                self.set_cursor(pass.base, windows, w);
            }
            let input = if i == 0 && w > 0 {
                let embedding_key = self.key("embed_tokens.weight", "token_embd.weight");
                let rows: Vec<usize> = tokens_w.iter().map(|&t| t as usize).collect();
                let e = self.weights.lookup_rows(&embedding_key, &rows)?.reshape(&[
                    1,
                    tokens_w.len() as i64,
                    self.config.hidden_size as i64,
                ])?;
                let stream = MxArray::tile(&e, &[1, 1, self.config.hc_count as i32])?;
                streams.push(stream.clone());
                stream
            } else {
                streams[w].clone()
            };
            let carried_in = if carry_next_norm {
                prefill_normed.take()
            } else {
                None
            };
            let (stream, normed) = self.attention_matrix_for_mlp_with_norm(
                &input,
                tokens_w,
                i,
                cache,
                true,
                carried_in.as_ref(),
            )?;
            if !union {
                let (next_stream, next_norm) = self.mlp_matrix_for_attention_next(
                    &stream,
                    i,
                    normed.as_ref(),
                    carry_next_norm,
                )?;
                streams[w] = next_stream;
                if carry_next_norm {
                    *prefill_normed = next_norm;
                }
            }
            let first = self.paged_chunk_start;
            self.write_window_kv(i, cache, tokens_w.len(), first, pass)?;
            if !union {
                continue;
            }
            let (mixed, gate) = self.mlp_mix(&stream, i, normed.as_ref())?;
            let (selected, scores) = self.route_window(&mixed, i)?;
            let ids = selected.to_uint32()?.to_vec();
            routed.push((stream, gate, selected));
            inputs.push((mixed, ids, scores));
        }
        if !union {
            return Ok(());
        }
        *prefill_normed = None;
        let sums = match self.union_expert_rows(&inputs, i)? {
            Some(sums) => sums,
            None => {
                let mut sums = Vec::with_capacity(inputs.len());
                for ((mixed, _, scores), (_, _, selected)) in inputs.iter().zip(&routed) {
                    sums.push(self.expert_rows_for(mixed, selected, scores, i)?);
                }
                sums
            }
        };
        #[cfg(test)]
        if let Some(held) = self.group_held.last_mut() {
            let arrays = streams
                .iter()
                .chain(
                    routed
                        .iter()
                        .flat_map(|(stream, gate, selected)| {
                            [Some(stream), gate.as_ref(), Some(selected)]
                        })
                        .flatten(),
                )
                .chain(inputs.iter().flat_map(|(mixed, _, scores)| [mixed, scores]))
                .chain(&sums);
            *held += arrays.map(|a| a.nbytes() as u64).sum::<u64>()
                + inputs
                    .iter()
                    .map(|(_, ids, _)| 4 * ids.len() as u64)
                    .sum::<u64>();
        }
        for (w, (((stream, gate, _), (mixed, _, _)), sum)) in
            routed.into_iter().zip(&inputs).zip(sums).enumerate()
        {
            let branch = self.with_shared_expert(mixed, &sum, i)?;
            streams[w] = self.mlp_inject(&stream, &branch, gate, i)?;
        }
        Ok(())
    }

    /// One pass over `windows`, `for layer { for window }`. A group of one
    /// window is the window-major pass; a group of more runs every window of
    /// a layer before the next layer, each window at its own cursor, so the
    /// GDN, PLE, indexer and page state flows from (layer, w) to
    /// (layer, w + 1) as a window-major run carries it between passes.
    pub(super) fn prefill_group_inner(
        &mut self,
        windows: &[&[u32]],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        if let [[token]] = windows
            && self.verification.is_none()
        {
            return self.decode_token(*token, next, embeddings, true, project_logits);
        }
        if self.verification.is_none() {
            self.settle_cuda_pending_gdn()?;
        }
        let base = self.history.len();
        let started = std::time::Instant::now();
        let total: usize = windows.iter().map(|w| w.len()).sum();
        let _cuda_graphs_guard =
            Self::cuda_prefill_graphs_guard(self.verification.is_none() && total > 16);
        let rows = (self.weights.lookup.prefetched(), self.weights.lookup.late());
        let fills = self.fill_records();
        let group = windows.len() > 1;
        let tokens = *windows
            .first()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill group has no windows"))?;
        let layers = self.config.num_hidden_layers;
        let submit_span = self.decode_submit_span();
        let async_window = windows.iter().all(|w| w.len() > 8) && self.async_device_prefill();
        let defer_final = self.deferred_forward_completion();
        let max_chunk = self
            .cuda_prefill_chunk_cap()
            .max(super::super::MAX_PREFILL_CHUNK);
        if windows.iter().any(|w| w.is_empty() || w.len() > max_chunk)
            || base + total > self.context_window
            || windows
                .iter()
                .flat_map(|w| w.iter())
                .any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(Error::from_reason(
                "Qwen4 prefill chunk exceeds token/context budget",
            ));
        }
        if let Some(e) = embeddings {
            if group {
                return Err(Error::from_reason(
                    "Qwen4 prefill group takes no media embeddings",
                ));
            }
            if *e.shape()? != [1, tokens.len() as i64, self.config.hidden_size as i64] {
                return Err(Error::from_reason("Qwen4 media embedding shape mismatch"));
            }
        }
        self.paged_chunk_start = base;
        if let Some(adapter) = &mut self.paged {
            if adapter.request_tokens() != self.history {
                return Err(Error::from_reason(
                    "Qwen4 paged/recurrent frontier mismatch",
                ));
            }
            if group {
                adapter
                    .record_tokens(&windows.concat())
                    .map_err(Error::from_reason)?;
            } else {
                adapter.record_tokens(tokens).map_err(Error::from_reason)?;
            }
        }
        if group {
            self.prefetch_group_rows(windows, next)?;
        } else {
            self.prefetch_window_rows(tokens, next, embeddings.is_none())?;
        }
        let carry = tokens.len() > 1 && self.verification.is_none();
        if group && !carry {
            return Err(Error::from_reason(
                "Qwen4 prefill group needs the carried window path",
            ));
        }
        let mut hidden = Vec::with_capacity(if carry { 0 } else { tokens.len() });
        let embedding_key = self.key("embed_tokens.weight", "token_embd.weight");
        let hc = self.config.hc_count as i32;
        let window_embedding = if embeddings.is_none() && tokens.len() > 1 {
            Some(
                self.weights
                    .lookup_rows(
                        &embedding_key,
                        &tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
                    )?
                    .reshape(&[1, tokens.len() as i64, self.config.hidden_size as i64])?,
            )
        } else {
            None
        };
        // One residual stream per window; the other windows' streams start
        // at their layer-0 step, from the rows the fetcher holds by then.
        let mut streams: Vec<MxArray> = Vec::with_capacity(windows.len());
        let mut window: Option<MxArray> = None;
        if carry {
            let embeddings = embeddings.or(window_embedding.as_ref()).ok_or_else(|| {
                Error::from_reason("Qwen4 prefill window is missing its token embeddings")
            })?;
            streams.push(MxArray::tile(embeddings, &[1, 1, hc])?);
        } else if self.verification.is_some()
            && let Some(e) = embeddings.or(window_embedding.as_ref())
        {
            window = Some(MxArray::tile(e, &[1, 1, hc])?);
        } else {
            for (j, &token) in tokens.iter().enumerate() {
                self.check_cancelled()?;
                let row = match embeddings.or(window_embedding.as_ref()) {
                    Some(e) => e.slice_axis(1, j as i64, j as i64 + 1)?,
                    None => self.embed_token(token)?,
                };
                hidden.push(MxArray::tile(&row, &[1, 1, hc])?);
            }
            if self.verification.is_some() {
                window = if hidden.len() == 1 {
                    hidden.pop()
                } else {
                    Some(MxArray::concatenate_many(hidden.iter().collect(), Some(1))?)
                };
                hidden.clear();
            }
        }
        let deferred_verify = self.cuda_verify_gdn_tape.is_some();
        if self.verification.is_some() && !deferred_verify {
            self.verification = Some(
                (0..tokens.len())
                    .map(|j| DecoderState {
                        caches: (0..self.config.num_hidden_layers)
                            .map(|_| LayerCache::default())
                            .collect(),
                        history: self
                            .history
                            .iter()
                            .chain(tokens[..=j].iter())
                            .copied()
                            .collect(),
                        positions: self.positions.clone(),
                        rope_delta: self.rope_delta,
                        media_digests: self.media_digests.clone(),
                        last_chunk_hidden: Vec::new(),
                        cancelled: self.cancelled.clone(),
                        route_guard: self.route_guard.clone(),
                    })
                    .collect(),
            );
        }
        self.begin_taps();
        let mut prefill_normed: Option<MxArray> = None;
        let mut verify_normed: Option<MxArray> = None;
        for i in 0..layers {
            self.check_cancelled()?;
            let mut cache = std::mem::take(&mut self.caches[i]);
            if carry {
                let pass = GroupPass {
                    base,
                    async_window,
                    submit_span,
                    defer_final,
                };
                self.group_layer(
                    i,
                    &mut cache,
                    windows,
                    &mut streams,
                    &mut prefill_normed,
                    pass,
                )?;
                #[cfg(test)]
                if let Some(passes) = &mut self.pass_outputs {
                    passes.push((i, streams.clone()));
                }
                self.tap_layer(i, &streams)?;
            } else {
                let input = window
                    .take()
                    .ok_or_else(|| Error::from_reason("Qwen4 verification lost its window"))?;
                let (stream, next_normed, saved) =
                    self.verify_layer(&input, tokens, i, &mut cache, verify_normed.as_ref())?;
                let should_submit = if deferred_verify {
                    // Submit Layer 0 right away so GPU overlaps with Layer 1 PLE
                    // reads, Layer 3 to keep GPU saturated while Layers 4..11
                    // build, then every 12 layers (5 async batches vs 48).
                    i == 0 || i == 3 || ((i + 1).is_multiple_of(12) && i + 1 < layers)
                } else {
                    true
                };
                if should_submit {
                    if let Some(ref normed) = next_normed {
                        self.submit(&[&stream, normed], "qwen4::decoder::verify")?;
                    } else {
                        self.submit(&[&stream], "qwen4::decoder::verify")?;
                    }
                }
                #[cfg(test)]
                if let Some(passes) = &mut self.pass_outputs {
                    passes.push((i, vec![stream.clone()]));
                }
                self.tap_layer(i, std::slice::from_ref(&stream))?;
                window = Some(stream);
                verify_normed = next_normed;
                if let Some(row_caches) = &mut self.cuda_verify_row_caches {
                    for (caches_j, saved_j) in row_caches.iter_mut().zip(saved) {
                        caches_j[i] = saved_j;
                    }
                } else if let Some(states) = &mut self.verification {
                    for (state, saved_j) in states.iter_mut().zip(saved) {
                        state.caches[i] = saved_j;
                    }
                }
            }
            self.history.truncate(base);
            if group {
                self.paged_chunk_start = base;
            }
            self.caches[i] = cache;
            super::super::memory::maintain_freelist(self.weights.plan.physical_bytes);
            if group {
                #[cfg(test)]
                if let Some(mut hook) = self.after_sweep.take() {
                    hook(self, i);
                    self.after_sweep = Some(hook);
                }
            }
        }
        self.finish_taps(base)?;
        let last_tokens = windows[windows.len() - 1];
        let final_window = if carry {
            streams.last().cloned()
        } else {
            window
        };
        if self.verification.is_some() {
            self.cuda_verify_final_window = final_window.clone();
            self.cuda_verify_final_normed = verify_normed;
            if deferred_verify {
                let out = final_window
                    .ok_or_else(|| Error::from_reason("Qwen4 deferred verify lost its window"))?;
                self.history.extend_from_slice(tokens);
                return Ok(out);
            }
        }
        if let Some(window) = final_window {
            // Only public/MTP frontier capture needs token views. Keep one
            // matrix across layer boundaries instead of rebuilding it twice
            // per layer from thousands of token-sized lazy array objects.
            hidden = (0..last_tokens.len())
                .map(|j| window.slice_axis(1, j as i64, j as i64 + 1))
                .collect::<Result<Vec<_>>>()?;
        }
        if let Some(states) = &mut self.verification {
            for (state, h) in states.iter_mut().zip(&hidden) {
                state.last_chunk_hidden = vec![h.clone()];
            }
        }
        let last_hidden = hidden
            .last()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no hidden rows"))?
            .clone();
        self.last_chunk_hidden = hidden;
        if self.verification.is_some() {
            self.history.extend_from_slice(tokens);
            return Ok(last_hidden);
        }
        let (mixed, _) = self.hyper(&last_hidden, "hyper_connection_mixer", "output_hc", false)?;
        let logits = if project_logits {
            self.linear(&mixed, "lm_head.weight", "output.weight")?
        } else {
            mixed
        };
        if !defer_final {
            if super::super::cuda_qwen4_fast_enabled() {
                let mut completed = vec![&logits];
                completed.extend(self.pending_state_arrays());
                if let Some(taps) = &self.hidden_taps {
                    completed.extend(taps.last_pass_arrays().iter());
                }
                MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::logits")?;
            } else {
                MxArray::eval_arrays_with_context(&[&logits], "qwen4::decoder::logits")?;
            }
        }
        let filled = self.fill_records();
        tracing::debug!(
            "qwen4 prefill {}: {total} tokens at {base} in {:.1?}, rows prefetched +{} late +{}, slot misses +{} fill bytes +{} fill time +{:.1} ms",
            if group {
                format!("group of {} windows", windows.len())
            } else {
                "window".to_owned()
            },
            started.elapsed(),
            self.weights.lookup.prefetched() - rows.0,
            self.weights.lookup.late() - rows.1,
            filled[0].saturating_sub(fills[0]),
            filled[1].saturating_sub(fills[1]),
            filled[2].saturating_sub(fills[2]) as f64 / 1000.0
        );

        for window in windows {
            self.history.extend_from_slice(window);
        }
        self.group_streams = if group { streams } else { Vec::new() };
        Ok(logits)
    }
}
