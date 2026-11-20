//! What the DFlash2 draft reads of the target: after each tapped layer
//! completes, the mean of its hyper-connection streams, summed in fp32 in
//! stream order and stored as bf16 (Splash `addStreamCapture`,
//! `runtime/model/Qwen4ExpTarget.cpp:54-65`, kernel
//! `hyper_connection_stream_mean`, `runtime/metal/kernels/shared/hyper_connection.metal:278-289`).
//! A pass's evaluation list does not change; its taps are evaluated right
//! after it.
use super::*;

/// The taps of one finished pass.
pub(in super::super) struct TappedRows {
    /// The sequence row of the pass's first row.
    pub(in super::super) first: usize,
    /// One `[1, rows, hidden]` bf16 array per tapped layer, in tap order.
    pub(in super::super) hidden: Vec<MxArray>,
}

impl TappedRows {
    pub(in super::super) fn rows(&self) -> Result<usize> {
        Ok(self.hidden[0].shape_at(1)? as usize)
    }
}

/// The tapped layers, the pass in flight and the finished passes since the
/// last take. Finished passes cover consecutive rows: a pass that starts at
/// row `r` drops what earlier passes recorded from `r` on (a replay, or rows
/// a verify commit rolled back).
pub(super) struct HiddenTaps {
    layers: Vec<usize>,
    running: Vec<Option<MxArray>>,
    passes: Vec<TappedRows>,
    /// Rows before the newest finished row's last `keep` are dropped with
    /// their records: a lazy tap holds its whole layer output alive.
    keep: Option<usize>,
}

impl HiddenTaps {
    pub(super) fn new(layers: &[usize], num_layers: usize) -> Result<Self> {
        if layers.is_empty()
            || layers.windows(2).any(|pair| pair[0] >= pair[1])
            || layers.iter().any(|&layer| layer >= num_layers)
        {
            return Err(Error::from_reason(format!(
                "Qwen4 hidden taps {layers:?} are not ascending layers below {num_layers}"
            )));
        }
        Ok(Self {
            layers: layers.to_vec(),
            running: vec![None; layers.len()],
            passes: Vec::new(),
            keep: None,
        })
    }

    pub(super) fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// Keep the records of the last `rows` finished rows only; `None` keeps
    /// every record until the next take.
    pub(super) fn keep_rows(&mut self, rows: Option<usize>) {
        self.keep = rows;
    }

    /// A pass starts at its first layer.
    pub(super) fn begin(&mut self) {
        self.running.iter_mut().for_each(|tap| *tap = None);
    }

    /// Layer `layer` of the pass in flight finished: `outputs` are its
    /// residual streams `[1, rows, hc * hidden]`, in row order. A layer run
    /// again (a replay) replaces what the earlier run tapped.
    pub(super) fn tap(
        &mut self,
        layer: usize,
        outputs: &[MxArray],
        hc: usize,
        hidden: usize,
    ) -> Result<()> {
        let Some(slot) = self.layers.iter().position(|&l| l == layer) else {
            return Ok(());
        };
        let means = outputs
            .iter()
            .map(|x| stream_mean(x, hc, hidden))
            .collect::<Result<Vec<_>>>()?;
        self.running[slot] = Some(match means.as_slice() {
            [one] => one.clone(),
            _ => MxArray::concatenate_many(means.iter().collect(), Some(1))?,
        });
        Ok(())
    }

    /// The pass in flight, whose first row is sequence row `first`, ran its
    /// last layer.
    pub(super) fn finish(&mut self, first: usize) -> Result<()> {
        let hidden = self
            .running
            .iter()
            .zip(&self.layers)
            .map(|(tap, &layer)| {
                tap.clone().ok_or_else(|| {
                    Error::from_reason(format!("Qwen4 hidden tap at layer {layer} was not taken"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        while self.passes.last().is_some_and(|pass| pass.first >= first) {
            self.passes.pop();
        }
        if let Some(last) = self.passes.last_mut() {
            let keep = first - last.first;
            if last.rows()? > keep {
                for tap in &mut last.hidden {
                    *tap = tap.slice_axis(1, 0, keep as i64)?;
                }
            }
        }
        let end = first + hidden[0].shape_at(1)? as usize;
        self.passes.push(TappedRows { first, hidden });
        if let Some(keep) = self.keep {
            let floor = end.saturating_sub(keep);
            let mut dropped = 0;
            for pass in &self.passes {
                if pass.first + pass.rows()? > floor {
                    break;
                }
                dropped += 1;
            }
            self.passes.drain(..dropped);
        }
        // A lazy tap holds its layer's whole output until evaluated.
        if let Some(pass) = self.passes.last().filter(|pass| pass.first == first) {
            MxArray::async_eval_arrays(&pass.hidden.iter().collect::<Vec<_>>());
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn passes(&self) -> &[TappedRows] {
        &self.passes
    }

    /// Bytes of the running taps and the finished passes' records.
    pub(super) fn held_bytes(&self) -> u64 {
        self.running
            .iter()
            .flatten()
            .chain(self.passes.iter().flat_map(|pass| &pass.hidden))
            .map(|tap| tap.nbytes() as u64)
            .sum()
    }

    /// The finished passes since the last take.
    pub(super) fn take(&mut self) -> Vec<TappedRows> {
        std::mem::take(&mut self.passes)
    }

    /// The decoder dropped its sequence.
    pub(super) fn clear(&mut self) {
        self.begin();
        self.passes.clear();
    }
}

/// `[1, rows, hidden]` bf16: the mean of the `hc` streams of `x`
/// (`[1, rows, hc * hidden]`, stream `s` at columns `s * hidden..`), its
/// fp32 sum taken in stream order.
fn stream_mean(x: &MxArray, hc: usize, hidden: usize) -> Result<MxArray> {
    #[cfg(test)]
    STREAM_MEANS.with(|n| n.set(n.get() + 1));
    let stream = |s: usize| {
        x.slice_axis(2, (s * hidden) as i64, ((s + 1) * hidden) as i64)?
            .astype(DType::Float32)
    };
    let mut sum = stream(0)?;
    for s in 1..hc {
        sum = sum.add(&stream(s)?)?;
    }
    sum.div_scalar(hc as f64)?.astype(DType::BFloat16)
}

#[cfg(test)]
thread_local! {
    static STREAM_MEANS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only: the stream means built on this thread.
#[cfg(test)]
pub(in super::super) fn stream_means() -> u64 {
    STREAM_MEANS.with(std::cell::Cell::get)
}
