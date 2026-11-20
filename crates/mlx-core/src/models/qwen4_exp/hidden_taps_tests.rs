use super::*;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;

/// The tiny fixture's taps: GDN layer 0, the PLE layer 1 and the last
/// layer, attention.
const TINY_TAPS: [usize; 3] = [0, 1, 3];

/// f32 rounded to bf16, nearest even, as f32.
fn bf16_round(x: f32) -> u32 {
    let bits = x.to_bits();
    if x.is_nan() {
        return (bits | 0x0040_0000) & 0xFFFF_0000;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb)) & 0xFFFF_0000
}

/// The host's mean of `hc` streams per row of every array in `outputs`
/// (`[1, rows, hc * hidden]`, in row order), summed in f32 in stream order
/// and rounded to bf16: one f32 bit pattern per (row, column).
fn host_means(outputs: &[MxArray], hc: usize, hidden: usize) -> Vec<u32> {
    let mut means = Vec::new();
    for x in outputs {
        let values = x.to_float32().unwrap();
        let rows = x.shape_at(1).unwrap() as usize;
        assert_eq!(values.len(), rows * hc * hidden);
        for r in 0..rows {
            let row = &values[r * hc * hidden..(r + 1) * hc * hidden];
            for d in 0..hidden {
                let mut sum = row[d];
                for s in 1..hc {
                    sum += row[s * hidden + d];
                }
                means.push(bf16_round(sum / hc as f32));
            }
        }
    }
    means
}

/// Each layer's outputs from the last run of it in `entries` (a replay runs
/// layers again after the pass it replays).
fn last_writes(entries: &[(usize, Vec<MxArray>)], layers: usize) -> Vec<Vec<MxArray>> {
    let mut last: Vec<Option<Vec<MxArray>>> = vec![None; layers];
    for (layer, outputs) in entries {
        last[*layer] = Some(outputs.clone());
    }
    last.into_iter()
        .enumerate()
        .map(|(layer, outputs)| outputs.unwrap_or_else(|| panic!("layer {layer} ran in no pass")))
        .collect()
}

/// The direct reads of the decoder's layer outputs since the last call: the
/// one-token passes' and the multi-row passes'.
fn drain_outputs(decoder: &mut decoder::Decoder) -> Vec<(usize, Vec<MxArray>)> {
    let mut entries: Vec<(usize, Vec<MxArray>)> = decoder
        .layer_outputs
        .replace(Vec::new())
        .unwrap()
        .into_iter()
        .map(|(layer, x)| (layer, vec![x]))
        .collect();
    entries.extend(decoder.pass_outputs.replace(Vec::new()).unwrap());
    entries
}

/// `pass` is the tapped pass of rows `first..first + rows` whose layers'
/// last outputs are `outputs`: one bf16 `[1, rows, hidden]` tap per tapped
/// layer, each the stream mean of that layer's output.
fn assert_pass(
    pass: &decoder::TappedRows,
    outputs: &[Vec<MxArray>],
    taps: &[usize],
    first: usize,
    rows: usize,
    c: &config::Config,
    what: &str,
) {
    assert_eq!(pass.first, first, "{what}: first row");
    assert_eq!(pass.hidden.len(), taps.len(), "{what}: taps");
    for (tap, &layer) in pass.hidden.iter().zip(taps) {
        assert_eq!(
            tap.shape().unwrap().to_vec(),
            vec![1, rows as i64, c.hidden_size as i64],
            "{what}: layer {layer} shape"
        );
        assert_eq!(
            tap.dtype().unwrap(),
            crate::array::DType::BFloat16,
            "{what}: layer {layer}"
        );
        let got = f32_bits(tap);
        let want = host_means(&outputs[layer], c.hc_count, c.hidden_size);
        assert_eq!(got.len(), want.len(), "{what}: layer {layer} values");
        let diffs = (0..got.len()).filter(|&k| got[k] != want[k]).count();
        assert_eq!(
            diffs,
            0,
            "{what}: layer {layer}: {diffs} of {} tap values differ from the mean of its output",
            got.len()
        );
    }
}

/// `count` prompt tokens inside the fixtures' vocabulary.
fn prompt(count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| ((i * 13 + i / 4) % 61 + 2) as u32)
        .collect()
}

/// The machine runs the tentative prefill kernels.
fn tentative_kernels_available() -> bool {
    crate::engine::persistence::compiled_forward_backend_available()
        && unsafe { mlx_sys::mlx_metal_is_nax_available() }
        && super::super::tf32_enabled()
}

/// Both decoders record their layer outputs.
fn record_outputs(decoders: [&mut decoder::Decoder; 2]) {
    for decoder in decoders {
        decoder.layer_outputs = Some(Vec::new());
        decoder.pass_outputs = Some(Vec::new());
    }
}

/// On the tiny fixture, paged and flat, a decoder with taps after layers 0,
/// 1 and 3 gives the bits of one with no taps on every pass kind: a
/// layer-major group of two windows and its tail, a prefill window, decode
/// steps, verify rows of widths 1-4 (with the commit's rollback), a host
/// replay from layer 1, 2 and 3 after a poisoned pass, and a device prefill
/// replay: logits and every state array. Each pass leaves one record whose
/// taps are, bit for bit, the bf16 stream mean of the layer's output as the
/// last run of the layer left it; a pass after a verify commit cuts the
/// verify's record to the kept rows.
/// MUTATION: tap the layer input → red.
/// MUTATION: a replayed layer keeps the first run's tap → red.
/// MUTATION: a pass keeps the records of the rows it runs again → red.
#[test]
fn taps_read_each_layer_output_and_move_no_bit() {
    for paged in [false, true] {
        let mut on = tiny_inner();
        let mut off = tiny_inner();
        for inner in [&mut on, &mut off] {
            if paged {
                inner.activate_paged_seq(1).unwrap();
            } else {
                inner.decoder.paged = None;
            }
        }
        on.decoder.set_hidden_taps(Some(&TINY_TAPS[..])).unwrap();
        record_outputs([&mut on.decoder, &mut off.decoder]);
        let c = on.decoder.config.clone();
        let layers = c.num_hidden_layers;
        let p = |what: &str| format!("paged={paged} {what}");

        // A group of two windows and its tail: two passes.
        let w0: Vec<u32> = (0..8).map(|i| 3 + (i % 10) as u32).collect();
        let w1: Vec<u32> = (0..8).map(|i| 5 + (i % 9) as u32).collect();
        let tail = [9, 4, 12];
        let got = on
            .decoder
            .prefill_group(&[&w0, &w1], &tail, true, false)
            .unwrap()
            .0;
        let want = off
            .decoder
            .prefill_group(&[&w0, &w1], &tail, true, false)
            .unwrap()
            .0;
        assert_same_bits(&got, &want, &p("group logits"));
        route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &p("group"));
        let entries = drain_outputs(&mut on.decoder);
        drain_outputs(&mut off.decoder);
        assert_eq!(entries.len(), 2 * layers, "{}", p("group + tail layers"));
        let passes = on.decoder.take_hidden_taps();
        assert_eq!(passes.len(), 2, "{}", p("group + tail passes"));
        assert_pass(
            &passes[0],
            &last_writes(&entries[..layers], layers),
            &TINY_TAPS,
            0,
            16,
            &c,
            &p("group"),
        );
        assert_pass(
            &passes[1],
            &last_writes(&entries[layers..], layers),
            &TINY_TAPS,
            16,
            3,
            &c,
            &p("tail"),
        );

        // One prefill window.
        let window = [7, 3, 11, 6, 8, 2];
        let got = on.decoder.prefill_chunk(&window, None, true).unwrap();
        let want = off.decoder.prefill_chunk(&window, None, true).unwrap();
        assert_same_bits(&got, &want, &p("window logits"));
        route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &p("window"));
        let entries = drain_outputs(&mut on.decoder);
        drain_outputs(&mut off.decoder);
        let passes = on.decoder.take_hidden_taps();
        assert_eq!(passes.len(), 1);
        assert_pass(
            &passes[0],
            &last_writes(&entries, layers),
            &TINY_TAPS,
            19,
            6,
            &c,
            &p("window"),
        );

        // Decode steps.
        let mut token = 5;
        for step in 0..6 {
            let at = on.decoder.history.len();
            let got = on.decoder.step(token).unwrap();
            let want = off.decoder.step(token).unwrap();
            let what = p(&format!("step {step}"));
            assert_same_bits(&got, &want, &format!("{what}: logits"));
            route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &what);
            let entries = drain_outputs(&mut on.decoder);
            drain_outputs(&mut off.decoder);
            let passes = on.decoder.take_hidden_taps();
            assert_eq!(passes.len(), 1, "{what}");
            assert_pass(
                &passes[0],
                &last_writes(&entries, layers),
                &TINY_TAPS,
                at,
                1,
                &c,
                &what,
            );
            token = argmax(&got);
        }

        // Verify rows, then the commit of `keep` rows and a step.
        for width in 1..=4usize {
            let keep = width.div_ceil(2);
            let tokens: Vec<u32> = (0..width)
                .map(|j| 3 + ((token as usize + j) % 11) as u32)
                .collect();
            let at = on.decoder.history.len();
            let (got, got_states) = on.decoder.verify_chunk(&tokens).unwrap();
            let (want, want_states) = off.decoder.verify_chunk(&tokens).unwrap();
            let what = p(&format!("verify width {width}"));
            assert_same_bits(&got, &want, &format!("{what}: logits"));
            let entries = drain_outputs(&mut on.decoder);
            drain_outputs(&mut off.decoder);
            for (inner, states) in [(&mut on, got_states), (&mut off, want_states)] {
                if let Some(adapter) = &mut inner.decoder.paged {
                    adapter.rollback_last_tokens((width - keep) as u32).unwrap();
                }
                inner.decoder.restore_state(states[keep - 1].clone());
            }
            route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &what);
            let next = 4 + (width as u32 % 7);
            let got = on.decoder.step(next).unwrap();
            let want = off.decoder.step(next).unwrap();
            assert_same_bits(&got, &want, &format!("{what}: step after the commit"));
            let step_entries = drain_outputs(&mut on.decoder);
            drain_outputs(&mut off.decoder);
            let passes = on.decoder.take_hidden_taps();
            assert_eq!(passes.len(), 2, "{what}: verify + step");
            let rows = last_writes(&entries, layers);
            assert_eq!(passes[0].rows().unwrap(), keep, "{what}: the commit's rows");
            let kept: Vec<Vec<MxArray>> = rows
                .iter()
                .map(|outputs| vec![outputs[0].slice_axis(1, 0, keep as i64).unwrap()])
                .collect();
            assert_pass(&passes[0], &kept, &TINY_TAPS, at, keep, &c, &what);
            assert_pass(
                &passes[1],
                &last_writes(&step_entries, layers),
                &TINY_TAPS,
                at + keep,
                1,
                &c,
                &format!("{what}: step"),
            );
            token = argmax(&got);
        }

        // A poisoned tentative pass replayed on host routes from `from`.
        for from in 1..layers {
            let at = on.decoder.history.len();
            let mut replayed = Vec::new();
            for inner in [&mut on, &mut off] {
                let d = &mut inner.decoder;
                let base = d.snapshot();
                d.layer_inputs_pin = Some(true);
                d.poison_moe_at = Some(from);
                d.step(token).unwrap();
                d.layer_inputs_pin = None;
                let inputs = d.layer_inputs.take();
                let prior = std::mem::replace(&mut d.decode_pass, true);
                replayed.push(
                    d.replay_from_layer(&base, from, inputs, token, true)
                        .unwrap(),
                );
                d.decode_pass = prior;
            }
            let what = p(&format!("replay from {from}"));
            assert_same_bits(&replayed[0], &replayed[1], &format!("{what}: logits"));
            route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &what);
            let entries = drain_outputs(&mut on.decoder);
            drain_outputs(&mut off.decoder);
            assert_eq!(
                entries.len(),
                2 * layers - from,
                "{what}: pass + replay layers"
            );
            let passes = on.decoder.take_hidden_taps();
            assert_eq!(
                passes.len(),
                1,
                "{what}: the replay replaces the poisoned pass"
            );
            assert_pass(
                &passes[0],
                &last_writes(&entries, layers),
                &TINY_TAPS,
                at,
                1,
                &c,
                &what,
            );
            token = argmax(&replayed[0]);
        }

        // A device prefill replay of a window over the window it replaces.
        let at = on.decoder.history.len();
        let wanted = [6, 10, 3, 8, 5];
        let mut replayed = Vec::new();
        for inner in [&mut on, &mut off] {
            let base = inner.decoder.snapshot();
            inner.decoder.prefill_chunk(&[11; 5], None, true).unwrap();
            replayed.push(
                inner
                    .decoder
                    .replay_device_prefill(base, &wanted, None, true)
                    .unwrap(),
            );
        }
        let what = p("device prefill replay");
        assert_same_bits(&replayed[0], &replayed[1], &format!("{what}: logits"));
        route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &what);
        let entries = drain_outputs(&mut on.decoder);
        drain_outputs(&mut off.decoder);
        assert_eq!(entries.len(), 2 * layers, "{what}: two passes ran");
        let passes = on.decoder.take_hidden_taps();
        assert_eq!(passes.len(), 1, "{what}: the replay replaces the pass");
        assert_pass(
            &passes[0],
            &last_writes(&entries[layers..], layers),
            &TINY_TAPS,
            at,
            5,
            &c,
            &what,
        );

        assert!(off.decoder.take_hidden_taps().is_empty());
    }
}

/// On the Flash Next MoE shape (hidden 2560) with every expert in the
/// slots, a tentative device prefill window of 256 rows (where the machine
/// runs its kernels) and device-route decode with taps after layers 0, 1
/// and 3 give the bits of a decoder with none over 16 greedy steps, one of which misses in
/// layer 2 and replays it and layer 3 on device routes: logits, every layer output, every
/// state array; each step's taps are the stream means of its layers' last
/// outputs.
/// MUTATION: a replayed layer keeps the first run's tap → red.
#[test]
fn taps_on_device_routes_and_their_replay_move_no_bit() {
    let _host = memory::scoped_fixture_host();
    let mut on = route_step::moe_shape_inner(12);
    let mut off = route_step::moe_shape_inner(12);
    let c = on.decoder.config.clone();
    let layers = c.num_hidden_layers;
    let taps = [0, 1, 3];
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    for inner in [&mut on, &mut off] {
        // The fixture's trained window is 256 rows; the window and the steps need more.
        inner.decoder.context_window = 512;
        inner.activate_paged_seq(1).unwrap();
        route_step::fill_slots(&mut inner.decoder, &all);
    }
    on.decoder.set_hidden_taps(Some(&taps[..])).unwrap();
    record_outputs([&mut on.decoder, &mut off.decoder]);
    let replays = |inner: &Inner| {
        inner
            .decoder
            .weights
            .live
            .route_replays
            .load(Ordering::Relaxed)
    };
    if tentative_kernels_available() {
        let prompt = prompt(256);
        on.decoder.tentative_commits.clear();
        let got = on.decoder.prefill_chunk(&prompt, None, true).unwrap();
        let want = off.decoder.prefill_chunk(&prompt, None, true).unwrap();
        assert_eq!(
            on.decoder.tentative_commits,
            [1],
            "the window ran tentatively"
        );
        assert_same_bits(&got, &want, "tentative window logits");
        route_step::assert_same_decoder_state(
            &mut on.decoder,
            &mut off.decoder,
            "tentative window",
        );
        let entries = drain_outputs(&mut on.decoder);
        drain_outputs(&mut off.decoder);
        let passes = on.decoder.take_hidden_taps();
        assert_eq!(passes.len(), 1);
        assert_pass(
            &passes[0],
            &last_writes(&entries, layers),
            &taps,
            0,
            256,
            &c,
            "tentative window",
        );
    } else {
        eprintln!("no tentative prefill kernels on this machine: window part skipped");
    }
    let start = replays(&on);
    const MISS_AT: usize = 8;
    let mut token = 3;
    for step in 0..16 {
        if step == MISS_AT {
            for inner in [&mut on, &mut off] {
                for expert in [9, 10, 11] {
                    inner.decoder.weights.evict_for_test(2, expert).unwrap();
                }
            }
        }
        let at = on.decoder.history.len();
        let before = replays(&on);
        let got = on.decoder.step(token).unwrap();
        let want = off.decoder.step(token).unwrap();
        let what = format!("step {step} token {token}");
        assert_same_bits(&got, &want, &format!("{what}: logits"));
        route_step::assert_same_decoder_state(&mut on.decoder, &mut off.decoder, &what);
        let entries = drain_outputs(&mut on.decoder);
        let reference = drain_outputs(&mut off.decoder);
        let (a, b) = (
            last_writes(&entries, layers),
            last_writes(&reference, layers),
        );
        for (layer, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_same_bits(&x[0], &y[0], &format!("{what}: layer {layer} output"));
        }
        let passes = on.decoder.take_hidden_taps();
        assert_eq!(passes.len(), 1, "{what}");
        assert_pass(&passes[0], &a, &taps, at, 1, &c, &what);
        if step == MISS_AT {
            assert_eq!(replays(&on), before + 1, "{what}: the miss replays");
            assert!(entries.len() > 2, "{what}: the replay ran layers again");
        }
        token = argmax(&got);
    }
    assert!(replays(&on) > start, "a step replayed");
    assert_eq!(replays(&on), replays(&off));
}

/// Without taps no pass builds a tap: the whole pass sequence of
/// [`taps_read_each_layer_output_and_move_no_bit`]'s first half builds no
/// stream mean, keeps no record, and submits what the tapped decoder
/// submits (the taps ride no submit). With taps each tapped layer of each
/// pass builds one mean per window.
/// MUTATION: the hook builds the mean with no taps set → red.
#[test]
fn no_taps_build_nothing() {
    let run = |taps: bool| {
        let mut inner = tiny_inner();
        inner.activate_paged_seq(1).unwrap();
        if taps {
            inner.decoder.set_hidden_taps(Some(&TINY_TAPS[..])).unwrap();
        }
        inner.decoder.submit_log.borrow_mut().clear();
        let before = hidden_taps_means();
        let w0: Vec<u32> = (0..8).map(|i| 3 + (i % 10) as u32).collect();
        let w1: Vec<u32> = (0..8).map(|i| 5 + (i % 9) as u32).collect();
        inner
            .decoder
            .prefill_group(&[&w0, &w1], &[9, 4, 12], true, false)
            .unwrap();
        let mut token = 5;
        for _ in 0..4 {
            token = argmax(&inner.decoder.step(token).unwrap());
        }
        inner.decoder.verify_chunk(&[token, 4, 7]).unwrap();
        let means = hidden_taps_means() - before;
        let records = inner.decoder.take_hidden_taps().len();
        let log = inner.decoder.submit_log.borrow().clone();
        (means, records, log)
    };
    let (off_means, off_records, off_log) = run(false);
    let (on_means, on_records, on_log) = run(true);
    assert_eq!(
        [off_means, off_records as u64],
        [0, 0],
        "no taps: nothing built"
    );
    let taps = TINY_TAPS.len() as u64;
    assert_eq!(
        on_means,
        taps * (2 + 1 + 4 + 1),
        "one mean per tapped layer and window"
    );
    assert_eq!(on_records, 1 + 1 + 4 + 1);
    assert_eq!(on_log, off_log, "the taps add no submit");
}

fn hidden_taps_means() -> u64 {
    decoder::stream_means()
}
