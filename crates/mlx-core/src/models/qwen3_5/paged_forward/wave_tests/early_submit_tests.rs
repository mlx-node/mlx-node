//! Which layers a paged forward queues while the host builds the next ones:
//! a lone DFlash2 verify every layer but the last, a verify wave, an MTP
//! verify and an AR step the first four; and a verify keeps every bit when
//! the queued layers change, alone and in waves of 2, 3 and 4 lanes.
use super::*;

/// The early layers of `maybe_submit_paged_decode_layer` with no override.
const DECODE_LAYERS: u64 = 4;
const WINDOW: u32 = 262_144;

fn submits() -> u64 {
    PAGED_LAYER_SUBMITS.with(std::cell::Cell::get)
}

/// Sets the DFlash2 verify submit of this thread until dropped.
struct SubmitOverride;

impl SubmitOverride {
    fn set(submit: VerifySubmit) -> Self {
        DFLASH2_VERIFY_SUBMIT_OVERRIDE.with(|o| o.set(Some(submit)));
        SubmitOverride
    }
}

impl Drop for SubmitOverride {
    fn drop(&mut self) {
        DFLASH2_VERIFY_SUBMIT_OVERRIDE.with(|o| o.set(None));
    }
}

const WAVES: [&[usize]; 3] = [&[0, 1], &[1, 2, 3], &[0, 1, 2, 3]];

#[test]
fn a_lone_dflash2_verify_queues_every_layer_but_the_last() -> Result<()> {
    if !metal("a_lone_dflash2_verify_queues_every_layer_but_the_last") {
        return Ok(());
    }
    let mut model = model()?;
    let every = BLOCKS.len() as u64 - 1;
    let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
    for lane in 0..LANES.len() {
        let before = submits();
        alone(&mut model, &mut r, &[lane], &[verify_ids(lane, 0, 7)])?;
        let got = submits() - before;
        println!("SUBMITS recorded lane {lane} {got}");
        assert_eq!(
            got, every,
            "a recorded verify of lane {lane} queued {got} layers"
        );
    }
    let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
    let seq = LANES[0].0;
    r.adapter
        .activate_request(seq)
        .map_err(Error::from_reason)?;
    let ids = verify_ids(0, 1, 3);
    let l = &mut r.lanes[0];
    let before = submits();
    let (logits, _) = run_paged_dflash2_verify_step(
        VerifyInput::Host(&ids),
        &model.embed,
        &mut model.layers,
        &mut l.caches,
        &model.final_norm,
        &model.lm_head,
        &model.kinds,
        &mut r.adapter,
        &mut l.tape,
        l.rope_deltas,
        &TAPS,
        &l.gdn,
        WINDOW,
    )?;
    let got = submits() - before;
    logits.eval();
    println!("SUBMITS unrecorded {got}");
    assert_eq!(got, every, "an unrecorded verify queued {got} layers");
    Ok(())
}

#[test]
fn a_verify_wave_queues_the_decode_layers() -> Result<()> {
    if !metal("a_verify_wave_queues_the_decode_layers") {
        return Ok(());
    }
    let model = model()?;
    for order in WAVES {
        let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let before = submits();
        wave(&model, &mut r, order, &ids)?;
        let got = submits() - before;
        println!("SUBMITS wave lanes={} {got}", order.len());
        assert_eq!(
            got,
            DECODE_LAYERS,
            "a wave of {} lanes queued {got} layers",
            order.len()
        );
    }
    Ok(())
}

#[test]
fn decode_forwards_and_an_mtp_verify_queue_the_decode_layers() -> Result<()> {
    if !metal("decode_forwards_and_an_mtp_verify_queue_the_decode_layers") {
        return Ok(());
    }
    let layers = BLOCKS.len();
    let probe = MxArray::from_float32(&[1.0], &[1])?;
    let before = submits();
    for layer in 0..layers {
        maybe_submit_paged_decode_layer(layer, layers, &probe);
    }
    let got = submits() - before;
    println!("SUBMITS decode default {got}");
    assert_eq!(got, DECODE_LAYERS, "the decode default queued {got} layers");

    let mut model = model()?;
    let (seq, _, delta) = LANES[0];
    let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
    r.adapter
        .activate_request(seq)
        .map_err(Error::from_reason)?;
    let before = submits();
    let logits = run_paged_decode_step(
        17,
        &model.embed,
        &mut model.layers,
        &mut r.lanes[0].caches,
        &model.final_norm,
        &model.lm_head,
        &model.kinds,
        &mut r.adapter,
        delta,
    )?;
    let got = submits() - before;
    logits.eval();
    println!("SUBMITS ar step {got}");
    assert_eq!(got, DECODE_LAYERS, "an AR step queued {got} layers");

    let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
    r.adapter
        .activate_request(seq)
        .map_err(Error::from_reason)?;
    let mut tape = Vec::new();
    let before = submits();
    let out = run_paged_verify_step(
        &verify_ids(0, 3, 11),
        &model.embed,
        &mut model.layers,
        &mut r.lanes[0].caches,
        &model.final_norm,
        &model.lm_head,
        &model.kinds,
        &mut r.adapter,
        &mut tape,
        delta,
    )?;
    let got = submits() - before;
    out.logits.eval();
    println!("SUBMITS mtp verify {got}");
    assert_eq!(got, DECODE_LAYERS, "an MTP verify queued {got} layers");
    Ok(())
}

/// Logits, taps, tapes, live GDN slots and stored K/V rows of a verify with
/// the first four layers queued equal those with every layer queued: four
/// lanes alone, and waves of 2, 3 and 4 lanes.
#[test]
fn a_verify_keeps_its_bits_when_every_layer_is_queued() -> Result<()> {
    if !metal("a_verify_keeps_its_bits_when_every_layer_is_queued") {
        return Ok(());
    }
    let mut model = model()?;
    let every = BLOCKS.len() as u64 - 1;
    let order = [0usize, 1, 2, 3];
    let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
    let mut compared = 0usize;

    let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
    let before = submits();
    let queued_every = alone(&mut model, &mut r, &order, &ids)?;
    assert_eq!(submits() - before, every * order.len() as u64);
    let queued_decode = {
        let _decode = SubmitOverride::set(VerifySubmit::Decode);
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let before = submits();
        let held = alone(&mut model, &mut r, &order, &ids)?;
        assert_eq!(submits() - before, DECODE_LAYERS * order.len() as u64);
        held
    };
    for (lane, (got, want)) in queued_every.iter().zip(&queued_decode).enumerate() {
        assert_same(got, want, &format!("every layer queued, alone lane {lane}"));
        compared += got.len();
    }

    for order in WAVES {
        let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let queued_decode = wave(&model, &mut r, order, &ids)?;
        let queued_every = {
            let _every = SubmitOverride::set(VerifySubmit::EveryLayer);
            let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
            let before = submits();
            let held = wave(&model, &mut r, order, &ids)?;
            assert_eq!(submits() - before, every);
            held
        };
        for (&lane, (got, want)) in order.iter().zip(queued_every.iter().zip(&queued_decode)) {
            assert_same(
                got,
                want,
                &format!("every layer queued, wave of {} lane {lane}", order.len()),
            );
            compared += got.len();
        }
    }
    println!("EARLY_BITS equal arrays={compared}");
    Ok(())
}
