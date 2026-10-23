use super::*;
use crate::array::DType;
fn bf(values: &[f32], shape: &[i64]) -> MxArray {
    MxArray::from_float32(values, shape)
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap()
}
fn bits(x: &MxArray) -> Vec<u16> {
    x.to_uint16_native().unwrap()
}

#[test]
fn source_mixer_hidden_is_normalized_once_without_second_residual() -> Result<()> {
    let norm = RMSNorm::from_weight(&bf(&[1.125; 2048], &[2048]), Some(1e-6))?;
    let residual = bf(&vec![2.0; 8 * 2048], &[1, 8, 2048]);
    let delta = bf(
        &(0..8 * 2048)
            .map(|i| ((i % 47) as f32 - 23.0) * 0.03125)
            .collect::<Vec<_>>(),
        &[1, 8, 2048],
    );
    let hidden = residual.add(&delta)?;
    let (got, normalized) = DecoderLayer::finish_mixer(&norm, &residual, hidden.clone(), true)?;
    assert_eq!(bits(&got), bits(&hidden));
    assert_eq!(bits(&normalized), bits(&norm.forward(&hidden)?));
    assert_ne!(bits(&got), bits(&residual.add(&hidden)?));
    let (native, native_norm) = DecoderLayer::finish_mixer(&norm, &residual, delta, false)?;
    assert_eq!(bits(&native), bits(&hidden));
    assert_eq!(bits(&native_norm), bits(&normalized));
    // The next FFN carrier and final norm preserve the one-add meaning.
    let output = VerifyLayerOutput::Hidden(got);
    let (tap, final_norm) = output.normalize(&norm)?;
    assert_eq!(bits(&tap), bits(&hidden));
    assert_eq!(bits(&final_norm), bits(&norm.forward(&hidden)?));
    Ok(())
}

#[test]
fn source_mixer_hidden_one_trace_preserves_changing_input_and_raw_tap() -> Result<()> {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return Ok(());
    }
    let norm = RMSNorm::from_weight(&bf(&[0.875; 2048], &[2048]), Some(1e-6))?;
    norm.get_weight().eval();
    let key = 0x5150_534d_4958_0001;
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    let mut traces = 0;
    for seed in [3, 17, 29, 41] {
        let residual = bf(
            &(0..8 * 2048)
                .map(|i| ((i * 3 + seed) % 71) as f32 * 0.125 - 4.0)
                .collect::<Vec<_>>(),
            &[1, 8, 2048],
        );
        let hidden = bf(
            &(0..8 * 2048)
                .map(|i| ((i * 11 + seed) % 61) as f32 * 0.0625 - 2.0)
                .collect::<Vec<_>>(),
            &[1, 8, 2048],
        );
        let mut build = |a: &[MxArray]| -> Result<Vec<MxArray>> {
            traces += 1;
            let (h, n) = DecoderLayer::finish_mixer(&norm, &a[0], a[1].clone(), true)?;
            Ok(vec![a[0].clone(), h, n])
        };
        let got = crate::compiled_graph::invoke_compiled_graph(
            key,
            &[&residual, &hidden],
            3,
            true,
            &mut build,
        )?
        .unwrap();
        assert_eq!(bits(&got[0]), bits(&residual));
        assert_eq!(bits(&got[1]), bits(&hidden));
        assert_eq!(bits(&got[2]), bits(&norm.forward(&hidden)?));
    }
    assert_eq!(traces, 1);
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    Ok(())
}
