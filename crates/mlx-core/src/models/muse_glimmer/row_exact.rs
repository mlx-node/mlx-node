use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Apply a decode operation to independent `[1, 1, H]` rows and restore the
/// scheduler's `[N, 1, H]` layout. Packed projections use this to retain the
/// same Metal graph as serial inference without serializing paged attention.
pub(super) fn forward_rows_independently(
    x: &MxArray,
    mut forward: impl FnMut(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    let batch = usize::try_from(x.shape_at(0)?).map_err(|_| {
        Error::from_reason("Muse-Glimmer row-exact decode received a negative batch")
    })?;
    if batch == 0 {
        return Err(Error::from_reason(
            "Muse-Glimmer row-exact decode requires at least one row",
        ));
    }
    if batch == 1 {
        return forward(x);
    }
    let rows = (0..batch)
        .map(|row| {
            let row = x.slice_axis(0, row as i64, row as i64 + 1)?;
            forward(&row)
        })
        .collect::<Result<Vec<_>>>()?;
    MxArray::concatenate_many(rows.iter().collect(), Some(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_rows_in_scheduler_order() {
        let input =
            MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]).expect("row-exact input");
        let output = forward_rows_independently(&input, |row| row.mul_scalar(2.0))
            .expect("row-exact output");
        assert_eq!(output.shape().expect("shape").as_ref(), [2, 1, 2]);
        assert_eq!(
            output.to_float32().expect("values").as_ref(),
            [2.0, 4.0, 6.0, 8.0]
        );
    }
}
