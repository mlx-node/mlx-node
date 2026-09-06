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

/// Preserve each owner's `[1,T,H]` projection shape while attention consumes
/// a packed ragged wave. Token rows within an owner must remain together.
pub(super) fn forward_owner_spans(
    x: &MxArray,
    owners: &[crate::transformer::paged_kv_cache_adapter::PagedRaggedRow],
    mut forward: impl FnMut(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    let mut offset = 0i64;
    let mut outputs = Vec::with_capacity(owners.len());
    for owner in owners {
        let end = offset + i64::from(owner.query_len);
        if owner.query_len == 0 || end > x.shape_at(0)? {
            return Err(Error::from_reason("invalid ragged projection span"));
        }
        let span = x.slice_axis(0, offset, end)?.transpose(Some(&[1, 0, 2]))?;
        outputs.push(forward(&span)?.transpose(Some(&[1, 0, 2]))?);
        offset = end;
    }
    if outputs.is_empty() || offset != x.shape_at(0)? {
        return Err(Error::from_reason(
            "ragged projection spans do not cover input",
        ));
    }
    MxArray::concatenate_many(outputs.iter().collect(), Some(0))
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
