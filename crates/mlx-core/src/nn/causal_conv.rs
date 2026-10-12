//! Explicit state for causal NLC convolutions. State belongs to a stream, not weights.
use crate::array::MxArray;
use crate::nn::{Conv1d, conv_transpose1d::ConvTranspose1d};
use napi::{Error, Result};

// History arrays are immutable: each convolution step replaces the handle.
#[derive(Default, Clone)]
pub struct ConvState {
    pub history: Option<MxArray>,
}

pub struct CausalConv1d {
    conv: Conv1d,
    history_len: i64,
}

impl CausalConv1d {
    pub fn new(
        weight: &MxArray,
        bias: Option<&MxArray>,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 3 || dilation == 0 || groups == 0 {
            return Err(Error::from_reason("Invalid causal convolution geometry"));
        }
        Ok(Self {
            conv: Conv1d::from_weights(
                weight,
                bias,
                Some(1),
                Some(0),
                Some(dilation),
                Some(groups),
            )?,
            history_len: (shape[1] - 1) * i64::from(dilation),
        })
    }
    pub fn step(&self, input: &MxArray, state: &mut ConvState) -> Result<MxArray> {
        let n = input.shape()?[1];
        if n == 0 {
            return Err(Error::from_reason(
                "Causal convolution requires nonempty input",
            ));
        }
        let x = match &state.history {
            Some(history) => MxArray::concatenate_many(vec![history, input], Some(1))?,
            None => input.pad(&[0, 0, self.history_len as i32, 0, 0, 0], 0.)?,
        };
        if self.history_len > 0 {
            state.history = Some(x.slice_axis(1, n, n + self.history_len)?.deep_copy()?);
        }
        self.conv.forward(&x)
    }
}

pub struct CausalTransposeConv1d {
    conv: ConvTranspose1d,
}

impl CausalTransposeConv1d {
    pub fn new(weight: MxArray, bias: Option<MxArray>, stride: i32) -> Result<Self> {
        Ok(Self {
            conv: ConvTranspose1d::from_weights(weight, bias, stride)?,
        })
    }
    pub fn step(&self, input: &MxArray, state: &mut ConvState) -> Result<MxArray> {
        let emit = input.shape()?[1] * i64::from(self.conv.stride);
        let overlap = i64::from(self.conv.kernel - self.conv.stride);
        let mut output = self.conv.forward_without_bias(input)?;
        if let Some(tail) = &state.history {
            let length = output.shape()?[1];
            output = output.add(&tail.pad(&[0, 0, 0, (length - overlap) as i32, 0, 0], 0.)?)?;
        }
        if overlap > 0 {
            state.history = Some(output.slice_axis(1, emit, emit + overlap)?.deep_copy()?);
        }
        // Bias is added only once, after overlap accumulation. Final overlap is the
        // causal right trim; it is discarded on end, never emitted as extra audio.
        self.conv.add_bias(output.slice_axis(1, 0, emit)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_preserves_dilation_overlap_and_bias() {
        let input = MxArray::from_float32(&[1., -2., 3., 4., -5., 6., 7.], &[1, 7, 1]).unwrap();
        let weight = MxArray::from_float32(&[0.1, 0.2, 0.3, 0.4, 0.5], &[1, 5, 1]).unwrap();
        let bias = MxArray::from_float32(&[1.25], &[1]).unwrap();
        let causal = CausalConv1d::new(&weight, Some(&bias), 3, 1).unwrap();
        // Kernel larger than twice the stride exercises a tail that spans
        // several short input chunks, with a nonzero bias added exactly once.
        let transpose = CausalTransposeConv1d::new(weight, Some(bias), 2).unwrap();
        for transposed in [false, true] {
            let apply = |x: &MxArray, state: &mut ConvState| {
                if transposed {
                    transpose.step(x, state)
                } else {
                    causal.step(x, state)
                }
                .unwrap()
            };
            let expected = apply(&input, &mut ConvState::default())
                .to_float32()
                .unwrap();
            for chunks in [vec![1; 7], vec![2, 1, 4], vec![7]] {
                let mut state = ConvState::default();
                let mut at = 0;
                let mut actual = vec![];
                for n in chunks {
                    let out = apply(&input.slice_axis(1, at, at + n).unwrap(), &mut state);
                    actual.extend_from_slice(&out.to_float32().unwrap());
                    at += n;
                }
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
    }
}
